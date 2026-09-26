from __future__ import annotations

import asyncio
import json
from dataclasses import fields
from typing import Literal

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from llm_fakes import TEST_STRUCTURED_OUTPUT_METHOD, FakeChatModel

from agnostic_market.agents.capabilities import (
    CapabilityEntry,
    CapabilityRegistry,
    CapabilitySpec,
)
from agnostic_market.agents.recovery import CommerceEffectFinishers, clear_automation_state
from agnostic_market.agents.routing import (
    _CAPABILITY_DEFINITIONS,
    COMMERCE_EFFECT_CAPABILITIES,
    CONTEXT_PROJECTOR_VERSION,
    ROUTE_SCHEMA_FINGERPRINT,
    ROUTER_PROMPT_FINGERPRINT,
    ROUTER_SYSTEM_PROMPT,
    UNSAFE_MISROUTE_CAPABILITIES,
    RoutingAttempt,
    RoutingRecognizer,
    RoutingSession,
    SemanticRouter,
    materialize_route,
    project_routing_context,
    registry_fingerprint,
    resolve_route,
)
from agnostic_market.agents.telemetry import InMemoryTelemetrySink, TenantTelemetry
from agnostic_market.commerce.cart import CartStore
from agnostic_market.commerce.identity import BoundIdentity, CallerIdentityStore
from agnostic_market.commerce.orders import RecentOrderContext
from agnostic_market.dtos.config import ProviderModel
from agnostic_market.dtos.events import CommittedTurn
from agnostic_market.dtos.orchestration import (
    AbortCurrent,
    ActiveInvocation,
    AnswerQuestion,
    CancelOrders,
    CapabilityId,
    ChangeProfile,
    Converse,
    DiscloseAiIdentity,
    FocusedOrderSet,
    IntentRequestModel,
    ListOrders,
    ModifyCart,
    PlaceOrder,
    RecentOrderSet,
    RefundOrder,
    RequestPerson,
    ReturnOrder,
    RouteDecision,
    RouteProposal,
    RoutingContext,
    RoutingFailure,
    SearchCatalog,
    SwitchAccount,
    VerifyIdentity,
    VerifyOrderStatus,
    ViewCart,
    ViewIdentityStatus,
)
from agnostic_market.dtos.state import ProductOffer, ProductReference, ReasoningState

_SELECTION = ProviderModel(provider="fake", model="router")


class _RecordingRecognizer:
    def __init__(self, attempt: RoutingAttempt) -> None:
        self._attempt = attempt
        self.contexts: list[RoutingContext] = []

    async def route(self, context: RoutingContext) -> RoutingAttempt:
        self.contexts.append(context)
        return self._attempt


def _attempt(resolution: RouteDecision | RoutingFailure) -> RoutingAttempt:
    return RoutingAttempt(
        resolution=resolution,
        provider="fake",
        model="recognizer",
        structured_output_method=TEST_STRUCTURED_OUTPUT_METHOD,
        elapsed_ms=12.5,
        input_tokens=21,
        cache_read_tokens=3,
        output_tokens=4,
        route_schema_fingerprint="route-schema",
        prompt_fingerprint="recognizer-fingerprint",
        registry_fingerprint="registry-fingerprint",
        input_max_chars=2048,
        timeout_seconds=1.0,
        provider_call_outcome="completed",
    )


def _registry(*request_types: type[IntentRequestModel]) -> CapabilityRegistry:
    return CapabilityRegistry(
        CapabilitySpec(
            request_type.model_fields["kind"].default,
            request_type,
            CapabilityEntry(f"{request_type.model_fields['kind'].default.value}_entry"),
        )
        for request_type in request_types
    )


def _context(
    *capabilities: CapabilityId,
    active: CapabilityId | None = None,
    routing_scope: Literal["ordinary", "confirmation_escape"] = "ordinary",
) -> RoutingContext:
    return RoutingContext(
        routing_scope=routing_scope,
        utterance="cancel all my orders",
        bound_customer=False,
        active_capability=active,
        recent_order_operation=None,
        recent_order_count=0,
        cart_state="empty",
        available_capabilities=capabilities,
    )


def _router(
    model: FakeChatModel,
    registry: CapabilityRegistry,
    *,
    timeout_seconds: float = 1.0,
    input_max_chars: int = 2048,
) -> SemanticRouter:
    return SemanticRouter(
        model,
        selection=_SELECTION,
        structured_output_method=TEST_STRUCTURED_OUTPUT_METHOD,
        timeout_seconds=timeout_seconds,
        input_max_chars=input_max_chars,
        registry=registry,
    )


def test_projector_uses_the_admitted_turn_and_real_session_state() -> None:
    registry = _registry(SearchCatalog, ViewCart)
    identity = CallerIdentityStore()
    identity.bind(BoundIdentity(customer_ref="CUST-001", masked_contact="***0119"))
    cart = CartStore()
    cart.add_item(sku="SKU-1", name="Trail shoe", price_usd=79.0, quantity=1)
    recent = RecentOrderContext(max_refs=3)
    recent.record(("ORD-1001", "ORD-1002"), operation="read")
    state = ReasoningState(
        messages=[HumanMessage("stale text", id="older-turn")],
        consumed_turn_ids=("older-turn",),
        active_invocation=ActiveInvocation(
            request=SearchCatalog(query=None),
            opened_turn_id="older-turn",
        ),
    )

    projected = project_routing_context(
        CommittedTurn(text="show me another trail shoe", message_id="fresh-turn"),
        state,
        identity_store=identity,
        cart_store=cart,
        recent_orders=recent,
        registry=registry,
    )

    assert projected == RoutingContext(
        utterance="show me another trail shoe",
        bound_customer=True,
        active_capability=CapabilityId.SEARCH_CATALOG,
        recent_order_operation="read",
        recent_order_count=2,
        cart_state="nonempty",
        available_capabilities=(CapabilityId.SEARCH_CATALOG, CapabilityId.VIEW_CART),
    )


def test_projector_rejects_missing_turn_identity_before_model_work() -> None:
    result = project_routing_context(
        CommittedTurn(text="show my cart", message_id=None),
        ReasoningState(),
        identity_store=CallerIdentityStore(),
        cart_store=CartStore(),
        recent_orders=RecentOrderContext(max_refs=1),
        registry=_registry(ViewCart),
    )

    assert result == RoutingFailure(reason="context_invalid")


def test_projector_rejects_active_owner_missing_from_the_registry() -> None:
    state = ReasoningState(
        consumed_turn_ids=("turn-1",),
        active_invocation=ActiveInvocation(
            request=SearchCatalog(query="trail shoes"),
            opened_turn_id="turn-1",
        ),
    )

    result = project_routing_context(
        CommittedTurn(text="show another", message_id="turn-2"),
        state,
        identity_store=CallerIdentityStore(),
        cart_store=CartStore(),
        recent_orders=RecentOrderContext(max_refs=1),
        registry=_registry(ViewCart),
    )

    assert result == RoutingFailure(reason="context_invalid")


def test_projector_rejects_terminal_state_before_reading_live_stores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("terminal projection read a live store")

    monkeypatch.setattr(CallerIdentityStore, "current", forbidden)
    monkeypatch.setattr(CartStore, "is_empty", forbidden)
    monkeypatch.setattr(RecentOrderContext, "snapshot", forbidden)

    result = project_routing_context(
        CommittedTurn(text="show my cart", message_id="terminal-turn"),
        ReasoningState(automation_terminal=True),
        identity_store=CallerIdentityStore(),
        cart_store=CartStore(),
        recent_orders=RecentOrderContext(max_refs=1),
        registry=_registry(ViewCart),
    )

    assert result == RoutingFailure(reason="context_invalid")


async def test_routing_session_resolves_and_emits_only_closed_route_fields() -> None:
    sink = InMemoryTelemetrySink()
    telemetry = TenantTelemetry("acme_store", sink, sink).bind_session("route-session")
    registry = _registry(SearchCatalog)
    recognizer: RoutingRecognizer = _RecordingRecognizer(
        _attempt(RouteDecision.direct(SearchCatalog(query="private product wording")))
    )
    routing = RoutingSession(
        recognizer,
        identity_store=CallerIdentityStore(),
        cart_store=CartStore(),
        recent_orders=RecentOrderContext(max_refs=1),
        registry=registry,
        telemetry=telemetry.routing_evidence,
    )
    turn = CommittedTurn(
        text="find private product wording and call me on +44 7700 900123",
        message_id="route-turn-1",
    )

    projection = routing.project(turn, ReasoningState())
    assert isinstance(projection, RoutingContext)
    resolution = await routing.resolve(turn, ReasoningState())

    assert isinstance(recognizer, _RecordingRecognizer)
    assert recognizer.contexts == [projection]
    assert resolution == RouteDecision.direct(SearchCatalog(query="private product wording"))
    records = [record.flattened() for record in sink.records]
    assert records == [
        {
            "schema_version": 1,
            "purpose": "routing_evidence",
            "tenant_id": "acme_store",
            "session_id": "route-session",
            "event": "semantic_route",
            "turn_id": "route-turn-1",
            "decision_source": "active",
            "decision": "direct",
            "capability": "search_catalog",
            "clarification_reason": None,
            "failure_reason": None,
            "provider": "fake",
            "model": "recognizer",
            "reasoning_effort": None,
            "structured_output_method": TEST_STRUCTURED_OUTPUT_METHOD,
            "latency_ms": 12.5,
            "input_tokens": 21,
            "cache_read_tokens": 3,
            "output_tokens": 4,
            "route_schema_fingerprint": "route-schema",
            "router_prompt_fingerprint": "recognizer-fingerprint",
            "registry_fingerprint": "registry-fingerprint",
            "context_projector_version": CONTEXT_PROJECTOR_VERSION,
            "provider_call_outcome": "completed",
        }
    ]
    serialized = json.dumps(records)
    assert "private product wording" not in serialized
    assert "7700" not in serialized
    assert "utterance" not in records[0]
    assert "request" not in records[0]


async def test_routing_projection_failure_calls_no_recognizer() -> None:
    recognizer = _RecordingRecognizer(_attempt(RouteDecision.direct(ViewCart())))
    routing = RoutingSession(
        recognizer,
        identity_store=CallerIdentityStore(),
        cart_store=CartStore(),
        recent_orders=RecentOrderContext(max_refs=1),
        registry=_registry(ViewCart),
        telemetry=TenantTelemetry("acme_store", InMemoryTelemetrySink(), InMemoryTelemetrySink())
        .bind_session("projection-failure")
        .routing_evidence,
    )

    turn = CommittedTurn(text="show my cart", message_id="terminal-route-turn")
    projection = routing.project(
        turn,
        ReasoningState(automation_terminal=True),
    )

    assert projection == RoutingFailure(reason="context_invalid")
    assert await routing.resolve(turn, ReasoningState(automation_terminal=True)) == RoutingFailure(
        reason="context_invalid"
    )
    assert recognizer.contexts == []


def test_route_resolver_clarifies_unavailable_direct_and_rejects_ownerless_continue() -> None:
    unavailable = resolve_route(
        _context(CapabilityId.VIEW_CART),
        RouteDecision.direct(SearchCatalog(query="trail shoes")),
    )
    ownerless = resolve_route(
        _context(CapabilityId.VIEW_CART),
        RouteDecision.continue_current(),
    )

    assert unavailable == RouteDecision.clarify("unsupported_capability")
    assert ownerless == RoutingFailure(reason="decision_rejected")


def test_route_resolver_preserves_executable_decisions() -> None:
    direct = RouteDecision.direct(ViewCart())
    continuation = RouteDecision.continue_current()

    assert resolve_route(_context(CapabilityId.VIEW_CART), direct) is direct
    assert (
        resolve_route(
            _context(CapabilityId.VIEW_CART, active=CapabilityId.VIEW_CART),
            continuation,
        )
        is continuation
    )


@pytest.mark.parametrize(
    "target",
    (FocusedOrderSet(), RecentOrderSet()),
)
def test_route_resolver_normalizes_recent_order_selector_without_recent_context(
    target: FocusedOrderSet | RecentOrderSet,
) -> None:
    context = _context(CapabilityId.VERIFY_ORDER_STATUS)

    assert resolve_route(
        context,
        RouteDecision.direct(VerifyOrderStatus(target=target)),
    ) == RouteDecision.direct(VerifyOrderStatus())


@pytest.mark.parametrize(
    "decision",
    (
        RouteDecision.direct(ViewCart()),
        RouteDecision.continue_current(),
        RouteDecision.clarify("missing_value"),
    ),
)
def test_confirmation_scope_allows_only_request_person(decision: RouteDecision) -> None:
    context = _context(
        CapabilityId.VIEW_CART,
        CapabilityId.REQUEST_PERSON,
        active=CapabilityId.VIEW_CART,
        routing_scope="confirmation_escape",
    )

    assert resolve_route(context, decision) == RouteDecision.clarify("ambiguous_intent")
    person = RouteDecision.direct(RequestPerson())
    assert resolve_route(context, person) is person


async def test_confirmation_escape_projects_its_scope_for_the_one_recognizer() -> None:
    recognizer = _RecordingRecognizer(_attempt(RouteDecision.direct(ViewCart())))
    routing = RoutingSession(
        recognizer,
        identity_store=CallerIdentityStore(),
        cart_store=CartStore(),
        recent_orders=RecentOrderContext(max_refs=1),
        registry=_registry(ViewCart, RequestPerson),
        telemetry=TenantTelemetry("acme_store", InMemoryTelemetrySink(), InMemoryTelemetrySink())
        .bind_session("confirmation-escape")
        .routing_evidence,
    )

    result = await routing.resolve_confirmation_escape(
        CommittedTurn(text="sure", message_id="confirmation-turn"),
        ReasoningState(),
    )

    assert result == RouteDecision.clarify("ambiguous_intent")
    assert recognizer.contexts[0].routing_scope == "confirmation_escape"


def test_route_materializer_covers_every_capability_from_one_coarse_contract() -> None:
    # A focused target is only executable with a focused order, so the context must supply one
    # for the focused arm of this contract to survive resolve_route.
    context = _context(*CapabilityId).model_copy(
        update={
            "recent_order_operation": "read",
            "recent_order_count": 2,
            "has_focused_order": True,
        }
    )
    cases = (
        (
            RouteProposal(
                decision="direct",
                capability=CapabilityId.ANSWER_QUESTION,
                answer_topic="general",
            ),
            AnswerQuestion(topic="general"),
        ),
        (
            RouteProposal(decision="direct", capability=CapabilityId.SEARCH_CATALOG),
            SearchCatalog(),
        ),
        (
            RouteProposal(
                decision="direct",
                capability=CapabilityId.VERIFY_ORDER_STATUS,
                order_status_selector="explicit",
            ),
            VerifyOrderStatus(),
        ),
        (
            RouteProposal(
                decision="direct",
                capability=CapabilityId.VERIFY_ORDER_STATUS,
                order_status_selector="focused",
            ),
            VerifyOrderStatus(target=FocusedOrderSet()),
        ),
        (
            RouteProposal(
                decision="direct",
                capability=CapabilityId.VERIFY_ORDER_STATUS,
                order_status_selector="recent",
            ),
            VerifyOrderStatus(target=RecentOrderSet()),
        ),
        (
            RouteProposal(
                decision="direct",
                capability=CapabilityId.LIST_ORDERS,
                list_scope="account",
            ),
            ListOrders(scope="account"),
        ),
        (RouteProposal(decision="direct", capability=CapabilityId.VIEW_CART), ViewCart()),
        (
            RouteProposal(
                decision="direct",
                capability=CapabilityId.MODIFY_CART,
                cart_operation="add",
            ),
            ModifyCart(operation="add"),
        ),
        (RouteProposal(decision="direct", capability=CapabilityId.PLACE_ORDER), PlaceOrder()),
        (
            RouteProposal(
                decision="direct",
                capability=CapabilityId.CANCEL_ORDERS,
                cancel_selector="explicit",
            ),
            CancelOrders(),
        ),
        (
            RouteProposal(decision="direct", capability=CapabilityId.REFUND_ORDER),
            RefundOrder(),
        ),
        (
            RouteProposal(decision="direct", capability=CapabilityId.RETURN_ORDER),
            ReturnOrder(),
        ),
        (
            RouteProposal(
                decision="direct",
                capability=CapabilityId.CHANGE_PROFILE,
                profile_field="address",
            ),
            ChangeProfile(field="address"),
        ),
        (
            RouteProposal(decision="direct", capability=CapabilityId.VERIFY_IDENTITY),
            VerifyIdentity(),
        ),
        (
            RouteProposal(decision="direct", capability=CapabilityId.SWITCH_ACCOUNT),
            SwitchAccount(),
        ),
        (
            RouteProposal(decision="direct", capability=CapabilityId.VIEW_IDENTITY_STATUS),
            ViewIdentityStatus(),
        ),
        (
            RouteProposal(decision="direct", capability=CapabilityId.ABORT_CURRENT),
            AbortCurrent(),
        ),
        (
            RouteProposal(
                decision="direct",
                capability=CapabilityId.CONVERSE,
                conversation_act="greeting",
            ),
            Converse(act="greeting"),
        ),
        (
            RouteProposal(decision="direct", capability=CapabilityId.DISCLOSE_AI_IDENTITY),
            DiscloseAiIdentity(),
        ),
        (
            RouteProposal(decision="direct", capability=CapabilityId.REQUEST_PERSON),
            RequestPerson(),
        ),
    )

    for proposal, request in cases:
        assert materialize_route(context, proposal) == RouteDecision.direct(request)

    assert {proposal.capability for proposal, _request in cases} == set(CapabilityId)


@pytest.mark.parametrize(
    "proposal, context",
    (
        (
            RouteProposal(decision="direct", capability=CapabilityId.LIST_ORDERS),
            _context(CapabilityId.LIST_ORDERS),
        ),
        (
            RouteProposal(
                decision="direct",
                capability=CapabilityId.VIEW_CART,
                list_scope="account",
            ),
            _context(CapabilityId.VIEW_CART),
        ),
        (RouteProposal(decision="direct"), _context(CapabilityId.VIEW_CART)),
        (
            RouteProposal(
                decision="clarify",
                capability=CapabilityId.VIEW_CART,
                clarification_reason="ambiguous_intent",
            ),
            _context(CapabilityId.VIEW_CART),
        ),
        (
            RouteProposal(decision="continue", cart_operation="add"),
            _context(CapabilityId.MODIFY_CART, active=CapabilityId.MODIFY_CART),
        ),
        (
            RouteProposal(decision="continue"),
            _context(CapabilityId.MODIFY_CART),
        ),
    ),
)
def test_route_materializer_rejects_missing_or_irrelevant_coarse_fields(
    proposal: RouteProposal,
    context: RoutingContext,
) -> None:
    assert materialize_route(context, proposal) == RoutingFailure(reason="decision_rejected")


@pytest.mark.parametrize(
    "reason",
    ("invalid_output", "routing_unavailable", "context_invalid", "decision_rejected"),
)
def test_routing_failures_are_code_authored_and_not_in_the_route_schema(reason: str) -> None:
    assert RoutingFailure(reason=reason).reason == reason  # type: ignore[arg-type]
    route_schema = RouteProposal.model_json_schema()
    assert reason not in str(route_schema)


def test_router_capability_meanings_are_total_and_byte_stable() -> None:
    meaning_block = ROUTER_SYSTEM_PROMPT.split("Capability definitions:\n", 1)[1].split(
        "\n\nNever ", 1
    )[0]

    assert ROUTER_PROMPT_FINGERPRINT == (
        "80bb783df1bce2bececb2591b552cc70df522b1330645c518c64512ee34f68c1"
    )
    assert all(meaning_block.count(capability_id.value) == 1 for capability_id in CapabilityId)


def test_projector_carries_focus_presence_so_a_focused_route_stays_executable() -> None:
    """Two recent orders with a focus and two without project identically without this.

    A caller asking about "my order" right after placing one met the owner asking for a
    reference the session already held, because the focus never reached the router.
    """

    registry = _registry(VerifyOrderStatus)
    state = ReasoningState(
        messages=[HumanMessage("older", id="older-turn")],
        consumed_turn_ids=("older-turn",),
    )

    def project(recent: RecentOrderContext) -> RoutingContext:
        result = project_routing_context(
            CommittedTurn(text="any news on my order?", message_id="fresh-turn"),
            state,
            identity_store=CallerIdentityStore(),
            cart_store=CartStore(),
            recent_orders=recent,
            registry=registry,
        )
        assert isinstance(result, RoutingContext)
        return result

    single = RecentOrderContext(max_refs=3)
    single.record(("ORD-1001",), operation="place")
    assert single.snapshot().focused_order_ref == "ORD-1001"
    assert project(single).has_focused_order is True

    unfocused = RecentOrderContext(max_refs=3)
    unfocused.record(("ORD-1001", "ORD-1002"), operation="list")
    assert unfocused.snapshot().focused_order_ref is None
    assert project(unfocused).has_focused_order is False

    # The two contexts must not be indistinguishable, which was the defect.
    assert project(single) != project(unfocused)


def test_projector_reports_an_offer_only_on_the_turn_that_answers_it() -> None:
    """The offer is retired by turn adjacency, so no flow has to remember to clear it.

    Without the adjacency rule a recorded offer would keep resolving a later "yes" that was
    answering something else entirely.
    """

    registry = _registry(VerifyOrderStatus)
    offer = ProductOffer(skus=("SKU-RED-42",), turn_id="offer-turn")

    def project(state: ReasoningState, message_id: str) -> RoutingContext:
        result = project_routing_context(
            CommittedTurn(text="yes please", message_id=message_id),
            state,
            identity_store=CallerIdentityStore(),
            cart_store=CartStore(),
            recent_orders=RecentOrderContext(max_refs=3),
            registry=registry,
        )
        assert isinstance(result, RoutingContext)
        return result

    answering = ReasoningState(
        messages=[HumanMessage("what socks do you have?", id="offer-turn")],
        consumed_turn_ids=("offer-turn", "reply-turn"),
        product_offer=offer,
    )
    assert project(answering, "reply-turn").has_offered_product is True

    intervened = ReasoningState(
        messages=[HumanMessage("what socks do you have?", id="offer-turn")],
        consumed_turn_ids=("offer-turn", "unrelated-turn", "reply-turn"),
        product_offer=offer,
    )
    assert project(intervened, "reply-turn").has_offered_product is False

    without = ReasoningState(
        messages=[HumanMessage("what socks do you have?", id="offer-turn")],
        consumed_turn_ids=("offer-turn", "reply-turn"),
    )
    assert project(without, "reply-turn").has_offered_product is False


def test_projector_keeps_product_reference_separate_from_actionable_offer() -> None:
    registry = _registry(VerifyOrderStatus)
    state = ReasoningState(
        messages=[HumanMessage("It is $129.00.", id="price-turn")],
        consumed_turn_ids=("price-turn", "reply-turn"),
        product_reference=ProductReference(skus=("SKU-BLU-07",), turn_id="price-turn"),
    )
    context = project_routing_context(
        CommittedTurn(text="add it", message_id="reply-turn"),
        state,
        identity_store=CallerIdentityStore(),
        cart_store=CartStore(),
        recent_orders=RecentOrderContext(max_refs=3),
        registry=registry,
    )
    assert isinstance(context, RoutingContext)
    assert context.has_product_reference is True
    assert context.has_offered_product is False
    assert state.live_product_reference("reply-turn") is not None
    assert state.live_product_offer("reply-turn") is None


def test_projector_exposes_only_a_live_open_help_prompt() -> None:
    registry = _registry(Converse, ViewCart)

    def project(*turn_ids: str) -> RoutingContext:
        state = ReasoningState.model_validate(
            {
                "consumed_turn_ids": turn_ids,
                "assistant_prompt": {"kind": "open_help", "turn_id": "invitation-turn"},
            }
        )
        result = project_routing_context(
            CommittedTurn(text="Yes.", message_id="reply-turn"),
            state,
            identity_store=CallerIdentityStore(),
            cart_store=CartStore(),
            recent_orders=RecentOrderContext(max_refs=3),
            registry=registry,
            assistant_prompt_completed=True,
        )
        assert isinstance(result, RoutingContext)
        return result

    answering = project("invitation-turn", "reply-turn")
    assert answering.awaiting_reply_kind == "open_help"
    assert answering.has_offered_product is False

    stale = project("invitation-turn", "intervening-turn", "reply-turn")
    assert stale.awaiting_reply_kind is None


@pytest.mark.parametrize("completed", [False, True])
def test_projector_requires_completed_speech_for_open_help(completed: bool) -> None:
    state = ReasoningState.model_validate(
        {
            "consumed_turn_ids": ("invitation-turn", "reply-turn"),
            "assistant_prompt": {"kind": "open_help", "turn_id": "invitation-turn"},
        }
    )
    context = project_routing_context(
        CommittedTurn(text="Yes.", message_id="reply-turn"),
        state,
        identity_store=CallerIdentityStore(),
        cart_store=CartStore(),
        recent_orders=RecentOrderContext(max_refs=3),
        registry=_registry(Converse, ViewCart),
        assistant_prompt_completed=completed,
    )
    assert isinstance(context, RoutingContext)
    assert context.awaiting_reply_kind == ("open_help" if completed else None)


def test_a_product_offer_survives_the_automation_reset_that_dispatch_applies() -> None:
    """A `pending_`-prefixed name would be wiped on the very turn that routes the "yes".

    `validate_automation_state_clear` forces every prefixed field into the reset map, and
    `capability_dispatch` applies that map before handing the turn to the cart owner.
    """

    offer = ProductOffer(skus=("SKU-RED-42",), turn_id="offer-turn")
    reference = ProductReference(skus=("SKU-RED-42",), turn_id="offer-turn")
    state = ReasoningState(
        messages=[HumanMessage("what socks do you have?", id="offer-turn")],
        consumed_turn_ids=("offer-turn", "reply-turn"),
        product_offer=offer,
        product_reference=reference,
    )

    survived = state.model_copy(update=clear_automation_state())

    assert survived.product_offer == offer
    assert survived.product_reference == reference
    assert survived.active_invocation is None
    assert survived.pending_cart_mutation is None


def test_every_declared_discriminator_has_a_prompt_example() -> None:
    """A required field no example teaches is a live rejection waiting to happen.

    `converse` shipped without one and every greeting was rejected as decision_rejected,
    because the schema marks the field optional while the capability table requires it.
    """

    examples = ROUTER_SYSTEM_PROMPT.split("Contrastive examples:", 1)[1]
    taught: dict[str, set[str]] = {}
    for line in examples.splitlines():
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        payload = json.loads(stripped)
        capability = payload.get("capability")
        if capability is None:
            continue
        taught.setdefault(capability, set()).update(payload.keys())

    untaught = sorted(
        f"{capability.value}.{field}"
        for capability, definition in _CAPABILITY_DEFINITIONS.items()
        for field in definition.discriminators
        if field not in taught.get(capability.value, set())
    )
    assert untaught == []


def test_commerce_effect_capabilities_match_the_post_commit_boundary() -> None:
    """The effect set is declared on the capability table; this pins why those six.

    A capability is a commerce effect when its flow owns one of the post-commit
    projection boundaries in CommerceEffectFinishers. That dataclass is the
    independent source: it is defined for recovery, not for routing, so it moves
    only when the commit boundary itself moves.

    Identity capabilities are deliberately absent. verify_identity and
    switch_account do mutate, via identity_apply to transition_principal to
    begin_principal_retirement, which clears cart, recent and guest state. That
    commit is to session state, a different subsystem, and owns no finisher here.
    """
    boundary_to_capability = {
        "cart_mutation": CapabilityId.MODIFY_CART,
        "placement": CapabilityId.PLACE_ORDER,
        "cancel": CapabilityId.CANCEL_ORDERS,
        "refund": CapabilityId.REFUND_ORDER,
        "return_": CapabilityId.RETURN_ORDER,
        "profile_change": CapabilityId.CHANGE_PROFILE,
    }
    finisher_fields = {field.name for field in fields(CommerceEffectFinishers)}

    assert set(boundary_to_capability) == finisher_fields
    assert frozenset(boundary_to_capability.values()) == COMMERCE_EFFECT_CAPABILITIES
    assert CapabilityId.VERIFY_IDENTITY not in COMMERCE_EFFECT_CAPABILITIES
    assert CapabilityId.SWITCH_ACCOUNT not in COMMERCE_EFFECT_CAPABILITIES


def test_unsafe_misroute_targets_cover_effects_and_session_control() -> None:
    assert (
        COMMERCE_EFFECT_CAPABILITIES
        | {
            CapabilityId.VERIFY_IDENTITY,
            CapabilityId.SWITCH_ACCOUNT,
            CapabilityId.ABORT_CURRENT,
            CapabilityId.REQUEST_PERSON,
        }
        == UNSAFE_MISROUTE_CAPABILITIES
    )


def test_router_prompt_distinguishes_reported_speech_and_context_gaps() -> None:
    prompt = " ".join(ROUTER_SYSTEM_PROMPT.split())

    assert "A quoted command is not the caller's request" in prompt
    assert "A declarative report with no requested action" in prompt
    assert "ambiguous among capability owners" in prompt
    assert "requires unavailable live account membership" in prompt
    assert "no recent order context" in prompt


async def test_attempt_records_when_the_resolver_rewrote_the_models_decision() -> None:
    """The resolver may rewrite a decision for runtime safety; evidence must say so.

    Without this the report records the rewritten route as the model's own proposal,
    so a wrong selector is indistinguishable from a right one.
    """
    model = FakeChatModel(
        structured_args={
            "RouteProposal": (
                {
                    "decision": "direct",
                    "capability": "verify_order_status",
                    "order_status_selector": "focused",
                },
            )
        }
    )
    registry = _registry(VerifyOrderStatus)
    router = _router(model, registry)

    # recent_order_count is 0, so a focused selector cannot be executed as proposed.
    attempt = await router.route(_context(CapabilityId.VERIFY_ORDER_STATUS))

    assert attempt.resolution == RouteDecision.direct(VerifyOrderStatus())
    assert attempt.resolution_adjusted is True


async def test_semantic_router_forwards_transport_and_returns_sanitized_attempt() -> None:
    model = FakeChatModel(record_prompts=True)
    registry = _registry(SearchCatalog)
    router = _router(model, registry)

    attempt = await router.route(_context(CapabilityId.SEARCH_CATALOG))

    assert isinstance(attempt, RoutingAttempt)
    assert attempt.resolution == RouteDecision.direct(SearchCatalog())
    assert attempt.provider == "fake"
    assert attempt.model == "router"
    assert attempt.structured_output_method == TEST_STRUCTURED_OUTPUT_METHOD
    assert attempt.input_tokens is None
    assert attempt.cache_read_tokens is None
    assert attempt.output_tokens is None
    assert attempt.route_schema_fingerprint == ROUTE_SCHEMA_FINGERPRINT
    assert attempt.prompt_fingerprint == ROUTER_PROMPT_FINGERPRINT
    assert attempt.registry_fingerprint == registry_fingerprint(registry)
    assert attempt.input_max_chars == 2048
    assert attempt.timeout_seconds == 1.0
    assert attempt.provider_call_outcome == "completed"
    assert attempt.resolution_adjusted is False
    assert attempt.projector_version == CONTEXT_PROJECTOR_VERSION
    assert model.structured_methods == (TEST_STRUCTURED_OUTPUT_METHOD,)
    assert "cancel all my orders" in model._seen_prompts[-1]
    assert set(attempt.__dataclass_fields__) == {
        "resolution",
        "provider",
        "model",
        "structured_output_method",
        "elapsed_ms",
        "input_tokens",
        "cache_read_tokens",
        "output_tokens",
        "route_schema_fingerprint",
        "prompt_fingerprint",
        "registry_fingerprint",
        "input_max_chars",
        "timeout_seconds",
        "provider_call_outcome",
        "projector_version",
        "reasoning_effort",
        "temperature",
        "observed_at",
        "provider_error_category",
        "provider_request_id",
        "provider_retry_count",
        "resolution_adjusted",
    }
    assert attempt.observed_at is not None
    assert attempt.observed_at.tzinfo is not None
    assert attempt.provider_error_category is None
    assert attempt.provider_retry_count is None


async def test_semantic_router_extracts_standardized_usage_only() -> None:
    class UsageModel(FakeChatModel):
        def _respond(self, messages, **kwargs):  # type: ignore[override]
            response = super()._respond(messages, **kwargs)
            return AIMessage(
                content=response.content,
                tool_calls=response.tool_calls,
                usage_metadata={
                    "input_tokens": 41,
                    "output_tokens": 7,
                    "total_tokens": 48,
                    "input_token_details": {"cache_read": 11},
                },
            )

    attempt = await _router(UsageModel(), _registry(SearchCatalog)).route(
        _context(CapabilityId.SEARCH_CATALOG)
    )

    assert (attempt.input_tokens, attempt.cache_read_tokens, attempt.output_tokens) == (41, 11, 7)


async def test_semantic_router_classifies_invalid_and_unavailable_without_fallback() -> None:
    invalid = await _router(
        FakeChatModel(
            structured_args={"RouteProposal": ({"decision": "direct", "capability": "not_real"},)}
        ),
        _registry(CancelOrders),
    ).route(_context(CapabilityId.CANCEL_ORDERS))
    unavailable_model = FakeChatModel(raise_transport=True)
    unavailable = await _router(unavailable_model, _registry(CancelOrders)).route(
        _context(CapabilityId.CANCEL_ORDERS)
    )

    assert invalid.resolution == RoutingFailure(reason="invalid_output")
    assert invalid.provider_call_outcome == "completed"
    assert unavailable.resolution == RoutingFailure(reason="routing_unavailable")
    assert unavailable.provider_call_outcome == "provider_error"
    assert unavailable.provider_error_category == "ConnectionError"
    assert unavailable_model.invoke_count == 1


async def test_semantic_router_rejects_oversized_input_without_truncation_or_provider_work() -> (
    None
):
    registry = _registry(CancelOrders)
    rejected_model = FakeChatModel(record_prompts=True)
    rejected_context = _context(CapabilityId.CANCEL_ORDERS).model_copy(
        update={"utterance": "x" * 11}
    )

    rejected = await _router(rejected_model, registry, input_max_chars=10).route(rejected_context)

    assert rejected.resolution == RoutingFailure(reason="context_invalid")
    assert rejected.input_max_chars == 10
    assert rejected.provider_call_outcome == "not_attempted"
    assert rejected_model.invoke_count == 0
    assert not rejected_model._seen_prompts

    accepted_model = FakeChatModel(record_prompts=True)
    accepted_context = rejected_context.model_copy(update={"utterance": "x" * 10})
    accepted = await _router(
        accepted_model,
        _registry(SearchCatalog),
        input_max_chars=10,
    ).route(
        accepted_context.model_copy(
            update={"available_capabilities": (CapabilityId.SEARCH_CATALOG,)}
        )
    )

    assert isinstance(accepted.resolution, RouteDecision)
    assert accepted_model.invoke_count == 1
    assert "x" * 10 in accepted_model._seen_prompts[-1]


async def test_semantic_router_timeout_is_closed_but_external_cancellation_propagates() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def block(_messages):
        started.set()
        await release.wait()
        return {}

    class BlockingModel(FakeChatModel):
        def with_structured_output(self, schema, *, include_raw=False, **kwargs):
            return RunnableLambda(block)

    async def provider_timeout(_messages):
        raise TimeoutError("upstream transport timed out")

    class ProviderTimeoutModel(FakeChatModel):
        def with_structured_output(self, schema, *, include_raw=False, **kwargs):
            return RunnableLambda(provider_timeout)

    registry = _registry(CancelOrders)
    provider_timed_out = await _router(ProviderTimeoutModel(), registry).route(
        _context(CapabilityId.CANCEL_ORDERS)
    )
    assert provider_timed_out.resolution == RoutingFailure(reason="routing_unavailable")
    assert provider_timed_out.provider_call_outcome == "provider_error"
    assert provider_timed_out.provider_error_category == "TimeoutError"

    timed_out = await _router(BlockingModel(), registry, timeout_seconds=0.01).route(
        _context(CapabilityId.CANCEL_ORDERS)
    )
    assert timed_out.resolution == RoutingFailure(reason="routing_unavailable")
    assert timed_out.timeout_seconds == 0.01
    assert timed_out.provider_call_outcome == "deadline_exceeded"
    assert timed_out.provider_error_category == "TimeoutError"

    started.clear()
    task = asyncio.create_task(
        _router(BlockingModel(), registry, timeout_seconds=1.0).route(
            _context(CapabilityId.CANCEL_ORDERS)
        )
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def test_structured_wrapper_configuration_failure_is_not_a_turn_failure() -> None:
    class BrokenConfigurationModel(FakeChatModel):
        def with_structured_output(self, schema, *, include_raw=False, **kwargs):
            raise ValueError("unsupported structured-output method")

    with pytest.raises(ValueError, match="unsupported structured-output method"):
        _router(BrokenConfigurationModel(), _registry(CancelOrders))

    with pytest.raises(ValueError, match="input limit must be positive"):
        _router(FakeChatModel(), _registry(CancelOrders), input_max_chars=0)
