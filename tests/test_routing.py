from __future__ import annotations

import asyncio

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from llm_fakes import TEST_STRUCTURED_OUTPUT_METHOD, FakeChatModel

from agnostic_market.agents.capabilities import (
    CapabilityEntry,
    CapabilityRegistry,
    CapabilitySpec,
)
from agnostic_market.agents.routing import (
    CONTEXT_PROJECTOR_VERSION,
    ROUTE_SCHEMA_FINGERPRINT,
    ROUTER_PROMPT_FINGERPRINT,
    ROUTER_SYSTEM_PROMPT,
    RoutingAttempt,
    SemanticRouter,
    materialize_route,
    project_routing_context,
    registry_fingerprint,
    resolve_route,
)
from agnostic_market.commerce.cart import CartStore
from agnostic_market.commerce.identity import BoundIdentity, CallerIdentityStore
from agnostic_market.commerce.orders import RecentOrderContext
from agnostic_market.dtos.config import ProviderModel
from agnostic_market.dtos.events import CommittedTurn
from agnostic_market.dtos.orchestration import (
    ActiveInvocation,
    AnswerQuestion,
    CancelOrders,
    CapabilityId,
    ChangeProfile,
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
from agnostic_market.dtos.state import ReasoningState

_SELECTION = ProviderModel(provider="fake", model="router")


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
) -> RoutingContext:
    return RoutingContext(
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


def test_route_materializer_covers_every_capability_from_one_coarse_contract() -> None:
    context = _context(*CapabilityId)
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
            RouteProposal(decision="direct", capability=CapabilityId.CANCEL_ORDERS),
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
        "ba46871d7a2f5b707e5caa5527aed4a7a5b374d1e6871b30fbfbcea23e58f0b1"
    )
    assert all(meaning_block.count(capability_id.value) == 1 for capability_id in CapabilityId)


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
    }


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

    timed_out = await _router(BlockingModel(), registry, timeout_seconds=0.01).route(
        _context(CapabilityId.CANCEL_ORDERS)
    )
    assert timed_out.resolution == RoutingFailure(reason="routing_unavailable")
    assert timed_out.timeout_seconds == 0.01
    assert timed_out.provider_call_outcome == "deadline_exceeded"

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
