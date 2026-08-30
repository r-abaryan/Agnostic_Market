"""Frontline graph structural safety, typed routing, and owner contracts. Zero network."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from types import MappingProxyType
from typing import NoReturn, get_args

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END
from langgraph.types import Command
from llm_fakes import (
    TEST_CALLER_AUDIBLE_MODEL_TEXT_MAX_CHARS,
    TEST_STRUCTURED_OUTPUT_METHOD,
    FakeChatModel,
    NativeAsyncOnlyFakeChatModel,
)
from policy_helpers import make_policy
from telemetry_helpers import make_session_telemetry

from agnostic_market.agents.frontline import build_frontline_graph, read_flow
from agnostic_market.agents.frontline import graph as frontline_graph
from agnostic_market.agents.recovery import (
    AUTOMATION_TERMINAL_LINE,
    RECOVERY_NODE_NAME,
    RECOVERY_TERMINALIZER_NODE_NAME,
    clear_automation_state,
)
from agnostic_market.agents.support import flow as support_flow
from agnostic_market.agents.telemetry import (
    InMemoryTelemetrySink,
    TelemetryPurpose,
    TenantTelemetry,
)
from agnostic_market.commerce.cart import CartStore
from agnostic_market.commerce.catalog import FixtureCatalog
from agnostic_market.commerce.identity import (
    BoundIdentity,
    CallerIdentityStore,
    CustomerDirectory,
    load_customers_fixture,
)
from agnostic_market.commerce.orders import (
    GuestOrderScope,
    OrdersFixture,
    OrderStore,
    RecentOrderContext,
    load_orders_fixture,
    render_cart_line,
)
from agnostic_market.commerce.payment_instruments import (
    PaymentInstrumentDirectory,
    load_payment_instruments_fixture,
)
from agnostic_market.commerce.profile import ProfileStore, load_profile_fixture
from agnostic_market.commerce.receipts import CommittedReceipt
from agnostic_market.commerce.verification import OtpProvider, RiskProvider, VerificationStore
from agnostic_market.dtos.llm import StructuredOutputMethod
from agnostic_market.dtos.orchestration import (
    AbortCurrent,
    ActiveInvocation,
    AnswerQuestion,
    CancelOrders,
    CapabilityDispatchEnvelope,
    CapabilityId,
    CartItemChoices,
    CartItemQuery,
    ClarificationReason,
    DiscloseAiIdentity,
    ExplicitOrderSet,
    FocusedOrderSet,
    IntentRequest,
    InvocationClarificationOwner,
    ListOrders,
    ModifyCart,
    PlaceOrder,
    RecentOrderSet,
    RequestPerson,
    ResolvedCartItemRef,
    RoutingFailureReason,
    SearchCatalog,
    SwitchAccount,
    VerifyIdentity,
    VerifyOrderStatus,
    ViewCart,
    ViewIdentityStatus,
)
from agnostic_market.dtos.recovery import AbandonmentKind, ExceptionAction
from agnostic_market.dtos.state import (
    CartClarification,
    ClarificationLiveness,
    HandoffSource,
    PendingCartMutation,
    ReasoningState,
)
from agnostic_market.session import CallerContext

# A DEFERRING destination (planner) — these tests exercise the destination-agnostic handover
_READ_ARGS = {"order_status": {"order_id": "ORD-1001"}, "catalog_search": {"query": "shoes"}}
_TEST_OTP = "482913"
_DISPATCH_REJECTION_LINE = "I couldn't complete that request. Please try again."


def test_router_no_action_copy_exactly_covers_closed_reasons() -> None:
    expected = set(get_args(ClarificationReason)) | set(get_args(RoutingFailureReason))
    assert set(frontline_graph._ROUTER_NO_ACTION_LINES) == expected


def _granted(*order_ids: str) -> CallerIdentityStore:
    """A session identity store with rung-1 grants — for tests exercising what happens
    AFTER an authorized order read (the L3 render path), not the gate itself."""
    identity = CallerIdentityStore()
    identity.grant_orders(*order_ids)
    return identity


def _graph(config_root: Path, fake: FakeChatModel, **kwargs):
    fixture = load_orders_fixture(config_root, "acme_store")
    store = kwargs.pop("store", None) or OrderStore("acme_store", fixture.orders)
    catalog = kwargs.pop("catalog", None) or FixtureCatalog("acme_store", fixture)
    # Routing projection and graph owners share these session stores.
    cart = kwargs.pop("cart_store", None) or CartStore()
    policy = kwargs.pop("policy", None) or make_policy(refund_returnless_under_usd=50.0)
    recent_orders = kwargs.pop("recent_orders", None) or RecentOrderContext(
        max_refs=policy.cancel_batch_max
    )
    identity = kwargs.pop("identity", None) or CallerIdentityStore()
    otp = kwargs.pop("otp", None) or OtpProvider("acme_store", valid_code=_TEST_OTP)
    verification = kwargs.pop("verification_store", None) or VerificationStore(otp)
    telemetry = kwargs.pop("telemetry", None) or make_session_telemetry(
        "acme_store", "frontline-graph"
    )
    guest_orders = kwargs.pop("guest_orders", None) or GuestOrderScope(
        tenant_id="acme_store",
        session_id=telemetry.session_id,
    )
    caller_context = CallerContext(
        verification_store=verification,
        cart_store=cart,
        recent_orders=recent_orders,
        identity_store=identity,
        guest_orders=guest_orders,
        telemetry=telemetry.operational,
    )
    assembly = build_frontline_graph(
        fake,
        display_name="Acme Store",
        tenant_id="acme_store",
        guest_orders=guest_orders,
        cart_store=cart,
        recent_orders=recent_orders,
        verification_store=verification,
        risk=kwargs.pop("risk", None) or RiskProvider("acme_store"),
        identity_store=identity,
        customers=kwargs.pop("customers", None)
        or CustomerDirectory("acme_store", load_customers_fixture(config_root, "acme_store")),
        payment_instruments=kwargs.pop("payment_instruments", None)
        or PaymentInstrumentDirectory(
            "acme_store", load_payment_instruments_fixture(config_root, "acme_store")
        ),
        profile_store=kwargs.pop("profile_store", None)
        or ProfileStore("acme_store", load_profile_fixture(config_root, "acme_store")),
        # Frontline-path tests never reach checkout; a default fake keeps one graph shape.
        reasoning_model=kwargs.pop("reasoning_model", None) or FakeChatModel(),
        store=store,
        catalog=catalog,
        policy=policy,
        lifecycle=caller_context,
        structured_output_method=kwargs.pop(
            "structured_output_method", TEST_STRUCTURED_OUTPUT_METHOD
        ),
        caller_audible_model_text_max_chars=kwargs.pop(
            "caller_audible_model_text_max_chars",
            TEST_CALLER_AUDIBLE_MODEL_TEXT_MAX_CHARS,
        ),
        response_model_node_timeout_seconds=kwargs.pop("response_model_node_timeout_seconds", 2.0),
        reasoning_model_node_timeout_seconds=kwargs.pop(
            "reasoning_model_node_timeout_seconds", 6.0
        ),
        session_telemetry=telemetry,
        **kwargs,
    )
    graph = assembly.graph
    graph._test_telemetry_sinks = (
        telemetry.operational.sink,
        telemetry.routing_evidence.sink,
    )
    return graph


@pytest.mark.parametrize(
    "dependency",
    (
        "catalog",
        "store",
        "guest_orders",
        "customers",
        "payment_instruments",
        "profile_store",
        "otp",
        "risk",
        "telemetry",
    ),
)
def test_graph_rejects_cross_tenant_dependencies(config_root: Path, dependency: str) -> None:
    fixture = load_orders_fixture(config_root, "acme_store")
    dependencies = {
        "catalog": FixtureCatalog("other_store", fixture),
        "store": OrderStore("other_store", fixture.orders),
        "guest_orders": GuestOrderScope(tenant_id="other_store", session_id="foreign"),
        "customers": CustomerDirectory(
            "other_store", load_customers_fixture(config_root, "acme_store")
        ),
        "payment_instruments": PaymentInstrumentDirectory(
            "other_store", load_payment_instruments_fixture(config_root, "acme_store")
        ),
        "profile_store": ProfileStore(
            "other_store", load_profile_fixture(config_root, "acme_store")
        ),
        "otp": OtpProvider("other_store", valid_code=_TEST_OTP),
        "risk": RiskProvider("other_store"),
        "telemetry": make_session_telemetry("other_store", "frontline-graph"),
    }

    with pytest.raises(ValueError, match="tenant dependencies do not match"):
        _graph(config_root, FakeChatModel(), **{dependency: dependencies[dependency]})


def test_graph_rejects_telemetry_from_another_session(config_root: Path) -> None:
    telemetry = make_session_telemetry("acme_store", "other-session")
    guest_orders = GuestOrderScope(tenant_id="acme_store", session_id="frontline-graph")

    with pytest.raises(ValueError, match="session dependencies do not match"):
        _graph(
            config_root,
            FakeChatModel(),
            telemetry=telemetry,
            guest_orders=guest_orders,
        )


def _admitted_turn(text: str, *, turn_id: str, **state: object) -> dict[str, object]:
    return {
        "messages": [HumanMessage(content=text, id=turn_id)],
        "consumed_turn_ids": (turn_id,),
        **state,
    }


# --- the structural safety invariant (T1's structural half) --------------------------


def test_frontline_has_no_broad_model_or_tool_routing_surface(config_root: Path) -> None:
    graph = _graph(config_root, FakeChatModel())
    assert not hasattr(graph, "frontline_read_only_tools")
    assert {
        "gate",
        "model",
        "tools",
        "cross_switch",
        "read_render",
        "forced_status",
        "enumeration_gate",
    }.isdisjoint(graph.nodes)


@pytest.mark.parametrize(
    ("model_text", "max_chars"),
    (
        ("\u200b", TEST_CALLER_AUDIBLE_MODEL_TEXT_MAX_CHARS),
        ("x" * 41, 40),
    ),
)
async def test_typed_answer_owner_rejects_invalid_caller_audible_text(
    config_root: Path,
    model_text: str,
    max_chars: int,
) -> None:
    graph = _graph(
        config_root,
        FakeChatModel(emit_tool_calls=False, text_response=model_text),
        caller_audible_model_text_max_chars=max_chars,
    )

    result = await _typed_read(
        graph,
        AnswerQuestion(topic="general"),
        turn_id="bounded-model",
        text="Tell me a joke about shoes.",
    )

    assert _failed_nodes(graph) == ["answer_response"]
    assert "hit a snag" in _only_spoken(result)
    assert model_text not in _only_spoken(result)
    assert _answered_rows(graph) == []


def test_support_capability_registry_and_dispatch_topology_are_closed(
    config_root: Path,
) -> None:
    graph = _graph(config_root, FakeChatModel())
    registry = graph.capability_registry

    assert registry.capability_ids == (
        CapabilityId.LIST_ORDERS,
        CapabilityId.CANCEL_ORDERS,
        CapabilityId.REFUND_ORDER,
        CapabilityId.RETURN_ORDER,
        CapabilityId.CHANGE_PROFILE,
        CapabilityId.VIEW_CART,
        CapabilityId.VIEW_IDENTITY_STATUS,
        CapabilityId.VERIFY_IDENTITY,
        CapabilityId.SWITCH_ACCOUNT,
        CapabilityId.MODIFY_CART,
        CapabilityId.PLACE_ORDER,
        CapabilityId.SEARCH_CATALOG,
        CapabilityId.ANSWER_QUESTION,
        CapabilityId.VERIFY_ORDER_STATUS,
        CapabilityId.ABORT_CURRENT,
        CapabilityId.REQUEST_PERSON,
    )
    assert registry.entry_nodes == (
        "support_capability_entry",
        "cart_view_render",
        "identity_status_render",
        "identity_capability_entry",
        "cart_capability_entry",
        "catalog_entry",
        "answer_response",
        "order_status_entry",
        "abort_current",
        "request_person",
    )
    # Rendering hint only, and DERIVED: it must equal the registry's own entry nodes, and the
    # dispatcher must still own no executable outgoing route.
    assert graph.builder.nodes["capability_dispatch"].ends == registry.entry_nodes
    assert not any(source == "capability_dispatch" for source, _target in graph.builder.edges)
    assert "capability_dispatch" not in graph.builder.branches
    # An unmigrated id must be ABSENT, never resolving to some fallback owner.
    # The unmigrated set is named, not counted: a count stays green if one id gains an owner
    # while another is added, and it names nothing when it breaks.
    assert set(CapabilityId) - set(registry.capability_ids) == {
        CapabilityId.DISCLOSE_AI_IDENTITY,
    }
    assert registry.resolve(SearchCatalog(query="shoes")).node_name == "catalog_entry"
    assert graph.builder.nodes["catalog_entry"].ends == (
        "catalog_response",
        "catalog_query_reject",
    )
    assert graph.builder.nodes["catalog_response"].ends == (END,)
    assert registry.resolve(AnswerQuestion(topic="policy")).node_name == "answer_response"
    assert graph.builder.nodes["answer_response"].ends == (
        "answer_clarify",
        "answer_unsupported",
        END,
    )
    for command_node in ("catalog_entry", "catalog_response"):
        assert not any(source == command_node for source, _target in graph.builder.edges)
        assert command_node not in graph.builder.branches
    assert not any(source == "answer_response" for source, _target in graph.builder.edges)
    assert "answer_response" not in graph.builder.branches
    assert registry.resolve(VerifyOrderStatus()).node_name == "order_status_entry"
    assert registry.resolve(AbortCurrent()).node_name == "abort_current"
    assert registry.resolve(RequestPerson()).node_name == "request_person"
    assert graph.builder.nodes["order_status_entry"].ends == (
        "order_status_target_ask",
        "order_status_target_propose",
        "order_status_fulfill",
    )
    assert graph.builder.nodes["order_status_target_propose"].ends == (
        "order_status_target_reject",
        "order_status_fulfill",
    )
    assert graph.builder.nodes["order_status_target_confirm"].ends == (
        "order_status_target_reject",
        "order_status_fulfill",
        "handover",
    )
    assert graph.builder.nodes["order_status_fulfill"].ends == (
        "order_status_target_ask",
        "order_status_target_confirm",
        "order_status_target_reject",
        "order_status_fulfill",
        END,
    )
    for command_node in (
        "order_status_entry",
        "order_status_target_propose",
        "order_status_target_confirm",
        "order_status_fulfill",
    ):
        assert not any(source == command_node for source, _target in graph.builder.edges)
        assert command_node not in graph.builder.branches


def test_direct_dispatch_envelope_opens_invocation_and_routes_atomically(
    config_root: Path,
) -> None:
    graph = _graph(config_root, FakeChatModel())
    envelope = CapabilityDispatchEnvelope(
        turn_id="dispatch-direct",
        mode="direct",
        request=ViewCart(),
    )

    command = graph.nodes["capability_dispatch"].invoke(
        ReasoningState(
            messages=[HumanMessage("what is in my cart?", id="dispatch-direct")],
            consumed_turn_ids=("dispatch-direct",),
            pending_capability_dispatch=envelope,
        )
    )

    assert isinstance(command, Command)
    assert command.goto == "cart_view_render"
    assert command.update["pending_capability_dispatch"] is None
    invocation = command.update["active_invocation"]
    assert isinstance(invocation, ActiveInvocation)
    assert invocation.request == ViewCart()
    assert invocation.opened_turn_id == "dispatch-direct"


def test_compiled_entry_consumes_direct_dispatch_without_ordinary_routing(
    config_root: Path,
) -> None:
    frontline = FakeChatModel(emit_tool_calls=False, text_response="ordinary route ran")
    reasoning = FakeChatModel(emit_tool_calls=False)
    graph = _graph(config_root, frontline, reasoning_model=reasoning)
    turn_id = "dispatch-compiled"

    result = graph.invoke(
        _admitted_turn(
            "what is in my cart?",
            turn_id=turn_id,
            pending_capability_dispatch=CapabilityDispatchEnvelope(
                turn_id=turn_id,
                mode="direct",
                request=ViewCart(),
            ),
        )
    )

    assert frontline.invoke_count == 0 and reasoning.invoke_count == 0
    assert result["pending_capability_dispatch"] is None
    assert result["active_invocation"] is None
    assert _only_spoken(result) == "Your cart's empty at the moment."


def test_continue_dispatch_envelope_requires_and_preserves_the_observed_invocation(
    config_root: Path,
) -> None:
    graph = _graph(config_root, FakeChatModel())
    invocation = ActiveInvocation(request=ModifyCart(operation="add"), opened_turn_id="turn-1")
    envelope = CapabilityDispatchEnvelope(
        turn_id="turn-2",
        mode="continue",
        observed_invocation_id=invocation.invocation_id,
    )

    command = graph.nodes["capability_dispatch"].invoke(
        ReasoningState(
            messages=[HumanMessage("the shoes", id="turn-2")],
            consumed_turn_ids=("turn-1", "turn-2"),
            active_invocation=invocation,
            execution_owner="cart",
            clarification_liveness=ClarificationLiveness(
                owner=InvocationClarificationOwner(invocation_id=invocation.invocation_id),
                reasks=1,
            ),
            pending_capability_dispatch=envelope,
        )
    )

    assert isinstance(command, Command)
    assert command.goto == "cart_capability_entry"
    assert command.update == {"pending_capability_dispatch": None}


def test_continue_dispatch_retains_identity_detour_execution_ownership(
    config_root: Path,
) -> None:
    graph = _graph(config_root, FakeChatModel())
    invocation = ActiveInvocation(request=CancelOrders(), opened_turn_id="turn-1")

    command = graph.nodes["capability_dispatch"].invoke(
        ReasoningState(
            messages=[HumanMessage("casey@example.com", id="turn-2")],
            consumed_turn_ids=("turn-1", "turn-2"),
            active_invocation=invocation,
            execution_owner="identity",
            pending_capability_dispatch=CapabilityDispatchEnvelope(
                turn_id="turn-2",
                mode="continue",
                observed_invocation_id=invocation.invocation_id,
            ),
        )
    )

    assert isinstance(command, Command)
    assert command.goto == "identity_assemble"
    assert command.update == {"pending_capability_dispatch": None}


def test_direct_dispatch_replaces_invocation_and_clears_old_liveness(
    config_root: Path,
) -> None:
    graph = _graph(config_root, FakeChatModel())
    old = ActiveInvocation(request=ModifyCart(operation="add"), opened_turn_id="turn-1")
    envelope = CapabilityDispatchEnvelope(
        turn_id="turn-2",
        mode="direct",
        request=ViewCart(),
    )

    command = graph.nodes["capability_dispatch"].invoke(
        ReasoningState(
            messages=[HumanMessage("show my cart", id="turn-2")],
            consumed_turn_ids=("turn-1", "turn-2"),
            active_invocation=old,
            execution_owner="cart",
            identity_claim_misses=1,
            pending_ack="old response",
            pending_clarification=CartClarification(detail="item"),
            clarification_liveness=ClarificationLiveness(
                owner=InvocationClarificationOwner(invocation_id=old.invocation_id),
                reasks=1,
            ),
            pending_capability_dispatch=envelope,
        )
    )

    assert isinstance(command, Command)
    assert command.goto == "cart_view_render"
    replacement = command.update["active_invocation"]
    assert isinstance(replacement, ActiveInvocation)
    assert replacement.invocation_id != old.invocation_id
    assert replacement.request == ViewCart()
    assert {
        "execution_owner": command.update["execution_owner"],
        "identity_claim_misses": command.update["identity_claim_misses"],
        "pending_ack": command.update["pending_ack"],
        "pending_clarification": command.update["pending_clarification"],
        "clarification_liveness": command.update["clarification_liveness"],
    } == {
        "execution_owner": None,
        "identity_claim_misses": 0,
        "pending_ack": None,
        "pending_clarification": None,
        "clarification_liveness": None,
    }


@pytest.mark.parametrize(
    "failure",
    ("stale_turn", "stale_invocation", "unregistered", "malformed_envelope"),
)
def test_invalid_dispatch_envelope_closes_without_executing_an_owner(
    config_root: Path,
    failure: str,
) -> None:
    sink = InMemoryTelemetrySink()
    telemetry = TenantTelemetry("acme_store", sink, sink).bind_session("invalid-dispatch")
    graph = _graph(config_root, FakeChatModel(), telemetry=telemetry)
    invocation = ActiveInvocation(request=ViewCart(), opened_turn_id="turn-1")
    envelope: object = (
        CapabilityDispatchEnvelope(
            turn_id="stale-turn",
            mode="direct",
            request=ViewCart(),
        )
        if failure == "stale_turn"
        else CapabilityDispatchEnvelope(
            turn_id="turn-2",
            mode="continue",
            observed_invocation_id="stale-invocation",
        )
        if failure == "stale_invocation"
        else CapabilityDispatchEnvelope(
            turn_id="turn-2",
            mode="direct",
            request=DiscloseAiIdentity(),
        )
        if failure == "unregistered"
        else object()
    )

    command = graph.nodes["capability_dispatch"].invoke(
        ReasoningState.model_construct(
            messages=[HumanMessage("current", id="turn-2")],
            consumed_turn_ids=("turn-1", "turn-2"),
            active_invocation=invocation,
            pending_capability_dispatch=envelope,
        )
    )

    assert isinstance(command, Command)
    assert command.goto == END
    assert command.update == {
        **clear_automation_state(),
        "messages": [AIMessage(_DISPATCH_REJECTION_LINE)],
    }
    records = [{"event": record.event, **record.attributes} for record in sink.records]
    assert records == [
        {
            "event": "capability_dispatch_rejected",
            "reason": failure,
            "disposition": "closed",
        }
    ]


async def test_complete_typed_cart_add_resolves_live_catalog_without_a_model_call(
    config_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = load_orders_fixture(config_root, "acme_store")
    catalog = FixtureCatalog("acme_store", fixture)
    store = OrderStore("acme_store", fixture.orders)
    cart = CartStore()
    reasoning = FakeChatModel(emit_tool_calls=False)
    sink = InMemoryTelemetrySink()
    telemetry = TenantTelemetry("acme_store", sink, sink).bind_session("typed-cart-add")
    searches: list[str] = []
    resolutions: list[tuple[str, ...]] = []
    search = catalog.search
    resolve_products = catalog.resolve_products

    def observed_search(query: str):
        searches.append(query)
        return search(query)

    def observed_resolution(skus: tuple[str, ...]):
        resolutions.append(skus)
        return resolve_products(skus)

    monkeypatch.setattr(catalog, "search", observed_search)
    monkeypatch.setattr(catalog, "resolve_products", observed_resolution)
    graph = _graph(
        config_root,
        FakeChatModel(),
        store=store,
        catalog=catalog,
        cart_store=cart,
        reasoning_model=reasoning,
        telemetry=telemetry,
    )
    product = fixture.products[0]
    turn_id = "typed-cart-add"

    result = await graph.ainvoke(
        _admitted_turn(
            f"add one {product.name}",
            turn_id=turn_id,
            active_invocation=ActiveInvocation(
                request=ModifyCart(
                    operation="add",
                    item=CartItemQuery(query=product.name),
                    quantity=1,
                ),
                opened_turn_id=turn_id,
            ),
        )
    )

    assert reasoning.invoke_count == 0
    assert cart.is_empty()
    assert result["active_invocation"] is None
    assert result["execution_owner"] == "cart"
    pending = result["pending_cart_mutation"]
    assert isinstance(pending, PendingCartMutation)
    assert pending.sku == product.sku
    assert pending.price_usd == product.price_usd
    assert searches == [product.name]
    assert resolutions == [(product.sku,)]
    assert result["__interrupt__"][0].value == (
        f"Just to confirm: add 1 of {product.name} to your cart?"
    )
    assert sink.records == ()


async def test_resolved_typed_cart_add_revalidates_the_live_catalog_before_effect(
    config_root: Path,
) -> None:
    store = OrderStore("acme_store", load_orders_fixture(config_root, "acme_store").orders)
    cart = CartStore()
    graph = _graph(config_root, FakeChatModel(), store=store, cart_store=cart)
    product = load_orders_fixture(config_root, "acme_store").products[0]
    turn_id = "typed-cart-resolved"

    result = await graph.ainvoke(
        _admitted_turn(
            "add it",
            turn_id=turn_id,
            active_invocation=ActiveInvocation(
                request=ModifyCart(
                    operation="add",
                    item=ResolvedCartItemRef(sku=product.sku),
                    quantity=1,
                ),
                opened_turn_id=turn_id,
            ),
        )
    )

    assert cart.is_empty()
    assert result["active_invocation"] is None
    pending = result["pending_cart_mutation"]
    assert isinstance(pending, PendingCartMutation)
    assert pending.sku == product.sku


async def test_typed_cart_gathers_one_slot_per_committed_turn(config_root: Path) -> None:
    store = OrderStore("acme_store", load_orders_fixture(config_root, "acme_store").orders)
    cart = CartStore()
    reasoning = NativeAsyncOnlyFakeChatModel(
        scripted_calls=[
            [("provide_cart_item", {"candidate_key": "1"})],
            [("provide_cart_quantity", {"quantity": 2})],
        ]
    )
    graph = _graph(
        config_root,
        FakeChatModel(),
        store=store,
        cart_store=cart,
        reasoning_model=reasoning,
    )
    invocation = ActiveInvocation(
        request=ModifyCart(operation="add"),
        opened_turn_id="typed-cart-item",
    )
    first_state = ReasoningState(
        messages=[HumanMessage("add one of those", id="typed-cart-item")],
        consumed_turn_ids=("typed-cart-item",),
        active_invocation=invocation,
    )

    first = await graph.nodes["cart_capability_entry"].ainvoke(first_state)

    retained = first["active_invocation"]
    assert isinstance(retained, ActiveInvocation)
    assert isinstance(retained.request, ModifyCart)
    assert isinstance(retained.request.item, ResolvedCartItemRef)
    assert retained.request.quantity is None
    assert first["pending_clarification"].detail == "quantity"
    assert cart.is_empty()
    assert reasoning.invoke_count == 1

    second_state = ReasoningState(
        messages=[
            *first_state.messages,
            *first["messages"],
            HumanMessage("make it two", id="typed-cart-quantity"),
        ],
        consumed_turn_ids=("typed-cart-item", "typed-cart-quantity"),
        active_invocation=retained,
        execution_owner="cart",
        clarification_liveness=first["clarification_liveness"],
    )
    second = await graph.nodes["cart_capability_entry"].ainvoke(second_state)

    assert second["active_invocation"] is None
    assert second["execution_owner"] == "cart"
    pending = second["pending_cart_mutation"]
    assert isinstance(pending, PendingCartMutation)
    assert pending.quantity == 2
    assert pending.pre_confirm_quantity == 0
    assert cart.is_empty()
    assert reasoning.invoke_count == 2


@pytest.mark.parametrize(
    ("operation", "quantity"),
    (
        ("set_quantity", 3),
        ("set_quantity", 0),
        ("remove", None),
    ),
)
async def test_typed_cart_remove_and_set_resolve_only_live_cart_lines(
    config_root: Path,
    operation: str,
    quantity: int | None,
) -> None:
    store = OrderStore("acme_store", load_orders_fixture(config_root, "acme_store").orders)
    product = load_orders_fixture(config_root, "acme_store").products[0]
    cart = CartStore()
    cart.add_item(
        sku=product.sku,
        name=product.name,
        price_usd=product.price_usd,
        quantity=1,
    )
    reasoning = FakeChatModel(emit_tool_calls=False)
    sink = InMemoryTelemetrySink()
    telemetry = TenantTelemetry("acme_store", sink, sink).bind_session("typed-cart-update")
    graph = _graph(
        config_root,
        FakeChatModel(),
        store=store,
        cart_store=cart,
        reasoning_model=reasoning,
        telemetry=telemetry,
    )
    turn_id = f"typed-cart-{operation}-{quantity}"
    request = ModifyCart(
        operation=operation,
        item=CartItemQuery(query=product.name),
        quantity=quantity,
    )

    result = await graph.ainvoke(
        _admitted_turn(
            f"{operation} {product.name}",
            turn_id=turn_id,
            active_invocation=ActiveInvocation(request=request, opened_turn_id=turn_id),
        )
    )

    assert reasoning.invoke_count == 0
    assert result["active_invocation"] is None
    assert cart.view()[0].quantity == 1
    pending = result["pending_cart_mutation"]
    assert isinstance(pending, PendingCartMutation)
    assert pending.operation == operation
    assert pending.quantity == quantity
    assert pending.pre_confirm_quantity == 1
    assert sink.records == ()


async def test_typed_cart_no_match_resets_only_item_and_asks_in_code(config_root: Path) -> None:
    reasoning = FakeChatModel(emit_tool_calls=False)
    graph = _graph(config_root, FakeChatModel(), reasoning_model=reasoning)
    turn_id = "typed-cart-no-match"
    request = ModifyCart(
        operation="add",
        item=CartItemQuery(query="product that does not exist"),
        quantity=2,
    )

    result = await graph.ainvoke(
        _admitted_turn(
            "add two unavailable things",
            turn_id=turn_id,
            active_invocation=ActiveInvocation(request=request, opened_turn_id=turn_id),
        )
    )

    retained = result["active_invocation"]
    assert isinstance(retained, ActiveInvocation)
    assert retained.request == ModifyCart(operation="add", quantity=2)
    assert result["execution_owner"] == "cart"
    assert _only_spoken(result) == "Which item would you like?"
    assert reasoning.invoke_count == 0


async def test_typed_cart_duplicate_name_selection_retains_the_resolved_sku(
    config_root: Path,
) -> None:
    fixture = load_orders_fixture(config_root, "acme_store")
    payload = fixture.model_dump()
    duplicate = {
        **payload["products"][0],
        "sku": "SKU-DUPLICATE",
        "price_usd": payload["products"][0]["price_usd"] + 10,
    }
    payload["products"] = (*payload["products"], duplicate)
    custom_fixture = OrdersFixture.model_validate(payload)
    store = OrderStore("acme_store", custom_fixture.orders)
    catalog = FixtureCatalog("acme_store", custom_fixture)
    reasoning = FakeChatModel(
        scripted_calls=[
            [("provide_cart_item", {"candidate_key": "2"})],
            [("provide_cart_quantity", {"quantity": 2})],
        ],
        record_prompts=True,
    )
    cart = CartStore()
    graph = _graph(
        config_root,
        FakeChatModel(),
        store=store,
        catalog=catalog,
        cart_store=cart,
        reasoning_model=reasoning,
    )
    name = custom_fixture.products[0].name
    invocation = ActiveInvocation(
        request=ModifyCart(
            operation="add",
            item=CartItemQuery(query=name),
        ),
        opened_turn_id="typed-cart-duplicate",
    )
    first = await graph.nodes["cart_capability_entry"].ainvoke(
        ReasoningState(
            messages=[HumanMessage(f"add the {name}", id="typed-cart-duplicate")],
            consumed_turn_ids=("typed-cart-duplicate",),
            active_invocation=invocation,
        )
    )

    retained = first["active_invocation"]
    assert isinstance(retained, ActiveInvocation)
    assert retained.request == ModifyCart(
        operation="add",
        item=CartItemChoices(skus=(custom_fixture.products[0].sku, "SKU-DUPLICATE")),
    )
    assert first["pending_clarification"].detail == "item"
    assert reasoning.invoke_count == 0

    clarification = graph.nodes["cart_clarify"].invoke(
        ReasoningState(
            active_invocation=retained,
            consumed_turn_ids=("typed-cart-duplicate",),
            pending_clarification=first["pending_clarification"],
        )
    )
    spoken_choices = str(clarification["messages"][0].content)
    assert "option 1" in spoken_choices and "option 2" in spoken_choices
    assert name in spoken_choices
    assert "waterproof rain jacket" not in spoken_choices

    second = await graph.nodes["cart_capability_entry"].ainvoke(
        ReasoningState(
            messages=[HumanMessage("the second option", id="typed-cart-selection")],
            consumed_turn_ids=("typed-cart-duplicate", "typed-cart-selection"),
            active_invocation=retained,
            execution_owner="cart",
            clarification_liveness=first["clarification_liveness"],
        )
    )

    selected = second["active_invocation"]
    assert isinstance(selected, ActiveInvocation)
    assert selected.request == ModifyCart(
        operation="add",
        item=ResolvedCartItemRef(sku="SKU-DUPLICATE"),
    )
    assert second["pending_clarification"].detail == "quantity"
    assert cart.is_empty()
    assert reasoning.invoke_count == 1
    assert "waterproof rain jacket" not in reasoning._seen_prompts[-1]

    third = await graph.nodes["cart_capability_entry"].ainvoke(
        ReasoningState(
            messages=[HumanMessage("make it two", id="typed-cart-quantity")],
            consumed_turn_ids=(
                "typed-cart-duplicate",
                "typed-cart-selection",
                "typed-cart-quantity",
            ),
            active_invocation=selected,
            execution_owner="cart",
            clarification_liveness=second["clarification_liveness"],
        )
    )

    assert third["active_invocation"] is None
    assert cart.is_empty()
    pending = third["pending_cart_mutation"]
    assert isinstance(pending, PendingCartMutation)
    assert pending.sku == "SKU-DUPLICATE"
    assert pending.quantity == 2
    assert reasoning.invoke_count == 2


async def test_typed_cart_slot_model_sees_only_the_current_committed_utterance(
    config_root: Path,
) -> None:
    reasoning = FakeChatModel(
        scripted_calls=[[("provide_cart_item", {"candidate_key": "1"})]],
        record_prompts=True,
    )
    graph = _graph(config_root, FakeChatModel(), reasoning_model=reasoning)

    update = await graph.nodes["cart_capability_entry"].ainvoke(
        ReasoningState(
            messages=[
                HumanMessage("add the jacket", id="prior-turn"),
                AIMessage("Which item would you like?"),
                HumanMessage("the first option", id="current-turn"),
            ],
            consumed_turn_ids=("prior-turn", "current-turn"),
            active_invocation=ActiveInvocation(
                request=ModifyCart(operation="add"),
                opened_turn_id="prior-turn",
            ),
        )
    )

    assert update["active_invocation"] is not None
    assert "the first option" in reasoning._seen_prompts[-1]
    assert "add the jacket" not in reasoning._seen_prompts[-1]


async def test_typed_cart_boolean_quantity_performs_no_effect(config_root: Path) -> None:
    reasoning = FakeChatModel(
        scripted_calls=[
            [("provide_cart_quantity", {"quantity": True})],
            [("provide_cart_quantity", {"quantity": True})],
        ]
    )
    cart = CartStore()
    graph = _graph(
        config_root,
        FakeChatModel(),
        cart_store=cart,
        reasoning_model=reasoning,
    )
    product = load_orders_fixture(config_root, "acme_store").products[0]

    update = await graph.nodes["cart_capability_entry"].ainvoke(
        ReasoningState(
            messages=[HumanMessage("add it", id="typed-cart-boolean")],
            consumed_turn_ids=("typed-cart-boolean",),
            active_invocation=ActiveInvocation(
                request=ModifyCart(
                    operation="add",
                    item=ResolvedCartItemRef(sku=product.sku),
                ),
                opened_turn_id="typed-cart-boolean",
            ),
        )
    )

    assert cart.is_empty()
    assert update["pending_clarification"].detail == "quantity"
    assert reasoning.invoke_count == 2


async def test_stale_resolved_cart_sku_performs_no_effect_and_returns_to_selection(
    config_root: Path,
) -> None:
    reasoning = FakeChatModel(emit_tool_calls=False)
    graph = _graph(config_root, FakeChatModel(), reasoning_model=reasoning)
    invocation = ActiveInvocation(
        request=ModifyCart(
            operation="add",
            item=ResolvedCartItemRef(sku="SKU-NOT-LIVE"),
            quantity=1,
        ),
        opened_turn_id="typed-cart-stale",
    )

    update = await graph.nodes["cart_capability_entry"].ainvoke(
        ReasoningState(
            messages=[HumanMessage("add it")],
            consumed_turn_ids=("typed-cart-stale",),
            active_invocation=invocation,
        )
    )

    retained = update["active_invocation"]
    assert isinstance(retained, ActiveInvocation)
    assert retained.request == ModifyCart(operation="add", quantity=1)
    assert update["pending_clarification"].detail == "item"
    assert reasoning.invoke_count == 0


async def test_two_invalid_typed_cart_item_keys_enter_bounded_clarification(
    config_root: Path,
) -> None:
    reasoning = FakeChatModel(
        scripted_calls=[
            [("provide_cart_item", {"candidate_key": "999"})],
            [("provide_cart_item", {"candidate_key": "still-not-valid"})],
        ]
    )
    graph = _graph(config_root, FakeChatModel(), reasoning_model=reasoning)
    invocation = ActiveInvocation(
        request=ModifyCart(operation="add", quantity=1),
        opened_turn_id="typed-cart-invalid",
    )

    update = await graph.nodes["cart_capability_entry"].ainvoke(
        ReasoningState(
            messages=[HumanMessage("add something", id="typed-cart-invalid")],
            consumed_turn_ids=("typed-cart-invalid",),
            active_invocation=invocation,
        )
    )

    assert update["active_invocation"] == invocation
    assert update["pending_clarification"].detail == "item"
    assert reasoning.invoke_count == 2
    tool_uses = {
        call["id"]
        for message in update["messages"]
        if isinstance(message, AIMessage)
        for call in message.tool_calls
    }
    tool_results = {
        message.tool_call_id for message in update["messages"] if isinstance(message, ToolMessage)
    }
    assert tool_uses == tool_results


async def test_typed_cart_rejects_a_fixed_slot_proposal_then_accepts_the_missing_slot(
    config_root: Path,
) -> None:
    reasoning = FakeChatModel(
        scripted_calls=[
            [("provide_cart_quantity", {"quantity": 99})],
            [("provide_cart_item", {"candidate_key": "1"})],
        ]
    )
    cart = CartStore()
    graph = _graph(
        config_root,
        FakeChatModel(),
        cart_store=cart,
        reasoning_model=reasoning,
    )
    invocation = ActiveInvocation(
        request=ModifyCart(operation="add", quantity=2),
        opened_turn_id="typed-cart-fixed-slot",
    )

    update = await graph.nodes["cart_capability_entry"].ainvoke(
        ReasoningState(
            messages=[HumanMessage("add the first one", id="typed-cart-fixed-slot")],
            consumed_turn_ids=("typed-cart-fixed-slot",),
            active_invocation=invocation,
        )
    )

    assert update["active_invocation"] is None
    assert cart.is_empty()
    pending = update["pending_cart_mutation"]
    assert isinstance(pending, PendingCartMutation)
    assert pending.quantity == 2
    assert reasoning.invoke_count == 2
    assert any(
        isinstance(message, ToolMessage) and "Unavailable action" in str(message.content)
        for message in update["messages"]
    )


async def test_typed_cart_model_prose_is_replaced_by_code_clarification(
    config_root: Path,
) -> None:
    fabricated = "I added it to your cart."
    reasoning = FakeChatModel(emit_tool_calls=False, text_response=fabricated)
    graph = _graph(config_root, FakeChatModel(), reasoning_model=reasoning)
    turn_id = "typed-cart-prose"

    result = await graph.ainvoke(
        _admitted_turn(
            "add something",
            turn_id=turn_id,
            active_invocation=ActiveInvocation(
                request=ModifyCart(operation="add", quantity=1),
                opened_turn_id=turn_id,
            ),
        )
    )

    assert _only_spoken(result) == "Which item would you like?"
    assert fabricated not in _only_spoken(result)


async def test_typed_cart_empty_remove_uses_review_ack_and_clears(config_root: Path) -> None:
    reasoning = FakeChatModel(emit_tool_calls=False)
    graph = _graph(config_root, FakeChatModel(), reasoning_model=reasoning)
    turn_id = "typed-cart-empty-remove"

    result = await graph.ainvoke(
        _admitted_turn(
            "remove the shoes",
            turn_id=turn_id,
            active_invocation=ActiveInvocation(
                request=ModifyCart(
                    operation="remove",
                    item=CartItemQuery(query="shoes"),
                ),
                opened_turn_id=turn_id,
            ),
        )
    )

    assert result["active_invocation"] is None
    assert result["execution_owner"] is None
    assert _only_spoken(result) == "Your cart's empty right now - what would you like to add?"
    assert reasoning.invoke_count == 0


async def test_all_typed_cart_exit_shapes_clear_the_invocation(config_root: Path) -> None:
    request = ModifyCart(operation="add")
    invocation = ActiveInvocation(request=request, opened_turn_id="typed-cart-exit")

    leave_graph = _graph(
        config_root,
        FakeChatModel(),
        reasoning_model=FakeChatModel(scripted_calls=[[("leave_cart", {})]]),
    )
    leave = await leave_graph.nodes["cart_capability_entry"].ainvoke(
        ReasoningState(
            messages=[HumanMessage("never mind", id="typed-cart-exit")],
            consumed_turn_ids=("typed-cart-exit",),
            active_invocation=invocation,
        )
    )
    assert leave["active_invocation"] is None

    exhausted_graph = _graph(
        config_root,
        FakeChatModel(),
        reasoning_model=FakeChatModel(emit_tool_calls=False),
    )
    exhausted = await exhausted_graph.nodes["cart_capability_entry"].ainvoke(
        ReasoningState(
            messages=[HumanMessage("I still don't know", id="typed-cart-exit")],
            consumed_turn_ids=("typed-cart-exit",),
            active_invocation=invocation,
            execution_owner="cart",
            clarification_liveness=ClarificationLiveness(
                owner=InvocationClarificationOwner(invocation_id=invocation.invocation_id),
                reasks=2,
            ),
        )
    )
    assert exhausted["active_invocation"] is None
    assert exhausted["execution_owner"] is None

    graph = _graph(config_root, FakeChatModel())
    update = graph.nodes["abort_current"].invoke(
        ReasoningState(
            consumed_turn_ids=("typed-cart-exit",),
            active_invocation=ActiveInvocation(
                request=AbortCurrent(),
                opened_turn_id="typed-cart-exit",
            ),
            execution_owner="cart",
        )
    )
    assert update["active_invocation"] is None


@pytest.mark.parametrize("exit_kind", ("leave", "exhaustion"))
async def test_each_typed_cart_exit_clears_the_invocation_in_compiled_state(
    config_root: Path,
    exit_kind: str,
) -> None:
    reasoning = (
        FakeChatModel(scripted_calls=[[("leave_cart", {})]])
        if exit_kind == "leave"
        else FakeChatModel(emit_tool_calls=False)
    )
    graph = _graph(
        config_root,
        FakeChatModel(emit_tool_calls=False),
        reasoning_model=reasoning,
    )
    text = {
        "leave": "let's discuss something unrelated",
        "exhaustion": "I still cannot choose",
    }[exit_kind]
    state: dict[str, object] = {
        "execution_owner": "cart",
        "active_invocation": ActiveInvocation(
            request=ModifyCart(operation="add"),
            opened_turn_id=f"typed-cart-{exit_kind}",
        ),
    }
    if exit_kind == "exhaustion":
        invocation = state["active_invocation"]
        assert isinstance(invocation, ActiveInvocation)
        state["clarification_liveness"] = ClarificationLiveness(
            owner=InvocationClarificationOwner(invocation_id=invocation.invocation_id),
            reasks=2,
        )

    result = await graph.ainvoke(_admitted_turn(text, turn_id=f"typed-cart-{exit_kind}", **state))

    assert result["active_invocation"] is None


async def test_typed_place_order_snapshot_failure_recovers_without_effect_or_model(
    config_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = OrderStore("acme_store", load_orders_fixture(config_root, "acme_store").orders)
    product = load_orders_fixture(config_root, "acme_store").products[0]
    cart = CartStore()
    cart.add_item(
        sku=product.sku,
        name=product.name,
        price_usd=product.price_usd,
        quantity=1,
    )
    reasoning = FakeChatModel(emit_tool_calls=False)
    graph = _graph(
        config_root,
        FakeChatModel(),
        store=store,
        cart_store=cart,
        reasoning_model=reasoning,
    )

    def fail_snapshot() -> NoReturn:
        raise RuntimeError("injected typed placement snapshot failure")

    monkeypatch.setattr(cart, "snapshot", fail_snapshot)
    turn_id = "typed-place-recovery"
    result = await graph.ainvoke(
        _admitted_turn(
            "place my cart",
            turn_id=turn_id,
            active_invocation=ActiveInvocation(
                request=PlaceOrder(),
                opened_turn_id=turn_id,
            ),
        )
    )

    assert _failed_nodes(graph) == ["cart_capability_entry"]
    assert _only_spoken(result).endswith("Please review your cart before trying again.")
    assert cart.line_count == 1
    assert store.placed_count == 0
    assert reasoning.invoke_count == 0
    assert result["active_invocation"] is None
    assert result["pending_placement"] is None
    assert result.get("pending_recovery") is None
    assert "__interrupt__" not in result


@pytest.mark.parametrize("fails_after_mutation", (False, True))
async def test_typed_cart_recovery_reconciles_without_replaying_mutation(
    config_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    fails_after_mutation: bool,
) -> None:
    store = OrderStore("acme_store", load_orders_fixture(config_root, "acme_store").orders)
    cart = CartStore()
    graph = _graph(
        config_root,
        FakeChatModel(),
        store=store,
        cart_store=cart,
        checkpointer=InMemorySaver(),
    )
    product = load_orders_fixture(config_root, "acme_store").products[0]
    real_apply = cart.apply_confirmed_mutation
    if fails_after_mutation:

        def fail_after_mutation(*args, **kwargs):
            real_apply(*args, **kwargs)
            raise RuntimeError("injected post-mutation failure")

        monkeypatch.setattr(cart, "apply_confirmed_mutation", fail_after_mutation)
    else:

        def fail_before_mutation(*_args, **_kwargs):
            raise RuntimeError("injected pre-mutation failure")

        monkeypatch.setattr(cart, "apply_confirmed_mutation", fail_before_mutation)
    turn_id = f"typed-cart-recovery-{fails_after_mutation}"
    config = {"configurable": {"thread_id": turn_id}}

    first = await graph.ainvoke(
        _admitted_turn(
            f"add one {product.name}",
            turn_id=turn_id,
            active_invocation=ActiveInvocation(
                request=ModifyCart(
                    operation="add",
                    item=CartItemQuery(query=product.name),
                    quantity=1,
                ),
                opened_turn_id=turn_id,
            ),
        ),
        config,
    )

    assert cart.is_empty()
    assert first["__interrupt__"][0].value == (
        f"Just to confirm: add 1 of {product.name} to your cart?"
    )

    result = await graph.ainvoke(Command(resume={"text": "yes"}), config)

    assert cart.line_count == int(fails_after_mutation)
    assert result["active_invocation"] is None
    assert result["pending_cart_mutation"] is None
    line = _only_spoken(result)
    if fails_after_mutation:
        assert product.name in line
        assert "added" in line.lower()
    else:
        assert "Your cart is empty." in line
        assert "Please review your cart before trying again." in line


async def test_typed_cart_recovery_fails_closed_on_a_malformed_receipt(
    config_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = OrderStore("acme_store", load_orders_fixture(config_root, "acme_store").orders)
    cart = CartStore()
    graph = _graph(
        config_root,
        FakeChatModel(),
        store=store,
        cart_store=cart,
        checkpointer=InMemorySaver(),
    )
    product = load_orders_fixture(config_root, "acme_store").products[0]
    turn_id = "typed-cart-malformed-receipt"
    config = {"configurable": {"thread_id": turn_id}}
    await graph.ainvoke(
        _admitted_turn(
            f"add one {product.name}",
            turn_id=turn_id,
            active_invocation=ActiveInvocation(
                request=ModifyCart(
                    operation="add",
                    item=CartItemQuery(query=product.name),
                    quantity=1,
                ),
                opened_turn_id=turn_id,
            ),
        ),
        config,
    )

    def fail_before_mutation(*_args, **_kwargs):
        raise RuntimeError("injected effect failure")

    monkeypatch.setattr(cart, "apply_confirmed_mutation", fail_before_mutation)
    monkeypatch.setattr(
        cart,
        "mutation_receipt",
        lambda *_args, **_kwargs: CommittedReceipt(record="malformed"),
    )

    result = await graph.ainvoke(Command(resume={"text": "yes"}), config)

    assert cart.is_empty()
    assert result["automation_terminal"] is True
    assert result["pending_cart_mutation"] is None
    assert _only_spoken(result) == AUTOMATION_TERMINAL_LINE


def _only_spoken(result: dict[str, object]) -> str:
    """The turn's single caller-facing line. Fails if a turn spoke twice, which is itself a bug."""
    spoken = [message for message in result["messages"] if isinstance(message, AIMessage)]
    assert len(spoken) == 1
    return str(spoken[0].content)


async def test_dispatch_reaches_session_list_owner_without_a_model_call(
    config_root: Path,
) -> None:
    reasoning = FakeChatModel(emit_tool_calls=False)
    graph = _graph(config_root, FakeChatModel(), reasoning_model=reasoning)
    turn_id = "typed-session-list"
    invocation = ActiveInvocation(
        request=ListOrders(scope="session"),
        opened_turn_id=turn_id,
    )

    result = await graph.ainvoke(
        _admitted_turn(
            "what did I order on this call?",
            turn_id=turn_id,
            active_invocation=invocation,
        )
    )

    assert reasoning.invoke_count == 0
    assert result["active_invocation"] is None
    assert result["execution_owner"] is None
    assert "hit a snag" not in _only_spoken(result).lower()


def test_identity_capability_entry_is_preparation_only(config_root: Path) -> None:
    frontline = FakeChatModel()
    reasoning = FakeChatModel()
    graph = _graph(config_root, frontline, reasoning_model=reasoning)

    for request in (VerifyIdentity(), SwitchAccount()):
        state = ReasoningState(
            consumed_turn_ids=("identity-owner",),
            active_invocation=ActiveInvocation(
                request=request,
                opened_turn_id="identity-owner",
            ),
        )
        update = graph.nodes["identity_capability_entry"].invoke(state)

        assert update == {"execution_owner": "identity"}

    assert frontline.invoke_count == 0 and reasoning.invoke_count == 0


async def test_bound_verification_uses_the_typed_owner_without_otp_or_rotation(
    config_root: Path,
) -> None:
    identity = CallerIdentityStore()
    identity.bind(BoundIdentity(customer_ref="CUST-001", masked_contact="number ending 0119"))
    otp = OtpProvider("acme_store", valid_code=_TEST_OTP)
    verification = VerificationStore(otp)
    assert verification.verify_otp(_TEST_OTP)
    cart = CartStore()
    cart.add_item(sku="SKU-1", name="waterproof rain jacket", price_usd=129.0, quantity=1)
    frontline = FakeChatModel()
    reasoning = FakeChatModel()
    graph = _graph(
        config_root,
        frontline,
        reasoning_model=reasoning,
        identity=identity,
        otp=otp,
        verification_store=verification,
        cart_store=cart,
    )

    result = await graph.ainvoke(
        _admitted_turn(
            "verify me",
            turn_id="bound-verify",
            active_invocation=ActiveInvocation(
                request=VerifyIdentity(),
                opened_turn_id="bound-verify",
            ),
        )
    )

    assert frontline.invoke_count == 0 and reasoning.invoke_count == 0
    assert otp.dispatch_count == 0
    assert identity.current() == BoundIdentity(
        customer_ref="CUST-001", masked_contact="number ending 0119"
    )
    assert cart.line_count == 1
    assert result["active_invocation"] is None
    assert result["execution_owner"] is None
    assert _only_spoken(result) == "You're verified on this call."


# --- capability-dispatched answer owners ---------------------------------------------


async def _typed_read(
    graph,
    request: IntentRequest,
    *,
    turn_id: str,
    text: str,
) -> dict[str, object]:
    return await graph.ainvoke(
        _admitted_turn(
            text,
            turn_id=turn_id,
            active_invocation=ActiveInvocation(request=request, opened_turn_id=turn_id),
        )
    )


async def test_catalog_owner_uses_one_live_lookup_and_one_tool_incapable_model_call(
    config_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lookup_queries: list[str] = []
    fixture = load_orders_fixture(config_root, "acme_store")
    catalog = FixtureCatalog("acme_store", fixture)
    real_search = catalog.search

    def observed_search(query: str):
        lookup_queries.append(query)
        return real_search(query)

    monkeypatch.setattr(catalog, "search", observed_search)
    response_model = NativeAsyncOnlyFakeChatModel(
        emit_tool_calls=False,
        text_response="We carry trail running shoes for $89.99.",
        record_prompts=True,
    )
    graph = _graph(config_root, response_model, catalog=catalog)

    result = await _typed_read(
        graph,
        SearchCatalog(query="running"),
        turn_id="catalog-complete",
        text="Tell me about running shoes.",
    )

    assert response_model.invoke_count == 1
    assert lookup_queries == ["running"]
    assert response_model.emitted_messages[-1].tool_calls == []
    assert result["active_invocation"] is None
    assert _only_spoken(result) == "We carry trail running shoes for $89.99."
    prompt = response_model._seen_prompts[-1]
    assert "trail running shoes; SKU SKU-RED-42; price $89.99" in prompt
    assert "waterproof rain jacket" not in prompt


async def test_catalog_no_match_prompt_does_not_authorize_a_relevance_claim(
    config_root: Path,
) -> None:
    response_model = FakeChatModel(
        emit_tool_calls=False,
        text_response="No catalog name matched. The catalog contains trail running shoes.",
        record_prompts=True,
    )
    graph = _graph(config_root, response_model)

    result = await _typed_read(
        graph,
        SearchCatalog(query="walking shoes"),
        turn_id="catalog-no-match",
        text="Do you have walking shoes?",
    )

    assert _only_spoken(result) == (
        "No catalog name matched. The catalog contains trail running shoes."
    )
    prompt = response_model._seen_prompts[-1]
    assert "Do not claim that other products match" in prompt
    assert "Do not invent products" in prompt
    assert "- No matching catalog products." in prompt
    assert "trail running shoes" not in prompt
    assert "waterproof rain jacket" not in prompt


async def test_catalog_answer_telemetry_uses_the_id_matched_committed_turn(
    config_root: Path,
) -> None:
    graph = _graph(
        config_root,
        FakeChatModel(emit_tool_calls=False, text_response="We carry trail running shoes."),
    )

    result = await graph.ainvoke(
        {
            "messages": [
                HumanMessage("Tell me about running shoes.", id="catalog-current"),
                HumanMessage("unadmitted later history", id="catalog-later"),
            ],
            "consumed_turn_ids": ("catalog-current",),
            "active_invocation": ActiveInvocation(
                request=SearchCatalog(query="running"),
                opened_turn_id="catalog-current",
            ),
        }
    )

    assert _only_spoken(result) == "We carry trail running shoes."
    assert _answered_rows(graph) == [
        {
            "utterance": "Tell me about running shoes.",
            "outcome": "answered",
            "outcome_detail": "capability_answer",
            "capability": "search_catalog",
            "answer_source": "grounded_model_response",
        }
    ]


async def test_catalog_owner_fills_only_the_query_from_the_admitted_opening_turn(
    config_root: Path,
) -> None:
    response_model = FakeChatModel(emit_tool_calls=False, text_response="We carry everyday socks.")
    graph = _graph(config_root, response_model)
    opening = await _typed_read(
        graph,
        SearchCatalog(),
        turn_id="catalog-opening",
        text="What do you sell?",
    )

    assert response_model.invoke_count == 1
    assert opening["active_invocation"] is None
    assert _only_spoken(opening) == "We carry everyday socks."
    assert _answered_rows(graph) == [
        {
            "utterance": "What do you sell?",
            "outcome": "answered",
            "outcome_detail": "capability_answer",
            "capability": "search_catalog",
            "answer_source": "grounded_model_response",
        }
    ]


async def test_catalog_owner_rejects_a_blank_followup_without_a_model_call(
    config_root: Path,
) -> None:
    response_model = FakeChatModel(emit_tool_calls=False)
    graph = _graph(config_root, response_model)
    invocation = ActiveInvocation(
        request=SearchCatalog(),
        opened_turn_id="catalog-opening",
    )

    result = await graph.ainvoke(
        _admitted_turn(
            "   ",
            turn_id="catalog-blank",
            consumed_turn_ids=("catalog-opening", "catalog-blank"),
            active_invocation=invocation,
        )
    )

    assert response_model.invoke_count == 0
    assert result["active_invocation"] is None
    assert _only_spoken(result) == "I didn't get a product to look for. What else can I help with?"


@pytest.mark.parametrize("text_response", ("", "   "))
async def test_catalog_owner_recovers_without_answer_telemetry_on_a_blank_model_response(
    config_root: Path,
    text_response: str,
) -> None:
    graph = _graph(
        config_root,
        FakeChatModel(emit_tool_calls=False, text_response=text_response),
    )

    result = await _typed_read(
        graph,
        SearchCatalog(query="running"),
        turn_id="catalog-blank-model",
        text="Tell me about running shoes.",
    )

    assert _failed_nodes(graph) == ["catalog_response"]
    assert "hit a snag" in _only_spoken(result)
    assert _answered_rows(graph) == []


async def test_catalog_owner_rejects_model_text_over_the_platform_limit(
    config_root: Path,
) -> None:
    limit = 40
    graph = _graph(
        config_root,
        FakeChatModel(emit_tool_calls=False, text_response="x" * (limit + 1)),
        caller_audible_model_text_max_chars=limit,
    )

    result = await _typed_read(
        graph,
        SearchCatalog(query="running"),
        turn_id="catalog-over-limit",
        text="Tell me about running shoes.",
    )

    assert _failed_nodes(graph) == ["catalog_response"]
    assert "hit a snag" in _only_spoken(result)
    assert _answered_rows(graph) == []


class _UnexpectedCatalogToolCall(FakeChatModel):
    def _respond(self, messages: list, **kwargs: object) -> AIMessage:
        self._invoke_count += 1
        response = AIMessage(
            content="I changed something.",
            tool_calls=[
                {
                    "name": "invented_effect",
                    "args": {},
                    "id": "unexpected-call",
                    "type": "tool_call",
                }
            ],
        )
        self._emitted_messages.append(response)
        return response


async def test_catalog_owner_rejects_an_unexpected_tool_call_before_speech_or_telemetry(
    config_root: Path,
) -> None:
    graph = _graph(config_root, _UnexpectedCatalogToolCall())

    result = await _typed_read(
        graph,
        SearchCatalog(query="running"),
        turn_id="catalog-tool-call",
        text="Tell me about running shoes.",
    )

    assert _failed_nodes(graph) == ["catalog_response"]
    assert _only_spoken(result) != "I changed something."
    assert _answered_rows(graph) == []


async def test_catalog_owner_fails_closed_when_the_admitted_turn_has_no_matching_message(
    config_root: Path,
) -> None:
    response_model = FakeChatModel(emit_tool_calls=False)
    graph = _graph(config_root, response_model)

    result = await graph.ainvoke(
        {
            "messages": [HumanMessage("an older question", id="older")],
            "consumed_turn_ids": ("older", "missing-current"),
            "active_invocation": ActiveInvocation(
                request=SearchCatalog(query="running"),
                opened_turn_id="older",
            ),
        }
    )

    assert response_model.invoke_count == 0
    assert _failed_nodes(graph) == ["catalog_response"]
    assert "hit a snag" in _only_spoken(result)
    assert _answered_rows(graph) == []


def test_catalog_speech_authority_is_owned_by_the_response_and_code_rejection_nodes(
    config_root: Path,
) -> None:
    graph = _graph(config_root, FakeChatModel())

    assert "catalog_query_reject" in graph.speakable_nodes
    assert "catalog_response" in graph.model_speech_nodes
    assert "catalog_response" not in graph.speakable_nodes
    assert {"catalog_entry", "catalog_response"}.isdisjoint(
        frontline_graph.NON_SPEAKING_MODEL_NODES
    )


@pytest.mark.parametrize(
    ("topic", "answer_source", "answer"),
    (
        ("policy", "grounded_model_response", "Returns are accepted within 30 days."),
        ("general", "general_model_response", "A shoe midsole cushions each step."),
    ),
)
async def test_answer_owner_uses_one_bounded_model_call_and_records_truthful_provenance(
    config_root: Path,
    topic: str,
    answer_source: str,
    answer: str,
) -> None:
    response_model = NativeAsyncOnlyFakeChatModel(
        structured_args={"AnswerResponse": ({"decision": "answer", "answer": answer},)},
        record_prompts=True,
    )
    graph = _graph(config_root, response_model)
    utterance = "What is your return policy?" if topic == "policy" else "What is a shoe midsole?"

    result = await _typed_read(
        graph,
        AnswerQuestion(topic=topic),  # type: ignore[arg-type]
        turn_id=f"answer-{topic}",
        text=utterance,
    )

    assert response_model.invoke_count == 1
    assert result["active_invocation"] is None
    assert _only_spoken(result) == answer
    prompt = response_model._seen_prompts[-1]
    assert utterance in prompt
    assert "does not apply to this terminal owner" in prompt
    assert "Never promise to check, handle, transfer, or follow up" in prompt
    if topic == "policy":
        assert "answer only from the approved merchant policy facts" in prompt
        assert "may give a low-risk explanation" not in prompt
    else:
        assert "may give a low-risk explanation" in prompt
        assert "answer only from the approved merchant policy facts" not in prompt
        assert "unsupported takes precedence" in prompt
    assert _answered_rows(graph) == [
        {
            "utterance": utterance,
            "outcome": "answered",
            "outcome_detail": "capability_answer",
            "capability": "answer_question",
            "answer_source": answer_source,
        }
    ]


async def test_unknown_policy_detail_is_an_answer_not_a_context_clarification(
    config_root: Path,
) -> None:
    line = "That policy detail is not available."
    graph = _graph(
        config_root,
        FakeChatModel(
            structured_args={"AnswerResponse": ({"decision": "answer", "answer": line},)}
        ),
    )

    result = await _typed_read(
        graph,
        AnswerQuestion(topic="policy"),
        turn_id="answer-unknown-policy",
        text="Does the return policy cover monogrammed products?",
    )

    assert _only_spoken(result) == line
    assert _failed_nodes(graph) == []
    assert _answered_rows(graph)[0]["answer_source"] == "grounded_model_response"


@pytest.mark.parametrize(
    ("decision", "line", "node"),
    (
        ("clarify", "What would you like me to explain?", "answer_clarify"),
        (
            "unsupported",
            "Please restate that as a specific store request, including the order, product, "
            "account, or cart detail I should use.",
            "answer_unsupported",
        ),
    ),
)
async def test_answer_no_answer_decisions_use_only_their_code_authored_destination(
    config_root: Path,
    decision: str,
    line: str,
    node: str,
) -> None:
    response_model = FakeChatModel(
        structured_args={"AnswerResponse": ({"decision": decision, "answer": None},)}
    )
    graph = _graph(config_root, response_model)

    result = await _typed_read(
        graph,
        AnswerQuestion(topic="general"),
        turn_id=f"answer-{decision}",
        text="What does that mean?",
    )

    assert response_model.invoke_count == 1
    assert result["active_invocation"] is None
    assert _only_spoken(result) == line
    assert _answered_rows(graph) == []
    assert _failed_nodes(graph) == []
    assert node in graph.speakable_nodes


async def test_answer_model_failure_uses_safe_abort_not_the_unsupported_line(
    config_root: Path,
) -> None:
    graph = _graph(config_root, FakeChatModel(raise_transport=True))

    result = await _typed_read(
        graph,
        AnswerQuestion(topic="general"),
        turn_id="answer-transport-failure",
        text="What is a shoe midsole?",
    )

    spoken = _only_spoken(result)
    assert "hit a snag" in spoken
    assert "specific store request" not in spoken
    assert result["active_invocation"] is None
    assert _failed_nodes(graph) == ["answer_response"]
    assert _answered_rows(graph) == []


@pytest.mark.parametrize(
    "answer",
    (
        "\u200b",
        ". . .",
    ),
)
async def test_answer_owner_rejects_model_text_without_lexical_content(
    config_root: Path,
    answer: str,
) -> None:
    graph = _graph(
        config_root,
        FakeChatModel(
            structured_args={"AnswerResponse": ({"decision": "answer", "answer": answer},)}
        ),
    )

    result = await _typed_read(
        graph,
        AnswerQuestion(topic="general"),
        turn_id="answer-non-lexical",
        text="What is a shoe midsole?",
    )

    assert "hit a snag" in _only_spoken(result)
    assert _failed_nodes(graph) == ["answer_response"]
    assert _answered_rows(graph) == []


async def test_answer_owner_rejects_model_text_over_the_platform_limit(
    config_root: Path,
) -> None:
    limit = 40
    graph = _graph(
        config_root,
        FakeChatModel(
            structured_args={
                "AnswerResponse": ({"decision": "answer", "answer": "x" * (limit + 1)},)
            }
        ),
        caller_audible_model_text_max_chars=limit,
    )

    result = await _typed_read(
        graph,
        AnswerQuestion(topic="general"),
        turn_id="answer-over-limit",
        text="What is a shoe midsole?",
    )

    assert "hit a snag" in _only_spoken(result)
    assert _failed_nodes(graph) == ["answer_response"]
    assert _answered_rows(graph) == []


class _RawAnswerMappingModel(FakeChatModel):
    def with_structured_output(self, schema, *, include_raw=False, **kwargs):
        return RunnableLambda(lambda _messages: {"decision": "answer", "answer": "unvalidated"})


async def test_answer_owner_rejects_a_raw_mapping_from_the_structured_wrapper(
    config_root: Path,
) -> None:
    graph = _graph(config_root, _RawAnswerMappingModel())

    result = await _typed_read(
        graph,
        AnswerQuestion(topic="general"),
        turn_id="answer-raw-mapping",
        text="What is a shoe midsole?",
    )

    assert "hit a snag" in _only_spoken(result)
    assert _failed_nodes(graph) == ["answer_response"]
    assert _answered_rows(graph) == []


def test_answer_speech_authority_is_split_between_model_and_code_nodes(
    config_root: Path,
) -> None:
    graph = _graph(config_root, FakeChatModel())

    assert {"answer_clarify", "answer_unsupported"} <= graph.speakable_nodes
    assert "answer_response" in graph.model_speech_nodes
    assert "answer_response" not in graph.speakable_nodes
    assert "answer_response" not in frontline_graph.NON_SPEAKING_MODEL_NODES


def test_answer_owner_uses_the_required_configured_structured_transport(config_root: Path) -> None:
    response_model = FakeChatModel()
    configured_method: StructuredOutputMethod = "json_schema"

    _graph(config_root, response_model, structured_output_method=configured_method)

    assert response_model.structured_methods == (configured_method, configured_method)


async def test_order_status_owner_grants_and_renders_one_explicit_order_without_model_speech(
    config_root: Path,
) -> None:
    model = FakeChatModel()
    graph = _graph(config_root, model)

    result = await _typed_read(
        graph,
        VerifyOrderStatus(target=ExplicitOrderSet(order_refs=("ORD-1001",))),
        turn_id="status-explicit",
        text="ORD-1001, my phone is 555 010 0119",
    )

    assert model.invoke_count == 0
    assert result["active_invocation"] is None
    assert "Your order ORD-1001" in _only_spoken(result)
    assert _answered_rows(graph) == [
        {
            "utterance": "ORD-1001, my phone is [phone]",
            "outcome": "answered",
            "outcome_detail": "capability_answer",
            "capability": "verify_order_status",
            "answer_source": "code_authored_read",
        }
    ]


async def test_order_status_spoken_email_after_contact_phrase_matches_end_to_end(
    config_root: Path,
) -> None:
    identity = CallerIdentityStore()
    graph = _graph(config_root, FakeChatModel(), identity=identity)

    result = await _typed_read(
        graph,
        VerifyOrderStatus(target=ExplicitOrderSet(order_refs=("ORD-1002",))),
        turn_id="status-spoken-email",
        text="ORD-1002, contact me at casey at example dot com",
    )

    assert identity.order_granted("ORD-1002")
    assert "Your order ORD-1002" in _only_spoken(result)


async def _pause_unwitnessed_order_status_target(config_root: Path, *, thread_id: str):
    identity = CallerIdentityStore()
    graph = _graph(
        config_root,
        FakeChatModel(),
        identity=identity,
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": thread_id}}
    paused = await graph.ainvoke(
        _admitted_turn(
            "I do not know the order number. My email is casey@example.com.",
            turn_id=thread_id,
            active_invocation=ActiveInvocation(
                request=VerifyOrderStatus(
                    target=ExplicitOrderSet(order_refs=("ORD-1002",)),
                ),
                opened_turn_id=thread_id,
            ),
        ),
        config,
    )
    return graph, identity, config, paused


async def test_unwitnessed_router_target_requires_caller_confirmation_before_guest_grant(
    config_root: Path,
) -> None:
    graph, identity, config, paused = await _pause_unwitnessed_order_status_target(
        config_root,
        thread_id="status-router-confirm",
    )

    assert "I heard ORD-1002" in str(paused["__interrupt__"][0].value)
    assert paused["active_invocation"].request.explicit_target_turn_id == "status-router-confirm"
    assert not identity.order_granted("ORD-1002")
    assert _answered_rows(graph) == []

    result = await graph.ainvoke(
        Command(
            resume={"text": "yes"},
            update={"consumed_turn_ids": ("status-router-confirm-answer",)},
        ),
        config,
    )

    assert identity.order_granted("ORD-1002")
    assert result["active_invocation"] is None
    assert "Your order ORD-1002" in _only_spoken(result)


async def test_model_only_order_status_target_cannot_grant_before_confirmation(
    config_root: Path,
) -> None:
    identity = CallerIdentityStore()
    model = FakeChatModel(
        structured_args={
            "OrderTargetProposal": ({"relationship": "single", "order_refs": ["ORD-1002"]},)
        }
    )
    graph = _graph(
        config_root,
        model,
        identity=identity,
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "status-model-confirm"}}
    invocation = ActiveInvocation(
        request=VerifyOrderStatus(),
        opened_turn_id="status-opening",
    )

    paused = await graph.ainvoke(
        _admitted_turn(
            "I do not know the order number. My email is casey@example.com.",
            turn_id="status-followup",
            consumed_turn_ids=("status-opening", "status-followup"),
            active_invocation=invocation,
        ),
        config,
    )

    assert model.invoke_count == 1
    assert "I heard ORD-1002" in str(paused["__interrupt__"][0].value)
    assert paused["active_invocation"].request.explicit_target_turn_id == "status-followup"
    assert not identity.order_granted("ORD-1002")
    assert _answered_rows(graph) == []


async def test_confirmed_order_status_target_can_collect_contact_on_a_later_turn(
    config_root: Path,
) -> None:
    identity = CallerIdentityStore()
    graph = _graph(
        config_root,
        FakeChatModel(),
        identity=identity,
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "status-confirm-then-contact"}}
    paused = await graph.ainvoke(
        _admitted_turn(
            "I do not know the order number.",
            turn_id="status-target-source",
            active_invocation=ActiveInvocation(
                request=VerifyOrderStatus(
                    target=ExplicitOrderSet(order_refs=("ORD-1002",)),
                ),
                opened_turn_id="status-target-source",
            ),
        ),
        config,
    )
    assert "I heard ORD-1002" in str(paused["__interrupt__"][0].value)

    contact_question = await graph.ainvoke(
        Command(
            resume={"text": "yes"},
            update={"consumed_turn_ids": ("status-target-confirmation",)},
        ),
        config,
    )

    assert _only_spoken(contact_question) == "What email address or phone number is on the account?"
    assert contact_question["active_invocation"].request.explicit_target_confirmed
    assert not identity.order_granted("ORD-1002")
    prior_message_count = len(contact_question["messages"])

    result = await graph.ainvoke(
        _admitted_turn(
            "casey@example.com",
            turn_id="status-contact-followup",
        ),
        config,
    )

    assert identity.order_granted("ORD-1002")
    new_spoken = [
        message
        for message in result["messages"][prior_message_count:]
        if isinstance(message, AIMessage)
    ]
    assert len(new_spoken) == 1
    assert "Your order ORD-1002" in str(new_spoken[0].content)


async def test_declined_order_status_target_confirmation_grants_nothing(
    config_root: Path,
) -> None:
    graph, identity, config, _paused = await _pause_unwitnessed_order_status_target(
        config_root,
        thread_id="status-confirm-no",
    )

    result = await graph.ainvoke(
        Command(
            resume={"text": "no"},
            update={"consumed_turn_ids": ("status-confirm-no-answer",)},
        ),
        config,
    )

    assert not identity.order_granted("ORD-1002")
    assert result["active_invocation"] is None
    assert _only_spoken(result) == "What is the order number, for example ORD-1234?"


async def test_unclear_order_status_target_confirmation_is_bounded(
    config_root: Path,
) -> None:
    graph, identity, config, _paused = await _pause_unwitnessed_order_status_target(
        config_root,
        thread_id="status-confirm-unclear",
    )

    retry = await graph.ainvoke(
        Command(
            resume={"text": "maybe"},
            update={"consumed_turn_ids": ("status-confirm-unclear-1",)},
        ),
        config,
    )

    assert "Please say yes or no" in str(retry["__interrupt__"][0].value)
    assert not identity.order_granted("ORD-1002")

    result = await graph.ainvoke(
        Command(
            resume={"text": "I am not sure"},
            update={"consumed_turn_ids": ("status-confirm-unclear-2",)},
        ),
        config,
    )

    assert not identity.order_granted("ORD-1002")
    assert result["active_invocation"] is None
    assert _only_spoken(result) == "What is the order number, for example ORD-1234?"


async def test_interrupted_order_status_target_readback_reconfirms_before_grant(
    config_root: Path,
) -> None:
    graph, identity, config, _paused = await _pause_unwitnessed_order_status_target(
        config_root,
        thread_id="status-confirm-interrupted",
    )

    retry = await graph.ainvoke(
        Command(
            resume={"text": "yes", "readback_interrupted": True},
            update={"consumed_turn_ids": ("status-confirm-interrupted-1",)},
        ),
        config,
    )

    assert "Please say yes or no" in str(retry["__interrupt__"][0].value)
    assert not identity.order_granted("ORD-1002")

    result = await graph.ainvoke(
        Command(
            resume={"text": "yes"},
            update={"consumed_turn_ids": ("status-confirm-interrupted-2",)},
        ),
        config,
    )

    assert identity.order_granted("ORD-1002")
    assert "Your order ORD-1002" in _only_spoken(result)


async def test_order_status_target_confirmation_human_escape_uses_terminal_handover(
    config_root: Path,
) -> None:
    graph, identity, config, _paused = await _pause_unwitnessed_order_status_target(
        config_root,
        thread_id="status-confirm-human",
    )

    result = await graph.ainvoke(
        Command(
            resume={
                "text": "I want a person",
                "handoff_source": HandoffSource.SEMANTIC_ROUTER.value,
            },
            update={"consumed_turn_ids": ("status-confirm-human-answer",)},
        ),
        config,
    )

    assert not identity.order_granted("ORD-1002")
    assert result["active_invocation"] is None
    assert result["automation_terminal"] is True


async def test_order_status_target_confirmation_failure_safe_aborts_before_grant(
    config_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph, identity, config, _paused = await _pause_unwitnessed_order_status_target(
        config_root,
        thread_id="status-confirm-failure",
    )

    def _fail_consent(_answer: object, **_kwargs: object) -> NoReturn:
        raise RuntimeError("injected confirmation failure")

    monkeypatch.setattr(read_flow, "classify_confirmation", _fail_consent)
    result = await graph.ainvoke(
        Command(
            resume={"text": "yes"},
            update={"consumed_turn_ids": ("status-confirm-failure-answer",)},
        ),
        config,
    )

    assert not identity.order_granted("ORD-1002")
    assert result["active_invocation"] is None
    assert "hit a snag" in _only_spoken(result)
    assert _failed_nodes(graph) == ["order_status_target_confirm"]


async def test_order_status_owner_gathers_target_then_uses_one_non_speaking_proposal(
    config_root: Path,
) -> None:
    model = FakeChatModel(
        structured_args={
            "OrderTargetProposal": (
                {"relationship": "ambiguous", "order_refs": []},
                {"relationship": "single", "order_refs": ["ORD-1002"]},
            )
        }
    )
    graph = _graph(config_root, model)
    opening = await _typed_read(
        graph,
        VerifyOrderStatus(),
        turn_id="status-opening",
        text="Where is my order?",
    )
    assert _only_spoken(opening) == "What is the order number, for example ORD-1234?"
    retained = opening["active_invocation"]
    assert retained is not None

    result = await graph.ainvoke(
        _admitted_turn(
            "ORD-1002, casey@example.com",
            turn_id="status-followup",
            consumed_turn_ids=("status-opening", "status-followup"),
            active_invocation=retained,
        )
    )

    assert model.invoke_count == 2
    assert result["active_invocation"] is None
    assert "Your order ORD-1002" in _only_spoken(result)
    assert "order_status_target_propose" in frontline_graph.NON_SPEAKING_MODEL_NODES
    assert "order_status_target_propose" not in graph.speakable_nodes


async def test_order_status_focused_selector_fails_closed_without_live_focus(
    config_root: Path,
) -> None:
    graph = _graph(config_root, FakeChatModel())

    result = await _typed_read(
        graph,
        VerifyOrderStatus(target=FocusedOrderSet()),
        turn_id="status-no-focus",
        text="Where is that order?",
    )

    assert result["active_invocation"] is None
    assert _only_spoken(result) == "What is the order number, for example ORD-1234?"


async def test_order_status_plural_grant_is_atomic_and_does_not_invent_focus(
    config_root: Path,
) -> None:
    recent = RecentOrderContext(max_refs=3)
    identity = CallerIdentityStore()
    graph = _graph(config_root, FakeChatModel(), recent_orders=recent, identity=identity)

    result = await _typed_read(
        graph,
        VerifyOrderStatus(target=ExplicitOrderSet(order_refs=("ORD-1001", "ORD-1003"))),
        turn_id="status-plural",
        text="ORD-1001 and ORD-1003, phone 555 010 0119",
    )

    line = _only_spoken(result)
    assert "Your order ORD-1001" in line and "Your order ORD-1003" in line
    assert identity.order_granted("ORD-1001") and identity.order_granted("ORD-1003")
    assert recent.snapshot().order_refs == ("ORD-1001", "ORD-1003")
    assert recent.snapshot().focused_order_ref is None
    assert len(_answered_rows(graph)) == 1


async def test_order_status_bound_principal_cannot_widen_with_a_foreign_contact(
    config_root: Path,
) -> None:
    identity = CallerIdentityStore()
    identity.bind(BoundIdentity(customer_ref="CUST-001", masked_contact="number ending 0119"))
    graph = _graph(config_root, FakeChatModel(), identity=identity)

    result = await _typed_read(
        graph,
        VerifyOrderStatus(target=ExplicitOrderSet(order_refs=("ORD-1002",))),
        turn_id="status-bound-foreign",
        text="ORD-1002, casey@example.com",
    )

    assert "couldn't retrieve" in _only_spoken(result)
    assert not identity.order_granted("ORD-1002")
    assert identity.current() is not None


async def test_order_status_recent_selector_uses_the_complete_bounded_set(
    config_root: Path,
) -> None:
    recent = RecentOrderContext(max_refs=3)
    recent.record(("ORD-1001", "ORD-1003"), operation="list")
    graph = _graph(config_root, FakeChatModel(), recent_orders=recent)

    result = await _typed_read(
        graph,
        VerifyOrderStatus(target=RecentOrderSet()),
        turn_id="status-recent",
        text="check those orders, phone 555 010 0119",
    )

    line = _only_spoken(result)
    assert "Your order ORD-1001" in line and "Your order ORD-1003" in line


async def test_order_status_alternatives_retain_the_owner_and_ask_without_read_or_grant(
    config_root: Path,
) -> None:
    identity = CallerIdentityStore()
    model = FakeChatModel(
        structured_args={
            "OrderTargetProposal": (
                {
                    "relationship": "alternative",
                    "order_refs": ["ORD-1001", "ORD-1002"],
                },
            )
        }
    )
    graph = _graph(config_root, model, identity=identity)
    invocation = ActiveInvocation(
        request=VerifyOrderStatus(),
        opened_turn_id="status-opening",
    )

    result = await graph.ainvoke(
        _admitted_turn(
            "Either ORD-1001 or ORD-1002",
            turn_id="status-alternative",
            consumed_turn_ids=("status-opening", "status-alternative"),
            active_invocation=invocation,
        )
    )

    assert result["active_invocation"] == invocation
    assert _only_spoken(result) == "What is the order number, for example ORD-1234?"
    assert not identity.order_granted("ORD-1001")
    assert not identity.order_granted("ORD-1002")


async def test_order_status_multiple_contact_claims_clarify_without_matching(
    config_root: Path,
) -> None:
    identity = CallerIdentityStore()
    graph = _graph(config_root, FakeChatModel(), identity=identity)

    result = await _typed_read(
        graph,
        VerifyOrderStatus(target=ExplicitOrderSet(order_refs=("ORD-1001",))),
        turn_id="status-multiple-contact",
        text="ORD-1001, casey@example.com or phone 555 010 0119",
    )

    assert _only_spoken(result) == "What email address or phone number is on the account?"
    assert result["active_invocation"] is not None
    assert not identity.order_granted("ORD-1001")


async def test_cart_view_owner_speaks_the_live_cart_without_a_model_call(
    config_root: Path,
) -> None:
    cart = CartStore()
    cart.add_item(sku="SKU-1", name="waterproof rain jacket", price_usd=129.0, quantity=1)
    frontline = FakeChatModel()
    reasoning = FakeChatModel()
    graph = _graph(config_root, frontline, cart_store=cart, reasoning_model=reasoning)

    result = await _typed_read(graph, ViewCart(), turn_id="typed-cart", text="what's in my cart?")

    assert frontline.invoke_count == 0 and reasoning.invoke_count == 0
    assert result["active_invocation"] is None
    line = _only_spoken(result)
    assert "waterproof rain jacket" in line and "129.00" in line
    assert cart.line_count == 1  # a read mutates nothing


async def test_cart_view_owner_re_reads_the_store_on_every_turn(config_root: Path) -> None:
    # Live-read freshness: the owner must render the CURRENT cart, never a value captured when
    # the invocation was opened.
    cart = CartStore()
    cart.add_item(sku="SKU-1", name="waterproof rain jacket", price_usd=129.0, quantity=1)
    graph = _graph(config_root, FakeChatModel(), cart_store=cart)

    first = _only_spoken(await _typed_read(graph, ViewCart(), turn_id="cart-1", text="my cart?"))
    cart.add_item(sku="SKU-2", name="trail running shoes", price_usd=95.0, quantity=1)
    second = _only_spoken(await _typed_read(graph, ViewCart(), turn_id="cart-2", text="and now?"))

    assert "trail running shoes" not in first
    assert "trail running shoes" in second and "waterproof rain jacket" in second


async def test_cart_view_owner_speaks_the_empty_line_with_no_close(config_root: Path) -> None:
    from agnostic_market.agents._copy import all_closes

    graph = _graph(config_root, FakeChatModel(), cart_store=CartStore())

    line = _only_spoken(await _typed_read(graph, ViewCart(), turn_id="cart-empty", text="my cart?"))

    assert line == "Your cart's empty at the moment."
    assert not any(line.endswith(close) for close in all_closes())


async def test_typed_cart_read_uses_the_shared_live_renderer(config_root: Path) -> None:
    cart = CartStore()
    cart.add_item(sku="SKU-1", name="waterproof rain jacket", price_usd=129.0, quantity=1)
    graph = _graph(config_root, FakeChatModel(), cart_store=cart)

    typed_line = _only_spoken(
        await _typed_read(graph, ViewCart(), turn_id="same-typed", text="my cart?")
    )

    assert typed_line.startswith(render_cart_line(cart.view(), cart.cart_total()))


async def test_typed_cart_read_takes_exactly_one_close(
    config_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []

    def counting_close() -> str:
        calls.append(1)
        return "Anything else I can help with?"

    monkeypatch.setattr(frontline_graph, "warm_close", counting_close)
    cart = CartStore()
    cart.add_item(sku="SKU-1", name="waterproof rain jacket", price_usd=129.0, quantity=1)

    await _typed_read(
        _graph(config_root, FakeChatModel(), cart_store=cart),
        ViewCart(),
        turn_id="close-typed",
        text="my cart?",
    )
    assert len(calls) == 1


async def test_identity_status_owner_reports_the_live_binding_only(config_root: Path) -> None:
    from agnostic_market.agents._copy import (
        IDENTITY_STATUS_UNVERIFIED,
        IDENTITY_STATUS_VERIFIED,
        all_closes,
    )

    identity = CallerIdentityStore()
    frontline = FakeChatModel()
    reasoning = FakeChatModel()
    graph = _graph(config_root, frontline, identity=identity, reasoning_model=reasoning)

    unbound_result = await _typed_read(
        graph, ViewIdentityStatus(), turn_id="id-1", text="am I verified?"
    )
    unbound = _only_spoken(unbound_result)
    identity.bind(BoundIdentity(customer_ref="CUST-001", masked_contact="number ending 0119"))
    bound_result = await _typed_read(
        graph, ViewIdentityStatus(), turn_id="id-2", text="am I verified now?"
    )
    bound = _only_spoken(bound_result)

    assert frontline.invoke_count == 0 and reasoning.invoke_count == 0
    assert unbound_result["active_invocation"] is None
    assert bound_result["active_invocation"] is None
    # The unverified branch carries its own invitation, so it takes no warm close.
    assert unbound == IDENTITY_STATUS_UNVERIFIED
    # A resolved answer closes like every other code-authored line.
    assert bound.startswith(IDENTITY_STATUS_VERIFIED)
    assert any(bound.endswith(close) for close in all_closes())
    # Bound-or-unbound ONLY: never the customer reference, never the contact on file.
    assert "CUST-001" not in bound and "0119" not in bound


def _telemetry_records(graph):
    records = []
    for sink in graph._test_telemetry_sinks:
        assert isinstance(sink, InMemoryTelemetrySink)
        records.extend(sink.records)
    return records


def _rows(graph) -> list[dict[str, object]]:
    return [{"event": record.event, **record.attributes} for record in _telemetry_records(graph)]


def _answered_rows(graph) -> list[dict[str, object]]:
    return [
        {key: value for key, value in row.items() if key != "event"}
        for row in _rows(graph)
        if row.get("outcome") == "answered"
    ]


def _failed_nodes(graph) -> list[str]:
    return [str(row["node"]) for row in _rows(graph) if row.get("event") == "turn_failed"]


async def test_every_capability_answer_owner_records_one_answered_turn(config_root: Path) -> None:
    # These nodes END, bypassing finalize_node, so without their own record a typed read would
    # leave no negative in the classifier dataset. All FOUR owners, because the parity is the
    # point: one missing call is invisible in any test that only asserts a row's absence.
    cart = CartStore()
    cart.add_item(sku="SKU-1", name="waterproof rain jacket", price_usd=129.0, quantity=1)
    graph = _graph(config_root, FakeChatModel(), cart_store=cart)

    await _typed_read(graph, ViewCart(), turn_id="tel-cart", text="what's in my cart?")
    await _typed_read(graph, ViewIdentityStatus(), turn_id="tel-id", text="am I verified?")
    await _typed_read(
        graph,
        ListOrders(scope="session"),
        turn_id="tel-list",
        text="what have I ordered?",
    )
    await _typed_read(
        graph,
        SearchCatalog(query="running"),
        turn_id="tel-catalog",
        text="what running shoes do you have?",
    )

    assert _answered_rows(graph) == [
        {
            "utterance": "what's in my cart?",
            "outcome": "answered",
            # NOT "code_render": that slug is the tool-driven path's and carries a `tool` key.
            "outcome_detail": "capability_answer",
            "capability": "view_cart",
            "answer_source": "code_authored_read",
        },
        {
            "utterance": "am I verified?",
            "outcome": "answered",
            "outcome_detail": "capability_answer",
            "capability": "view_identity_status",
            "answer_source": "code_authored_read",
        },
        {
            "utterance": "what have I ordered?",
            "outcome": "answered",
            "outcome_detail": "capability_answer",
            "capability": "list_orders",
            "answer_source": "code_authored_read",
        },
        {
            "utterance": "what running shoes do you have?",
            "outcome": "answered",
            "outcome_detail": "capability_answer",
            "capability": "search_catalog",
            "answer_source": "grounded_model_response",
        },
    ]
    # No tool ran on any path, so claiming one would corrupt any tool-usage analysis.
    assert all("tool" not in row for row in _answered_rows(graph))
    assert all(
        record.purpose is TelemetryPurpose.ROUTING_EVIDENCE
        for record in _telemetry_records(graph)
        if record.event == "capability_answered"
    )


async def test_rotated_read_continuation_records_no_blank_utterance(config_root: Path) -> None:
    # An ACCOUNT list is the only read `project_principal_transition` lets survive rotation
    # (view_cart, view_identity_status and even a SESSION list are refused), and the engine seeds
    # that fresh thread with no messages. No in-thread utterance means no label, and an
    # empty-string row is a mislabelled classifier negative, so none is written.
    identity = CallerIdentityStore()
    identity.bind(BoundIdentity(customer_ref="CUST-001", masked_contact="number ending 0119"))
    graph = _graph(config_root, FakeChatModel(), identity=identity)

    result = await graph.ainvoke(
        {
            "messages": [],
            "consumed_turn_ids": ("rotated",),
            "active_invocation": ActiveInvocation(
                request=ListOrders(scope="account"), opened_turn_id="rotated"
            ),
        }
    )

    # Non-vacuous: the owner really did run and answer, it just recorded nothing.
    assert "ORD-1001" in _only_spoken(result)
    assert _answered_rows(graph) == []


def _failing_render(*_args: object, **_kwargs: object) -> NoReturn:
    raise RuntimeError("render failed")


async def test_a_failed_cart_render_records_no_answered_turn(
    config_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The record must FOLLOW the line it reports, as the tool path's does. Written first it
    # would claim "answered" for a turn whose render blew up and whose caller heard the snag.
    monkeypatch.setattr(frontline_graph, "render_cart_line", _failing_render)
    cart = CartStore()
    cart.add_item(sku="SKU-1", name="waterproof rain jacket", price_usd=129.0, quantity=1)
    graph = _graph(config_root, FakeChatModel(), cart_store=cart)

    result = await _typed_read(graph, ViewCart(), turn_id="cart-fail", text="what's in my cart?")

    # Pinned to the owner: "hit a snag" alone would also pass if some EARLIER node had broken.
    assert _failed_nodes(graph) == ["cart_view_render"]
    assert "hit a snag" in _only_spoken(result)
    assert _answered_rows(graph) == []


async def test_a_failed_order_list_render_records_no_answered_turn(
    config_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(support_flow, "render_order_list_line", _failing_render)
    identity = CallerIdentityStore()
    identity.bind(BoundIdentity(customer_ref="CUST-001", masked_contact="number ending 0119"))
    graph = _graph(config_root, FakeChatModel(), identity=identity)

    result = await _typed_read(
        graph, ListOrders(scope="account"), turn_id="list-fail", text="what are my orders?"
    )

    assert _failed_nodes(graph) == ["support_capability_render"]
    assert "hit a snag" in _only_spoken(result)
    assert _answered_rows(graph) == []


def test_all_regular_nodes_have_the_reviewed_recovery_policy(config_root: Path) -> None:
    graph = _graph(config_root, FakeChatModel())
    policies = graph.node_recovery_policies
    assert {
        "ordinary_abort",
        "cart_abort",
        "support_abort",
        "identity_abort",
    }.isdisjoint(graph.get_graph().nodes)
    expected_abandonment = {
        AbandonmentKind.PURE_ABORT: {
            "entry",
            "request_person",
            "abort_current",
            "owner_declined",
            "router_no_action",
            "cart_clarify",
            "cart_guardrail",
            "capability_dispatch",
            "support_capability_entry",
            "identity_capability_entry",
            "support_capability_render",
            "cart_view_render",
            "identity_status_render",
            "catalog_entry",
            "catalog_query_reject",
            "catalog_response",
            "answer_response",
            "answer_clarify",
            "answer_unsupported",
            "order_status_entry",
            "order_status_target_ask",
            "order_status_target_propose",
            "order_status_target_reject",
            "order_status_fulfill",
            "support_clarify",
            "support_guardrail",
            "support_risk_check",
            "support_cancel_guardrail",
            "support_resolve",
            "support_return_guardrail",
            "support_profile_guardrail",
            "support_profile_risk_check",
            "identity_assemble",
            "identity_ask_contact",
            "identity_reask",
            "identity_guardrail",
            "identity_risk_check",
        },
        AbandonmentKind.CART_REVIEW: {
            "cart_capability_entry",
            "cart_ack",
        },
        AbandonmentKind.AUTHORITATIVE_RECONCILE: {
            "cart_mutation_apply",
            "cart_place",
            "support_place",
            "support_cancel_void",
            "support_return_place",
            "support_profile_place",
        },
        AbandonmentKind.LIFECYCLE_SPECIAL: {
            "principal_warning",
            "order_status_target_confirm",
            "cart_mutation_confirm",
            "cart_confirm",
            "support_dispatch",
            "support_collect",
            "support_confirm",
            "support_cancel_confirm",
            "support_return_confirm",
            "support_profile_dispatch",
            "support_profile_collect",
            "support_profile_confirm",
            "identity_dispatch",
            "identity_collect",
            "identity_apply",
        },
        AbandonmentKind.TERMINAL: {
            "handover",
            "automation_terminal_response",
        },
    }
    expected_exception = {
        ExceptionAction.SAFE_ABORT: {
            *expected_abandonment[AbandonmentKind.PURE_ABORT],
            "order_status_target_confirm",
        },
        ExceptionAction.CART_REVIEW: {
            "cart_capability_entry",
            "cart_ack",
            "cart_mutation_confirm",
            "cart_mutation_apply",
        },
        ExceptionAction.RECONCILE_PLACEMENT: {"cart_place"},
        ExceptionAction.RECONCILE_REFUND: {"support_place"},
        ExceptionAction.RECONCILE_CANCEL: {"support_cancel_void"},
        ExceptionAction.RECONCILE_RETURN: {"support_return_place"},
        ExceptionAction.RECONCILE_PROFILE_CHANGE: {"support_profile_place"},
        ExceptionAction.ABORT_PRINCIPAL_WARNING: {"principal_warning"},
        ExceptionAction.ABORT_PLACEMENT_CONFIRMATION: {"cart_confirm"},
        ExceptionAction.ABORT_REFUND_VERIFICATION: {"support_dispatch", "support_collect"},
        ExceptionAction.ABORT_REFUND_CONFIRMATION: {"support_confirm"},
        ExceptionAction.ABORT_CANCEL_CONFIRMATION: {"support_cancel_confirm"},
        ExceptionAction.ABORT_RETURN_CONFIRMATION: {"support_return_confirm"},
        ExceptionAction.ABORT_PROFILE_VERIFICATION: {
            "support_profile_dispatch",
            "support_profile_collect",
        },
        ExceptionAction.ABORT_PROFILE_CONFIRMATION: {"support_profile_confirm"},
        ExceptionAction.ABORT_IDENTITY_VERIFICATION: {
            "identity_dispatch",
            "identity_collect",
        },
        ExceptionAction.RECONCILE_PRINCIPAL_TRANSITION: {"identity_apply"},
        ExceptionAction.TERMINAL: expected_abandonment[AbandonmentKind.TERMINAL]
        - {"automation_terminal_response"},
        ExceptionAction.ENGINE_LAST_RESORT: {"automation_terminal_response"},
    }

    assert isinstance(policies, MappingProxyType)
    assert RECOVERY_NODE_NAME in graph.get_graph().nodes
    assert graph.builder.nodes[RECOVERY_NODE_NAME].ends == (graph.recovery_entry_node, END)
    assert not any(source == RECOVERY_NODE_NAME for source, _target in graph.builder.edges)
    assert RECOVERY_NODE_NAME not in graph.builder.branches
    assert graph.recovery_infrastructure_nodes == frozenset(
        {
            RECOVERY_NODE_NAME,
            RECOVERY_TERMINALIZER_NODE_NAME,
            graph.principal_seed_complete_node,
        }
    )
    assert set(policies) == set().union(*expected_exception.values())
    assert graph.recovery_handled_nodes == frozenset(
        set(policies) - {"automation_terminal_response"}
    )
    assert graph.recovery_handled_infrastructure_nodes == frozenset({RECOVERY_NODE_NAME})
    assert graph.consent_interrupt_kinds == {
        "principal_warning": "standard",
        "order_status_target_confirm": "standard",
        "cart_mutation_confirm": "standard",
        "cart_confirm": "standard",
        "support_confirm": "standard",
        "support_cancel_confirm": "cancel",
        "support_return_confirm": "standard",
        "support_profile_confirm": "standard",
    }
    assert graph.node_execution_tracker.tracked_node_names == frozenset(
        set(policies) - expected_abandonment[AbandonmentKind.PURE_ABORT]
    )
    assert Counter(policy.on_abandonment for policy in policies.values()) == {
        kind: len(nodes) for kind, nodes in expected_abandonment.items()
    }
    for kind, names in expected_abandonment.items():
        assert {name for name, policy in policies.items() if policy.on_abandonment == kind} == names
    for action, names in expected_exception.items():
        assert {name for name, policy in policies.items() if policy.on_exception == action} == names
    for name, policy in policies.items():
        expected_cancellation = (
            ExceptionAction.TERMINAL
            if policy.on_abandonment == AbandonmentKind.TERMINAL
            else policy.on_exception
        )
        assert policy.on_cancellation == expected_cancellation, name
