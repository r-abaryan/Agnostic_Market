"""Frontline graph: structural safety + routing (gate / read-only / model-handover). Zero net."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from types import MappingProxyType
from typing import NoReturn

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END
from llm_fakes import FakeChatModel
from policy_helpers import make_policy

from agnostic_market.agents.cart import flow as cart_flow
from agnostic_market.agents.frontline import build_frontline_graph, read_flow
from agnostic_market.agents.frontline import graph as frontline_graph
from agnostic_market.agents.recovery import (
    RECOVERY_NODE_NAME,
    RECOVERY_TERMINALIZER_NODE_NAME,
    clear_automation_state,
)
from agnostic_market.agents.support import flow as support_flow
from agnostic_market.agents.tooling import wrap_readonly_tool
from agnostic_market.commerce.cart import CartStore
from agnostic_market.commerce.identity import (
    BoundIdentity,
    CallerIdentityStore,
    CustomerDirectory,
    load_customers_fixture,
)
from agnostic_market.commerce.orders import (
    CatalogLookup,
    OrdersFixture,
    OrderStore,
    RecentOrderContext,
    load_orders_fixture,
)
from agnostic_market.commerce.payment_instruments import (
    PaymentInstrumentDirectory,
    load_payment_instruments_fixture,
)
from agnostic_market.commerce.profile import ProfileStore, load_profile_fixture
from agnostic_market.commerce.verification import OtpProvider, VerificationStore
from agnostic_market.dtos.orchestration import (
    ActiveInvocation,
    CancelOrders,
    CapabilityId,
    CartItemChoices,
    CartItemQuery,
    IntentRequest,
    ListOrders,
    ModifyCart,
    PlaceOrder,
    ResolvedCartItemRef,
    SearchCatalog,
    SwitchAccount,
    VerifyIdentity,
    ViewCart,
    ViewIdentityStatus,
)
from agnostic_market.dtos.recovery import AbandonmentKind, ExceptionAction
from agnostic_market.dtos.state import (
    ClarificationProgress,
    HandoffDestination,
    HandoffReasonCode,
    HandoffRequest,
    PolicyContext,
    ReasoningState,
)
from agnostic_market.voice.context import CallerContext
from agnostic_market.voice.tools import build_voice_tools

# A DEFERRING destination (planner) — these tests exercise the destination-agnostic handover
# CONTROL mechanism (routing through Command, deferral speak, history hygiene), NOT a specific
# flow. checkout/support destinations ENTER their flows (3b/3c) instead of deferring, so a
# mechanism test must use a destination that still ends at the spoken deferral.
_HANDOVER_ARGS = {"request_handover": {"destination": "planner", "reason_code": "multi_step"}}
_READ_ARGS = {"order_status": {"order_id": "ORD-1001"}, "catalog_search": {"query": "shoes"}}
_TEST_OTP = "482913"


def _granted(*order_ids: str) -> CallerIdentityStore:
    """A session identity store with rung-1 grants — for tests exercising what happens
    AFTER an authorized order read (the L3 render path), not the gate itself."""
    identity = CallerIdentityStore()
    for oid in order_ids:
        identity.grant_order(oid)
    return identity


def _tools(
    config_root: Path,
    store: OrderStore,
    cart: CartStore,
    recent_orders: RecentOrderContext,
    identity: CallerIdentityStore,
) -> list:
    customers = CustomerDirectory(load_customers_fixture(config_root, "acme_store"))
    return [
        wrap_readonly_tool(t, "acme_store")
        for t in build_voice_tools(store, cart, recent_orders, identity, customers)
    ]


def _graph(config_root: Path, fake: FakeChatModel, **kwargs):
    store = kwargs.pop("store", None) or OrderStore(load_orders_fixture(config_root, "acme_store"))
    # The SAME cart instance must reach build_voice_tools AND the graph, or view_cart reads a
    # different cart than the render node (split-brain, graph docstring). Same for the
    # identity store (P7): the tool grants into it, the render router reads it.
    cart = kwargs.pop("cart_store", None) or CartStore()
    policy = kwargs.pop("policy", None) or make_policy(refund_returnless_under_usd=50.0)
    recent_orders = kwargs.pop("recent_orders", None) or RecentOrderContext(
        max_refs=policy.cancel_batch_max
    )
    identity = kwargs.pop("identity", None) or CallerIdentityStore()
    otp = kwargs.pop("otp", None) or OtpProvider(valid_code=_TEST_OTP)
    verification = kwargs.pop("verification_store", None) or VerificationStore(otp)
    caller_context = CallerContext(
        verification_store=verification,
        cart_store=cart,
        recent_orders=recent_orders,
        identity_store=identity,
        order_store=store,
    )
    return build_frontline_graph(
        fake,
        _tools(config_root, store, cart, recent_orders, identity),
        display_name="Acme Store",
        tenant_id="acme_store",
        cart_store=cart,
        recent_orders=recent_orders,
        otp=otp,
        verification_store=verification,
        identity_store=identity,
        customers=CustomerDirectory(load_customers_fixture(config_root, "acme_store")),
        payment_instruments=PaymentInstrumentDirectory(
            load_payment_instruments_fixture(config_root, "acme_store")
        ),
        profile_store=ProfileStore(load_profile_fixture(config_root, "acme_store")),
        # Frontline-path tests never reach checkout; a default fake keeps one graph shape.
        reasoning_model=kwargs.pop("reasoning_model", None) or FakeChatModel(),
        store=store,
        policy=policy,
        lifecycle=caller_context,
        **kwargs,
    ).graph


def _admitted_turn(text: str, *, turn_id: str, **state: object) -> dict[str, object]:
    return {
        "messages": [HumanMessage(content=text, id=turn_id)],
        "consumed_turn_ids": (turn_id,),
        **state,
    }


# --- the structural safety invariant (T1's structural half) --------------------------


def test_frontline_holds_no_sensitive_tool(config_root: Path) -> None:
    graph = _graph(config_root, FakeChatModel())
    # The only tools the frontline can call are the read-only ones + request_handover
    # (a control signal, not a mutation). NO cart-write / place-order / refund / profile.
    assert graph.frontline_read_only_tools == {
        "order_status",
        "list_orders",
        "catalog_search",
        "view_cart",
    }


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
    )
    assert registry.entry_nodes == (
        "support_capability_entry",
        "cart_view_render",
        "identity_status_render",
        "identity_capability_entry",
        "cart_capability_entry",
        "catalog_entry",
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
        CapabilityId.ANSWER_QUESTION,
        CapabilityId.VERIFY_ORDER_STATUS,
        CapabilityId.DISCLOSE_AI_IDENTITY,
        CapabilityId.REQUEST_PERSON,
    }
    assert registry.resolve(SearchCatalog(query="shoes")).node_name == "catalog_entry"
    assert graph.builder.nodes["catalog_entry"].ends == (
        "catalog_response",
        "catalog_query_clarify",
        "catalog_query_reject",
    )
    assert graph.builder.nodes["catalog_response"].ends == (END,)
    for command_node in ("catalog_entry", "catalog_response"):
        assert not any(source == command_node for source, _target in graph.builder.edges)
        assert command_node not in graph.builder.branches


def test_complete_typed_cart_add_resolves_live_catalog_without_a_model_call(
    config_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = OrderStore(load_orders_fixture(config_root, "acme_store"))
    cart = CartStore()
    reasoning = FakeChatModel(emit_tool_calls=False)
    records: list[dict[str, object]] = []
    monkeypatch.setattr(cart_flow, "write_event", records.append)
    graph = _graph(
        config_root,
        FakeChatModel(),
        store=store,
        cart_store=cart,
        reasoning_model=reasoning,
    )
    product = store.fixture.products[0]
    turn_id = "typed-cart-add"

    result = graph.invoke(
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
    assert cart.view()[0].sku == product.sku
    assert cart.view()[0].price_usd == product.price_usd
    assert result["active_invocation"] is None
    assert result["active_flow"] is None
    assert product.name in _only_spoken(result)
    assert records == [{"event": "cart_item_added", "sku": product.sku}]


def test_resolved_typed_cart_add_revalidates_the_live_catalog_before_effect(
    config_root: Path,
) -> None:
    store = OrderStore(load_orders_fixture(config_root, "acme_store"))
    cart = CartStore()
    graph = _graph(config_root, FakeChatModel(), store=store, cart_store=cart)
    product = store.fixture.products[0]
    turn_id = "typed-cart-resolved"

    result = graph.invoke(
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

    assert cart.view()[0].sku == product.sku
    assert result["active_invocation"] is None


def test_typed_cart_gathers_one_slot_per_committed_turn(config_root: Path) -> None:
    store = OrderStore(load_orders_fixture(config_root, "acme_store"))
    cart = CartStore()
    reasoning = FakeChatModel(
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

    first = graph.nodes["cart_capability_entry"].invoke(first_state)

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
        active_flow="cart",
        clarification_progress=first["clarification_progress"],
    )
    second = graph.nodes["cart_capability_entry"].invoke(second_state)

    assert second["active_invocation"] is None
    assert second["active_flow"] is None
    assert second["pending_ack"]
    assert cart.view()[0].quantity == 2
    assert reasoning.invoke_count == 2


@pytest.mark.parametrize(
    ("operation", "quantity", "expected_quantity", "expected_event"),
    (
        ("set_quantity", 3, 3, "cart_quantity_set"),
        ("set_quantity", 0, None, "cart_quantity_set"),
        ("remove", None, None, "cart_item_removed"),
    ),
)
def test_typed_cart_remove_and_set_resolve_only_live_cart_lines(
    config_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    quantity: int | None,
    expected_quantity: int | None,
    expected_event: str,
) -> None:
    store = OrderStore(load_orders_fixture(config_root, "acme_store"))
    product = store.fixture.products[0]
    cart = CartStore()
    cart.add_item(
        sku=product.sku,
        name=product.name,
        price_usd=product.price_usd,
        quantity=1,
    )
    reasoning = FakeChatModel(emit_tool_calls=False)
    records: list[dict[str, object]] = []
    monkeypatch.setattr(cart_flow, "write_event", records.append)
    graph = _graph(
        config_root,
        FakeChatModel(),
        store=store,
        cart_store=cart,
        reasoning_model=reasoning,
    )
    turn_id = f"typed-cart-{operation}-{quantity}"
    request = ModifyCart(
        operation=operation,
        item=CartItemQuery(query=product.name),
        quantity=quantity,
    )

    result = graph.invoke(
        _admitted_turn(
            f"{operation} {product.name}",
            turn_id=turn_id,
            active_invocation=ActiveInvocation(request=request, opened_turn_id=turn_id),
        )
    )

    assert reasoning.invoke_count == 0
    assert result["active_invocation"] is None
    lines = cart.view()
    if expected_quantity is None:
        assert lines == ()
    else:
        assert lines[0].quantity == expected_quantity
    assert records == [{"event": expected_event, "sku": product.sku}]


def test_typed_cart_no_match_resets_only_item_and_asks_in_code(config_root: Path) -> None:
    reasoning = FakeChatModel(emit_tool_calls=False)
    graph = _graph(config_root, FakeChatModel(), reasoning_model=reasoning)
    turn_id = "typed-cart-no-match"
    request = ModifyCart(
        operation="add",
        item=CartItemQuery(query="product that does not exist"),
        quantity=2,
    )

    result = graph.invoke(
        _admitted_turn(
            "add two unavailable things",
            turn_id=turn_id,
            active_invocation=ActiveInvocation(request=request, opened_turn_id=turn_id),
        )
    )

    retained = result["active_invocation"]
    assert isinstance(retained, ActiveInvocation)
    assert retained.request == ModifyCart(operation="add", quantity=2)
    assert result["active_flow"] == "cart"
    assert _only_spoken(result) == "Which item would you like?"
    assert reasoning.invoke_count == 0


def test_typed_cart_duplicate_name_selection_retains_the_resolved_sku(
    config_root: Path,
) -> None:
    fixture = load_orders_fixture(config_root, "acme_store")
    payload = fixture.model_dump()
    duplicate = {
        **payload["products"][0],
        "sku": "SKU-DUPLICATE",
        "price_usd": payload["products"][0]["price_usd"] + 10,
    }
    payload["products"].append(duplicate)
    store = OrderStore(OrdersFixture.model_validate(payload))
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
        cart_store=cart,
        reasoning_model=reasoning,
    )
    name = store.fixture.products[0].name
    invocation = ActiveInvocation(
        request=ModifyCart(
            operation="add",
            item=CartItemQuery(query=name),
        ),
        opened_turn_id="typed-cart-duplicate",
    )
    first = graph.nodes["cart_capability_entry"].invoke(
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
        item=CartItemChoices(skus=(store.fixture.products[0].sku, "SKU-DUPLICATE")),
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

    second = graph.nodes["cart_capability_entry"].invoke(
        ReasoningState(
            messages=[HumanMessage("the second option", id="typed-cart-selection")],
            consumed_turn_ids=("typed-cart-duplicate", "typed-cart-selection"),
            active_invocation=retained,
            active_flow="cart",
            clarification_progress=first["clarification_progress"],
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

    third = graph.nodes["cart_capability_entry"].invoke(
        ReasoningState(
            messages=[HumanMessage("make it two", id="typed-cart-quantity")],
            consumed_turn_ids=(
                "typed-cart-duplicate",
                "typed-cart-selection",
                "typed-cart-quantity",
            ),
            active_invocation=selected,
            active_flow="cart",
            clarification_progress=second["clarification_progress"],
        )
    )

    assert third["active_invocation"] is None
    assert cart.view()[0].sku == "SKU-DUPLICATE"
    assert cart.view()[0].quantity == 2
    assert reasoning.invoke_count == 2


def test_typed_cart_slot_model_sees_only_the_current_committed_utterance(
    config_root: Path,
) -> None:
    reasoning = FakeChatModel(
        scripted_calls=[[("provide_cart_item", {"candidate_key": "1"})]],
        record_prompts=True,
    )
    graph = _graph(config_root, FakeChatModel(), reasoning_model=reasoning)

    update = graph.nodes["cart_capability_entry"].invoke(
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


def test_typed_cart_boolean_quantity_performs_no_effect(config_root: Path) -> None:
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

    update = graph.nodes["cart_capability_entry"].invoke(
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


def test_stale_resolved_cart_sku_performs_no_effect_and_returns_to_selection(
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

    update = graph.nodes["cart_capability_entry"].invoke(
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


def test_two_invalid_typed_cart_item_keys_enter_bounded_clarification(
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

    update = graph.nodes["cart_capability_entry"].invoke(
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


def test_typed_cart_rejects_a_fixed_slot_proposal_then_accepts_the_missing_slot(
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

    update = graph.nodes["cart_capability_entry"].invoke(
        ReasoningState(
            messages=[HumanMessage("add the first one", id="typed-cart-fixed-slot")],
            consumed_turn_ids=("typed-cart-fixed-slot",),
            active_invocation=invocation,
        )
    )

    assert update["active_invocation"] is None
    assert cart.view()[0].quantity == 2
    assert reasoning.invoke_count == 2
    assert any(
        isinstance(message, ToolMessage) and "Unavailable action" in str(message.content)
        for message in update["messages"]
    )


def test_typed_cart_model_prose_is_replaced_by_code_clarification(config_root: Path) -> None:
    fabricated = "I added it to your cart."
    reasoning = FakeChatModel(emit_tool_calls=False, text_response=fabricated)
    graph = _graph(config_root, FakeChatModel(), reasoning_model=reasoning)
    turn_id = "typed-cart-prose"

    result = graph.invoke(
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


def test_typed_cart_empty_remove_uses_review_ack_and_clears(config_root: Path) -> None:
    reasoning = FakeChatModel(emit_tool_calls=False)
    graph = _graph(config_root, FakeChatModel(), reasoning_model=reasoning)
    turn_id = "typed-cart-empty-remove"

    result = graph.invoke(
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
    assert result["active_flow"] is None
    assert _only_spoken(result) == "Your cart's empty right now - what would you like to add?"
    assert reasoning.invoke_count == 0


def test_all_typed_cart_exit_shapes_clear_the_invocation(config_root: Path) -> None:
    request = ModifyCart(operation="add")
    invocation = ActiveInvocation(request=request, opened_turn_id="typed-cart-exit")

    leave_graph = _graph(
        config_root,
        FakeChatModel(),
        reasoning_model=FakeChatModel(scripted_calls=[[("leave_cart", {})]]),
    )
    leave = leave_graph.nodes["cart_capability_entry"].invoke(
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
    exhausted = exhausted_graph.nodes["cart_capability_entry"].invoke(
        ReasoningState(
            messages=[HumanMessage("I still don't know", id="typed-cart-exit")],
            consumed_turn_ids=("typed-cart-exit",),
            active_invocation=invocation,
            active_flow="cart",
            clarification_progress=ClarificationProgress(flow="cart", reasks=2),
        )
    )
    assert exhausted["active_invocation"] is None
    assert exhausted["active_flow"] == "left_cart"

    for node_name in ("cart_abort", "cart_escape_human"):
        graph = _graph(config_root, FakeChatModel())
        update = graph.nodes[node_name].invoke(
            ReasoningState(
                consumed_turn_ids=("typed-cart-exit",),
                active_invocation=invocation,
                active_flow="cart",
            )
        )
        assert update["active_invocation"] is None


@pytest.mark.parametrize("exit_kind", ("leave", "exhaustion", "abort", "human"))
def test_each_typed_cart_exit_clears_the_invocation_in_compiled_state(
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
        "abort": "never mind",
        "human": "get me a person",
    }[exit_kind]
    state: dict[str, object] = {
        "active_flow": "cart",
        "active_invocation": ActiveInvocation(
            request=ModifyCart(operation="add"),
            opened_turn_id=f"typed-cart-{exit_kind}",
        ),
    }
    if exit_kind == "exhaustion":
        state["clarification_progress"] = ClarificationProgress(flow="cart", reasks=2)

    result = graph.invoke(_admitted_turn(text, turn_id=f"typed-cart-{exit_kind}", **state))

    assert result["active_invocation"] is None
    if exit_kind == "human":
        assert result["automation_terminal"] is True


def test_cross_flow_switch_clears_a_typed_cart_invocation(config_root: Path) -> None:
    reasoning = FakeChatModel(emit_tool_calls=False)
    cart = CartStore()
    graph = _graph(
        config_root,
        FakeChatModel(emit_tool_calls=False),
        cart_store=cart,
        reasoning_model=reasoning,
    )
    turn_id = "typed-cart-cross-flow"

    result = graph.invoke(
        _admitted_turn(
            "actually I want a refund",
            turn_id=turn_id,
            active_flow="cart",
            active_invocation=ActiveInvocation(
                request=ModifyCart(operation="add"),
                opened_turn_id=turn_id,
            ),
        )
    )

    assert result["active_invocation"] is None
    assert result["active_flow"] == "support"
    assert cart.is_empty()


def test_mid_slot_checkout_replaces_typed_cart_without_a_model_call(
    config_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = OrderStore(load_orders_fixture(config_root, "acme_store"))
    product = store.fixture.products[0]
    cart = CartStore()
    cart.add_item(
        sku=product.sku,
        name=product.name,
        price_usd=product.price_usd,
        quantity=1,
    )
    reasoning = FakeChatModel(emit_tool_calls=False)
    records: list[dict[str, object]] = []
    monkeypatch.setattr(frontline_graph, "write_event", records.append)
    graph = _graph(
        config_root,
        FakeChatModel(),
        store=store,
        cart_store=cart,
        reasoning_model=reasoning,
    )
    turn_id = "typed-cart-to-checkout"

    result = graph.invoke(
        _admitted_turn(
            "just check out",
            turn_id=turn_id,
            active_flow="cart",
            active_invocation=ActiveInvocation(
                request=ModifyCart(operation="add"),
                opened_turn_id=turn_id,
            ),
        )
    )

    assert result["active_invocation"] is None
    assert result["pending_placement"] is not None
    assert len(result["__interrupt__"]) == 1
    assert cart.line_count == 1
    assert reasoning.invoke_count == 0
    assert [record for record in records if record.get("event") == "capability_replaced"] == [
        {
            "event": "capability_replaced",
            "from": "modify_cart",
            "destination": "checkout",
            "reason_code": "cart_write",
            "source": "gate",
        }
    ]
    assert not any(record.get("event") == "flow_cross_switch" for record in records)


def test_mid_slot_checkout_opens_a_fresh_place_order_invocation(
    config_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records: list[dict[str, object]] = []
    monkeypatch.setattr(frontline_graph, "write_event", records.append)
    graph = _graph(config_root, FakeChatModel())
    turn_id = "typed-cart-replacement"
    original = ActiveInvocation(
        request=ModifyCart(operation="add"),
        opened_turn_id=turn_id,
    )

    update = graph.nodes["cross_switch"].invoke(
        ReasoningState(
            messages=[HumanMessage("just check out", id=turn_id)],
            consumed_turn_ids=(turn_id,),
            active_flow="cart",
            active_invocation=original,
        )
    )

    replacement = update["active_invocation"]
    assert isinstance(replacement, ActiveInvocation)
    assert type(replacement.request) is PlaceOrder
    assert replacement.invocation_id != original.invocation_id
    assert replacement.opened_turn_id == turn_id
    assert update["handover"] == HandoffRequest(
        destination="checkout",
        reason_code="cart_write",
        source="gate",
    )
    assert records == [
        {
            "event": "capability_replaced",
            "from": "modify_cart",
            "destination": "checkout",
            "reason_code": "cart_write",
            "source": "gate",
        }
    ]


def test_mid_slot_checkout_with_an_empty_cart_uses_the_checkout_line_without_a_model(
    config_root: Path,
) -> None:
    cart = CartStore()
    reasoning = FakeChatModel(emit_tool_calls=False)
    graph = _graph(
        config_root,
        FakeChatModel(),
        cart_store=cart,
        reasoning_model=reasoning,
    )
    turn_id = "typed-empty-cart-to-checkout"

    result = graph.invoke(
        _admitted_turn(
            "just check out",
            turn_id=turn_id,
            active_flow="cart",
            active_invocation=ActiveInvocation(
                request=ModifyCart(operation="add"),
                opened_turn_id=turn_id,
            ),
        )
    )

    assert _only_spoken(result) == "Your cart's empty - what would you like to add?"
    assert reasoning.invoke_count == 0
    assert result["active_invocation"] is None
    assert result["active_flow"] is None
    assert result["pending_placement"] is None
    assert result["pending_clarification"] is None
    assert "__interrupt__" not in result


def test_typed_place_order_snapshot_failure_recovers_without_effect_or_model(
    config_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = OrderStore(load_orders_fixture(config_root, "acme_store"))
    product = store.fixture.products[0]
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
    result = graph.invoke(
        _admitted_turn(
            "place my cart",
            turn_id=turn_id,
            active_invocation=ActiveInvocation(
                request=PlaceOrder(),
                opened_turn_id=turn_id,
            ),
        )
    )

    assert _failed_nodes(tmp_path) == ["cart_capability_entry"]
    assert _only_spoken(result).endswith("Please review your cart before trying again.")
    assert cart.line_count == 1
    assert store.placed_count == 0
    assert reasoning.invoke_count == 0
    assert result["active_invocation"] is None
    assert result["pending_placement"] is None
    assert result.get("pending_recovery") is None
    assert "__interrupt__" not in result


@pytest.mark.parametrize("fails_after_mutation", (False, True))
def test_typed_cart_recovery_reads_live_cart_without_replaying_mutation(
    config_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    fails_after_mutation: bool,
) -> None:
    store = OrderStore(load_orders_fixture(config_root, "acme_store"))
    cart = CartStore()
    graph = _graph(config_root, FakeChatModel(), store=store, cart_store=cart)
    product = store.fixture.products[0]
    if fails_after_mutation:
        real_write = cart_flow.write_event

        def fail_after_write(record: dict[str, object]) -> None:
            if record.get("event") == "cart_item_added":
                raise RuntimeError("injected post-mutation telemetry failure")
            real_write(record)

        monkeypatch.setattr(cart_flow, "write_event", fail_after_write)
    else:

        def fail_before_mutation(*_args, **_kwargs):
            raise RuntimeError("injected pre-mutation failure")

        monkeypatch.setattr(cart_flow, "_apply_resolved_mutation", fail_before_mutation)
    turn_id = f"typed-cart-recovery-{fails_after_mutation}"

    result = graph.invoke(
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

    assert cart.line_count == int(fails_after_mutation)
    assert result["active_invocation"] is None
    line = _only_spoken(result)
    assert "Please review your cart before trying again." in line
    if fails_after_mutation:
        assert product.name in line
    else:
        assert "Your cart is empty." in line


def _only_spoken(result: dict[str, object]) -> str:
    """The turn's single caller-facing line. Fails if a turn spoke twice, which is itself a bug."""
    spoken = [message for message in result["messages"] if isinstance(message, AIMessage)]
    assert len(spoken) == 1
    return str(spoken[0].content)


def test_dispatch_reaches_session_list_owner_without_a_model_call(config_root: Path) -> None:
    reasoning = FakeChatModel(emit_tool_calls=False)
    graph = _graph(config_root, FakeChatModel(), reasoning_model=reasoning)
    turn_id = "typed-session-list"
    invocation = ActiveInvocation(
        request=ListOrders(scope="session"),
        opened_turn_id=turn_id,
    )

    result = graph.invoke(
        _admitted_turn(
            "what did I order on this call?",
            turn_id=turn_id,
            active_invocation=invocation,
        )
    )

    assert reasoning.invoke_count == 0
    assert result["active_invocation"] is None
    assert result["active_flow"] is None
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

        assert update == {"active_flow": "identity"}

    assert frontline.invoke_count == 0 and reasoning.invoke_count == 0


def test_bound_verification_uses_the_typed_owner_without_otp_or_rotation(
    config_root: Path,
) -> None:
    identity = CallerIdentityStore()
    identity.bind(BoundIdentity(customer_ref="CUST-001", masked_contact="number ending 0119"))
    otp = OtpProvider(valid_code=_TEST_OTP)
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

    result = graph.invoke(
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
    assert result["active_flow"] is None
    assert _only_spoken(result) == "You're verified on this call."


# --- capability-dispatched answer owners ---------------------------------------------


def _typed_read(graph, request: IntentRequest, *, turn_id: str, text: str) -> dict[str, object]:
    return graph.invoke(
        _admitted_turn(
            text,
            turn_id=turn_id,
            active_invocation=ActiveInvocation(request=request, opened_turn_id=turn_id),
        )
    )


def test_catalog_owner_uses_one_live_lookup_and_one_tool_incapable_model_call(
    config_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lookup_queries: list[str] = []
    real_lookup = read_flow.lookup_catalog

    def observed_lookup(fixture: OrdersFixture, query: str) -> CatalogLookup:
        lookup_queries.append(query)
        return real_lookup(fixture, query)

    monkeypatch.setattr(read_flow, "lookup_catalog", observed_lookup)
    response_model = FakeChatModel(
        emit_tool_calls=False,
        text_response="We carry trail running shoes for $89.99.",
        record_prompts=True,
    )
    graph = _graph(config_root, response_model)

    result = _typed_read(
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


def test_catalog_no_match_prompt_does_not_authorize_a_relevance_claim(
    config_root: Path,
) -> None:
    response_model = FakeChatModel(
        emit_tool_calls=False,
        text_response="No catalog name matched. The catalog contains trail running shoes.",
        record_prompts=True,
    )
    graph = _graph(config_root, response_model)

    result = _typed_read(
        graph,
        SearchCatalog(query="walking shoes"),
        turn_id="catalog-no-match",
        text="Do you have walking shoes?",
    )

    assert _only_spoken(result) == (
        "No catalog name matched. The catalog contains trail running shoes."
    )
    prompt = response_model._seen_prompts[-1]
    assert "do not claim they match the request" in prompt
    assert "or are relevant alternatives" in prompt
    assert "trail running shoes" in prompt
    assert "waterproof rain jacket" in prompt


def test_catalog_answer_telemetry_uses_the_id_matched_committed_turn(
    config_root: Path,
    tmp_path: Path,
) -> None:
    graph = _graph(
        config_root,
        FakeChatModel(emit_tool_calls=False, text_response="We carry trail running shoes."),
    )

    result = graph.invoke(
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
    assert _answered_rows(tmp_path) == [
        {
            "utterance": "Tell me about running shoes.",
            "outcome": "answered",
            "outcome_detail": "capability_answer",
            "capability": "search_catalog",
            "answer_source": "grounded_model_response",
        }
    ]


def test_catalog_owner_asks_once_then_fills_only_the_query_from_the_next_committed_turn(
    config_root: Path,
    tmp_path: Path,
) -> None:
    response_model = FakeChatModel(emit_tool_calls=False, text_response="We carry everyday socks.")
    graph = _graph(config_root, response_model)
    opening = _typed_read(
        graph,
        SearchCatalog(),
        turn_id="catalog-opening",
        text="What do you sell?",
    )

    assert _only_spoken(opening) == "What product would you like me to look for?"
    assert _answered_rows(tmp_path) == []
    retained = opening["active_invocation"]
    assert retained is not None and retained.request == SearchCatalog()
    continuation = graph.invoke(
        _admitted_turn(
            "everyday socks",
            turn_id="catalog-followup",
            active_invocation=retained,
            consumed_turn_ids=("catalog-opening", "catalog-followup"),
        )
    )

    assert response_model.invoke_count == 1
    assert continuation["active_invocation"] is None
    assert _only_spoken(continuation) == "We carry everyday socks."
    assert _answered_rows(tmp_path) == [
        {
            "utterance": "everyday socks",
            "outcome": "answered",
            "outcome_detail": "capability_answer",
            "capability": "search_catalog",
            "answer_source": "grounded_model_response",
        }
    ]


def test_catalog_owner_rejects_a_blank_followup_without_a_model_call(config_root: Path) -> None:
    response_model = FakeChatModel(emit_tool_calls=False)
    graph = _graph(config_root, response_model)
    invocation = ActiveInvocation(
        request=SearchCatalog(),
        opened_turn_id="catalog-opening",
    )

    result = graph.invoke(
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
def test_catalog_owner_recovers_without_answer_telemetry_on_a_blank_model_response(
    config_root: Path,
    tmp_path: Path,
    text_response: str,
) -> None:
    graph = _graph(
        config_root,
        FakeChatModel(emit_tool_calls=False, text_response=text_response),
    )

    result = _typed_read(
        graph,
        SearchCatalog(query="running"),
        turn_id="catalog-blank-model",
        text="Tell me about running shoes.",
    )

    assert _failed_nodes(tmp_path) == ["catalog_response"]
    assert "hit a snag" in _only_spoken(result)
    assert _answered_rows(tmp_path) == []


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


def test_catalog_owner_rejects_an_unexpected_tool_call_before_speech_or_telemetry(
    config_root: Path,
    tmp_path: Path,
) -> None:
    graph = _graph(config_root, _UnexpectedCatalogToolCall())

    result = _typed_read(
        graph,
        SearchCatalog(query="running"),
        turn_id="catalog-tool-call",
        text="Tell me about running shoes.",
    )

    assert _failed_nodes(tmp_path) == ["catalog_response"]
    assert _only_spoken(result) != "I changed something."
    assert _answered_rows(tmp_path) == []


def test_catalog_owner_fails_closed_when_the_admitted_turn_has_no_matching_message(
    config_root: Path,
    tmp_path: Path,
) -> None:
    response_model = FakeChatModel(emit_tool_calls=False)
    graph = _graph(config_root, response_model)

    result = graph.invoke(
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
    assert _failed_nodes(tmp_path) == ["catalog_response"]
    assert "hit a snag" in _only_spoken(result)
    assert _answered_rows(tmp_path) == []


def test_catalog_speech_authority_is_owned_by_the_response_and_code_question_nodes(
    config_root: Path,
) -> None:
    graph = _graph(config_root, FakeChatModel())

    assert {"catalog_query_clarify", "catalog_query_reject"} <= graph.speakable_nodes
    assert "catalog_response" in graph.model_speech_nodes
    assert "catalog_response" not in graph.speakable_nodes
    assert {"catalog_entry", "catalog_response"}.isdisjoint(
        frontline_graph.TRANSACTIONAL_MODEL_NODES
    )


def test_cart_view_owner_speaks_the_live_cart_without_a_model_call(config_root: Path) -> None:
    cart = CartStore()
    cart.add_item(sku="SKU-1", name="waterproof rain jacket", price_usd=129.0, quantity=1)
    frontline = FakeChatModel()
    reasoning = FakeChatModel()
    graph = _graph(config_root, frontline, cart_store=cart, reasoning_model=reasoning)

    result = _typed_read(graph, ViewCart(), turn_id="typed-cart", text="what's in my cart?")

    assert frontline.invoke_count == 0 and reasoning.invoke_count == 0
    assert result["active_invocation"] is None
    line = _only_spoken(result)
    assert "waterproof rain jacket" in line and "129.00" in line
    assert cart.line_count == 1  # a read mutates nothing


def test_cart_view_owner_re_reads_the_store_on_every_turn(config_root: Path) -> None:
    # Live-read freshness: the owner must render the CURRENT cart, never a value captured when
    # the invocation was opened.
    cart = CartStore()
    cart.add_item(sku="SKU-1", name="waterproof rain jacket", price_usd=129.0, quantity=1)
    graph = _graph(config_root, FakeChatModel(), cart_store=cart)

    first = _only_spoken(_typed_read(graph, ViewCart(), turn_id="cart-1", text="my cart?"))
    cart.add_item(sku="SKU-2", name="trail running shoes", price_usd=95.0, quantity=1)
    second = _only_spoken(_typed_read(graph, ViewCart(), turn_id="cart-2", text="and now?"))

    assert "trail running shoes" not in first
    assert "trail running shoes" in second and "waterproof rain jacket" in second


def test_cart_view_owner_speaks_the_empty_line_with_no_close(config_root: Path) -> None:
    from agnostic_market.agents._copy import all_closes

    graph = _graph(config_root, FakeChatModel(), cart_store=CartStore())

    line = _only_spoken(_typed_read(graph, ViewCart(), turn_id="cart-empty", text="my cart?"))

    assert line == "Your cart's empty at the moment."
    assert not any(line.endswith(close) for close in all_closes())


def test_both_cart_read_paths_speak_the_same_line(config_root: Path) -> None:
    # The tool-driven render and the typed owner must not drift: one helper authors both.
    def build(model: FakeChatModel):
        cart = CartStore()
        cart.add_item(sku="SKU-1", name="waterproof rain jacket", price_usd=129.0, quantity=1)
        return _graph(config_root, model, cart_store=cart)

    typed_line = _only_spoken(
        _typed_read(build(FakeChatModel()), ViewCart(), turn_id="same-typed", text="my cart?")
    )
    tool_result = build(FakeChatModel(scripted_calls=[[("view_cart", {})]])).invoke(
        _admitted_turn("what's in my cart?", turn_id="same-tool")
    )
    tool_line = str(
        [m for m in tool_result["messages"] if isinstance(m, AIMessage) and not m.tool_calls][
            -1
        ].content
    )

    # Both must really be the rendered cart, not an empty line or model prose.
    assert "waterproof rain jacket" in typed_line and "waterproof rain jacket" in tool_line
    # Closes rotate, so compare the sentence before the close.
    assert typed_line.split(" in total.")[0] == tool_line.split(" in total.")[0]


def test_each_cart_read_path_takes_exactly_one_close(
    config_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The close is computed ONCE per turn in read_render_node, before its branch. A helper that
    # called warm_close() itself would advance the module-level rotation twice on a cart read,
    # which no membership assertion would catch.
    calls: list[int] = []

    def counting_close() -> str:
        calls.append(1)
        return "Anything else I can help with?"

    monkeypatch.setattr(frontline_graph, "warm_close", counting_close)
    cart = CartStore()
    cart.add_item(sku="SKU-1", name="waterproof rain jacket", price_usd=129.0, quantity=1)

    _typed_read(
        _graph(config_root, FakeChatModel(), cart_store=cart),
        ViewCart(),
        turn_id="close-typed",
        text="my cart?",
    )
    assert len(calls) == 1

    calls.clear()
    _graph(
        config_root,
        FakeChatModel(scripted_calls=[[("view_cart", {})]]),
        cart_store=cart,
    ).invoke(_admitted_turn("what's in my cart?", turn_id="close-tool"))
    assert len(calls) == 1


def test_identity_status_owner_reports_the_live_binding_only(config_root: Path) -> None:
    from agnostic_market.agents._copy import (
        IDENTITY_STATUS_UNVERIFIED,
        IDENTITY_STATUS_VERIFIED,
        all_closes,
    )

    identity = CallerIdentityStore()
    frontline = FakeChatModel()
    reasoning = FakeChatModel()
    graph = _graph(config_root, frontline, identity=identity, reasoning_model=reasoning)

    unbound_result = _typed_read(graph, ViewIdentityStatus(), turn_id="id-1", text="am I verified?")
    unbound = _only_spoken(unbound_result)
    identity.bind(BoundIdentity(customer_ref="CUST-001", masked_contact="number ending 0119"))
    bound_result = _typed_read(
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


def _rows(tmp_path: Path) -> list[dict]:
    path = tmp_path / "telemetry.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _answered_rows(tmp_path: Path) -> list[dict]:
    return [row for row in _rows(tmp_path) if row.get("outcome") == "answered"]


def _failed_nodes(tmp_path: Path) -> list[str]:
    return [str(row["node"]) for row in _rows(tmp_path) if row.get("event") == "turn_failed"]


def test_every_capability_answer_owner_records_one_answered_turn(
    config_root: Path, tmp_path: Path
) -> None:
    # These nodes END, bypassing finalize_node, so without their own record a typed read would
    # leave no negative in the classifier dataset. All THREE owners, because the parity is the
    # point: one missing call is invisible in any test that only asserts a row's absence.
    cart = CartStore()
    cart.add_item(sku="SKU-1", name="waterproof rain jacket", price_usd=129.0, quantity=1)
    graph = _graph(config_root, FakeChatModel(), cart_store=cart)

    _typed_read(graph, ViewCart(), turn_id="tel-cart", text="what's in my cart?")
    _typed_read(graph, ViewIdentityStatus(), turn_id="tel-id", text="am I verified?")
    _typed_read(graph, ListOrders(scope="session"), turn_id="tel-list", text="what have I ordered?")
    _typed_read(
        graph,
        SearchCatalog(query="running"),
        turn_id="tel-catalog",
        text="what running shoes do you have?",
    )

    assert _answered_rows(tmp_path) == [
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
    assert all("tool" not in row for row in _answered_rows(tmp_path))


def test_rotated_read_continuation_records_no_blank_utterance(
    config_root: Path, tmp_path: Path
) -> None:
    # An ACCOUNT list is the only read `project_principal_transition` lets survive rotation
    # (view_cart, view_identity_status and even a SESSION list are refused), and the engine seeds
    # that fresh thread with no messages. No in-thread utterance means no label, and an
    # empty-string row is a mislabelled classifier negative, so none is written.
    identity = CallerIdentityStore()
    identity.bind(BoundIdentity(customer_ref="CUST-001", masked_contact="number ending 0119"))
    graph = _graph(config_root, FakeChatModel(), identity=identity)

    result = graph.invoke(
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
    assert _answered_rows(tmp_path) == []


def _failing_render(*_args: object, **_kwargs: object) -> NoReturn:
    raise RuntimeError("render failed")


def test_a_failed_cart_render_records_no_answered_turn(
    config_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The record must FOLLOW the line it reports, as the tool path's does. Written first it
    # would claim "answered" for a turn whose render blew up and whose caller heard the snag.
    monkeypatch.setattr(frontline_graph, "render_cart_line", _failing_render)
    cart = CartStore()
    cart.add_item(sku="SKU-1", name="waterproof rain jacket", price_usd=129.0, quantity=1)
    graph = _graph(config_root, FakeChatModel(), cart_store=cart)

    result = _typed_read(graph, ViewCart(), turn_id="cart-fail", text="what's in my cart?")

    # Pinned to the owner: "hit a snag" alone would also pass if some EARLIER node had broken.
    assert _failed_nodes(tmp_path) == ["cart_view_render"]
    assert "hit a snag" in _only_spoken(result)
    assert _answered_rows(tmp_path) == []


def test_a_failed_order_list_render_records_no_answered_turn(
    config_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(support_flow, "render_order_list_line", _failing_render)
    identity = CallerIdentityStore()
    identity.bind(BoundIdentity(customer_ref="CUST-001", masked_contact="number ending 0119"))
    graph = _graph(config_root, FakeChatModel(), identity=identity)

    result = _typed_read(
        graph, ListOrders(scope="account"), turn_id="list-fail", text="what are my orders?"
    )

    assert _failed_nodes(tmp_path) == ["support_capability_render"]
    assert "hit a snag" in _only_spoken(result)
    assert _answered_rows(tmp_path) == []


def test_all_regular_nodes_have_the_reviewed_recovery_policy(config_root: Path) -> None:
    graph = _graph(config_root, FakeChatModel())
    policies = graph.node_recovery_policies
    expected_abandonment = {
        AbandonmentKind.PURE_ABORT: {
            "entry",
            "cross_switch",
            "gate",
            "model",
            "finalize",
            "read_render",
            "forced_status",
            "enumeration_gate",
            "cart_clarify",
            "cart_guardrail",
            "cart_abort",
            "support_assemble",
            "capability_dispatch",
            "support_capability_entry",
            "identity_capability_entry",
            "support_capability_render",
            "cart_view_render",
            "identity_status_render",
            "catalog_entry",
            "catalog_query_clarify",
            "catalog_query_reject",
            "catalog_response",
            "support_clarify",
            "support_guardrail",
            "support_risk_check",
            "support_cancel_guardrail",
            "support_resolve",
            "support_return_guardrail",
            "support_profile_guardrail",
            "support_profile_risk_check",
            "support_abort",
            "identity_assemble",
            "identity_ask_contact",
            "identity_reask",
            "identity_guardrail",
            "identity_risk_check",
            "identity_abort",
        },
        AbandonmentKind.CART_REVIEW: {
            "cart_assemble",
            "cart_capability_entry",
            "cart_ack",
        },
        AbandonmentKind.AUTHORITATIVE_RECONCILE: {
            "cart_place",
            "support_place",
            "support_cancel_void",
            "support_return_place",
            "support_profile_place",
        },
        AbandonmentKind.LIFECYCLE_SPECIAL: {
            "tools",
            "principal_warning",
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
            "cart_escape_human",
            "support_escape_human",
            "identity_escape_human",
        },
    }
    expected_exception = {
        ExceptionAction.SAFE_ABORT: {
            *expected_abandonment[AbandonmentKind.PURE_ABORT],
            "tools",
        },
        ExceptionAction.CART_REVIEW: {
            "cart_assemble",
            "cart_capability_entry",
            "cart_ack",
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
    assert len(policies) == 64
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
    assert len(graph.recovery_handled_nodes) == 63
    assert set(policies) == set().union(*expected_exception.values())
    assert graph.recovery_handled_nodes == frozenset(
        set(policies) - {"automation_terminal_response"}
    )
    assert graph.recovery_handled_infrastructure_nodes == frozenset({RECOVERY_NODE_NAME})
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


def _handover_update(
    graph,
    destination: HandoffDestination,
    reason_code: HandoffReasonCode,
) -> dict[str, object]:
    state = ReasoningState(
        consumed_turn_ids=("handover-turn",),
        handover=HandoffRequest(
            destination=destination,
            reason_code=reason_code,
            source="gate",
        ),
    )
    return graph.nodes["handover"].invoke(state)


def test_total_clear_is_used_only_for_terminal_handover_and_cross_switch(
    config_root: Path,
) -> None:
    graph = _graph(config_root, FakeChatModel())
    assert _handover_update(graph, "human", "other") == {
        **clear_automation_state(),
        "automation_terminal": True,
    }

    cross_switch = graph.nodes["cross_switch"].invoke(
        ReasoningState(
            messages=[HumanMessage("refund my order")],
            active_flow="cart",
        )
    )
    assert cross_switch == {
        **clear_automation_state(),
        "handover": HandoffRequest(
            destination="support",
            reason_code="refund",
            source="gate",
        ),
    }


def test_non_human_handover_entries_remain_partial_updates(config_root: Path) -> None:
    graph = _graph(config_root, FakeChatModel())

    assert _handover_update(graph, "checkout", "cart_write") == {
        "active_flow": "cart",
        "handover": None,
        "clarification_progress": None,
    }
    switch = _handover_update(graph, "support", "switch_account")
    switch_invocation = switch.pop("active_invocation")
    assert switch == {
        "active_flow": "identity",
        "handover": None,
        "identity_claim_misses": 0,
        "clarification_progress": None,
    }
    assert isinstance(switch_invocation, ActiveInvocation)
    assert switch_invocation.request == SwitchAccount()
    assert switch_invocation.opened_turn_id == "handover-turn"

    list_orders = _handover_update(graph, "support", "list_orders")
    list_invocation = list_orders.pop("active_invocation")
    assert list_orders == {
        "active_flow": "identity",
        "handover": None,
        "identity_claim_misses": 0,
        "clarification_progress": None,
    }
    assert isinstance(list_invocation, ActiveInvocation)
    assert list_invocation.request == ListOrders(scope="account")
    assert list_invocation.opened_turn_id == "handover-turn"
    for reason_code in ("refund", "cancel_order", "address_change", "contact_change"):
        assert _handover_update(graph, "support", reason_code) == {
            "active_flow": "support",
            "handover": None,
            "clarification_progress": None,
        }


def test_bound_and_session_list_readbacks_remain_partial_updates(config_root: Path) -> None:
    identity = CallerIdentityStore()
    identity.bind(
        BoundIdentity(customer_ref="CUST-002", masked_contact="email ending example dot com")
    )
    bound_graph = _graph(config_root, FakeChatModel(), identity=identity)
    bound = _handover_update(bound_graph, "support", "list_orders")

    store = OrderStore(load_orders_fixture(config_root, "acme_store"))
    _place_two_session_orders(store)
    session_graph = _graph(config_root, FakeChatModel(), store=store)
    session = _handover_update(session_graph, "support", "list_orders")

    expected_keys = {"messages", "active_flow", "handover", "identity_claim_misses"}
    assert set(bound) == expected_keys
    assert set(session) == expected_keys
    for update in (bound, session):
        assert update["active_flow"] is None
        assert update["handover"] is None
        assert update["identity_claim_misses"] == 0


# --- routing paths -------------------------------------------------------------------


async def test_gate_trip_skips_model_and_hands_over(config_root: Path) -> None:
    # The slim gate trips on high-certainty IRREVERSIBLE requests (here: cancel) BEFORE any
    # generation: the frontline model is never invoked, and the turn enters the support
    # flow directly (cancel is a BUILT capability — it no longer defers; and if support's
    # model bounces and leaves, the gate-skip hands the answer to the frontline model
    # rather than a canned deferral — see checkout's gate-skip test).
    fake = FakeChatModel(tool_call_limit=1)
    reasoning = FakeChatModel(emit_tool_calls=False)  # support clarifies; stays in flow
    graph = _graph(config_root, fake, reasoning_model=reasoning)
    out = await graph.ainvoke({"messages": [HumanMessage("cancel my order please")]})
    assert out["active_flow"] == "support"  # entered by the gate, pre-generation
    assert fake._tool_calls_made == 0  # the frontline model never ran


def test_ambiguous_pronoun_cancel_does_not_force_an_order_handover(config_root: Path) -> None:
    graph = _graph(config_root, FakeChatModel())

    update = graph.nodes["gate"].invoke(
        ReasoningState(messages=[HumanMessage("Actually, you know what? Let's cancel it.")])
    )

    assert update == {}


async def test_cancel_order_enters_the_support_flow(config_root: Path) -> None:
    # Group A: cancel_order is now a BUILT support capability, so a cancel_order handover
    # ENTERS the support flow (it no longer defers — that was the 3c-only behavior). The
    # support assemble model runs and proposes the cancel. (Group C: address/contact change
    # enter too; only payment_change defers.) Fix 2: ORD-1002 is CUST-002's and this session is
    # UNBOUND, so the mutation cannot authorize on a rung-1 pair — it DETOURS into the identity
    # OTP flow (no pending minted, "no pending before auth"); it resumes + mints after the bind.
    reasoning = FakeChatModel(
        force_tool="propose_cancel",
        canned_args={"propose_cancel": {"order_keys": ["ORD-1002"]}},
        tool_call_limit=1,
    )
    graph = _graph(config_root, FakeChatModel(tool_call_limit=1), reasoning_model=reasoning)
    out = await graph.ainvoke(
        {
            "messages": [HumanMessage("actually cancel order ORD-1002", id="cancel-detour-turn")],
            "consumed_turn_ids": ("cancel-detour-turn",),
        }
    )
    assert reasoning._tool_calls_made == 1  # the support model ran and proposed a cancel
    assert out.get("active_flow") == "identity"  # detoured to verify (rung-2 required)
    assert out.get("pending_cancel") is None  # nothing staged before the caller is bound
    invocation = out.get("active_invocation")
    assert isinstance(invocation, ActiveInvocation)
    assert invocation.opened_turn_id == "cancel-detour-turn"
    request = invocation.request
    assert isinstance(request, CancelOrders)
    assert request.target.order_refs == ("ORD-1002",)


async def test_address_change_enters_the_support_flow(config_root: Path) -> None:
    # Group C: address_change flipped from defer -> enter (the profile flow is built).
    frontline = FakeChatModel(
        force_tool="request_handover",
        canned_args={
            "request_handover": {"destination": "support", "reason_code": "address_change"}
        },
        tool_call_limit=1,
    )
    reasoning = FakeChatModel(emit_tool_calls=False)  # support clarifies; stays in flow
    graph = _graph(config_root, frontline, reasoning_model=reasoning)
    out = await graph.ainvoke({"messages": [HumanMessage("I need to update my address")]})
    assert out.get("active_flow") == "support"  # ENTERED (no deferral line)
    texts = [str(m.content) for m in out["messages"] if isinstance(m, AIMessage) and m.content]
    assert not any("support team" in t for t in texts)  # the old deferral is gone


async def test_payment_change_still_defers(config_root: Path) -> None:
    # Phase 5 boundary pin: payment_change must NOT enter (its flow doesn't exist — entering
    # would bounce off the assemble and double-speak). The honest deferral speaks once.
    frontline = FakeChatModel(
        force_tool="request_handover",
        canned_args={
            "request_handover": {"destination": "support", "reason_code": "payment_change"}
        },
        tool_call_limit=1,
    )
    graph = _graph(config_root, frontline)
    out = await graph.ainvoke({"messages": [HumanMessage("put my new card on the account")]})
    assert out.get("active_flow") is None
    texts = [str(m.content) for m in out["messages"] if isinstance(m, AIMessage) and m.content]
    assert sum("support team" in t for t in texts) == 1  # exactly one deferral line


async def test_read_only_turn_answers_without_handover(config_root: Path) -> None:
    fake = FakeChatModel(tool_call_limit=1, canned_args=_READ_ARGS)
    graph = _graph(config_root, fake, identity=_granted("ORD-1001"))
    out = await graph.ainvoke({"messages": [HumanMessage("status of order ORD-1001")]})
    assert out.get("handover") is None
    assert [type(m).__name__ for m in out["messages"]] == [
        "HumanMessage",
        "AIMessage",
        "ToolMessage",
        "AIMessage",
    ]


async def test_model_handover_routes_through_command(config_root: Path) -> None:
    # Trigger-free phrasing the gate can't pattern -> the model calls request_handover.
    # This fake emits an EMPTY-content tool call (no narration), so the node's canned
    # deferral fires as the fallback — the caller is never left silent.
    fake = FakeChatModel(
        force_tool="request_handover", tool_call_limit=1, canned_args=_HANDOVER_ARGS
    )
    graph = _graph(config_root, fake)
    out = await graph.ainvoke(
        {"messages": [HumanMessage("I moved recently, make sure it goes to the right place")]}
    )
    assert out["handover"].source == "model"
    # Proper tool_use/tool_result pairing (G1): the executed handover tool left a ToolMessage.
    assert any(isinstance(m, ToolMessage) for m in out["messages"])
    assert "picked up" in out["messages"][-1].content  # the planner deferral line


async def test_model_narration_is_not_double_spoken(config_root: Path) -> None:
    # Live 2026-07-08 bug: when the model narrates its handover AND the node appends the
    # canned deferral, the caller hears two deferrals. If the model already spoke, the
    # node must stay silent (its deferral is a fallback only).
    class NarratingFake(FakeChatModel):
        def _respond(self, messages, **kwargs):  # type: ignore[override]
            msg = super()._respond(messages, **kwargs)
            if msg.tool_calls:  # attach spoken narration alongside the handover tool call
                msg.content = "I'll connect you with our support team for that."
            return msg

    fake = NarratingFake(
        force_tool="request_handover", tool_call_limit=1, canned_args=_HANDOVER_ARGS
    )
    graph = _graph(config_root, fake)
    out = await graph.ainvoke({"messages": [HumanMessage("plan a whole trip outfit for me")]})
    assert out["handover"].source == "model"
    # The node did NOT append its canned deferral (the model's narration is the deferral).
    canned = "I'll make sure it's picked up"  # the planner deferral line
    assert not any(
        isinstance(m, AIMessage) and canned in (m.content or "") for m in out["messages"]
    )


async def test_two_turn_history_stays_clean(config_root: Path) -> None:
    # F2 regression: after a model-handover turn, the next turn runs without a dangling
    # tool_use breaking the model call.
    fake = FakeChatModel(
        force_tool="request_handover", tool_call_limit=1, canned_args=_HANDOVER_ARGS
    )
    graph = _graph(config_root, fake)
    first = await graph.ainvoke({"messages": [HumanMessage("send it to my work address")]})
    second = await graph.ainvoke(
        {"messages": [*first["messages"], HumanMessage("what's the status of order ORD-1001")]}
    )
    assert second["messages"]  # completed without error


async def test_model_node_prepends_platform_system_prompt(config_root: Path) -> None:
    # F1: the prompt lives inside the graph, so eval and production share one prompt path.
    seen: dict[str, object] = {}

    class RecordingFake(FakeChatModel):
        def _respond(self, messages, **kwargs):  # type: ignore[override]
            seen["first"] = messages[0]
            return super()._respond(messages, **kwargs)

    fake = RecordingFake(tool_call_limit=0, canned_args=_READ_ARGS)
    graph = _graph(config_root, fake)
    await graph.ainvoke({"messages": [HumanMessage("hello")]})
    assert isinstance(seen["first"], SystemMessage)
    assert "Acme Store" in seen["first"].content


async def test_plain_answer_ends_without_tools(config_root: Path) -> None:
    fake = FakeChatModel(emit_tool_calls=False)
    graph = _graph(config_root, fake)
    out = await graph.ainvoke({"messages": [HumanMessage("hi there")]})
    assert out.get("handover") is None
    assert isinstance(out["messages"][-1], AIMessage)
    assert out["messages"][-1].content


async def test_runaway_tool_loop_is_bounded(config_root: Path) -> None:
    # A model that never stops calling read tools must not spin forever: the hop guard ends
    # the turn after policy.max_tool_hops round-trips (no framework loop protection here).
    # Forces catalog_search (NON-renderable — stays on the model->tools->model loop; a
    # single renderable read like order_status would divert to read_render and END after one
    # hop, which is a STRONGER bound, not the loop this guard protects).
    default_hops = make_policy().max_tool_hops
    fake = FakeChatModel(force_tool="catalog_search", canned_args=_READ_ARGS)  # no limit set
    graph = _graph(config_root, fake)
    out = await graph.ainvoke({"messages": [HumanMessage("do you have shoes")]})
    hops = sum(1 for m in out["messages"] if isinstance(m, AIMessage) and m.tool_calls)
    assert hops == default_hops
    assert isinstance(out["messages"][-1], AIMessage)
    assert out["messages"][-1].content  # limit produces a final answer, not a dangling call
    for index, message in enumerate(out["messages"]):
        if isinstance(message, AIMessage) and message.tool_calls:
            assert isinstance(out["messages"][index + 1], ToolMessage)


async def test_tool_hop_bound_tracks_the_policy_knob(config_root: Path) -> None:
    # The bound is config-driven (policies.security.max_tool_hops), not a hardcoded constant:
    # a tightened knob ends the turn sooner. Pins that the value actually THREADS to the guard.
    fake = FakeChatModel(force_tool="catalog_search", canned_args=_READ_ARGS)
    graph = _graph(config_root, fake, policy=make_policy(max_tool_hops=2))
    out = await graph.ainvoke({"messages": [HumanMessage("do you have shoes")]})
    hops = sum(1 for m in out["messages"] if isinstance(m, AIMessage) and m.tool_calls)
    assert hops == 2


async def test_tool_hop_bound_strips_a_provider_tool_call_from_the_final_pass(
    config_root: Path,
) -> None:
    class ToolCallingWithoutToolsFake(FakeChatModel):
        def _respond(self, messages, **kwargs):  # type: ignore[override]
            if not kwargs.get("tools"):
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "catalog_search",
                            "args": {"query": "shoes"},
                            "id": "invalid_final_call",
                            "type": "tool_call",
                        }
                    ],
                )
            return super()._respond(messages, **kwargs)

    fake = ToolCallingWithoutToolsFake(
        force_tool="catalog_search", canned_args=_READ_ARGS, tool_call_limit=None
    )
    graph = _graph(config_root, fake, policy=make_policy(max_tool_hops=1))
    out = await graph.ainvoke({"messages": [HumanMessage("do you have shoes")]})
    tool_messages = [message for message in out["messages"] if isinstance(message, ToolMessage)]
    assert len(tool_messages) == 1
    final = out["messages"][-1]
    assert isinstance(final, AIMessage) and not final.tool_calls and final.content


async def test_state_followup_reads_recent_context_without_calling_the_model(
    config_root: Path,
) -> None:
    store = OrderStore(load_orders_fixture(config_root, "acme_store"))
    store.cancel_order("cancel-1", order_id="ORD-1002")
    recent_orders = RecentOrderContext(max_refs=make_policy().cancel_batch_max)
    recent_orders.record(["ORD-1002"], operation="read")
    identity = _granted("ORD-1002")
    graph = _graph(
        config_root,
        FakeChatModel(raise_transport=True),
        store=store,
        recent_orders=recent_orders,
        identity=identity,
    )
    out = await graph.ainvoke({"messages": [HumanMessage("is it cancelled?")]})
    assert "ORD-1002" in out["messages"][-1].content
    assert "cancelled" in out["messages"][-1].content


async def test_plural_state_followup_corrects_a_memory_claim_from_live_store(
    config_root: Path,
) -> None:
    store = OrderStore(load_orders_fixture(config_root, "acme_store"))
    store.cancel_order("cancel-1", order_id="ORD-1002")
    recent_orders = RecentOrderContext(max_refs=make_policy().cancel_batch_max)
    recent_orders.record(["ORD-1001", "ORD-1002"], operation="list")
    graph = _graph(
        config_root,
        FakeChatModel(raise_transport=True),
        store=store,
        recent_orders=recent_orders,
        identity=_granted("ORD-1001", "ORD-1002"),
    )
    out = await graph.ainvoke(
        {
            "messages": [
                AIMessage("ORD-1001 and ORD-1002 are both cancelled."),
                HumanMessage("so both are cancelled?"),
            ]
        }
    )
    line = out["messages"][-1].content
    assert "ORD-1001" in line and "ORD-1002" in line
    assert "ORD-1001" in line and "was expected" in line
    assert "ORD-1002" in line and "cancelled" in line


async def test_unverified_all_orders_state_check_enters_identity_flow(
    config_root: Path,
) -> None:
    graph = _graph(
        config_root,
        FakeChatModel(raise_transport=True),
        reasoning_model=FakeChatModel(emit_tool_calls=False),
    )
    out = await graph.ainvoke(
        _admitted_turn(
            "are all my orders cancelled?",
            turn_id="all-orders-state-check",
        )
    )
    assert out.get("active_flow") == "identity"


async def test_unverified_explicit_state_check_does_not_bypass_order_authorization(
    config_root: Path,
) -> None:
    graph = _graph(config_root, FakeChatModel(emit_tool_calls=False))
    out = await graph.ainvoke({"messages": [HumanMessage("is ORD-1002 cancelled?")]})
    final = out["messages"][-1]
    assert final.content == _TEXT_RESPONSE
    assert "waterproof rain jacket" not in final.content


async def test_answered_turn_writes_telemetry_negative(config_root: Path, tmp_path) -> None:
    # The classifier dataset needs NEGATIVES: an answered (non-escalated) turn must leave
    # a telemetry line too, not only handovers. (Telemetry is redirected to tmp by conftest.)
    import json

    from agnostic_market.agents import telemetry

    fake = FakeChatModel(emit_tool_calls=False)
    graph = _graph(config_root, fake)
    await graph.ainvoke({"messages": [HumanMessage("hi there")]})
    lines = [
        json.loads(line)
        for line in telemetry._TELEMETRY_PATH.read_text(encoding="utf-8").splitlines()
    ]
    assert any(rec["outcome"] == "answered" and rec["utterance"] == "hi there" for rec in lines)


# --- policy grounding: DERIVED from enforced values (no drift), + free-text extras --------


def _policy(**over) -> PolicyContext:
    # The policy-grounding suite's default carries a spoken_policy_extra; everything else
    # (incl. the security knobs) comes from the shared factory. Any field overridable —
    # `over` wins over these defaults (a test may set spoken_policy_extra=None).
    return make_policy(
        **{
            "refund_returnless_under_usd": 50.0,
            "spoken_policy_extra": "Refunds take 5 to 7 business days.",
            **over,
        }
    )


def test_prompt_grounds_policy_from_enforced_values() -> None:
    from agnostic_market.agents.frontline.prompt import compose_system_prompt

    prompt = compose_system_prompt("Acme Store", _policy())
    assert "Refunds take 5 to 7 business days." in prompt  # the free-text extra
    assert "$50" in prompt  # the returnless threshold, DERIVED from the enforced value
    assert "$200" in prompt  # the human-review line, DERIVED
    assert "ONLY policy statements" in prompt


def test_spoken_policy_tracks_the_enforced_value_no_drift() -> None:
    # The whole point: change the ENFORCED number, the spoken sentence changes with it —
    # they are one source, so a merchant can't state a threshold the guardrail won't honor.
    from agnostic_market.agents.spoken_policy import compose_spoken_policy

    assert "$50" in compose_spoken_policy(_policy(refund_returnless_under_usd=50.0))
    assert "$120" in compose_spoken_policy(_policy(refund_returnless_under_usd=120.0))
    # returnless 0 => return-first for every shipped refund (no dollar threshold spoken).
    zero = compose_spoken_policy(_policy(refund_returnless_under_usd=0.0))
    assert "issued once the return is arranged" in zero
    # The return window (Group C: enforced by the returns guardrail) is derived the same way.
    assert "within 30 days of delivery" in compose_spoken_policy(_policy(return_window_days=30))
    assert "within 60 days of delivery" in compose_spoken_policy(_policy(return_window_days=60))


def test_prompt_speaks_derived_policy_even_without_free_text() -> None:
    from agnostic_market.agents.frontline.prompt import compose_system_prompt

    prompt = compose_system_prompt("Acme Store", _policy(spoken_policy_extra=None))
    assert "$50" in prompt  # derived sentences exist for every merchant
    assert "NEVER invent" in prompt


def test_prompt_forbids_order_status_from_confirming_account_ownership() -> None:
    # Fix 4 (live 2026-07-18): a guest read ORD-1002 via order#+email, then asked "is it on
    # this account, under that email?" and the model answered "Yes, it's on the account for
    # casey@example.com" - an ownership ORACLE (anyone with an order number could learn whose
    # account it is) + a PII echo. The frontline prompt must forbid confirming the account link
    # and repeating the caller's contact; an order_status read speaks order STATE only.
    from agnostic_market.agents.frontline.prompt import compose_system_prompt

    prompt = compose_system_prompt("Acme Store", _policy())
    assert "STATE ONLY" in prompt
    assert "NEVER confirms, denies, or discusses WHOSE account" in prompt
    assert "Never repeat the caller's email or phone number" in prompt
    # the contrastive few-shot for the ownership probe is present
    assert "that order is on this account" in prompt


def test_shared_context_carries_todays_date_and_past_eta_rule() -> None:
    # Live call #9 P6: with no "today" in any prompt, a stored ETA of July 9 was spoken as
    # a FUTURE arrival on July 13. The date is read at compose time (per turn).
    from datetime import datetime

    from agnostic_market.agents._shared_prompt import compose_shared_context

    context = compose_shared_context("Acme Store", _policy())
    assert f"Today's date: {datetime.now():%A %d %B %Y}" in context
    assert "BEFORE today is in the PAST" in context


def test_shared_context_reaches_every_agent_prompt() -> None:
    # Knowledge must not be tier-local (live 2026-07-10: policy facts lived only in the
    # frontline prompt, so a policy question that gate-routed into support met a model
    # with zero policy knowledge). All three composers carry the SAME shared block:
    # persona continuity + the DERIVED policy summary.
    from agnostic_market.agents.cart.prompt import compose_cart_prompt
    from agnostic_market.agents.frontline.prompt import compose_system_prompt
    from agnostic_market.agents.support.prompt import compose_support_prompt
    from agnostic_market.commerce.cart import CartStore
    from agnostic_market.commerce.orders import Candidate, OrderCandidate

    policy = _policy()
    prompts = [
        compose_system_prompt("Acme Store", policy),
        compose_cart_prompt(
            "Acme Store",
            [Candidate(key="1", sku="SKU-1", name="thing", price_usd=1.0)],
            CartStore(),
            policy,
        ),
        compose_support_prompt(
            "Acme Store",
            [
                OrderCandidate(
                    key="1",
                    order_id="ORD-1",
                    summary="a thing",
                    total_usd=1.0,
                    status="processing",
                )
            ],
            policy,
        ),
    ]
    for prompt in prompts:
        assert "ONE continuous assistant" in prompt
        assert "Refunds take 5 to 7 business days." in prompt
        assert "$50" in prompt  # the derived enforced sentence reaches every tier


def test_support_prompt_teaches_batch_cancel_tool() -> None:
    # F-16.2 batch: the support model is told to cancel MULTIPLE orders in ONE propose_cancel
    # call (not one-at-a-time across turns), and to never report an order done from memory. A
    # content pin — the structural fix is code; this guards the wording against regression to
    # the retired (and non-functional) one-at-a-time continuation promise.
    from agnostic_market.agents.support.prompt import compose_support_prompt
    from agnostic_market.commerce.orders import OrderCandidate

    orders = [
        OrderCandidate(key="1", order_id="ORD-1", summary="x", total_usd=1.0, status="processing")
    ]
    prompt = compose_support_prompt("Acme Store", orders, _policy())
    assert "propose_cancel ONCE with ALL their option numbers" in prompt
    assert "NEVER report an order as done from memory" in prompt
    # The retired false promise must not creep back in.
    assert "on the NEXT turn propose the next one" not in prompt


# --- L3 deterministic read renderers: a single order_status/view_cart read is rendered in
#     CODE and ENDs, skipping the second model pass (latency + grounding win). -------------

from llm_fakes import _TEXT_RESPONSE  # noqa: E402  the fake's narration text (single source)


async def test_single_order_status_renders_in_code_and_skips_second_model_pass(
    config_root: Path,
) -> None:
    fake = FakeChatModel(tool_call_limit=1, canned_args=_READ_ARGS)
    # An AUTHORIZED read (P7): render tests exercise the L3 path, not the object-binding
    # gate — the gate's own pins live below and in test_voice_tools.py.
    graph = _graph(config_root, fake, identity=_granted("ORD-1001"))
    out = await graph.ainvoke({"messages": [HumanMessage("status of order ORD-1001")]})
    final = out["messages"][-1]
    # The final line is the CODE render (contains the order id + a store-derived status),
    # NOT the model's narration text — proving the second model pass was skipped.
    assert isinstance(final, AIMessage) and "ORD-1001" in final.content
    assert _TEXT_RESPONSE not in final.content
    # The model was invoked exactly ONCE (the tool-call turn); no narration invoke followed.
    assert fake._tool_calls_made == 1


async def test_render_node_is_speakable(config_root: Path) -> None:
    graph = _graph(config_root, FakeChatModel(tool_call_limit=1, canned_args=_READ_ARGS))
    assert "read_render" in graph.speakable_nodes


async def test_render_appends_a_factual_close_no_product_opinion(config_root: Path) -> None:
    # A status read ends with a warm FACTUAL close — never a product opinion ("nice
    # choice"/"good pick"), which reads as scripted and is nonsensical after a read (live
    # 2026-07-15: appreciative closes dropped).
    from agnostic_market.agents._copy import all_closes

    fake = FakeChatModel(tool_call_limit=1, canned_args=_READ_ARGS)
    graph = _graph(config_root, fake, identity=_granted("ORD-1001"))
    out = await graph.ainvoke({"messages": [HumanMessage("status of order ORD-1001")]})
    line = out["messages"][-1].content
    assert any(line.endswith(c) for c in all_closes())
    assert "pick" not in line.lower() and "choice" not in line.lower()  # no product opinion


async def test_multi_intent_read_still_goes_to_the_model(config_root: Path) -> None:
    # "status of my order AND do you have socks?" -> the model emits TWO tool calls in one
    # response; the ==1 guard fails, so the turn does NOT divert to read_render after the
    # tools run — it returns to the model, which composes both (its narration text). The
    # tool_call_limit makes that post-tools model invoke return text (not loop).
    fake = FakeChatModel(
        scripted_calls=[
            [("order_status", {"order_id": "ORD-1001"}), ("catalog_search", {"query": "socks"})]
        ],
        tool_call_limit=0,  # after the scripted multi-call, the next invoke narrates
    )
    graph = _graph(config_root, fake)
    out = await graph.ainvoke(
        {"messages": [HumanMessage("status of ORD-1001 and do you have socks")]}
    )
    # The model composed the answer (its narration text), NOT a code render.
    assert out["messages"][-1].content == _TEXT_RESPONSE


async def test_single_catalog_search_stays_model_narrated(config_root: Path) -> None:
    # catalog_search is NOT renderable (fuzzy discovery needs framing) — even a single call
    # routes back to the model.
    fake = FakeChatModel(tool_call_limit=1, force_tool="catalog_search", canned_args=_READ_ARGS)
    graph = _graph(config_root, fake)
    out = await graph.ainvoke({"messages": [HumanMessage("do you have shoes")]})
    assert out["messages"][-1].content == _TEXT_RESPONSE


async def test_single_view_cart_renders_in_code(config_root: Path) -> None:
    cart = CartStore()
    cart.add_item(sku="SKU-BLU-07", name="rain jacket", price_usd=129.0, quantity=2)
    fake = FakeChatModel(tool_call_limit=1, force_tool="view_cart", canned_args=_READ_ARGS)
    graph = _graph(config_root, fake, cart_store=cart)
    out = await graph.ainvoke({"messages": [HumanMessage("what's in my cart")]})
    final = out["messages"][-1]
    assert "2 rain jackets" in final.content and "$258.00" in final.content
    assert _TEXT_RESPONSE not in final.content


async def test_render_cannot_invent_forward_state(config_root: Path) -> None:
    # A processing order must render "being prepared", never "on the way" — the phrase is
    # derived from the store field, so the embellishment class is structurally impossible.
    fake = FakeChatModel(
        tool_call_limit=1,
        canned_args={"order_status": {"order_id": "ORD-1002"}},
    )
    graph = _graph(config_root, fake, identity=_granted("ORD-1002"))
    out = await graph.ainvoke({"messages": [HumanMessage("status of ORD-1002")]})
    line = out["messages"][-1].content
    assert "being prepared" in line and "on its way" not in line


async def test_handover_turn_never_renders(config_root: Path) -> None:
    # A request_handover turn sets `handover`; the shared predicate excludes it, so the divert
    # never steals a handover turn — it routes to the handover sink (spoken deferral).
    fake = FakeChatModel(
        tool_call_limit=1, force_tool="request_handover", canned_args=_HANDOVER_ARGS
    )
    graph = _graph(config_root, fake)
    out = await graph.ainvoke({"messages": [HumanMessage("I need a human")]})
    # Deferral spoken (planner destination), NOT a code render.
    assert "picked up" in out["messages"][-1].content.lower()


async def test_unauthorized_order_read_never_renders(config_root: Path) -> None:
    # THE P7 render-gate pin: read_render re-derives the line from the STORE, so a declined
    # order_status followed by the render divert would LEAK the order around the tool's
    # object-binding gate. `_render_ready` requires authorization — an unverified single
    # order_status falls back to the model, which narrates the tool's ask-for-contact
    # instruction (the fake's text), and NO store-derived order line is ever spoken.
    fake = FakeChatModel(tool_call_limit=1, canned_args=_READ_ARGS)
    graph = _graph(config_root, fake)  # identity store fresh: unverified session
    out = await graph.ainvoke({"messages": [HumanMessage("status of order ORD-1001")]})
    final = out["messages"][-1]
    assert final.content == _TEXT_RESPONSE  # model narration, NOT a code render
    spoken = [str(m.content) for m in out["messages"] if isinstance(m, AIMessage) and m.content]
    assert not any("trail running shoes" in t for t in spoken)  # no order data leaked


async def test_list_orders_handover_enters_the_identity_flow(config_root: Path) -> None:
    # P7 rung 2: a list_orders handover ENTERS the identity flow (no deferral); the identity
    # model runs and asks for the contact on the account (clarify -> stays sticky).
    frontline = FakeChatModel(
        force_tool="request_handover",
        canned_args={"request_handover": {"destination": "support", "reason_code": "list_orders"}},
        tool_call_limit=1,
    )
    reasoning = FakeChatModel(emit_tool_calls=False)  # identity model asks its ONE question
    graph = _graph(config_root, frontline, reasoning_model=reasoning)
    out = await graph.ainvoke(
        _admitted_turn("what orders do I have", turn_id="list-orders-handover")
    )
    assert out.get("active_flow") == "identity"  # ENTERED (sticky, awaiting the claim)
    texts = [str(m.content) for m in out["messages"] if isinstance(m, AIMessage) and m.content]
    assert not any("support team" in t for t in texts)  # no stale deferral


async def test_identity_apply_is_speakable(config_root: Path) -> None:
    graph = _graph(config_root, FakeChatModel())
    assert "identity_apply" in graph.speakable_nodes
    assert "identity_reask" in graph.speakable_nodes
    assert "identity_assemble" not in graph.speakable_nodes  # double-speak (cart_ack lesson)


async def test_unverified_enumeration_diverts_without_a_relay_pass(config_root: Path) -> None:
    # THE deterministic enumeration divert (call #13 latency + F-12.3 closed structurally):
    # a single unverified list_orders probe routes STRAIGHT to the identity flow in code.
    # The high-confidence detector avoids the frontline model entirely, so it cannot relay
    # the tool's instruction, ask for the email itself, or narrate.
    frontline = FakeChatModel(force_tool="list_orders", tool_call_limit=1)
    reasoning = FakeChatModel(emit_tool_calls=False)  # identity model asks its ONE question
    graph = _graph(config_root, frontline, reasoning_model=reasoning)
    out = await graph.ainvoke(_admitted_turn("what orders do I have", turn_id="enumeration-divert"))
    assert out.get("active_flow") == "identity"  # entered via the code-set handover
    assert frontline._tool_calls_made == 0
    spoken = [str(m.content) for m in out["messages"] if isinstance(m, AIMessage) and m.content]
    assert not any("email or phone" in t and "verification" in t for t in spoken)


async def test_explicit_enumeration_phrase_skips_frontline_model(config_root: Path) -> None:
    # Live call #17: this exact intent was answered "I can't list orders" because only a
    # model-emitted list_orders tool call triggered the deterministic identity divert.
    frontline = FakeChatModel(raise_transport=True)
    reasoning = FakeChatModel(emit_tool_calls=False)
    graph = _graph(config_root, frontline, reasoning_model=reasoning)
    out = await graph.ainvoke(
        _admitted_turn(
            "tell me what order numbers are available",
            turn_id="explicit-enumeration",
        )
    )
    assert out.get("active_flow") == "identity"


async def test_enumeration_cross_switches_out_of_sticky_support(config_root: Path) -> None:
    # The live failure occurred after a cancel authorization denial left support sticky.
    # Enumeration belongs to identity even though both use the "support" handover destination.
    frontline = FakeChatModel(raise_transport=True)
    reasoning = FakeChatModel(emit_tool_calls=False)
    graph = _graph(config_root, frontline, reasoning_model=reasoning)
    out = await graph.ainvoke(
        _admitted_turn(
            "tell me what orders are available",
            turn_id="enumeration-cross-switch",
            active_flow="support",
        )
    )
    assert out.get("active_flow") == "identity"
    assert out.get("pending_cancel") is None


async def test_bound_enumeration_renders_the_list_in_code(config_root: Path) -> None:
    # A BOUND session's clear enumeration ask is code-authored from the scoped store view
    # and the turn ENDs without re-entering Identity or invoking the frontline model.
    from agnostic_market.commerce.identity import BoundIdentity

    identity = CallerIdentityStore()
    identity.bind(
        BoundIdentity(customer_ref="CUST-002", masked_contact="email ending example dot com")
    )
    fake = FakeChatModel(force_tool="list_orders", tool_call_limit=1)
    graph = _graph(config_root, fake, identity=identity)
    out = await graph.ainvoke({"messages": [HumanMessage("what orders do I have")]})
    final = out["messages"][-1]
    assert "ORD-1002" in final.content and "ORD-1001" not in final.content  # scoped
    assert _TEXT_RESPONSE not in final.content  # code render, not model narration
    assert fake._tool_calls_made == 0


async def test_bound_enumeration_escapes_sticky_support_without_otp(config_root: Path) -> None:
    from agnostic_market.commerce.identity import BoundIdentity

    identity = CallerIdentityStore()
    identity.bind(
        BoundIdentity(customer_ref="CUST-002", masked_contact="email ending example dot com")
    )
    graph = _graph(config_root, FakeChatModel(raise_transport=True), identity=identity)
    out = await graph.ainvoke(
        {
            "messages": [HumanMessage("tell me what orders are available")],
            "active_flow": "support",
        }
    )
    final = out["messages"][-1]
    assert "ORD-1002" in final.content and "ORD-1001" not in final.content
    assert out.get("active_flow") is None
    assert out.get("pending_identity") is None


# --- Fix 3: a GUEST lists the orders they placed THIS session (no verification) ------------


def _place_two_session_orders(store: OrderStore) -> tuple[str, str]:
    from agnostic_market.dtos.state import CartLine

    a = store.place_cart(
        "s1",
        lines=[CartLine(sku="SKU-BLU-07", name="rain jacket", price_usd=129.0, quantity=2)],
        total_usd=258.0,
    )
    b = store.place_cart(
        "s2",
        lines=[CartLine(sku="SKU-RED-42", name="trail shoes", price_usd=89.99, quantity=1)],
        total_usd=89.99,
    )
    return a.order_id, b.order_id


async def test_guest_lists_session_placed_orders_without_verification(
    config_root: Path, tmp_path: Path
) -> None:
    # THE Fix-3 pin (live trace 2026-07-18): an UNBOUND caller who placed orders this call asks
    # to list them and hears them read back from CODE — no identity handover, no OTP. The spoken
    # line DISCLOSES this-call scope (not "you've got N orders", which implies full history) and
    # carries exactly ONE closing invitation (the verify-for-more line, NOT also warm_close).
    import json

    from agnostic_market.agents._copy import GUEST_LIST_CLOSE, all_closes

    store = OrderStore(load_orders_fixture(config_root, "acme_store"))
    a, b = _place_two_session_orders(store)
    graph = _graph(config_root, FakeChatModel(raise_transport=True), store=store)
    out = await graph.ainvoke({"messages": [HumanMessage("tell me what orders are there")]})
    final = out["messages"][-1]
    assert a in final.content and b in final.content  # both session orders read back
    assert out.get("active_flow") is None  # NO identity detour
    assert out.get("pending_identity") is None
    assert "on this call" in final.content  # scope disclosed (not full-history phrasing)
    assert GUEST_LIST_CLOSE in final.content  # the verify-for-more invite
    assert not any(c in final.content for c in all_closes())  # and NOT also a warm_close
    scope = [
        json.loads(line)
        for line in (tmp_path / "telemetry.jsonl").read_text(encoding="utf-8").splitlines()
        if '"order_scope"' in line
    ]
    assert scope and scope[-1]["order_scope"] == "session"


async def test_guest_enumeration_never_lists_fixture_orders(config_root: Path) -> None:
    # SECURITY: a guest's session list is drawn ONLY from what they placed — never an account's
    # fixture orders (a different code path). Place ONE order; the spoken list must not name any
    # fixture order id.
    from agnostic_market.dtos.state import CartLine

    store = OrderStore(load_orders_fixture(config_root, "acme_store"))
    placed = store.place_cart(
        "one",
        lines=[CartLine(sku="SKU-BLU-07", name="rain jacket", price_usd=129.0, quantity=1)],
        total_usd=129.0,
    )
    graph = _graph(config_root, FakeChatModel(raise_transport=True), store=store)
    out = await graph.ainvoke({"messages": [HumanMessage("what orders do I have")]})
    final = out["messages"][-1]
    assert placed.order_id in final.content
    assert not any(x in final.content for x in ("ORD-1001", "ORD-1002", "ORD-1003"))


async def test_guest_with_no_placed_orders_still_enters_identity(config_root: Path) -> None:
    # An unbound caller who placed NOTHING has no session orders to read — so enumeration still
    # enters the identity flow (today's behavior; nothing to list without an account).
    frontline = FakeChatModel(
        force_tool="request_handover",
        canned_args={"request_handover": {"destination": "support", "reason_code": "list_orders"}},
        tool_call_limit=1,
    )
    graph = _graph(config_root, frontline, reasoning_model=FakeChatModel(emit_tool_calls=False))
    out = await graph.ainvoke(
        _admitted_turn("what orders do I have", turn_id="guest-empty-enumeration")
    )
    assert out.get("active_flow") == "identity"


async def test_guest_status_list_reads_session_placed(config_root: Path) -> None:
    # Symmetry with the state-verification path: an unbound guest asking "are they both shipped?"
    # reads the session-placed orders (forced-status), not nothing.
    store = OrderStore(load_orders_fixture(config_root, "acme_store"))
    a, b = _place_two_session_orders(store)
    graph = _graph(config_root, FakeChatModel(raise_transport=True), store=store)
    out = await graph.ainvoke({"messages": [HumanMessage("are both of my orders shipped?")]})
    final = out["messages"][-1]
    assert a in final.content and b in final.content
    assert out.get("active_flow") is None  # answered in code, no identity


async def test_render_emits_exactly_one_answered_event(config_root: Path, tmp_path: Path) -> None:
    # The render path ENDs at read_render (bypassing finalize_node); it must emit exactly ONE
    # answered-telemetry event, not zero and not a duplicate. (Telemetry is redirected to
    # tmp_path by the autouse conftest fixture.)
    import json

    fake = FakeChatModel(tool_call_limit=1, canned_args=_READ_ARGS)
    graph = _graph(config_root, fake, identity=_granted("ORD-1001"))
    await graph.ainvoke({"messages": [HumanMessage("status of order ORD-1001")]})
    sink = tmp_path / "telemetry.jsonl"
    answered = [
        json.loads(line)
        for line in sink.read_text(encoding="utf-8").splitlines()
        if '"outcome": "answered"' in line
    ]
    assert len(answered) == 1
    assert answered[0]["outcome_detail"] == "code_render"
