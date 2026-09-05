"""Cart flow at the GRAPH level (Group B): mutation acks, the whole-cart placement tail,
direct-buy, abort-keeps-cart, stickiness/escapes/cross-switch, view_cart, A10a. Zero network
(fake models + InMemorySaver). Folds the old test_checkout_flow.py — the single-line checkout
flow is now the cart flow's placement tail.

Entry: 'checkout now' trips the GATE (cart_write -> the cart flow) with no frontline fake
needed; the flow's reasoning fake then decides what happens inside (mutate / place / clarify).
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END
from langgraph.types import Command
from llm_fakes import (
    TEST_CALLER_AUDIBLE_MODEL_TEXT_MAX_CHARS,
    TEST_STRUCTURED_OUTPUT_METHOD,
    FakeChatModel,
)
from policy_helpers import make_policy
from pydantic import ValidationError
from routing_helpers import make_routing_session
from telemetry_helpers import make_session_telemetry
from turn_helpers import (
    TEST_CANCELLATION_QUIESCENCE_TIMEOUT_SECONDS,
    engine_events,
)
from verification_helpers import make_otp_provider

from agnostic_market.agents.cart import flow as cart_flow
from agnostic_market.agents.engine import ReasoningEngine
from agnostic_market.agents.frontline import build_frontline_graph
from agnostic_market.agents.telemetry import InMemoryTelemetrySink, TenantTelemetry
from agnostic_market.commerce.cart import CartMutationError, CartStore
from agnostic_market.commerce.catalog import FixtureCatalog
from agnostic_market.commerce.identity import (
    CallerIdentityStore,
    CustomerDirectory,
    load_customers_fixture,
)
from agnostic_market.commerce.orders import (
    GuestOrderScope,
    OrderStore,
    RecentOrderContext,
    load_orders_fixture,
    speak_quantity,
)
from agnostic_market.commerce.payment_instruments import (
    PaymentInstrumentDirectory,
    load_payment_instruments_fixture,
)
from agnostic_market.commerce.profile import ProfileStore, load_profile_fixture
from agnostic_market.commerce.receipts import IndeterminateReceipt, NotCommittedReceipt
from agnostic_market.commerce.verification import RiskProvider, VerificationStore
from agnostic_market.dtos.events import InterruptEvent, SpokenMessageEvent, TokenEvent
from agnostic_market.dtos.orchestration import (
    ActiveInvocation,
    CartItemQuery,
    ModifyCart,
    PlaceOrder,
)
from agnostic_market.dtos.state import PendingCartMutation
from agnostic_market.durability.session_state import SessionStateCoordinator
from agnostic_market.session import CallerContext

_POLICY = make_policy(refund_returnless_under_usd=50.0)
_CFG = {"configurable": {"thread_id": "t1"}}
_CART_CLARIFICATION_LINES = {
    "action": "What would you like to do with your cart?",
    "item": "Which item would you like?",
    "quantity": "How many would you like?",
}


def _tool_fake(name: str, args: dict, *, limit: int = 1) -> FakeChatModel:
    return FakeChatModel(force_tool=name, canned_args={name: args}, tool_call_limit=limit)


def _build(
    config_root: Path,
    *,
    frontline: FakeChatModel | None = None,
    reasoning: FakeChatModel | None = None,
    cart: CartStore | None = None,
    recent_orders: RecentOrderContext | None = None,
    telemetry=None,
):
    fixture = load_orders_fixture(config_root, "acme_store")
    catalog = FixtureCatalog("acme_store", fixture)
    store = OrderStore("acme_store", fixture.orders)
    cart = cart or CartStore()
    recent_orders = recent_orders or RecentOrderContext(max_refs=_POLICY.cancel_batch_max)
    identity = CallerIdentityStore()
    telemetry = telemetry or make_session_telemetry("acme_store", "t1")
    guest_orders = GuestOrderScope(
        tenant_id="acme_store",
        session_id=telemetry.session_id,
    )
    customers = CustomerDirectory("acme_store", load_customers_fixture(config_root, "acme_store"))
    otp = make_otp_provider()
    verification = VerificationStore(otp, session_id=telemetry.session_id)
    caller_context = CallerContext(
        verification_store=verification,
        session_state=SessionStateCoordinator(cart, recent_orders, guest_orders),
        identity_store=identity,
        telemetry=telemetry.operational,
    )
    assembly = build_frontline_graph(
        frontline or FakeChatModel(emit_tool_calls=False),
        display_name="Acme Store",
        tenant_id="acme_store",
        reasoning_model=reasoning or FakeChatModel(emit_tool_calls=False),
        store=store,
        catalog=catalog,
        guest_orders=guest_orders,
        cart_store=cart,  # the SAME cart the view_cart tool reads (no split-brain)
        verification_store=verification,
        risk=RiskProvider("acme_store"),
        recent_orders=recent_orders,
        identity_store=identity,  # the SAME store the order_status gate grants into (P7)
        customers=customers,
        payment_instruments=PaymentInstrumentDirectory(
            "acme_store", load_payment_instruments_fixture(config_root, "acme_store")
        ),
        profile_store=ProfileStore("acme_store", load_profile_fixture(config_root, "acme_store")),
        policy=_POLICY,
        lifecycle=caller_context,
        structured_output_method=TEST_STRUCTURED_OUTPUT_METHOD,
        caller_audible_model_text_max_chars=TEST_CALLER_AUDIBLE_MODEL_TEXT_MAX_CHARS,
        response_model_node_timeout_seconds=2.0,
        reasoning_model_node_timeout_seconds=6.0,
        session_telemetry=telemetry,
        checkpointer=InMemorySaver(),
    )
    return assembly.graph, store, cart


def _flow(graph) -> str | None:
    return graph.get_state(_CFG).values.get("execution_owner")


def _ai_texts(out) -> list[str]:
    return [str(m.content) for m in out["messages"] if isinstance(m, AIMessage) and m.content]


async def _events(engine: ReasoningEngine, text: str) -> list:
    return await engine_events(engine, text)


def _reasoning_engine(graph) -> ReasoningEngine:
    telemetry = make_session_telemetry("acme_store", "t1")
    return ReasoningEngine(
        graph,
        tenant_id="acme_store",
        deployment_id="test-deployment",
        thread_id="t1",
        checkpoint_io_timeout_seconds=2.0,
        cancellation_quiescence_timeout_seconds=(TEST_CANCELLATION_QUIESCENCE_TIMEOUT_SECONDS),
        routing=make_routing_session(
            graph.capability_registry,
            identity_store=CallerIdentityStore(),
            cart_store=CartStore(),
            recent_orders=RecentOrderContext(max_refs=_POLICY.cancel_batch_max),
            telemetry=telemetry.routing_evidence,
        ),
        telemetry=telemetry.operational,
    )


def _assert_tool_calls_paired(state: dict) -> None:
    tool_use_ids = Counter(
        call["id"]
        for message in state["messages"]
        if isinstance(message, AIMessage)
        for call in message.tool_calls
    )
    tool_result_ids = Counter(
        message.tool_call_id for message in state["messages"] if isinstance(message, ToolMessage)
    )
    assert tool_use_ids == tool_result_ids


def _assert_cart_clarification(
    graph, store: OrderStore, cart: CartStore, events: list, line: str
) -> None:
    assert not any(isinstance(event, TokenEvent | InterruptEvent) for event in events)
    assert [
        (event.node, event.text) for event in events if isinstance(event, SpokenMessageEvent)
    ] == [("cart_clarify", line)]
    snapshot = graph.get_state(_CFG)
    state = snapshot.values
    assert snapshot.next == ()
    assert state.get("execution_owner") == "cart"
    assert state.get("pending_clarification") is None
    for field in (
        "pending_placement",
        "pending_refund",
        "pending_cancel",
        "pending_return",
        "pending_profile_change",
        "pending_identity",
    ):
        assert state.get(field) is None
    assert not snapshot.interrupts
    assert cart.is_empty()
    assert store.placed_count == 0


async def _enter_cart(graph) -> None:
    """Enter the cart flow via the gate ('checkout now' -> cart_write -> cart flow)."""
    await graph.ainvoke({"messages": [HumanMessage("checkout now please")]}, _CFG)
    assert _flow(graph) == "cart"


# --- speech-native rendering (VOICE_PIPELINE §7 — no 'x' for TTS; no double plural) -------


@pytest.mark.parametrize(
    ("qty", "name", "expected"),
    [
        (1, "waterproof rain jacket", "1 waterproof rain jacket"),
        (2, "waterproof rain jacket", "2 waterproof rain jackets"),
        (2, "trail running shoes", "2 trail running shoes"),  # already plural -> no 'shoess'
    ],
)
def test_speak_quantity_pluralizes_without_x_or_double_s(qty, name, expected) -> None:
    result = speak_quantity(qty, name)
    assert result == expected
    assert " x " not in result


# --- code-authored Cart clarification ----------------------------------------------------


@pytest.mark.parametrize(
    ("proposal_model", "arguments"),
    (
        (cart_flow._ProposeKey, {"candidate_key": "1", "unexpected": True}),
        (cart_flow._ProposeQuantity, {"quantity": 1, "unexpected": True}),
    ),
)
def test_cart_slot_models_reject_undeclared_fields(
    proposal_model: type, arguments: dict[str, object]
) -> None:
    with pytest.raises(ValidationError):
        proposal_model.model_validate(arguments)


@pytest.mark.parametrize("quantity", (True, False, "2"))
def test_cart_quantity_slot_is_a_strict_integer(quantity: object) -> None:
    with pytest.raises(ValidationError, match="quantity"):
        cart_flow._ProposeQuantity.model_validate({"quantity": quantity})


# --- whole-cart placement ----------------------------------------------------------------


async def test_typed_cart_mutation_requires_confirmation_before_effect(
    config_root: Path,
) -> None:
    graph, _store, cart = _build(config_root)
    product = load_orders_fixture(config_root, "acme_store").products[0]
    turn_id = "typed-cart-confirmation"

    await graph.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content=f"add two {product.name}",
                    id=turn_id,
                )
            ],
            "consumed_turn_ids": (turn_id,),
            "active_invocation": ActiveInvocation(
                request=ModifyCart(
                    operation="add",
                    item=CartItemQuery(query=product.name),
                    quantity=2,
                ),
                opened_turn_id=turn_id,
            ),
        },
        _CFG,
    )

    paused = graph.get_state(_CFG)
    assert cart.is_empty()
    assert paused.interrupts
    assert paused.interrupts[0].value == (f"Just to confirm: add 2 of {product.name} to your cart?")
    assert paused.values["active_invocation"] is None
    assert paused.values["execution_owner"] == "cart"
    assert paused.values["pending_cart_mutation"].sku == product.sku

    out = await graph.ainvoke(Command(resume={"text": "yes"}), _CFG)

    assert cart.view()[0].quantity == 2
    assert graph.get_state(_CFG).values.get("pending_cart_mutation") is None
    assert any(product.name in line for line in _ai_texts(out))


def test_confirmed_cart_mutation_is_store_idempotent() -> None:
    cart = CartStore()

    first = cart.apply_confirmed_mutation(
        "cart-mutation-1",
        operation="add",
        sku="SKU-1",
        name="trail shoes",
        price_usd=79.0,
        quantity=2,
        pre_confirm_quantity=0,
    )
    replay = cart.apply_confirmed_mutation(
        "cart-mutation-1",
        operation="add",
        sku="SKU-1",
        name="trail shoes",
        price_usd=79.0,
        quantity=2,
        pre_confirm_quantity=0,
    )

    assert replay == first
    assert cart.view()[0].quantity == 2
    assert (
        cart.mutation_receipt(
            "cart-mutation-1",
            operation="add",
            sku="SKU-1",
            name="trail shoes",
            price_usd=79.0,
            quantity=2,
            pre_confirm_quantity=0,
        ).record
        == first
    )


def test_cart_mutation_receipts_reject_key_conflicts_and_report_absence() -> None:
    cart = CartStore()
    cart.apply_confirmed_mutation(
        "cart-mutation-1",
        operation="add",
        sku="SKU-1",
        name="trail shoes",
        price_usd=79.0,
        quantity=2,
        pre_confirm_quantity=0,
    )

    conflict = cart.mutation_receipt(
        "cart-mutation-1",
        operation="add",
        sku="SKU-1",
        name="trail shoes",
        price_usd=79.0,
        quantity=3,
        pre_confirm_quantity=0,
    )
    proposal_state_conflict = cart.mutation_receipt(
        "cart-mutation-1",
        operation="add",
        sku="SKU-1",
        name="trail shoes",
        price_usd=79.0,
        quantity=2,
        pre_confirm_quantity=1,
    )
    absent = cart.mutation_receipt(
        "cart-mutation-absent",
        operation="remove",
        sku="SKU-2",
        name="hiking socks",
        price_usd=14.5,
        quantity=None,
        pre_confirm_quantity=0,
    )

    assert conflict == IndeterminateReceipt(reason="key_conflict")
    assert proposal_state_conflict == IndeterminateReceipt(reason="key_conflict")
    assert isinstance(absent, NotCommittedReceipt)
    with pytest.raises(CartMutationError, match="different parameters"):
        cart.apply_confirmed_mutation(
            "cart-mutation-1",
            operation="add",
            sku="SKU-1",
            name="trail shoes",
            price_usd=79.0,
            quantity=3,
            pre_confirm_quantity=0,
        )
    assert cart.view()[0].quantity == 2
    with pytest.raises(CartMutationError, match="must not be blank"):
        cart.mutation_receipt(
            " ",
            operation="add",
            sku="SKU-1",
            name="trail shoes",
            price_usd=79.0,
            quantity=2,
            pre_confirm_quantity=0,
        )


def test_confirmed_remove_of_absent_line_is_a_truthful_noop() -> None:
    cart = CartStore()

    record = cart.apply_confirmed_mutation(
        "cart-remove-absent",
        operation="remove",
        sku="SKU-1",
        name="trail shoes",
        price_usd=79.0,
        quantity=None,
        pre_confirm_quantity=1,
    )

    assert record.outcome == "unchanged"
    assert record.previous_quantity == 0
    assert record.final_quantity == 0
    assert cart.is_empty()


def test_invalid_confirmed_mutation_fails_before_changing_cart() -> None:
    cart = CartStore()

    with pytest.raises(ValidationError):
        cart.apply_confirmed_mutation(
            "cart-invalid",
            operation="add",
            sku="SKU-1",
            name="trail shoes",
            price_usd=79.0,
            quantity=1,
            pre_confirm_quantity=-1,
        )

    assert cart.is_empty()
    assert isinstance(
        cart.mutation_receipt(
            "cart-invalid",
            operation="add",
            sku="SKU-1",
            name="trail shoes",
            price_usd=79.0,
            quantity=1,
            pre_confirm_quantity=0,
        ),
        NotCommittedReceipt,
    )


def test_cart_clear_discards_confirmed_mutation_receipts() -> None:
    cart = CartStore()
    cart.apply_confirmed_mutation(
        "cart-mutation-1",
        operation="add",
        sku="SKU-1",
        name="trail shoes",
        price_usd=79.0,
        quantity=1,
        pre_confirm_quantity=0,
    )

    cart.clear()

    receipt = cart.mutation_receipt(
        "cart-mutation-1",
        operation="add",
        sku="SKU-1",
        name="trail shoes",
        price_usd=79.0,
        quantity=1,
        pre_confirm_quantity=0,
    )
    assert isinstance(receipt, NotCommittedReceipt)


@pytest.mark.parametrize(
    "invalid_shape",
    (
        {"operation": "add", "quantity": None},
        {"operation": "add", "quantity": 0},
        {"operation": "remove", "quantity": 1},
        {"operation": "set_quantity", "quantity": None},
    ),
)
def test_pending_cart_mutation_rejects_incoherent_quantity_shapes(
    invalid_shape: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        PendingCartMutation(
            sku="SKU-1",
            name="trail shoes",
            price_usd=79.0,
            pre_confirm_quantity=0,
            idempotency_key="cart-mutation-1",
            created_at=1.0,
            **invalid_shape,
        )


def test_pending_cart_mutation_is_frozen_and_extra_forbidden() -> None:
    pending = PendingCartMutation(
        operation="add",
        sku="SKU-1",
        name="trail shoes",
        price_usd=79.0,
        quantity=1,
        pre_confirm_quantity=0,
        idempotency_key="cart-mutation-1",
        created_at=1.0,
    )

    with pytest.raises(ValidationError, match="frozen"):
        pending.quantity = 2
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        PendingCartMutation.model_validate({**pending.model_dump(), "caller_text": "add it"})
    with pytest.raises(ValidationError, match="must not be blank"):
        PendingCartMutation.model_validate({**pending.model_dump(), "idempotency_key": " "})


@pytest.mark.parametrize(
    ("operation", "quantity", "starting_quantity", "final_quantity", "event", "action"),
    (
        ("add", 2, 0, 2, "cart_item_added", "add 2 of {name} to your cart"),
        ("remove", None, 2, 0, "cart_item_removed", "remove {name} from your cart"),
        ("set_quantity", 3, 2, 3, "cart_quantity_set", "set {name} to 3 in your cart"),
        ("set_quantity", 0, 2, 0, "cart_quantity_set", "set {name} to 0 in your cart"),
    ),
)
async def test_typed_cart_mutations_confirm_then_apply_one_authoritative_effect(
    config_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    quantity: int | None,
    starting_quantity: int,
    final_quantity: int,
    event: str,
    action: str,
) -> None:
    sink = InMemoryTelemetrySink()
    telemetry = TenantTelemetry("acme_store", sink, sink).bind_session("typed-cart-mutation")
    graph, _store, cart = _build(config_root, telemetry=telemetry)
    product = load_orders_fixture(config_root, "acme_store").products[0]
    if starting_quantity:
        cart.add_item(
            sku=product.sku,
            name=product.name,
            price_usd=product.price_usd,
            quantity=starting_quantity,
        )
    turn_id = f"typed-{operation}-{quantity}"

    await graph.ainvoke(
        {
            "messages": [HumanMessage(f"{operation} {product.name}", id=turn_id)],
            "consumed_turn_ids": (turn_id,),
            "active_invocation": ActiveInvocation(
                request=ModifyCart(
                    operation=operation,
                    item=CartItemQuery(query=product.name),
                    quantity=quantity,
                ),
                opened_turn_id=turn_id,
            ),
        },
        _CFG,
    )

    paused = graph.get_state(_CFG)
    pending = paused.values["pending_cart_mutation"]
    assert isinstance(pending, PendingCartMutation)
    assert pending.pre_confirm_quantity == starting_quantity
    assert paused.interrupts[0].value == (f"Just to confirm: {action.format(name=product.name)}?")
    if starting_quantity:
        assert cart.view()[0].quantity == starting_quantity
    else:
        assert cart.is_empty()
    assert sink.records == ()

    out = await graph.ainvoke(Command(resume={"text": "yes"}), _CFG)

    lines = cart.view()
    assert (lines[0].quantity if lines else 0) == final_quantity
    assert [{"event": record.event, **record.attributes} for record in sink.records] == [
        {"event": event, "sku": product.sku}
    ]
    assert graph.get_state(_CFG).values.get("pending_cart_mutation") is None
    assert graph.get_state(_CFG).values.get("execution_owner") is None
    assert len(_ai_texts(out)) == 1
    assert product.name in _ai_texts(out)[0]


async def test_cart_mutation_decline_is_exact_and_has_no_effect(config_root: Path) -> None:
    graph, _store, cart = _build(config_root)
    product = load_orders_fixture(config_root, "acme_store").products[0]
    turn_id = "typed-cart-decline"
    await graph.ainvoke(
        {
            "messages": [HumanMessage("add it", id=turn_id)],
            "consumed_turn_ids": (turn_id,),
            "active_invocation": ActiveInvocation(
                request=ModifyCart(
                    operation="add",
                    item=CartItemQuery(query=product.name),
                    quantity=1,
                ),
                opened_turn_id=turn_id,
            ),
        },
        _CFG,
    )

    out = await graph.ainvoke(Command(resume={"text": "no"}), _CFG)

    assert cart.is_empty()
    assert _ai_texts(out) == ["Okay, I won't change your cart."]
    assert graph.get_state(_CFG).values.get("pending_cart_mutation") is None


async def test_cart_mutation_unclear_twice_uses_one_fixed_retry_then_declines(
    config_root: Path,
) -> None:
    graph, _store, cart = _build(config_root)
    product = load_orders_fixture(config_root, "acme_store").products[0]
    turn_id = "typed-cart-unclear"
    await graph.ainvoke(
        {
            "messages": [HumanMessage("add it", id=turn_id)],
            "consumed_turn_ids": (turn_id,),
            "active_invocation": ActiveInvocation(
                request=ModifyCart(
                    operation="add",
                    item=CartItemQuery(query=product.name),
                    quantity=1,
                ),
                opened_turn_id=turn_id,
            ),
        },
        _CFG,
    )

    await graph.ainvoke(Command(resume={"text": "maybe"}), _CFG)
    retry = graph.get_state(_CFG).interrupts
    assert len(retry) == 1
    assert retry[0].value == (
        f"Sorry - just to be clear: add 1 of {product.name} to your cart. Please say yes or no."
    )
    assert cart.is_empty()

    out = await graph.ainvoke(Command(resume={"text": "still maybe"}), _CFG)

    assert _ai_texts(out) == ["Okay, I won't change your cart."]
    assert cart.is_empty()
    assert graph.get_state(_CFG).values.get("pending_cart_mutation") is None


async def test_cart_mutation_human_reply_uses_the_existing_terminal_path(
    config_root: Path,
) -> None:
    graph, _store, cart = _build(config_root)
    product = load_orders_fixture(config_root, "acme_store").products[0]
    turn_id = "typed-cart-human"
    await graph.ainvoke(
        {
            "messages": [HumanMessage("add it", id=turn_id)],
            "consumed_turn_ids": (turn_id,),
            "active_invocation": ActiveInvocation(
                request=ModifyCart(
                    operation="add",
                    item=CartItemQuery(query=product.name),
                    quantity=1,
                ),
                opened_turn_id=turn_id,
            ),
        },
        _CFG,
    )

    out = await graph.ainvoke(
        Command(resume={"text": "I want a person", "handoff_source": "semantic_router"}),
        _CFG,
    )

    assert cart.is_empty()
    assert out["automation_terminal"] is True
    assert graph.get_state(_CFG).values.get("pending_cart_mutation") is None


async def test_cart_mutation_expiry_is_exact_and_does_not_consume_yes(
    config_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph, _store, cart = _build(config_root)
    product = load_orders_fixture(config_root, "acme_store").products[0]
    turn_id = "typed-cart-expiry"
    await graph.ainvoke(
        {
            "messages": [HumanMessage("add it", id=turn_id)],
            "consumed_turn_ids": (turn_id,),
            "active_invocation": ActiveInvocation(
                request=ModifyCart(
                    operation="add",
                    item=CartItemQuery(query=product.name),
                    quantity=1,
                ),
                opened_turn_id=turn_id,
            ),
        },
        _CFG,
    )
    pending = graph.get_state(_CFG).values["pending_cart_mutation"]
    monkeypatch.setattr(
        cart_flow.time,
        "time",
        lambda: pending.created_at + _POLICY.pending_ttl_seconds + 1,
    )

    out = await graph.ainvoke(Command(resume={"text": "yes"}), _CFG)

    assert _ai_texts(out) == ["That confirmation expired, so I haven't changed your cart."]
    assert cart.is_empty()
    assert graph.get_state(_CFG).values.get("pending_cart_mutation") is None


async def test_barged_cart_mutation_readback_reconfirms_before_effect(
    config_root: Path,
) -> None:
    graph, _store, cart = _build(config_root)
    product = load_orders_fixture(config_root, "acme_store").products[0]
    turn_id = "typed-cart-barged"
    await graph.ainvoke(
        {
            "messages": [HumanMessage("add it", id=turn_id)],
            "consumed_turn_ids": (turn_id,),
            "active_invocation": ActiveInvocation(
                request=ModifyCart(
                    operation="add",
                    item=CartItemQuery(query=product.name),
                    quantity=1,
                ),
                opened_turn_id=turn_id,
            ),
        },
        _CFG,
    )

    await graph.ainvoke(
        Command(resume={"text": "yes", "readback_interrupted": True}),
        _CFG,
    )

    retry = graph.get_state(_CFG).interrupts
    assert len(retry) == 1
    assert "Please say yes or no." in retry[0].value
    assert cart.is_empty()

    await graph.ainvoke(Command(resume={"text": "yes"}), _CFG)
    assert cart.view()[0].quantity == 1


async def test_wrong_negative_family_nomination_cannot_mutate_without_consent(
    config_root: Path,
) -> None:
    graph, _store, cart = _build(config_root)
    product = load_orders_fixture(config_root, "acme_store").products[0]
    turn_id = "typed-cart-wrong-negative-nomination"

    await graph.ainvoke(
        {
            "messages": [HumanMessage("Do not add anything to my cart", id=turn_id)],
            "consumed_turn_ids": (turn_id,),
            "active_invocation": ActiveInvocation(
                request=ModifyCart(
                    operation="add",
                    item=CartItemQuery(query=product.name),
                    quantity=1,
                ),
                opened_turn_id=turn_id,
            ),
        },
        _CFG,
    )

    assert cart.is_empty()
    assert graph.get_state(_CFG).interrupts
    assert isinstance(
        graph.get_state(_CFG).values.get("pending_cart_mutation"),
        PendingCartMutation,
    )


# --- abort keeps the cart (D6) -----------------------------------------------------------


# --- stickiness + escapes + cross-switch -------------------------------------------------


# --- view_cart read answered, not escalated ----------------------------------------------


# --- A10a structure ----------------------------------------------------------------------


async def test_cart_mutation_effect_has_one_ack_route_and_no_back_edge(
    config_root: Path,
) -> None:
    graph, _, _ = _build(config_root)

    assert {
        target for source, target in graph.builder.edges if source == "cart_mutation_apply"
    } == {"cart_ack"}
    assert "cart_mutation_apply" not in graph.builder.branches
    assert not any(source == "cart_mutation_confirm" for source, _target in graph.builder.edges)
    mutation_targets = {
        edge.target
        for edge in graph.get_graph().edges
        if edge.source in {"cart_mutation_confirm", "cart_mutation_apply"}
    }
    assert mutation_targets == {"handover", "cart_mutation_apply", "cart_ack", END}
    assert mutation_targets.isdisjoint({"cart_capability_entry", "cart_mutation_confirm"})


async def _begin_typed_placement(graph, *, turn_id: str = "typed-place") -> dict:
    return await graph.ainvoke(
        {
            "messages": [HumanMessage("place my cart", id=turn_id)],
            "consumed_turn_ids": (turn_id,),
            "active_invocation": ActiveInvocation(
                request=PlaceOrder(),
                opened_turn_id=turn_id,
            ),
        },
        _CFG,
    )


async def test_typed_place_order_reads_back_then_places_the_whole_cart(
    config_root: Path,
) -> None:
    cart = CartStore()
    cart.add_item(
        sku="SKU-BLU-07",
        name="waterproof rain jacket",
        price_usd=129.0,
        quantity=2,
    )
    cart.add_item(
        sku="SKU-GRN-15",
        name="merino hiking socks",
        price_usd=14.5,
        quantity=1,
    )
    graph, store, _ = _build(config_root, cart=cart)

    await _begin_typed_placement(graph)

    paused = graph.get_state(_CFG)
    assert paused.interrupts
    prompt = str(paused.interrupts[0].value)
    assert "2 waterproof rain jackets and 1 merino hiking socks" in prompt
    assert "$272.50" in prompt
    assert store.placed_count == 0

    out = await graph.ainvoke(Command(resume={"text": "yes"}), _CFG)

    assert store.placed_count == 1
    assert cart.is_empty()
    assert any("ORD-9001" in text for text in _ai_texts(out))


async def test_typed_place_order_decline_preserves_the_cart(config_root: Path) -> None:
    cart = CartStore()
    cart.add_item(
        sku="SKU-BLU-07",
        name="waterproof rain jacket",
        price_usd=129.0,
        quantity=1,
    )
    graph, store, _ = _build(config_root, cart=cart)
    await _begin_typed_placement(graph, turn_id="typed-place-decline")

    out = await graph.ainvoke(Command(resume={"text": "no"}), _CFG)

    assert store.placed_count == 0
    assert cart.line_count == 1
    assert any("your cart's still saved" in text for text in _ai_texts(out))


async def test_typed_place_order_rejects_an_empty_cart(config_root: Path) -> None:
    graph, store, cart = _build(config_root)

    out = await _begin_typed_placement(graph, turn_id="typed-place-empty")

    assert not graph.get_state(_CFG).interrupts
    assert store.placed_count == 0
    assert cart.is_empty()
    assert _ai_texts(out) == ["Your cart's empty - what would you like to add?"]


async def test_typed_place_order_enforces_the_value_cap_before_confirmation(
    config_root: Path,
) -> None:
    cart = CartStore()
    cart.add_item(
        sku="SKU-BLU-07",
        name="waterproof rain jacket",
        price_usd=129.0,
        quantity=100,
    )
    graph, store, _ = _build(config_root, cart=cart)

    out = await _begin_typed_placement(graph, turn_id="typed-place-cap")

    assert not graph.get_state(_CFG).interrupts
    assert store.placed_count == 0
    assert any("more than I'm able to place" in text for text in _ai_texts(out))


async def test_typed_place_order_records_recent_order_context(config_root: Path) -> None:
    recent_orders = RecentOrderContext(max_refs=_POLICY.cancel_batch_max)
    cart = CartStore()
    cart.add_item(
        sku="SKU-BLU-07",
        name="waterproof rain jacket",
        price_usd=129.0,
        quantity=1,
    )
    graph, _, _ = _build(config_root, cart=cart, recent_orders=recent_orders)
    await _begin_typed_placement(graph, turn_id="typed-place-context")
    assert recent_orders.snapshot().focused_order_ref is None

    await graph.ainvoke(Command(resume={"text": "yes"}), _CFG)

    assert recent_orders.snapshot().focused_order_ref == "ORD-9001"
