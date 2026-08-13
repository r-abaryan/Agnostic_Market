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
from langgraph.types import Command
from llm_fakes import (
    TEST_CALLER_AUDIBLE_MODEL_TEXT_MAX_CHARS,
    TEST_STRUCTURED_OUTPUT_METHOD,
    FakeChatModel,
)
from policy_helpers import make_policy
from pydantic import ValidationError
from turn_helpers import (
    TEST_CANCELLATION_QUIESCENCE_TIMEOUT_SECONDS,
    engine_events,
    next_committed_turn,
)

from agnostic_market.agents._copy import all_closes
from agnostic_market.agents.cart import flow as cart_flow
from agnostic_market.agents.engine import ReasoningEngine, build_checkpointer
from agnostic_market.agents.frontline import build_frontline_graph
from agnostic_market.agents.tooling import wrap_readonly_tool
from agnostic_market.commerce.cart import CartStore
from agnostic_market.commerce.identity import (
    CallerIdentityStore,
    CustomerDirectory,
    load_customers_fixture,
)
from agnostic_market.commerce.orders import (
    Candidate,
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
from agnostic_market.commerce.verification import OtpProvider, VerificationStore
from agnostic_market.dtos.events import InterruptEvent, SpokenMessageEvent, TokenEvent, TurnFacts
from agnostic_market.dtos.state import SupportClarification
from agnostic_market.voice.context import CallerContext
from agnostic_market.voice.tools import build_voice_tools

_POLICY = make_policy(refund_returnless_under_usd=50.0)
_CFG = {"configurable": {"thread_id": "t1"}}
_TEST_OTP = "482913"
_CART_CLARIFICATION_LINES = {
    "action": "What would you like to do with your cart?",
    "item": "Which item would you like?",
    "quantity": "How many would you like?",
}


def _tool_fake(name: str, args: dict, *, limit: int = 1) -> FakeChatModel:
    return FakeChatModel(force_tool=name, canned_args={name: args}, tool_call_limit=limit)


# A frontline fake that hands over to the cart flow (cart_write) — for entering the flow on
# a message the gate does NOT catch (e.g. "buy the blue jacket"). limit high so a later turn
# on the same thread can also hand over if the test drives multiple entries.
def _handover_frontline() -> FakeChatModel:
    return FakeChatModel(
        force_tool="request_handover",
        tool_call_limit=99,
        canned_args={"request_handover": {"destination": "checkout", "reason_code": "cart_write"}},
    )


def _build(
    config_root: Path,
    *,
    frontline: FakeChatModel | None = None,
    reasoning: FakeChatModel | None = None,
    cart: CartStore | None = None,
    recent_orders: RecentOrderContext | None = None,
):
    store = OrderStore(load_orders_fixture(config_root, "acme_store"))
    cart = cart or CartStore()
    recent_orders = recent_orders or RecentOrderContext(max_refs=_POLICY.cancel_batch_max)
    identity = CallerIdentityStore()
    customers = CustomerDirectory(load_customers_fixture(config_root, "acme_store"))
    tools = [
        wrap_readonly_tool(t, "acme_store")
        for t in build_voice_tools(store, cart, recent_orders, identity, customers)
    ]
    otp = OtpProvider(valid_code=_TEST_OTP)
    verification = VerificationStore(otp)
    caller_context = CallerContext(
        verification_store=verification,
        cart_store=cart,
        recent_orders=recent_orders,
        identity_store=identity,
        order_store=store,
    )
    assembly = build_frontline_graph(
        frontline or FakeChatModel(emit_tool_calls=False),
        tools,
        display_name="Acme Store",
        tenant_id="acme_store",
        reasoning_model=reasoning or FakeChatModel(emit_tool_calls=False),
        store=store,
        cart_store=cart,  # the SAME cart the view_cart tool reads (no split-brain)
        otp=otp,
        verification_store=verification,
        recent_orders=recent_orders,
        identity_store=identity,  # the SAME store the order_status gate grants into (P7)
        customers=customers,
        payment_instruments=PaymentInstrumentDirectory(
            load_payment_instruments_fixture(config_root, "acme_store")
        ),
        profile_store=ProfileStore(load_profile_fixture(config_root, "acme_store")),
        policy=_POLICY,
        lifecycle=caller_context,
        structured_output_method=TEST_STRUCTURED_OUTPUT_METHOD,
        caller_audible_model_text_max_chars=TEST_CALLER_AUDIBLE_MODEL_TEXT_MAX_CHARS,
        checkpointer=build_checkpointer(),
    )
    return assembly.graph, store, cart


def _flow(graph) -> str | None:
    return graph.get_state(_CFG).values.get("active_flow")


def _ai_texts(out) -> list[str]:
    return [str(m.content) for m in out["messages"] if isinstance(m, AIMessage) and m.content]


async def _events(engine: ReasoningEngine, text: str) -> list:
    return await engine_events(engine, text)


def _reasoning_engine(graph) -> ReasoningEngine:
    return ReasoningEngine(
        graph,
        thread_id="t1",
        cancellation_quiescence_timeout_seconds=(TEST_CANCELLATION_QUIESCENCE_TIMEOUT_SECONDS),
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
    assert state.get("active_flow") == "cart"
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


@pytest.mark.parametrize("detail", ("action", "item", "quantity"))
async def test_cart_clarification_tool_selects_one_code_authored_line(
    config_root: Path, detail: str
) -> None:
    reasoning = FakeChatModel(
        force_tool="request_cart_clarification",
        canned_args={"request_cart_clarification": {"detail": detail}},
        tool_call_limit=1,
    )
    graph, store, cart = _build(config_root, reasoning=reasoning)
    events = await _events(_reasoning_engine(graph), "checkout now please")
    _assert_cart_clarification(graph, store, cart, events, _CART_CLARIFICATION_LINES[detail])


def test_cart_clarification_tool_schema_is_closed(config_root: Path) -> None:
    reasoning = FakeChatModel(emit_tool_calls=False)
    _build(config_root, reasoning=reasoning)
    detail_schema = reasoning.bound_tools["request_cart_clarification"]["function"]["parameters"][
        "properties"
    ]["detail"]
    assert set(detail_schema["enum"]) == set(_CART_CLARIFICATION_LINES)


def test_cart_prompt_routes_explicit_browsing_through_the_existing_leave_seam() -> None:
    from agnostic_market.agents.cart.prompt import compose_cart_prompt
    from agnostic_market.commerce.orders import Candidate

    prompt = compose_cart_prompt(
        "Acme Store",
        [Candidate(key="1", sku="SKU-1", name="trail running shoes", price_usd=89.99)],
        CartStore(),
        _POLICY,
    )

    assert "mentions a product but gives no clear cart action" in prompt
    assert "only browsing, comparing products, or asking what is available" in prompt
    assert "call leave_cart" in prompt


def test_cart_router_rejects_a_cross_flow_clarification(config_root: Path) -> None:
    graph, _, _ = _build(config_root)
    with pytest.raises(TypeError, match="non-cart clarification"):
        graph.update_state(
            _CFG,
            {
                "active_flow": "cart",
                "pending_clarification": SupportClarification(detail="order"),
            },
            as_node="cart_assemble",
        )


async def test_cart_no_tool_prose_falls_back_to_code_authored_action_question(
    config_root: Path,
) -> None:
    raw_model_text = "I added the item and placed your order."
    reasoning = FakeChatModel(emit_tool_calls=False, text_response=raw_model_text)
    graph, store, cart = _build(config_root, reasoning=reasoning)
    events = await _events(_reasoning_engine(graph), "checkout now please")
    _assert_cart_clarification(graph, store, cart, events, _CART_CLARIFICATION_LINES["action"])
    assert reasoning.emitted_messages[0].content == raw_model_text
    state = graph.get_state(_CFG).values
    assert not any(
        isinstance(message, AIMessage) and message.content == raw_model_text
        for message in state["messages"]
    )


async def test_repeated_cart_clarification_returns_to_frontline_without_mutation(
    config_root: Path, tmp_path: Path
) -> None:
    frontline = FakeChatModel(emit_tool_calls=False, text_response="Frontline fallback.")
    graph, store, cart = _build(
        config_root,
        frontline=frontline,
        reasoning=FakeChatModel(emit_tool_calls=False),
    )
    engine = _reasoning_engine(graph)

    first = await _events(engine, "checkout now please")
    second = await _events(engine, "What do you have?")
    third = await _events(engine, "What products are available?")
    exhausted = await _events(engine, "Can you show me the catalog?")

    assert [event.node for event in first if isinstance(event, SpokenMessageEvent)] == [
        "cart_clarify"
    ]
    assert [event.node for event in second if isinstance(event, SpokenMessageEvent)] == [
        "cart_clarify"
    ]
    assert [event.node for event in third if isinstance(event, SpokenMessageEvent)] == [
        "cart_clarify"
    ]
    assert [event.text for event in exhausted if isinstance(event, TokenEvent)] == [
        "Frontline fallback."
    ]
    state = graph.get_state(_CFG).values
    assert state.get("active_flow") == "left_cart"
    assert state.get("pending_clarification") is None
    assert state.get("clarification_progress") is None
    assert cart.is_empty()
    assert store.placed_count == 0
    assert store.cancel_count == store.refund_count == store.return_count == 0
    exhausted_events = [
        line
        for line in (tmp_path / "telemetry.jsonl").read_text(encoding="utf-8").splitlines()
        if '"event": "clarification_exhausted"' in line
    ]
    assert exhausted_events == [
        '{"event": "clarification_exhausted", "flow": "cart", "consumed_reasks": 2, "limit": 2}'
    ]


async def test_valid_cart_mutation_clears_prior_clarification_progress(
    config_root: Path,
) -> None:
    reasoning = FakeChatModel(
        scripted_calls=[
            [("request_cart_clarification", {"detail": "item"})],
            [("add_to_cart", {"candidate_key": "1", "quantity": 1})],
        ]
    )
    graph, _, cart = _build(config_root, reasoning=reasoning)
    engine = _reasoning_engine(graph)

    await _events(engine, "checkout now please")
    before = graph.get_state(_CFG).values
    assert before["clarification_progress"].flow == "cart"
    events = await _events(engine, "Add the first item.")

    assert any(
        isinstance(event, SpokenMessageEvent) and event.node == "cart_ack" for event in events
    )
    after = graph.get_state(_CFG).values
    assert after.get("clarification_progress") is None
    assert cart.line_count == 1


async def test_two_malformed_cart_clarifications_are_paired_and_stay_sticky(
    config_root: Path,
) -> None:
    malformed = [
        (
            "request_cart_clarification",
            {"detail": "item", "unexpected": "not allowed"},
        )
    ]
    reasoning = FakeChatModel(scripted_calls=[malformed, malformed])
    graph, store, cart = _build(config_root, reasoning=reasoning)
    events = await _events(_reasoning_engine(graph), "checkout now please")
    _assert_cart_clarification(graph, store, cart, events, _CART_CLARIFICATION_LINES["action"])
    assert reasoning.invoke_count == 2
    state = graph.get_state(_CFG).values
    _assert_tool_calls_paired(state)
    assert state["clarification_progress"].reasks == 0


@pytest.mark.parametrize(
    ("proposal_model", "arguments"),
    (
        (
            cart_flow._ProposeItem,
            {"candidate_key": "1", "quantity": 1, "unexpected": True},
        ),
        (cart_flow._ProposeKey, {"candidate_key": "1", "unexpected": True}),
        (
            cart_flow._RequestCartClarification,
            {"detail": "item", "unexpected": True},
        ),
    ),
)
def test_cart_proposal_models_reject_undeclared_fields(
    proposal_model: type, arguments: dict
) -> None:
    with pytest.raises(ValidationError):
        proposal_model.model_validate(arguments)


@pytest.mark.parametrize("proposal_model", (cart_flow._ProposeItem, cart_flow._ProposeQuantity))
@pytest.mark.parametrize("quantity", (True, False, "2"))
def test_cart_proposal_quantities_are_strict_integers(
    proposal_model: type,
    quantity: object,
) -> None:
    payload = {"quantity": quantity}
    if proposal_model is cart_flow._ProposeItem:
        payload["candidate_key"] = "1"
    with pytest.raises(ValidationError, match="quantity"):
        proposal_model.model_validate(payload)


@pytest.mark.parametrize("tool_name", ("add_to_cart", "buy_now"))
@pytest.mark.parametrize(
    "arguments",
    (
        {"candidate_key": "1", "quantity": 1, "unexpected": "not allowed"},
        {"candidate_key": "1", "quantity": True},
    ),
)
async def test_malformed_item_proposal_cannot_reach_cart_or_placement(
    config_root: Path,
    tool_name: str,
    arguments: dict[str, object],
) -> None:
    malformed = [(tool_name, arguments)]
    reasoning = FakeChatModel(scripted_calls=[malformed, malformed])
    graph, store, cart = _build(config_root, reasoning=reasoning)
    events = await _events(_reasoning_engine(graph), "checkout now please")
    _assert_cart_clarification(graph, store, cart, events, _CART_CLARIFICATION_LINES["action"])
    _assert_tool_calls_paired(graph.get_state(_CFG).values)


async def test_clarification_leading_a_mutation_response_does_not_mutate(
    config_root: Path,
) -> None:
    reasoning = FakeChatModel(
        scripted_calls=[
            [
                ("request_cart_clarification", {"detail": "item"}),
                ("add_to_cart", {"candidate_key": "1", "quantity": 1}),
            ]
        ]
    )
    graph, store, cart = _build(config_root, reasoning=reasoning)
    events = await _events(_reasoning_engine(graph), "checkout now please")
    _assert_cart_clarification(graph, store, cart, events, _CART_CLARIFICATION_LINES["item"])
    _assert_tool_calls_paired(graph.get_state(_CFG).values)


async def test_mutation_leading_a_clarification_response_keeps_batch_contract(
    config_root: Path,
) -> None:
    reasoning = FakeChatModel(
        scripted_calls=[
            [
                ("add_to_cart", {"candidate_key": "1", "quantity": 1}),
                ("request_cart_clarification", {"detail": "quantity"}),
            ]
        ]
    )
    graph, store, cart = _build(config_root, reasoning=reasoning)
    events = await _events(_reasoning_engine(graph), "checkout now please")
    assert cart.line_count == 1
    assert store.placed_count == 0
    assert not any(
        isinstance(event, SpokenMessageEvent) and event.node == "cart_clarify" for event in events
    )
    assert not any(isinstance(event, InterruptEvent) for event in events)
    _assert_tool_calls_paired(graph.get_state(_CFG).values)


# --- mutation ack ------------------------------------------------------------------------


async def test_add_to_cart_acks_and_stays_sticky(config_root: Path) -> None:
    reasoning = _tool_fake("add_to_cart", {"candidate_key": "1", "quantity": 2})
    graph, _, cart = _build(config_root, reasoning=reasoning)
    out = await graph.ainvoke({"messages": [HumanMessage("checkout now please")]}, _CFG)
    assert _flow(graph) == "cart"  # sticky
    assert graph.get_state(_CFG).values.get("pending_placement") is None
    assert cart.line_count == 1
    acks = [t for t in _ai_texts(out) if "Added" in t]
    assert len(acks) == 1
    # Ends with one of the rotating warm closes (lowercased mid-sentence after " - ").
    assert any(acks[-1].lower().endswith(c.lower()) for c in all_closes())


async def test_repeat_add_increments(config_root: Path) -> None:
    reasoning = _tool_fake("add_to_cart", {"candidate_key": "1", "quantity": 1}, limit=99)
    graph, _, cart = _build(config_root, reasoning=reasoning)
    await graph.ainvoke({"messages": [HumanMessage("checkout now please")]}, _CFG)
    await graph.ainvoke({"messages": [HumanMessage("one more please")]}, _CFG)
    assert cart.line_count == 1
    assert cart.view()[0].quantity == 2


def test_resolved_add_rejects_cart_line_and_uses_current_catalog_price(
    config_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    product = load_orders_fixture(config_root, "acme_store").products[0]
    stale_price = round(product.price_usd / 2, 2)
    assert stale_price != product.price_usd
    cart = CartStore()
    stale_line = cart.add_item(
        sku=product.sku,
        name=product.name,
        price_usd=stale_price,
        quantity=1,
    )
    records: list[dict[str, object]] = []
    monkeypatch.setattr(cart_flow, "write_event", records.append)

    with pytest.raises(TypeError, match="catalog candidate"):
        cart_flow._apply_resolved_mutation(cart, "add", stale_line, 1)

    assert cart.view() == (stale_line,)
    assert records == []

    outcome = cart_flow._apply_resolved_mutation(
        cart,
        "add",
        Candidate(
            key="1",
            sku=product.sku,
            name=product.name,
            price_usd=product.price_usd,
        ),
        1,
    )

    assert outcome.line.quantity == 2
    assert outcome.line.price_usd == product.price_usd
    assert records == [{"event": "cart_item_added", "sku": product.sku}]


async def test_set_quantity_zero_removes_the_resolved_cart_line(config_root: Path) -> None:
    cart = CartStore()
    cart.add_item(
        sku="SKU-RED-42",
        name="trail running shoes",
        price_usd=89.99,
        quantity=2,
    )
    reasoning = _tool_fake("set_quantity", {"candidate_key": "1", "quantity": 0})
    graph, _, _ = _build(config_root, reasoning=reasoning, cart=cart)

    out = await graph.ainvoke({"messages": [HumanMessage("checkout now please")]}, _CFG)

    assert cart.is_empty()
    assert any("Removed the trail running shoes from your cart" in text for text in _ai_texts(out))


@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    (
        ("remove_from_cart", {"candidate_key": "2"}),
        ("set_quantity", {"candidate_key": "2", "quantity": 5}),
        ("set_quantity", {"candidate_key": "2", "quantity": 0}),
    ),
)
async def test_absent_item_mutation_is_a_truthful_noop_without_mutation_event(
    config_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    tool_name: str,
    arguments: dict,
) -> None:
    cart = CartStore()
    cart.add_item(
        sku="SKU-RED-42",
        name="trail running shoes",
        price_usd=89.99,
        quantity=2,
    )
    before = cart.view()
    records: list[dict[str, object]] = []
    monkeypatch.setattr(cart_flow, "write_event", records.append)
    graph, _, _ = _build(
        config_root,
        reasoning=_tool_fake(tool_name, arguments),
        cart=cart,
    )

    out = await graph.ainvoke({"messages": [HumanMessage("checkout now please")]}, _CFG)

    assert cart.view() == before
    assert any(
        "waterproof rain jacket wasn't in your cart" in text.lower() for text in _ai_texts(out)
    )
    assert any(
        "waterproof rain jacket wasn't in your cart" in str(message.content).lower()
        for message in out["messages"]
        if isinstance(message, ToolMessage)
    )
    assert records == []


@pytest.mark.parametrize(
    ("tool_name", "arguments", "starting_quantity", "event_name"),
    (
        ("add_to_cart", {"candidate_key": "1", "quantity": 1}, None, "cart_item_added"),
        ("remove_from_cart", {"candidate_key": "1"}, 1, "cart_item_removed"),
        ("set_quantity", {"candidate_key": "1", "quantity": 3}, 1, "cart_quantity_set"),
        ("set_quantity", {"candidate_key": "1", "quantity": 0}, 1, "cart_quantity_set"),
    ),
)
async def test_each_resolved_mutation_emits_one_matching_audit_event(
    config_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    tool_name: str,
    arguments: dict,
    starting_quantity: int | None,
    event_name: str,
) -> None:
    cart = CartStore()
    if starting_quantity is not None:
        cart.add_item(
            sku="SKU-RED-42",
            name="trail running shoes",
            price_usd=89.99,
            quantity=starting_quantity,
        )
    records: list[dict[str, object]] = []
    monkeypatch.setattr(cart_flow, "write_event", records.append)
    graph, _, _ = _build(
        config_root,
        reasoning=_tool_fake(tool_name, arguments),
        cart=cart,
    )

    await graph.ainvoke({"messages": [HumanMessage("checkout now please")]}, _CFG)

    assert records == [{"event": event_name, "sku": "SKU-RED-42"}]


async def test_batch_adds_apply_all_with_one_ack(config_root: Path) -> None:
    # Live call #9 P3: "one from each" — the model emitted N add_to_cart calls and N-1 were
    # silently dropped. A mutation-led response must apply EVERY mutation, one combined ack.
    reasoning = FakeChatModel(
        scripted_calls=[
            [
                ("add_to_cart", {"candidate_key": "1", "quantity": 1}),
                ("add_to_cart", {"candidate_key": "2", "quantity": 1}),
                ("add_to_cart", {"candidate_key": "3", "quantity": 1}),
            ]
        ]
    )
    graph, _, cart = _build(config_root, reasoning=reasoning)
    out = await graph.ainvoke({"messages": [HumanMessage("checkout now please")]}, _CFG)
    assert cart.line_count == 3
    acks = [t for t in _ai_texts(out) if t.startswith("Added")]
    assert len(acks) == 1  # ONE combined ack, not three
    for name in ("trail running shoes", "waterproof rain jacket", "merino hiking socks"):
        assert name in acks[0]
    assert _flow(graph) == "cart"  # sticky, no placement minted
    assert graph.get_state(_CFG).values.get("pending_placement") is None


async def test_batch_add_and_remove_combined_ack(config_root: Path) -> None:
    cart = CartStore()
    cart.add_item(sku="SKU-GRN-15", name="merino hiking socks", price_usd=14.50, quantity=1)
    reasoning = FakeChatModel(
        scripted_calls=[
            [
                ("add_to_cart", {"candidate_key": "2", "quantity": 1}),
                ("remove_from_cart", {"candidate_key": "3"}),
            ]
        ]
    )
    graph, _, _ = _build(config_root, reasoning=reasoning, cart=cart)
    out = await graph.ainvoke({"messages": [HumanMessage("checkout now please")]}, _CFG)
    assert cart.line_count == 1  # socks removed, jacket added
    ack = next(t for t in _ai_texts(out) if "Added" in t)
    assert "waterproof rain jacket" in ack
    assert "removed the merino hiking socks" in ack


async def test_control_call_behind_mutations_is_answered_not_acted(config_root: Path) -> None:
    # go_to_checkout riding BEHIND adds: the mutations apply; the control call gets a
    # tool_result (F-4) but does NOT act — control intent must lead its own turn.
    reasoning = FakeChatModel(
        scripted_calls=[
            [
                ("add_to_cart", {"candidate_key": "1", "quantity": 1}),
                ("go_to_checkout", {}),
            ]
        ]
    )
    graph, store, cart = _build(config_root, reasoning=reasoning)
    await graph.ainvoke({"messages": [HumanMessage("checkout now please")]}, _CFG)
    state = graph.get_state(_CFG)
    assert cart.line_count == 1
    assert store.placed_count == 0
    assert state.values.get("pending_placement") is None
    assert not state.interrupts
    _assert_tool_calls_paired(state.values)


async def test_batch_with_invalid_key_applies_valid_and_says_so(config_root: Path) -> None:
    reasoning = FakeChatModel(
        scripted_calls=[
            [
                ("add_to_cart", {"candidate_key": "1", "quantity": 1}),
                ("add_to_cart", {"candidate_key": "99", "quantity": 1}),  # no such option
            ]
        ]
    )
    graph, _, cart = _build(config_root, reasoning=reasoning)
    out = await graph.ainvoke({"messages": [HumanMessage("checkout now please")]}, _CFG)
    assert cart.line_count == 1  # the valid add landed
    ack = next(t for t in _ai_texts(out) if t.startswith("Added"))
    assert "didn't go through" in ack  # the failure is SPOKEN, never silent


async def test_all_invalid_batch_gets_one_corrective_retry(config_root: Path) -> None:
    reasoning = FakeChatModel(
        scripted_calls=[
            [("add_to_cart", {"candidate_key": "99", "quantity": 1})],  # all invalid -> re-prompt
            [("add_to_cart", {"candidate_key": "1", "quantity": 1})],  # corrected on retry
        ]
    )
    graph, _, cart = _build(config_root, reasoning=reasoning)
    await graph.ainvoke({"messages": [HumanMessage("checkout now please")]}, _CFG)
    assert cart.line_count == 1


async def test_empty_cart_checkout_stays_and_asks(config_root: Path) -> None:
    reasoning = _tool_fake("go_to_checkout", {})
    graph, store, _ = _build(config_root, reasoning=reasoning)
    out = await graph.ainvoke({"messages": [HumanMessage("checkout now please")]}, _CFG)
    assert store.placed_count == 0
    assert _flow(graph) == "cart"  # stayed, no placement
    assert "Your cart's empty - what would you like to add?" in _ai_texts(out)


async def test_review_cart_lists_contents(config_root: Path) -> None:
    cart = CartStore()
    cart.add_item(sku="SKU-BLU-07", name="waterproof rain jacket", price_usd=129.0, quantity=2)
    reasoning = _tool_fake("review_cart", {})
    graph, _, _ = _build(config_root, reasoning=reasoning, cart=cart)
    out = await graph.ainvoke({"messages": [HumanMessage("checkout now please")]}, _CFG)
    assert any("2 waterproof rain jackets" in t and "$258.00" in t for t in _ai_texts(out))


# --- whole-cart placement ----------------------------------------------------------------


async def test_buy_now_places_one_order_after_readback(config_root: Path) -> None:
    # candidate_key "2" of the full catalog for "buy it now" = waterproof rain jacket $129.
    reasoning = _tool_fake("buy_now", {"candidate_key": "2", "quantity": 2})
    graph, store, cart = _build(config_root, reasoning=reasoning)
    await graph.ainvoke({"messages": [HumanMessage("buy it now")]}, _CFG)
    state = graph.get_state(_CFG)
    assert state.interrupts  # paused at the placement readback
    rb = str(state.interrupts[0].value)
    assert "2 waterproof rain jackets" in rb and "$258.00" in rb
    assert store.placed_count == 0
    out = await graph.ainvoke(Command(resume={"text": "yes"}), _CFG)
    assert store.placed_count == 1
    assert cart.is_empty()  # cleared on placement
    assert any("ORD-9001" in t for t in _ai_texts(out))


async def test_multi_line_place_reads_back_all_lines(config_root: Path) -> None:
    cart = CartStore()
    cart.add_item(sku="SKU-BLU-07", name="waterproof rain jacket", price_usd=129.0, quantity=2)
    cart.add_item(sku="SKU-GRN-15", name="merino hiking socks", price_usd=14.5, quantity=1)
    reasoning = _tool_fake("go_to_checkout", {})
    graph, store, _ = _build(config_root, reasoning=reasoning, cart=cart)
    await graph.ainvoke({"messages": [HumanMessage("checkout now please")]}, _CFG)
    rb = str(graph.get_state(_CFG).interrupts[0].value)
    assert "2 waterproof rain jackets and 1 merino hiking socks" in rb
    assert "$272.50" in rb
    await graph.ainvoke(Command(resume={"text": "yes"}), _CFG)
    assert store.placed_count == 1
    assert len(next(iter(store._placed_by_key.values())).lines) == 2


async def test_order_over_value_cap_is_denied_in_code(config_root: Path) -> None:
    reasoning = _tool_fake("buy_now", {"candidate_key": "2", "quantity": 100})  # > $500
    graph, store, _ = _build(config_root, reasoning=reasoning)
    out = await graph.ainvoke({"messages": [HumanMessage("buy it now")]}, _CFG)
    assert store.placed_count == 0
    assert not graph.get_state(_CFG).interrupts  # never reached the readback
    assert any("more than I'm able to place" in t for t in _ai_texts(out))


async def test_buy_now_with_existing_cart_places_whole_cart(config_root: Path) -> None:
    cart = CartStore()
    cart.add_item(sku="SKU-GRN-15", name="merino hiking socks", price_usd=14.5, quantity=1)
    reasoning = _tool_fake("buy_now", {"candidate_key": "2", "quantity": 1})  # rain jacket
    graph, store, _ = _build(config_root, reasoning=reasoning, cart=cart)
    await graph.ainvoke({"messages": [HumanMessage("buy it now")]}, _CFG)
    rb = str(graph.get_state(_CFG).interrupts[0].value)
    assert "socks" in rb and "jacket" in rb  # both the existing line and the bought one
    await graph.ainvoke(Command(resume={"text": "yes"}), _CFG)
    assert store.placed_count == 1


async def test_duplicate_whole_cart_reads_back_second_order(config_root: Path) -> None:
    reasoning = _tool_fake("buy_now", {"candidate_key": "1", "quantity": 2}, limit=99)
    graph, store, _ = _build(config_root, reasoning=reasoning)
    await graph.ainvoke({"messages": [HumanMessage("buy it now")]}, _CFG)
    await graph.ainvoke(Command(resume={"text": "yes"}), _CFG)
    assert store.placed_count == 1
    # buy the same set again -> the guardrail flags it, readback names a SECOND order
    await graph.ainvoke({"messages": [HumanMessage("buy it now")]}, _CFG)
    rb = str(graph.get_state(_CFG).interrupts[0].value)
    assert "SECOND" in rb and "ORD-9001" in rb


# --- abort keeps the cart (D6) -----------------------------------------------------------


async def test_abort_at_readback_keeps_the_cart(config_root: Path) -> None:
    reasoning = _tool_fake("buy_now", {"candidate_key": "1", "quantity": 2})
    graph, store, cart = _build(config_root, reasoning=reasoning)
    await graph.ainvoke({"messages": [HumanMessage("buy it now")]}, _CFG)
    assert graph.get_state(_CFG).interrupts
    out = await graph.ainvoke({"messages": [HumanMessage("actually never mind, stop")]}, _CFG)
    assert _flow(graph) is None
    assert store.placed_count == 0
    assert cart.line_count == 1  # the cart SURVIVES the abort
    assert any("still saved" in t for t in _ai_texts(out))


# --- stickiness + escapes + cross-switch -------------------------------------------------


async def test_followup_turn_stays_inside_cart(config_root: Path) -> None:
    frontline = FakeChatModel(emit_tool_calls=False)
    graph, _, _ = _build(config_root, frontline=frontline)
    await _enter_cart(graph)  # gate entry, reasoning clarifies -> sticky
    await graph.ainvoke({"messages": [HumanMessage("the rain jacket, two")]}, _CFG)
    assert _flow(graph) == "cart"
    assert frontline._tool_calls_made == 0  # entry router bypassed the frontline tier


async def test_checkout_now_while_sticky_in_cart_stays_inside(config_root: Path) -> None:
    # THE _gate_owner test: "checkout now" trips the gate to the "checkout" DESTINATION,
    # which this SAME cart flow serves -> NOT a cross-switch (stays inside).
    cart = CartStore()
    cart.add_item(sku="SKU-BLU-07", name="waterproof rain jacket", price_usd=129.0, quantity=1)
    reasoning = _tool_fake("go_to_checkout", {}, limit=99)
    graph, _, _ = _build(config_root, reasoning=reasoning, cart=cart)
    await _enter_cart(graph)  # first checkout-now enters + places-readback? no: go_to_checkout
    out = await graph.ainvoke({"messages": [HumanMessage("checkout now please")]}, _CFG)
    assert out.get("handover") is None  # did NOT cross-switch out


async def test_refund_while_sticky_in_cart_cross_switches_to_support(config_root: Path) -> None:
    graph, _, _ = _build(config_root)
    await _enter_cart(graph)
    await graph.ainvoke({"messages": [HumanMessage("actually I want a refund")]}, _CFG)
    assert _flow(graph) == "support"  # cross-switched


async def test_human_escape_breaks_stickiness(config_root: Path) -> None:
    graph, _, _ = _build(config_root)
    await _enter_cart(graph)
    out = await graph.ainvoke({"messages": [HumanMessage("get me a person please")]}, _CFG)
    assert _flow(graph) is None
    assert out.get("handover") is None
    assert out.get("automation_terminal") is True


# --- view_cart read answered, not escalated ----------------------------------------------


async def test_view_cart_read_is_answered_not_escalated(config_root: Path) -> None:
    frontline = _tool_fake("view_cart", {})
    cart = CartStore()
    cart.add_item(sku="SKU-BLU-07", name="waterproof rain jacket", price_usd=129.0, quantity=2)
    graph, _, _ = _build(config_root, frontline=frontline, cart=cart)
    out = await graph.ainvoke({"messages": [HumanMessage("what's in my cart")]}, _CFG)
    assert _flow(graph) is None  # never entered a flow
    assert out.get("handover") is None
    assert any(
        isinstance(m, ToolMessage) and "rain jacket" in str(m.content) for m in out["messages"]
    )


# --- A10a structure ----------------------------------------------------------------------


async def test_place_replay_yields_the_same_single_order(config_root: Path) -> None:
    reasoning = _tool_fake("buy_now", {"candidate_key": "1", "quantity": 2}, limit=99)
    graph, store, _ = _build(config_root, reasoning=reasoning)
    await graph.ainvoke({"messages": [HumanMessage("buy it now")]}, _CFG)
    await graph.ainvoke(Command(resume={"text": "yes"}), _CFG)
    await graph.ainvoke({"messages": [HumanMessage("yes")]}, _CFG)  # stray extra, must not re-place
    assert store.placed_count == 1


async def test_barged_readback_reconfirms_before_placing(config_root: Path) -> None:
    graph, store, _ = _build(
        config_root, reasoning=_tool_fake("buy_now", {"candidate_key": "2", "quantity": 2})
    )
    engine = _reasoning_engine(graph)
    [_ async for _ in engine.stream_turn(next_committed_turn(engine, "buy it now"), TurnFacts())]
    events = [
        e
        async for e in engine.stream_turn(
            next_committed_turn(engine, "yes"),
            TurnFacts(readback_interrupted=True),
        )
    ]
    assert store.placed_count == 0
    reconfirms = [e for e in events if isinstance(e, InterruptEvent)]
    assert len(reconfirms) == 1
    assert "yes or no" in reconfirms[0].prompt.lower()
    assert "2 waterproof rain jackets" in reconfirms[0].prompt
    assert "$258.00" in reconfirms[0].prompt
    [_ async for _ in engine.stream_turn(next_committed_turn(engine, "yes"), TurnFacts())]
    assert store.placed_count == 1


async def test_double_tool_call_response_is_fully_acked(config_root: Path) -> None:
    reasoning = FakeChatModel(
        force_tool="buy_now",
        tool_call_limit=1,
        double_tool_calls=True,
        canned_args={"buy_now": {"candidate_key": "1", "quantity": 1}},
    )
    graph, _, _ = _build(config_root, reasoning=reasoning)
    await graph.ainvoke({"messages": [HumanMessage("buy it now")]}, _CFG)
    state = graph.get_state(_CFG)
    assert state.interrupts  # flow still proceeded to the readback
    _assert_tool_calls_paired(state.values)


async def test_speakable_nodes_and_read_only_tools(config_root: Path) -> None:
    graph, _, _ = _build(config_root)
    assert {
        "handover",
        "cart_ack",
        "cart_clarify",
        "cart_guardrail",
        "cart_confirm",
        "cart_place",
        "cart_abort",
    } <= graph.speakable_nodes
    assert graph.frontline_read_only_tools == {
        "order_status",
        "list_orders",
        "catalog_search",
        "view_cart",
    }
    assert "cart_capability_entry" not in graph.speakable_nodes


@pytest.mark.parametrize(
    ("second_call", "unknown_results"),
    [
        ([("request_cart_clarification", {"detail": "action"})], 1),
        ([("catalog_lookup", {"query": "trail shoes"})], 2),
    ],
)
async def test_unknown_cart_tool_uses_bounded_correction_without_an_effect(
    config_root: Path,
    second_call: list[tuple[str, dict]],
    unknown_results: int,
) -> None:
    reasoning = FakeChatModel(
        scripted_calls=[
            [("catalog_lookup", {"query": "trail shoes"})],
            second_call,
        ]
    )
    graph, store, cart = _build(config_root, reasoning=reasoning)
    engine = _reasoning_engine(graph)

    events = await _events(engine, "checkout now please")

    state = graph.get_state(_CFG).values
    _assert_tool_calls_paired(state)
    assert reasoning.invoke_count == 2
    assert (
        sum(
            isinstance(message, ToolMessage)
            and str(message.content).startswith("Unavailable action.")
            for message in state["messages"]
        )
        == unknown_results
    )
    assert [
        (event.node, event.text) for event in events if isinstance(event, SpokenMessageEvent)
    ] == [("cart_clarify", _CART_CLARIFICATION_LINES["action"])]
    assert not any(isinstance(event, TokenEvent | InterruptEvent) for event in events)
    assert cart.is_empty()
    assert store.placed_count == 0
    assert store.cancel_count == store.refund_count == store.return_count == 0


async def test_unknown_cart_browse_tool_can_correct_to_leave_and_frontline_answer(
    config_root: Path,
) -> None:
    reasoning = FakeChatModel(
        scripted_calls=[
            [("request_cart_clarification", {"detail": "action"})],
            [("catalog_lookup", {"query": "trail shoes"})],
            [("leave_cart", {})],
        ]
    )
    frontline = FakeChatModel(
        emit_tool_calls=False,
        text_response="We have trail running shoes available.",
    )
    graph, store, cart = _build(config_root, frontline=frontline, reasoning=reasoning)
    engine = _reasoning_engine(graph)
    await _events(engine, "checkout now please")

    events = await _events(engine, "I'm only asking what shoes are available.")

    state = graph.get_state(_CFG).values
    _assert_tool_calls_paired(state)
    assert [event.text for event in events if isinstance(event, TokenEvent)] == [
        "We have trail running shoes available."
    ]
    assert state.get("active_flow") == "left_cart"
    assert state.get("clarification_progress") is None
    assert reasoning.invoke_count == 3
    assert frontline.invoke_count == 1
    assert cart.is_empty()
    assert store.placed_count == 0
    assert store.cancel_count == store.refund_count == store.return_count == 0


async def test_place_records_recent_order_context(config_root: Path) -> None:
    # Group C L4: a just-placed order becomes "the order most recently discussed".
    recent_orders = RecentOrderContext(max_refs=_POLICY.cancel_batch_max)
    reasoning = _tool_fake("buy_now", {"candidate_key": "2", "quantity": 1})
    graph, _, _ = _build(config_root, reasoning=reasoning, recent_orders=recent_orders)
    await graph.ainvoke({"messages": [HumanMessage("buy it now")]}, _CFG)
    assert recent_orders.snapshot().focused_order_ref is None
    await graph.ainvoke(Command(resume={"text": "yes"}), _CFG)
    assert recent_orders.snapshot().focused_order_ref == "ORD-9001"
