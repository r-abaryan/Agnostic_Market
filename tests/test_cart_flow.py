"""Cart flow at the GRAPH level (Group B): mutation acks, the whole-cart placement tail,
direct-buy, abort-keeps-cart, stickiness/escapes/cross-switch, view_cart, A10a. Zero network
(fake models + InMemorySaver). Folds the old test_checkout_flow.py — the single-line checkout
flow is now the cart flow's placement tail.

Entry: 'checkout now' trips the GATE (cart_write -> the cart flow) with no frontline fake
needed; the flow's reasoning fake then decides what happens inside (mutate / place / clarify).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command
from llm_fakes import FakeChatModel

from agnostic_market.agents._copy import all_closes
from agnostic_market.agents.engine import ReasoningEngine, build_checkpointer
from agnostic_market.agents.frontline import build_frontline_graph
from agnostic_market.agents.tooling import wrap_readonly_tool
from agnostic_market.commerce.cart import CartStore
from agnostic_market.commerce.orders import (
    LastOrderPointer,
    OrderStore,
    load_orders_fixture,
    speak_quantity,
)
from agnostic_market.dtos.events import InterruptEvent, TurnFacts
from agnostic_market.dtos.state import PolicyContext
from agnostic_market.voice.tools import build_voice_tools

_POLICY = PolicyContext(
    max_order_value_usd=500.0,
    allow_ai_merchant_handoff=True,
    refund_auto_approve_under_usd=50.0,
    refund_require_human_above_usd=200.0,
    refund_returnless_under_usd=50.0,
    return_window_days=30,
    pending_ttl_seconds=120.0,
)
_CFG = {"configurable": {"thread_id": "t1"}}


def _tool_fake(name: str, args: dict, *, limit: int = 1) -> FakeChatModel:
    return FakeChatModel(force_tool=name, canned_args={name: args}, tool_call_limit=limit)


# A frontline fake that hands over to the cart flow (cart_write) — for entering the flow on
# a message the gate does NOT catch (e.g. "buy the blue jacket"). limit high so a later turn
# on the same thread can also hand over if the test drives multiple entries.
def _handover_frontline() -> FakeChatModel:
    return FakeChatModel(
        force_tool="request_handover", tool_call_limit=99,
        canned_args={"request_handover": {"destination": "checkout", "reason_code": "cart_write"}})


def _build(
    config_root: Path,
    *,
    frontline: FakeChatModel | None = None,
    reasoning: FakeChatModel | None = None,
    cart: CartStore | None = None,
    pointer: LastOrderPointer | None = None,
):
    store = OrderStore(load_orders_fixture(config_root, "acme_store"))
    cart = cart or CartStore()
    pointer = pointer or LastOrderPointer()
    tools = [
        wrap_readonly_tool(t, "acme_store") for t in build_voice_tools(store, cart, pointer)
    ]
    graph = build_frontline_graph(
        frontline or FakeChatModel(emit_tool_calls=False),
        tools,
        display_name="Acme Store",
        tenant_id="acme_store",
        reasoning_model=reasoning or FakeChatModel(emit_tool_calls=False),
        store=store,
        cart_store=cart,  # the SAME cart the view_cart tool reads (no split-brain)
        pointer=pointer,  # the SAME pointer order_status sets (Group C L4)
        policy=_POLICY,
        checkpointer=build_checkpointer(),
    )
    return graph, store, cart


def _flow(graph) -> str | None:
    return graph.get_state(_CFG).values.get("active_flow")


def _ai_texts(out) -> list[str]:
    return [str(m.content) for m in out["messages"] if isinstance(m, AIMessage) and m.content]


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


async def test_batch_adds_apply_all_with_one_ack(config_root: Path) -> None:
    # Live call #9 P3: "one from each" — the model emitted N add_to_cart calls and N-1 were
    # silently dropped. A mutation-led response must apply EVERY mutation, one combined ack.
    reasoning = FakeChatModel(scripted_calls=[[
        ("add_to_cart", {"candidate_key": "1", "quantity": 1}),
        ("add_to_cart", {"candidate_key": "2", "quantity": 1}),
        ("add_to_cart", {"candidate_key": "3", "quantity": 1}),
    ]])
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
    reasoning = FakeChatModel(scripted_calls=[[
        ("add_to_cart", {"candidate_key": "2", "quantity": 1}),
        ("remove_from_cart", {"candidate_key": "3"}),
    ]])
    graph, _, _ = _build(config_root, reasoning=reasoning, cart=cart)
    out = await graph.ainvoke({"messages": [HumanMessage("checkout now please")]}, _CFG)
    assert cart.line_count == 1  # socks removed, jacket added
    ack = next(t for t in _ai_texts(out) if "Added" in t)
    assert "waterproof rain jacket" in ack
    assert "removed the merino hiking socks" in ack


async def test_control_call_behind_mutations_is_answered_not_acted(config_root: Path) -> None:
    # go_to_checkout riding BEHIND adds: the mutations apply; the control call gets a
    # tool_result (F-4) but does NOT act — control intent must lead its own turn.
    reasoning = FakeChatModel(scripted_calls=[[
        ("add_to_cart", {"candidate_key": "1", "quantity": 1}),
        ("go_to_checkout", {}),
    ]])
    graph, store, cart = _build(config_root, reasoning=reasoning)
    await graph.ainvoke({"messages": [HumanMessage("checkout now please")]}, _CFG)
    state = graph.get_state(_CFG)
    assert cart.line_count == 1
    assert store.placed_count == 0
    assert state.values.get("pending_placement") is None
    assert not state.interrupts
    tool_use_ids = {
        c["id"] for m in state.values["messages"] if isinstance(m, AIMessage) for c in m.tool_calls
    }
    tool_result_ids = {
        m.tool_call_id for m in state.values["messages"] if isinstance(m, ToolMessage)
    }
    assert tool_use_ids <= tool_result_ids  # nothing dangling in the persisted thread


async def test_batch_with_invalid_key_applies_valid_and_says_so(config_root: Path) -> None:
    reasoning = FakeChatModel(scripted_calls=[[
        ("add_to_cart", {"candidate_key": "1", "quantity": 1}),
        ("add_to_cart", {"candidate_key": "99", "quantity": 1}),  # no such option
    ]])
    graph, _, cart = _build(config_root, reasoning=reasoning)
    out = await graph.ainvoke({"messages": [HumanMessage("checkout now please")]}, _CFG)
    assert cart.line_count == 1  # the valid add landed
    ack = next(t for t in _ai_texts(out) if t.startswith("Added"))
    assert "didn't go through" in ack  # the failure is SPOKEN, never silent


async def test_all_invalid_batch_gets_one_corrective_retry(config_root: Path) -> None:
    reasoning = FakeChatModel(scripted_calls=[
        [("add_to_cart", {"candidate_key": "99", "quantity": 1})],  # all invalid -> re-prompt
        [("add_to_cart", {"candidate_key": "1", "quantity": 1})],  # corrected on retry
    ])
    graph, _, cart = _build(config_root, reasoning=reasoning)
    await graph.ainvoke({"messages": [HumanMessage("checkout now please")]}, _CFG)
    assert cart.line_count == 1


async def test_empty_cart_checkout_stays_and_asks(config_root: Path) -> None:
    reasoning = _tool_fake("go_to_checkout", {})
    graph, store, _ = _build(config_root, reasoning=reasoning)
    out = await graph.ainvoke({"messages": [HumanMessage("checkout now please")]}, _CFG)
    assert store.placed_count == 0
    assert _flow(graph) == "cart"  # stayed, no placement
    assert any("empty" in t.lower() for t in _ai_texts(out))


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
    assert out["handover"].destination == "human"


# --- view_cart read answered, not escalated ----------------------------------------------


async def test_view_cart_read_is_answered_not_escalated(config_root: Path) -> None:
    frontline = _tool_fake("view_cart", {})
    cart = CartStore()
    cart.add_item(sku="SKU-BLU-07", name="waterproof rain jacket", price_usd=129.0, quantity=2)
    graph, _, _ = _build(config_root, frontline=frontline, cart=cart)
    out = await graph.ainvoke({"messages": [HumanMessage("what's in my cart")]}, _CFG)
    assert _flow(graph) is None  # never entered a flow
    assert out.get("handover") is None
    assert any(isinstance(m, ToolMessage) and "rain jacket" in str(m.content)
               for m in out["messages"])


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
        config_root, reasoning=_tool_fake("buy_now", {"candidate_key": "2", "quantity": 2}))
    engine = ReasoningEngine(graph, thread_id="t1")
    [_ async for _ in engine.stream_turn("buy it now", TurnFacts())]
    events = [e async for e in engine.stream_turn("yes", TurnFacts(readback_interrupted=True))]
    assert store.placed_count == 0
    assert any(isinstance(e, InterruptEvent) and "yes or no" in e.prompt.lower() for e in events)
    [_ async for _ in engine.stream_turn("yes", TurnFacts())]
    assert store.placed_count == 1


async def test_double_tool_call_response_is_fully_acked(config_root: Path) -> None:
    reasoning = FakeChatModel(
        force_tool="buy_now", tool_call_limit=1, double_tool_calls=True,
        canned_args={"buy_now": {"candidate_key": "1", "quantity": 1}})
    graph, _, _ = _build(config_root, reasoning=reasoning)
    await graph.ainvoke({"messages": [HumanMessage("buy it now")]}, _CFG)
    state = graph.get_state(_CFG)
    assert state.interrupts  # flow still proceeded to the readback
    tool_use_ids = {
        c["id"] for m in state.values["messages"] if isinstance(m, AIMessage) for c in m.tool_calls
    }
    tool_result_ids = {
        m.tool_call_id for m in state.values["messages"] if isinstance(m, ToolMessage)
    }
    assert tool_use_ids <= tool_result_ids  # nothing dangling in the persisted thread


async def test_speakable_nodes_and_read_only_tools(config_root: Path) -> None:
    graph, _, _ = _build(config_root)
    assert {"handover", "cart_ack", "cart_guardrail", "cart_confirm", "cart_place",
            "cart_abort"} <= graph.speakable_nodes
    assert graph.frontline_read_only_tools == {"order_status", "catalog_search", "view_cart"}


async def test_unhandled_tool_fails_loud_not_silent_misroute(config_root: Path) -> None:
    # The terminal guard: a tool name with no branch (here: a future tool the fake forces)
    # must RAISE, not silently fall into the last branch (remove_from_cart) — a silent
    # wrong-mutation is the worst debug class. force_tool bypasses bind_tools, so this
    # simulates an 8th tool bound without a handler.
    reasoning = _tool_fake("some_new_unhandled_tool", {"candidate_key": "1"})
    graph, _, _ = _build(config_root, reasoning=reasoning)
    with pytest.raises(ValueError, match="unhandled tool"):
        await graph.ainvoke({"messages": [HumanMessage("checkout now please")]}, _CFG)


async def test_place_sets_the_last_order_pointer(config_root: Path) -> None:
    # Group C L4: a just-placed order becomes "the order most recently discussed".
    pointer = LastOrderPointer()
    reasoning = _tool_fake("buy_now", {"candidate_key": "2", "quantity": 1})
    graph, _, _ = _build(config_root, reasoning=reasoning, pointer=pointer)
    await graph.ainvoke({"messages": [HumanMessage("buy it now")]}, _CFG)
    assert pointer.get() is None  # nothing placed yet (paused at the readback)
    await graph.ainvoke(Command(resume={"text": "yes"}), _CFG)
    assert pointer.get() == "ORD-9001"
