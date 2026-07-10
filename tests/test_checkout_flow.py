"""Checkout flow at the GRAPH level: entry routing, escapes, stickiness, guardrail,
SKU discipline, structural A10a shape. Zero network (fake models + InMemorySaver)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command
from llm_fakes import FakeChatModel

from agnostic_market.agents import telemetry
from agnostic_market.agents.checkout import build_checkout_nodes, speak_quantity
from agnostic_market.agents.engine import build_checkpointer
from agnostic_market.agents.frontline import build_frontline_graph
from agnostic_market.agents.tooling import wrap_readonly_tool
from agnostic_market.commerce.orders import OrderStore, load_orders_fixture
from agnostic_market.dtos.state import PendingAction, PolicyContext, ReasoningState
from agnostic_market.voice.tools import build_voice_tools

_POLICY = PolicyContext(
    max_order_value_usd=500.0,
    allow_ai_merchant_handoff=True,
    refund_auto_approve_under_usd=50.0,
    refund_require_human_above_usd=200.0,
    refund_returnless_under_usd=50.0,
    pending_ttl_seconds=120.0,
)
_CFG = {"configurable": {"thread_id": "t1"}}


def _build(
    config_root: Path,
    *,
    frontline: FakeChatModel | None = None,
    reasoning: FakeChatModel | None = None,
):
    store = OrderStore(load_orders_fixture(config_root, "acme_store"))
    tools = [wrap_readonly_tool(t, "acme_store") for t in build_voice_tools(store)]
    graph = build_frontline_graph(
        frontline or FakeChatModel(emit_tool_calls=False),
        tools,
        display_name="Acme Store",
        reasoning_model=reasoning or FakeChatModel(emit_tool_calls=False),
        store=store,
        policy=_POLICY,
        checkpointer=build_checkpointer(),
    )
    return graph, store


# --- speech-native rendering (VOICE_PIPELINE §7 — no 'x' for TTS to voice as 'ex') -------


@pytest.mark.parametrize(
    ("qty", "name", "expected"),
    [
        (1, "waterproof rain jacket", "1 waterproof rain jacket"),
        (2, "waterproof rain jacket", "2 waterproof rain jackets"),
        (3, "merino hiking sock", "3 merino hiking socks"),
    ],
)
def test_speak_quantity_pluralizes_without_x(qty: int, name: str, expected: str) -> None:
    result = speak_quantity(qty, name)
    assert result == expected
    assert " x " not in result  # the 'x' separator TTS mis-voices is gone


# --- entering the flow -------------------------------------------------------------------


async def test_model_handover_to_checkout_enters_the_flow(config_root: Path) -> None:
    # Trigger-free purchase intent: the FRONTLINE model hands over to checkout, and the
    # same turn continues INTO assemble -> confirm interrupt (no deferral spoken).
    frontline = FakeChatModel(
        force_tool="request_handover",
        tool_call_limit=1,
        canned_args={"request_handover": {"destination": "checkout", "reason_code": "cart_write"}},
    )
    reasoning = FakeChatModel(
        force_tool="propose_order",
        tool_call_limit=1,
        canned_args={"propose_order": {"candidate_key": "1", "quantity": 1}},
    )
    graph, store = _build(config_root, frontline=frontline, reasoning=reasoning)
    await graph.ainvoke(
        {"messages": [HumanMessage("I'd like to get the waterproof rain jacket")]}, _CFG
    )
    state = graph.get_state(_CFG)
    assert state.interrupts  # paused at the readback
    pending = state.values["pending_action"]
    assert pending.sku == "SKU-BLU-07"  # candidates narrowed to the jacket; key "1" = it
    assert pending.total_usd == 129.00  # fixture price, code arithmetic
    assert store.placed_count == 0


# --- guardrail (code-enforced cap) -------------------------------------------------------


async def test_order_over_value_cap_is_denied_in_code(config_root: Path) -> None:
    reasoning = FakeChatModel(
        force_tool="propose_order",
        tool_call_limit=1,
        canned_args={"propose_order": {"candidate_key": "2", "quantity": 100}},  # $12,900
    )
    graph, store = _build(config_root, reasoning=reasoning)
    out = await graph.ainvoke({"messages": [HumanMessage("checkout now please")]}, _CFG)
    assert store.placed_count == 0
    assert out.get("pending_action") is None
    assert out.get("active_flow") is None
    assert not graph.get_state(_CFG).interrupts  # never even reached the confirm gate
    assert any(
        isinstance(m, AIMessage) and "more than I'm able to place" in str(m.content)
        for m in out["messages"]
    )


# --- SKU discipline (unoffered key is structurally unreachable) ---------------------------


async def test_unoffered_candidate_key_never_places(config_root: Path) -> None:
    # The fake keeps proposing key "99" (not in the candidate list): one corrective
    # re-prompt, then assemble LEAVES the flow — the gate skips a re-trip back into the
    # flow just left (see the gate-skip test below), so the FRONTLINE model answers the
    # turn instead of a cycle or a canned deferral.
    reasoning = FakeChatModel(
        force_tool="propose_order",
        canned_args={"propose_order": {"candidate_key": "99", "quantity": 1}},
    )
    frontline = FakeChatModel(emit_tool_calls=False)
    graph, store = _build(config_root, frontline=frontline, reasoning=reasoning)
    out = await graph.ainvoke({"messages": [HumanMessage("checkout now please")]}, _CFG)
    assert store.placed_count == 0
    assert out.get("pending_action") is None
    assert not graph.get_state(_CFG).interrupts
    assert frontline._tool_calls_made == 0  # frontline RAN (answered); no checkout re-entry
    assert isinstance(out["messages"][-1], AIMessage) and out["messages"][-1].content


# --- gate skip-when-just-left (2026-07-10 live: "complete the purchase" on an already-
# ---  placed order -> model left -> gate re-tripped -> stale "I'll pass it along" line) ----


async def test_gate_does_not_retrip_into_a_flow_the_model_just_left(config_root: Path) -> None:
    # The checkout model LEAVES (it saw there is nothing to buy); the same gate-certain
    # text must not bounce straight back in or end at the canned deferral — the frontline
    # model owns the answer (it holds no write tools; safe by construction).
    reasoning = FakeChatModel(force_tool="leave_checkout", tool_call_limit=1, canned_args={})
    frontline = FakeChatModel(emit_tool_calls=False)
    graph, store = _build(config_root, frontline=frontline, reasoning=reasoning)
    out = await graph.ainvoke(
        {"messages": [HumanMessage("go ahead and complete the purchase")]}, _CFG
    )
    assert store.placed_count == 0
    texts = [str(m.content) for m in out["messages"] if isinstance(m, AIMessage)]
    assert not any("pass it along" in t for t in texts)  # the stale deferral is gone
    assert texts and texts[-1]  # the frontline model spoke the answer
    assert reasoning._tool_calls_made == 1  # checkout ran once (the leave), no re-entry


# --- duplicate-order disambiguation (2026-07-10 live: "complete the purchase" after the
# ---  order was already placed silently created a second identical $387 order) ------------


async def test_second_identical_order_readback_disambiguates(config_root: Path) -> None:
    reasoning = FakeChatModel(
        force_tool="propose_order",
        tool_call_limit=2,
        canned_args={"propose_order": {"candidate_key": "1", "quantity": 3}},
    )
    graph, store = _build(config_root, reasoning=reasoning)
    await graph.ainvoke({"messages": [HumanMessage("checkout now please")]}, _CFG)
    first = str(graph.get_state(_CFG).interrupts[0].value)
    assert "SECOND" not in first  # the first order reads back normally
    await graph.ainvoke(Command(resume={"text": "yes"}), _CFG)
    assert store.placed_count == 1  # ORD-9001 placed
    # The same proposal again this session: the readback must name the existing order and
    # make "second order" unmissable — a misread "complete the purchase" hears exactly
    # what a yes would do.
    await graph.ainvoke({"messages": [HumanMessage("checkout now please")]}, _CFG)
    second = str(graph.get_state(_CFG).interrupts[0].value)
    assert "SECOND" in second
    assert "ORD-9001" in second
    await graph.ainvoke(Command(resume={"text": "yes"}), _CFG)
    assert store.placed_count == 2  # an explicit yes to a second order is legitimate


async def test_duplicate_flag_declined_places_nothing_more(config_root: Path) -> None:
    reasoning = FakeChatModel(
        force_tool="propose_order",
        tool_call_limit=2,
        canned_args={"propose_order": {"candidate_key": "1", "quantity": 3}},
    )
    graph, store = _build(config_root, reasoning=reasoning)
    await graph.ainvoke({"messages": [HumanMessage("checkout now please")]}, _CFG)
    await graph.ainvoke(Command(resume={"text": "yes"}), _CFG)
    await graph.ainvoke({"messages": [HumanMessage("checkout now please")]}, _CFG)
    out = await graph.ainvoke(Command(resume={"text": "no, that was the same one"}), _CFG)
    assert store.placed_count == 1  # the decline kept it at one
    assert any(
        isinstance(m, AIMessage) and "won't place it" in str(m.content)
        for m in out["messages"]
    )


# --- F-4: a multi-tool-call response must not leave a dangling tool_use -------------------


async def test_double_tool_call_response_is_fully_acked(config_root: Path) -> None:
    # A misbehaving model emits TWO tool calls in one response. The assemble acts on the
    # first and must ack the second — a persisted tool_use with no tool_result fails
    # provider-side history validation on EVERY later model call in the session.
    reasoning = FakeChatModel(
        force_tool="propose_order",
        tool_call_limit=1,
        double_tool_calls=True,
        canned_args={"propose_order": {"candidate_key": "1", "quantity": 1}},
    )
    graph, store = _build(config_root, reasoning=reasoning)
    await graph.ainvoke({"messages": [HumanMessage("checkout now please")]}, _CFG)
    state = graph.get_state(_CFG)
    assert state.interrupts  # the flow still proceeded to the readback
    tool_use_ids = {
        c["id"] for m in state.values["messages"] if isinstance(m, AIMessage) for c in m.tool_calls
    }
    tool_result_ids = {
        m.tool_call_id for m in state.values["messages"] if isinstance(m, ToolMessage)
    }
    assert tool_use_ids <= tool_result_ids  # nothing dangling in the persisted thread
    assert store.placed_count == 0  # and the extra call minted no second proposal


# --- stickiness + escapes (multi-turn on one thread) --------------------------------------


async def _clarify_turn(graph) -> None:
    out = await graph.ainvoke({"messages": [HumanMessage("checkout now please")]}, _CFG)
    assert out["active_flow"] == "checkout"  # in flow, awaiting item/quantity


async def test_followup_turn_stays_inside_checkout(config_root: Path) -> None:
    frontline = FakeChatModel(emit_tool_calls=False)
    graph, _ = _build(config_root, frontline=frontline)
    await _clarify_turn(graph)
    out = await graph.ainvoke({"messages": [HumanMessage("the rain jacket, two of them")]}, _CFG)
    assert out["active_flow"] == "checkout"  # still inside (clarify fake keeps asking)
    assert out.get("handover") is None
    # The frontline tier was NEVER consulted on either turn: the entry router bypassed it.
    assert frontline._tool_calls_made == 0


async def test_abort_escape_breaks_stickiness(config_root: Path) -> None:
    graph, store = _build(config_root)
    await _clarify_turn(graph)
    out = await graph.ainvoke({"messages": [HumanMessage("actually never mind, stop")]}, _CFG)
    assert out.get("active_flow") is None
    assert store.placed_count == 0
    assert any(
        isinstance(m, AIMessage) and "Nothing has been ordered" in str(m.content)
        for m in out["messages"]
    )


async def test_human_escape_breaks_stickiness_and_hands_over(config_root: Path) -> None:
    graph, _ = _build(config_root)
    await _clarify_turn(graph)
    out = await graph.ainvoke(
        {"messages": [HumanMessage("I want to talk to a human please")]}, _CFG
    )
    assert out.get("active_flow") is None
    assert out["handover"].destination == "human"  # §A9: never trapped
    assert any(
        isinstance(m, AIMessage) and "person" in str(m.content) for m in out["messages"]
    )


# --- the cross-flow escape (live 2026-07-09 bug: sticky checkout swallowed a refund) -------


async def test_refund_while_sticky_in_checkout_cross_switches_to_support(
    config_root: Path,
) -> None:
    # THE live bug: sticky in checkout (assemble clarified, no tool call), the caller asks
    # for a refund. Pre-fix, the entry router sent this to checkout_assemble (gate never
    # consulted) and the checkout model refused/narrated. Now the gate-certain cross-flow
    # intent deterministically abandons checkout and enters support.
    graph, _ = _build(config_root)
    await _clarify_turn(graph)  # active_flow == "checkout"
    out = await graph.ainvoke(
        {"messages": [HumanMessage("Can I get refund for this purchase?")]}, _CFG
    )
    assert out["active_flow"] == "support"  # switched INTO the refund flow (clarify keeps it)
    assert out.get("pending_action") is None  # nothing from checkout lingers


async def test_cancel_order_while_sticky_in_checkout_cross_switches(config_root: Path) -> None:
    # "cancel my order" is NOT an abort phrasing (abort = cancel that/it/this) — it's a
    # placed-order cancel, which the gate routes to support's cancel path.
    graph, _ = _build(config_root)
    await _clarify_turn(graph)
    out = await graph.ainvoke({"messages": [HumanMessage("please cancel my order")]}, _CFG)
    assert out["active_flow"] == "support"


async def test_same_flow_gate_trip_stays_inside_checkout(config_root: Path) -> None:
    # A checkout-destined utterance while already sticky in checkout must NOT switch —
    # it falls through to assemble (already in the right flow).
    frontline = FakeChatModel(emit_tool_calls=False)
    graph, _ = _build(config_root, frontline=frontline)
    await _clarify_turn(graph)
    out = await graph.ainvoke({"messages": [HumanMessage("checkout now please")]}, _CFG)
    assert out["active_flow"] == "checkout"
    assert frontline._tool_calls_made == 0  # frontline still bypassed


async def test_abort_precedes_cross_switch(config_root: Path) -> None:
    # "cancel it" matches BOTH the abort escape and the gate's cancel pattern. Mid-flow it
    # means "abort the in-flight thing" — the abort check runs FIRST, so it aborts locally
    # instead of cross-switching to support's placed-order cancel.
    graph, store = _build(config_root)
    await _clarify_turn(graph)
    out = await graph.ainvoke({"messages": [HumanMessage("actually cancel it")]}, _CFG)
    assert out.get("active_flow") is None  # aborted, NOT switched into support
    assert store.cancel_count == 0  # no placed-order cancel happened
    assert any(
        isinstance(m, AIMessage) and "Nothing has been ordered" in str(m.content)
        for m in out["messages"]
    )


async def test_checkout_request_while_sticky_in_support_cross_switches(
    config_root: Path,
) -> None:
    # Symmetric: sticky in support (refund clarify), the caller pivots to buying.
    graph, _ = _build(config_root)
    out = await graph.ainvoke({"messages": [HumanMessage("I want a refund")]}, _CFG)
    assert out["active_flow"] == "support"  # in the support flow, clarifying
    out = await graph.ainvoke({"messages": [HumanMessage("checkout now please")]}, _CFG)
    assert out["active_flow"] == "checkout"  # switched into checkout
    assert out.get("pending_refund") is None  # nothing from support lingers


# --- A10a structure + the effect node ----------------------------------------------------


def test_place_is_reachable_only_through_confirm(config_root: Path) -> None:
    graph, _ = _build(config_root)
    edges = graph.get_graph().edges
    sources_into_place = {e.source for e in edges if e.target == "checkout_place"}
    assert sources_into_place == {"checkout_confirm"}


def test_place_node_replay_yields_the_same_single_order(config_root: Path) -> None:
    # A10a: the effect node may re-run (crash between effect and checkpoint); the store's
    # key dedup makes the replay return the ORIGINAL order — same spoken id, one record.
    store = OrderStore(load_orders_fixture(config_root, "acme_store"))
    nodes = build_checkout_nodes(FakeChatModel(), store, _POLICY, display_name="Acme Store")
    state = ReasoningState(
        messages=[],
        pending_action=PendingAction(
            sku="SKU-BLU-07",
            name="waterproof rain jacket",
            quantity=2,
            total_usd=258.0,
            idempotency_key="fixed-key",
            created_at=time.time(),
        ),
    )
    first = nodes.place(state)
    replay = nodes.place(state)
    assert store.placed_count == 1
    assert str(first["messages"][0].content) == str(replay["messages"][0].content)
    # The confirmed event landed in telemetry (redirected to tmp by conftest).
    lines = [
        json.loads(line)
        for line in telemetry._TELEMETRY_PATH.read_text(encoding="utf-8").splitlines()
    ]
    assert any(rec.get("event") == "checkout_confirmed" for rec in lines)


def test_speakable_nodes_are_graph_declared(config_root: Path) -> None:
    graph, _ = _build(config_root)
    # Checkout's caller-facing nodes are declared speakable (support declares its own — see
    # the support suite). The graph's set is the single source of truth the engine reads.
    assert {
        "handover",
        "checkout_guardrail",
        "checkout_confirm",
        "checkout_place",
        "checkout_abort",
    } <= graph.speakable_nodes
    # The structural safety set is unchanged: still no state-changing tool on the frontline.
    assert graph.frontline_read_only_tools == {"order_status", "catalog_search"}
