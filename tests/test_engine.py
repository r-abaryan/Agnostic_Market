"""ReasoningEngine over the REAL graph (fake models, InMemorySaver, zero network).

Covers the exit behaviors: interrupt/resume, §4a re-confirm, TTL expiry (Clock A),
kill-mid-placement (Clock B reap), idempotent placement, and the seam's zero-LiveKit claim.
Group B: the placement path is the cart flow (buy_now → guardrail → confirm → place_cart).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated, TypedDict

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langgraph.graph.message import add_messages
from llm_fakes import FakeChatModel
from policy_helpers import make_policy

from agnostic_market.agents.cart import flow as cart_flow
from agnostic_market.agents.engine import (
    ReasoningEngine,
    _GraphSpans,
    _TurnSpeech,
    build_checkpointer,
)
from agnostic_market.agents.frontline import build_frontline_graph
from agnostic_market.agents.tooling import wrap_readonly_tool
from agnostic_market.commerce.cart import CartStore
from agnostic_market.commerce.identity import (
    CallerIdentityStore,
    CustomerDirectory,
    load_customers_fixture,
)
from agnostic_market.commerce.orders import OrderStore, RecentOrderContext, load_orders_fixture
from agnostic_market.commerce.profile import ProfileStore, load_profile_fixture
from agnostic_market.commerce.verification import OtpProvider, VerificationStore
from agnostic_market.dtos.events import InterruptEvent, SpokenMessageEvent, TokenEvent, TurnFacts
from agnostic_market.voice.context import CallerContext
from agnostic_market.voice.tools import build_voice_tools

# The reasoning fake buys option 2 (waterproof rain jacket, $129.00) x2 = $258.00 -> straight
# to the placement tail via buy_now.
_PROPOSE = {"buy_now": {"candidate_key": "2", "quantity": 2}}
_FACTS = TurnFacts()
_TEST_OTP = "482913"


def _engine(
    config_root: Path,
    *,
    frontline: FakeChatModel | None = None,
    reasoning: FakeChatModel | None = None,
    identity: CallerIdentityStore | None = None,
    thread_id: str = "session-1",
) -> tuple[ReasoningEngine, OrderStore]:
    store = OrderStore(load_orders_fixture(config_root, "acme_store"))
    policy = make_policy(refund_returnless_under_usd=50.0)
    recent_orders = RecentOrderContext(max_refs=policy.cancel_batch_max)
    cart = CartStore()
    identity = identity or CallerIdentityStore()
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
    graph = build_frontline_graph(
        frontline or FakeChatModel(emit_tool_calls=False),
        tools,
        display_name="Acme Store",
        tenant_id="acme_store",
        reasoning_model=reasoning
        or FakeChatModel(force_tool="buy_now", canned_args=_PROPOSE, tool_call_limit=1),
        store=store,
        otp=otp,
        verification_store=verification,
        cart_store=cart,
        recent_orders=recent_orders,
        identity_store=identity,
        customers=customers,
        profile_store=ProfileStore(load_profile_fixture(config_root, "acme_store")),
        policy=policy,
        transition_principal=caller_context.transition_principal,
        principal_state_will_be_discarded=caller_context.has_discardable_state,
        checkpointer=build_checkpointer(),
    )
    engine = ReasoningEngine(graph, thread_id=thread_id, lifecycle=caller_context)
    caller_context.attach_engine(engine)
    return engine, store


async def _events(engine: ReasoningEngine, text: str, facts: TurnFacts = _FACTS) -> list:
    return [event async for event in engine.stream_turn(text, facts)]


async def _pause_at_confirmation(engine: ReasoningEngine) -> list:
    """Drive the graph to the readback interrupt via the gate's checkout trigger."""
    return await _events(engine, "checkout now please")


# --- the turn-failure boundary (live call #13 F-13.1: a 529 died in SILENCE) -------------


class _ExplodingFake(FakeChatModel):
    """Raises once (a provider outage surviving SDK retries), then behaves normally."""

    def _respond(self, messages, **kwargs):  # type: ignore[override]
        if not getattr(self, "_exploded", False):
            object.__setattr__(self, "_exploded", True)
            raise RuntimeError("simulated provider 529 overloaded")
        return super()._respond(messages, **kwargs)


async def test_failed_turn_speaks_the_fallback_never_silence(
    config_root: Path, tmp_path: Path
) -> None:
    import json

    from agnostic_market.agents import telemetry
    from agnostic_market.dtos.events import SpokenMessageEvent

    engine, _ = _engine(config_root, frontline=_ExplodingFake(emit_tool_calls=False))
    events = await _events(engine, "hi there")  # the graph dies mid-turn...
    spoken = [e for e in events if isinstance(e, SpokenMessageEvent)]
    assert len(spoken) == 1 and spoken[0].node == "turn_fallback"
    assert "say that again" in spoken[0].text  # ...but the caller hears the fallback
    lines = [
        json.loads(line)
        for line in telemetry._TELEMETRY_PATH.read_text(encoding="utf-8").splitlines()
    ]
    assert any(rec.get("event") == "turn_failed" for rec in lines)  # loud, not swallowed
    # The session SURVIVES: the next turn runs normally on the same thread.
    retry = await _events(engine, "hi again")
    assert any(isinstance(e, TokenEvent) and e.text for e in retry)


# --- plain turns -----------------------------------------------------------------------


async def test_plain_answer_is_spoken_exactly_once(config_root: Path) -> None:
    # A non-streaming model's answer arrives as ONE full message; the engine must speak it
    # once (fallback) and never twice (the 3a double-speak class).
    engine, _ = _engine(config_root)
    events = await _events(engine, "hi there")
    tokens = [e for e in events if isinstance(e, TokenEvent)]
    assert len(tokens) == 1
    assert tokens[0].text  # the fake's canned answer


# --- the interrupt path (V1/V2/V3) ------------------------------------------------------


async def test_checkout_pauses_with_graph_authored_readback(config_root: Path) -> None:
    engine, store = _engine(config_root)
    events = await _pause_at_confirmation(engine)
    interrupts = [e for e in events if isinstance(e, InterruptEvent)]
    assert len(interrupts) == 1
    # The readback is GRAPH-authored, SPEECH-native (§7: "2 waterproof rain jackets", not
    # "2 x ..." which TTS voices "ex"), and covers the declared confirm fields (§7a):
    # quantity + total — total computed by CODE from the fixture price (2 x $129.00).
    assert "2 waterproof rain jackets" in interrupts[0].prompt
    assert "$258.00" in interrupts[0].prompt
    assert engine.pending_interrupt()
    assert store.placed_count == 0  # nothing placed before consent


async def test_resume_yes_places_exactly_once(config_root: Path) -> None:
    reasoning = FakeChatModel(force_tool="buy_now", canned_args=_PROPOSE, tool_call_limit=1)
    engine, store = _engine(config_root, reasoning=reasoning)
    await _pause_at_confirmation(engine)
    events = await _events(engine, "yes please")
    assert store.placed_count == 1
    assert not engine.pending_interrupt()
    # The success line is node-authored by the place node and spoken.
    spoken = [e for e in events if isinstance(e, SpokenMessageEvent)]
    assert any("ORD-9001" in e.text and e.node == "cart_place" for e in spoken)
    # No NEW interrupt fires on the resume (the readback is not re-spoken - V2).
    assert not any(isinstance(e, InterruptEvent) for e in events)
    # A10a/V4: assemble did NOT re-run on resume (its model was invoked exactly once).
    assert reasoning._tool_calls_made == 1


async def test_placed_order_is_queryable_same_session(config_root: Path) -> None:
    engine, store = _engine(config_root)
    await _pause_at_confirmation(engine)
    await _events(engine, "yes")
    assert "rain jacket" in (store.order_summary("ORD-9001") or "")


async def test_resume_no_cancels_without_placing(config_root: Path) -> None:
    engine, store = _engine(config_root)
    await _pause_at_confirmation(engine)
    events = await _events(engine, "no, don't do it")
    assert store.placed_count == 0
    assert not engine.pending_interrupt()
    spoken = [e for e in events if isinstance(e, SpokenMessageEvent)]
    assert any("won't place it" in e.text.lower() for e in spoken)


# --- §4a + re-confirm-once (the user-decided consent UX) --------------------------------


async def test_barged_readback_makes_yes_invalid_and_reconfirms(config_root: Path) -> None:
    engine, store = _engine(config_root)
    await _pause_at_confirmation(engine)
    # Caller said "yes" but LiveKit marked the readback barged-over: NOT consent (§4a).
    events = await _events(engine, "yes", TurnFacts(readback_interrupted=True))
    assert store.placed_count == 0
    reconfirms = [e for e in events if isinstance(e, InterruptEvent)]
    assert len(reconfirms) == 1
    assert "yes or no" in reconfirms[0].prompt.lower()
    # A clean committed yes on the re-confirm places it.
    await _events(engine, "yes")
    assert store.placed_count == 1


async def test_unclear_answer_reconfirms_once_then_cancels(config_root: Path) -> None:
    engine, store = _engine(config_root)
    await _pause_at_confirmation(engine)
    first = await _events(engine, "wait, do you have it in blue?")
    assert any(isinstance(e, InterruptEvent) for e in first)  # ONE re-confirm
    second = await _events(engine, "hmm what about the weather")
    assert store.placed_count == 0
    assert not engine.pending_interrupt()  # cancelled, not trapped
    spoken = [e for e in second if isinstance(e, SpokenMessageEvent)]
    assert any("won't place it" in e.text.lower() for e in spoken)


async def test_human_request_at_confirmation_escapes(config_root: Path) -> None:
    engine, store = _engine(config_root)
    await _pause_at_confirmation(engine)
    events = await _events(engine, "just get me a real person please")
    assert store.placed_count == 0
    assert not engine.pending_interrupt()
    spoken = [e for e in events if isinstance(e, SpokenMessageEvent)]
    assert any(e.node == "handover" and "person" in e.text for e in spoken)


# --- Clock A: pending-confirmation TTL --------------------------------------------------


async def test_expired_pending_cancels_before_speaking(
    config_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine, store = _engine(config_root)
    await _pause_at_confirmation(engine)
    # TTL is policy-driven (default 120s); jump the flow's clock past it so the resume finds
    # the pending expired. Capture real now FIRST (checkout_flow.time is the global module —
    # a self-referential lambda would recurse). Clear-before-speak: a stale yes must NOT place.
    future = time.time() + 10_000
    monkeypatch.setattr(cart_flow.time, "time", lambda: future)
    events = await _events(engine, "yes")
    assert store.placed_count == 0
    assert not engine.pending_interrupt()  # cleared (clear-before-speak)
    spoken = [e for e in events if isinstance(e, SpokenMessageEvent)]
    assert any("haven't placed anything" in e.text for e in spoken)


# --- Clock B: kill mid-checkout (the BUILD_PLAN-named exit test) -------------------------


async def test_kill_mid_checkout_leaves_no_ghost_order(config_root: Path) -> None:
    engine, store = _engine(config_root)
    await _pause_at_confirmation(engine)
    assert engine.pending_interrupt()
    # The call drops: Clock-B teardown reaps the thread unconditionally.
    engine.delete_thread()
    assert store.placed_count == 0
    assert not engine.pending_interrupt()  # nothing resumable survives the drop
    # Reap is idempotent at the checkpointer (verified) - a double-fired hook is safe.
    engine.delete_thread()
    # A fresh session (new thread) starts clean: no ghost order, no stale pending.
    engine2, store2 = _engine(config_root, thread_id="session-2")
    events = await _events(engine2, "hi there")
    assert store2.placed_count == 0
    assert not engine2.pending_interrupt()
    assert any(isinstance(e, TokenEvent) for e in events)


# --- checkpoint serde: our DTOs are registered (no 'unregistered type' warning) ----------


async def test_checkpointed_dtos_deserialize_without_unregistered_warning(
    config_root: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # build_checkpointer's serde allowlists PendingPlacement/CartLine/PendingRefund/
    # HandoffRequest/ReasoningState — a checkpointed interrupt/resume (which deserializes
    # PendingPlacement + nested CartLine + HandoffRequest live) must NOT emit langgraph's
    # "Deserializing unregistered type" warning, which is slated to become a hard block.
    engine, _ = _engine(config_root)
    with caplog.at_level("WARNING", logger="langgraph.checkpoint.serde.jsonplus"):
        await _pause_at_confirmation(engine)  # writes + reads PendingAction across the interrupt
        await _events(engine, "yes")  # resume re-reads the checkpoint
    assert not any("unregistered" in r.getMessage().lower() for r in caplog.records)


# --- buffer-before-speak (_TurnSpeech, live call #9 P2) ----------------------------------
# The chunk path can't be driven through the graph with non-streaming fakes, so the
# buffering unit is tested directly with synthetic stream items.


_SPEAKABLE = frozenset({"support_guardrail"})
_ASSEMBLE_META = {"langgraph_node": "support_assemble"}


def _chunk(text: str, msg_id: str, *, tool_call: bool = False) -> AIMessageChunk:
    chunks = (
        [{"name": "propose_refund", "args": "", "id": "tc1", "index": 0}] if tool_call else []
    )
    return AIMessageChunk(content=text, id=msg_id, tool_call_chunks=chunks)


def test_streamed_clarify_speaks_once_at_message_completion() -> None:
    speech = _TurnSpeech(_SPEAKABLE)
    assert speech.feed(_chunk("Which ", "m1"), _ASSEMBLE_META) is None
    assert speech.feed(_chunk("order?", "m1"), _ASSEMBLE_META) is None
    event = speech.feed(AIMessage(content="Which order?", id="m1"), _ASSEMBLE_META)
    assert isinstance(event, TokenEvent)
    assert event.text == "Which order?"
    assert list(speech.flush()) == []  # never re-spoken


def test_toolcall_narration_never_reaches_the_caller() -> None:
    # Live call #9 P2: "Shall I set the refund...?" streamed alongside propose_refund and
    # was heard OVER the guardrail's return-first decline. The narration must be dropped.
    speech = _TurnSpeech(_SPEAKABLE)
    speech.feed(_chunk("Shall I refund?", "m1"), _ASSEMBLE_META)
    done = AIMessage(
        content="Shall I refund?",
        id="m1",
        tool_calls=[{"name": "propose_refund", "args": {}, "id": "tc1", "type": "tool_call"}],
    )
    assert speech.feed(done, _ASSEMBLE_META) is None
    assert list(speech.flush()) == []  # and not resurrected at flush


def test_toolcall_detected_from_chunks_alone_stays_dropped() -> None:
    # The completed message never arrives (or arrives under another id): chunks that
    # carried tool_call_chunks must stay dropped at flush.
    speech = _TurnSpeech(_SPEAKABLE)
    speech.feed(_chunk("Refunding now.", "m1", tool_call=True), _ASSEMBLE_META)
    assert list(speech.flush()) == []


def test_unstreamed_answer_speaks_once() -> None:
    # Non-streaming provider / test fake: the answer arrives only as one full message.
    speech = _TurnSpeech(_SPEAKABLE)
    event = speech.feed(AIMessage(content="Hello!"), {"langgraph_node": "frontline"})
    assert isinstance(event, TokenEvent)
    assert event.text == "Hello!"


def test_speakable_node_line_is_a_spoken_message() -> None:
    speech = _TurnSpeech(_SPEAKABLE)
    event = speech.feed(
        AIMessage(content="Refund declined.", id="g1"), {"langgraph_node": "support_guardrail"}
    )
    assert isinstance(event, SpokenMessageEvent)
    assert event.node == "support_guardrail"


def test_orphan_streamed_clarify_flushes_exactly_once() -> None:
    # Defensive: chunks streamed but no completed message echoed — the clarify must not be
    # swallowed silently.
    speech = _TurnSpeech(_SPEAKABLE)
    speech.feed(_chunk("Which order?", "m1"), _ASSEMBLE_META)
    assert [e.text for e in speech.flush()] == ["Which order?"]


def test_id_change_between_chunk_and_completion_does_not_double_speak() -> None:
    speech = _TurnSpeech(_SPEAKABLE)
    speech.feed(_chunk("Which order?", "m1"), _ASSEMBLE_META)
    event = speech.feed(AIMessage(content="Which order?", id="m2"), _ASSEMBLE_META)
    assert isinstance(event, TokenEvent)  # spoken at completion under the new id
    assert list(speech.flush()) == []  # the orphaned m1 buffer must not re-speak it


def test_tool_messages_never_surface() -> None:
    speech = _TurnSpeech(_SPEAKABLE)
    assert speech.feed(ToolMessage("raw", tool_call_id="t1"), _ASSEMBLE_META) is None
    assert list(speech.flush()) == []


# --- graph latency spans (_GraphSpans, live call #10) ------------------------------------


_MODEL = {"langgraph_node": "model"}  # the only node that runs the LLM


def test_graph_spans_counts_tools_and_ttf_model() -> None:
    spans = _GraphSpans()
    spans.observe(AIMessage(content="", tool_calls=[
        {"name": "order_status", "args": {}, "id": "t1", "type": "tool_call"}]), _MODEL)
    spans.observe(ToolMessage("processing", tool_call_id="t1"), {"langgraph_node": "tools"})
    spans.observe(AIMessage(content="It's processing."), _MODEL)  # a real second model pass
    # A tool ran, and the second model pass (rendering the result) is timed separately.
    assert spans._tool_count == 1
    assert spans._ttf_model is not None
    assert spans._tool_to_next_model is not None  # tool-result -> next model activity


def test_graph_spans_single_pass_has_no_tool_span() -> None:
    spans = _GraphSpans()
    spans.observe(AIMessage(content="Hello!"), _MODEL)
    assert spans._tool_count == 0
    assert spans._ttf_model is not None
    assert spans._tool_to_next_model is None  # no tool -> no second-pass span


def test_graph_spans_ignores_empty_and_tool_only_for_ttf() -> None:
    spans = _GraphSpans()
    spans.observe(AIMessage(content=""), _MODEL)  # empty: not model output
    assert spans._ttf_model is None
    spans.observe(AIMessage(content="", tool_calls=[
        {"name": "x", "args": {}, "id": "t1", "type": "tool_call"}]), _MODEL)
    assert spans._ttf_model is not None  # a tool-call message IS model activity


def test_graph_spans_render_node_is_not_a_model_pass() -> None:
    # L3: a deterministic read renderer (read_render node) authors the post-tool line INSTEAD
    # of a second model pass. That node-authored AIMessage must NOT be timed as model
    # activity, or every rendered turn shows a phantom tool_to_next_model cost.
    spans = _GraphSpans()
    spans.observe(AIMessage(content="", tool_calls=[
        {"name": "order_status", "args": {}, "id": "t1", "type": "tool_call"}]), _MODEL)
    spans.observe(ToolMessage("...", tool_call_id="t1"), {"langgraph_node": "tools"})
    spans.observe(AIMessage(content="Your order ORD-1001 is on its way."),
                  {"langgraph_node": "read_render"})
    assert spans._tool_count == 1
    assert spans._tool_to_next_model is None  # rendered, NOT a second model pass


# --- the seam's zero-LiveKit claim ------------------------------------------------------


def test_engine_module_imports_no_livekit() -> None:
    # The dependency arrow is voice -> engine ONLY (AGENTS §A0): the engine must be
    # importable and testable without the voice plane.
    import agnostic_market.agents.engine as engine_module

    source = Path(engine_module.__file__).read_text(encoding="utf-8")
    assert "livekit" not in source.lower()


# --- Fix 5 Milestone 0: LangGraph thread-rotation CONTRACT (framework behavior gate) --------
#
# This is a self-contained FRAMEWORK contract test (a mini-graph below; NO production graph, NO
# src/ dependency beyond build_checkpointer's production serde). It pins the LangGraph behaviors
# Fix 5's principal-context rotation (Milestone D) will rely on, so a LangGraph upgrade that
# changes any of them fails HERE instead of silently breaking rotation:
#   1. a FRESH thread is seeded with ONLY a typed continuation via update_state(as_node=START);
#   2. that seed routes DETERMINISTICALLY into the intended bootstrap node;
#   3. a confirmation interrupt is created AND resumed IN THE NEW thread, its effect running once;
#   4. a DELETED old thread cannot recover its messages/interrupt/continuation and does NOT run the
#      old effect (it may restart from START — pinned precisely so a behavior change is caught).
# The active-thread-id SWAP that makes "old cannot resume" a hard guarantee is PRODUCTION code
# (Milestone D) with its own test; deletion alone is what this contract covers.


def _rot_append(a: list | None, b: list | None) -> list:
    return (a or []) + (b or [])


class _RotState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    continuation: str | None  # the typed continuation SEED (a proposal, not authority)
    resolved: str | None  # what the bootstrap froze from the continuation
    effect_done: int  # idempotence witness
    visited: Annotated[list[str], _rot_append]


def _rotation_contract_graph():
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import interrupt

    def resolver(state: _RotState) -> dict:  # the continuation bootstrap node
        return {"resolved": f"target::{state.get('continuation')}", "visited": ["resolver"]}

    def confirm(state: _RotState) -> dict:
        decision = interrupt({"ask": f"confirm {state.get('resolved')}?"})
        return {"visited": ["confirm"], "messages": [("human", str(decision))]}

    def route_after_confirm(state: _RotState) -> str:
        last = state["messages"][-1]
        text = (last.content if hasattr(last, "content") else str(last)).lower()
        return "effect" if "yes" in text else END

    def effect(state: _RotState) -> dict:
        return {"effect_done": state.get("effect_done", 0) + 1, "visited": ["effect"]}

    g = StateGraph(_RotState)
    g.add_node("resolver", resolver)
    g.add_node("confirm", confirm)
    g.add_node("effect", effect)
    g.add_edge(START, "resolver")
    g.add_edge("resolver", "confirm")
    g.add_conditional_edges("confirm", route_after_confirm, {"effect": "effect", END: END})
    g.add_edge("effect", END)
    return g.compile(checkpointer=build_checkpointer())


def test_langgraph_thread_rotation_contract() -> None:
    from langgraph.types import Command

    graph = _rotation_contract_graph()
    old = {"configurable": {"thread_id": "OLD"}}
    new = {"configurable": {"thread_id": "NEW"}}

    # OLD thread accumulates prior-principal state and pauses at an interrupt.
    graph.update_state(old, {"messages": [("human", "A-private")], "continuation": "A"})
    graph.invoke(None, old)
    assert graph.get_state(old).next == ("confirm",)
    assert len(graph.get_state(old).interrupts) == 1

    # ROTATE: delete OLD, seed a FRESH thread with ONLY the typed continuation.
    graph.checkpointer.delete_thread("OLD")
    graph.update_state(new, {"continuation": "B-cancel-all", "effect_done": 0}, as_node="__start__")
    seeded = graph.get_state(new)
    assert seeded.values.get("continuation") == "B-cancel-all"
    assert seeded.values.get("messages") in (None, [])  # NO prior-principal message bleed
    assert seeded.next == ("resolver",)  # (2) deterministic bootstrap into the intended node

    # Drive NEW with no input: resolver -> confirm -> interrupt IN THE NEW THREAD.
    graph.invoke(None, new)
    st = graph.get_state(new)
    assert [i.value["ask"] for i in st.interrupts] == ["confirm target::B-cancel-all?"]
    assert st.next == ("confirm",)

    # (4) DELETED OLD cannot recover: a stray resume restarts from START — old messages/interrupt/
    # continuation gone, and the old effect is NOT executed.
    old_after = graph.invoke(Command(resume="yes"), old)
    assert old_after.get("effect_done") is None  # old effect never ran
    assert old_after.get("visited") == ["resolver"]  # restarted from START, not resumed at confirm
    assert old_after.get("messages", []) == []  # no "A-private" recovered

    # (3) Resume the NEW thread's interrupt -> effect runs EXACTLY once.
    final = graph.invoke(Command(resume="yes"), new)
    assert final.get("effect_done") == 1
    assert final.get("visited") == ["resolver", "confirm", "effect"]
    assert graph.get_state(new).next == ()  # completed
