"""ReasoningEngine over the REAL graph (fake models, InMemorySaver, zero network).

Covers the 3b exit behaviors: interrupt/resume, §4a re-confirm, TTL expiry (Clock A),
kill-mid-checkout (Clock B reap), idempotent placement, and the seam's zero-LiveKit claim.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from llm_fakes import FakeChatModel

from agnostic_market.agents.checkout import flow as checkout_flow
from agnostic_market.agents.engine import ReasoningEngine, build_checkpointer
from agnostic_market.agents.frontline import build_frontline_graph
from agnostic_market.agents.tooling import wrap_readonly_tool
from agnostic_market.commerce.orders import OrderStore, load_orders_fixture
from agnostic_market.dtos.events import InterruptEvent, SpokenMessageEvent, TokenEvent, TurnFacts
from agnostic_market.dtos.state import PolicyContext
from agnostic_market.voice.tools import build_voice_tools

# The reasoning fake proposes option 2 (waterproof rain jacket, $129.00) x2 = $258.00.
_PROPOSE = {"propose_order": {"candidate_key": "2", "quantity": 2}}
_FACTS = TurnFacts()


def _engine(
    config_root: Path,
    *,
    frontline: FakeChatModel | None = None,
    reasoning: FakeChatModel | None = None,
    thread_id: str = "session-1",
) -> tuple[ReasoningEngine, OrderStore]:
    store = OrderStore(load_orders_fixture(config_root, "acme_store"))
    tools = [wrap_readonly_tool(t, "acme_store") for t in build_voice_tools(store)]
    graph = build_frontline_graph(
        frontline or FakeChatModel(emit_tool_calls=False),
        tools,
        display_name="Acme Store",
        reasoning_model=reasoning
        or FakeChatModel(force_tool="propose_order", canned_args=_PROPOSE, tool_call_limit=1),
        store=store,
        policy=PolicyContext(
            max_order_value_usd=500.0,
            allow_ai_merchant_handoff=True,
            refund_auto_approve_under_usd=50.0,
            refund_require_human_above_usd=200.0,
            pending_ttl_seconds=120.0,
        ),
        checkpointer=build_checkpointer(),
    )
    return ReasoningEngine(graph, thread_id=thread_id), store


async def _events(engine: ReasoningEngine, text: str, facts: TurnFacts = _FACTS) -> list:
    return [event async for event in engine.stream_turn(text, facts)]


async def _pause_at_confirmation(engine: ReasoningEngine) -> list:
    """Drive the graph to the readback interrupt via the gate's checkout trigger."""
    return await _events(engine, "checkout now please")


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
    reasoning = FakeChatModel(force_tool="propose_order", canned_args=_PROPOSE, tool_call_limit=1)
    engine, store = _engine(config_root, reasoning=reasoning)
    await _pause_at_confirmation(engine)
    events = await _events(engine, "yes please")
    assert store.placed_count == 1
    assert not engine.pending_interrupt()
    # The success line is node-authored by the place node and spoken.
    spoken = [e for e in events if isinstance(e, SpokenMessageEvent)]
    assert any("ORD-9001" in e.text and e.node == "checkout_place" for e in spoken)
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
    assert any("nothing has been ordered" in e.text.lower() for e in spoken)


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
    assert any("nothing has been ordered" in e.text.lower() for e in spoken)


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
    monkeypatch.setattr(checkout_flow.time, "time", lambda: future)
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
    # build_checkpointer's serde allowlists PendingAction/PendingRefund/HandoffRequest/
    # ReasoningState — a checkpointed interrupt/resume (which deserializes PendingAction +
    # HandoffRequest live) must NOT emit langgraph's "Deserializing unregistered type"
    # warning, which is slated to become a hard block.
    engine, _ = _engine(config_root)
    with caplog.at_level("WARNING", logger="langgraph.checkpoint.serde.jsonplus"):
        await _pause_at_confirmation(engine)  # writes + reads PendingAction across the interrupt
        await _events(engine, "yes")  # resume re-reads the checkpoint
    assert not any("unregistered" in r.getMessage().lower() for r in caplog.records)


# --- the seam's zero-LiveKit claim ------------------------------------------------------


def test_engine_module_imports_no_livekit() -> None:
    # The dependency arrow is voice -> engine ONLY (AGENTS §A0): the engine must be
    # importable and testable without the voice plane.
    import agnostic_market.agents.engine as engine_module

    source = Path(engine_module.__file__).read_text(encoding="utf-8")
    assert "livekit" not in source.lower()
