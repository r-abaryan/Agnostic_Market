"""Support flow (T3) at the ENGINE level: the step-up verification loop, refund-to-new-
instrument, and its exit behaviors. Zero network (fake models + InMemorySaver).

Drives the real graph through the ReasoningEngine so the interrupt/resume path is exercised
exactly as production does. The reasoning fake proposes a refund via `propose_refund`; the
frontline gate trips on "refund ..." and enters the support flow.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from llm_fakes import FakeChatModel

from agnostic_market.agents.engine import ReasoningEngine, build_checkpointer
from agnostic_market.agents.frontline import build_frontline_graph
from agnostic_market.agents.support import flow as support_flow
from agnostic_market.agents.tooling import wrap_readonly_tool
from agnostic_market.commerce.orders import OrderStore, load_orders_fixture
from agnostic_market.commerce.verification import OtpProvider, RiskProvider, VerificationStore
from agnostic_market.dtos.events import (
    InterruptEvent,
    SpokenMessageEvent,
    TokenEvent,
    TurnFacts,
)
from agnostic_market.dtos.state import PolicyContext
from agnostic_market.voice.tools import build_voice_tools

_POLICY = PolicyContext(
    max_order_value_usd=500.0,
    allow_ai_merchant_handoff=True,
    refund_auto_approve_under_usd=50.0,
    refund_require_human_above_usd=200.0,
    pending_ttl_seconds=120.0,
)
_FACTS = TurnFacts()
_VALID_OTP = "482913"
# The reasoning fake proposes a refund of $129.00 on order key "2" (ORD-1002) to a NEW card.
_PROPOSE = {
    "propose_refund": {"order_key": "2", "amount_usd": 129.0, "destination": "new_instrument"}
}
_REFUND_REQUEST = "I'd like a refund to a different card"


def _engine(
    config_root: Path,
    *,
    reasoning: FakeChatModel | None = None,
    risk_flagged: bool = False,
    policy: PolicyContext = _POLICY,
    thread_id: str = "support-1",
) -> tuple[ReasoningEngine, OrderStore, VerificationStore, OtpProvider]:
    store = OrderStore(load_orders_fixture(config_root, "acme_store"))
    tools = [wrap_readonly_tool(t, "acme_store") for t in build_voice_tools(store)]
    otp = OtpProvider()
    verification = VerificationStore(otp)
    graph = build_frontline_graph(
        FakeChatModel(emit_tool_calls=False),
        tools,
        display_name="Acme Store",
        reasoning_model=reasoning
        or FakeChatModel(force_tool="propose_refund", canned_args=_PROPOSE, tool_call_limit=1),
        store=store,
        policy=policy,
        verification_store=verification,
        otp=otp,
        risk=RiskProvider(flagged=risk_flagged),
        checkpointer=build_checkpointer(),
    )
    return ReasoningEngine(graph, thread_id=thread_id), store, verification, otp


async def _events(engine: ReasoningEngine, text: str, facts: TurnFacts = _FACTS) -> list:
    return [event async for event in engine.stream_turn(text, facts)]


async def _pause_at_otp(engine: ReasoningEngine) -> list:
    return await _events(engine, _REFUND_REQUEST)


# --- the happy T3 path -------------------------------------------------------------------


async def test_stepup_asks_for_otp_before_touching_money(config_root: Path) -> None:
    engine, store, verification, otp = _engine(config_root)
    events = await _pause_at_otp(engine)
    interrupts = [e for e in events if isinstance(e, InterruptEvent)]
    assert len(interrupts) == 1
    assert "6-digit code" in interrupts[0].prompt
    assert otp.dispatch_count == 1  # OTP dispatched exactly once
    assert verification.current_level() == 1  # not raised yet
    assert store.refund_count == 0  # nothing refunded before verification


async def test_committed_otp_raises_level_then_reads_back_the_refund(config_root: Path) -> None:
    engine, store, verification, _ = _engine(config_root)
    await _pause_at_otp(engine)
    events = await _events(engine, _VALID_OTP)
    assert verification.current_level() == 2  # raised mid-flow
    interrupts = [e for e in events if isinstance(e, InterruptEvent)]
    assert len(interrupts) == 1
    # GRAPH-authored, speech-native readback (§7a): amount + masked instrument, no PAN.
    assert "$129.00" in interrupts[0].prompt
    assert "card ending 4471" in interrupts[0].prompt
    assert store.refund_count == 0  # still nothing placed before the yes


async def test_yes_after_stepup_issues_exactly_one_refund(config_root: Path) -> None:
    engine, store, _, _ = _engine(config_root)
    await _pause_at_otp(engine)
    await _events(engine, _VALID_OTP)
    events = await _events(engine, "yes please")
    assert store.refund_count == 1
    assert store.refunded_so_far("ORD-1002") == 129.0
    assert not engine.pending_interrupt()
    spoken = [e for e in events if isinstance(e, SpokenMessageEvent)]
    assert any(e.node == "support_place" and "refund" in e.text.lower() for e in spoken)


async def test_double_resume_of_the_confirm_never_double_refunds(config_root: Path) -> None:
    engine, store, _, _ = _engine(config_root)
    await _pause_at_otp(engine)
    await _events(engine, _VALID_OTP)
    await _events(engine, "yes")
    # A stray extra turn after the flow completed must not re-place (no interrupt pending).
    await _events(engine, "yes")
    assert store.refund_count == 1


# --- the security branches (T3 failure variants) ----------------------------------------


async def test_sim_swap_flag_blocks_otp_and_hands_to_human(config_root: Path) -> None:
    engine, store, verification, otp = _engine(config_root, risk_flagged=True)
    events = await _events(engine, _REFUND_REQUEST)
    # No OTP dispatched, no interrupt (we don't collect a code we can't trust), no refund.
    assert otp.dispatch_count == 0
    assert not any(isinstance(e, InterruptEvent) for e in events)
    assert store.refund_count == 0
    assert verification.current_level() == 1


async def test_wrong_otp_twice_stays_l1_and_never_refunds(config_root: Path) -> None:
    engine, store, verification, _ = _engine(config_root)
    await _pause_at_otp(engine)
    first = await _events(engine, "000000")  # wrong -> re-collect
    assert any(isinstance(e, InterruptEvent) for e in first)  # a second OTP ask
    await _events(engine, "111111")  # wrong again -> exhausted
    assert verification.current_level() == 1
    assert store.refund_count == 0
    assert not engine.pending_interrupt()  # not trapped


async def test_live_read_blocks_a_lapsed_grant_at_place(config_root: Path) -> None:
    # The T3 security property: the level is read LIVE. If the grant is revoked after step-up
    # but before the effect, the place node re-validates and refuses to move money.
    engine, store, verification, _ = _engine(config_root)
    await _pause_at_otp(engine)
    await _events(engine, _VALID_OTP)
    assert verification.current_level() == 2
    verification.clear()  # session grant revoked (SIM-swap flagged / session invalidated)
    await _events(engine, "yes")
    assert store.refund_count == 0  # money did NOT move on a lapsed level


# --- §4a committed-consent ---------------------------------------------------------------


async def test_barged_readback_reconfirms_before_refunding(config_root: Path) -> None:
    engine, store, _, _ = _engine(config_root)
    await _pause_at_otp(engine)
    await _events(engine, _VALID_OTP)
    # "yes" over a barged-over readback is not consent (§4a) -> re-confirm, not refund.
    events = await _events(engine, "yes", TurnFacts(readback_interrupted=True))
    assert store.refund_count == 0
    reconfirms = [e for e in events if isinstance(e, InterruptEvent)]
    assert len(reconfirms) == 1
    assert "yes or no" in reconfirms[0].prompt.lower()
    await _events(engine, "yes")  # a clean committed yes places it
    assert store.refund_count == 1


async def test_no_at_readback_cancels_without_refunding(config_root: Path) -> None:
    engine, store, _, _ = _engine(config_root)
    await _pause_at_otp(engine)
    await _events(engine, _VALID_OTP)
    events = await _events(engine, "no, don't")
    assert store.refund_count == 0
    assert not engine.pending_interrupt()
    spoken = [e for e in events if isinstance(e, SpokenMessageEvent)]
    assert any("nothing has changed" in e.text.lower() for e in spoken)


# --- kill mid-step-up (Clock B) ----------------------------------------------------------


async def test_kill_mid_stepup_leaves_no_ghost_refund_and_no_free_level(config_root: Path) -> None:
    engine, store, verification, _ = _engine(config_root)
    await _pause_at_otp(engine)  # paused at the OTP collect interrupt
    assert engine.pending_interrupt()
    engine.delete_thread()  # Clock-B teardown
    verification.clear()  # pipeline reaper also clears the grant
    assert store.refund_count == 0
    assert verification.current_level() == 1
    # A fresh session starts clean.
    engine2, store2, verification2, _ = _engine(config_root, thread_id="support-2")
    events = await _events(engine2, "hi there")
    assert store2.refund_count == 0
    assert verification2.current_level() == 1
    assert any(isinstance(e, TokenEvent) for e in events)


# --- Group A: refund-to-ORIGINAL (L1, no step-up) ----------------------------------------

# ORD-1002 is "processing" (cancellable), $129.00 captured.
_REFUND_ORIGINAL = {
    "propose_refund": {"order_key": "2", "amount_usd": 50.0, "destination": "original"}
}
_CANCEL_PROCESSING = {"propose_cancel": {"order_key": "2"}}  # ORD-1002, processing
_CANCEL_SHIPPED = {"propose_cancel": {"order_key": "1"}}  # ORD-1001, shipped


def _cancel_engine(config_root, args, *, thread_id, risk_flagged=False):
    return _engine(
        config_root,
        reasoning=FakeChatModel(
            force_tool=next(iter(args)), canned_args=args, tool_call_limit=1
        ),
        risk_flagged=risk_flagged,
        thread_id=thread_id,
    )


async def test_refund_to_original_is_l1_no_stepup(config_root: Path) -> None:
    engine, store, verification, otp = _engine(
        config_root,
        reasoning=FakeChatModel(
            force_tool="propose_refund", canned_args=_REFUND_ORIGINAL, tool_call_limit=1
        ),
        thread_id="ro-1",
    )
    events = await _events(engine, "I want a refund to my original card")
    # NO OTP dispatched (L1 already satisfies refund-to-original) — straight to the readback.
    assert otp.dispatch_count == 0
    assert verification.current_level() == 1
    interrupts = [e for e in events if isinstance(e, InterruptEvent)]
    assert len(interrupts) == 1
    assert "$50.00" in interrupts[0].prompt
    assert "your your" not in interrupts[0].prompt  # the double-'your' bug stays fixed
    await _events(engine, "yes")
    assert store.refund_count == 1
    assert store.refunded_so_far("ORD-1002") == 50.0


# --- Group A: cancel-order ---------------------------------------------------------------


async def test_cancel_processing_order_voids_after_readback(config_root: Path) -> None:
    engine, store, _, _ = _cancel_engine(config_root, _CANCEL_PROCESSING, thread_id="cx-1")
    events = await _events(engine, "cancel my rain jacket order")
    interrupts = [e for e in events if isinstance(e, InterruptEvent)]
    assert len(interrupts) == 1
    assert "ORD-1002" in interrupts[0].prompt and "can't be undone" in interrupts[0].prompt
    assert store.cancel_count == 0  # nothing voided before consent
    events = await _events(engine, "yes")
    assert store.cancel_count == 1
    assert store.order_status("ORD-1002") == "cancelled"
    spoken = [e for e in events if isinstance(e, SpokenMessageEvent)]
    assert any(e.node == "support_cancel_void" and "cancelled" in e.text.lower() for e in spoken)


async def test_cancel_shipped_order_declines_without_voiding(config_root: Path) -> None:
    engine, store, _, _ = _cancel_engine(config_root, _CANCEL_SHIPPED, thread_id="cx-2")
    events = await _events(engine, "cancel my shoes order")
    # No interrupt (we don't read back a cancel we can't do), no void, one honest line.
    assert not any(isinstance(e, InterruptEvent) for e in events)
    assert store.cancel_count == 0
    spoken = [e for e in events if isinstance(e, SpokenMessageEvent)]
    assert any("already shipped" in e.text.lower() for e in spoken)


async def test_cancel_risk_flagged_hands_to_human_without_voiding(config_root: Path) -> None:
    engine, store, _, _ = _cancel_engine(
        config_root, _CANCEL_PROCESSING, thread_id="cx-3", risk_flagged=True
    )
    events = await _events(engine, "cancel my rain jacket order")
    assert not any(isinstance(e, InterruptEvent) for e in events)
    assert store.cancel_count == 0  # a risk-flagged session gets no silent void
    spoken = [e for e in events if isinstance(e, SpokenMessageEvent)]
    assert any(e.node == "handover" and "person" in e.text.lower() for e in spoken)


async def test_cancel_no_at_readback_leaves_order_untouched(config_root: Path) -> None:
    engine, store, _, _ = _cancel_engine(config_root, _CANCEL_PROCESSING, thread_id="cx-4")
    await _events(engine, "cancel my rain jacket order")
    events = await _events(engine, "no, leave it")
    assert store.cancel_count == 0
    assert store.order_status("ORD-1002") == "processing"  # untouched
    spoken = [e for e in events if isinstance(e, SpokenMessageEvent)]
    assert any("leave that order" in e.text.lower() for e in spoken)


async def test_cancel_barged_readback_reconfirms_before_voiding(config_root: Path) -> None:
    engine, store, _, _ = _cancel_engine(config_root, _CANCEL_PROCESSING, thread_id="cx-5")
    await _events(engine, "cancel my rain jacket order")
    # "yes" over a barged-over readback is not consent (§4a) -> re-confirm, not void.
    events = await _events(engine, "yes", TurnFacts(readback_interrupted=True))
    assert store.cancel_count == 0
    reconfirms = [e for e in events if isinstance(e, InterruptEvent)]
    assert len(reconfirms) == 1
    assert "yes or no" in reconfirms[0].prompt.lower()
    await _events(engine, "yes")  # a clean committed yes voids it
    assert store.cancel_count == 1


async def test_cancel_is_idempotent_across_double_resume(config_root: Path) -> None:
    engine, store, _, _ = _cancel_engine(config_root, _CANCEL_PROCESSING, thread_id="cx-6")
    await _events(engine, "cancel my rain jacket order")
    await _events(engine, "yes")
    await _events(engine, "yes")  # a stray extra turn after completion must not re-void
    assert store.cancel_count == 1


async def test_kill_mid_cancel_leaves_no_ghost(config_root: Path) -> None:
    engine, store, _, _ = _cancel_engine(config_root, _CANCEL_PROCESSING, thread_id="cx-7")
    await _events(engine, "cancel my rain jacket order")  # paused at the cancel readback
    assert engine.pending_interrupt()
    engine.delete_thread()
    assert store.cancel_count == 0
    assert store.order_status("ORD-1002") == "processing"  # nothing voided on the drop


# --- F-1: refund amount gate (merchant policy, within platform bounds) --------------------


def _refund_engine(config_root, amount, dest, *, thread_id, policy=_POLICY):
    # ORD-1001 (key "1") is shipped but total 179.98; refunds don't need it cancellable.
    return _engine(
        config_root,
        reasoning=FakeChatModel(
            force_tool="propose_refund",
            canned_args={
                "propose_refund": {"order_key": "1", "amount_usd": amount, "destination": dest}
            },
            tool_call_limit=1,
        ),
        policy=policy,
        thread_id=thread_id,
    )


async def test_refund_in_band_reaches_readback(config_root: Path) -> None:
    # $150 <= require_human_above (200) -> agent processes behind the readback.
    engine, store, _, _ = _refund_engine(config_root, 150.0, "original", thread_id="amt-1")
    events = await _events(engine, "I want a refund to my original card")
    interrupts = [e for e in events if isinstance(e, InterruptEvent)]
    assert len(interrupts) == 1
    assert "$150.00" in interrupts[0].prompt
    assert store.refund_count == 0  # not until the yes


async def test_refund_over_human_threshold_routes_to_person_no_refund(config_root: Path) -> None:
    # $250 > require_human_above (200) -> a person, NO readback, NO refund, specific line.
    engine, store, _, otp = _refund_engine(config_root, 250.0, "original", thread_id="amt-2")
    events = await _events(engine, "I want a refund to my original card")
    assert not any(isinstance(e, InterruptEvent) for e in events)
    assert store.refund_count == 0
    assert otp.dispatch_count == 0  # no step-up either — it just goes to a human
    spoken = [e for e in events if isinstance(e, SpokenMessageEvent)]
    assert any("above what I can process" in e.text for e in spoken)


async def test_refund_amount_gate_precedes_stepup(config_root: Path) -> None:
    # An over-threshold refund to a NEW instrument (would need L2) must NOT enter the OTP
    # loop — the amount gate routes to a human FIRST (no point verifying a refund a human
    # must handle anyway).
    engine, store, _, otp = _refund_engine(config_root, 300.0, "new_instrument", thread_id="amt-3")
    events = await _events(engine, "refund my order to a different card")
    assert otp.dispatch_count == 0  # amount gate fired before dispatch
    assert store.refund_count == 0
    assert not any(isinstance(e, InterruptEvent) for e in events)


# --- F-2: Clock-A TTL on the support confirm nodes (clear-before-speak) -------------------


async def test_stale_refund_readback_expires_before_placing(
    config_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine, store, _, _ = _engine(config_root)
    await _pause_at_otp(engine)
    await _events(engine, _VALID_OTP)  # raised to L2, now paused at the refund readback
    # Jump the flow clock past the TTL; the resume must find the pending expired.
    future = time.time() + 10_000
    monkeypatch.setattr(support_flow.time, "time", lambda: future)
    events = await _events(engine, "yes")  # a stale yes must NOT refund
    assert store.refund_count == 0
    assert not engine.pending_interrupt()
    spoken = [e for e in events if isinstance(e, SpokenMessageEvent)]
    assert any("sat for a while" in e.text for e in spoken)


async def test_stale_cancel_readback_expires_before_voiding(
    config_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine, store, _, _ = _cancel_engine(config_root, _CANCEL_PROCESSING, thread_id="ttl-c")
    await _events(engine, "cancel my rain jacket order")  # paused at the cancel readback
    future = time.time() + 10_000
    monkeypatch.setattr(support_flow.time, "time", lambda: future)
    events = await _events(engine, "yes")  # a stale yes must NOT void
    assert store.cancel_count == 0
    assert store.order_status("ORD-1002") == "processing"
    assert not engine.pending_interrupt()
    spoken = [e for e in events if isinstance(e, SpokenMessageEvent)]
    assert any("sat for a while" in e.text for e in spoken)
