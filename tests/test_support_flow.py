"""Support flow (T3) at the ENGINE level: the step-up verification loop, refund-to-new-
instrument, and its exit behaviors. Zero network (fake models + InMemorySaver).

Drives the real graph through the ReasoningEngine so the interrupt/resume path is exercised
exactly as production does. The reasoning fake proposes a refund via `propose_refund`; the
frontline gate trips on "refund ..." and enters the support flow.
"""

from __future__ import annotations

from pathlib import Path

from llm_fakes import FakeChatModel

from agnostic_market.agents.engine import ReasoningEngine, build_checkpointer
from agnostic_market.agents.frontline import build_frontline_graph
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

_POLICY = PolicyContext(max_order_value_usd=500.0, allow_ai_merchant_handoff=True)
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
        policy=_POLICY,
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
