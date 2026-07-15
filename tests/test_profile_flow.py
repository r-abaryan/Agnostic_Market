"""Profile-change sub-path (Group C) at the ENGINE level: L2 step-up on the EXISTING factor
before any change, the readback consent, the idempotent effect, PII discipline, and the
no-crossover claim of the shared step-up factory. Zero network."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from llm_fakes import FakeChatModel
from support_helpers import SupportHarness, build_support_engine

from agnostic_market.agents.support import flow as support_flow
from agnostic_market.commerce.profile import ProfileError, ProfileStore
from agnostic_market.dtos.events import InterruptEvent, SpokenMessageEvent, TurnFacts
from agnostic_market.dtos.state import PolicyContext

_POLICY = PolicyContext(
    max_order_value_usd=500.0,
    allow_ai_merchant_handoff=True,
    refund_auto_approve_under_usd=50.0,
    refund_require_human_above_usd=200.0,
    refund_returnless_under_usd=50.0,
    return_window_days=30,
    pending_ttl_seconds=120.0,
)
_FACTS = TurnFacts()
_VALID_OTP = "482913"
_NEW_ADDRESS = "7 Elm Street, Dover"
# Profile changes have NO gate patterns by design — entry is the MODEL's request_handover.
_REQUEST = "I moved recently, please update my delivery details"


def _profile_harness(
    config_root: Path,
    field: str = "address",
    new_value: str = _NEW_ADDRESS,
    *,
    reason_code: str = "address_change",
    risk_flagged: bool = False,
    thread_id: str = "prof-1",
) -> SupportHarness:
    return build_support_engine(
        config_root,
        policy=_POLICY,
        frontline=FakeChatModel(
            force_tool="request_handover",
            canned_args={
                "request_handover": {"destination": "support", "reason_code": reason_code}
            },
            tool_call_limit=99,
        ),
        reasoning=FakeChatModel(
            force_tool="propose_profile_change",
            canned_args={"propose_profile_change": {"field": field, "new_value": new_value}},
            tool_call_limit=1,
        ),
        risk_flagged=risk_flagged,
        thread_id=thread_id,
    )


async def _events(engine, text: str, facts: TurnFacts = _FACTS) -> list:
    return [e async for e in engine.stream_turn(text, facts)]


def _telemetry(tmp_path: Path) -> str:
    path = tmp_path / "telemetry.jsonl"
    return path.read_text(encoding="utf-8") if path.exists() else ""


# --- the happy path: OTP on the old factor BEFORE any change ------------------------------


async def test_address_change_steps_up_before_touching_the_profile(config_root: Path) -> None:
    h = _profile_harness(config_root)
    events = await _events(h.engine, _REQUEST)
    interrupts = [e for e in events if isinstance(e, InterruptEvent)]
    assert len(interrupts) == 1
    assert "6-digit code" in interrupts[0].prompt
    assert h.otp.dispatch_count == 1
    assert h.profile.change_count == 0  # nothing changed before verification
    assert h.verification.current_level() == 1


async def test_committed_otp_then_readback_then_one_change(config_root: Path) -> None:
    h = _profile_harness(config_root)
    await _events(h.engine, _REQUEST)
    events = await _events(h.engine, _VALID_OTP)
    assert h.verification.current_level() == 2  # raised mid-flow, store-as-truth
    interrupts = [e for e in events if isinstance(e, InterruptEvent)]
    assert len(interrupts) == 1
    # The new VALUE is a declared critical field — literally spoken in the readback.
    assert _NEW_ADDRESS in interrupts[0].prompt
    assert "delivery address" in interrupts[0].prompt
    assert h.profile.change_count == 0
    done = await _events(h.engine, "yes")
    assert h.profile.change_count == 1
    assert h.profile.address_on_file() == _NEW_ADDRESS
    spoken = [e for e in done if isinstance(e, SpokenMessageEvent)]
    assert any(
        e.node == "support_profile_place" and _NEW_ADDRESS in e.text for e in spoken
    )


async def test_contact_change_updates_the_factor_reference(config_root: Path) -> None:
    h = _profile_harness(
        config_root, field="contact", new_value="555-0187", reason_code="contact_change",
        thread_id="prof-contact-1",
    )
    old_factor = h.profile.contact_on_file()
    await _events(h.engine, "I've got a new phone number, put it on my account")
    await _events(h.engine, _VALID_OTP)  # the OTP went to the OLD factor
    await _events(h.engine, "yes")
    assert h.profile.contact_on_file() == "555-0187"
    assert h.profile.contact_on_file() != old_factor


# --- security branches ---------------------------------------------------------------------


async def test_wrong_otp_twice_hands_to_human_without_changing(
    config_root: Path, tmp_path: Path
) -> None:
    h = _profile_harness(config_root)
    await _events(h.engine, _REQUEST)
    first = await _events(h.engine, "000000")
    assert any(isinstance(e, InterruptEvent) for e in first)  # ONE re-collect
    await _events(h.engine, "111111")
    assert h.profile.change_count == 0
    assert h.verification.current_level() == 1  # no free level
    assert not h.engine.pending_interrupt()  # not trapped
    # The on-ramp package fired (2nd converging path besides the consent-"human" exit).
    onramps = [
        json.loads(line)
        for line in _telemetry(tmp_path).splitlines()
        if '"human_onramp"' in line
    ]
    assert len(onramps) == 1
    assert onramps[0]["reason_code"] == "verification_required"
    assert onramps[0]["schema_version"] == 1


async def test_sim_swap_flag_blocks_the_otp_entirely(config_root: Path) -> None:
    h = _profile_harness(config_root, risk_flagged=True, thread_id="prof-risk-1")
    events = await _events(h.engine, _REQUEST)
    assert h.otp.dispatch_count == 0  # no OTP to a possibly-hijacked number
    assert not any(isinstance(e, InterruptEvent) for e in events)
    assert h.profile.change_count == 0


async def test_lapsed_level_at_place_blocks_the_change(config_root: Path) -> None:
    h = _profile_harness(config_root)
    await _events(h.engine, _REQUEST)
    await _events(h.engine, _VALID_OTP)
    h.verification.clear()  # grant revoked between step-up and the effect
    await _events(h.engine, "yes")
    assert h.profile.change_count == 0  # §A4c live re-validation at place


async def test_kill_mid_stepup_leaves_no_ghost_change_and_no_free_level(
    config_root: Path,
) -> None:
    h = _profile_harness(config_root)
    await _events(h.engine, _REQUEST)  # paused at OTP collect
    h.engine.delete_thread()  # Clock B: the call dropped
    assert h.profile.change_count == 0
    assert h.verification.current_level() == 1
    assert not h.engine.pending_interrupt()


# --- consent exits --------------------------------------------------------------------------


async def test_stale_profile_readback_expires_before_changing(
    config_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = _profile_harness(config_root)
    await _events(h.engine, _REQUEST)
    await _events(h.engine, _VALID_OTP)
    monkeypatch.setattr(support_flow.time, "time", lambda: 9_999_999_999.0)
    events = await _events(h.engine, "yes")
    assert h.profile.change_count == 0
    assert not h.engine.pending_interrupt()
    spoken = [e for e in events if isinstance(e, SpokenMessageEvent)]
    assert any("sat for a while" in e.text for e in spoken)


async def test_no_at_readback_changes_nothing(config_root: Path) -> None:
    h = _profile_harness(config_root)
    await _events(h.engine, _REQUEST)
    await _events(h.engine, _VALID_OTP)
    events = await _events(h.engine, "no, keep it as it is")
    assert h.profile.change_count == 0
    spoken = [e for e in events if isinstance(e, SpokenMessageEvent)]
    assert any("leave your details as they are" in e.text for e in spoken)


async def test_payment_change_still_defers_honestly(config_root: Path) -> None:
    # Phase 5: payment_change must NOT enter the flow — the honest deferral speaks once.
    h = _profile_harness(config_root, reason_code="payment_change", thread_id="prof-pay-1")
    events = await _events(h.engine, "put my new card on the account")
    spoken = [e for e in events if isinstance(e, SpokenMessageEvent)]
    assert any(e.node == "handover" and "support team" in e.text for e in spoken)
    assert h.profile.change_count == 0
    assert not h.engine.pending_interrupt()


# --- PII discipline (value spoken, never persisted to observability) -----------------------


async def test_new_value_never_reaches_telemetry(config_root: Path, tmp_path: Path) -> None:
    h = _profile_harness(config_root)
    await _events(h.engine, _REQUEST)
    await _events(h.engine, _VALID_OTP)
    await _events(h.engine, "yes")
    assert h.profile.change_count == 1
    raw = _telemetry(tmp_path)
    assert "Elm" not in raw  # the address value must NEVER be logged/telemetered
    confirmed = [json.loads(x) for x in raw.splitlines() if '"profile_change_confirmed"' in x]
    assert len(confirmed) == 1
    assert confirmed[0]["field"] == "address"  # the slug only
    assert "new_value" not in confirmed[0]


def test_profile_error_string_carries_no_value() -> None:
    with pytest.raises(ProfileError) as err:
        ProfileStore().update_profile("k1", field="address", new_value="   ")
    assert "address" in str(err.value)  # the field slug, never a caller value


# --- crossover isolation (the factory shares BODIES, never state) --------------------------


async def test_profile_stepup_failure_never_touches_refund_state(
    config_root: Path, tmp_path: Path
) -> None:
    h = _profile_harness(config_root)
    await _events(h.engine, _REQUEST)
    await _events(h.engine, "000000")
    await _events(h.engine, "111111")  # exhausted -> human
    raw = _telemetry(tmp_path)
    events = [json.loads(x) for x in raw.splitlines()]
    assert any(e.get("event") == "profile_stepup_failed" for e in events)
    assert not any(e.get("event", "").startswith("refund_") for e in events)
    assert h.store.refund_count == 0
