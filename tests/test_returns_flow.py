"""Returns sub-path (Group C) at the ENGINE level: the direct door (propose_return) and the
steered door (refund tier-4), the eligibility guardrail tiers, the idempotent RMA effect,
and the refund interplay. Zero network (fake models + InMemorySaver)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from llm_fakes import FakeChatModel
from policy_helpers import make_policy
from support_helpers import (
    TEST_OTP,
    SupportHarness,
    authorize_fixture_orders,
    build_support_engine,
)

from agnostic_market.agents.support import flow as support_flow
from agnostic_market.commerce.profile import ProfileStore
from agnostic_market.dtos.events import InterruptEvent, SpokenMessageEvent, TurnFacts
from agnostic_market.dtos.state import PolicyContext

# returnless high (the default): the return tests drive the RETURN door directly.
_POLICY = make_policy()
_FACTS = TurnFacts()
# ORD-1003 was delivered 2026-07-01T00:00:00Z (fixture) = this UTC epoch; window tests
# freeze the flow clock relative to it (fixture dates age against wall clock).
_DELIVERED_EPOCH = 1782864000.0
_DAY = 86400.0


def _return_harness(
    config_root: Path,
    order_key: str = "1",  # ORD-1001, shipped, $179.98
    *,
    policy: PolicyContext = _POLICY,
    thread_id: str = "ret-1",
) -> SupportHarness:
    # Fixture orders pre-authorized (rung-1): this suite pins the returns money logic;
    # the selection gate has its own suite (test_support_scoping.py).
    return authorize_fixture_orders(
        build_support_engine(
            config_root,
            policy=policy,
            reasoning=FakeChatModel(
                force_tool="propose_return",
                canned_args={"propose_return": {"order_key": order_key}},
                tool_call_limit=1,
            ),
            thread_id=thread_id,
        )
    )


async def _events(engine, text: str, facts: TurnFacts = _FACTS) -> list:
    return [e async for e in engine.stream_turn(text, facts)]


def _telemetry_events(tmp_path: Path) -> list[dict]:
    path = tmp_path / "telemetry.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# --- the direct door: propose_return -> readback -> RMA ----------------------------------


async def test_direct_return_creates_one_rma_after_readback(config_root: Path) -> None:
    h = _return_harness(config_root)
    events = await _events(h.engine, "I need to return this order")
    interrupts = [e for e in events if isinstance(e, InterruptEvent)]
    assert len(interrupts) == 1
    # GRAPH-authored readback: order named, CODE-computed refund stated, v1 destination
    # constant spoken — nothing created yet.
    assert "ORD-1001" in interrupts[0].prompt
    assert "$179.98" in interrupts[0].prompt
    assert "original payment method" in interrupts[0].prompt
    assert h.store.return_count == 0
    done = await _events(h.engine, "yes")
    assert h.store.return_count == 1
    assert h.store.refund_count == 0  # the refund RELEASES at Phase 4, never here
    spoken = [e for e in done if isinstance(e, SpokenMessageEvent)]
    assert any("RMA-3001" in e.text and e.node == "support_return_place" for e in spoken)
    assert h.recent_orders.snapshot().focused_order_ref == "ORD-1001"


async def test_stray_turn_after_completion_never_double_creates(config_root: Path) -> None:
    h = _return_harness(config_root)
    await _events(h.engine, "I need to return this order")
    await _events(h.engine, "yes")
    await _events(h.engine, "yes")  # stray extra turn: no interrupt pending, no re-place
    assert h.store.return_count == 1


async def test_barged_readback_reconfirms_before_creating(config_root: Path) -> None:
    h = _return_harness(config_root)
    await _events(h.engine, "I need to return this order")
    events = await _events(h.engine, "yes", TurnFacts(readback_interrupted=True))
    assert h.store.return_count == 0  # §4a: consent over a barged readback is not consent
    reconfirms = [e for e in events if isinstance(e, InterruptEvent)]
    assert len(reconfirms) == 1
    assert "yes or no" in reconfirms[0].prompt.lower()
    await _events(h.engine, "yes")
    assert h.store.return_count == 1


# --- guardrail tiers ----------------------------------------------------------------------


async def test_return_on_unshipped_order_steers_to_cancel(config_root: Path) -> None:
    h = _return_harness(config_root, order_key="2")  # ORD-1002, processing
    events = await _events(h.engine, "I need to return this order")
    spoken = [e for e in events if isinstance(e, SpokenMessageEvent)]
    assert any("nothing to send back" in e.text for e in spoken)  # the steer line
    interrupts = [e for e in events if isinstance(e, InterruptEvent)]
    assert len(interrupts) == 1
    assert "can't be undone" in interrupts[0].prompt  # the CANCEL readback
    await _events(h.engine, "yes")
    assert h.store.order_status("ORD-1002") == "cancelled"
    assert h.store.return_count == 0  # a cancel, not a return


async def test_return_on_cancelled_order_declines(config_root: Path) -> None:
    h = _return_harness(config_root, order_key="2")
    h.store.cancel_order("ck-1", order_id="ORD-1002")
    events = await _events(h.engine, "I need to return this order")
    spoken = [e for e in events if isinstance(e, SpokenMessageEvent)]
    assert any("nothing to return" in e.text for e in spoken)
    assert h.store.return_count == 0
    assert not h.engine.pending_interrupt()


async def test_second_return_names_the_open_rma(config_root: Path) -> None:
    h = _return_harness(config_root)
    first = h.store.create_return(
        "rk-0", order_id="ORD-1001", refund_due_usd=100.0, destination="original"
    )
    events = await _events(h.engine, "I need to return this order")
    spoken = [e for e in events if isinstance(e, SpokenMessageEvent)]
    assert any(first.rma_id in e.text and "already set up" in e.text for e in spoken)
    assert h.store.return_count == 1  # only the pre-existing one


async def test_out_of_window_declines_honestly(
    config_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 40 days after delivery (window 30): no self-serve return at ANY amount.
    monkeypatch.setattr(support_flow.time, "time", lambda: _DELIVERED_EPOCH + 40 * _DAY)
    h = _return_harness(config_root, order_key="3")  # ORD-1003, delivered
    events = await _events(h.engine, "I need to return this order")
    spoken = [e for e in events if isinstance(e, SpokenMessageEvent)]
    assert any("more than 30 days ago" in e.text for e in spoken)
    assert not any(isinstance(e, InterruptEvent) for e in events)
    assert h.store.return_count == 0


async def test_within_window_is_eligible(
    config_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(support_flow.time, "time", lambda: _DELIVERED_EPOCH + 5 * _DAY)
    h = _return_harness(config_root, order_key="3")
    events = await _events(h.engine, "I need to return this order")
    interrupts = [e for e in events if isinstance(e, InterruptEvent)]
    assert len(interrupts) == 1
    assert "ORD-1003" in interrupts[0].prompt


async def test_oversized_return_refund_needs_a_person(config_root: Path) -> None:
    # The refund promise ($179.98) is over the (tightened) human line — same authorization
    # ceiling as the refund flow's amount gate, for DIRECT proposals.
    tight = _POLICY.model_copy(update={"refund_require_human_above_usd": 100.0})
    h = _return_harness(config_root, policy=tight)
    events = await _events(h.engine, "I need to return this order")
    spoken = [e for e in events if isinstance(e, SpokenMessageEvent)]
    assert any("above what I can set up" in e.text for e in spoken)
    assert h.store.return_count == 0
    assert not any(isinstance(e, InterruptEvent) for e in events)


# --- the steered door: refund tier-4 -> the returns path ---------------------------------


async def test_steered_refund_out_of_window_declines_before_promising(
    config_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The ask-then-decline guard (live call #9 P2 class): the steer line must NOT be spoken
    # when the return guardrail would decline — eligibility is checked BEFORE the promise.
    monkeypatch.setattr(support_flow.time, "time", lambda: _DELIVERED_EPOCH + 40 * _DAY)
    tight = _POLICY.model_copy(update={"refund_returnless_under_usd": 10.0})
    h = authorize_fixture_orders(
        build_support_engine(
            config_root,
            policy=tight,
            reasoning=FakeChatModel(
                force_tool="propose_refund",
                canned_args={
                    "propose_refund": {
                        "order_key": "3", "amount_usd": 40.0, "destination": "original"
                    }
                },
                tool_call_limit=1,
            ),
            thread_id="ret-steer-1",
        )
    )
    events = await _events(h.engine, "I want a refund for my socks order")
    spoken = [e for e in events if isinstance(e, SpokenMessageEvent)]
    assert any("outside the 30-day return window" in e.text for e in spoken)
    assert not any("arrange that return for you now" in e.text for e in spoken)
    assert not any(isinstance(e, InterruptEvent) for e in events)
    assert h.store.return_count == 0
    assert h.store.refund_count == 0


async def test_refund_with_open_return_points_at_it(config_root: Path) -> None:
    h = authorize_fixture_orders(
        build_support_engine(
            config_root,
            policy=_POLICY,
            reasoning=FakeChatModel(
                force_tool="propose_refund",
                canned_args={
                    "propose_refund": {
                        "order_key": "1", "amount_usd": 150.0, "destination": "original"
                    }
                },
                tool_call_limit=1,
            ),
            thread_id="ret-open-1",
        )
    )
    existing = h.store.create_return(
        "rk-0", order_id="ORD-1001", refund_due_usd=179.98, destination="original"
    )
    events = await _events(h.engine, "I want a refund to my original card")
    spoken = [e for e in events if isinstance(e, SpokenMessageEvent)]
    assert any(existing.rma_id in e.text and "already set up" in e.text for e in spoken)
    assert h.store.refund_count == 0
    assert not any(isinstance(e, InterruptEvent) for e in events)


# --- exits ---------------------------------------------------------------------------------


async def test_stale_return_readback_expires_before_creating(
    config_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = _return_harness(config_root)
    await _events(h.engine, "I need to return this order")
    future = 9_999_999_999.0  # far past any TTL
    monkeypatch.setattr(support_flow.time, "time", lambda: future)
    events = await _events(h.engine, "yes")
    assert h.store.return_count == 0
    assert not h.engine.pending_interrupt()  # cleared (clear-before-speak)
    spoken = [e for e in events if isinstance(e, SpokenMessageEvent)]
    assert any("sat for a while" in e.text for e in spoken)


async def test_no_at_readback_creates_nothing(config_root: Path) -> None:
    h = _return_harness(config_root)
    await _events(h.engine, "I need to return this order")
    events = await _events(h.engine, "no, leave it")
    assert h.store.return_count == 0
    assert not h.engine.pending_interrupt()
    spoken = [e for e in events if isinstance(e, SpokenMessageEvent)]
    assert any("won't set up a return" in e.text for e in spoken)


async def test_human_at_readback_escapes_with_onramp_package(
    config_root: Path, tmp_path: Path
) -> None:
    h = _return_harness(config_root)
    await _events(h.engine, "I need to return this order")
    events = await _events(h.engine, "just get me a real person")
    assert h.store.return_count == 0
    assert not h.engine.pending_interrupt()  # §A9: never trapped
    spoken = [e for e in events if isinstance(e, SpokenMessageEvent)]
    assert any(e.node == "handover" and "person" in e.text for e in spoken)
    # The warm-transfer context package (Group C on-ramp): closed slugs, exact keys, no PII.
    onramps = [e for e in _telemetry_events(tmp_path) if e.get("event") == "human_onramp"]
    assert len(onramps) == 1
    assert set(onramps[0]) == {
        "event", "schema_version", "tenant", "verification_level",
        "active_flow", "reason_code", "source",
    }
    assert onramps[0]["schema_version"] == 1
    assert onramps[0]["tenant"] == "acme_store"


# --- crossover isolation (the step-up factory shares bodies, never state) ------------------


async def test_refund_stepup_emits_no_profile_events(config_root: Path, tmp_path: Path) -> None:
    h = authorize_fixture_orders(
        build_support_engine(
            config_root,
            policy=_POLICY,
            reasoning=FakeChatModel(
                force_tool="propose_refund",
                canned_args={
                    "propose_refund": {
                        "order_key": "2", "amount_usd": 129.0, "destination": "new_instrument"
                    }
                },
                tool_call_limit=1,
            ),
            thread_id="ret-cross-1",
        )
    )
    await _events(h.engine, "I'd like a refund to a different card")  # pauses at OTP
    await _events(h.engine, TEST_OTP)
    await _events(h.engine, "yes")
    assert h.store.refund_count == 1
    events = _telemetry_events(tmp_path)
    assert any(e.get("event", "").startswith("refund_stepup") for e in events)
    assert not any(e.get("event", "").startswith("profile_") for e in events)
    assert h.profile.change_count == 0
    assert isinstance(h.profile, ProfileStore)  # untouched profile store, wrong-family-proof


def test_render_orders_marks_the_pointed_order() -> None:
    # The support model's order list marks the recent focused order so a bare "that order"
    # resolves by REFERENCE, not by conversational salience (live call #10 L4).
    from agnostic_market.agents.support.prompt import render_orders
    from agnostic_market.commerce.orders import OrderCandidate

    orders = [
        OrderCandidate(key="1", order_id="ORD-1001", summary="shoes", total_usd=179.98,
                       status="shipped"),
        OrderCandidate(key="2", order_id="ORD-1002", summary="jacket", total_usd=129.0,
                       status="processing"),
    ]
    marked = render_orders(orders, "ORD-1002")
    assert "ORD-1002 - jacket ($129.00, processing) - the order most recently discussed" in marked
    assert "ORD-1001 - shoes ($179.98, shipped)\n" in marked  # unmarked line untouched
    assert "most recently discussed" not in render_orders(orders)  # no context -> no mark
