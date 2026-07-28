"""Milestone 6C-b ordinary node-failure recovery contracts."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from langgraph.types import Command
from llm_fakes import FakeChatModel
from policy_helpers import make_policy
from pydantic import ValidationError
from support_helpers import build_support_engine
from turn_helpers import engine_events

from agnostic_market.agents import telemetry
from agnostic_market.agents.cart import flow as cart_flow
from agnostic_market.agents.frontline import graph as frontline_graph
from agnostic_market.agents.recovery import (
    AUTOMATION_TERMINAL_LINE,
    RECOVERY_NODE_NAME,
    TURN_FALLBACK_LINE,
    NodeExecutionTracker,
    clear_automation_state,
)
from agnostic_market.agents.support import _stepup as support_stepup
from agnostic_market.commerce.orders import render_cart_line
from agnostic_market.dtos.events import SpokenMessageEvent
from agnostic_market.dtos.orchestration import ListOrders
from agnostic_market.dtos.recovery import ExceptionAction, PendingRecovery
from agnostic_market.dtos.state import PendingRefund, ReasoningState

_ACTION_CASES = (
    ("model", ExceptionAction.SAFE_ABORT, TURN_FALLBACK_LINE),
    ("cart_assemble", ExceptionAction.CART_REVIEW, None),
    (
        "principal_warning",
        ExceptionAction.ABORT_PRINCIPAL_WARNING,
        "The account verification or switch confirmation did not complete. "
        "Please try that request again.",
    ),
    (
        "cart_confirm",
        ExceptionAction.ABORT_PLACEMENT_CONFIRMATION,
        "That order placement request did not complete. Your cart is still saved for review.",
    ),
    (
        "support_dispatch",
        ExceptionAction.ABORT_REFUND_VERIFICATION,
        "That refund request did not complete. Please try it again.",
    ),
    (
        "support_confirm",
        ExceptionAction.ABORT_REFUND_CONFIRMATION,
        "That refund request did not complete. Please try it again.",
    ),
    (
        "support_cancel_confirm",
        ExceptionAction.ABORT_CANCEL_CONFIRMATION,
        "That cancellation request did not complete. Your orders are unchanged.",
    ),
    (
        "support_return_confirm",
        ExceptionAction.ABORT_RETURN_CONFIRMATION,
        "That return request did not complete. No return was created.",
    ),
    (
        "support_profile_dispatch",
        ExceptionAction.ABORT_PROFILE_VERIFICATION,
        "That profile update did not complete. Your profile is unchanged.",
    ),
    (
        "support_profile_confirm",
        ExceptionAction.ABORT_PROFILE_CONFIRMATION,
        "That profile update did not complete. Your profile is unchanged.",
    ),
    (
        "identity_dispatch",
        ExceptionAction.ABORT_IDENTITY_VERIFICATION,
        "Account verification did not complete. Please try that request again.",
    ),
    ("handover", ExceptionAction.TERMINAL, AUTOMATION_TERMINAL_LINE),
)


def test_full_idle_waits_for_turn_span_across_mutable_zero_crossings() -> None:
    tracker = NodeExecutionTracker()
    callbacks: list[str] = []

    with tracker.turn_span() as admitted:
        assert admitted is True
        tracker.defer_until_fully_idle(lambda: callbacks.append("closed"))
        tracker._run("first", lambda: None)
        assert callbacks == []
        tracker._run("second", lambda: None)
        assert callbacks == []

    assert callbacks == ["closed"]


def test_mutable_idle_does_not_wait_on_the_owning_turn_span() -> None:
    tracker = NodeExecutionTracker()

    with tracker.turn_span() as admitted:
        assert admitted is True
        assert tracker.wait_until_mutable_idle(0.01) is True


def test_stopped_turn_admission_rejects_without_changing_full_idle() -> None:
    tracker = NodeExecutionTracker()
    tracker.stop_turn_admission()

    with tracker.turn_span() as admitted:
        assert admitted is False

    callbacks: list[str] = []
    tracker.defer_until_fully_idle(lambda: callbacks.append("closed"))
    assert callbacks == ["closed"]


def _telemetry_records() -> list[dict[str, object]]:
    if not telemetry._TELEMETRY_PATH.exists():
        return []
    return [
        json.loads(line)
        for line in telemetry._TELEMETRY_PATH.read_text(encoding="utf-8").splitlines()
    ]


async def _events(engine, text: str) -> list:
    return await engine_events(engine, text)


async def _run_seeded_continuation(harness) -> None:
    async for _update in harness.engine._graph.astream(
        Command(update={"consumed_turn_ids": ()}),
        harness.engine._config,
        stream_mode="updates",
    ):
        pass


def _commerce_counts(harness) -> tuple[int, int, int, int, int]:
    return (
        harness.store.placed_count,
        harness.store.refund_count,
        harness.store.cancel_count,
        harness.store.return_count,
        harness.profile.change_count,
    )


@pytest.mark.parametrize(("origin", "action", "expected_line"), _ACTION_CASES)
def test_every_ordinary_recovery_action_has_one_closed_result(
    config_root: Path,
    origin: str,
    action: ExceptionAction,
    expected_line: str | None,
) -> None:
    frontline = FakeChatModel(emit_tool_calls=False)
    reasoning = FakeChatModel(emit_tool_calls=False)
    harness = build_support_engine(
        config_root,
        policy=make_policy(),
        frontline=frontline,
        reasoning=reasoning,
        thread_id=f"recover-{action}",
    )
    cart = harness.caller_context.cart_store
    if action == ExceptionAction.CART_REVIEW:
        cart.add_item(
            sku="SKU-BLU-07",
            name="blue wool hat",
            price_usd=29.0,
            quantity=1,
        )
        expected_line = (
            f"{render_cart_line(cart.view(), cart.cart_total())} "
            "Please review your cart before trying checkout again."
        )
    before_counts = _commerce_counts(harness)
    before_cart = cart.view()
    state = ReasoningState(
        pending_recovery=PendingRecovery(
            origin_node=origin,
            action=action,
            trigger="node_exception",
        ),
        active_flow="support",
        pending_ack="stale",
        identity_claim_misses=1,
    )

    update = harness.engine._graph.nodes[RECOVERY_NODE_NAME].invoke(state)

    for field, value in clear_automation_state().items():
        assert update[field] == value
    assert update["messages"] == [AIMessage(expected_line)]
    assert update.get("automation_terminal", False) is (action == ExceptionAction.TERMINAL)
    assert _commerce_counts(harness) == before_counts
    assert cart.view() == before_cart
    assert frontline.invoke_count == 0
    assert reasoning.invoke_count == 0
    assert [record for record in _telemetry_records() if record["event"] == "turn_failed"] == [
        {
            "event": "turn_failed",
            "reason": "node_exception",
            "node": origin,
            "action": action,
        }
    ]


@pytest.mark.parametrize(
    "marker",
    (
        None,
        PendingRecovery(
            origin_node="unknown_node",
            action=ExceptionAction.SAFE_ABORT,
            trigger="node_exception",
        ),
        PendingRecovery(
            origin_node="model",
            action=ExceptionAction.CART_REVIEW,
            trigger="node_exception",
        ),
        PendingRecovery(
            origin_node="model",
            action=ExceptionAction.SAFE_ABORT,
            trigger="stream_cancelled",
            abandoned_message_id="cancelled-turn",
        ),
        {"origin_node": "model", "action": "safe_abort", "trigger": "node_exception"},
    ),
)
def test_invalid_recovery_markers_fail_terminal_without_echoing_marker_data(
    config_root: Path,
    marker: object,
) -> None:
    harness = build_support_engine(
        config_root,
        policy=make_policy(),
        thread_id="invalid-recovery",
    )
    state = ReasoningState.model_construct(pending_recovery=marker)

    update = harness.engine._graph.nodes[RECOVERY_NODE_NAME].invoke(state)

    assert update["automation_terminal"] is True
    assert update["pending_recovery"] is None
    assert update["messages"] == [AIMessage(AUTOMATION_TERMINAL_LINE)]
    assert _telemetry_records() == [
        {
            "event": "turn_failed",
            "reason": "recovery_contract_invalid",
            "action": "terminal",
        }
    ]


def test_pending_recovery_is_strict_and_checkpoint_safe(
    config_root: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with pytest.raises(ValidationError):
        PendingRecovery.model_validate(
            {
                "origin_node": "model",
                "action": "safe_abort",
                "trigger": "node_exception",
                "exception_text": "must not be checkpointed",
            }
        )
    with pytest.raises(ValidationError, match="requires"):
        PendingRecovery(
            origin_node="model",
            action=ExceptionAction.SAFE_ABORT,
            trigger="stream_cancelled",
        )
    with pytest.raises(ValidationError, match="nonblank"):
        PendingRecovery(
            origin_node="model",
            action=ExceptionAction.SAFE_ABORT,
            trigger="stream_cancelled",
            abandoned_message_id=" ",
        )
    with pytest.raises(ValidationError, match="forbids"):
        PendingRecovery(
            origin_node="model",
            action=ExceptionAction.SAFE_ABORT,
            trigger="node_exception",
            abandoned_message_id="unexpected-turn",
        )
    harness = build_support_engine(
        config_root,
        policy=make_policy(),
        thread_id="recovery-serde",
    )
    markers = (
        PendingRecovery(
            origin_node="model",
            action=ExceptionAction.SAFE_ABORT,
            trigger="node_exception",
        ),
        PendingRecovery(
            origin_node="model",
            action=ExceptionAction.SAFE_ABORT,
            trigger="stream_cancelled",
            abandoned_message_id="transport-turn",
        ),
    )

    with caplog.at_level("WARNING", logger="langgraph.checkpoint.serde.jsonplus"):
        for index, marker in enumerate(markers):
            config = {"configurable": {"thread_id": f"recovery-serde-{index}"}}
            harness.engine._graph.update_state(
                config,
                {"pending_recovery": marker},
                as_node="__start__",
            )
            restored = harness.engine._graph.get_state(config)
            assert restored.values["pending_recovery"] == marker

    assert not any("unregistered" in record.getMessage().lower() for record in caplog.records)


async def test_seeded_recovery_precedes_a_pending_continuation_and_clears_it(
    config_root: Path,
) -> None:
    frontline = FakeChatModel(emit_tool_calls=False)
    reasoning = FakeChatModel(emit_tool_calls=False)
    harness = build_support_engine(
        config_root,
        policy=make_policy(),
        frontline=frontline,
        reasoning=reasoning,
        thread_id="recovery-route-precedence",
    )
    harness.engine._graph.update_state(
        harness.engine._config,
        {
            "pending_recovery": PendingRecovery(
                origin_node="model",
                action=ExceptionAction.SAFE_ABORT,
                trigger="node_exception",
            ),
            "pending_request": ListOrders(scope="account"),
        },
        as_node="__start__",
    )

    events = await _events(harness.engine, "continue")
    snapshot = harness.engine._graph.get_state(harness.engine._config)

    assert [event.text for event in events if isinstance(event, SpokenMessageEvent)] == [
        TURN_FALLBACK_LINE
    ]
    assert frontline.invoke_count == 0
    assert reasoning.invoke_count == 0
    assert snapshot.values.get("pending_recovery") is None
    assert snapshot.values.get("pending_request") is None
    assert snapshot.next == ()


async def test_cart_mutation_then_exception_preserves_live_cart_and_requires_review(
    config_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reasoning = FakeChatModel(
        force_tool="add_to_cart",
        canned_args={"add_to_cart": {"candidate_key": "2", "quantity": 1}},
        tool_call_limit=1,
    )
    harness = build_support_engine(
        config_root,
        policy=make_policy(),
        reasoning=reasoning,
        thread_id="cart-review-after-write",
    )
    cart = harness.caller_context.cart_store
    original_add = cart.add_item

    def write_then_fail(**kwargs):
        original_add(**kwargs)
        raise RuntimeError("simulated failure after reversible cart write")

    monkeypatch.setattr(cart, "add_item", write_then_fail)

    events = await _events(harness.engine, "checkout now please")
    spoken = [event for event in events if isinstance(event, SpokenMessageEvent)]
    snapshot = harness.engine._graph.get_state(harness.engine._config)

    assert cart.line_count == 1
    assert harness.store.placed_count == 0
    assert len(spoken) == 1 and spoken[0].node == RECOVERY_NODE_NAME
    assert "waterproof rain jacket" in spoken[0].text
    assert "review your cart" in spoken[0].text.lower()
    assert snapshot.values.get("active_flow") is None
    assert snapshot.values.get("pending_recovery") is None
    assert snapshot.next == ()


async def test_confirmation_exception_aborts_placement_but_preserves_cart(
    config_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reasoning = FakeChatModel(
        force_tool="buy_now",
        canned_args={"buy_now": {"candidate_key": "2", "quantity": 1}},
        tool_call_limit=1,
    )
    harness = build_support_engine(
        config_root,
        policy=make_policy(),
        reasoning=reasoning,
        thread_id="placement-confirm-failure",
    )

    def fail_confirmation(_prompt: str):
        raise RuntimeError("simulated confirmation failure")

    monkeypatch.setattr(cart_flow, "interrupt", fail_confirmation)

    events = await _events(harness.engine, "checkout now please")
    spoken = [event for event in events if isinstance(event, SpokenMessageEvent)]
    snapshot = harness.engine._graph.get_state(harness.engine._config)

    assert harness.store.placed_count == 0
    assert harness.caller_context.cart_store.line_count == 1
    assert [event.text for event in spoken] == [
        "That order placement request did not complete. Your cart is still saved for review."
    ]
    assert snapshot.values.get("pending_placement") is None
    assert snapshot.next == ()


def _pending_refund(harness) -> PendingRefund:
    instrument_ref = harness.payment_instruments.new_instrument_ref("CUST-002")
    assert instrument_ref is not None
    return PendingRefund(
        order_id="ORD-1002",
        summary="a waterproof rain jacket",
        amount_usd=129.0,
        destination="new_instrument",
        instrument_ref=instrument_ref,
        idempotency_key="refund-recovery",
        attempt_key="otp-recovery",
        created_at=time.time(),
    )


async def test_otp_dispatch_exception_does_not_retry_or_enter_collection(
    config_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = build_support_engine(
        config_root,
        policy=make_policy(),
        thread_id="otp-dispatch-failure",
    )
    harness.engine._graph.update_state(
        harness.engine._config,
        {"pending_refund": _pending_refund(harness), "active_flow": "support"},
        as_node="support_risk_check",
    )

    def fail_dispatch(_attempt_key: str) -> None:
        raise RuntimeError("simulated OTP provider failure")

    monkeypatch.setattr(harness.otp, "dispatch", fail_dispatch)

    await _run_seeded_continuation(harness)
    snapshot = harness.engine._graph.get_state(harness.engine._config)

    assert harness.otp.dispatch_count == 0
    assert not snapshot.interrupts
    assert [str(message.content) for message in snapshot.values["messages"][-1:]] == [
        "That refund request did not complete. Please try it again."
    ]
    assert snapshot.values.get("pending_refund") is None
    assert snapshot.next == ()


async def test_otp_collect_exception_does_not_verify_or_redispatch(
    config_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = build_support_engine(
        config_root,
        policy=make_policy(),
        thread_id="otp-collect-failure",
    )
    harness.engine._graph.update_state(
        harness.engine._config,
        {"pending_refund": _pending_refund(harness), "active_flow": "support"},
        as_node="support_risk_check",
    )

    def fail_collection(_prompt: str):
        raise RuntimeError("simulated OTP collection failure")

    monkeypatch.setattr(support_stepup, "interrupt", fail_collection)

    await _run_seeded_continuation(harness)
    snapshot = harness.engine._graph.get_state(harness.engine._config)

    assert harness.otp.dispatch_count == 1
    assert harness.verification.current_level() == 1
    assert not snapshot.interrupts
    assert [str(message.content) for message in snapshot.values["messages"][-1:]] == [
        "That refund request did not complete. Please try it again."
    ]
    assert snapshot.values.get("pending_refund") is None
    assert snapshot.next == ()


def _identity_verification_harness(config_root: Path, *, thread_id: str):
    return build_support_engine(
        config_root,
        policy=make_policy(),
        frontline=FakeChatModel(
            force_tool="request_handover",
            canned_args={
                "request_handover": {
                    "destination": "support",
                    "reason_code": "list_orders",
                }
            },
            tool_call_limit=99,
        ),
        reasoning=FakeChatModel(
            force_tool="propose_identity",
            canned_args={"propose_identity": {"contact_claim": "+1 555 010 0119"}},
            tool_call_limit=99,
        ),
        thread_id=thread_id,
    )


async def test_cancelled_otp_dispatch_is_not_replayed_by_recovery(
    config_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _identity_verification_harness(
        config_root,
        thread_id="cancelled-otp-dispatch",
    )
    entered = threading.Event()
    release = threading.Event()
    real_dispatch = harness.otp.dispatch

    def dispatch_then_pause(attempt_key: str) -> None:
        real_dispatch(attempt_key)
        entered.set()
        if not release.wait(timeout=5):
            raise RuntimeError("test did not release OTP dispatch")

    monkeypatch.setattr(harness.otp, "dispatch", dispatch_then_pause)
    dispatching = asyncio.create_task(_events(harness.engine, "list my account orders"))
    assert await asyncio.to_thread(entered.wait, 5)

    dispatching.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await dispatching
    marker = harness.engine._graph.get_state(harness.engine._config).values["pending_recovery"]
    assert marker.origin_node == "identity_dispatch"
    assert marker.trigger == "stream_cancelled"

    recovered = await _events(harness.engine, "continue")
    snapshot = harness.engine._graph.get_state(harness.engine._config)

    assert [event.text for event in recovered if isinstance(event, SpokenMessageEvent)] == [
        "Account verification did not complete. Please try that request again."
    ]
    assert harness.otp.dispatch_count == 1
    assert harness.verification.current_level() == 1
    assert harness.verification.grants == []
    assert harness.identity.current() is None
    assert _commerce_counts(harness) == (0, 0, 0, 0, 0)
    assert snapshot.values.get("pending_recovery") is None
    assert snapshot.values.get("pending_identity") is None
    assert snapshot.next == ()


async def test_cancelled_otp_collect_does_not_reverify_bind_or_resume_the_action(
    config_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _identity_verification_harness(
        config_root,
        thread_id="cancelled-otp-collect",
    )
    await _events(harness.engine, "list my account orders")
    assert harness.engine.pending_interrupt()
    assert harness.otp.dispatch_count == 1
    entered = threading.Event()
    release = threading.Event()
    verify_calls = 0
    real_verify = harness.verification.verify_otp

    def verify_then_pause(code: str) -> bool:
        nonlocal verify_calls
        verify_calls += 1
        verified = real_verify(code)
        entered.set()
        if not release.wait(timeout=5):
            raise RuntimeError("test did not release OTP verification")
        return verified

    monkeypatch.setattr(harness.verification, "verify_otp", verify_then_pause)
    collecting = asyncio.create_task(_events(harness.engine, "482913"))
    assert await asyncio.to_thread(entered.wait, 5)

    collecting.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await collecting
    abandoned = harness.engine._graph.get_state(harness.engine._config)
    marker = abandoned.values.get("pending_recovery")
    assert isinstance(marker, PendingRecovery), abandoned
    assert marker.origin_node == "identity_collect"
    assert marker.trigger == "stream_cancelled"

    recovered = await _events(harness.engine, "continue")
    snapshot = harness.engine._graph.get_state(harness.engine._config)

    assert [event.text for event in recovered if isinstance(event, SpokenMessageEvent)] == [
        "Account verification did not complete. Please try that request again."
    ]
    assert verify_calls == 1
    assert harness.otp.dispatch_count == 1
    assert harness.verification.current_level() == 2
    assert len(harness.verification.grants) == 1
    assert harness.identity.current() is None
    assert _commerce_counts(harness) == (0, 0, 0, 0, 0)
    assert snapshot.values.get("pending_recovery") is None
    assert snapshot.values.get("pending_identity") is None
    assert snapshot.values.get("pending_request") is None
    assert snapshot.next == ()


async def test_terminal_node_exception_stays_terminal_without_reentry(
    config_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = build_support_engine(
        config_root,
        policy=make_policy(),
        thread_id="terminal-node-failure",
    )
    harness.engine._graph.update_state(
        harness.engine._config,
        {"automation_terminal": True},
        as_node="__start__",
    )

    def fail_terminal_event(_record: dict[str, object]) -> None:
        raise RuntimeError("simulated terminal-node failure")

    monkeypatch.setattr(frontline_graph, "write_event", fail_terminal_event)

    events = await _events(harness.engine, "continue")
    snapshot = harness.engine._graph.get_state(harness.engine._config)

    assert [event.text for event in events if isinstance(event, SpokenMessageEvent)] == [
        AUTOMATION_TERMINAL_LINE
    ]
    assert snapshot.values["automation_terminal"] is True
    assert snapshot.values.get("pending_recovery") is None
    assert snapshot.next == ()
