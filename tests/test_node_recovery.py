"""Milestone 6C-b ordinary node-failure recovery contracts."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
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
    clear_automation_state,
)
from agnostic_market.agents.support import _stepup as support_stepup
from agnostic_market.commerce.orders import render_cart_line
from agnostic_market.dtos.events import InterruptEvent, SpokenMessageEvent
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


def _telemetry_records() -> list[dict[str, object]]:
    if not telemetry._TELEMETRY_PATH.exists():
        return []
    return [
        json.loads(line)
        for line in telemetry._TELEMETRY_PATH.read_text(encoding="utf-8").splitlines()
    ]


async def _events(engine, text: str) -> list:
    return await engine_events(engine, text)


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
    harness = build_support_engine(
        config_root,
        policy=make_policy(),
        thread_id="recovery-serde",
    )
    marker = PendingRecovery(
        origin_node="model",
        action=ExceptionAction.SAFE_ABORT,
        trigger="node_exception",
    )

    with caplog.at_level("WARNING", logger="langgraph.checkpoint.serde.jsonplus"):
        harness.engine._graph.update_state(
            harness.engine._config,
            {"pending_recovery": marker},
            as_node="__start__",
        )
        restored = harness.engine._graph.get_state(harness.engine._config)

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

    events = await _events(harness.engine, "continue")
    snapshot = harness.engine._graph.get_state(harness.engine._config)

    assert harness.otp.dispatch_count == 0
    assert not any(isinstance(event, InterruptEvent) for event in events)
    assert [event.text for event in events if isinstance(event, SpokenMessageEvent)] == [
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

    events = await _events(harness.engine, "continue")
    snapshot = harness.engine._graph.get_state(harness.engine._config)

    assert harness.otp.dispatch_count == 1
    assert harness.verification.current_level() == 1
    assert not any(isinstance(event, InterruptEvent) for event in events)
    assert [event.text for event in events if isinstance(event, SpokenMessageEvent)] == [
        "That refund request did not complete. Please try it again."
    ]
    assert snapshot.values.get("pending_refund") is None
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
