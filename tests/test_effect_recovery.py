"""Milestone 6C-c authoritative commerce-effect reconciliation contracts."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command
from policy_helpers import make_policy
from support_helpers import SupportHarness, build_support_engine
from turn_helpers import engine_events

from agnostic_market.agents import telemetry
from agnostic_market.agents.recovery import (
    AUTOMATION_TERMINAL_LINE,
    RECOVERY_NODE_NAME,
)
from agnostic_market.commerce.identity import BoundIdentity
from agnostic_market.commerce.receipts import CommittedReceipt, IndeterminateReceipt
from agnostic_market.dtos.events import SpokenMessageEvent
from agnostic_market.dtos.recovery import ExceptionAction, PendingRecovery
from agnostic_market.dtos.state import (
    CHECKPOINT_SCHEMA_VERSION,
    BatchCancelOutcome,
    CancelTarget,
    CartLine,
    PendingCancelBatch,
    PendingPlacement,
    PendingProfileChange,
    PendingRefund,
    PendingReturn,
    ReasoningState,
)

EffectName = Literal["placement", "refund", "cancel", "return", "profile"]
_EFFECTS: tuple[EffectName, ...] = ("placement", "refund", "cancel", "return", "profile")
_OWNER = "CUST-001"
_WAIT_TIMEOUT_SECONDS = 5.0
_RECEIPT_NAMES: dict[EffectName, str] = {
    "placement": "placement_receipt",
    "refund": "refund_receipt",
    "cancel": "cancel_receipt",
    "return": "return_receipt",
    "profile": "profile_change_receipt",
}


@dataclass(frozen=True)
class _EffectCase:
    pending_field: str
    pending: object
    predecessor: str
    origin_node: str
    owner: object
    mutator_name: str
    count_name: str
    action: ExceptionAction
    success_event: str
    committed_text: str
    not_committed_text: str


def _records() -> list[dict[str, object]]:
    if not telemetry._TELEMETRY_PATH.exists():
        return []
    return [
        json.loads(line)
        for line in telemetry._TELEMETRY_PATH.read_text(encoding="utf-8").splitlines()
    ]


async def _run_seeded_effect(harness: SupportHarness) -> list[str]:
    before = harness.engine._graph.get_state(harness.engine._config)
    message_count = len(before.values.get("messages", ()))
    async for _update in harness.engine._graph.astream(
        Command(update={"consumed_turn_ids": ()}),
        harness.engine._config,
        stream_mode="updates",
    ):
        pass
    after = harness.engine._graph.get_state(harness.engine._config)
    return [
        str(message.content)
        for message in after.values.get("messages", ())[message_count:]
        if isinstance(message, AIMessage)
    ]


def _case(harness: SupportHarness, effect: EffectName) -> _EffectCase:
    now = time.time()
    if effect == "placement":
        line = harness.caller_context.cart_store.add_item(
            sku="SKU-BLU-07",
            name="waterproof rain jacket",
            price_usd=129.0,
            quantity=1,
        )
        return _EffectCase(
            pending_field="pending_placement",
            pending=PendingPlacement(
                lines=(line,),
                total_usd=129.0,
                idempotency_key="recover-placement",
                created_at=now,
            ),
            predecessor="cart_confirm",
            origin_node="cart_place",
            owner=harness.store,
            mutator_name="place_cart",
            count_name="placed_count",
            action=ExceptionAction.RECONCILE_PLACEMENT,
            success_event="checkout_confirmed",
            committed_text="ORD-9001",
            not_committed_text=(
                "That order placement request did not complete. "
                "Your cart is still saved for review."
            ),
        )
    if effect == "refund":
        return _EffectCase(
            pending_field="pending_refund",
            pending=PendingRefund(
                order_id="ORD-1002",
                summary="1 waterproof rain jacket",
                amount_usd=40.0,
                destination="original",
                instrument_ref="original payment method",
                idempotency_key="recover-refund",
                attempt_key="recover-refund-otp",
                created_at=now,
            ),
            predecessor="support_confirm",
            origin_node="support_place",
            owner=harness.store,
            mutator_name="issue_refund",
            count_name="refund_count",
            action=ExceptionAction.RECONCILE_REFUND,
            success_event="refund_confirmed",
            committed_text="R-7001",
            not_committed_text="That refund request did not complete.",
        )
    if effect == "cancel":
        return _EffectCase(
            pending_field="pending_cancel",
            pending=PendingCancelBatch(
                targets=(
                    CancelTarget(
                        order_id="ORD-1002",
                        summary="1 waterproof rain jacket",
                        idempotency_key="recover-cancel",
                    ),
                ),
                created_at=now,
            ),
            predecessor="support_cancel_confirm",
            origin_node="support_cancel_void",
            owner=harness.store,
            mutator_name="cancel_order",
            count_name="cancel_count",
            action=ExceptionAction.RECONCILE_CANCEL,
            success_event="cancel_confirmed",
            committed_text="cancelled",
            not_committed_text=(
                "The cancellation request for your order for 1 waterproof rain jacket "
                "(ORD-1002) did not complete."
            ),
        )
    if effect == "return":
        return _EffectCase(
            pending_field="pending_return",
            pending=PendingReturn(
                order_id="ORD-1001",
                summary="wrong pending summary",
                refund_due_usd=100.0,
                idempotency_key="recover-return",
                created_at=now,
            ),
            predecessor="support_return_confirm",
            origin_node="support_return_place",
            owner=harness.store,
            mutator_name="create_return",
            count_name="return_count",
            action=ExceptionAction.RECONCILE_RETURN,
            success_event="return_confirmed",
            committed_text="2 pairs of trail running shoes",
            not_committed_text="That return request did not complete.",
        )

    harness.identity.bind(BoundIdentity(customer_ref=_OWNER, masked_contact="number ending 0119"))
    assert harness.verification.verify_otp("482913")
    return _EffectCase(
        pending_field="pending_profile_change",
        pending=PendingProfileChange(
            customer_ref=_OWNER,
            field="address",
            new_value="7 Elm Street",
            factor_ref="number ending 0119",
            idempotency_key="recover-profile",
            attempt_key="recover-profile-otp",
            created_at=now,
        ),
        predecessor="support_profile_confirm",
        origin_node="support_profile_place",
        owner=harness.profile,
        mutator_name="update_profile",
        count_name="change_count",
        action=ExceptionAction.RECONCILE_PROFILE_CHANGE,
        success_event="profile_change_confirmed",
        committed_text="7 Elm Street",
        not_committed_text="That profile update request did not complete.",
    )


def _seed(
    harness: SupportHarness,
    case: _EffectCase,
    *,
    consumed_turn_ids: tuple[str, ...] = (),
) -> None:
    harness.engine._graph.update_state(
        harness.engine._config,
        {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            case.pending_field: case.pending,
            "consumed_turn_ids": consumed_turn_ids,
            "execution_owner": "cart"
            if case.action == (ExceptionAction.RECONCILE_PLACEMENT)
            else "support",
        },
        as_node=case.predecessor,
    )


def _effect_count(case: _EffectCase) -> int:
    return int(getattr(case.owner, case.count_name))


@pytest.mark.parametrize("effect", _EFFECTS)
@pytest.mark.parametrize("commits_before_cancellation", (False, True))
async def test_external_cancellation_reconciles_every_effect_without_replaying_mutator(
    config_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    effect: EffectName,
    commits_before_cancellation: bool,
) -> None:
    harness = build_support_engine(
        config_root,
        policy=make_policy(),
        thread_id=f"{effect}-cancel-{'post' if commits_before_cancellation else 'pre'}",
    )
    case = _case(harness, effect)
    original = getattr(case.owner, case.mutator_name)
    entered = threading.Event()
    release = threading.Event()
    calls = 0

    def blocked_effect(*args, **kwargs):
        nonlocal calls
        calls += 1
        if commits_before_cancellation:
            result = original(*args, **kwargs)
            entered.set()
            if not release.wait(timeout=_WAIT_TIMEOUT_SECONDS):
                raise TimeoutError("test did not release committed effect")
            return result
        entered.set()
        if not release.wait(timeout=_WAIT_TIMEOUT_SECONDS):
            raise TimeoutError("test did not release uncommitted effect")
        raise RuntimeError("simulated cancellation before effect commit")

    monkeypatch.setattr(case.owner, case.mutator_name, blocked_effect)
    abandoned_message_id = f"{effect}-cancelled-turn"
    _seed(
        harness,
        case,
        consumed_turn_ids=(abandoned_message_id,),
    )

    async def run_effect() -> None:
        async for _update in harness.engine._graph.astream(
            Command(update={"consumed_turn_ids": ()}),
            harness.engine._config,
            stream_mode="updates",
        ):
            pass

    abandoned = asyncio.create_task(run_effect())
    assert await asyncio.to_thread(entered.wait, _WAIT_TIMEOUT_SECONDS)
    abandoned.cancel()
    with pytest.raises(asyncio.CancelledError):
        await abandoned
    active_nodes = harness.engine._node_execution_tracker.active_node_names
    takeover = asyncio.create_task(
        harness.engine._take_over_cancelled_stream(
            owned_thread_id=harness.engine.thread_id,
            abandoned_message_id=abandoned_message_id,
            active_node_names=active_nodes,
            resumed_interrupt_node=None,
        )
    )
    release.set()
    await asyncio.wait_for(takeover, timeout=_WAIT_TIMEOUT_SECONDS)

    seeded = harness.engine._graph.get_state(harness.engine._config)
    marker = seeded.values.get("pending_recovery")
    assert isinstance(marker, PendingRecovery)
    assert marker.trigger == "stream_cancelled"
    assert marker.origin_node == case.origin_node
    assert marker.action == case.action

    events = await engine_events(harness.engine, "premature follow-up")
    snapshot = harness.engine._graph.get_state(harness.engine._config)
    spoken = [event.text for event in events if isinstance(event, SpokenMessageEvent)]
    human_texts = [
        str(message.content)
        for message in snapshot.values.get("messages", ())
        if isinstance(message, HumanMessage)
    ]

    assert calls == 1
    assert _effect_count(case) == int(commits_before_cancellation)
    expected = case.committed_text if commits_before_cancellation else case.not_committed_text
    assert len(spoken) == 1 and expected in spoken[0]
    assert "premature follow-up" not in human_texts
    assert snapshot.values.get(case.pending_field) is None
    assert snapshot.values.get("pending_recovery") is None
    assert snapshot.next == ()

    failures = [record for record in _records() if record["event"] == "turn_failed"]
    assert failures == [
        {
            "event": "turn_failed",
            "reason": "stream_cancelled",
            "node": case.origin_node,
            "action": case.action,
        }
    ]


@pytest.mark.parametrize("effect", _EFFECTS)
@pytest.mark.parametrize("commits_before_failure", (False, True))
async def test_effect_failure_reconciles_from_receipt_without_replaying_mutator(
    config_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    effect: EffectName,
    commits_before_failure: bool,
) -> None:
    harness = build_support_engine(
        config_root,
        policy=make_policy(),
        thread_id=f"{effect}-{'post' if commits_before_failure else 'pre'}-effect",
    )
    case = _case(harness, effect)
    original = getattr(case.owner, case.mutator_name)
    calls = 0

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if commits_before_failure:
            original(*args, **kwargs)
        raise RuntimeError("simulated effect boundary failure")

    monkeypatch.setattr(case.owner, case.mutator_name, fail_once)
    _seed(harness, case)

    spoken = await _run_seeded_effect(harness)
    snapshot = harness.engine._graph.get_state(harness.engine._config)

    assert calls == 1
    assert _effect_count(case) == int(commits_before_failure)
    assert len(spoken) == 1
    expected = case.committed_text if commits_before_failure else case.not_committed_text
    assert expected in spoken[0]
    if effect == "return" and commits_before_failure:
        assert "wrong pending summary" not in spoken[0]
    assert snapshot.values.get(case.pending_field) is None
    assert snapshot.values.get("pending_recovery") is None
    assert snapshot.next == ()

    records = _records()
    failures = [record for record in records if record["event"] == "turn_failed"]
    successes = [record for record in records if record["event"] == case.success_event]
    assert failures == [
        {
            "event": "turn_failed",
            "reason": "node_exception",
            "node": case.origin_node,
            "action": case.action,
        }
    ]
    assert len(successes) == int(commits_before_failure)
    if effect == "profile":
        assert "7 Elm Street" not in json.dumps(records)


@pytest.mark.parametrize("effect", ("placement", "refund", "cancel", "return"))
async def test_post_commit_projection_failure_reuses_finisher_and_logs_success_once(
    config_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    effect: EffectName,
) -> None:
    harness = build_support_engine(
        config_root,
        policy=make_policy(),
        thread_id=f"{effect}-projection-recovery",
    )
    case = _case(harness, effect)
    original_record = harness.recent_orders.record
    projection_calls = 0

    def fail_first_projection(*args, **kwargs):
        nonlocal projection_calls
        projection_calls += 1
        if projection_calls == 1:
            raise RuntimeError("simulated post-commit projection failure")
        return original_record(*args, **kwargs)

    monkeypatch.setattr(harness.recent_orders, "record", fail_first_projection)
    _seed(harness, case)

    spoken = await _run_seeded_effect(harness)

    assert projection_calls == 2
    assert _effect_count(case) == 1
    if effect == "placement":
        assert harness.caller_context.cart_store.is_empty()
    assert any(case.committed_text in text for text in spoken)
    assert len([record for record in _records() if record["event"] == case.success_event]) == 1


@pytest.mark.parametrize("effect", _EFFECTS)
@pytest.mark.parametrize("reason", ("key_conflict", "pending", "unavailable"))
def test_indeterminate_receipt_terminalizes_without_an_effect_claim(
    config_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    effect: EffectName,
    reason: str,
) -> None:
    harness = build_support_engine(
        config_root,
        policy=make_policy(),
        thread_id=f"{effect}-indeterminate-{reason}",
    )
    case = _case(harness, effect)
    monkeypatch.setattr(
        case.owner,
        _RECEIPT_NAMES[effect],
        lambda *_args, **_kwargs: IndeterminateReceipt(reason=reason),
    )
    state = ReasoningState.model_validate(
        {
            case.pending_field: case.pending,
            "pending_recovery": PendingRecovery(
                origin_node=case.origin_node,
                action=case.action,
                trigger="node_exception",
            ),
        }
    )

    update = harness.engine._graph.nodes[RECOVERY_NODE_NAME].invoke(state)

    assert update["automation_terminal"] is True
    assert update[case.pending_field] is None
    assert update["messages"] == [AIMessage(AUTOMATION_TERMINAL_LINE)]
    assert _effect_count(case) == 0


@pytest.mark.parametrize("effect", _EFFECTS)
def test_committed_receipt_with_wrong_record_shape_terminalizes(
    config_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    effect: EffectName,
) -> None:
    harness = build_support_engine(
        config_root,
        policy=make_policy(),
        thread_id=f"{effect}-wrong-receipt-shape",
    )
    case = _case(harness, effect)
    monkeypatch.setattr(
        case.owner,
        _RECEIPT_NAMES[effect],
        lambda *_args, **_kwargs: CommittedReceipt(record="wrong record type"),
    )
    state = ReasoningState.model_validate(
        {
            case.pending_field: case.pending,
            "pending_recovery": PendingRecovery(
                origin_node=case.origin_node,
                action=case.action,
                trigger="node_exception",
            ),
        }
    )

    update = harness.engine._graph.nodes[RECOVERY_NODE_NAME].invoke(state)

    assert update["automation_terminal"] is True
    assert update[case.pending_field] is None
    assert update["messages"] == [AIMessage(AUTOMATION_TERMINAL_LINE)]
    assert _effect_count(case) == 0


def test_missing_pending_effect_contract_terminalizes(
    config_root: Path,
) -> None:
    harness = build_support_engine(
        config_root,
        policy=make_policy(),
        thread_id="missing-effect-contract",
    )
    state = ReasoningState(
        pending_recovery=PendingRecovery(
            origin_node="cart_place",
            action=ExceptionAction.RECONCILE_PLACEMENT,
            trigger="node_exception",
        )
    )

    update = harness.engine._graph.nodes[RECOVERY_NODE_NAME].invoke(state)

    assert update["automation_terminal"] is True
    assert update["messages"] == [AIMessage(AUTOMATION_TERMINAL_LINE)]
    assert harness.store.placed_count == 0


async def test_external_cancellation_mid_batch_reconciles_current_and_aborts_remainder(
    config_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = build_support_engine(
        config_root,
        policy=make_policy(),
        thread_id="external-cancel-mid-batch",
    )
    line = CartLine(
        sku="SKU-BLU-07",
        name="waterproof rain jacket",
        price_usd=129.0,
        quantity=1,
    )
    current = harness.store.place_cart("placed-current", lines=(line,), total_usd=129.0)
    remainder = harness.store.place_cart("placed-remainder", lines=(line,), total_usd=129.0)
    prior_record = harness.store.cancel_order("cancel-prior", order_id="ORD-1002")
    pending = PendingCancelBatch(
        targets=(
            CancelTarget(
                order_id=prior_record.order_id,
                summary=prior_record.summary,
                idempotency_key="cancel-prior",
            ),
            CancelTarget(
                order_id=current.order_id,
                summary="1 waterproof rain jacket",
                idempotency_key="cancel-current",
            ),
            CancelTarget(
                order_id=remainder.order_id,
                summary="1 waterproof rain jacket",
                idempotency_key="cancel-remainder",
            ),
        ),
        outcomes=(
            BatchCancelOutcome(
                order_id=prior_record.order_id,
                summary=prior_record.summary,
                outcome="cancelled",
                amount_usd=prior_record.total_usd,
            ),
        ),
        created_at=time.time(),
    )
    abandoned_message_id = "cancel-batch-current"
    harness.engine._graph.update_state(
        harness.engine._config,
        {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "pending_cancel": pending,
            "execution_owner": "support",
            "consumed_turn_ids": (abandoned_message_id,),
        },
        as_node="support_cancel_confirm",
    )
    assert harness.engine._graph.get_state(harness.engine._config).next == ("support_cancel_void",)
    entered = threading.Event()
    release = threading.Event()
    real_cancel = harness.store.cancel_order
    calls = 0

    def commit_then_pause(idempotency_key: str, *, order_id: str):
        nonlocal calls
        calls += 1
        record = real_cancel(idempotency_key, order_id=order_id)
        entered.set()
        if not release.wait(timeout=_WAIT_TIMEOUT_SECONDS):
            raise RuntimeError("test did not release batch cancellation")
        return record

    monkeypatch.setattr(harness.store, "cancel_order", commit_then_pause)

    async def run_current_target() -> None:
        async for _update in harness.engine._graph.astream(
            Command(update={"consumed_turn_ids": ()}),
            harness.engine._config,
            stream_mode="updates",
        ):
            pass

    abandoned = asyncio.create_task(run_current_target())
    assert await asyncio.to_thread(entered.wait, _WAIT_TIMEOUT_SECONDS)
    abandoned.cancel()
    with pytest.raises(asyncio.CancelledError):
        await abandoned
    active_nodes = harness.engine._node_execution_tracker.active_node_names
    takeover = asyncio.create_task(
        harness.engine._take_over_cancelled_stream(
            owned_thread_id=harness.engine.thread_id,
            abandoned_message_id=abandoned_message_id,
            active_node_names=active_nodes,
            resumed_interrupt_node=None,
        )
    )
    release.set()
    await asyncio.wait_for(takeover, timeout=_WAIT_TIMEOUT_SECONDS)

    events = await engine_events(harness.engine, "cancel the remainder too")
    snapshot = harness.engine._graph.get_state(harness.engine._config)
    spoken = [event.text for event in events if isinstance(event, SpokenMessageEvent)]
    human_texts = [
        str(message.content)
        for message in snapshot.values.get("messages", ())
        if isinstance(message, HumanMessage)
    ]

    assert calls == 1
    assert harness.store.cancel_count == 2
    assert harness.store.order_status(current.order_id) == "cancelled"
    assert harness.store.order_status(remainder.order_id) == "processing"
    assert len(spoken) == 1
    assert prior_record.order_id in spoken[0] and "is cancelled" in spoken[0]
    assert current.order_id in spoken[0] and "is cancelled" in spoken[0]
    assert remainder.order_id in spoken[0] and "did not complete" in spoken[0]
    assert "cancel the remainder too" not in human_texts
    assert snapshot.values.get("pending_cancel") is None
    assert snapshot.values.get("pending_recovery") is None
    assert snapshot.next == ()


@pytest.mark.parametrize("trigger", ("node_exception", "stream_cancelled"))
@pytest.mark.parametrize("current_committed", (False, True))
def test_cancel_recovery_preserves_prior_outcomes_and_aborts_the_remainder(
    config_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    current_committed: bool,
    trigger: str,
) -> None:
    harness = build_support_engine(
        config_root,
        policy=make_policy(),
        thread_id=f"cancel-batch-recovery-{current_committed}",
    )
    line = CartLine(
        sku="SKU-BLU-07",
        name="waterproof rain jacket",
        price_usd=129.0,
        quantity=1,
    )
    current = harness.store.place_cart("placed-current", lines=(line,), total_usd=129.0)
    remainder = harness.store.place_cart("placed-remainder", lines=(line,), total_usd=129.0)
    prior_record = harness.store.cancel_order("cancel-prior", order_id="ORD-1002")
    prior = BatchCancelOutcome(
        order_id=prior_record.order_id,
        summary=prior_record.summary,
        outcome="cancelled",
        amount_usd=prior_record.total_usd,
    )
    pending = PendingCancelBatch(
        targets=(
            CancelTarget(
                order_id=prior.order_id,
                summary=prior.summary,
                idempotency_key="cancel-prior",
            ),
            CancelTarget(
                order_id=current.order_id,
                summary="1 waterproof rain jacket",
                idempotency_key="cancel-current",
            ),
            CancelTarget(
                order_id=remainder.order_id,
                summary="1 waterproof rain jacket",
                idempotency_key="cancel-remainder",
            ),
        ),
        outcomes=(prior,),
        created_at=time.time(),
    )
    if current_committed:
        harness.store.cancel_order("cancel-current", order_id=current.order_id)
    monkeypatch.setattr(
        harness.store,
        "cancel_order",
        lambda *_args, **_kwargs: pytest.fail("recovery replayed cancel_order"),
    )
    abandoned_message_id = "cancel-batch-abandoned" if trigger == "stream_cancelled" else None
    state = ReasoningState(
        consumed_turn_ids=(abandoned_message_id,) if abandoned_message_id is not None else (),
        pending_cancel=pending,
        pending_recovery=PendingRecovery(
            origin_node="support_cancel_void",
            action=ExceptionAction.RECONCILE_CANCEL,
            trigger=trigger,
            abandoned_message_id=abandoned_message_id,
        ),
    )

    update = harness.engine._graph.nodes[RECOVERY_NODE_NAME].invoke(state)
    text = str(update["messages"][0].content)

    assert update["pending_cancel"] is None
    assert harness.store.cancel_count == 1 + int(current_committed)
    assert harness.store.order_status(remainder.order_id) == "processing"
    assert text.count("did not complete") == (1 if current_committed else 2)
    if current_committed:
        assert current.order_id in text and "is cancelled" in text
    assert remainder.order_id in text
