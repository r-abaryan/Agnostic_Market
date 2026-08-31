"""Milestone 6C-d principal-transition and engine hard-failure contracts."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import pytest
from llm_fakes import FakeChatModel
from policy_helpers import make_policy
from pydantic import ValidationError
from support_helpers import SupportHarness, build_support_engine
from turn_helpers import engine_events
from verification_helpers import TEST_FACTOR_REFS, TEST_OTP_CODES, grant_verification

from agnostic_market.agents import engine as engine_module
from agnostic_market.agents import recovery
from agnostic_market.agents.identity import flow as identity_flow
from agnostic_market.agents.recovery import (
    AUTOMATION_TERMINAL_LINE,
    RECOVERY_NODE_NAME,
)
from agnostic_market.agents.telemetry import TelemetryRecord
from agnostic_market.checkpoints import CheckpointScopeError
from agnostic_market.commerce.identity import BoundIdentity
from agnostic_market.commerce.verification import RiskProvider
from agnostic_market.dtos.events import InterruptEvent, SpokenMessageEvent, TurnFacts
from agnostic_market.dtos.orchestration import (
    FocusedOrderTarget,
    ListOrders,
    RefundOrder,
    SwitchAccount,
    VerifyIdentity,
)
from agnostic_market.dtos.recovery import ExceptionAction, PendingRecovery
from agnostic_market.dtos.state import (
    CHECKPOINT_SCHEMA_VERSION,
    PendingIdentity,
    ReasoningState,
    open_active_invocation,
)

_FACTS = TurnFacts()
_CUST1_OTP = TEST_OTP_CODES["CUST-001"]
_CUST2_OTP = TEST_OTP_CODES["CUST-002"]
_CUST1 = BoundIdentity(customer_ref="CUST-001", masked_contact="number ending 0119")


def _assert_checkpoint_thread_retired(engine, config) -> None:
    with pytest.raises(CheckpointScopeError, match="namespace"):
        engine._graph.get_state(config)


_CUST2 = BoundIdentity(customer_ref="CUST-002", masked_contact="email ending example dot com")


def _identity_harness(
    config_root: Path,
    *,
    customer_claim: str = "+1 555 010 0119",
    thread_id: str,
    reasoning: FakeChatModel | None = None,
) -> SupportHarness:
    return build_support_engine(
        config_root,
        policy=make_policy(refund_returnless_under_usd=50.0),
        reasoning=reasoning
        or FakeChatModel(
            force_tool="propose_identity",
            canned_args={"propose_identity": {"contact_claim": customer_claim}},
            tool_call_limit=99,
        ),
        thread_id=thread_id,
    )


async def _events(harness: SupportHarness, text: str) -> list:
    return await engine_events(harness.engine, text, _FACTS)


def _spoken(events: list) -> list[SpokenMessageEvent]:
    return [event for event in events if isinstance(event, SpokenMessageEvent)]


def _telemetry(harness: SupportHarness) -> list[dict[str, object]]:
    return [{"event": record.event, **record.attributes} for record in harness.telemetry.records]


async def test_transition_inspection_requires_every_postcondition_and_invalidation_is_total(
    config_root: Path,
) -> None:
    harness = _identity_harness(config_root, thread_id="transition-inspection")
    context = harness.caller_context

    assert context.inspect_principal_transition().outcome == "none"
    await grant_verification(harness.verification)
    proof = harness.verification.grants[-1]
    transition = await context.transition_principal(
        _CUST1,
        proof,
        ListOrders(scope="account"),
    )

    inspection = context.inspect_principal_transition()
    assert inspection.outcome == "coherent"
    assert inspection.transition == transition

    harness.identity.grant_orders("ORD-1001")
    assert context.inspect_principal_transition().outcome == "inconsistent"
    assert await context.invalidate_principal_transition(transition.transition_id) is True
    assert context.pending_transition() is None
    assert harness.identity.current() is None
    assert not harness.identity.has_residual_order_authority()
    assert harness.verification.current_level() == 1
    assert harness.verification.grants == []


async def test_rejected_initiating_request_does_not_retire_existing_principal(
    config_root: Path,
) -> None:
    harness = _identity_harness(config_root, thread_id="transition-request-rejected")
    harness.identity.bind(_CUST1)
    await grant_verification(harness.verification)
    proof = harness.verification.grants[-1]

    with pytest.raises(ValidationError, match="cannot continue"):
        await harness.caller_context.transition_principal(
            _CUST2,
            proof,
            RefundOrder(target=FocusedOrderTarget()),
        )

    assert harness.caller_context.pending_transition() is None
    assert harness.identity.current() == _CUST1
    assert harness.verification.grants == [proof]


@pytest.mark.parametrize(
    "invocation_request",
    (None, RefundOrder(target=FocusedOrderTarget())),
    ids=("missing-invocation", "unsupported-invocation"),
)
async def test_principal_recovery_rejects_missing_or_mismatched_initiating_request(
    config_root: Path,
    invocation_request: RefundOrder | None,
) -> None:
    case = "missing" if invocation_request is None else "unsupported"
    harness = _identity_harness(config_root, thread_id=f"transition-request-{case}")
    await grant_verification(harness.verification)
    transition = await harness.caller_context.transition_principal(
        _CUST1,
        harness.verification.grants[-1],
        SwitchAccount(),
    )
    consumed_turn_ids = ("transition-opening-turn",) if invocation_request is not None else ()
    invocation = (
        open_active_invocation(invocation_request, consumed_turn_ids=consumed_turn_ids)
        if invocation_request is not None
        else None
    )
    state = ReasoningState(
        consumed_turn_ids=consumed_turn_ids,
        execution_owner="identity",
        active_invocation=invocation,
        pending_identity=PendingIdentity(
            customer_ref=_CUST1.customer_ref,
            masked_contact=_CUST1.masked_contact,
            factor_ref=TEST_FACTOR_REFS[_CUST1.customer_ref],
            attempt_key="transition-request-attempt",
            challenge_id=None,
        ),
        pending_recovery=PendingRecovery(
            origin_node="identity_apply",
            action=ExceptionAction.RECONCILE_PRINCIPAL_TRANSITION,
            trigger="node_exception",
        ),
    )

    update = await harness.engine._graph.nodes[RECOVERY_NODE_NAME].ainvoke(state)

    assert update["automation_terminal"] is True
    assert [message.content for message in update["messages"]] == [AUTOMATION_TERMINAL_LINE]
    assert harness.caller_context.pending_transition() is None
    assert harness.identity.current() is None
    assert harness.verification.current_level() == 1
    assert harness.verification.grants == []
    assert transition.transition_id


@pytest.mark.parametrize(
    "corruption",
    ("ephemeral", "identity", "proof", "order_grant"),
)
async def test_every_partial_transition_terminalizes_without_rotation(
    config_root: Path,
    corruption: str,
) -> None:
    harness = _identity_harness(
        config_root,
        thread_id=f"partial-transition-{corruption}",
    )
    old_thread = harness.engine.thread_id
    await grant_verification(harness.verification)
    transition = await harness.caller_context.transition_principal(
        _CUST1,
        harness.verification.grants[-1],
        ListOrders(scope="account"),
    )
    if corruption == "ephemeral":
        harness.caller_context.cart_store.add_item(
            sku="SKU-BLU-07",
            name="blue wool hat",
            price_usd=29.0,
            quantity=1,
        )
    elif corruption == "identity":
        harness.identity.clear()
    elif corruption == "proof":
        await harness.verification.clear()
    else:
        harness.identity.grant_orders("ORD-1001")

    events = await _events(harness, "queued request")
    snapshot = harness.engine._graph.get_state(harness.engine._config)

    assert [event.text for event in _spoken(events)] == [AUTOMATION_TERMINAL_LINE]
    assert harness.engine.thread_id == old_thread
    assert harness.caller_context.pending_transition() is None
    assert harness.identity.current() is None
    assert not harness.identity.has_residual_order_authority()
    assert harness.verification.current_level() == 1
    assert snapshot.values["automation_terminal"] is True
    assert snapshot.next == ()
    assert transition.transition_id


async def test_identity_apply_failure_before_publication_preserves_original_principal(
    config_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _identity_harness(
        config_root,
        customer_claim="casey@example.com",
        thread_id="transition-before-publish",
    )
    await grant_verification(harness.verification)
    old_proof = harness.verification.grants[-1]
    harness.identity.bind(_CUST1)
    harness.caller_context.cart_store.add_item(
        sku="SKU-BLU-07",
        name="blue wool hat",
        price_usd=29.0,
        quantity=1,
    )
    old_thread = harness.engine.thread_id

    warning = await _events(harness, "switch my account")
    assert any(isinstance(event, InterruptEvent) for event in warning)
    dispatched = await _events(harness, "yes")
    assert any(isinstance(event, InterruptEvent) for event in dispatched)

    def fail_before_publication(_challenge_id: str):
        raise RuntimeError("injected failure before transition publication")

    monkeypatch.setattr(harness.verification, "proof_for_challenge", fail_before_publication)
    recovered = await _events(harness, _CUST2_OTP)
    snapshot = harness.engine._graph.get_state(harness.engine._config)

    assert [event.text for event in _spoken(recovered)] == [
        "I couldn't finish that account request; please try it again."
    ]
    assert harness.engine.thread_id == old_thread
    assert harness.identity.current() == _CUST1
    assert not harness.caller_context.cart_store.is_empty()
    assert old_proof in harness.verification.grants
    assert harness.caller_context.pending_transition() is None
    assert snapshot.values.get("pending_identity") is None
    assert snapshot.values.get("active_invocation") is None
    assert snapshot.next == ()


async def test_identity_apply_missing_invocation_fails_before_transition_publication(
    config_root: Path,
) -> None:
    harness = _identity_harness(
        config_root,
        thread_id="transition-missing-invocation-before-publish",
    )
    await grant_verification(harness.verification)
    nodes = identity_flow.build_identity_nodes(
        FakeChatModel(emit_tool_calls=False),
        harness.verification,
        RiskProvider("acme_store", flagged=False),
        harness.customers,
        harness.identity,
        make_policy(refund_returnless_under_usd=50.0),
        harness.caller_context.transition_principal,
        display_name="Acme Store",
        telemetry=harness.caller_context.telemetry,
    )
    state = ReasoningState(
        execution_owner="identity",
        pending_identity=PendingIdentity(
            customer_ref=_CUST1.customer_ref,
            masked_contact=_CUST1.masked_contact,
            factor_ref=TEST_FACTOR_REFS[_CUST1.customer_ref],
            attempt_key="missing-invocation-attempt",
            challenge_id=None,
        ),
    )

    with pytest.raises(RuntimeError, match="requires an active invocation"):
        await nodes.apply(state)

    assert harness.caller_context.pending_transition() is None
    assert harness.identity.current() is None


async def test_identity_apply_failure_after_coherent_publish_rotates_and_continues_once(
    config_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _identity_harness(config_root, thread_id="transition-after-publish")
    old_thread = harness.engine.thread_id
    old_config = harness.engine._config
    await _events(harness, "list my account orders")

    real_write = harness.telemetry.emit

    def fail_after_publication(record: TelemetryRecord) -> None:
        if record.event == "identity_bound":
            raise RuntimeError("injected failure after coherent publication")
        real_write(record)

    monkeypatch.setattr(harness.telemetry, "emit", fail_after_publication)
    recovered = await _events(harness, _CUST1_OTP)

    assert harness.engine.thread_id != old_thread
    assert harness.caller_context.pending_transition() is None
    assert harness.identity.current() == _CUST1
    assert len(harness.verification.grants) == 1
    _assert_checkpoint_thread_retired(harness.engine, old_config)
    lines = [event.text for event in _spoken(recovered)]
    assert len(lines) == 1
    assert "ORD-1001" in lines[0] and "ORD-1003" in lines[0]
    assert "ORD-1002" not in lines[0]
    assert harness.otp.dispatch_count == 1
    assert harness.engine._graph.get_state(harness.engine._config).next == ()
    reconciled = [
        event
        for event in _telemetry(harness)
        if event.get("event") == "principal_transition_reconciled"
    ]
    rotated = [
        event for event in _telemetry(harness) if event.get("event") == "reasoning_context_rotated"
    ]
    assert [event["outcome"] for event in reconciled] == ["coherent"]
    assert len(rotated) == 1
    assert rotated[0]["transition_id"] == reconciled[0]["transition_id"]


async def test_switch_apply_failure_after_coherent_publish_preserves_acknowledgement(
    config_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _identity_harness(
        config_root,
        customer_claim="casey@example.com",
        thread_id="switch-after-publish",
    )
    await grant_verification(harness.verification)
    harness.identity.bind(_CUST1)
    old_thread = harness.engine.thread_id
    old_config = harness.engine._config
    warning = await _events(harness, "switch my account")
    assert any(isinstance(event, InterruptEvent) for event in warning)
    dispatched = await _events(harness, "yes")
    assert any(isinstance(event, InterruptEvent) for event in dispatched)
    real_write = harness.telemetry.emit

    def fail_after_publication(record: TelemetryRecord) -> None:
        if record.event == "identity_bound":
            raise RuntimeError("injected switch tail failure")
        real_write(record)

    monkeypatch.setattr(harness.telemetry, "emit", fail_after_publication)
    recovered = await _events(harness, _CUST2_OTP)

    assert [event.text for event in _spoken(recovered)] == [
        "You're now verified on the new account."
    ]
    assert harness.engine.thread_id != old_thread
    assert harness.identity.current() == _CUST2
    assert harness.caller_context.pending_transition() is None
    _assert_checkpoint_thread_retired(harness.engine, old_config)


async def test_verify_apply_failure_after_coherent_publish_preserves_acknowledgement(
    config_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _identity_harness(config_root, thread_id="verify-after-publish")
    opening_turn_id = "verify-after-publish-opening"
    consumed_turn_ids = (opening_turn_id,)
    harness.engine._graph.update_state(
        harness.engine._config,
        {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "consumed_turn_ids": consumed_turn_ids,
            "active_invocation": open_active_invocation(
                VerifyIdentity(),
                consumed_turn_ids=consumed_turn_ids,
            ),
        },
        as_node="__start__",
    )
    old_thread = harness.engine.thread_id
    old_config = harness.engine._config
    dispatched = await _events(harness, "continue")
    assert any(isinstance(event, InterruptEvent) for event in dispatched)
    real_write = harness.telemetry.emit

    def fail_after_publication(record: TelemetryRecord) -> None:
        if record.event == "identity_bound":
            raise RuntimeError("injected verification tail failure")
        real_write(record)

    monkeypatch.setattr(harness.telemetry, "emit", fail_after_publication)
    recovered = await _events(harness, _CUST1_OTP)

    assert [event.text for event in _spoken(recovered)] == ["You're now verified."]
    assert harness.engine.thread_id != old_thread
    assert harness.identity.current() == _CUST1
    assert harness.caller_context.pending_transition() is None
    _assert_checkpoint_thread_retired(harness.engine, old_config)
    new_state = harness.engine._graph.get_state(harness.engine._config)
    assert new_state.next == ()
    assert new_state.values.get("active_invocation") is None


async def test_cancelled_native_async_identity_apply_invalidates_partial_transition(
    config_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _identity_harness(
        config_root,
        customer_claim="casey@example.com",
        thread_id="cancelled-native-async-principal",
    )
    await grant_verification(harness.verification)
    harness.identity.bind(_CUST1)
    old_thread = harness.engine.thread_id
    warning = await _events(harness, "switch my account")
    assert any(isinstance(event, InterruptEvent) for event in warning)
    dispatched = await _events(harness, "yes")
    assert any(isinstance(event, InterruptEvent) for event in dispatched)

    entered = asyncio.Event()
    release = asyncio.Event()
    real_retain_only = harness.otp.retain_only

    async def pause_during_provider_cleanup(
        session_id: str,
        challenge_ids: tuple[str, ...],
    ) -> None:
        entered.set()
        await release.wait()
        await real_retain_only(session_id, challenge_ids)

    monkeypatch.setattr(harness.otp, "retain_only", pause_during_provider_cleanup)

    applying = asyncio.create_task(_events(harness, _CUST2_OTP))
    await asyncio.wait_for(entered.wait(), timeout=5)
    assert harness.caller_context.inspect_principal_transition().outcome == "inconsistent"
    applying.cancel("cancelled-identity-apply")
    release.set()
    with pytest.raises(asyncio.CancelledError) as cancelled:
        await applying
    assert cancelled.value.args == ("cancelled-identity-apply",)
    assert harness.engine._node_execution_tracker.active_node_names == frozenset()
    assert harness.engine.thread_id == old_thread
    assert harness.identity.current() is None
    assert not harness.identity.has_residual_order_authority()
    assert harness.verification.current_level() == 1
    assert harness.verification.grants == []
    assert harness.otp.active_challenge_count == 0
    assert harness.caller_context.pending_transition() is None
    snapshot = harness.engine._graph.get_state(harness.engine._config)
    assert snapshot.values["automation_terminal"] is True
    assert snapshot.next == ()
    repeated = await _events(harness, "continue")
    assert [event.text for event in _spoken(repeated)] == [AUTOMATION_TERMINAL_LINE]


@pytest.mark.parametrize("wait_outcome", ("timeout", "failure"))
@pytest.mark.parametrize("provider_window", ("before", "after"))
async def test_native_async_identity_apply_deferred_cleanup_is_fail_closed(
    config_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    wait_outcome: str,
    provider_window: str,
) -> None:
    harness = _identity_harness(
        config_root,
        customer_claim="casey@example.com",
        thread_id=f"async-principal-{wait_outcome}-{provider_window}",
    )
    await grant_verification(harness.verification)
    harness.identity.bind(_CUST1)
    await _events(harness, "switch my account")
    await _events(harness, "yes")

    entered = asyncio.Event()
    release = asyncio.Event()
    real_retain_only = harness.verification.retain_only

    async def pause_retain_only(proof) -> None:
        if provider_window == "after":
            await real_retain_only(proof)
        entered.set()
        await release.wait()
        if provider_window == "before":
            await real_retain_only(proof)

    monkeypatch.setattr(harness.verification, "retain_only", pause_retain_only)
    terminal_persisted = asyncio.Event()
    real_finalize = harness.engine._afinalize_last_resort_state

    async def observe_terminal_persistence(*, replace_checkpoint: bool = False) -> None:
        await real_finalize(replace_checkpoint=replace_checkpoint)
        terminal_persisted.set()

    monkeypatch.setattr(
        harness.engine,
        "_afinalize_last_resort_state",
        observe_terminal_persistence,
    )

    if wait_outcome == "timeout":
        monkeypatch.setattr(
            harness.engine._node_execution_tracker,
            "wait_until_mutable_idle",
            lambda _timeout_seconds: False,
        )
    else:

        def fail_wait(_timeout_seconds: float) -> bool:
            raise RuntimeError("injected quiescence wait failure")

        monkeypatch.setattr(
            harness.engine._node_execution_tracker,
            "wait_until_mutable_idle",
            fail_wait,
        )

    applying = asyncio.create_task(_events(harness, _CUST2_OTP))
    await asyncio.wait_for(entered.wait(), timeout=5)
    assert harness.caller_context.inspect_principal_transition().outcome == "inconsistent"
    applying.cancel("unquiescent-identity-apply")
    release.set()
    with pytest.raises(asyncio.CancelledError) as cancelled:
        await applying
    assert cancelled.value.args == ("unquiescent-identity-apply",)

    assert harness.engine._terminal_latched is True
    assert harness.engine._node_execution_tracker.active_node_names == frozenset()
    cleanup_complete = threading.Event()
    harness.engine._node_execution_tracker.defer_until_fully_idle(cleanup_complete.set)
    assert await asyncio.to_thread(cleanup_complete.wait, 5)
    await asyncio.wait_for(terminal_persisted.wait(), timeout=5)

    assert harness.identity.current() is None
    assert not harness.identity.has_residual_order_authority()
    assert harness.verification.current_level() == 1
    assert harness.verification.grants == []
    assert harness.otp.active_challenge_count == 0
    assert harness.caller_context.pending_transition() is None
    stable = harness.engine._graph.get_state(harness.engine._config)
    assert stable.values.get("pending_recovery") is None
    assert stable.values["automation_terminal"] is True
    assert stable.next == ()
    blocked = await _events(harness, "continue after deferred cleanup")
    assert [event.text for event in _spoken(blocked)] == [AUTOMATION_TERMINAL_LINE]
    unchanged = harness.engine._graph.get_state(harness.engine._config)
    assert unchanged.values == stable.values
    assert unchanged.next == stable.next
    assert unchanged.tasks == stable.tasks
    assert unchanged.interrupts == stable.interrupts


async def test_native_async_deferred_cleanup_yields_to_session_close(
    config_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _identity_harness(
        config_root,
        customer_claim="casey@example.com",
        thread_id="async-principal-close-race",
    )
    await grant_verification(harness.verification)
    harness.identity.bind(_CUST1)
    await _events(harness, "switch my account")
    await _events(harness, "yes")

    entered = asyncio.Event()
    release = asyncio.Event()
    real_retain_only = harness.verification.retain_only

    async def pause_before_provider_cleanup(proof) -> None:
        entered.set()
        await release.wait()
        await real_retain_only(proof)

    monkeypatch.setattr(
        harness.verification,
        "retain_only",
        pause_before_provider_cleanup,
    )
    monkeypatch.setattr(
        harness.engine._node_execution_tracker,
        "wait_until_mutable_idle",
        lambda _timeout_seconds: False,
    )

    applying = asyncio.create_task(_events(harness, _CUST2_OTP))
    await asyncio.wait_for(entered.wait(), timeout=5)
    assert harness.caller_context.inspect_principal_transition().outcome == "inconsistent"
    applying.cancel("unquiescent-close-race")
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await applying

    closing = asyncio.create_task(harness.caller_context.aclose_session())
    await asyncio.sleep(0)
    assert harness.engine._node_execution_tracker.active_node_names == frozenset()
    assert harness.engine._graph.get_state(harness.engine._config).values

    close_complete = threading.Event()
    harness.engine._node_execution_tracker.defer_until_fully_idle(close_complete.set)
    assert await asyncio.to_thread(close_complete.wait, 5)
    await closing
    await asyncio.sleep(0)
    if harness.engine._background_tasks:
        await asyncio.gather(*tuple(harness.engine._background_tasks))

    assert harness.identity.current() is None
    assert not harness.identity.has_residual_order_authority()
    assert harness.verification.current_level() == 1
    assert harness.verification.grants == []
    assert harness.otp.active_challenge_count == 0
    assert harness.caller_context.pending_transition() is None
    _assert_checkpoint_thread_retired(harness.engine, harness.engine._config)
    terminal = await _events(harness, "continue")
    assert _spoken(terminal) == []
    _assert_checkpoint_thread_retired(harness.engine, harness.engine._config)


async def test_old_thread_yes_cannot_resume_new_thread_confirmation(
    config_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _identity_harness(
        config_root,
        thread_id="cross-thread-consent",
        reasoning=FakeChatModel(
            scripted_calls=[
                [("propose_cancel", {"order_keys": ["ORD-1002"]})],
                [
                    (
                        "propose_identity",
                        {"contact_claim": "casey@example.com"},
                    )
                ],
            ],
            tool_call_limit=99,
        ),
    )
    engine = harness.engine
    old_thread = engine.thread_id
    pending = await _events(harness, "cancel order ORD-1002")
    assert any(isinstance(event, InterruptEvent) for event in pending)

    published = threading.Event()
    release_apply = threading.Event()
    real_write = harness.telemetry.emit

    def pause_after_publication(record: TelemetryRecord) -> None:
        if record.event == "identity_bound":
            published.set()
            if not release_apply.wait(timeout=5):
                raise RuntimeError("test did not release identity_apply")
        real_write(record)

    monkeypatch.setattr(harness.telemetry, "emit", pause_after_publication)
    completing = asyncio.create_task(_events(harness, _CUST2_OTP))
    assert await asyncio.to_thread(published.wait, 5)
    queued_arrived = asyncio.Event()
    real_get_state = engine._graph.aget_state

    async def observe_queued_arrival(config):
        task = asyncio.current_task()
        if task is not None and task.get_name() == "queued-old-thread":
            queued_arrived.set()
        return await real_get_state(config)

    monkeypatch.setattr(engine._graph, "aget_state", observe_queued_arrival)
    queued = asyncio.create_task(_events(harness, "yes"), name="queued-old-thread")
    await asyncio.wait_for(queued_arrived.wait(), timeout=5)
    release_apply.set()

    first_events = await completing
    events = await queued

    assert engine.thread_id != old_thread
    assert harness.caller_context.pending_transition() is None
    assert harness.store.cancel_count == 0
    first_prompts = [event.prompt for event in first_events if isinstance(event, InterruptEvent)]
    prompts = [event.prompt for event in events if isinstance(event, InterruptEvent)]
    assert len(first_prompts) == 1 and "ORD-1002" in first_prompts[0]
    assert len(prompts) == 1 and "ORD-1002" in prompts[0]
    assert prompts == first_prompts
    assert await engine.apending_interrupt()
    assert tuple(engine._graph.get_state(engine._config).values["consumed_turn_ids"]) == (
        "test-turn-1",
        "test-turn-2",
        "test-turn-3",
    )

    await _events(harness, "yes")
    assert harness.store.cancel_count == 1
    assert tuple(engine._graph.get_state(engine._config).values["consumed_turn_ids"]) == (
        "test-turn-1",
        "test-turn-2",
        "test-turn-3",
        "test-turn-4",
    )
    consumed = [
        event for event in _telemetry(harness) if event.get("event") == "cross_thread_turn_consumed"
    ]
    assert len(consumed) == 1


async def test_checkpointer_and_takeover_failure_latches_terminal_without_later_graph_work(
    config_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _identity_harness(config_root, thread_id="terminal-latch")
    read_calls = 0
    takeover_calls = 0

    async def fail_get_state(_config):
        nonlocal read_calls
        read_calls += 1
        raise RuntimeError("injected checkpointer read failure")

    async def fail_takeover(_config, _values, *, as_node=None):
        nonlocal takeover_calls
        takeover_calls += 1
        assert as_node == "automation_terminal_response"
        raise RuntimeError("injected checkpoint takeover failure")

    monkeypatch.setattr(harness.engine._graph, "aget_state", fail_get_state)
    monkeypatch.setattr(harness.engine._graph, "aupdate_state", fail_takeover)

    first = await _events(harness, "hello")
    second = await _events(harness, "try again")

    assert [event.text for event in _spoken(first)] == [AUTOMATION_TERMINAL_LINE]
    assert [event.text for event in _spoken(second)] == [AUTOMATION_TERMINAL_LINE]
    assert read_calls == 1
    assert takeover_calls == 1


async def test_recovery_node_failure_terminalizes_and_invalidates_coherent_transition(
    config_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _identity_harness(config_root, thread_id="recovery-node-failure")
    old_thread = harness.engine.thread_id
    await _events(harness, "list my account orders")
    real_identity_write = harness.telemetry.emit
    identity_failed = False

    def fail_identity_tail(record: TelemetryRecord) -> None:
        nonlocal identity_failed
        if record.event == "identity_bound":
            identity_failed = True
            raise RuntimeError("injected identity tail failure")
        if identity_failed:
            raise RuntimeError("injected recovery infrastructure failure")
        real_identity_write(record)

    monkeypatch.setattr(harness.telemetry, "emit", fail_identity_tail)
    events = await _events(harness, _CUST1_OTP)
    snapshot = harness.engine._graph.get_state(harness.engine._config)

    assert [event.text for event in _spoken(events)] == [AUTOMATION_TERMINAL_LINE]
    assert harness.engine.thread_id == old_thread
    assert harness.caller_context.pending_transition() is None
    assert harness.identity.current() is None
    assert harness.verification.current_level() == 1
    assert snapshot.values["automation_terminal"] is True
    assert snapshot.values.get("pending_recovery") is None
    assert snapshot.next == ()


async def test_terminalizer_failure_escapes_once_to_engine_takeover(
    config_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _identity_harness(config_root, thread_id="terminalizer-failure")
    await _events(harness, "list my account orders")
    real_identity_write = harness.telemetry.emit

    def fail_identity_tail(record: TelemetryRecord) -> None:
        if record.event == "identity_bound":
            raise RuntimeError("injected identity tail failure")
        real_identity_write(record)

    def fail_clear() -> dict[str, object]:
        raise RuntimeError("injected terminalizer failure")

    monkeypatch.setattr(harness.telemetry, "emit", fail_identity_tail)
    monkeypatch.setattr(recovery, "clear_automation_state", fail_clear)
    events = await _events(harness, _CUST1_OTP)
    snapshot = harness.engine._graph.get_state(harness.engine._config)
    repeated = await _events(harness, "try again")

    assert [event.text for event in _spoken(events)] == [AUTOMATION_TERMINAL_LINE]
    assert [event.text for event in _spoken(repeated)] == [AUTOMATION_TERMINAL_LINE]
    assert harness.caller_context.pending_transition() is None
    assert harness.identity.current() is None
    assert snapshot.values["automation_terminal"] is True
    assert snapshot.next == ()


async def test_terminal_response_failure_escapes_once_to_engine_takeover(
    config_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _identity_harness(config_root, thread_id="terminal-response-failure")
    harness.engine._graph.update_state(
        harness.engine._config,
        {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "automation_terminal": True,
        },
        as_node="entry",
    )
    real_write = harness.telemetry.emit

    def fail_terminal_response(record: TelemetryRecord) -> None:
        if record.event == "automation_terminal_response":
            raise RuntimeError("injected terminal response failure")
        real_write(record)

    monkeypatch.setattr(harness.telemetry, "emit", fail_terminal_response)
    events = await _events(harness, "try again")
    snapshot = harness.engine._graph.get_state(harness.engine._config)
    repeated = await _events(harness, "still there")

    assert [event.text for event in _spoken(events)] == [AUTOMATION_TERMINAL_LINE]
    assert [event.text for event in _spoken(repeated)] == [AUTOMATION_TERMINAL_LINE]
    assert snapshot.values["automation_terminal"] is True
    assert snapshot.values.get("pending_recovery") is None
    assert snapshot.next == ()


@pytest.mark.parametrize(
    ("failure_point", "thread_switches"),
    (
        ("seed", False),
        ("contamination", False),
        ("delete", False),
        ("complete", True),
    ),
)
async def test_rotation_failure_is_one_shot_and_terminal(
    config_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
    thread_switches: bool,
) -> None:
    harness = _identity_harness(
        config_root,
        thread_id=f"rotation-failure-{failure_point}",
    )
    engine = harness.engine
    old_thread = engine.thread_id
    old_storage_thread = engine._config["configurable"]["thread_id"]
    await grant_verification(harness.verification)
    transition = await harness.caller_context.transition_principal(
        _CUST1,
        harness.verification.grants[-1],
        ListOrders(scope="account"),
    )
    generated_ids: list[str] = []

    def one_thread_id() -> str:
        value = f"new-thread-{len(generated_ids) + 1}"
        generated_ids.append(value)
        return value

    monkeypatch.setattr(engine_module, "_new_thread_id", one_thread_id)
    if failure_point in {"seed", "contamination"}:
        real_update = engine._graph.aupdate_state

        async def intercept_seed(config, values, *, as_node=None):
            if config["configurable"]["thread_id"] != old_storage_thread:
                if failure_point == "seed":
                    raise RuntimeError("injected rotation seed failure")
                await real_update(config, values, as_node=as_node)
                return await real_update(
                    config,
                    {"identity_claim_misses": 1},
                    as_node="__start__",
                )
            return await real_update(config, values, as_node=as_node)

        monkeypatch.setattr(engine._graph, "aupdate_state", intercept_seed)
    elif failure_point == "delete":
        real_delete = engine._graph.checkpointer.adelete_thread

        async def fail_old_delete(thread_id: str) -> None:
            if thread_id == old_storage_thread:
                raise RuntimeError("injected old-thread delete failure")
            await real_delete(thread_id)

        monkeypatch.setattr(engine._graph.checkpointer, "adelete_thread", fail_old_delete)
    else:

        def fail_completion(transition_id: str) -> None:
            assert transition_id == transition.transition_id
            raise RuntimeError("injected transition completion failure")

        monkeypatch.setattr(harness.caller_context, "complete_transition", fail_completion)

    events = await _events(harness, "queued request")
    repeated = await _events(harness, "try again")

    assert [event.text for event in _spoken(events)] == [AUTOMATION_TERMINAL_LINE]
    assert [event.text for event in _spoken(repeated)] == [AUTOMATION_TERMINAL_LINE]
    assert generated_ids == ["new-thread-1"]
    assert (engine.thread_id != old_thread) is thread_switches
    assert harness.caller_context.pending_transition() is None
    assert harness.identity.current() is None
    assert harness.verification.current_level() == 1
    snapshot = engine._graph.get_state(engine._config)
    assert snapshot.values["automation_terminal"] is True
    assert snapshot.next == ()
