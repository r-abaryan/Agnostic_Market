"""Milestone 6C-d principal-transition and engine hard-failure contracts."""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

import pytest
from llm_fakes import FakeChatModel
from policy_helpers import make_policy
from support_helpers import SupportHarness, build_support_engine
from turn_helpers import engine_events

from agnostic_market.agents import engine as engine_module
from agnostic_market.agents import recovery
from agnostic_market.agents.frontline import graph as frontline_graph
from agnostic_market.agents.identity import flow as identity_flow
from agnostic_market.agents.recovery import AUTOMATION_TERMINAL_LINE
from agnostic_market.commerce.identity import BoundIdentity
from agnostic_market.dtos.events import InterruptEvent, SpokenMessageEvent, TurnFacts
from agnostic_market.dtos.orchestration import ListOrders

_FACTS = TurnFacts()
_OTP = "482913"
_CUST1 = BoundIdentity(customer_ref="CUST-001", masked_contact="number ending 0119")
_CUST2 = BoundIdentity(customer_ref="CUST-002", masked_contact="email ending example dot com")


def _identity_harness(
    config_root: Path,
    *,
    customer_claim: str = "+1 555 010 0119",
    reason_code: str = "list_orders",
    thread_id: str,
    reasoning: FakeChatModel | None = None,
) -> SupportHarness:
    return build_support_engine(
        config_root,
        policy=make_policy(refund_returnless_under_usd=50.0),
        frontline=FakeChatModel(
            force_tool="request_handover",
            canned_args={
                "request_handover": {
                    "destination": "support",
                    "reason_code": reason_code,
                }
            },
            tool_call_limit=99,
        ),
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


def _telemetry(tmp_path: Path) -> list[dict[str, object]]:
    path = tmp_path / "telemetry.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_transition_inspection_requires_every_postcondition_and_invalidation_is_total(
    config_root: Path,
) -> None:
    harness = _identity_harness(config_root, thread_id="transition-inspection")
    context = harness.caller_context

    assert context.inspect_principal_transition().outcome == "none"
    assert harness.verification.verify_otp(_OTP)
    proof = harness.verification.grants[-1]
    transition = context.transition_principal(
        _CUST1,
        proof,
        ListOrders(scope="account"),
    )

    inspection = context.inspect_principal_transition()
    assert inspection.outcome == "coherent"
    assert inspection.transition == transition

    harness.identity.grant_order("ORD-1001")
    assert context.inspect_principal_transition().outcome == "inconsistent"
    assert context.invalidate_principal_transition(transition.transition_id) is True
    assert context.pending_transition() is None
    assert harness.identity.current() is None
    assert not harness.identity.has_residual_order_authority()
    assert harness.verification.current_level() == 1
    assert harness.verification.grants == []


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
    assert harness.verification.verify_otp(_OTP)
    transition = harness.caller_context.transition_principal(
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
        harness.verification.clear()
    else:
        harness.identity.grant_order("ORD-1001")

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
        reason_code="switch_account",
        thread_id="transition-before-publish",
    )
    assert harness.verification.verify_otp(_OTP)
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

    def fail_before_publication(_grant_count: int):
        raise RuntimeError("injected failure before transition publication")

    monkeypatch.setattr(harness.verification, "fresh_proof_since", fail_before_publication)
    recovered = await _events(harness, _OTP)
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
    assert snapshot.values.get("pending_request") is None
    assert snapshot.next == ()


async def test_identity_apply_failure_after_coherent_publish_rotates_and_continues_once(
    config_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _identity_harness(config_root, thread_id="transition-after-publish")
    old_thread = harness.engine.thread_id
    await _events(harness, "list my account orders")

    real_write = identity_flow.write_event

    def fail_after_publication(record: dict[str, object]) -> None:
        if record.get("event") == "identity_bound":
            raise RuntimeError("injected failure after coherent publication")
        real_write(record)

    monkeypatch.setattr(identity_flow, "write_event", fail_after_publication)
    recovered = await _events(harness, _OTP)

    assert harness.engine.thread_id != old_thread
    assert harness.caller_context.pending_transition() is None
    assert harness.identity.current() == _CUST1
    assert len(harness.verification.grants) == 1
    assert harness.engine._graph.get_state({"configurable": {"thread_id": old_thread}}).values == {}
    lines = [event.text for event in _spoken(recovered)]
    assert len(lines) == 1
    assert "ORD-1001" in lines[0] and "ORD-1003" in lines[0]
    assert "ORD-1002" not in lines[0]
    assert harness.otp.dispatch_count == 1
    assert harness.engine._graph.get_state(harness.engine._config).next == ()
    reconciled = [
        event
        for event in _telemetry(tmp_path)
        if event.get("event") == "principal_transition_reconciled"
    ]
    rotated = [
        event for event in _telemetry(tmp_path) if event.get("event") == "reasoning_context_rotated"
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
        reason_code="switch_account",
        thread_id="switch-after-publish",
    )
    assert harness.verification.verify_otp(_OTP)
    harness.identity.bind(_CUST1)
    old_thread = harness.engine.thread_id
    warning = await _events(harness, "switch my account")
    assert any(isinstance(event, InterruptEvent) for event in warning)
    dispatched = await _events(harness, "yes")
    assert any(isinstance(event, InterruptEvent) for event in dispatched)
    real_write = identity_flow.write_event

    def fail_after_publication(record: dict[str, object]) -> None:
        if record.get("event") == "identity_bound":
            raise RuntimeError("injected switch tail failure")
        real_write(record)

    monkeypatch.setattr(identity_flow, "write_event", fail_after_publication)
    recovered = await _events(harness, _OTP)

    assert [event.text for event in _spoken(recovered)] == [
        "You're now verified on the new account."
    ]
    assert harness.engine.thread_id != old_thread
    assert harness.identity.current() == _CUST2
    assert harness.caller_context.pending_transition() is None
    assert harness.engine._graph.get_state({"configurable": {"thread_id": old_thread}}).values == {}


@pytest.mark.parametrize("publication", ("none", "coherent", "inconsistent"))
async def test_cancelled_identity_apply_resolves_principal_publication_fail_closed(
    config_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    publication: str,
) -> None:
    harness = _identity_harness(
        config_root,
        customer_claim="casey@example.com",
        reason_code="switch_account",
        thread_id=f"cancelled-principal-{publication}",
    )
    assert harness.verification.verify_otp(_OTP)
    old_proof = harness.verification.grants[-1]
    harness.identity.bind(_CUST1)
    old_thread = harness.engine.thread_id
    warning = await _events(harness, "switch my account")
    assert any(isinstance(event, InterruptEvent) for event in warning)
    dispatched = await _events(harness, "yes")
    assert any(isinstance(event, InterruptEvent) for event in dispatched)

    entered = threading.Event()
    release = threading.Event()
    if publication == "none":

        def pause_without_publication(_grant_count: int):
            entered.set()
            if not release.wait(timeout=5):
                raise RuntimeError("test did not release identity_apply")
            return None

        monkeypatch.setattr(
            harness.verification,
            "fresh_proof_since",
            pause_without_publication,
        )
    else:
        real_write = identity_flow.write_event

        def pause_after_publication(record: dict[str, object]) -> None:
            if record.get("event") == "identity_bound":
                entered.set()
                if not release.wait(timeout=5):
                    raise RuntimeError("test did not release identity_apply")
            real_write(record)

        monkeypatch.setattr(identity_flow, "write_event", pause_after_publication)

    applying = asyncio.create_task(_events(harness, _OTP))
    assert await asyncio.to_thread(entered.wait, 5)
    if publication == "inconsistent":
        harness.identity.grant_order("ORD-1002")
    applying.cancel("cancelled-identity-apply")
    release.set()
    with pytest.raises(asyncio.CancelledError) as cancelled:
        await applying
    assert cancelled.value.args == ("cancelled-identity-apply",)

    if publication == "none":
        marker = harness.engine._graph.get_state(harness.engine._config).values["pending_recovery"]
        assert marker.trigger == "stream_cancelled"
        assert marker.origin_node == "identity_apply"
        recovered = await _events(harness, "continue")
        assert [event.text for event in _spoken(recovered)] == [
            "I couldn't finish that account request; please try it again."
        ]
        assert harness.engine.thread_id == old_thread
        assert harness.identity.current() == _CUST1
        assert old_proof in harness.verification.grants
        assert harness.caller_context.pending_transition() is None
        assert harness.engine._graph.get_state(harness.engine._config).next == ()
    elif publication == "coherent":
        assert harness.engine.thread_id != old_thread
        assert harness.identity.current() == _CUST2
        assert harness.caller_context.pending_transition() is None
        assert harness.engine._graph.get_state(harness.engine._config).next == ()
        assert (
            harness.engine._graph.get_state({"configurable": {"thread_id": old_thread}}).values
            == {}
        )
    else:
        terminal = await _events(harness, "continue")
        assert [event.text for event in _spoken(terminal)] == [AUTOMATION_TERMINAL_LINE]
        assert harness.engine.thread_id == old_thread
        assert harness.identity.current() is None
        assert not harness.identity.has_residual_order_authority()
        assert harness.verification.current_level() == 1
        assert harness.caller_context.pending_transition() is None
        snapshot = harness.engine._graph.get_state(harness.engine._config)
        assert snapshot.values["automation_terminal"] is True
        assert snapshot.next == ()


@pytest.mark.parametrize("wait_outcome", ("timeout", "failure"))
@pytest.mark.parametrize("publication", ("before", "after"))
async def test_unquiescent_identity_apply_defers_terminal_cleanup_until_worker_exit(
    config_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    wait_outcome: str,
    publication: str,
) -> None:
    harness = _identity_harness(
        config_root,
        customer_claim="casey@example.com",
        reason_code="switch_account",
        thread_id=f"unquiescent-principal-{wait_outcome}-{publication}",
    )
    assert harness.verification.verify_otp(_OTP)
    harness.identity.bind(_CUST1)
    await _events(harness, "switch my account")
    await _events(harness, "yes")

    entered = threading.Event()
    release = threading.Event()
    if publication == "before":
        real_fresh_proof_since = harness.verification.fresh_proof_since

        def pause_before_publication(grant_count: int):
            entered.set()
            if not release.wait(timeout=5):
                raise RuntimeError("test did not release identity_apply")
            return real_fresh_proof_since(grant_count)

        monkeypatch.setattr(
            harness.verification,
            "fresh_proof_since",
            pause_before_publication,
        )
    else:
        real_write = identity_flow.write_event

        def pause_after_publication(record: dict[str, object]) -> None:
            if record.get("event") == "identity_bound":
                entered.set()
                if not release.wait(timeout=5):
                    raise RuntimeError("test did not release identity_apply")
            real_write(record)

        monkeypatch.setattr(identity_flow, "write_event", pause_after_publication)

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

    applying = asyncio.create_task(_events(harness, _OTP))
    assert await asyncio.to_thread(entered.wait, 5)
    applying.cancel("unquiescent-identity-apply")
    with pytest.raises(asyncio.CancelledError) as cancelled:
        await applying
    assert cancelled.value.args == ("unquiescent-identity-apply",)

    assert harness.engine._terminal_latched is True
    assert harness.engine._node_execution_tracker.active_node_names == frozenset({"identity_apply"})
    if publication == "before":
        assert harness.identity.current() == _CUST1
        assert harness.caller_context.pending_transition() is None
    else:
        assert harness.identity.current() == _CUST2
        assert harness.caller_context.inspect_principal_transition().outcome == "coherent"
    unstable = harness.engine._graph.get_state(harness.engine._config)
    assert unstable.values.get("automation_terminal", False) is False
    assert unstable.values.get("pending_recovery") is None
    blocked = await _events(harness, "continue while cleanup is pending")
    assert [event.text for event in _spoken(blocked)] == [AUTOMATION_TERMINAL_LINE]
    assert harness.engine._node_execution_tracker.active_node_names == frozenset({"identity_apply"})
    unchanged = harness.engine._graph.get_state(harness.engine._config)
    assert unchanged.values == unstable.values
    assert unchanged.next == unstable.next
    assert unchanged.tasks == unstable.tasks
    assert unchanged.interrupts == unstable.interrupts

    cleanup_complete = threading.Event()
    harness.engine._node_execution_tracker.defer_until_fully_idle(cleanup_complete.set)
    release.set()
    assert await asyncio.to_thread(cleanup_complete.wait, 5)

    assert harness.engine._node_execution_tracker.active_node_names == frozenset()
    assert harness.identity.current() is None
    assert not harness.identity.has_residual_order_authority()
    assert harness.verification.current_level() == 1
    assert harness.verification.grants == []
    assert harness.caller_context.pending_transition() is None
    snapshot = harness.engine._graph.get_state(harness.engine._config)
    assert snapshot.values.get("pending_recovery") is None
    assert snapshot.values["automation_terminal"] is True
    assert snapshot.next == ()
    repeated = await _events(harness, "continue")
    assert [event.text for event in _spoken(repeated)] == [AUTOMATION_TERMINAL_LINE]


async def test_deferred_terminal_cleanup_yields_to_session_close_without_recreating_thread(
    config_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _identity_harness(
        config_root,
        customer_claim="casey@example.com",
        reason_code="switch_account",
        thread_id="unquiescent-principal-close-race",
    )
    assert harness.verification.verify_otp(_OTP)
    harness.identity.bind(_CUST1)
    await _events(harness, "switch my account")
    await _events(harness, "yes")

    entered = threading.Event()
    release = threading.Event()
    real_fresh_proof_since = harness.verification.fresh_proof_since

    def pause_before_publication(grant_count: int):
        entered.set()
        if not release.wait(timeout=5):
            raise RuntimeError("test did not release identity_apply")
        return real_fresh_proof_since(grant_count)

    monkeypatch.setattr(
        harness.verification,
        "fresh_proof_since",
        pause_before_publication,
    )
    monkeypatch.setattr(
        harness.engine._node_execution_tracker,
        "wait_until_mutable_idle",
        lambda _timeout_seconds: False,
    )

    applying = asyncio.create_task(_events(harness, _OTP))
    assert await asyncio.to_thread(entered.wait, 5)
    applying.cancel("unquiescent-close-race")
    with pytest.raises(asyncio.CancelledError):
        await applying

    harness.caller_context.close_session()
    assert harness.engine._node_execution_tracker.active_node_names == frozenset({"identity_apply"})
    assert harness.engine._graph.get_state(harness.engine._config).values

    close_complete = threading.Event()
    harness.engine._node_execution_tracker.defer_until_fully_idle(close_complete.set)
    release.set()
    assert await asyncio.to_thread(close_complete.wait, 5)

    assert harness.identity.current() is None
    assert not harness.identity.has_residual_order_authority()
    assert harness.verification.current_level() == 1
    assert harness.caller_context.pending_transition() is None
    assert harness.engine._graph.get_state(harness.engine._config).values == {}
    terminal = await _events(harness, "continue")
    assert _spoken(terminal) == []
    assert harness.engine._graph.get_state(harness.engine._config).values == {}


async def test_old_thread_yes_cannot_resume_new_thread_confirmation(
    config_root: Path,
    tmp_path: Path,
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
    real_write = identity_flow.write_event

    def pause_after_publication(record: dict[str, object]) -> None:
        if record.get("event") == "identity_bound":
            published.set()
            if not release_apply.wait(timeout=5):
                raise RuntimeError("test did not release identity_apply")
        real_write(record)

    monkeypatch.setattr(identity_flow, "write_event", pause_after_publication)
    completing = asyncio.create_task(_events(harness, _OTP))
    assert await asyncio.to_thread(published.wait, 5)
    queued_arrived = asyncio.Event()
    real_get_state = engine._graph.get_state

    def observe_queued_arrival(config):
        task = asyncio.current_task()
        if task is not None and task.get_name() == "queued-old-thread":
            queued_arrived.set()
        return real_get_state(config)

    monkeypatch.setattr(engine._graph, "get_state", observe_queued_arrival)
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
    assert engine.pending_interrupt()
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
        event
        for event in _telemetry(tmp_path)
        if event.get("event") == "cross_thread_turn_consumed"
    ]
    assert len(consumed) == 1


async def test_checkpointer_and_takeover_failure_latches_terminal_without_later_graph_work(
    config_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _identity_harness(config_root, thread_id="terminal-latch")
    read_calls = 0
    takeover_calls = 0

    def fail_get_state(_config):
        nonlocal read_calls
        read_calls += 1
        raise RuntimeError("injected checkpointer read failure")

    def fail_takeover(_config, _values, *, as_node=None):
        nonlocal takeover_calls
        takeover_calls += 1
        assert as_node == "automation_terminal_response"
        raise RuntimeError("injected checkpoint takeover failure")

    monkeypatch.setattr(harness.engine._graph, "get_state", fail_get_state)
    monkeypatch.setattr(harness.engine._graph, "update_state", fail_takeover)

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
    real_identity_write = identity_flow.write_event

    def fail_identity_tail(record: dict[str, object]) -> None:
        if record.get("event") == "identity_bound":
            raise RuntimeError("injected identity tail failure")
        real_identity_write(record)

    def fail_recovery_event(_record: dict[str, object]) -> None:
        raise RuntimeError("injected recovery infrastructure failure")

    monkeypatch.setattr(identity_flow, "write_event", fail_identity_tail)
    monkeypatch.setattr(recovery, "write_event", fail_recovery_event)
    events = await _events(harness, _OTP)
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
    real_identity_write = identity_flow.write_event

    def fail_identity_tail(record: dict[str, object]) -> None:
        if record.get("event") == "identity_bound":
            raise RuntimeError("injected identity tail failure")
        real_identity_write(record)

    def fail_clear() -> dict[str, object]:
        raise RuntimeError("injected terminalizer failure")

    monkeypatch.setattr(identity_flow, "write_event", fail_identity_tail)
    monkeypatch.setattr(recovery, "clear_automation_state", fail_clear)
    events = await _events(harness, _OTP)
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
        {"automation_terminal": True},
        as_node="entry",
    )
    real_write = frontline_graph.write_event

    def fail_terminal_response(record: dict[str, object]) -> None:
        if record.get("event") == "automation_terminal_response":
            raise RuntimeError("injected terminal response failure")
        real_write(record)

    monkeypatch.setattr(frontline_graph, "write_event", fail_terminal_response)
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
    assert harness.verification.verify_otp(_OTP)
    transition = harness.caller_context.transition_principal(
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
        real_update = engine._graph.update_state

        def intercept_seed(config, values, *, as_node=None):
            if config["configurable"]["thread_id"].startswith("new-thread-"):
                if failure_point == "seed":
                    raise RuntimeError("injected rotation seed failure")
                real_update(config, values, as_node=as_node)
                return real_update(
                    config,
                    {"identity_claim_misses": 1},
                    as_node="__start__",
                )
            return real_update(config, values, as_node=as_node)

        monkeypatch.setattr(engine._graph, "update_state", intercept_seed)
    elif failure_point == "delete":
        real_delete = engine._graph.checkpointer.delete_thread

        def fail_old_delete(thread_id: str) -> None:
            if thread_id == old_thread:
                raise RuntimeError("injected old-thread delete failure")
            real_delete(thread_id)

        monkeypatch.setattr(engine._graph.checkpointer, "delete_thread", fail_old_delete)
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
