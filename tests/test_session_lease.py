"""Process-local supervision of one durable session lease."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from agnostic_market.dtos.session import AdmittedSessionAuthority, TransportAuthority
from agnostic_market.durability.encryption import SessionEnvelope
from agnostic_market.durability.session_lease import (
    LeaseLoss,
    LeaseLossReason,
    SessionLeaseSupervisor,
)
from agnostic_market.durability.session_registry import (
    LeaseAdmissionError,
    LeaseAdmissionReason,
    SessionLeaseRenewal,
    SessionLifecycle,
    SessionRegistryRecord,
)


def _renewal() -> SessionLeaseRenewal:
    return SessionLeaseRenewal(
        tenant_id="acme_store",
        authority=AdmittedSessionAuthority(
            logical_session_id="AD_session",
            transport=TransportAuthority(
                provider="livekit",
                room_id="RM_room",
                assignment_id="AJ_job",
                worker_id="AW_worker",
            ),
        ),
        deployment_id="deployment-a",
        graph_contract="graph-a",
        config_version="config-a",
        lease_owner_id="lease-owner",
        fencing_generation=1,
        duration_seconds=30.0,
    )


def _record(*, lease_seconds: float = 30.0) -> SessionRegistryRecord:
    now = datetime.now(UTC)
    return SessionRegistryRecord(
        tenant_id="acme_store",
        authority=_renewal().authority,
        lifecycle=SessionLifecycle.ACTIVE,
        checkpoint_namespace="AD_session::fence::1",
        deployment_id="deployment-a",
        graph_contract="graph-a",
        config_version="config-a",
        principal_generation=0,
        session_revision=0,
        fencing_generation=1,
        lease_owner_id="lease-owner",
        lease_expires_at=now + timedelta(seconds=lease_seconds),
        expires_at=now + timedelta(hours=1),
        envelope=SessionEnvelope(
            format="aes_256_gcm_v1",
            key_version="key-v1",
            payload_schema_version=1,
            nonce=b"0" * 12,
            ciphertext=b"0" * 16,
        ),
        created_at=now - timedelta(seconds=1),
        updated_at=now,
    )


class _Registry:
    def __init__(self, outcomes: list[object] | None = None) -> None:
        self.outcomes = list(outcomes or ())
        self.renewed = asyncio.Event()
        self.renewed_twice = asyncio.Event()
        self.calls = 0

    async def renew(self, renewal: SessionLeaseRenewal):
        self.calls += 1
        self.renewed.set()
        if self.calls >= 2:
            self.renewed_twice.set()
        outcome = self.outcomes.pop(0) if self.outcomes else _record()
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


async def test_supervisor_renews_until_closed_without_reporting_lease_loss() -> None:
    registry = _Registry()
    losses: list[LeaseLoss] = []
    supervisor = SessionLeaseSupervisor(
        registry,
        _renewal(),
        renewal_interval_seconds=0.001,
        on_lease_lost=losses.append,
    )

    supervisor.start()
    async with asyncio.timeout(0.1):
        await registry.renewed_twice.wait()
    await supervisor.aclose()

    assert registry.calls >= 2
    assert losses == []


async def test_supervisor_reports_a_closed_rejection_once() -> None:
    registry = _Registry([LeaseAdmissionError(LeaseAdmissionReason.STALE_FENCE)])
    losses: list[LeaseLoss] = []
    supervisor = SessionLeaseSupervisor(
        registry,
        _renewal(),
        renewal_interval_seconds=0.001,
        on_lease_lost=losses.append,
    )

    supervisor.start()
    await supervisor.wait()
    await supervisor.aclose()

    assert losses == [
        LeaseLoss(
            reason=LeaseLossReason.REJECTED,
            rejection=LeaseAdmissionReason.STALE_FENCE,
        )
    ]


async def test_supervisor_retires_before_the_database_lease_deadline() -> None:
    registry = _Registry([_record(lease_seconds=0.01)])
    losses: list[LeaseLoss] = []
    loss_reported = asyncio.Event()

    def record_loss(loss: LeaseLoss) -> None:
        losses.append(loss)
        loss_reported.set()

    supervisor = SessionLeaseSupervisor(
        registry,
        _renewal(),
        renewal_interval_seconds=0.05,
        on_lease_lost=record_loss,
    )

    supervisor.start()
    async with asyncio.timeout(0.03):
        await loss_reported.wait()
    await supervisor.aclose()

    assert [loss.reason.value for loss in losses] == ["lease_deadline"]
    assert registry.calls == 1


async def test_supervisor_reports_unexpected_renewal_failure_without_exposing_it() -> None:
    registry = _Registry([RuntimeError("sensitive provider detail")])
    losses: list[LeaseLoss] = []
    supervisor = SessionLeaseSupervisor(
        registry,
        _renewal(),
        renewal_interval_seconds=0.001,
        on_lease_lost=losses.append,
    )

    supervisor.start()
    await supervisor.wait()

    assert losses == [LeaseLoss(reason=LeaseLossReason.RENEWAL_FAILED)]
    assert "sensitive" not in repr(losses[0])


@pytest.mark.parametrize(
    ("failure", "reason"),
    (
        (TimeoutError(), LeaseLossReason.RENEWAL_TIMEOUT),
        (asyncio.CancelledError(), LeaseLossReason.CANCELLED),
    ),
)
async def test_supervisor_reports_timeout_or_cancelled_renewal_as_lease_loss(
    failure: BaseException,
    reason: LeaseLossReason,
) -> None:
    registry = _Registry([failure])
    losses: list[LeaseLoss] = []
    loss_reported = asyncio.Event()

    def record_loss(loss: LeaseLoss) -> None:
        losses.append(loss)
        loss_reported.set()

    supervisor = SessionLeaseSupervisor(
        registry,
        _renewal(),
        renewal_interval_seconds=0.001,
        on_lease_lost=record_loss,
    )

    supervisor.start()
    await loss_reported.wait()
    await supervisor.aclose()

    assert losses == [LeaseLoss(reason=reason)]


async def test_supervisor_rejects_duplicate_start() -> None:
    supervisor = SessionLeaseSupervisor(
        _Registry(),
        _renewal(),
        renewal_interval_seconds=1.0,
        on_lease_lost=lambda _loss: None,
    )
    supervisor.start()
    try:
        with pytest.raises(RuntimeError, match="already started"):
            supervisor.start()
    finally:
        await supervisor.aclose()
