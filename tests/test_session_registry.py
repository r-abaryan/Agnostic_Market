"""Platform-session registry schema and domain contracts."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from agnostic_market.checkpoints import CheckpointBinding
from agnostic_market.dtos.session import AdmittedSessionAuthority, TransportAuthority
from agnostic_market.durability.encryption import AesGcmSessionCipher, SessionEnvelopeContext
from agnostic_market.durability.migrations import (
    PLATFORM_MIGRATIONS,
    PLATFORM_SESSION_SCHEMA_VERSION,
)
from agnostic_market.durability.session_lifecycle import DurableSessionLifecycleCoordinator
from agnostic_market.durability.session_payload import (
    SESSION_PAYLOAD_SCHEMA_VERSION,
    DurableSessionPayload,
    SessionOperationReceiptPayload,
)
from agnostic_market.durability.session_registry import (
    ExpiredSessionCandidate,
    InMemoryCheckpointGenerationAuthority,
    LeaseAdmissionError,
    LeaseAdmissionReason,
    SessionCloseError,
    SessionCloseReason,
    SessionLeaseAuthority,
    SessionLeaseRenewal,
    SessionLeaseRequest,
    SessionLifecycle,
    SessionRegistration,
    SessionRegistryRecord,
    SessionStateWriteError,
    SessionStateWriteReason,
)

_KEY = bytes(range(32))


class _ReaperRegistry:
    def __init__(self) -> None:
        self.candidates = (
            ExpiredSessionCandidate(
                tenant_id="acme_store",
                logical_session_id="AD_poisoned",
            ),
            ExpiredSessionCandidate(
                tenant_id="acme_store",
                logical_session_id="AD_healthy",
            ),
        )
        self.purge_calls = 0

    async def expired_sessions(self, tenant_id: str, *, limit: int):
        assert tenant_id == "acme_store"
        assert limit == 100
        return self.candidates

    async def claim_expired(self, candidate, _request):
        return candidate

    async def purge_closed_tombstones(self, tenant_id: str, *, limit: int) -> int:
        assert tenant_id == "acme_store"
        assert limit == 100
        self.purge_calls += 1
        return 0


def _authority() -> AdmittedSessionAuthority:
    return AdmittedSessionAuthority(
        logical_session_id="AD_session",
        transport=TransportAuthority(
            provider="livekit",
            room_id="RM_room",
            assignment_id="AJ_job",
            worker_id="AW_worker",
        ),
    )


def _envelope():
    context = SessionEnvelopeContext(
        tenant_id="acme_store",
        logical_session_id="AD_session",
        checkpoint_namespace="cp_generation_0",
        payload_schema_version=SESSION_PAYLOAD_SCHEMA_VERSION,
        payload_purpose="session_projection",
        session_revision=0,
    )
    return AesGcmSessionCipher(active_key_version="key-v1", keys={"key-v1": _KEY}).encrypt(
        b'{"cart":[]}',
        context,
    )


def test_platform_migration_inventory_is_contiguous_and_complete() -> None:
    assert tuple(migration.version for migration in PLATFORM_MIGRATIONS) == tuple(
        range(1, PLATFORM_SESSION_SCHEMA_VERSION + 1)
    )
    assert all(len(migration.checksum) == 64 for migration in PLATFORM_MIGRATIONS)


async def test_reaper_isolates_candidate_failures_and_still_purges_tombstones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _ReaperRegistry()
    coordinator = DurableSessionLifecycleCoordinator(
        None,  # type: ignore[arg-type]
        registry,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        checkpoint_io_timeout_seconds=1.0,
        close_lease_duration_seconds=1.0,
        tombstone_retention_seconds=1.0,
    )
    finalized: list[str] = []

    async def finalize(candidate: ExpiredSessionCandidate):
        finalized.append(candidate.logical_session_id)
        if candidate.logical_session_id == "AD_poisoned":
            raise RuntimeError("poisoned checkpoint")

    monkeypatch.setattr(coordinator, "finalize", finalize)

    with pytest.raises(ExceptionGroup, match="cleanup operations failed") as failed:
        await coordinator.reap_expired("acme_store")

    assert finalized == ["AD_poisoned", "AD_healthy"]
    assert registry.purge_calls == 1
    assert len(failed.value.exceptions) == 1
    assert failed.value.exceptions[0].__notes__ == [
        "failed reaper candidate acme_store/AD_poisoned"
    ]


async def test_reaper_treats_a_concurrent_close_claim_as_a_benign_race(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _ReaperRegistry()
    coordinator = DurableSessionLifecycleCoordinator(
        None,  # type: ignore[arg-type]
        registry,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        checkpoint_io_timeout_seconds=1.0,
        close_lease_duration_seconds=1.0,
        tombstone_retention_seconds=1.0,
    )
    finalized: list[str] = []

    async def claim_expired(candidate: ExpiredSessionCandidate, _request):
        if candidate.logical_session_id == "AD_poisoned":
            raise SessionCloseError(SessionCloseReason.WRONG_CLOSE_OWNER)
        return candidate

    async def finalize(candidate: ExpiredSessionCandidate):
        finalized.append(candidate.logical_session_id)

    monkeypatch.setattr(registry, "claim_expired", claim_expired)
    monkeypatch.setattr(coordinator, "finalize", finalize)

    assert await coordinator.reap_expired("acme_store") == 1
    assert finalized == ["AD_healthy"]
    assert registry.purge_calls == 1


async def test_reaper_treats_ownership_loss_during_finalization_as_a_benign_race(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _ReaperRegistry()
    coordinator = DurableSessionLifecycleCoordinator(
        None,  # type: ignore[arg-type]
        registry,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        checkpoint_io_timeout_seconds=1.0,
        close_lease_duration_seconds=1.0,
        tombstone_retention_seconds=1.0,
    )
    finalized: list[str] = []

    async def finalize(candidate: ExpiredSessionCandidate):
        finalized.append(candidate.logical_session_id)
        if candidate.logical_session_id == "AD_poisoned":
            raise SessionCloseError(SessionCloseReason.WRONG_CLOSE_OWNER)

    monkeypatch.setattr(coordinator, "finalize", finalize)

    assert await coordinator.reap_expired("acme_store") == 1
    assert finalized == ["AD_poisoned", "AD_healthy"]
    assert registry.purge_calls == 1


async def test_in_memory_checkpoint_authority_models_one_registry_owned_rotation() -> None:
    authority = InMemoryCheckpointGenerationAuthority.from_binding(
        CheckpointBinding(
            tenant_id="acme_store",
            logical_session_id="AD_session",
            deployment_id="deployment-a",
            graph_contract="graph-a",
            thread_id="AD_session::fence::0",
        )
    )

    destination = await authority.begin_checkpoint_rotation(
        expected_revision=1,
        transition_id="principal-transition-a",
    )
    replay = await authority.begin_checkpoint_rotation(
        expected_revision=1,
        transition_id="principal-transition-a",
    )

    assert replay == destination
    assert destination.state == "pending"
    assert authority.current_generation.principal_generation == 0
    with pytest.raises(SessionStateWriteError) as pending_rejected:
        await authority.begin_checkpoint_rotation(
            expected_revision=1,
            transition_id="principal-transition-b",
        )
    assert pending_rejected.value.reason is SessionStateWriteReason.ROTATION_PENDING

    rotation = await authority.switch_checkpoint_generation("principal-transition-a")
    deleted = await authority.record_checkpoint_deletion("principal-transition-a")

    assert rotation.source.state == "retired"
    assert rotation.destination.state == "current"
    assert authority.current_generation == rotation.destination
    assert deleted.checkpoint_namespace == rotation.source.checkpoint_namespace
    assert deleted.state == "deleted"

    second = await authority.begin_checkpoint_rotation(
        expected_revision=2,
        transition_id="principal-transition-b",
    )
    await authority.switch_checkpoint_generation("principal-transition-b")

    assert second.principal_generation == 2
    with pytest.raises(SessionStateWriteError) as replay_rejected:
        await authority.begin_checkpoint_rotation(
            expected_revision=1,
            transition_id="principal-transition-a",
        )
    assert replay_rejected.value.reason is SessionStateWriteReason.OPERATION_CONFLICT


@pytest.mark.parametrize("payload_type", [DurableSessionPayload, SessionOperationReceiptPayload])
def test_current_session_payload_models_reject_version_1(payload_type: type) -> None:
    with pytest.raises(ValidationError, match="schema_version"):
        payload_type.model_validate({"schema_version": 1})


def test_transport_authority_requires_a_room_identity() -> None:
    with pytest.raises(ValidationError, match="room_id"):
        TransportAuthority(
            provider="livekit",
            assignment_id="AJ_job",
            worker_id="AW_worker",
        )


def test_session_registration_requires_positive_finite_retention() -> None:
    for retention in (
        0.0,
        -1.0,
        float("inf"),
        float("nan"),
        timedelta.max.total_seconds(),
    ):
        with pytest.raises(ValidationError, match="retention_seconds"):
            SessionRegistration(
                tenant_id="acme_store",
                authority=_authority(),
                deployment_id="deployment-a",
                graph_contract="graph-a",
                config_version="config-a",
                principal_generation=0,
                session_revision=0,
                retention_seconds=retention,
            )


def test_session_registration_rejects_a_caller_selected_checkpoint_namespace() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        SessionRegistration(
            tenant_id="acme_store",
            authority=_authority(),
            deployment_id="deployment-a",
            graph_contract="graph-a",
            config_version="config-a",
            principal_generation=0,
            session_revision=0,
            retention_seconds=3600.0,
            checkpoint_namespace="caller-selected",
        )


def test_session_registration_does_not_accept_a_detached_envelope() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        SessionRegistration.model_validate(
            {
                "tenant_id": "acme_store",
                "authority": _authority(),
                "deployment_id": "deployment-a",
                "graph_contract": "graph-a",
                "config_version": "config-a",
                "principal_generation": 0,
                "session_revision": 0,
                "retention_seconds": 3600.0,
                "envelope": _envelope(),
            }
        )


@pytest.mark.parametrize(
    "lifecycle, envelope",
    (
        (SessionLifecycle.OPENING, None),
        (SessionLifecycle.ACTIVE, None),
        (SessionLifecycle.CLOSING, None),
        (SessionLifecycle.CLOSED, _envelope()),
    ),
)
def test_session_record_enforces_payload_lifecycle(
    lifecycle: SessionLifecycle,
    envelope: object,
) -> None:
    now = datetime.now(UTC)
    with pytest.raises(ValidationError, match="payload"):
        SessionRegistryRecord(
            tenant_id="acme_store",
            authority=_authority(),
            lifecycle=lifecycle,
            checkpoint_namespace="cp_generation_0",
            deployment_id="deployment-a",
            graph_contract="graph-a",
            config_version="config-a",
            principal_generation=0,
            session_revision=0,
            fencing_generation=0,
            lease_owner_id=None,
            lease_expires_at=None,
            expires_at=now + timedelta(hours=1),
            envelope=envelope,
            created_at=now,
            updated_at=now,
        )


def test_session_record_requires_complete_lease_identity() -> None:
    now = datetime.now(UTC)
    values = {
        "tenant_id": "acme_store",
        "authority": _authority(),
        "lifecycle": SessionLifecycle.OPENING,
        "checkpoint_namespace": "cp_generation_0",
        "deployment_id": "deployment-a",
        "graph_contract": "graph-a",
        "config_version": "config-a",
        "principal_generation": 0,
        "session_revision": 0,
        "fencing_generation": 0,
        "lease_owner_id": "lease-owner",
        "lease_expires_at": None,
        "expires_at": now + timedelta(hours=1),
        "envelope": _envelope(),
        "created_at": now,
        "updated_at": now,
    }

    with pytest.raises(ValidationError, match="lease owner"):
        SessionRegistryRecord.model_validate(values)


@pytest.mark.parametrize(
    ("lease_owner_id", "lease_expires_at", "fencing_generation", "message"),
    (
        (None, None, 1, "require a lease"),
        ("lease-owner", timedelta(minutes=1), 0, "positive fencing"),
    ),
)
def test_open_session_record_requires_fenced_lease_authority(
    lease_owner_id: str | None,
    lease_expires_at: timedelta | None,
    fencing_generation: int,
    message: str,
) -> None:
    now = datetime.now(UTC)

    with pytest.raises(ValidationError, match=message):
        SessionRegistryRecord(
            tenant_id="acme_store",
            authority=_authority(),
            lifecycle=SessionLifecycle.OPENING,
            checkpoint_namespace="AD_session::fence::1",
            deployment_id="deployment-a",
            graph_contract="graph-a",
            config_version="config-a",
            principal_generation=0,
            session_revision=0,
            fencing_generation=fencing_generation,
            lease_owner_id=lease_owner_id,
            lease_expires_at=(None if lease_expires_at is None else now + lease_expires_at),
            expires_at=now + timedelta(hours=1),
            envelope=_envelope(),
            created_at=now,
            updated_at=now,
        )


def test_closed_session_record_cannot_retain_a_lease() -> None:
    now = datetime.now(UTC)

    with pytest.raises(ValidationError, match="closed sessions cannot retain a lease"):
        SessionRegistryRecord(
            tenant_id="acme_store",
            authority=_authority(),
            lifecycle=SessionLifecycle.CLOSED,
            checkpoint_namespace="cp_generation_0",
            deployment_id="deployment-a",
            graph_contract="graph-a",
            config_version="config-a",
            principal_generation=0,
            session_revision=0,
            fencing_generation=1,
            lease_owner_id="lease-owner",
            lease_expires_at=now + timedelta(minutes=1),
            expires_at=now + timedelta(hours=1),
            envelope=None,
            created_at=now,
            updated_at=now,
        )


def test_session_record_rejects_a_lease_beyond_session_expiry() -> None:
    now = datetime.now(UTC)

    with pytest.raises(ValidationError, match="lease expiry"):
        SessionRegistryRecord(
            tenant_id="acme_store",
            authority=_authority(),
            lifecycle=SessionLifecycle.OPENING,
            checkpoint_namespace="AD_session::fence::1",
            deployment_id="deployment-a",
            graph_contract="graph-a",
            config_version="config-a",
            principal_generation=0,
            session_revision=0,
            fencing_generation=1,
            lease_owner_id="lease-owner",
            lease_expires_at=now + timedelta(hours=2),
            expires_at=now + timedelta(hours=1),
            envelope=_envelope(),
            created_at=now,
            updated_at=now,
        )


def test_session_record_rejects_a_namespace_outside_its_fence() -> None:
    now = datetime.now(UTC)

    with pytest.raises(ValidationError, match="namespace"):
        SessionRegistryRecord(
            tenant_id="acme_store",
            authority=_authority(),
            lifecycle=SessionLifecycle.OPENING,
            checkpoint_namespace="caller-selected",
            deployment_id="deployment-a",
            graph_contract="graph-a",
            config_version="config-a",
            principal_generation=0,
            session_revision=0,
            fencing_generation=1,
            lease_owner_id="lease-owner",
            lease_expires_at=now + timedelta(minutes=1),
            expires_at=now + timedelta(hours=1),
            envelope=_envelope(),
            created_at=now,
            updated_at=now,
        )


def test_session_record_rejects_a_lease_that_does_not_follow_creation() -> None:
    now = datetime.now(UTC)

    with pytest.raises(ValidationError, match="follow session creation"):
        SessionRegistryRecord(
            tenant_id="acme_store",
            authority=_authority(),
            lifecycle=SessionLifecycle.OPENING,
            checkpoint_namespace="AD_session::fence::1",
            deployment_id="deployment-a",
            graph_contract="graph-a",
            config_version="config-a",
            principal_generation=0,
            session_revision=0,
            fencing_generation=1,
            lease_owner_id="lease-owner",
            lease_expires_at=now,
            expires_at=now + timedelta(hours=1),
            envelope=_envelope(),
            created_at=now,
            updated_at=now,
        )


def test_lease_requests_reject_non_positive_or_non_finite_durations() -> None:
    for duration in (
        0.0,
        -1.0,
        float("inf"),
        float("nan"),
        timedelta.max.total_seconds(),
        1e308,
    ):
        with pytest.raises(ValidationError, match="duration_seconds"):
            SessionLeaseRequest(
                lease_owner_id="lease-owner",
                duration_seconds=duration,
            )


def test_lease_renewal_requires_a_positive_expected_fence() -> None:
    with pytest.raises(ValidationError, match="fencing_generation"):
        SessionLeaseRenewal(
            tenant_id="acme_store",
            authority=_authority(),
            deployment_id="deployment-a",
            graph_contract="graph-a",
            config_version="config-a",
            lease_owner_id="lease-owner",
            fencing_generation=0,
            duration_seconds=30.0,
        )


def test_lease_authority_does_not_accept_a_renewal_duration() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        SessionLeaseAuthority(
            tenant_id="acme_store",
            authority=_authority(),
            deployment_id="deployment-a",
            graph_contract="graph-a",
            config_version="config-a",
            lease_owner_id="lease-owner",
            fencing_generation=1,
            duration_seconds=30.0,
        )


def test_lease_admission_error_exposes_only_its_closed_reason() -> None:
    error = LeaseAdmissionError(LeaseAdmissionReason.STALE_FENCE)

    assert error.reason is LeaseAdmissionReason.STALE_FENCE
    assert str(error) == "session lease admission rejected: stale_fence"
