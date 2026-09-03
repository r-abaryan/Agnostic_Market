"""Platform-session registry schema and domain contracts."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from agnostic_market.dtos.session import AdmittedSessionAuthority, TransportAuthority
from agnostic_market.durability.encryption import AesGcmSessionCipher, SessionEnvelopeContext
from agnostic_market.durability.migrations import (
    PLATFORM_MIGRATIONS,
    PLATFORM_SESSION_SCHEMA_VERSION,
)
from agnostic_market.durability.session_registry import (
    SessionLifecycle,
    SessionRegistration,
    SessionRegistryRecord,
)

_KEY = bytes(range(32))


def _authority() -> AdmittedSessionAuthority:
    return AdmittedSessionAuthority(
        logical_session_id="AD_session",
        transport=TransportAuthority(
            provider="livekit",
            assignment_id="AJ_job",
            worker_id="AW_worker",
        ),
    )


def _envelope():
    context = SessionEnvelopeContext(
        tenant_id="acme_store",
        logical_session_id="AD_session",
        checkpoint_namespace="cp_generation_0",
        payload_schema_version=1,
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


def test_session_registration_requires_timezone_aware_expiry() -> None:
    with pytest.raises(ValidationError, match="timezone"):
        SessionRegistration(
            tenant_id="acme_store",
            authority=_authority(),
            checkpoint_namespace="cp_generation_0",
            deployment_id="deployment-a",
            graph_contract="graph-a",
            config_version="config-a",
            principal_generation=0,
            session_revision=0,
            expires_at=datetime.now(),
            payload_schema_version=1,
        )


def test_session_registration_does_not_accept_a_detached_envelope() -> None:
    now = datetime.now(UTC)

    with pytest.raises(ValidationError, match="extra_forbidden"):
        SessionRegistration.model_validate(
            {
                "tenant_id": "acme_store",
                "authority": _authority(),
                "checkpoint_namespace": "cp_generation_0",
                "deployment_id": "deployment-a",
                "graph_contract": "graph-a",
                "config_version": "config-a",
                "principal_generation": 0,
                "session_revision": 0,
                "expires_at": now + timedelta(hours=1),
                "payload_schema_version": 1,
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
