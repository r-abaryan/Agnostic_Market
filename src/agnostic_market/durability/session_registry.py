"""Authoritative platform-session registry boundary."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Mapping
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Annotated, LiteralString, Protocol, Self, runtime_checkable

from psycopg import Error as PsycopgError
from psycopg import sql
from psycopg.errors import UniqueViolation
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
)

from agnostic_market.dtos.session import (
    AdmittedSessionAuthority,
    AuthorityIdentifier,
)
from agnostic_market.durability.encryption import (
    AesGcmSessionCipher,
    SessionEnvelope,
    SessionEnvelopeContext,
)

_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True)
_AUTHORITY_IDENTIFIER = TypeAdapter(AuthorityIdentifier)
DurationSeconds = Annotated[
    float,
    Field(gt=0, lt=timedelta.max.total_seconds(), allow_inf_nan=False),
]

_RETURNING_COLUMNS: LiteralString = """
tenant_id,
logical_session_id,
lifecycle,
checkpoint_namespace,
deployment_id,
graph_contract,
config_version,
principal_generation,
session_revision,
fencing_generation,
lease_owner_id,
lease_expires_at,
transport_provider,
transport_room_id,
transport_assignment_id,
transport_worker_id,
expires_at,
envelope_format,
envelope_key_version,
payload_schema_version,
envelope_nonce,
encrypted_payload,
created_at,
updated_at
"""


class SessionRegistryError(RuntimeError):
    """A registry operation could not establish an authoritative result."""


class SessionRegistryDataError(SessionRegistryError):
    """A stored registry row violates the executable domain schema."""


class LeaseAdmissionReason(StrEnum):
    SESSION_EXISTS = "session_exists"
    SESSION_NOT_FOUND = "session_not_found"
    WRONG_DEPLOYMENT = "wrong_deployment"
    WRONG_GRAPH_CONTRACT = "wrong_graph_contract"
    STALE_CONFIG = "stale_config"
    SESSION_EXPIRED = "session_expired"
    LIFECYCLE_REJECTED = "lifecycle_rejected"
    WRONG_TRANSPORT = "wrong_transport"
    LEASE_EXPIRED = "lease_expired"
    WRONG_LEASE_OWNER = "wrong_lease_owner"
    STALE_FENCE = "stale_fence"


class LeaseAdmissionError(SessionRegistryError):
    def __init__(self, reason: LeaseAdmissionReason) -> None:
        self.reason = reason
        super().__init__(f"session lease admission rejected: {reason.value}")


class SessionLifecycle(StrEnum):
    OPENING = "opening"
    ACTIVE = "active"
    CLOSING = "closing"
    CLOSED = "closed"


def _require_aware(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("session registry timestamps must include a timezone")
    return value


class SessionContract(BaseModel):
    model_config = _STRICT

    tenant_id: AuthorityIdentifier
    authority: AdmittedSessionAuthority
    deployment_id: AuthorityIdentifier
    graph_contract: AuthorityIdentifier
    config_version: AuthorityIdentifier


class SessionRegistration(SessionContract):
    principal_generation: int = Field(ge=0)
    session_revision: int = Field(ge=0)
    retention_seconds: DurationSeconds
    payload_schema_version: int = Field(ge=1)


class SessionLeaseRequest(BaseModel):
    model_config = _STRICT

    lease_owner_id: AuthorityIdentifier
    duration_seconds: DurationSeconds


class SessionLeaseRenewal(SessionContract):
    lease: SessionLeaseRequest
    fencing_generation: int = Field(ge=1)


class SessionRegistryRecord(BaseModel):
    model_config = _STRICT

    tenant_id: AuthorityIdentifier
    authority: AdmittedSessionAuthority
    lifecycle: SessionLifecycle
    checkpoint_namespace: AuthorityIdentifier
    deployment_id: AuthorityIdentifier
    graph_contract: AuthorityIdentifier
    config_version: AuthorityIdentifier
    principal_generation: int = Field(ge=0)
    session_revision: int = Field(ge=0)
    fencing_generation: int = Field(ge=0)
    lease_owner_id: AuthorityIdentifier | None
    lease_expires_at: datetime | None
    expires_at: datetime
    envelope: SessionEnvelope | None
    created_at: datetime
    updated_at: datetime

    _validate_timestamps = field_validator(
        "lease_expires_at",
        "expires_at",
        "created_at",
        "updated_at",
    )(_require_aware)

    @model_validator(mode="after")
    def _validate_state(self) -> Self:
        if (self.lease_owner_id is None) != (self.lease_expires_at is None):
            raise ValueError("lease owner and expiry must be present together")
        if self.lifecycle is SessionLifecycle.CLOSED and self.envelope is not None:
            raise ValueError("closed sessions cannot retain encrypted payloads")
        if self.lifecycle is SessionLifecycle.CLOSED and self.lease_owner_id is not None:
            raise ValueError("closed sessions cannot retain a lease")
        if self.lifecycle is not SessionLifecycle.CLOSED and self.envelope is None:
            raise ValueError("open sessions require an encrypted payload")
        if self.lifecycle is not SessionLifecycle.CLOSED and self.lease_owner_id is None:
            raise ValueError("open sessions require a lease")
        if self.lifecycle is not SessionLifecycle.CLOSED and self.fencing_generation < 1:
            raise ValueError("open sessions require a positive fencing generation")
        if self.checkpoint_namespace != _checkpoint_namespace(
            self.authority.logical_session_id,
            self.fencing_generation,
        ):
            raise ValueError("checkpoint namespace does not match the session fence")
        if self.lease_expires_at is not None and self.lease_expires_at <= self.created_at:
            raise ValueError("lease expiry must follow session creation")
        if self.lease_expires_at is not None and self.lease_expires_at > self.expires_at:
            raise ValueError("lease expiry cannot exceed session expiry")
        if self.updated_at < self.created_at:
            raise ValueError("session update time cannot precede creation")
        return self


@runtime_checkable
class SessionRegistryPort(Protocol):
    async def register_and_acquire(
        self,
        registration: SessionRegistration,
        lease: SessionLeaseRequest,
        *,
        payload: bytes,
    ) -> SessionRegistryRecord: ...

    async def renew(self, renewal: SessionLeaseRenewal) -> SessionRegistryRecord: ...

    async def get(
        self,
        tenant_id: str,
        logical_session_id: str,
    ) -> SessionRegistryRecord | None: ...


def _record_from_row(row: Mapping[str, object]) -> SessionRegistryRecord:
    try:
        envelope = None
        if row["encrypted_payload"] is not None:
            envelope = {
                "format": row["envelope_format"],
                "key_version": row["envelope_key_version"],
                "payload_schema_version": row["payload_schema_version"],
                "nonce": row["envelope_nonce"],
                "ciphertext": row["encrypted_payload"],
            }
        return SessionRegistryRecord.model_validate(
            {
                "tenant_id": row["tenant_id"],
                "authority": {
                    "logical_session_id": row["logical_session_id"],
                    "transport": {
                        "provider": row["transport_provider"],
                        "room_id": row["transport_room_id"],
                        "assignment_id": row["transport_assignment_id"],
                        "worker_id": row["transport_worker_id"],
                    },
                },
                "lifecycle": SessionLifecycle(str(row["lifecycle"])),
                "checkpoint_namespace": row["checkpoint_namespace"],
                "deployment_id": row["deployment_id"],
                "graph_contract": row["graph_contract"],
                "config_version": row["config_version"],
                "principal_generation": row["principal_generation"],
                "session_revision": row["session_revision"],
                "fencing_generation": row["fencing_generation"],
                "lease_owner_id": row["lease_owner_id"],
                "lease_expires_at": row["lease_expires_at"],
                "expires_at": row["expires_at"],
                "envelope": envelope,
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SessionRegistryDataError("stored session registry row is invalid") from exc


def _checkpoint_namespace(logical_session_id: str, fencing_generation: int) -> str:
    return f"{logical_session_id}::fence::{fencing_generation}"


def _lease_duration(seconds: float) -> timedelta:
    if not math.isfinite(seconds) or seconds <= 0:
        raise SessionRegistryDataError("lease duration must be positive and finite")
    try:
        return timedelta(seconds=seconds)
    except OverflowError as exc:
        raise SessionRegistryDataError("lease duration is outside the supported range") from exc


def _transport_values(authority: AdmittedSessionAuthority) -> tuple[str, str, str, str]:
    transport = authority.transport
    return (
        transport.provider,
        transport.room_id,
        transport.assignment_id,
        transport.worker_id,
    )


def _renewal_rejection(
    record: SessionRegistryRecord,
    renewal: SessionLeaseRenewal,
    *,
    database_now: datetime,
) -> LeaseAdmissionReason | None:
    if record.expires_at <= database_now:
        return LeaseAdmissionReason.SESSION_EXPIRED
    if record.lifecycle not in {SessionLifecycle.OPENING, SessionLifecycle.ACTIVE}:
        return LeaseAdmissionReason.LIFECYCLE_REJECTED
    if record.deployment_id != renewal.deployment_id:
        return LeaseAdmissionReason.WRONG_DEPLOYMENT
    if record.graph_contract != renewal.graph_contract:
        return LeaseAdmissionReason.WRONG_GRAPH_CONTRACT
    if record.config_version != renewal.config_version:
        return LeaseAdmissionReason.STALE_CONFIG
    if record.authority != renewal.authority:
        return LeaseAdmissionReason.WRONG_TRANSPORT
    if record.lease_expires_at is None or record.lease_expires_at <= database_now:
        return LeaseAdmissionReason.LEASE_EXPIRED
    if record.lease_owner_id != renewal.lease.lease_owner_id:
        return LeaseAdmissionReason.WRONG_LEASE_OWNER
    if record.fencing_generation != renewal.fencing_generation:
        return LeaseAdmissionReason.STALE_FENCE
    return None


class PostgresSessionRegistry(SessionRegistryPort):
    def __init__(
        self,
        pool: AsyncConnectionPool,
        *,
        cipher: AesGcmSessionCipher,
        operation_timeout_seconds: float,
    ) -> None:
        if not math.isfinite(operation_timeout_seconds) or operation_timeout_seconds <= 0:
            raise ValueError("registry operation timeout must be positive")
        self._pool = pool
        self._cipher = cipher
        self._operation_timeout_seconds = operation_timeout_seconds

    async def register_and_acquire(
        self,
        registration: SessionRegistration,
        lease: SessionLeaseRequest,
        *,
        payload: bytes,
    ) -> SessionRegistryRecord:
        row = None
        try:
            async with asyncio.timeout(self._operation_timeout_seconds):
                fencing_generation = 1
                checkpoint_namespace = _checkpoint_namespace(
                    registration.authority.logical_session_id,
                    fencing_generation,
                )
                lease_duration = _lease_duration(lease.duration_seconds)
                retention_duration = _lease_duration(registration.retention_seconds)
                if lease_duration >= retention_duration:
                    raise SessionRegistryDataError(
                        "session retention must be longer than the initial lease"
                    )
                envelope = self._cipher.encrypt(
                    payload,
                    SessionEnvelopeContext(
                        tenant_id=registration.tenant_id,
                        logical_session_id=registration.authority.logical_session_id,
                        checkpoint_namespace=checkpoint_namespace,
                        payload_schema_version=registration.payload_schema_version,
                    ),
                )
                async with (
                    self._pool.connection() as connection,
                    connection.transaction(),
                ):
                    await connection.execute(
                        "SELECT set_config('agnostic_market.tenant_id', %s, true)",
                        (registration.tenant_id,),
                    )
                    async with connection.cursor(row_factory=dict_row) as cursor:
                        query = sql.SQL(
                            """
                            WITH authority_time AS (
                                SELECT clock_timestamp() AS now
                            )
                            INSERT INTO platform_sessions (
                                tenant_id,
                                logical_session_id,
                                lifecycle,
                                checkpoint_namespace,
                                deployment_id,
                                graph_contract,
                                config_version,
                                principal_generation,
                                session_revision,
                                fencing_generation,
                                transport_provider,
                                transport_room_id,
                                transport_assignment_id,
                                transport_worker_id,
                                lease_owner_id,
                                lease_expires_at,
                                expires_at,
                                envelope_format,
                                envelope_key_version,
                                payload_schema_version,
                                envelope_nonce,
                                encrypted_payload
                            ) SELECT
                                %s, %s, 'opening', %s, %s, %s, %s, %s, %s, %s,
                                %s, %s, %s, %s, %s,
                                authority_time.now + %s,
                                authority_time.now + %s,
                                %s, %s, %s, %s, %s
                            FROM authority_time
                            RETURNING {}
                            """
                        ).format(sql.SQL(_RETURNING_COLUMNS))
                        await cursor.execute(
                            query,
                            (
                                registration.tenant_id,
                                registration.authority.logical_session_id,
                                checkpoint_namespace,
                                registration.deployment_id,
                                registration.graph_contract,
                                registration.config_version,
                                registration.principal_generation,
                                registration.session_revision,
                                fencing_generation,
                                *_transport_values(registration.authority),
                                lease.lease_owner_id,
                                lease_duration,
                                retention_duration,
                                envelope.format,
                                envelope.key_version,
                                envelope.payload_schema_version,
                                envelope.nonce,
                                envelope.ciphertext,
                            ),
                        )
                        row = await cursor.fetchone()
        except UniqueViolation as exc:
            raise LeaseAdmissionError(LeaseAdmissionReason.SESSION_EXISTS) from exc
        except TimeoutError:
            raise
        except ValueError as exc:
            raise SessionRegistryDataError("session payload cannot be encrypted") from exc
        except PsycopgError as exc:
            raise SessionRegistryError("session registration failed") from exc
        if row is None:
            raise SessionRegistryError("session registration returned no authoritative row")
        return _record_from_row(row)

    async def renew(self, renewal: SessionLeaseRenewal) -> SessionRegistryRecord:
        row = None
        try:
            async with asyncio.timeout(self._operation_timeout_seconds):
                lease_duration = _lease_duration(renewal.lease.duration_seconds)
                async with (
                    self._pool.connection() as connection,
                    connection.transaction(),
                ):
                    await connection.execute(
                        "SELECT set_config('agnostic_market.tenant_id', %s, true)",
                        (renewal.tenant_id,),
                    )
                    async with connection.cursor(row_factory=dict_row) as cursor:
                        query = sql.SQL(
                            """
                            SELECT {}, clock_timestamp() AS database_now
                            FROM platform_sessions
                            WHERE tenant_id = %s AND logical_session_id = %s
                            FOR UPDATE
                            """
                        ).format(sql.SQL(_RETURNING_COLUMNS))
                        await cursor.execute(
                            query,
                            (
                                renewal.tenant_id,
                                renewal.authority.logical_session_id,
                            ),
                        )
                        locked = await cursor.fetchone()
                        if locked is None:
                            raise LeaseAdmissionError(LeaseAdmissionReason.SESSION_NOT_FOUND)
                        record = _record_from_row(locked)
                        database_now = locked.get("database_now")
                        if not isinstance(database_now, datetime):
                            raise SessionRegistryDataError(
                                "session lease read returned no database time"
                            )
                        if reason := _renewal_rejection(
                            record,
                            renewal,
                            database_now=database_now,
                        ):
                            raise LeaseAdmissionError(reason)
                        query = sql.SQL(
                            """
                            WITH authority_time AS (
                                SELECT clock_timestamp() AS now
                            )
                            UPDATE platform_sessions
                            SET lease_expires_at = GREATEST(
                                    lease_expires_at,
                                    LEAST(expires_at, authority_time.now + %s)
                                ),
                                updated_at = authority_time.now
                            FROM authority_time
                            WHERE tenant_id = %s
                              AND logical_session_id = %s
                              AND lease_owner_id = %s
                              AND fencing_generation = %s
                              AND expires_at > authority_time.now
                              AND lease_expires_at > authority_time.now
                            RETURNING {}
                            """
                        ).format(sql.SQL(_RETURNING_COLUMNS))
                        await cursor.execute(
                            query,
                            (
                                lease_duration,
                                renewal.tenant_id,
                                renewal.authority.logical_session_id,
                                renewal.lease.lease_owner_id,
                                renewal.fencing_generation,
                            ),
                        )
                        row = await cursor.fetchone()
                        if row is None:
                            await cursor.execute("SELECT clock_timestamp()")
                            current_time = await cursor.fetchone()
                            database_now = (
                                None
                                if current_time is None
                                else current_time.get("clock_timestamp")
                            )
                            if not isinstance(database_now, datetime):
                                raise SessionRegistryDataError(
                                    "session lease update returned no database time"
                                )
                            reason = (
                                LeaseAdmissionReason.SESSION_EXPIRED
                                if record.expires_at <= database_now
                                else LeaseAdmissionReason.LEASE_EXPIRED
                            )
                            raise LeaseAdmissionError(reason)
        except (LeaseAdmissionError, SessionRegistryDataError):
            raise
        except TimeoutError:
            raise
        except ValueError as exc:
            raise SessionRegistryDataError("session lease renewal is invalid") from exc
        except PsycopgError as exc:
            raise SessionRegistryError("session lease renewal failed") from exc
        if row is None:
            raise SessionRegistryError("session lease renewal returned no authoritative row")
        return _record_from_row(row)

    async def get(
        self,
        tenant_id: str,
        logical_session_id: str,
    ) -> SessionRegistryRecord | None:
        tenant_id = _AUTHORITY_IDENTIFIER.validate_python(tenant_id)
        logical_session_id = _AUTHORITY_IDENTIFIER.validate_python(logical_session_id)
        row = None
        try:
            async with asyncio.timeout(self._operation_timeout_seconds):
                async with self._pool.connection() as connection:
                    async with connection.transaction():
                        await connection.execute(
                            "SELECT set_config('agnostic_market.tenant_id', %s, true)",
                            (tenant_id,),
                        )
                        async with connection.cursor(row_factory=dict_row) as cursor:
                            query = sql.SQL(
                                """
                                SELECT {}
                                FROM platform_sessions
                                WHERE tenant_id = %s AND logical_session_id = %s
                                """
                            ).format(sql.SQL(_RETURNING_COLUMNS))
                            await cursor.execute(
                                query,
                                (tenant_id, logical_session_id),
                            )
                            row = await cursor.fetchone()
        except TimeoutError:
            raise
        except PsycopgError as exc:
            raise SessionRegistryError("session registry read failed") from exc
        return None if row is None else _record_from_row(row)
