"""Authoritative platform-session registry boundary."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Mapping
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Annotated, LiteralString, Protocol, Self, runtime_checkable

from psycopg import AsyncConnection, sql
from psycopg import Error as PsycopgError
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
    SessionEnvelopeError,
)
from agnostic_market.durability.session_payload import (
    SESSION_OPERATION_RESULT_SCHEMA_VERSION,
    SESSION_PAYLOAD_SCHEMA_VERSION,
    DurableSessionPayload,
    SessionOperationReceiptPayload,
    SessionOperationResult,
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


class SessionRestoreReason(StrEnum):
    DECRYPTION_FAILED = "decryption_failed"
    PAYLOAD_SCHEMA_INVALID = "payload_schema_invalid"
    REVISION_MISMATCH = "revision_mismatch"
    CHECKPOINT_INVALID = "checkpoint_invalid"
    RECONSTRUCTION_FAILED = "reconstruction_failed"


class SessionRestoreError(SessionRegistryError):
    def __init__(self, reason: SessionRestoreReason) -> None:
        self.reason = reason
        super().__init__(f"session restore rejected: {reason.value}")


class SessionStateWriteReason(StrEnum):
    STALE_REVISION = "stale_revision"
    OPERATION_CONFLICT = "operation_conflict"


class SessionStateWriteError(SessionRegistryError):
    def __init__(self, reason: SessionStateWriteReason) -> None:
        self.reason = reason
        super().__init__(f"session state publication rejected: {reason.value}")


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


class SessionLeaseRequest(BaseModel):
    model_config = _STRICT

    lease_owner_id: AuthorityIdentifier
    duration_seconds: DurationSeconds


class SessionLeaseAuthority(SessionContract):
    lease_owner_id: AuthorityIdentifier
    fencing_generation: int = Field(ge=1)


class SessionLeaseRenewal(SessionLeaseAuthority):
    duration_seconds: DurationSeconds


class SessionStatePublication(SessionLeaseAuthority):
    expected_revision: int = Field(ge=0)
    operation_id: AuthorityIdentifier
    request_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    payload: DurableSessionPayload
    operation_result: SessionOperationResult


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


class RestoredSessionState(BaseModel):
    model_config = _STRICT

    record: SessionRegistryRecord
    payload: DurableSessionPayload
    replayed: bool = False
    operation_revision: int | None = Field(default=None, ge=1)
    operation_result: SessionOperationResult | None = None

    @model_validator(mode="after")
    def operation_receipt_matches_replay(self) -> RestoredSessionState:
        has_complete_receipt = (
            self.operation_revision is not None and self.operation_result is not None
        )
        has_partial_receipt = (
            self.operation_revision is not None or self.operation_result is not None
        )
        if (self.replayed and not has_complete_receipt) or (
            not self.replayed and has_partial_receipt
        ):
            raise ValueError("only replayed session state carries an operation receipt")
        if (
            self.operation_revision is not None
            and self.operation_revision > self.record.session_revision
        ):
            raise ValueError("operation revision cannot exceed current session revision")
        return self


@runtime_checkable
class SessionRegistryPort(Protocol):
    async def register_and_acquire(
        self,
        registration: SessionRegistration,
        lease: SessionLeaseRequest,
        *,
        payload: DurableSessionPayload,
    ) -> SessionRegistryRecord: ...

    async def activate(self, authority: SessionLeaseAuthority) -> SessionRegistryRecord: ...

    async def renew(self, renewal: SessionLeaseRenewal) -> SessionRegistryRecord: ...

    async def restore(self, authority: SessionLeaseAuthority) -> RestoredSessionState: ...

    async def publish(self, publication: SessionStatePublication) -> RestoredSessionState: ...

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


def _lease_authority_rejection(
    record: SessionRegistryRecord,
    authority: SessionLeaseAuthority,
    *,
    database_now: datetime,
) -> LeaseAdmissionReason | None:
    if record.expires_at <= database_now:
        return LeaseAdmissionReason.SESSION_EXPIRED
    if record.lifecycle not in {SessionLifecycle.OPENING, SessionLifecycle.ACTIVE}:
        return LeaseAdmissionReason.LIFECYCLE_REJECTED
    if record.deployment_id != authority.deployment_id:
        return LeaseAdmissionReason.WRONG_DEPLOYMENT
    if record.graph_contract != authority.graph_contract:
        return LeaseAdmissionReason.WRONG_GRAPH_CONTRACT
    if record.config_version != authority.config_version:
        return LeaseAdmissionReason.STALE_CONFIG
    if record.authority != authority.authority:
        return LeaseAdmissionReason.WRONG_TRANSPORT
    if record.lease_expires_at is None or record.lease_expires_at <= database_now:
        return LeaseAdmissionReason.LEASE_EXPIRED
    if record.lease_owner_id != authority.lease_owner_id:
        return LeaseAdmissionReason.WRONG_LEASE_OWNER
    if record.fencing_generation != authority.fencing_generation:
        return LeaseAdmissionReason.STALE_FENCE
    return None


async def _database_now(connection: AsyncConnection) -> datetime:
    async with connection.cursor(row_factory=dict_row) as cursor:
        await cursor.execute("SELECT clock_timestamp() AS database_now")
        row = await cursor.fetchone()
    database_now = None if row is None else row.get("database_now")
    if not isinstance(database_now, datetime):
        raise SessionRegistryDataError("session lease operation returned no database time")
    return database_now


def _envelope_context(
    record: SessionRegistryRecord,
    *,
    checkpoint_namespace: str | None = None,
    payload_schema_version: int = SESSION_PAYLOAD_SCHEMA_VERSION,
) -> SessionEnvelopeContext:
    return SessionEnvelopeContext(
        tenant_id=record.tenant_id,
        logical_session_id=record.authority.logical_session_id,
        checkpoint_namespace=(
            record.checkpoint_namespace
            if checkpoint_namespace is None
            else _AUTHORITY_IDENTIFIER.validate_python(checkpoint_namespace)
        ),
        payload_schema_version=payload_schema_version,
    )


def _operation_result_from_row(
    cipher: AesGcmSessionCipher,
    record: SessionRegistryRecord,
    row: Mapping[str, object],
    *,
    operation_id: str,
    request_fingerprint: str,
) -> SessionOperationResult:
    try:
        envelope = SessionEnvelope.model_validate(
            {
                "format": row["result_envelope_format"],
                "key_version": row["result_envelope_key_version"],
                "payload_schema_version": row["result_schema_version"],
                "nonce": row["result_envelope_nonce"],
                "ciphertext": row["encrypted_result"],
            }
        )
        committed_checkpoint_namespace = _AUTHORITY_IDENTIFIER.validate_python(
            row["committed_checkpoint_namespace"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SessionRegistryDataError("stored session operation result is invalid") from exc
    if envelope.payload_schema_version != SESSION_OPERATION_RESULT_SCHEMA_VERSION:
        raise SessionRegistryDataError("stored session operation result has an unsupported schema")
    try:
        plaintext = cipher.decrypt(
            envelope,
            _envelope_context(
                record,
                checkpoint_namespace=committed_checkpoint_namespace,
                payload_schema_version=SESSION_OPERATION_RESULT_SCHEMA_VERSION,
            ),
        )
        receipt = SessionOperationReceiptPayload.from_bytes(plaintext)
    except (SessionEnvelopeError, TypeError, ValueError) as exc:
        raise SessionRegistryDataError("stored session operation result is invalid") from exc
    if receipt.operation_id != operation_id or receipt.request_fingerprint != request_fingerprint:
        raise SessionRegistryDataError("stored session operation result has invalid authority")
    return receipt.result


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
        payload: DurableSessionPayload,
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
                    payload.to_bytes(),
                    SessionEnvelopeContext(
                        tenant_id=registration.tenant_id,
                        logical_session_id=registration.authority.logical_session_id,
                        checkpoint_namespace=checkpoint_namespace,
                        payload_schema_version=SESSION_PAYLOAD_SCHEMA_VERSION,
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

    async def _locked_record(
        self,
        connection: AsyncConnection,
        authority: SessionLeaseAuthority,
    ) -> tuple[SessionRegistryRecord, datetime]:
        await connection.execute(
            "SELECT set_config('agnostic_market.tenant_id', %s, true)",
            (authority.tenant_id,),
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
                    authority.tenant_id,
                    authority.authority.logical_session_id,
                ),
            )
            locked = await cursor.fetchone()
        if locked is None:
            raise LeaseAdmissionError(LeaseAdmissionReason.SESSION_NOT_FOUND)
        record = _record_from_row(locked)
        database_now = locked.get("database_now")
        if not isinstance(database_now, datetime):
            raise SessionRegistryDataError("session lease read returned no database time")
        if reason := _lease_authority_rejection(
            record,
            authority,
            database_now=database_now,
        ):
            raise LeaseAdmissionError(reason)
        return record, database_now

    async def activate(self, authority: SessionLeaseAuthority) -> SessionRegistryRecord:
        row = None
        try:
            async with asyncio.timeout(self._operation_timeout_seconds):
                async with (
                    self._pool.connection() as connection,
                    connection.transaction(),
                ):
                    record, _ = await self._locked_record(connection, authority)
                    if record.lifecycle is SessionLifecycle.ACTIVE:
                        return record
                    async with connection.cursor(row_factory=dict_row) as cursor:
                        query = sql.SQL(
                            """
                            WITH authority_time AS (
                                SELECT clock_timestamp() AS now
                            )
                            UPDATE platform_sessions
                            SET lifecycle = 'active',
                                updated_at = authority_time.now
                            FROM authority_time
                            WHERE tenant_id = %s
                              AND logical_session_id = %s
                              AND lifecycle = 'opening'
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
                                authority.tenant_id,
                                authority.authority.logical_session_id,
                                authority.lease_owner_id,
                                authority.fencing_generation,
                            ),
                        )
                        row = await cursor.fetchone()
                        if row is None:
                            database_now = await _database_now(connection)
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
        except PsycopgError as exc:
            raise SessionRegistryError("session activation failed") from exc
        if row is None:
            raise SessionRegistryError("session activation returned no authoritative row")
        return _record_from_row(row)

    async def renew(self, renewal: SessionLeaseRenewal) -> SessionRegistryRecord:
        row = None
        try:
            async with asyncio.timeout(self._operation_timeout_seconds):
                lease_duration = _lease_duration(renewal.duration_seconds)
                async with (
                    self._pool.connection() as connection,
                    connection.transaction(),
                ):
                    record, _ = await self._locked_record(connection, renewal)
                    async with connection.cursor(row_factory=dict_row) as cursor:
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
                                renewal.lease_owner_id,
                                renewal.fencing_generation,
                            ),
                        )
                        row = await cursor.fetchone()
                        if row is None:
                            database_now = await _database_now(connection)
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

    def _restore_payload(self, record: SessionRegistryRecord) -> DurableSessionPayload:
        if record.envelope is None:
            raise SessionRestoreError(SessionRestoreReason.PAYLOAD_SCHEMA_INVALID)
        if record.envelope.payload_schema_version != SESSION_PAYLOAD_SCHEMA_VERSION:
            raise SessionRestoreError(SessionRestoreReason.PAYLOAD_SCHEMA_INVALID)
        try:
            plaintext = self._cipher.decrypt(record.envelope, _envelope_context(record))
        except SessionEnvelopeError as exc:
            raise SessionRestoreError(SessionRestoreReason.DECRYPTION_FAILED) from exc
        try:
            return DurableSessionPayload.from_bytes(plaintext)
        except (TypeError, ValueError) as exc:
            raise SessionRestoreError(SessionRestoreReason.PAYLOAD_SCHEMA_INVALID) from exc

    async def restore(self, authority: SessionLeaseAuthority) -> RestoredSessionState:
        try:
            async with asyncio.timeout(self._operation_timeout_seconds):
                async with (
                    self._pool.connection() as connection,
                    connection.transaction(),
                ):
                    record, _ = await self._locked_record(connection, authority)
                    payload = self._restore_payload(record)
        except (LeaseAdmissionError, SessionRegistryDataError, SessionRestoreError):
            raise
        except TimeoutError:
            raise
        except PsycopgError as exc:
            raise SessionRegistryError("session state restore failed") from exc
        return RestoredSessionState(record=record, payload=payload)

    async def publish(self, publication: SessionStatePublication) -> RestoredSessionState:
        row = None
        try:
            async with asyncio.timeout(self._operation_timeout_seconds):
                async with (
                    self._pool.connection() as connection,
                    connection.transaction(),
                ):
                    record, _ = await self._locked_record(connection, publication)
                    if record.lifecycle is not SessionLifecycle.ACTIVE:
                        raise LeaseAdmissionError(LeaseAdmissionReason.LIFECYCLE_REJECTED)
                    async with connection.cursor(row_factory=dict_row) as cursor:
                        await cursor.execute(
                            """
                            SELECT
                                request_fingerprint,
                                committed_revision,
                                committed_checkpoint_namespace,
                                result_envelope_format,
                                result_envelope_key_version,
                                result_schema_version,
                                result_envelope_nonce,
                                encrypted_result
                            FROM platform_session_operations
                            WHERE tenant_id = %s
                              AND logical_session_id = %s
                              AND operation_id = %s
                            """,
                            (
                                publication.tenant_id,
                                publication.authority.logical_session_id,
                                publication.operation_id,
                            ),
                        )
                        receipt = await cursor.fetchone()
                        if receipt is not None:
                            if (
                                receipt.get("request_fingerprint")
                                != publication.request_fingerprint
                            ):
                                raise SessionStateWriteError(
                                    SessionStateWriteReason.OPERATION_CONFLICT
                                )
                            operation_revision = receipt.get("committed_revision")
                            if not isinstance(operation_revision, int):
                                raise SessionRegistryDataError(
                                    "session operation receipt has an invalid revision"
                                )
                            return RestoredSessionState(
                                record=record,
                                payload=self._restore_payload(record),
                                replayed=True,
                                operation_revision=operation_revision,
                                operation_result=_operation_result_from_row(
                                    self._cipher,
                                    record,
                                    receipt,
                                    operation_id=publication.operation_id,
                                    request_fingerprint=publication.request_fingerprint,
                                ),
                            )
                        if publication.expected_revision != record.session_revision:
                            raise SessionStateWriteError(SessionStateWriteReason.STALE_REVISION)
                        next_revision = record.session_revision + 1
                        envelope = self._cipher.encrypt(
                            publication.payload.to_bytes(),
                            _envelope_context(record),
                        )
                        operation_result = SessionOperationReceiptPayload(
                            operation_id=publication.operation_id,
                            request_fingerprint=publication.request_fingerprint,
                            result=publication.operation_result,
                        )
                        result_envelope = self._cipher.encrypt(
                            operation_result.to_bytes(),
                            _envelope_context(
                                record,
                                payload_schema_version=SESSION_OPERATION_RESULT_SCHEMA_VERSION,
                            ),
                        )
                        query = sql.SQL(
                            """
                            WITH authority_time AS (
                                SELECT clock_timestamp() AS now
                            )
                            UPDATE platform_sessions
                            SET session_revision = %s,
                                envelope_format = %s,
                                envelope_key_version = %s,
                                payload_schema_version = %s,
                                envelope_nonce = %s,
                                encrypted_payload = %s,
                                updated_at = authority_time.now
                            FROM authority_time
                            WHERE tenant_id = %s
                              AND logical_session_id = %s
                              AND lifecycle = 'active'
                              AND lease_owner_id = %s
                              AND fencing_generation = %s
                              AND session_revision = %s
                              AND expires_at > authority_time.now
                              AND lease_expires_at > authority_time.now
                            RETURNING {}
                            """
                        ).format(sql.SQL(_RETURNING_COLUMNS))
                        await cursor.execute(
                            query,
                            (
                                next_revision,
                                envelope.format,
                                envelope.key_version,
                                envelope.payload_schema_version,
                                envelope.nonce,
                                envelope.ciphertext,
                                publication.tenant_id,
                                publication.authority.logical_session_id,
                                publication.lease_owner_id,
                                publication.fencing_generation,
                                publication.expected_revision,
                            ),
                        )
                        row = await cursor.fetchone()
                        if row is None:
                            database_now = await _database_now(connection)
                            reason = (
                                LeaseAdmissionReason.SESSION_EXPIRED
                                if record.expires_at <= database_now
                                else LeaseAdmissionReason.LEASE_EXPIRED
                            )
                            raise LeaseAdmissionError(reason)
                        await cursor.execute(
                            """
                            INSERT INTO platform_session_operations (
                                tenant_id,
                                logical_session_id,
                                operation_id,
                                request_fingerprint,
                                committed_revision,
                                committed_checkpoint_namespace,
                                result_envelope_format,
                                result_envelope_key_version,
                                result_schema_version,
                                result_envelope_nonce,
                                encrypted_result
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """,
                            (
                                publication.tenant_id,
                                publication.authority.logical_session_id,
                                publication.operation_id,
                                publication.request_fingerprint,
                                next_revision,
                                record.checkpoint_namespace,
                                result_envelope.format,
                                result_envelope.key_version,
                                result_envelope.payload_schema_version,
                                result_envelope.nonce,
                                result_envelope.ciphertext,
                            ),
                        )
        except (
            LeaseAdmissionError,
            SessionRegistryDataError,
            SessionRestoreError,
            SessionStateWriteError,
        ):
            raise
        except TimeoutError:
            raise
        except PsycopgError as exc:
            raise SessionRegistryError("session state publication failed") from exc
        if row is None:
            raise SessionRegistryError("session state publication returned no authoritative row")
        return RestoredSessionState(
            record=_record_from_row(row),
            payload=publication.payload,
        )

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
