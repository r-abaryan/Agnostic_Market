"""Authoritative platform-session registry boundary."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Mapping
from datetime import datetime
from enum import StrEnum
from typing import LiteralString, Protocol, Self, runtime_checkable

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


class SessionRegistryConflictError(SessionRegistryError):
    """The logical session or checkpoint namespace is already registered."""


class SessionRegistryDataError(SessionRegistryError):
    """A stored registry row violates the executable domain schema."""


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


class SessionRegistration(BaseModel):
    model_config = _STRICT

    tenant_id: AuthorityIdentifier
    authority: AdmittedSessionAuthority
    checkpoint_namespace: AuthorityIdentifier
    deployment_id: AuthorityIdentifier
    graph_contract: AuthorityIdentifier
    config_version: AuthorityIdentifier
    principal_generation: int = Field(ge=0)
    session_revision: int = Field(ge=0)
    expires_at: datetime
    payload_schema_version: int = Field(ge=1)

    _validate_expires_at = field_validator("expires_at")(_require_aware)


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
        if self.updated_at < self.created_at:
            raise ValueError("session update time cannot precede creation")
        return self


@runtime_checkable
class SessionRegistryPort(Protocol):
    async def create(
        self,
        registration: SessionRegistration,
        *,
        payload: bytes,
    ) -> SessionRegistryRecord: ...

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

    async def create(
        self,
        registration: SessionRegistration,
        *,
        payload: bytes,
    ) -> SessionRegistryRecord:
        row = None
        try:
            async with asyncio.timeout(self._operation_timeout_seconds):
                envelope = self._cipher.encrypt(
                    payload,
                    SessionEnvelopeContext(
                        tenant_id=registration.tenant_id,
                        logical_session_id=registration.authority.logical_session_id,
                        checkpoint_namespace=registration.checkpoint_namespace,
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
                                expires_at,
                                envelope_format,
                                envelope_key_version,
                                payload_schema_version,
                                envelope_nonce,
                                encrypted_payload
                            ) VALUES (
                                %s, %s, 'opening', %s, %s, %s, %s, %s, %s, 0,
                                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                            )
                            RETURNING {}
                            """
                        ).format(sql.SQL(_RETURNING_COLUMNS))
                        await cursor.execute(
                            query,
                            (
                                registration.tenant_id,
                                registration.authority.logical_session_id,
                                registration.checkpoint_namespace,
                                registration.deployment_id,
                                registration.graph_contract,
                                registration.config_version,
                                registration.principal_generation,
                                registration.session_revision,
                                registration.authority.transport.provider,
                                registration.authority.transport.room_id,
                                registration.authority.transport.assignment_id,
                                registration.authority.transport.worker_id,
                                registration.expires_at,
                                envelope.format,
                                envelope.key_version,
                                envelope.payload_schema_version,
                                envelope.nonce,
                                envelope.ciphertext,
                            ),
                        )
                        row = await cursor.fetchone()
        except UniqueViolation as exc:
            raise SessionRegistryConflictError(
                "session registration conflicts with existing state"
            ) from exc
        except TimeoutError:
            raise
        except ValueError as exc:
            raise SessionRegistryDataError("session payload cannot be encrypted") from exc
        except PsycopgError as exc:
            raise SessionRegistryError("session registration failed") from exc
        if row is None:
            raise SessionRegistryError("session registration returned no authoritative row")
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
