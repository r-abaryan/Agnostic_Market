"""Registry-fenced PostgreSQL checkpoint data plane."""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.base import SerializerProtocol
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from pydantic import ValidationError

from agnostic_market.dtos.state import CheckpointSchemaError
from agnostic_market.durability.checkpoint_encryption import (
    CHECKPOINT_PAYLOAD_ENVELOPE_SCHEMA_VERSION,
    CheckpointCipherScope,
    PendingWriteManifest,
    PendingWriteManifestEntry,
    seal_pending_write_manifest,
    verify_pending_write_manifest,
)
from agnostic_market.durability.encryption import AesGcmSessionCipher, SessionEnvelope
from agnostic_market.durability.session_registry import (
    CheckpointGeneration,
    SessionCloseAuthority,
    SessionLeaseAuthority,
    SessionLifecycle,
)


class CheckpointDataPlaneReason(StrEnum):
    SESSION_NOT_FOUND = "session_not_found"
    WRONG_DEPLOYMENT = "wrong_deployment"
    WRONG_GRAPH_CONTRACT = "wrong_graph_contract"
    STALE_CONFIG = "stale_config"
    WRONG_TRANSPORT = "wrong_transport"
    SESSION_EXPIRED = "session_expired"
    LIFECYCLE_REJECTED = "lifecycle_rejected"
    LEASE_EXPIRED = "lease_expired"
    WRONG_LEASE_OWNER = "wrong_lease_owner"
    STALE_FENCE = "stale_fence"
    UNREGISTERED_GENERATION = "unregistered_generation"
    GENERATION_NOT_WRITABLE = "generation_not_writable"
    GENERATION_NOT_DELETABLE = "generation_not_deletable"
    MANIFEST_MISSING = "manifest_missing"


class CheckpointDataPlaneError(RuntimeError):
    def __init__(self, reason: CheckpointDataPlaneReason) -> None:
        self.reason = reason
        super().__init__(f"checkpoint data-plane operation rejected: {reason.value}")


type CheckpointOperation = Literal["read", "write", "delete"]
type CheckpointAuthority = SessionLeaseAuthority | SessionCloseAuthority


def _config_value(config: RunnableConfig, name: str, *, default: str | None = None) -> str:
    configurable = config.get("configurable")
    if not isinstance(configurable, Mapping):
        raise CheckpointSchemaError("checkpoint storage configuration is malformed")
    value = configurable.get(name, default)
    if not isinstance(value, str) or (name != "checkpoint_ns" and not value):
        raise CheckpointSchemaError("checkpoint storage configuration is malformed")
    return value


def _manifest_entry(row: Mapping[str, object]) -> PendingWriteManifestEntry:
    try:
        blob = row["blob"]
        if blob is None:
            blob_bytes = b""
        elif isinstance(blob, (bytes, bytearray, memoryview)):
            blob_bytes = bytes(blob)
        else:
            raise TypeError("pending-write blob is not binary")
        return PendingWriteManifestEntry(
            task_id=row["task_id"],
            task_path=row["task_path"],
            index=row["idx"],
            channel=row["channel"],
            type_tag=row["type"],
            blob_digest=hashlib.sha256(blob_bytes).hexdigest(),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CheckpointSchemaError("persisted pending-write row is malformed") from exc


class FencedPostgresCheckpointSaver(BaseCheckpointSaver):
    """Delegate LangGraph storage inside one registry-authorized transaction."""

    def __init__(
        self,
        pool: AsyncConnectionPool,
        *,
        authority: CheckpointAuthority,
        cipher: AesGcmSessionCipher,
        serde: SerializerProtocol,
    ) -> None:
        super().__init__(serde=serde)
        self._pool = pool
        self._authority = authority
        self._cipher = cipher
        self._versioning = AsyncPostgresSaver(pool, serde=serde)

    @property
    def config_specs(self) -> list:
        return self._versioning.config_specs

    def get_next_version(self, current, channel):
        return self._versioning.get_next_version(current, channel)

    def _generation_from_row(self, row: Mapping[str, object]) -> CheckpointGeneration:
        try:
            return CheckpointGeneration(
                tenant_id=self._authority.tenant_id,
                logical_session_id=self._authority.authority.logical_session_id,
                deployment_id=self._authority.deployment_id,
                graph_contract=self._authority.graph_contract,
                checkpoint_namespace=row["checkpoint_namespace"],
                storage_thread_id=row["storage_thread_id"],
                fencing_generation=row["generation_fence"],
                principal_generation=row["generation_principal"],
                binding_version=row["binding_version"],
                state=row["generation_state"],
                transition_id=row["transition_id"],
                source_revision=row["source_revision"],
            )
        except (KeyError, TypeError, ValidationError) as exc:
            raise CheckpointSchemaError("checkpoint generation row is malformed") from exc

    def _validate_session(self, row: Mapping[str, object]) -> None:
        authority = self._authority
        transport = authority.authority.transport
        now = row.get("database_now")
        if not isinstance(now, datetime):
            raise CheckpointSchemaError("checkpoint authority query returned invalid time")
        if row.get("deployment_id") != authority.deployment_id:
            raise CheckpointDataPlaneError(CheckpointDataPlaneReason.WRONG_DEPLOYMENT)
        if row.get("graph_contract") != authority.graph_contract:
            raise CheckpointDataPlaneError(CheckpointDataPlaneReason.WRONG_GRAPH_CONTRACT)
        if row.get("config_version") != authority.config_version:
            raise CheckpointDataPlaneError(CheckpointDataPlaneReason.STALE_CONFIG)
        if (
            row.get("transport_provider") != transport.provider
            or row.get("transport_room_id") != transport.room_id
            or row.get("transport_assignment_id") != transport.assignment_id
            or row.get("transport_worker_id") != transport.worker_id
        ):
            raise CheckpointDataPlaneError(CheckpointDataPlaneReason.WRONG_TRANSPORT)
        lifecycle = row.get("lifecycle")
        if isinstance(authority, SessionCloseAuthority):
            if lifecycle != SessionLifecycle.CLOSING.value:
                raise CheckpointDataPlaneError(CheckpointDataPlaneReason.LIFECYCLE_REJECTED)
            if row.get("close_operation_id") != authority.operation_id:
                raise CheckpointDataPlaneError(CheckpointDataPlaneReason.WRONG_LEASE_OWNER)
        else:
            expires_at = row.get("expires_at")
            if not isinstance(expires_at, datetime) or expires_at <= now:
                raise CheckpointDataPlaneError(CheckpointDataPlaneReason.SESSION_EXPIRED)
            if lifecycle not in {
                SessionLifecycle.OPENING.value,
                SessionLifecycle.ACTIVE.value,
            }:
                raise CheckpointDataPlaneError(CheckpointDataPlaneReason.LIFECYCLE_REJECTED)
        lease_expires_at = row.get("lease_expires_at")
        if not isinstance(lease_expires_at, datetime) or lease_expires_at <= now:
            raise CheckpointDataPlaneError(CheckpointDataPlaneReason.LEASE_EXPIRED)
        if row.get("lease_owner_id") != authority.lease_owner_id:
            raise CheckpointDataPlaneError(CheckpointDataPlaneReason.WRONG_LEASE_OWNER)
        if row.get("session_fence") != authority.fencing_generation:
            raise CheckpointDataPlaneError(CheckpointDataPlaneReason.STALE_FENCE)

    async def _authorized_generation(
        self,
        connection: AsyncConnection,
        thread_id: str,
        *,
        operation: CheckpointOperation,
    ) -> CheckpointGeneration:
        authority = self._authority
        await connection.execute(
            "SELECT set_config('agnostic_market.tenant_id', %s, true)",
            (authority.tenant_id,),
        )
        async with connection.cursor(row_factory=dict_row) as cursor:
            await cursor.execute(
                """
                SELECT
                    session_row.lifecycle,
                    session_row.deployment_id,
                    session_row.graph_contract,
                    session_row.config_version,
                    session_row.fencing_generation AS session_fence,
                    session_row.lease_owner_id,
                    session_row.lease_expires_at,
                    session_row.close_operation_id,
                    session_row.expires_at,
                    session_row.transport_provider,
                    session_row.transport_room_id,
                    session_row.transport_assignment_id,
                    session_row.transport_worker_id,
                    generation.checkpoint_namespace,
                    generation.storage_thread_id,
                    generation.fencing_generation AS generation_fence,
                    generation.principal_generation AS generation_principal,
                    generation.binding_version,
                    generation.state AS generation_state,
                    generation.transition_id,
                    generation.source_revision,
                    clock_timestamp() AS database_now
                FROM platform_sessions AS session_row
                LEFT JOIN platform_checkpoint_generations AS generation
                  ON generation.tenant_id = session_row.tenant_id
                 AND generation.logical_session_id = session_row.logical_session_id
                WHERE session_row.tenant_id = %s
                  AND session_row.logical_session_id = %s
                FOR UPDATE OF session_row
                """,
                (authority.tenant_id, authority.authority.logical_session_id),
            )
            rows = await cursor.fetchall()
        if not rows:
            raise CheckpointDataPlaneError(CheckpointDataPlaneReason.SESSION_NOT_FOUND)
        self._validate_session(rows[0])
        generation = next(
            (
                candidate
                for row in rows
                if row.get("checkpoint_namespace") is not None
                and (candidate := self._generation_from_row(row)).storage_thread_id == thread_id
            ),
            None,
        )
        if generation is None:
            raise CheckpointDataPlaneError(CheckpointDataPlaneReason.UNREGISTERED_GENERATION)
        if operation == "write" and (
            isinstance(authority, SessionCloseAuthority)
            or generation.state not in {"current", "pending"}
        ):
            raise CheckpointDataPlaneError(CheckpointDataPlaneReason.GENERATION_NOT_WRITABLE)
        if (
            operation == "delete"
            and not isinstance(authority, SessionCloseAuthority)
            and generation.state not in {"retired", "deleted"}
        ):
            raise CheckpointDataPlaneError(CheckpointDataPlaneReason.GENERATION_NOT_DELETABLE)
        return generation

    @asynccontextmanager
    async def _operation(
        self,
        thread_id: str,
        *,
        operation: CheckpointOperation,
    ) -> AsyncIterator[tuple[AsyncConnection, AsyncPostgresSaver, CheckpointGeneration]]:
        async with self._pool.connection() as connection, connection.transaction():
            generation = await self._authorized_generation(
                connection,
                thread_id,
                operation=operation,
            )
            yield connection, AsyncPostgresSaver(connection, serde=self.serde), generation

    @staticmethod
    def _scope(
        generation: CheckpointGeneration,
        langgraph_namespace: str,
    ) -> CheckpointCipherScope:
        return generation.binding.encryption_scope(langgraph_namespace)

    async def _observed_manifest(
        self,
        connection: AsyncConnection,
        *,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
    ) -> PendingWriteManifest:
        async with connection.cursor(row_factory=dict_row) as cursor:
            await cursor.execute(
                """
                SELECT task_id, task_path, idx, channel, type, blob
                FROM checkpoint_writes
                WHERE thread_id = %s AND checkpoint_ns = %s AND checkpoint_id = %s
                ORDER BY task_id, idx
                """,
                (thread_id, checkpoint_ns, checkpoint_id),
            )
            return PendingWriteManifest(
                entries=tuple(_manifest_entry(row) for row in await cursor.fetchall())
            )

    async def _store_manifest(
        self,
        connection: AsyncConnection,
        generation: CheckpointGeneration,
        *,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
    ) -> None:
        manifest = await self._observed_manifest(
            connection,
            thread_id=thread_id,
            checkpoint_ns=checkpoint_ns,
            checkpoint_id=checkpoint_id,
        )
        envelope = seal_pending_write_manifest(
            manifest,
            self._scope(generation, checkpoint_ns),
            checkpoint_id,
            self._cipher,
        )
        await connection.execute(
            """
            INSERT INTO platform_checkpoint_write_manifests (
                tenant_id, logical_session_id, checkpoint_generation_namespace,
                langgraph_checkpoint_namespace, checkpoint_id, envelope_format,
                envelope_key_version, payload_schema_version, envelope_nonce,
                encrypted_manifest, updated_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, clock_timestamp())
            ON CONFLICT (
                tenant_id, logical_session_id, checkpoint_generation_namespace,
                langgraph_checkpoint_namespace, checkpoint_id
            ) DO UPDATE SET
                envelope_format = EXCLUDED.envelope_format,
                envelope_key_version = EXCLUDED.envelope_key_version,
                payload_schema_version = EXCLUDED.payload_schema_version,
                envelope_nonce = EXCLUDED.envelope_nonce,
                encrypted_manifest = EXCLUDED.encrypted_manifest,
                updated_at = clock_timestamp()
            """,
            (
                generation.tenant_id,
                generation.logical_session_id,
                generation.checkpoint_namespace,
                checkpoint_ns,
                checkpoint_id,
                envelope.format,
                envelope.key_version,
                envelope.payload_schema_version,
                envelope.nonce,
                envelope.ciphertext,
            ),
        )

    async def _load_manifest(
        self,
        connection: AsyncConnection,
        generation: CheckpointGeneration,
        *,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
    ) -> SessionEnvelope | None:
        async with connection.cursor(row_factory=dict_row) as cursor:
            await cursor.execute(
                """
                SELECT envelope_format, envelope_key_version, payload_schema_version,
                    envelope_nonce, encrypted_manifest
                FROM platform_checkpoint_write_manifests
                WHERE tenant_id = %s AND logical_session_id = %s
                  AND checkpoint_generation_namespace = %s
                  AND langgraph_checkpoint_namespace = %s AND checkpoint_id = %s
                """,
                (
                    generation.tenant_id,
                    generation.logical_session_id,
                    generation.checkpoint_namespace,
                    checkpoint_ns,
                    checkpoint_id,
                ),
            )
            row = await cursor.fetchone()
        if row is None:
            return None
        try:
            envelope = SessionEnvelope.model_validate(
                {
                    "format": row["envelope_format"],
                    "key_version": row["envelope_key_version"],
                    "payload_schema_version": row["payload_schema_version"],
                    "nonce": row["envelope_nonce"],
                    "ciphertext": row["encrypted_manifest"],
                }
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CheckpointSchemaError("persisted pending-write manifest is malformed") from exc
        if envelope.payload_schema_version != CHECKPOINT_PAYLOAD_ENVELOPE_SCHEMA_VERSION:
            raise CheckpointSchemaError("persisted pending-write manifest schema is unsupported")
        return envelope

    async def _verify_manifest(
        self,
        connection: AsyncConnection,
        generation: CheckpointGeneration,
        *,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
    ) -> None:
        envelope = await self._load_manifest(
            connection,
            generation,
            thread_id=thread_id,
            checkpoint_ns=checkpoint_ns,
            checkpoint_id=checkpoint_id,
        )
        if envelope is None:
            raise CheckpointDataPlaneError(CheckpointDataPlaneReason.MANIFEST_MISSING)
        await self._verify_observed_manifest(
            connection,
            generation,
            envelope,
            thread_id=thread_id,
            checkpoint_ns=checkpoint_ns,
            checkpoint_id=checkpoint_id,
        )

    async def _verify_observed_manifest(
        self,
        connection: AsyncConnection,
        generation: CheckpointGeneration,
        envelope: SessionEnvelope,
        *,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
    ) -> None:
        observed = await self._observed_manifest(
            connection,
            thread_id=thread_id,
            checkpoint_ns=checkpoint_ns,
            checkpoint_id=checkpoint_id,
        )
        verify_pending_write_manifest(
            envelope,
            observed,
            self._scope(generation, checkpoint_ns),
            checkpoint_id,
            self._cipher,
        )

    async def _authorize_checkpoint_manifest_transition(
        self,
        connection: AsyncConnection,
        generation: CheckpointGeneration,
        *,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
    ) -> None:
        envelope = await self._load_manifest(
            connection,
            generation,
            thread_id=thread_id,
            checkpoint_ns=checkpoint_ns,
            checkpoint_id=checkpoint_id,
        )
        if envelope is not None:
            await self._verify_observed_manifest(
                connection,
                generation,
                envelope,
                thread_id=thread_id,
                checkpoint_ns=checkpoint_ns,
                checkpoint_id=checkpoint_id,
            )
            return
        async with connection.cursor(row_factory=dict_row) as cursor:
            await cursor.execute(
                """
                SELECT
                    EXISTS (
                        SELECT 1 FROM checkpoints
                        WHERE thread_id = %s AND checkpoint_ns = %s AND checkpoint_id = %s
                    ) AS checkpoint_exists,
                    EXISTS (
                        SELECT 1 FROM checkpoint_writes
                        WHERE thread_id = %s AND checkpoint_ns = %s AND checkpoint_id = %s
                    ) AS pending_writes_exist
                """,
                (
                    thread_id,
                    checkpoint_ns,
                    checkpoint_id,
                    thread_id,
                    checkpoint_ns,
                    checkpoint_id,
                ),
            )
            existing = await cursor.fetchone()
        if existing is None or existing["checkpoint_exists"] or existing["pending_writes_exist"]:
            raise CheckpointDataPlaneError(CheckpointDataPlaneReason.MANIFEST_MISSING)

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        thread_id = _config_value(config, "thread_id")
        async with self._operation(thread_id, operation="read") as (
            connection,
            delegate,
            generation,
        ):
            saved = await delegate.aget_tuple(config)
            if saved is not None:
                saved_config = saved.config
                await self._verify_manifest(
                    connection,
                    generation,
                    thread_id=thread_id,
                    checkpoint_ns=_config_value(saved_config, "checkpoint_ns", default=""),
                    checkpoint_id=_config_value(saved_config, "checkpoint_id"),
                )
            return saved

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        if config is None:
            raise CheckpointSchemaError("unscoped checkpoint listing is forbidden")
        thread_id = _config_value(config, "thread_id")
        saved_items: list[CheckpointTuple] = []
        async with self._operation(thread_id, operation="read") as (
            connection,
            delegate,
            generation,
        ):
            async for saved in delegate.alist(
                config,
                filter=filter,
                before=before,
                limit=limit,
            ):
                await self._verify_manifest(
                    connection,
                    generation,
                    thread_id=thread_id,
                    checkpoint_ns=_config_value(saved.config, "checkpoint_ns", default=""),
                    checkpoint_id=_config_value(saved.config, "checkpoint_id"),
                )
                saved_items.append(saved)
        for saved in saved_items:
            yield saved

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        thread_id = _config_value(config, "thread_id")
        checkpoint_ns = _config_value(config, "checkpoint_ns", default="")
        checkpoint_id = checkpoint.get("id")
        if not isinstance(checkpoint_id, str) or not checkpoint_id:
            raise CheckpointSchemaError("checkpoint id is malformed")
        async with self._operation(thread_id, operation="write") as (
            connection,
            delegate,
            generation,
        ):
            await self._authorize_checkpoint_manifest_transition(
                connection,
                generation,
                thread_id=thread_id,
                checkpoint_ns=checkpoint_ns,
                checkpoint_id=checkpoint_id,
            )
            next_config = await delegate.aput(config, checkpoint, metadata, new_versions)
            await self._store_manifest(
                connection,
                generation,
                thread_id=thread_id,
                checkpoint_ns=checkpoint_ns,
                checkpoint_id=_config_value(next_config, "checkpoint_id"),
            )
            return next_config

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        thread_id = _config_value(config, "thread_id")
        checkpoint_ns = _config_value(config, "checkpoint_ns", default="")
        checkpoint_id = _config_value(config, "checkpoint_id")
        async with self._operation(thread_id, operation="write") as (
            connection,
            delegate,
            generation,
        ):
            await self._authorize_checkpoint_manifest_transition(
                connection,
                generation,
                thread_id=thread_id,
                checkpoint_ns=checkpoint_ns,
                checkpoint_id=checkpoint_id,
            )
            await delegate.aput_writes(config, writes, task_id, task_path)
            await self._store_manifest(
                connection,
                generation,
                thread_id=thread_id,
                checkpoint_ns=checkpoint_ns,
                checkpoint_id=checkpoint_id,
            )

    async def adelete_thread(self, thread_id: str) -> None:
        async with self._operation(thread_id, operation="delete") as (
            connection,
            delegate,
            generation,
        ):
            await delegate.adelete_thread(thread_id)
            await connection.execute(
                """
                DELETE FROM platform_checkpoint_write_manifests
                WHERE tenant_id = %s AND logical_session_id = %s
                  AND checkpoint_generation_namespace = %s
                """,
                (
                    generation.tenant_id,
                    generation.logical_session_id,
                    generation.checkpoint_namespace,
                ),
            )
