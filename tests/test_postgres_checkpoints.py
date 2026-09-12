"""Conformance for the pinned asynchronous PostgreSQL checkpoint backend."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import uuid
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import TypedDict

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import CheckpointMetadata, CheckpointTuple, empty_checkpoint
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.base import SerializerProtocol
from langgraph.graph import START, StateGraph
from psycopg import AsyncConnection, sql
from psycopg.errors import InsufficientPrivilege
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool, PoolTimeout

from agnostic_market.checkpoints import (
    CheckpointBinding,
    SchemaValidatedCheckpointSaver,
    SynchronousCheckpointOperationError,
    build_checkpoint_serializer,
    build_checkpointer,
    graph_contract_fingerprint,
)
from agnostic_market.dtos.session import AdmittedSessionAuthority, TransportAuthority
from agnostic_market.dtos.state import CheckpointSchemaError
from agnostic_market.durability.encryption import AesGcmSessionCipher
from agnostic_market.durability.migrations import (
    apply_platform_migrations,
    grant_platform_application_role,
)
from agnostic_market.durability.postgres_checkpoints import (
    CheckpointDataPlaneError,
    CheckpointDataPlaneReason,
    FencedPostgresCheckpointSaver,
)
from agnostic_market.durability.session_payload import (
    DurableSessionPayload,
    EmptySessionOperationResult,
    PrincipalRetirementMarker,
)
from agnostic_market.durability.session_registry import (
    BoundCheckpointGenerationAuthority,
    PostgresSessionRegistry,
    SessionLeaseAuthority,
    SessionLeaseRequest,
    SessionRegistration,
    SessionStatePublication,
)

_POSTGRES_DSN_ENV = "PHASE4C_POSTGRES_DSN"
_CHECKPOINT_IO_TIMEOUT_SECONDS = 2.0
_STALLED_READ_SECONDS = 10


async def _publish_retirement_marker(
    registry: PostgresSessionRegistry,
    authority: SessionLeaseAuthority,
    transition_id: str,
) -> None:
    await registry.publish(
        SessionStatePublication(
            **authority.model_dump(),
            expected_revision=0,
            operation_id=f"principal-retirement:{transition_id}",
            request_fingerprint=hashlib.sha256(transition_id.encode()).hexdigest(),
            payload=DurableSessionPayload(
                principal_retirement=PrincipalRetirementMarker(transition_id=transition_id)
            ),
            operation_result=EmptySessionOperationResult(),
        )
    )


class _State(TypedDict, total=False):
    value: int


class _SensitiveState(TypedDict, total=False):
    value: str


class _StalledReadPostgresSaver(AsyncPostgresSaver):
    def __init__(
        self,
        pool: AsyncConnectionPool,
        *,
        serde: SerializerProtocol,
    ) -> None:
        super().__init__(pool, serde=serde)
        self.stall_reads = False
        self.stalled_read_started = asyncio.Event()

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        if self.stall_reads:
            assert isinstance(self.conn, AsyncConnectionPool)
            async with self.conn.connection() as connection:
                self.stalled_read_started.set()
                await connection.execute("SELECT pg_sleep(%s)", (_STALLED_READ_SECONDS,))
        return await super().aget_tuple(config)


def _compiled(saver: SchemaValidatedCheckpointSaver):
    graph = StateGraph(_State)
    graph.add_node("increment", lambda state: {"value": state.get("value", 0) + 1})
    graph.add_edge(START, "increment")
    return graph.compile(checkpointer=saver)


def _sensitive_compiled(saver: SchemaValidatedCheckpointSaver):
    graph = StateGraph(_SensitiveState)
    graph.add_node("retain", lambda state: {"value": state["value"]})
    graph.add_edge(START, "retain")
    return graph.compile(checkpointer=saver)


def _checkpoint_cipher() -> AesGcmSessionCipher:
    return AesGcmSessionCipher(
        active_key_version="checkpoint-key-v1",
        keys={"checkpoint-key-v1": b"c" * 32},
    )


def _dsn() -> str:
    dsn = os.environ.get(_POSTGRES_DSN_ENV, "").strip()
    if not dsn:
        pytest.skip(f"{_POSTGRES_DSN_ENV} is provided by the disposable PostgreSQL harness")
    return dsn


async def _open_boundary(
    stack: AsyncExitStack,
    dsn: str,
    *,
    max_pool_size: int = 2,
    pool_timeout_seconds: float = _CHECKPOINT_IO_TIMEOUT_SECONDS,
    backend_type: type[AsyncPostgresSaver] = AsyncPostgresSaver,
    cipher: AesGcmSessionCipher | None = None,
) -> tuple[AsyncConnectionPool, AsyncPostgresSaver, SchemaValidatedCheckpointSaver]:
    pool = AsyncConnectionPool(
        dsn,
        min_size=1,
        max_size=max_pool_size,
        timeout=pool_timeout_seconds,
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
        },
        open=False,
    )
    await pool.open(wait=True, timeout=2.0)
    stack.push_async_callback(pool.close)
    backend = backend_type(pool, serde=build_checkpoint_serializer())
    return (
        pool,
        backend,
        build_checkpointer(
            backend,
            synchronous_operations=False,
            cipher=cipher,
        ),
    )


async def _open_fenced_boundary(
    stack: AsyncExitStack,
    dsn: str,
    *,
    logical_session_id: str,
    max_pool_size: int = 3,
    pool_timeout_seconds: float = _CHECKPOINT_IO_TIMEOUT_SECONDS,
) -> tuple[
    AsyncConnectionPool,
    AsyncPostgresSaver,
    SchemaValidatedCheckpointSaver,
    SessionLeaseAuthority,
    PostgresSessionRegistry,
]:
    pool = AsyncConnectionPool(
        dsn,
        min_size=1,
        max_size=max_pool_size,
        timeout=pool_timeout_seconds,
        kwargs={"autocommit": True, "prepare_threshold": 0},
        open=False,
    )
    await pool.open(wait=True, timeout=2.0)
    stack.push_async_callback(pool.close)
    async with pool.connection() as migration_connection:
        await apply_platform_migrations(migration_connection)
    serde = build_checkpoint_serializer()
    raw_backend = AsyncPostgresSaver(pool, serde=serde)
    authority = SessionLeaseAuthority(
        tenant_id="checkpoint-data-plane",
        authority=AdmittedSessionAuthority(
            logical_session_id=logical_session_id,
            transport=TransportAuthority(
                provider="livekit",
                room_id=f"RM_{logical_session_id}",
                assignment_id=f"AJ_{logical_session_id}",
                worker_id=f"AW_{logical_session_id}",
            ),
        ),
        deployment_id="phase4c-harness",
        graph_contract="checkpoint-data-plane-contract",
        config_version="config-a",
        lease_owner_id=f"owner-{logical_session_id}",
        fencing_generation=1,
    )
    registry = PostgresSessionRegistry(
        pool,
        cipher=_checkpoint_cipher(),
        operation_timeout_seconds=_CHECKPOINT_IO_TIMEOUT_SECONDS,
    )
    await registry.register_and_acquire(
        SessionRegistration(
            tenant_id=authority.tenant_id,
            authority=authority.authority,
            deployment_id=authority.deployment_id,
            graph_contract=authority.graph_contract,
            config_version=authority.config_version,
            principal_generation=0,
            session_revision=0,
            retention_seconds=3600.0,
        ),
        SessionLeaseRequest(
            lease_owner_id=authority.lease_owner_id,
            duration_seconds=300.0,
        ),
        payload=DurableSessionPayload(),
    )
    await registry.activate(authority)
    backend = FencedPostgresCheckpointSaver(
        pool,
        authority=authority,
        cipher=_checkpoint_cipher(),
        serde=serde,
    )
    return (
        pool,
        raw_backend,
        build_checkpointer(backend, synchronous_operations=False, cipher=_checkpoint_cipher()),
        authority,
        registry,
    )


def _bound_fenced_write_saver(
    pool: AsyncConnectionPool,
    authority: SessionLeaseAuthority,
    logical_session_id: str,
) -> tuple[
    SchemaValidatedCheckpointSaver,
    CheckpointBinding,
    RunnableConfig,
]:
    backend = FencedPostgresCheckpointSaver(
        pool,
        authority=authority,
        cipher=_checkpoint_cipher(),
        serde=build_checkpoint_serializer(),
    )
    saver = build_checkpointer(
        backend,
        synchronous_operations=False,
        cipher=_checkpoint_cipher(),
    )
    binding = CheckpointBinding(
        tenant_id=authority.tenant_id,
        logical_session_id=logical_session_id,
        deployment_id=authority.deployment_id,
        graph_contract=authority.graph_contract,
        thread_id=f"{logical_session_id}::fence::1",
    )
    saver.bind_checkpoint_contract(
        {"value"},
        binding=binding,
        io_timeout_seconds=1.0,
        required_state_keys=_SensitiveState.__annotations__,
    )
    write_config: RunnableConfig = {
        "configurable": {
            **binding.config["configurable"],
            "checkpoint_id": str(uuid.uuid4()),
        }
    }
    return saver, binding, write_config


@asynccontextmanager
async def _hold_session_row(
    pool: AsyncConnectionPool,
    authority: SessionLeaseAuthority,
) -> AsyncIterator[None]:
    async with pool.connection() as connection, connection.transaction():
        await connection.execute(
            "SELECT set_config('agnostic_market.tenant_id', %s, true)",
            (authority.tenant_id,),
        )
        await connection.execute(
            """
            SELECT 1 FROM platform_sessions
            WHERE tenant_id = %s AND logical_session_id = %s
            FOR UPDATE
            """,
            (authority.tenant_id, authority.authority.logical_session_id),
        )
        yield


async def _open_fenced_queue_boundary(
    stack: AsyncExitStack,
    prefix: str,
) -> tuple[
    AsyncConnectionPool,
    SchemaValidatedCheckpointSaver,
    SessionLeaseAuthority,
    RunnableConfig,
]:
    logical_session_id = f"{prefix}-{uuid.uuid4().hex}"
    pool, _raw_backend, _saver, authority, _registry = await _open_fenced_boundary(
        stack,
        _dsn(),
        logical_session_id=logical_session_id,
        max_pool_size=2,
        pool_timeout_seconds=0.1,
    )
    saver, _binding, write_config = _bound_fenced_write_saver(
        pool,
        authority,
        logical_session_id,
    )
    return pool, saver, authority, write_config


def _pending_write_task(
    saver: SchemaValidatedCheckpointSaver,
    config: RunnableConfig,
    task_id: str,
) -> asyncio.Task[None]:
    return asyncio.create_task(
        saver.aput_writes(
            config,
            (("value", task_id),),
            task_id,
            f"pull/{task_id}",
        )
    )


@pytest.mark.postgres
async def test_same_session_checkpoint_waiters_do_not_exhaust_the_pool() -> None:
    async with AsyncExitStack() as stack:
        pool, saver, authority, write_config = await _open_fenced_queue_boundary(
            stack,
            "checkpoint-queue",
        )

        async with _hold_session_row(pool, authority):
            first = _pending_write_task(saver, write_config, "queued-task-first")
            second = _pending_write_task(saver, write_config, "queued-task-second")
            await asyncio.sleep(0.2)
            assert not first.done()
            assert not second.done()

        await asyncio.gather(first, second)


@pytest.mark.postgres
async def test_cancelled_checkpoint_waiter_does_not_block_the_session_queue() -> None:
    async with AsyncExitStack() as stack:
        pool, saver, authority, write_config = await _open_fenced_queue_boundary(
            stack,
            "checkpoint-queued-cancel",
        )

        async with _hold_session_row(pool, authority):
            active = _pending_write_task(saver, write_config, "active-task")
            queued = _pending_write_task(saver, write_config, "queued-task")
            await asyncio.sleep(0.05)
            queued.cancel()
            with pytest.raises(asyncio.CancelledError):
                await queued
            assert not active.done()

        await active
        await saver.aput_writes(
            write_config,
            (("value", "after-cancel"),),
            "after-cancel-task",
            "pull/after-cancel-task",
        )


@pytest.mark.postgres
async def test_cancelled_active_checkpoint_operation_releases_pool_and_queue() -> None:
    async with AsyncExitStack() as stack:
        pool, saver, authority, write_config = await _open_fenced_queue_boundary(
            stack,
            "checkpoint-active-cancel",
        )

        async with _hold_session_row(pool, authority):
            active = _pending_write_task(
                saver,
                write_config,
                "cancelled-active-task",
            )
            await asyncio.sleep(0.05)
            active.cancel()
            with pytest.raises(asyncio.CancelledError):
                await active
            replacement = _pending_write_task(
                saver,
                write_config,
                "replacement-task",
            )
            await asyncio.sleep(0.2)
            assert not replacement.done()

        await replacement


@pytest.mark.postgres
async def test_async_postgres_saver_conforms_to_the_checkpoint_boundary() -> None:
    dsn = _dsn()
    async with AsyncExitStack() as stack:
        _first_pool, first_backend, first = await _open_boundary(stack, dsn)
        await first_backend.setup()
        _second_pool, _second_backend, second = await _open_boundary(stack, dsn)

        first_graph = _compiled(first)
        second_graph = _compiled(second)
        contract = graph_contract_fingerprint(first_graph)
        assert graph_contract_fingerprint(second_graph) == contract
        binding = CheckpointBinding(
            tenant_id="synthetic-tenant",
            deployment_id="phase4c-harness",
            graph_contract=contract,
            thread_id="shared-logical-thread",
        )
        for saver, graph in ((first, first_graph), (second, second_graph)):
            saver.bind_checkpoint_contract(
                graph.channels,
                binding=binding,
                io_timeout_seconds=_CHECKPOINT_IO_TIMEOUT_SECONDS,
                required_state_keys=_State.__annotations__,
            )

        await first_graph.ainvoke({"value": 1}, binding.config)
        assert (await second_graph.aget_state(binding.config)).values["value"] == 2

        saved = await second.aget_tuple(binding.config)
        assert saved is not None
        await first.aput_writes(saved.config, (("value", 3),), "synthetic-task")
        reread = await second.aget_tuple(binding.config)
        assert reread is not None
        assert any(write[1:] == ("value", 3) for write in reread.pending_writes)
        assert [item async for item in second.alist(binding.config, limit=1)]

        with pytest.raises(SynchronousCheckpointOperationError):
            first.get_tuple(binding.config)

        await second.adelete_thread(binding.storage_thread_id)
        assert await first.aget_tuple(binding.config) is None


@pytest.mark.postgres
async def test_encrypted_checkpoint_has_no_plaintext_payload_in_postgres() -> None:
    sentinel = "postgres-checkpoint-plaintext-sentinel-6b"
    async with AsyncExitStack() as stack:
        pool, backend, saver = await _open_boundary(
            stack,
            _dsn(),
            cipher=_checkpoint_cipher(),
        )
        await backend.setup()
        graph = _sensitive_compiled(saver)
        binding = CheckpointBinding(
            tenant_id="synthetic-tenant",
            logical_session_id="encrypted-logical-session",
            deployment_id="phase4c-harness",
            graph_contract=graph_contract_fingerprint(graph),
            thread_id="encrypted-logical-session::fence::1",
        )
        saver.bind_checkpoint_contract(
            graph.channels,
            binding=binding,
            io_timeout_seconds=_CHECKPOINT_IO_TIMEOUT_SECONDS,
            required_state_keys=_SensitiveState.__annotations__,
        )

        invoke_config: RunnableConfig = {
            "configurable": {
                "thread_id": binding.storage_thread_id,
                "checkpoint_ns": "",
                "config_marker": sentinel,
            },
            "metadata": {"request_marker": sentinel},
        }
        await graph.ainvoke({"value": sentinel}, invoke_config)
        saved = await saver.aget_tuple(binding.config)
        assert saved is not None
        next_checkpoint = {**saved.checkpoint, "id": str(uuid.uuid4())}
        next_config = await saver.aput(
            saved.config,
            next_checkpoint,
            {**saved.metadata, "sentinel": sentinel},
            saved.checkpoint["channel_versions"],
        )
        await saver.aput_writes(next_config, (("value", sentinel),), "sensitive-task")

        restored = await saver.aget_tuple(next_config)
        assert restored is not None
        assert restored.checkpoint["channel_values"]["value"] == sentinel
        assert restored.metadata["sentinel"] == sentinel
        assert restored.metadata["config_marker"] == sentinel
        assert restored.metadata["request_marker"] == sentinel
        assert any(write[1:] == ("value", sentinel) for write in restored.pending_writes)
        assert [item async for item in saver.alist(next_config, limit=1)]

        async with pool.connection() as connection:
            checkpoint_rows = await (
                await connection.execute(
                    """
                    SELECT checkpoint::text, metadata::text
                    FROM checkpoints WHERE thread_id = %s
                    """,
                    (binding.storage_thread_id,),
                )
            ).fetchall()
            blob_rows = await (
                await connection.execute(
                    """
                    SELECT blob FROM checkpoint_blobs WHERE thread_id = %s
                    UNION ALL
                    SELECT blob FROM checkpoint_writes WHERE thread_id = %s
                    """,
                    (binding.storage_thread_id, binding.storage_thread_id),
                )
            ).fetchall()
        assert checkpoint_rows
        assert blob_rows
        assert all(
            set(json.loads(row["metadata"])) == {"__agnostic_checkpoint_payload__"}
            for row in checkpoint_rows
        )
        assert all(sentinel not in str(value) for row in checkpoint_rows for value in row.values())
        assert all(
            sentinel.encode() not in row["blob"] for row in blob_rows if row["blob"] is not None
        )

        async with pool.connection() as connection:
            await connection.execute(
                """
                UPDATE checkpoints SET parent_checkpoint_id = %s
                WHERE thread_id = %s AND checkpoint_id = %s
                """,
                (
                    "tampered-parent",
                    binding.storage_thread_id,
                    next_config["configurable"]["checkpoint_id"],
                ),
            )
        with pytest.raises(CheckpointSchemaError, match="parent does not authenticate"):
            await saver.aget_tuple(next_config)

        await saver.adelete_thread(binding.storage_thread_id)


@pytest.mark.postgres
@pytest.mark.parametrize("mutation", ["delete", "duplicate", "relocate", "swap"])
async def test_fenced_checkpoint_rejects_pending_write_collection_tampering(
    mutation: str,
) -> None:
    logical_session_id = f"manifest-{mutation}-{uuid.uuid4().hex}"
    async with AsyncExitStack() as stack:
        pool, _raw_backend, saver, authority, _registry = await _open_fenced_boundary(
            stack,
            _dsn(),
            logical_session_id=logical_session_id,
        )
        binding = CheckpointBinding(
            tenant_id=authority.tenant_id,
            logical_session_id=logical_session_id,
            deployment_id=authority.deployment_id,
            graph_contract=authority.graph_contract,
            thread_id=f"{logical_session_id}::fence::1",
        )
        saver.bind_checkpoint_contract(
            {"value"},
            binding=binding,
            io_timeout_seconds=_CHECKPOINT_IO_TIMEOUT_SECONDS,
            required_state_keys=_SensitiveState.__annotations__,
        )
        checkpoint = empty_checkpoint()
        checkpoint["channel_values"] = {"value": "secret"}
        checkpoint["channel_versions"] = {"value": "1"}
        saved_config = await saver.aput(
            binding.config,
            checkpoint,
            CheckpointMetadata(),
            {"value": "1"},
        )
        await saver.aput_writes(
            saved_config,
            (("value", "first"), ("value", "second")),
            "manifest-task",
            "pull/manifest-task",
        )
        assert await saver.aget_tuple(saved_config) is not None

        checkpoint_id = saved_config["configurable"]["checkpoint_id"]
        async with pool.connection() as connection:
            if mutation == "delete":
                await connection.execute(
                    """
                    DELETE FROM checkpoint_writes
                    WHERE thread_id = %s AND checkpoint_ns = '' AND checkpoint_id = %s
                      AND task_id = 'manifest-task' AND idx = 0
                    """,
                    (binding.storage_thread_id, checkpoint_id),
                )
            elif mutation == "duplicate":
                await connection.execute(
                    """
                    INSERT INTO checkpoint_writes (
                        thread_id, checkpoint_ns, checkpoint_id, task_id,
                        task_path, idx, channel, type, blob
                    )
                    SELECT thread_id, checkpoint_ns, checkpoint_id, task_id,
                        task_path, idx + 100, channel, type, blob
                    FROM checkpoint_writes
                    WHERE thread_id = %s AND checkpoint_ns = '' AND checkpoint_id = %s
                      AND task_id = 'manifest-task' AND idx = 0
                    """,
                    (binding.storage_thread_id, checkpoint_id),
                )
            elif mutation == "relocate":
                await connection.execute(
                    """
                    UPDATE checkpoint_writes SET task_path = 'pull/foreign-task'
                    WHERE thread_id = %s AND checkpoint_ns = '' AND checkpoint_id = %s
                      AND task_id = 'manifest-task' AND idx = 0
                    """,
                    (binding.storage_thread_id, checkpoint_id),
                )
            else:
                await connection.execute(
                    """
                    UPDATE checkpoint_writes AS target
                    SET type = source.type, blob = source.blob
                    FROM checkpoint_writes AS source
                    WHERE target.thread_id = %s AND target.checkpoint_ns = ''
                      AND target.checkpoint_id = %s AND target.task_id = 'manifest-task'
                      AND target.idx IN (0, 1)
                      AND source.thread_id = target.thread_id
                      AND source.checkpoint_ns = target.checkpoint_ns
                      AND source.checkpoint_id = target.checkpoint_id
                      AND source.task_id = target.task_id
                      AND source.idx = 1 - target.idx
                    """,
                    (binding.storage_thread_id, checkpoint_id),
                )

        with pytest.raises(CheckpointSchemaError, match="pending-write collection"):
            await saver.aget_tuple(saved_config)


@pytest.mark.postgres
@pytest.mark.parametrize("authorized_operation", ["checkpoint", "pending_write"])
async def test_authorized_write_cannot_reseal_a_tampered_pending_write_collection(
    authorized_operation: str,
) -> None:
    logical_session_id = f"manifest-transition-{authorized_operation}-{uuid.uuid4().hex}"
    async with AsyncExitStack() as stack:
        pool, _raw_backend, saver, authority, _registry = await _open_fenced_boundary(
            stack,
            _dsn(),
            logical_session_id=logical_session_id,
        )
        binding = CheckpointBinding(
            tenant_id=authority.tenant_id,
            logical_session_id=logical_session_id,
            deployment_id=authority.deployment_id,
            graph_contract=authority.graph_contract,
            thread_id=f"{logical_session_id}::fence::1",
        )
        saver.bind_checkpoint_contract(
            {"value"},
            binding=binding,
            io_timeout_seconds=_CHECKPOINT_IO_TIMEOUT_SECONDS,
            required_state_keys=_SensitiveState.__annotations__,
        )
        checkpoint = empty_checkpoint()
        checkpoint["channel_values"] = {"value": "secret"}
        checkpoint["channel_versions"] = {"value": "1"}
        saved_config = await saver.aput(
            binding.config,
            checkpoint,
            CheckpointMetadata(),
            {"value": "1"},
        )
        await saver.aput_writes(
            saved_config,
            (("value", "first"),),
            "manifest-task",
            "pull/manifest-task",
        )
        checkpoint_id = saved_config["configurable"]["checkpoint_id"]
        async with pool.connection() as connection:
            await connection.execute(
                """
                INSERT INTO checkpoint_writes (
                    thread_id, checkpoint_ns, checkpoint_id, task_id,
                    task_path, idx, channel, type, blob
                )
                SELECT thread_id, checkpoint_ns, checkpoint_id, task_id,
                    task_path, idx + 100, channel, type, blob
                FROM checkpoint_writes
                WHERE thread_id = %s AND checkpoint_ns = '' AND checkpoint_id = %s
                  AND task_id = 'manifest-task' AND idx = 0
                """,
                (binding.storage_thread_id, checkpoint_id),
            )
            original_manifest = await (
                await connection.execute(
                    """
                    SELECT envelope_format, envelope_key_version, payload_schema_version,
                        envelope_nonce, encrypted_manifest, updated_at
                    FROM platform_checkpoint_write_manifests
                    WHERE tenant_id = %s AND logical_session_id = %s
                      AND checkpoint_generation_namespace = %s
                      AND langgraph_checkpoint_namespace = '' AND checkpoint_id = %s
                    """,
                    (
                        authority.tenant_id,
                        logical_session_id,
                        binding.thread_id,
                        checkpoint_id,
                    ),
                )
            ).fetchone()
        assert original_manifest is not None

        with pytest.raises(CheckpointSchemaError, match="pending-write collection"):
            if authorized_operation == "checkpoint":
                await saver.aput(
                    saved_config,
                    checkpoint,
                    CheckpointMetadata(),
                    {"value": "1"},
                )
            else:
                await saver.aput_writes(
                    saved_config,
                    (("value", "second"),),
                    "authorized-task",
                    "pull/authorized-task",
                )

        async with pool.connection() as connection:
            current_manifest = await (
                await connection.execute(
                    """
                    SELECT envelope_format, envelope_key_version, payload_schema_version,
                        envelope_nonce, encrypted_manifest, updated_at
                    FROM platform_checkpoint_write_manifests
                    WHERE tenant_id = %s AND logical_session_id = %s
                      AND checkpoint_generation_namespace = %s
                      AND langgraph_checkpoint_namespace = '' AND checkpoint_id = %s
                    """,
                    (
                        authority.tenant_id,
                        logical_session_id,
                        binding.thread_id,
                        checkpoint_id,
                    ),
                )
            ).fetchone()
            authorized_rows = await (
                await connection.execute(
                    """
                    SELECT count(*) FROM checkpoint_writes
                    WHERE thread_id = %s AND checkpoint_ns = '' AND checkpoint_id = %s
                      AND task_id = 'authorized-task'
                    """,
                    (binding.storage_thread_id, checkpoint_id),
                )
            ).fetchone()
        assert current_manifest == original_manifest
        assert authorized_rows == (0,)


@pytest.mark.postgres
async def test_application_role_runs_fenced_saver_and_rejects_cross_tenant_storage() -> None:
    dsn = _dsn()
    logical_session_id = f"application-role-{uuid.uuid4().hex}"
    foreign_session_id = f"foreign-role-{uuid.uuid4().hex}"
    role_name = f"phase4c_checkpoint_{uuid.uuid4().hex[:16]}"
    application_pool: AsyncConnectionPool | None = None
    role_created = False
    async with AsyncExitStack() as stack:
        owner_pool, raw_backend, _owner_saver, authority, registry = await _open_fenced_boundary(
            stack,
            dsn,
            logical_session_id=logical_session_id,
        )
        source = (await registry.checkpoint_generations(authority))[0].binding
        foreign_authority = SessionLeaseAuthority(
            tenant_id="checkpoint-data-plane-foreign",
            authority=AdmittedSessionAuthority(
                logical_session_id=foreign_session_id,
                transport=TransportAuthority(
                    provider="livekit",
                    room_id=f"RM_{foreign_session_id}",
                    assignment_id=f"AJ_{foreign_session_id}",
                    worker_id=f"AW_{foreign_session_id}",
                ),
            ),
            deployment_id=authority.deployment_id,
            graph_contract=authority.graph_contract,
            config_version=authority.config_version,
            lease_owner_id=f"owner-{foreign_session_id}",
            fencing_generation=1,
        )
        await registry.register_and_acquire(
            SessionRegistration(
                tenant_id=foreign_authority.tenant_id,
                authority=foreign_authority.authority,
                deployment_id=foreign_authority.deployment_id,
                graph_contract=foreign_authority.graph_contract,
                config_version=foreign_authority.config_version,
                principal_generation=0,
                session_revision=0,
                retention_seconds=3600.0,
            ),
            SessionLeaseRequest(
                lease_owner_id=foreign_authority.lease_owner_id,
                duration_seconds=300.0,
            ),
            payload=DurableSessionPayload(),
        )
        await registry.activate(foreign_authority)
        foreign = (await registry.checkpoint_generations(foreign_authority))[0].binding
        foreign_backend = FencedPostgresCheckpointSaver(
            owner_pool,
            authority=foreign_authority,
            cipher=_checkpoint_cipher(),
            serde=build_checkpoint_serializer(),
        )
        foreign_saver = build_checkpointer(
            foreign_backend,
            synchronous_operations=False,
            cipher=_checkpoint_cipher(),
        )
        foreign_saver.bind_checkpoint_contract(
            {"value"},
            binding=foreign,
            io_timeout_seconds=_CHECKPOINT_IO_TIMEOUT_SECONDS,
            required_state_keys=_SensitiveState.__annotations__,
        )
        foreign_checkpoint = empty_checkpoint()
        foreign_checkpoint["channel_values"] = {"value": "foreign"}
        foreign_checkpoint["channel_versions"] = {"value": "1"}
        foreign_config = await foreign_saver.aput(
            foreign.config,
            foreign_checkpoint,
            CheckpointMetadata(),
            {"value": "1"},
        )
        await foreign_saver.aput_writes(
            foreign_config,
            (("value", "foreign-pending"),),
            "foreign-application-role-task",
        )

        async with owner_pool.connection() as connection:
            schema_cursor = await connection.execute("SELECT current_schema()")
            schema_row = await schema_cursor.fetchone()
            assert schema_row is not None
            schema_name = str(schema_row[0])
            await connection.execute(
                sql.SQL("CREATE ROLE {} NOLOGIN").format(sql.Identifier(role_name))
            )
            role_created = True
            await connection.execute(
                sql.SQL(
                    "GRANT ALL PRIVILEGES ON TABLE checkpoints, checkpoint_blobs, "
                    "checkpoint_writes TO {}"
                ).format(sql.Identifier(role_name))
            )
            await connection.execute(
                sql.SQL("GRANT CREATE ON SCHEMA {} TO {}").format(
                    sql.Identifier(schema_name),
                    sql.Identifier(role_name),
                )
            )
            await grant_platform_application_role(
                connection,
                schema_name=schema_name,
                role_name=role_name,
            )

        async def configure_application_role(connection: AsyncConnection) -> None:
            await connection.execute(sql.SQL("SET ROLE {}").format(sql.Identifier(role_name)))
            await connection.execute(
                sql.SQL("SET search_path TO {}, pg_catalog").format(sql.Identifier(schema_name))
            )

        try:
            application_pool = AsyncConnectionPool(
                dsn,
                min_size=1,
                max_size=2,
                timeout=_CHECKPOINT_IO_TIMEOUT_SECONDS,
                kwargs={"autocommit": True, "prepare_threshold": 0},
                configure=configure_application_role,
                open=False,
            )
            await application_pool.open(wait=True, timeout=2.0)
            application_backend = FencedPostgresCheckpointSaver(
                application_pool,
                authority=authority,
                cipher=_checkpoint_cipher(),
                serde=build_checkpoint_serializer(),
            )
            application_saver = build_checkpointer(
                application_backend,
                synchronous_operations=False,
                cipher=_checkpoint_cipher(),
            )
            application_saver.bind_checkpoint_contract(
                {"value"},
                binding=source,
                io_timeout_seconds=_CHECKPOINT_IO_TIMEOUT_SECONDS,
                required_state_keys=_SensitiveState.__annotations__,
            )
            checkpoint = empty_checkpoint()
            checkpoint["channel_values"] = {"value": "same-tenant"}
            checkpoint["channel_versions"] = {"value": "1"}
            saved_config = await application_saver.aput(
                source.config,
                checkpoint,
                CheckpointMetadata(),
                {"value": "1"},
            )
            await application_saver.aput_writes(
                saved_config,
                (("value", "pending"),),
                "application-role-task",
            )
            restored = await application_saver.aget_tuple(saved_config)
            assert restored is not None
            assert any(write[1:] == ("value", "pending") for write in restored.pending_writes)
            assert [item async for item in application_saver.alist(source.config)]

            async with application_pool.connection() as application:
                await application.execute(
                    "SELECT set_config('agnostic_market.tenant_id', %s, false)",
                    (authority.tenant_id,),
                )
                role_cursor = await application.execute(
                    "SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname = current_user"
                )
                assert await role_cursor.fetchone() == (False, False)
                schema_cursor = await application.execute(
                    "SELECT has_schema_privilege(current_user, %s, 'CREATE')",
                    (schema_name,),
                )
                assert await schema_cursor.fetchone() == (False,)
                rls_cursor = await application.execute(
                    """
                    SELECT relname, relrowsecurity, relforcerowsecurity
                    FROM pg_class AS relation
                    JOIN pg_namespace AS namespace ON namespace.oid = relation.relnamespace
                    WHERE namespace.nspname = %s
                      AND relname IN ('checkpoints', 'checkpoint_blobs', 'checkpoint_writes')
                    ORDER BY relname
                    """,
                    (schema_name,),
                )
                assert await rls_cursor.fetchall() == [
                    ("checkpoint_blobs", True, True),
                    ("checkpoint_writes", True, True),
                    ("checkpoints", True, True),
                ]
                expected_privileges = {
                    "checkpoints": (True, True, True, True),
                    "checkpoint_blobs": (True, True, False, True),
                    "checkpoint_writes": (True, True, True, True),
                }
                for table_name, expected in expected_privileges.items():
                    privilege_cursor = await application.execute(
                        """
                        SELECT
                            has_table_privilege(current_user, %s, 'SELECT'),
                            has_table_privilege(current_user, %s, 'INSERT'),
                            has_table_privilege(current_user, %s, 'UPDATE'),
                            has_table_privilege(current_user, %s, 'DELETE'),
                            has_table_privilege(current_user, %s, 'TRUNCATE'),
                            has_table_privilege(current_user, %s, 'REFERENCES'),
                            has_table_privilege(current_user, %s, 'TRIGGER'),
                            has_table_privilege(current_user, %s, 'MAINTAIN')
                        """,
                        (f"{schema_name}.{table_name}",) * 8,
                    )
                    assert await privilege_cursor.fetchone() == (
                        *expected,
                        False,
                        False,
                        False,
                        False,
                    )
                    read_cursor = await application.execute(
                        sql.SQL("SELECT 1 FROM {} WHERE thread_id = %s").format(
                            sql.Identifier(table_name)
                        ),
                        (foreign.storage_thread_id,),
                    )
                    assert await read_cursor.fetchall() == []

                cross_tenant_inserts = (
                    (
                        """
                        INSERT INTO checkpoints (
                            thread_id, checkpoint_ns, checkpoint_id,
                            parent_checkpoint_id, type, checkpoint, metadata
                        )
                        SELECT %s, checkpoint_ns, %s,
                            parent_checkpoint_id, type, checkpoint, metadata
                        FROM checkpoints WHERE thread_id = %s
                        LIMIT 1
                        """,
                        (
                            foreign.storage_thread_id,
                            str(uuid.uuid4()),
                            source.storage_thread_id,
                        ),
                    ),
                    (
                        """
                        INSERT INTO checkpoint_blobs (
                            thread_id, checkpoint_ns, channel, version, type, blob
                        )
                        SELECT %s, checkpoint_ns, channel || '-cross-insert',
                            version, type, blob
                        FROM checkpoint_blobs WHERE thread_id = %s
                        LIMIT 1
                        """,
                        (foreign.storage_thread_id, source.storage_thread_id),
                    ),
                    (
                        """
                        INSERT INTO checkpoint_writes (
                            thread_id, checkpoint_ns, checkpoint_id, task_id,
                            task_path, idx, channel, type, blob
                        )
                        SELECT %s, checkpoint_ns, %s, 'cross-tenant-insert',
                            task_path, idx, channel, type, blob
                        FROM checkpoint_writes WHERE thread_id = %s
                        LIMIT 1
                        """,
                        (
                            foreign.storage_thread_id,
                            str(uuid.uuid4()),
                            source.storage_thread_id,
                        ),
                    ),
                )
                for statement, parameters in cross_tenant_inserts:
                    with pytest.raises(InsufficientPrivilege):
                        await application.execute(statement, parameters)
                for table_name in expected_privileges:
                    with pytest.raises(InsufficientPrivilege):
                        await application.execute(
                            sql.SQL("UPDATE {} SET thread_id = %s WHERE thread_id = %s").format(
                                sql.Identifier(table_name)
                            ),
                            (foreign.storage_thread_id, source.storage_thread_id),
                        )
                    delete_cursor = await application.execute(
                        sql.SQL("DELETE FROM {} WHERE thread_id = %s").format(
                            sql.Identifier(table_name)
                        ),
                        (foreign.storage_thread_id,),
                    )
                    assert delete_cursor.rowcount == 0
            foreign_restored = await foreign_saver.aget_tuple(foreign_config)
            assert foreign_restored is not None
            assert any(
                write[1:] == ("value", "foreign-pending")
                for write in foreign_restored.pending_writes
            )

            await _publish_retirement_marker(registry, authority, "application-role-deletion")
            destination_generation = await registry.begin_checkpoint_rotation(
                authority,
                expected_revision=1,
                expected_principal_generation=0,
                transition_id="application-role-deletion",
            )
            destination = destination_generation.binding
            application_saver.bind_checkpoint_contract(
                {"value"},
                binding=destination,
                io_timeout_seconds=_CHECKPOINT_IO_TIMEOUT_SECONDS,
                required_state_keys=_SensitiveState.__annotations__,
            )
            destination_checkpoint = empty_checkpoint()
            destination_checkpoint["channel_values"] = {"value": "destination"}
            destination_checkpoint["channel_versions"] = {"value": "1"}
            await application_saver.aput(
                destination.config,
                destination_checkpoint,
                CheckpointMetadata(),
                {"value": "1"},
            )
            await registry.switch_checkpoint_generation(
                authority,
                "application-role-deletion",
            )
            await application_saver.adelete_thread(source.storage_thread_id)
            assert await raw_backend.aget_tuple(source.config) is None
            async with owner_pool.connection() as connection:
                deletion_cursor = await connection.execute(
                    """
                    SELECT
                        (SELECT count(*) FROM checkpoints WHERE thread_id = %s),
                        (SELECT count(*) FROM checkpoint_blobs WHERE thread_id = %s),
                        (SELECT count(*) FROM checkpoint_writes WHERE thread_id = %s),
                        (
                            SELECT count(*) FROM platform_checkpoint_write_manifests
                            WHERE tenant_id = %s AND logical_session_id = %s
                              AND checkpoint_generation_namespace = %s
                        )
                    """,
                    (
                        source.storage_thread_id,
                        source.storage_thread_id,
                        source.storage_thread_id,
                        authority.tenant_id,
                        logical_session_id,
                        source.thread_id,
                    ),
                )
                assert await deletion_cursor.fetchone() == (0, 0, 0, 0)
        finally:
            if application_pool is not None:
                await application_pool.close()
            if role_created:
                async with owner_pool.connection() as connection:
                    role = sql.Identifier(role_name)
                    await connection.execute(sql.SQL("DROP OWNED BY {}").format(role))
                    await connection.execute(sql.SQL("DROP ROLE {}").format(role))


@pytest.mark.postgres
async def test_closing_fence_rejects_a_writer_already_waiting_on_registry_authority() -> None:
    logical_session_id = f"closing-writer-{uuid.uuid4().hex}"
    async with AsyncExitStack() as stack:
        pool, raw_backend, saver, authority, _registry = await _open_fenced_boundary(
            stack,
            _dsn(),
            logical_session_id=logical_session_id,
        )
        binding = CheckpointBinding(
            tenant_id=authority.tenant_id,
            logical_session_id=logical_session_id,
            deployment_id=authority.deployment_id,
            graph_contract=authority.graph_contract,
            thread_id=f"{logical_session_id}::fence::1",
        )
        saver.bind_checkpoint_contract(
            {"value"},
            binding=binding,
            io_timeout_seconds=_CHECKPOINT_IO_TIMEOUT_SECONDS,
            required_state_keys=_SensitiveState.__annotations__,
        )
        checkpoint = empty_checkpoint()
        checkpoint["channel_values"] = {"value": "late"}
        checkpoint["channel_versions"] = {"value": "1"}

        async with pool.connection() as connection, connection.transaction():
            await connection.execute(
                "SELECT set_config('agnostic_market.tenant_id', %s, true)",
                (authority.tenant_id,),
            )
            await connection.execute(
                """
                SELECT 1 FROM platform_sessions
                WHERE tenant_id = %s AND logical_session_id = %s
                FOR UPDATE
                """,
                (authority.tenant_id, logical_session_id),
            )
            delayed_write = asyncio.create_task(
                saver.aput(
                    binding.config,
                    checkpoint,
                    CheckpointMetadata(),
                    {"value": "1"},
                )
            )
            await asyncio.sleep(0.05)
            assert not delayed_write.done()
            await connection.execute(
                """
                UPDATE platform_sessions SET lifecycle = 'closing'
                WHERE tenant_id = %s AND logical_session_id = %s
                """,
                (authority.tenant_id, logical_session_id),
            )

        with pytest.raises(CheckpointDataPlaneError) as rejected:
            await delayed_write
        assert rejected.value.reason is CheckpointDataPlaneReason.LIFECYCLE_REJECTED
        assert await raw_backend.aget_tuple(binding.config) is None


@pytest.mark.postgres
async def test_fenced_checkpoint_requires_an_encrypted_pending_write_manifest() -> None:
    sentinel = "pending-write-manifest-plaintext-sentinel"
    logical_session_id = f"manifest-required-{uuid.uuid4().hex}"
    async with AsyncExitStack() as stack:
        pool, _raw_backend, saver, authority, _registry = await _open_fenced_boundary(
            stack,
            _dsn(),
            logical_session_id=logical_session_id,
        )
        binding = CheckpointBinding(
            tenant_id=authority.tenant_id,
            logical_session_id=logical_session_id,
            deployment_id=authority.deployment_id,
            graph_contract=authority.graph_contract,
            thread_id=f"{logical_session_id}::fence::1",
        )
        saver.bind_checkpoint_contract(
            {"value"},
            binding=binding,
            io_timeout_seconds=_CHECKPOINT_IO_TIMEOUT_SECONDS,
            required_state_keys=_SensitiveState.__annotations__,
        )
        checkpoint = empty_checkpoint()
        checkpoint["channel_values"] = {"value": sentinel}
        checkpoint["channel_versions"] = {"value": "1"}
        saved_config = await saver.aput(
            binding.config,
            checkpoint,
            CheckpointMetadata(),
            {"value": "1"},
        )
        await saver.aput_writes(
            saved_config,
            (("value", sentinel),),
            "manifest-sentinel-task",
            "pull/manifest-sentinel-task",
        )
        checkpoint_id = saved_config["configurable"]["checkpoint_id"]
        async with pool.connection() as connection:
            manifest = await (
                await connection.execute(
                    """
                    SELECT encrypted_manifest
                    FROM platform_checkpoint_write_manifests
                    WHERE tenant_id = %s AND logical_session_id = %s
                      AND checkpoint_id = %s
                    """,
                    (authority.tenant_id, logical_session_id, checkpoint_id),
                )
            ).fetchone()
            pending_write = await (
                await connection.execute(
                    """
                    SELECT blob FROM checkpoint_writes
                    WHERE thread_id = %s AND checkpoint_id = %s
                    """,
                    (binding.storage_thread_id, checkpoint_id),
                )
            ).fetchone()
            assert manifest is not None
            assert pending_write is not None
            assert sentinel.encode() not in bytes(manifest[0])
            assert sentinel.encode() not in bytes(pending_write[0])
            await connection.execute(
                """
                DELETE FROM platform_checkpoint_write_manifests
                WHERE tenant_id = %s AND logical_session_id = %s
                  AND checkpoint_id = %s
                """,
                (authority.tenant_id, logical_session_id, checkpoint_id),
            )

        with pytest.raises(CheckpointDataPlaneError) as rejected:
            await saver.aget_tuple(saved_config)
        assert rejected.value.reason is CheckpointDataPlaneReason.MANIFEST_MISSING

        with pytest.raises(CheckpointDataPlaneError) as write_rejected:
            await saver.aput_writes(
                saved_config,
                (("value", "unsealed-write"),),
                "unsealed-write-task",
                "pull/unsealed-write-task",
            )
        assert write_rejected.value.reason is CheckpointDataPlaneReason.MANIFEST_MISSING


@pytest.mark.postgres
async def test_fenced_checkpoint_accepts_langgraph_write_before_checkpoint_ordering() -> None:
    logical_session_id = f"write-before-checkpoint-{uuid.uuid4().hex}"
    async with AsyncExitStack() as stack:
        _pool, _raw_backend, saver, authority, _registry = await _open_fenced_boundary(
            stack,
            _dsn(),
            logical_session_id=logical_session_id,
        )
        binding = CheckpointBinding(
            tenant_id=authority.tenant_id,
            logical_session_id=logical_session_id,
            deployment_id=authority.deployment_id,
            graph_contract=authority.graph_contract,
            thread_id=f"{logical_session_id}::fence::1",
        )
        saver.bind_checkpoint_contract(
            {"value"},
            binding=binding,
            io_timeout_seconds=_CHECKPOINT_IO_TIMEOUT_SECONDS,
            required_state_keys=_SensitiveState.__annotations__,
        )
        checkpoint = empty_checkpoint()
        checkpoint["channel_values"] = {"value": "checkpoint-value"}
        checkpoint["channel_versions"] = {"value": "1"}
        write_config = {
            "configurable": {
                **binding.config["configurable"],
                "checkpoint_id": checkpoint["id"],
            }
        }

        await saver.aput_writes(
            write_config,
            (("value", "pending-value"),),
            "write-before-checkpoint-task",
            "pull/write-before-checkpoint-task",
        )
        saved_config = await saver.aput(
            binding.config,
            checkpoint,
            CheckpointMetadata(),
            {"value": "1"},
        )

        restored = await saver.aget_tuple(saved_config)
        assert restored is not None
        assert any(write[1:] == ("value", "pending-value") for write in restored.pending_writes)


@pytest.mark.postgres
async def test_registry_generation_switch_retires_source_checkpoint_write_authority() -> None:
    logical_session_id = f"generation-switch-{uuid.uuid4().hex}"
    async with AsyncExitStack() as stack:
        _pool, _raw_backend, saver, authority, registry = await _open_fenced_boundary(
            stack,
            _dsn(),
            logical_session_id=logical_session_id,
        )
        source = CheckpointBinding(
            tenant_id=authority.tenant_id,
            logical_session_id=logical_session_id,
            deployment_id=authority.deployment_id,
            graph_contract=authority.graph_contract,
            thread_id=f"{logical_session_id}::fence::1",
        )
        saver.bind_checkpoint_contract(
            {"value"},
            binding=source,
            io_timeout_seconds=_CHECKPOINT_IO_TIMEOUT_SECONDS,
            required_state_keys=_SensitiveState.__annotations__,
        )
        await _publish_retirement_marker(registry, authority, "checkpoint-data-plane-rotation")
        generations = await registry.checkpoint_generations(authority)
        generation_authority = BoundCheckpointGenerationAuthority(
            registry,
            authority,
            generations[0],
        )
        destination_generation = await generation_authority.begin_checkpoint_rotation(
            expected_revision=1,
            transition_id="checkpoint-data-plane-rotation",
        )
        destination = destination_generation.binding
        saver.bind_checkpoint_contract(
            {"value"},
            binding=destination,
            io_timeout_seconds=_CHECKPOINT_IO_TIMEOUT_SECONDS,
            required_state_keys=_SensitiveState.__annotations__,
        )
        checkpoint = empty_checkpoint()
        checkpoint["channel_values"] = {"value": "destination"}
        checkpoint["channel_versions"] = {"value": "1"}
        destination_config = await saver.aput(
            destination.config,
            checkpoint,
            CheckpointMetadata(),
            {"value": "1"},
        )
        await generation_authority.switch_checkpoint_generation(
            "checkpoint-data-plane-rotation",
        )

        source_checkpoint = empty_checkpoint()
        source_checkpoint["channel_values"] = {"value": "stale"}
        source_checkpoint["channel_versions"] = {"value": "1"}
        with pytest.raises(CheckpointDataPlaneError) as rejected:
            await saver.aput(
                source.config,
                source_checkpoint,
                CheckpointMetadata(),
                {"value": "1"},
            )
        assert rejected.value.reason is CheckpointDataPlaneReason.GENERATION_NOT_WRITABLE
        assert await saver.aget_tuple(destination_config) is not None

        await saver.adelete_thread(source.storage_thread_id)
        deleted = await generation_authority.record_checkpoint_deletion(
            "checkpoint-data-plane-rotation",
        )
        assert deleted.state == "deleted"
        assert generation_authority.current_generation == destination_generation.model_copy(
            update={"state": "current"}
        )


@pytest.mark.postgres
async def test_checkpoint_boundary_translates_a_malformed_generation_row() -> None:
    logical_session_id = f"malformed-generation-{uuid.uuid4().hex}"
    async with AsyncExitStack() as stack:
        pool, _raw_backend, saver, authority, _registry = await _open_fenced_boundary(
            stack,
            _dsn(),
            logical_session_id=logical_session_id,
        )
        binding = CheckpointBinding(
            tenant_id=authority.tenant_id,
            logical_session_id=logical_session_id,
            deployment_id=authority.deployment_id,
            graph_contract=authority.graph_contract,
            thread_id=f"{logical_session_id}::fence::1",
        )
        saver.bind_checkpoint_contract(
            {"value"},
            binding=binding,
            io_timeout_seconds=_CHECKPOINT_IO_TIMEOUT_SECONDS,
            required_state_keys=_SensitiveState.__annotations__,
        )
        async with pool.connection() as connection, connection.transaction():
            await connection.execute(
                "SELECT set_config('agnostic_market.tenant_id', %s, true)",
                (authority.tenant_id,),
            )
            await connection.execute(
                """
                UPDATE platform_checkpoint_generations
                SET storage_thread_id = %s
                WHERE tenant_id = %s AND logical_session_id = %s
                """,
                ("cp_" + "0" * 64, authority.tenant_id, logical_session_id),
            )

        with pytest.raises(CheckpointSchemaError, match="generation row is malformed"):
            await saver.aget_tuple(binding.config)


@pytest.mark.postgres
async def test_encrypted_checkpoint_rejects_cross_namespace_relocation_in_postgres() -> None:
    async with AsyncExitStack() as stack:
        _, backend, saver = await _open_boundary(
            stack,
            _dsn(),
            cipher=_checkpoint_cipher(),
        )
        await backend.setup()
        binding = CheckpointBinding(
            tenant_id="synthetic-tenant",
            logical_session_id="cross-namespace-session",
            deployment_id="phase4c-harness",
            graph_contract="cross-namespace-graph-contract",
            thread_id="cross-namespace-session::fence::1",
        )
        saver.bind_checkpoint_contract(
            {"value"},
            binding=binding,
            io_timeout_seconds=_CHECKPOINT_IO_TIMEOUT_SECONDS,
            required_state_keys=_SensitiveState.__annotations__,
        )
        source: RunnableConfig = {
            "configurable": {
                "thread_id": binding.storage_thread_id,
                "checkpoint_ns": "subgraph-a",
            }
        }
        checkpoint = empty_checkpoint()
        checkpoint["channel_values"] = {"value": "secret"}
        checkpoint["channel_versions"] = {"value": "1"}
        source_config = await saver.aput(
            source,
            checkpoint,
            CheckpointMetadata(),
            {"value": "1"},
        )
        raw = await backend.aget_tuple(source_config)
        assert raw is not None
        target: RunnableConfig = {
            "configurable": {
                "thread_id": binding.storage_thread_id,
                "checkpoint_ns": "subgraph-b",
            }
        }
        await backend.aput(
            target,
            raw.checkpoint,
            raw.metadata,
            raw.checkpoint["channel_versions"],
        )

        with pytest.raises(CheckpointSchemaError, match="authenticated"):
            await saver.aget_tuple(target)

        await saver.adelete_thread(binding.storage_thread_id)


@pytest.mark.postgres
async def test_encrypted_checkpoint_detects_a_missing_postgres_channel_blob() -> None:
    async with AsyncExitStack() as stack:
        pool, backend, saver = await _open_boundary(
            stack,
            _dsn(),
            cipher=_checkpoint_cipher(),
        )
        await backend.setup()
        graph = _sensitive_compiled(saver)
        binding = CheckpointBinding(
            tenant_id="synthetic-tenant",
            logical_session_id="missing-channel-session",
            deployment_id="phase4c-harness",
            graph_contract=graph_contract_fingerprint(graph),
            thread_id="missing-channel-session::fence::1",
        )
        saver.bind_checkpoint_contract(
            graph.channels,
            binding=binding,
            io_timeout_seconds=_CHECKPOINT_IO_TIMEOUT_SECONDS,
            required_state_keys=_SensitiveState.__annotations__,
        )

        await graph.ainvoke({"value": "secret"}, binding.config)
        saved = await saver.aget_tuple(binding.config)
        assert saved is not None
        version = saved.checkpoint["channel_versions"]["value"]
        async with pool.connection() as connection:
            await connection.execute(
                """
                DELETE FROM checkpoint_blobs
                WHERE thread_id = %s AND checkpoint_ns = %s
                  AND channel = %s AND version = %s
                """,
                (binding.storage_thread_id, "", "value", str(version)),
            )

        with pytest.raises(CheckpointSchemaError, match="channel manifest"):
            await saver.aget_tuple(binding.config)

        await saver.adelete_thread(binding.storage_thread_id)


@pytest.mark.postgres
async def test_checkpoint_deadline_and_cancellation_interrupt_a_blocked_database_read() -> None:
    dsn = _dsn()
    async with AsyncExitStack() as stack:
        _pool, backend, saver = await _open_boundary(
            stack,
            dsn,
            backend_type=_StalledReadPostgresSaver,
        )
        assert isinstance(backend, _StalledReadPostgresSaver)
        await backend.setup()
        graph = _compiled(saver)
        binding = CheckpointBinding(
            tenant_id="synthetic-tenant",
            deployment_id="phase4c-harness",
            graph_contract=graph_contract_fingerprint(graph),
            thread_id="blocked-read",
        )
        saver.bind_checkpoint_contract(
            graph.channels,
            binding=binding,
            io_timeout_seconds=_CHECKPOINT_IO_TIMEOUT_SECONDS,
            required_state_keys=_State.__annotations__,
        )
        await graph.ainvoke({"value": 1}, binding.config)

        backend.stall_reads = True
        with pytest.raises(TimeoutError):
            await saver.aget_tuple(binding.config)

        backend.stalled_read_started.clear()
        blocked_read = asyncio.create_task(saver.aget_tuple(binding.config))
        await asyncio.wait_for(
            backend.stalled_read_started.wait(),
            timeout=_CHECKPOINT_IO_TIMEOUT_SECONDS,
        )
        assert not blocked_read.done()
        blocked_read.cancel()
        with pytest.raises(asyncio.CancelledError):
            await blocked_read

        backend.stall_reads = False
        assert await saver.aget_tuple(binding.config) is not None


@pytest.mark.postgres
async def test_checkpoint_pool_exhaustion_fails_within_the_configured_acquisition_budget() -> None:
    dsn = _dsn()
    async with AsyncExitStack() as stack:
        pool, backend, saver = await _open_boundary(
            stack,
            dsn,
            max_pool_size=1,
            pool_timeout_seconds=0.05,
        )
        await backend.setup()
        graph = _compiled(saver)
        binding = CheckpointBinding(
            tenant_id="synthetic-tenant",
            deployment_id="phase4c-harness",
            graph_contract=graph_contract_fingerprint(graph),
            thread_id="pool-exhaustion",
        )
        saver.bind_checkpoint_contract(
            graph.channels,
            binding=binding,
            io_timeout_seconds=0.5,
            required_state_keys=_State.__annotations__,
        )

        async with pool.connection():
            with pytest.raises(PoolTimeout):
                await saver.aget_tuple(binding.config)


@pytest.mark.postgres
async def test_harness_runs_the_pinned_postgresql_release() -> None:
    async with await AsyncConnection.connect(_dsn(), autocommit=True) as connection:
        cursor = await connection.execute("SHOW server_version")
        row = await cursor.fetchone()
    assert row is not None
    assert str(row[0]).startswith("18.6")
