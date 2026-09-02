"""Conformance for the pinned asynchronous PostgreSQL checkpoint backend."""

from __future__ import annotations

import asyncio
import os
from contextlib import AsyncExitStack
from typing import TypedDict

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import CheckpointTuple
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.base import SerializerProtocol
from langgraph.graph import START, StateGraph
from psycopg import AsyncConnection
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

_POSTGRES_DSN_ENV = "PHASE4C_POSTGRES_DSN"
_CHECKPOINT_IO_TIMEOUT_SECONDS = 2.0
_STALLED_READ_SECONDS = 10


class _State(TypedDict, total=False):
    value: int


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
    return pool, backend, build_checkpointer(backend, synchronous_operations=False)


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
