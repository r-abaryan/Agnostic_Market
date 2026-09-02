"""Checkpoint persistence boundary contracts."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any, TypedDict

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.types import SCHEDULED
from langgraph.graph import START, StateGraph

from agnostic_market.checkpoints import (
    CheckpointBinding,
    CheckpointDeletionError,
    CheckpointScopeError,
    SchemaValidatedCheckpointSaver,
    SynchronousCheckpointOperationError,
    graph_contract_fingerprint,
)
from agnostic_market.dtos.state import CheckpointSchemaError


class _State(TypedDict, total=False):
    value: int


class _AsyncOnlySaver(InMemorySaver):
    """Remote-shaped test double whose synchronous surface must never be called."""

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        raise AssertionError("synchronous checkpoint read used")

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        raise AssertionError("synchronous checkpoint write used")

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        raise AssertionError("synchronous pending-write path used")

    def delete_thread(self, thread_id: str) -> None:
        raise AssertionError("synchronous checkpoint deletion used")

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        await asyncio.sleep(0.01)
        return InMemorySaver.get_tuple(self, config)

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        await asyncio.sleep(0.01)
        return InMemorySaver.put(self, config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        await asyncio.sleep(0.01)
        InMemorySaver.put_writes(self, config, writes, task_id, task_path)

    async def adelete_thread(self, thread_id: str) -> None:
        await asyncio.sleep(0.01)
        InMemorySaver.delete_thread(self, thread_id)


class _StalledReadSaver(_AsyncOnlySaver):
    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        await asyncio.sleep(60)
        return None


class _FailDeleteOnceSaver(_AsyncOnlySaver):
    def __init__(self) -> None:
        super().__init__()
        self.delete_attempts = 0

    async def adelete_thread(self, thread_id: str) -> None:
        self.delete_attempts += 1
        if self.delete_attempts == 1:
            raise TimeoutError("injected deletion failure")
        await super().adelete_thread(thread_id)


class _NoOpDeleteSaver(_AsyncOnlySaver):
    async def adelete_thread(self, thread_id: str) -> None:
        return None


def _compiled(saver: SchemaValidatedCheckpointSaver):
    graph = StateGraph(_State)
    graph.add_node("increment", lambda state: {"value": state.get("value", 0) + 1})
    graph.add_edge(START, "increment")
    return graph.compile(checkpointer=saver)


@pytest.mark.asyncio
async def test_async_graph_checkpointing_never_uses_sync_backend_methods() -> None:
    saver = SchemaValidatedCheckpointSaver(_AsyncOnlySaver())
    graph = _compiled(saver)
    contract = graph_contract_fingerprint(graph)
    binding = CheckpointBinding(
        tenant_id="tenant-a",
        deployment_id="deployment-a",
        graph_contract=contract,
        thread_id="thread-a",
    )
    saver.bind_checkpoint_contract(
        graph.channels,
        binding=binding,
        io_timeout_seconds=0.5,
        required_state_keys=_State.__annotations__,
    )

    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        for _ in range(10):
            await asyncio.sleep(0.002)
            ticks += 1

    _, _ = await asyncio.gather(
        graph.ainvoke({"value": 1}, binding.config),
        ticker(),
    )
    assert ticks == 10
    assert (await graph.aget_state(binding.config)).values["value"] == 2
    await saver.adelete_thread(binding.storage_thread_id)
    with pytest.raises(CheckpointScopeError, match="namespace"):
        await graph.aget_state(binding.config)


@pytest.mark.asyncio
async def test_async_only_boundary_rejects_every_synchronous_storage_operation() -> None:
    saver = SchemaValidatedCheckpointSaver(
        _AsyncOnlySaver(),
        synchronous_operations=False,
    )
    graph = _compiled(saver)
    binding = CheckpointBinding(
        tenant_id="tenant-a",
        deployment_id="deployment-a",
        graph_contract=graph_contract_fingerprint(graph),
        thread_id="thread-a",
    )
    saver.bind_checkpoint_contract(
        graph.channels,
        binding=binding,
        io_timeout_seconds=0.5,
        required_state_keys=_State.__annotations__,
    )
    await graph.ainvoke({"value": 1}, binding.config)
    saved = await saver.aget_tuple(binding.config)
    assert saved is not None

    with pytest.raises(SynchronousCheckpointOperationError):
        saver.get_tuple(binding.config)
    with pytest.raises(SynchronousCheckpointOperationError):
        list(saver.list(binding.config))
    with pytest.raises(SynchronousCheckpointOperationError):
        saver.put(saved.config, saved.checkpoint, saved.metadata, {})
    with pytest.raises(SynchronousCheckpointOperationError):
        saver.put_writes(saved.config, (("value", 2),), "task-a")
    with pytest.raises(SynchronousCheckpointOperationError):
        saver.delete_thread(binding.storage_thread_id)


@pytest.mark.asyncio
async def test_checkpoint_namespace_rejects_another_tenant_or_deployment() -> None:
    saver = SchemaValidatedCheckpointSaver(InMemorySaver())
    graph = _compiled(saver)
    contract = graph_contract_fingerprint(graph)
    expected = CheckpointBinding(
        tenant_id="tenant-a",
        deployment_id="deployment-a",
        graph_contract=contract,
        thread_id="shared-thread",
    )
    saver.bind_checkpoint_contract(
        graph.channels,
        binding=expected,
        io_timeout_seconds=0.5,
        required_state_keys=_State.__annotations__,
    )
    await graph.ainvoke({"value": 1}, expected.config)

    wrong_tenant = CheckpointBinding(
        tenant_id="tenant-b",
        deployment_id="deployment-a",
        graph_contract=contract,
        thread_id="shared-thread",
    )
    wrong_deployment = CheckpointBinding(
        tenant_id="tenant-a",
        deployment_id="deployment-b",
        graph_contract=contract,
        thread_id="shared-thread",
    )

    with pytest.raises(CheckpointScopeError, match="namespace"):
        await graph.aget_state(wrong_tenant.config)
    with pytest.raises(CheckpointScopeError, match="namespace"):
        await graph.aget_state(wrong_deployment.config)


@pytest.mark.asyncio
async def test_shared_backend_isolates_identical_logical_threads_across_tenants() -> None:
    saver = SchemaValidatedCheckpointSaver(InMemorySaver())
    graph = _compiled(saver)
    contract = graph_contract_fingerprint(graph)
    first = CheckpointBinding(
        tenant_id="tenant-a",
        deployment_id="deployment-a",
        graph_contract=contract,
        thread_id="shared-thread",
    )
    second = CheckpointBinding(
        tenant_id="tenant-b",
        deployment_id="deployment-a",
        graph_contract=contract,
        thread_id="shared-thread",
    )

    await asyncio.gather(
        asyncio.to_thread(
            saver.bind_checkpoint_contract,
            graph.channels,
            binding=first,
            io_timeout_seconds=0.5,
            required_state_keys=_State.__annotations__,
        ),
        asyncio.to_thread(
            saver.bind_checkpoint_contract,
            graph.channels,
            binding=second,
            io_timeout_seconds=0.5,
            required_state_keys=_State.__annotations__,
        ),
    )
    await asyncio.gather(
        graph.ainvoke({"value": 1}, first.config),
        graph.ainvoke({"value": 40}, second.config),
    )

    assert first.storage_thread_id != second.storage_thread_id
    assert (await graph.aget_state(first.config)).values["value"] == 2
    assert (await graph.aget_state(second.config)).values["value"] == 41


def test_graph_contract_changes_when_topology_changes() -> None:
    first_saver = SchemaValidatedCheckpointSaver(InMemorySaver())
    first = _compiled(first_saver)

    second_saver = SchemaValidatedCheckpointSaver(InMemorySaver())
    second_builder = StateGraph(_State)
    second_builder.add_node("increment", lambda state: {"value": state.get("value", 0) + 1})
    second_builder.add_node("finish", lambda _state: {})
    second_builder.add_edge(START, "increment")
    second_builder.add_edge("increment", "finish")
    second = second_builder.compile(checkpointer=second_saver)

    assert graph_contract_fingerprint(first) != graph_contract_fingerprint(second)


@pytest.mark.asyncio
async def test_checkpoint_backend_stall_is_bounded_by_the_bound_contract() -> None:
    saver = SchemaValidatedCheckpointSaver(_StalledReadSaver())
    graph = _compiled(saver)
    contract = graph_contract_fingerprint(graph)
    binding = CheckpointBinding(
        tenant_id="tenant-a",
        deployment_id="deployment-a",
        graph_contract=contract,
        thread_id="thread-a",
    )
    saver.bind_checkpoint_contract(
        graph.channels,
        binding=binding,
        io_timeout_seconds=0.01,
        required_state_keys=_State.__annotations__,
    )

    with pytest.raises(TimeoutError):
        await graph.aget_state(binding.config)


@pytest.mark.asyncio
async def test_async_listing_timeout_never_cancels_consumer_work() -> None:
    saver = SchemaValidatedCheckpointSaver(InMemorySaver())
    graph = _compiled(saver)
    binding = CheckpointBinding(
        tenant_id="tenant-a",
        deployment_id="deployment-a",
        graph_contract=graph_contract_fingerprint(graph),
        thread_id="thread-a",
    )
    saver.bind_checkpoint_contract(
        graph.channels,
        binding=binding,
        io_timeout_seconds=0.05,
        required_state_keys=_State.__annotations__,
    )
    await graph.ainvoke({"value": 1}, binding.config)

    checkpoints = saver.alist(binding.config, limit=1)
    assert await anext(checkpoints) is not None
    await asyncio.sleep(0.1)
    await checkpoints.aclose()


@pytest.mark.asyncio
async def test_pending_writes_reject_unknown_channels_on_write_and_read() -> None:
    backend = InMemorySaver()
    saver = SchemaValidatedCheckpointSaver(backend)
    graph = _compiled(saver)
    binding = CheckpointBinding(
        tenant_id="tenant-a",
        deployment_id="deployment-a",
        graph_contract=graph_contract_fingerprint(graph),
        thread_id="thread-a",
    )
    saver.bind_checkpoint_contract(
        graph.channels,
        binding=binding,
        io_timeout_seconds=0.5,
        required_state_keys=_State.__annotations__,
    )
    await graph.ainvoke({"value": 1}, binding.config)
    saved = await backend.aget_tuple(binding.config)
    assert saved is not None

    with pytest.raises(CheckpointSchemaError, match="pending write"):
        await saver.aput_writes(
            saved.config,
            (("unknown_channel", 7),),
            "unknown-write-task",
        )

    await backend.aput_writes(
        saved.config,
        (("unknown_channel", 7),),
        "injected-unknown-write-task",
    )
    with pytest.raises(CheckpointSchemaError, match="pending write"):
        await saver.aget_tuple(binding.config)


@pytest.mark.asyncio
async def test_langgraph_reserved_pending_write_channels_remain_accepted() -> None:
    backend = InMemorySaver()
    saver = SchemaValidatedCheckpointSaver(backend)
    graph = _compiled(saver)
    binding = CheckpointBinding(
        tenant_id="tenant-a",
        deployment_id="deployment-a",
        graph_contract=graph_contract_fingerprint(graph),
        thread_id="thread-a",
    )
    saver.bind_checkpoint_contract(
        graph.channels,
        binding=binding,
        io_timeout_seconds=0.5,
        required_state_keys=_State.__annotations__,
    )
    await graph.ainvoke({"value": 1}, binding.config)
    saved = await backend.aget_tuple(binding.config)
    assert saved is not None

    await saver.aput_writes(saved.config, ((SCHEDULED, None),), "scheduled-task")
    assert await saver.aget_tuple(binding.config) is not None


@pytest.mark.asyncio
async def test_successful_delete_retires_thread_authorization() -> None:
    saver = SchemaValidatedCheckpointSaver(InMemorySaver())
    graph = _compiled(saver)
    binding = CheckpointBinding(
        tenant_id="tenant-a",
        deployment_id="deployment-a",
        graph_contract=graph_contract_fingerprint(graph),
        thread_id="thread-a",
    )
    saver.bind_checkpoint_contract(
        graph.channels,
        binding=binding,
        io_timeout_seconds=0.5,
        required_state_keys=_State.__annotations__,
    )
    await graph.ainvoke({"value": 1}, binding.config)

    await saver.adelete_thread(binding.storage_thread_id)

    with pytest.raises(CheckpointScopeError, match="namespace"):
        await saver.aget_tuple(binding.config)


@pytest.mark.asyncio
async def test_delete_keeps_thread_authorized_when_backend_retains_checkpoint() -> None:
    backend = _NoOpDeleteSaver()
    saver = SchemaValidatedCheckpointSaver(backend)
    graph = _compiled(saver)
    binding = CheckpointBinding(
        tenant_id="tenant-a",
        deployment_id="deployment-a",
        graph_contract=graph_contract_fingerprint(graph),
        thread_id="thread-a",
    )
    saver.bind_checkpoint_contract(
        graph.channels,
        binding=binding,
        io_timeout_seconds=0.5,
        required_state_keys=_State.__annotations__,
    )
    await graph.ainvoke({"value": 1}, binding.config)

    with pytest.raises(CheckpointDeletionError, match="still contains checkpoints"):
        await saver.adelete_thread(binding.storage_thread_id)

    assert saver.thread_authorized(binding.storage_thread_id)
    assert await saver.aget_tuple(binding.config) is not None


@pytest.mark.asyncio
async def test_failed_delete_keeps_thread_authorized_for_retry() -> None:
    backend = _FailDeleteOnceSaver()
    saver = SchemaValidatedCheckpointSaver(backend)
    graph = _compiled(saver)
    binding = CheckpointBinding(
        tenant_id="tenant-a",
        deployment_id="deployment-a",
        graph_contract=graph_contract_fingerprint(graph),
        thread_id="thread-a",
    )
    saver.bind_checkpoint_contract(
        graph.channels,
        binding=binding,
        io_timeout_seconds=0.5,
        required_state_keys=_State.__annotations__,
    )

    with pytest.raises(TimeoutError, match="injected deletion failure"):
        await saver.adelete_thread(binding.storage_thread_id)
    assert await saver.aget_tuple(binding.config) is None

    await saver.adelete_thread(binding.storage_thread_id)
    with pytest.raises(CheckpointScopeError, match="namespace"):
        await saver.aget_tuple(binding.config)
