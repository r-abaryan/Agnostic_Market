"""Checkpoint persistence boundary contracts."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Sequence
from typing import Any, TypedDict

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    empty_checkpoint,
)
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.types import SCHEDULED
from langgraph.graph import START, StateGraph
from langgraph.types import interrupt

from agnostic_market.checkpoints import (
    CheckpointBinding,
    CheckpointDeletionError,
    CheckpointScopeError,
    SchemaValidatedCheckpointSaver,
    SynchronousCheckpointOperationError,
    build_checkpoint_serializer,
    build_checkpointer,
    graph_contract_fingerprint,
)
from agnostic_market.dtos.state import CheckpointSchemaError
from agnostic_market.durability.encryption import AesGcmSessionCipher


class _State(TypedDict, total=False):
    value: int


class _SensitiveState(TypedDict, total=False):
    value: str


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


class _CapturingSaver(InMemorySaver):
    last_checkpoint: Checkpoint | None = None

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        self.last_checkpoint = checkpoint
        return super().put(config, checkpoint, metadata, new_versions)


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


@pytest.mark.parametrize(
    ("tenant_id", "expected_storage_thread_id"),
    (
        (
            "upgrade_acme",
            "cp_58106c20014afb0edd1b0aaa2e9557d73cb5badbfb3d6555b5ec357afd5cfdf1",
        ),
        (
            "café_store",
            "cp_b0579695a2b69bad9ecf1a1d073b03811f33fc90eb031698ad86d08405342cee",
        ),
        (
            "商店",
            "cp_df78d3b8e6a732df024a75c25c01e849c55ea214d0b4a5000c1f81aa878e7f7f",
        ),
    ),
)
def test_checkpoint_binding_version_1_storage_key_is_frozen(
    tenant_id: str,
    expected_storage_thread_id: str,
) -> None:
    binding = CheckpointBinding(
        tenant_id=tenant_id,
        logical_session_id="AD_same_session",
        deployment_id="deployment-a",
        graph_contract="graph-a",
        thread_id="AD_same_session::fence::1",
    )

    assert binding.storage_thread_id == expected_storage_thread_id


def test_encrypted_boundary_requires_an_explicit_logical_session() -> None:
    saver = build_checkpointer(cipher=_checkpoint_cipher())
    graph = _sensitive_compiled(saver)
    binding = CheckpointBinding(
        tenant_id="tenant-a",
        deployment_id="deployment-a",
        graph_contract=graph_contract_fingerprint(graph),
        thread_id="logical-session-a::fence::1",
    )

    with pytest.raises(ValueError, match="logical session"):
        saver.bind_checkpoint_contract(
            graph.channels,
            binding=binding,
            io_timeout_seconds=0.5,
            required_state_keys=_SensitiveState.__annotations__,
        )


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
async def test_encrypted_boundary_covers_checkpoint_metadata_values_and_pending_writes() -> None:
    sentinel = "checkpoint-plaintext-sentinel-6b"
    backend = InMemorySaver(serde=build_checkpoint_serializer())
    saver = build_checkpointer(
        backend,
        synchronous_operations=False,
        cipher=_checkpoint_cipher(),
    )
    graph = _sensitive_compiled(saver)
    binding = CheckpointBinding(
        tenant_id="tenant-a",
        logical_session_id="logical-session-a",
        deployment_id="deployment-a",
        graph_contract=graph_contract_fingerprint(graph),
        thread_id="logical-session-a::fence::1",
    )
    saver.bind_checkpoint_contract(
        graph.channels,
        binding=binding,
        io_timeout_seconds=0.5,
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
    saved_config = await saver.aput(
        saved.config,
        next_checkpoint,
        {**saved.metadata, "writes": {"retain": sentinel}},
        {},
    )
    await saver.aput_writes(saved_config, (("value", sentinel),), "sensitive-task")

    restored = await saver.aget_tuple(saved_config)
    raw = await backend.aget_tuple(saved_config)
    assert restored is not None
    assert restored.checkpoint["channel_values"]["value"] == sentinel
    assert restored.metadata["config_marker"] == sentinel
    assert restored.metadata["request_marker"] == sentinel
    assert "writes" not in restored.metadata
    assert any(write[1:] == ("value", sentinel) for write in restored.pending_writes)
    assert raw is not None
    assert sentinel not in repr(raw.checkpoint)
    assert sentinel not in repr(raw.metadata)
    assert sentinel not in repr(raw.pending_writes)
    assert set(raw.metadata) == {"__agnostic_checkpoint_payload__"}
    with pytest.raises(NotImplementedError, match="metadata filtering"):
        _ = [item async for item in saver.alist(binding.config, filter={"source": "loop"})]


@pytest.mark.asyncio
async def test_encrypted_checkpoint_detects_a_missing_channel_blob() -> None:
    backend = InMemorySaver(serde=build_checkpoint_serializer())
    saver = build_checkpointer(
        backend,
        synchronous_operations=False,
        cipher=_checkpoint_cipher(),
    )
    graph = _sensitive_compiled(saver)
    binding = CheckpointBinding(
        tenant_id="tenant-a",
        logical_session_id="logical-session-a",
        deployment_id="deployment-a",
        graph_contract=graph_contract_fingerprint(graph),
        thread_id="logical-session-a::fence::1",
    )
    saver.bind_checkpoint_contract(
        graph.channels,
        binding=binding,
        io_timeout_seconds=0.5,
        required_state_keys=_SensitiveState.__annotations__,
    )

    await graph.ainvoke({"value": "secret"}, binding.config)
    raw = await backend.aget_tuple(binding.config)
    assert raw is not None
    version = raw.checkpoint["channel_versions"]["value"]
    del backend.blobs[(binding.storage_thread_id, "", "value", version)]

    with pytest.raises(CheckpointSchemaError, match="channel manifest"):
        await saver.aget_tuple(binding.config)


@pytest.mark.asyncio
async def test_encrypted_checkpoint_persists_only_new_channel_versions() -> None:
    backend = _CapturingSaver(serde=build_checkpoint_serializer())
    saver = build_checkpointer(
        backend,
        synchronous_operations=False,
        cipher=_checkpoint_cipher(),
    )
    graph = _sensitive_compiled(saver)
    binding = CheckpointBinding(
        tenant_id="tenant-a",
        logical_session_id="logical-session-a",
        deployment_id="deployment-a",
        graph_contract=graph_contract_fingerprint(graph),
        thread_id="logical-session-a::fence::1",
    )
    saver.bind_checkpoint_contract(
        graph.channels,
        binding=binding,
        io_timeout_seconds=0.5,
        required_state_keys=_SensitiveState.__annotations__,
    )
    first_checkpoint = empty_checkpoint()
    first_checkpoint["channel_values"] = {"value": "secret"}
    first_checkpoint["channel_versions"] = {"value": "1"}

    first_config = await saver.aput(
        binding.config,
        first_checkpoint,
        CheckpointMetadata(),
        {"value": "1"},
    )
    assert backend.last_checkpoint is not None
    assert set(backend.last_checkpoint["channel_values"]) == {"value"}

    second_checkpoint = first_checkpoint.copy()
    second_checkpoint["id"] = str(uuid.uuid4())
    second_config = await saver.aput(
        first_config,
        second_checkpoint,
        CheckpointMetadata(),
        {},
    )
    assert backend.last_checkpoint is not None
    assert backend.last_checkpoint["channel_values"] == {}
    restored = await saver.aget_tuple(second_config)
    assert restored is not None
    assert restored.checkpoint["channel_values"]["value"] == "secret"


@pytest.mark.asyncio
async def test_encrypted_checkpoint_rejects_a_value_without_a_channel_version() -> None:
    saver = build_checkpointer(
        synchronous_operations=False,
        cipher=_checkpoint_cipher(),
    )
    graph = _sensitive_compiled(saver)
    binding = CheckpointBinding(
        tenant_id="tenant-a",
        logical_session_id="logical-session-a",
        deployment_id="deployment-a",
        graph_contract=graph_contract_fingerprint(graph),
        thread_id="logical-session-a::fence::1",
    )
    saver.bind_checkpoint_contract(
        graph.channels,
        binding=binding,
        io_timeout_seconds=0.5,
        required_state_keys=_SensitiveState.__annotations__,
    )
    checkpoint = empty_checkpoint()
    checkpoint["channel_values"] = {"value": "secret"}

    with pytest.raises(CheckpointSchemaError, match="channel versions"):
        await saver.aput(binding.config, checkpoint, CheckpointMetadata(), {})

    checkpoint["channel_versions"] = {"value": "1"}
    with pytest.raises(CheckpointSchemaError, match="new versions"):
        await saver.aput(
            binding.config,
            checkpoint,
            CheckpointMetadata(),
            {"value": "2"},
        )


def test_checkpoint_binding_rejects_internal_whitespace_before_binding() -> None:
    with pytest.raises(ValueError, match="tenant_id"):
        CheckpointBinding(
            tenant_id="tenant a",
            logical_session_id="logical-session-a",
            deployment_id="deployment-a",
            graph_contract="graph-contract-a",
            thread_id="logical-session-a::fence::1",
        )


@pytest.mark.asyncio
async def test_encrypted_checkpoint_rejects_payload_moved_to_another_session_context() -> None:
    backend = InMemorySaver(serde=build_checkpoint_serializer())
    saver = build_checkpointer(
        backend,
        synchronous_operations=False,
        cipher=_checkpoint_cipher(),
    )
    graph = _sensitive_compiled(saver)
    contract = graph_contract_fingerprint(graph)
    first = CheckpointBinding(
        tenant_id="tenant-a",
        logical_session_id="logical-session-a",
        deployment_id="deployment-a",
        graph_contract=contract,
        thread_id="logical-session-a::fence::1",
    )
    second = CheckpointBinding(
        tenant_id="tenant-a",
        logical_session_id="logical-session-b",
        deployment_id="deployment-a",
        graph_contract=contract,
        thread_id="logical-session-b::fence::1",
    )
    for binding in (first, second):
        saver.bind_checkpoint_contract(
            graph.channels,
            binding=binding,
            io_timeout_seconds=0.5,
            required_state_keys=_SensitiveState.__annotations__,
        )

    await graph.ainvoke({"value": "secret"}, first.config)
    raw = await backend.aget_tuple(first.config)
    assert raw is not None
    await backend.aput(
        second.config,
        raw.checkpoint,
        raw.metadata,
        raw.checkpoint["channel_versions"],
    )

    with pytest.raises(CheckpointSchemaError, match="authenticated"):
        await saver.aget_tuple(second.config)


@pytest.mark.asyncio
async def test_encrypted_checkpoint_rejects_payload_moved_to_another_langgraph_namespace() -> None:
    backend = InMemorySaver(serde=build_checkpoint_serializer())
    saver = build_checkpointer(
        backend,
        synchronous_operations=False,
        cipher=_checkpoint_cipher(),
    )
    binding = CheckpointBinding(
        tenant_id="tenant-a",
        logical_session_id="logical-session-a",
        deployment_id="deployment-a",
        graph_contract="graph-contract-a",
        thread_id="logical-session-a::fence::1",
    )
    saver.bind_checkpoint_contract(
        {"value"},
        binding=binding,
        io_timeout_seconds=0.5,
        required_state_keys=_SensitiveState.__annotations__,
    )
    checkpoint = empty_checkpoint()
    checkpoint["channel_values"] = {"value": "secret"}
    checkpoint["channel_versions"] = {"value": "1"}
    source: RunnableConfig = {
        "configurable": {
            "thread_id": binding.storage_thread_id,
            "checkpoint_ns": "subgraph-a",
        }
    }
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


@pytest.mark.asyncio
async def test_encrypted_checkpoint_rejects_a_missing_checkpoint_version() -> None:
    saver = build_checkpointer(
        synchronous_operations=False,
        cipher=_checkpoint_cipher(),
    )
    binding = CheckpointBinding(
        tenant_id="tenant-a",
        logical_session_id="logical-session-a",
        deployment_id="deployment-a",
        graph_contract="graph-contract-a",
        thread_id="logical-session-a::fence::1",
    )
    saver.bind_checkpoint_contract(
        {"value"},
        binding=binding,
        io_timeout_seconds=0.5,
        required_state_keys=_SensitiveState.__annotations__,
    )
    checkpoint = empty_checkpoint()
    del checkpoint["v"]
    checkpoint["channel_values"] = {"value": "secret"}
    checkpoint["channel_versions"] = {"value": "1"}

    with pytest.raises(CheckpointSchemaError, match="version"):
        await saver.aput(
            binding.config,
            checkpoint,
            CheckpointMetadata(),
            {"value": "1"},
        )


@pytest.mark.asyncio
async def test_encrypted_checkpoint_lists_distinct_langgraph_namespaces() -> None:
    backend = InMemorySaver(serde=build_checkpoint_serializer())
    saver = build_checkpointer(
        backend,
        synchronous_operations=False,
        cipher=_checkpoint_cipher(),
    )
    binding = CheckpointBinding(
        tenant_id="tenant-a",
        logical_session_id="logical-session-a",
        deployment_id="deployment-a",
        graph_contract="graph-contract-a",
        thread_id="logical-session-a::fence::1",
    )
    saver.bind_checkpoint_contract(
        {"value"},
        binding=binding,
        io_timeout_seconds=0.5,
        required_state_keys=_SensitiveState.__annotations__,
    )
    for checkpoint_namespace, value in (
        ("subgraph-a", "first-secret"),
        ("subgraph-b", "second-secret"),
    ):
        checkpoint = empty_checkpoint()
        checkpoint["channel_values"] = {"value": value}
        checkpoint["channel_versions"] = {"value": "1"}
        await saver.aput(
            {
                "configurable": {
                    "thread_id": binding.storage_thread_id,
                    "checkpoint_ns": checkpoint_namespace,
                }
            },
            checkpoint,
            CheckpointMetadata(),
            {"value": "1"},
        )

    listed = [
        saved
        async for saved in saver.alist({"configurable": {"thread_id": binding.storage_thread_id}})
    ]

    assert {
        (
            saved.config["configurable"]["checkpoint_ns"],
            saved.checkpoint["channel_values"]["value"],
        )
        for saved in listed
    } == {
        ("subgraph-a", "first-secret"),
        ("subgraph-b", "second-secret"),
    }


@pytest.mark.asyncio
async def test_encrypted_checkpoint_point_read_defaults_to_the_root_namespace() -> None:
    saver = build_checkpointer(
        synchronous_operations=False,
        cipher=_checkpoint_cipher(),
    )
    binding = CheckpointBinding(
        tenant_id="tenant-a",
        logical_session_id="logical-session-a",
        deployment_id="deployment-a",
        graph_contract="graph-contract-a",
        thread_id="logical-session-a::fence::1",
    )
    saver.bind_checkpoint_contract(
        {"value"},
        binding=binding,
        io_timeout_seconds=0.5,
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

    restored = await saver.aget_tuple(
        {
            "configurable": {
                "thread_id": binding.storage_thread_id,
                "checkpoint_id": saved_config["configurable"]["checkpoint_id"],
            }
        }
    )

    assert restored is not None
    assert restored.checkpoint["channel_values"]["value"] == "secret"


@pytest.mark.asyncio
async def test_encrypted_checkpoint_rejects_another_deployment_or_graph_contract() -> None:
    backend = InMemorySaver(serde=build_checkpoint_serializer())
    source_saver = build_checkpointer(
        backend,
        synchronous_operations=False,
        cipher=_checkpoint_cipher(),
    )
    source_graph = _sensitive_compiled(source_saver)
    contract = graph_contract_fingerprint(source_graph)
    source = CheckpointBinding(
        tenant_id="tenant-a",
        logical_session_id="logical-session-a",
        deployment_id="deployment-a",
        graph_contract=contract,
        thread_id="logical-session-a::fence::1",
    )
    source_saver.bind_checkpoint_contract(
        source_graph.channels,
        binding=source,
        io_timeout_seconds=0.5,
        required_state_keys=_SensitiveState.__annotations__,
    )
    await source_graph.ainvoke({"value": "secret"}, source.config)
    raw = await backend.aget_tuple(source.config)
    assert raw is not None

    targets = (
        CheckpointBinding(
            tenant_id="tenant-a",
            logical_session_id="logical-session-a",
            deployment_id="deployment-b",
            graph_contract=contract,
            thread_id="logical-session-a::fence::1",
        ),
        CheckpointBinding(
            tenant_id="tenant-a",
            logical_session_id="logical-session-a",
            deployment_id="deployment-a",
            graph_contract="different-graph-contract",
            thread_id="logical-session-a::fence::1",
        ),
    )
    for target in targets:
        target_saver = build_checkpointer(
            backend,
            synchronous_operations=False,
            cipher=_checkpoint_cipher(),
        )
        target_graph = _sensitive_compiled(target_saver)
        target_saver.bind_checkpoint_contract(
            target_graph.channels,
            binding=target,
            io_timeout_seconds=0.5,
            required_state_keys=_SensitiveState.__annotations__,
        )
        await backend.aput(
            target.config,
            raw.checkpoint,
            raw.metadata,
            raw.checkpoint["channel_versions"],
        )

        with pytest.raises(CheckpointSchemaError, match="authenticated"):
            await target_saver.aget_tuple(target.config)


@pytest.mark.asyncio
async def test_encrypted_checkpoint_authenticates_its_parent_relationship() -> None:
    backend = InMemorySaver(serde=build_checkpoint_serializer())
    saver = build_checkpointer(
        backend,
        synchronous_operations=False,
        cipher=_checkpoint_cipher(),
    )
    graph = _sensitive_compiled(saver)
    binding = CheckpointBinding(
        tenant_id="tenant-a",
        logical_session_id="logical-session-a",
        deployment_id="deployment-a",
        graph_contract=graph_contract_fingerprint(graph),
        thread_id="logical-session-a::fence::1",
    )
    saver.bind_checkpoint_contract(
        graph.channels,
        binding=binding,
        io_timeout_seconds=0.5,
        required_state_keys=_SensitiveState.__annotations__,
    )

    await graph.ainvoke({"value": "secret"}, binding.config)
    raw = await backend.aget_tuple(binding.config)
    assert raw is not None
    checkpoint_id = raw.config["configurable"]["checkpoint_id"]
    stored_checkpoint, stored_metadata, parent_checkpoint_id = backend.storage[
        binding.storage_thread_id
    ][""][checkpoint_id]
    assert parent_checkpoint_id is not None
    backend.storage[binding.storage_thread_id][""][checkpoint_id] = (
        stored_checkpoint,
        stored_metadata,
        "tampered-parent",
    )

    with pytest.raises(CheckpointSchemaError, match="authenticate"):
        await saver.aget_tuple(binding.config)


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
async def test_interrupt_observation_does_not_project_graph_internal_state_channels() -> None:
    backend = InMemorySaver()
    graph_saver = SchemaValidatedCheckpointSaver(backend)
    graph = StateGraph(_State)
    graph.add_node("choose", lambda state: {"value": state.get("value", 0) + 1})
    graph.add_node("confirm", lambda _state: interrupt("confirm"))
    graph.add_edge(START, "choose")
    graph.add_conditional_edges(
        "choose",
        lambda _state: "confirm",
        {"confirm": "confirm"},
    )
    compiled = graph.compile(checkpointer=graph_saver)
    binding = CheckpointBinding(
        tenant_id="tenant-a",
        deployment_id="deployment-a",
        graph_contract=graph_contract_fingerprint(compiled),
        thread_id="thread-a",
    )
    graph_saver.bind_checkpoint_contract(
        compiled.channels,
        binding=binding,
        io_timeout_seconds=0.5,
        required_state_keys=_State.__annotations__,
    )
    await compiled.ainvoke({"value": 1}, binding.config)
    raw = await backend.aget_tuple(binding.config)
    assert raw is not None
    assert any(channel.startswith("branch:to:") for channel in raw.checkpoint["channel_values"])

    observer = SchemaValidatedCheckpointSaver(backend)
    observer.bind_checkpoint_contract(
        _State.__annotations__,
        binding=binding,
        io_timeout_seconds=0.5,
        required_state_keys=_State.__annotations__,
    )

    assert await observer.acheckpoint_has_pending_interrupt(binding.config) is True
    with pytest.raises(CheckpointSchemaError, match="unknown channels"):
        await observer.aget_tuple(binding.config)


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
