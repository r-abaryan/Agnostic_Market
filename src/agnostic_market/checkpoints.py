"""Tenant-scoped, schema-validating LangGraph checkpoint boundary."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import threading
from collections.abc import AsyncIterator, Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph._internal._constants import (
    ERROR,
    ERROR_SOURCE_NODE,
    INPUT,
    INTERRUPT,
    NO_WRITES,
    PREVIOUS,
    RESUME,
    RETURN,
    TASKS,
)
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.serde.types import SCHEDULED
from langgraph.graph.state import CompiledStateGraph

from agnostic_market.dtos.orchestration import (
    ActiveInvocation,
    CancellableOrderScope,
    CapabilityDispatchEnvelope,
    CapabilityId,
    RouterNoActionEnvelope,
)
from agnostic_market.dtos.recovery import ExceptionAction, PendingRecovery
from agnostic_market.dtos.state import (
    CHECKPOINT_SCHEMA_VERSION,
    CartClarification,
    CheckpointSchemaError,
    ClarificationLiveness,
    HandoffRequest,
    HandoffSource,
    IdentityClarification,
    PendingCancelBatch,
    PendingCartMutation,
    PendingIdentity,
    PendingPlacement,
    PendingProfileChange,
    PendingRefund,
    PendingReturn,
    ReasoningState,
    StateSchemaError,
    SupportClarification,
    validate_reasoning_state_keys,
)

CHECKPOINT_EXECUTION_CONTRACT_VERSION = "1"
_STORAGE_THREAD_RE = re.compile(r"cp_[0-9a-f]{64}\Z")
_LANGGRAPH_PENDING_WRITE_CHANNELS = frozenset(
    {
        INPUT,
        INTERRUPT,
        RESUME,
        ERROR,
        ERROR_SOURCE_NODE,
        NO_WRITES,
        TASKS,
        RETURN,
        PREVIOUS,
        SCHEDULED,
    }
)

# Top-level state DTOs are the serialized trust boundary. Nested models are reconstructed by
# their validated owner; custom enum values survive Pydantic's Python-mode dump independently.
_CHECKPOINT_CHANNEL_DTOS = (
    ActiveInvocation,
    CapabilityDispatchEnvelope,
    RouterNoActionEnvelope,
    PendingCartMutation,
    PendingPlacement,
    PendingRefund,
    CancellableOrderScope,
    PendingCancelBatch,
    PendingReturn,
    PendingProfileChange,
    PendingIdentity,
    PendingRecovery,
    IdentityClarification,
    SupportClarification,
    CartClarification,
    ClarificationLiveness,
    HandoffRequest,
)
_CHECKPOINT_NESTED_ENUMS = (
    CapabilityId,
    ExceptionAction,
    HandoffSource,
)


class CheckpointScopeError(ValueError):
    """A checkpoint operation escaped its tenant and deployment namespace."""


class CheckpointDeletionError(RuntimeError):
    """A backend acknowledged deletion but retained a readable checkpoint."""


class SynchronousCheckpointOperationError(RuntimeError):
    """A synchronous checkpoint operation reached an async-only boundary."""


def _fingerprint(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def graph_contract_fingerprint(graph: CompiledStateGraph) -> str:
    """Fingerprint resumability-relevant topology and state contracts."""
    topology = graph.get_graph().to_json()
    nodes = sorted(
        (str(node["id"]), str(node.get("type", ""))) for node in topology.get("nodes", ())
    )
    edges = sorted(
        (
            str(edge["source"]),
            str(edge["target"]),
            bool(edge.get("conditional", False)),
        )
        for edge in topology.get("edges", ())
    )
    return _fingerprint(
        {
            "execution_contract_version": CHECKPOINT_EXECUTION_CONTRACT_VERSION,
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "nodes": nodes,
            "edges": edges,
            "state_schema": ReasoningState.model_json_schema(),
        }
    )


@dataclass(frozen=True, slots=True)
class CheckpointBinding:
    """Logical session binding mapped to tenant-scoped physical checkpoint keys."""

    tenant_id: str
    deployment_id: str
    graph_contract: str
    thread_id: str

    def __post_init__(self) -> None:
        values = (self.tenant_id, self.deployment_id, self.graph_contract, self.thread_id)
        if any(not value.strip() for value in values):
            raise ValueError("checkpoint binding values must be non-empty")

    @property
    def namespace(self) -> str:
        return "ns_" + _fingerprint(
            {
                "tenant_id": self.tenant_id,
                "deployment_id": self.deployment_id,
                "graph_contract": self.graph_contract,
            }
        )

    @property
    def storage_thread_id(self) -> str:
        return "cp_" + _fingerprint(
            {
                "namespace": self.namespace,
                "thread_id": self.thread_id,
            }
        )

    @property
    def config(self) -> RunnableConfig:
        return {
            "configurable": {
                "thread_id": self.storage_thread_id,
                # LangGraph owns checkpoint_ns for subgraph traversal. Tenant and deployment
                # scope are encoded into the physical thread key and authorized by the wrapper.
                "checkpoint_ns": "",
            }
        }

    def rotate(self, thread_id: str) -> CheckpointBinding:
        return CheckpointBinding(
            tenant_id=self.tenant_id,
            deployment_id=self.deployment_id,
            graph_contract=self.graph_contract,
            thread_id=thread_id,
        )


class SchemaValidatedCheckpointSaver(BaseCheckpointSaver):
    """Validate one graph contract while delegating storage to any LangGraph saver."""

    def __init__(
        self,
        backend: BaseCheckpointSaver,
        *,
        synchronous_operations: bool = True,
    ) -> None:
        super().__init__(serde=backend.serde)
        self._backend = backend
        self._synchronous_operations = synchronous_operations
        self._allowed_checkpoint_channels: frozenset[str] | None = None
        self._graph_contract: str | None = None
        self._authorized_thread_ids: set[str] = set()
        self._io_timeout_seconds: float | None = None
        self._binding_lock = threading.RLock()

    @property
    def config_specs(self) -> list:
        return self._backend.config_specs

    def _require_synchronous_operations(self) -> None:
        if not self._synchronous_operations:
            raise SynchronousCheckpointOperationError(
                "this checkpoint boundary supports asynchronous operations only"
            )

    def bind_checkpoint_contract(
        self,
        channels: Collection[str],
        *,
        binding: CheckpointBinding,
        io_timeout_seconds: float,
        required_state_keys: Collection[str] = ReasoningState.model_fields,
    ) -> None:
        with self._binding_lock:
            if io_timeout_seconds <= 0:
                raise ValueError("checkpoint I/O timeout must be positive")
            allowed = frozenset(channels)
            missing = frozenset(required_state_keys) - allowed
            if missing:
                raise ValueError(
                    f"compiled graph omits reasoning-state fields: {sorted(missing)!r}"
                )
            if self._allowed_checkpoint_channels not in (None, allowed):
                raise ValueError("checkpointer is already bound to a different graph schema")
            if self._graph_contract not in (None, binding.graph_contract):
                raise ValueError("checkpointer is already bound to a different graph contract")
            if self._io_timeout_seconds not in (None, io_timeout_seconds):
                raise ValueError("checkpointer is already bound to a different I/O timeout")
            self._allowed_checkpoint_channels = allowed
            self._graph_contract = binding.graph_contract
            self._io_timeout_seconds = io_timeout_seconds
            self._authorized_thread_ids.add(binding.storage_thread_id)

    async def _bounded(self, operation):
        if self._io_timeout_seconds is None:
            raise CheckpointScopeError("checkpoint contract is not bound")
        async with asyncio.timeout(self._io_timeout_seconds):
            return await operation

    def _validate_config(self, config: RunnableConfig) -> None:
        configurable = config.get("configurable")
        if not isinstance(configurable, Mapping):
            raise CheckpointScopeError("checkpoint configuration has no namespace")
        thread_id = configurable.get("thread_id")
        with self._binding_lock:
            if self._graph_contract is None:
                raise CheckpointScopeError("checkpoint contract is not bound")
            if (
                not isinstance(thread_id, str)
                or _STORAGE_THREAD_RE.fullmatch(thread_id) is None
                or thread_id not in self._authorized_thread_ids
            ):
                raise CheckpointScopeError(
                    "checkpoint thread is outside the bound tenant and deployment namespace"
                )

    def _validate_checkpoint(self, checkpoint: Checkpoint) -> None:
        if self._allowed_checkpoint_channels is None:
            raise CheckpointScopeError("checkpoint contract is not bound")
        channel_values = checkpoint.get("channel_values")
        if not isinstance(channel_values, Mapping):
            raise CheckpointSchemaError("persisted checkpoint channels are malformed")
        try:
            validate_reasoning_state_keys(
                channel_values,
                allowed_keys=self._allowed_checkpoint_channels,
                source="persisted checkpoint",
            )
        except StateSchemaError as exc:
            raise CheckpointSchemaError("persisted checkpoint has unknown channels") from exc

    def _validate_pending_write_channels(
        self,
        channels: Collection[object],
    ) -> None:
        if self._allowed_checkpoint_channels is None:
            raise CheckpointScopeError("checkpoint contract is not bound")
        allowed = self._allowed_checkpoint_channels | _LANGGRAPH_PENDING_WRITE_CHANNELS
        if any(not isinstance(channel, str) or channel not in allowed for channel in channels):
            raise CheckpointSchemaError("pending write has an unknown channel")

    def _validate_tuple(self, saved: CheckpointTuple | None) -> CheckpointTuple | None:
        if saved is not None:
            self._validate_checkpoint(saved.checkpoint)
            self._validate_pending_write_channels(
                [write[1] for write in saved.pending_writes or ()]
            )
        return saved

    def _validate_storage_thread_id(self, thread_id: str) -> None:
        with self._binding_lock:
            if (
                _STORAGE_THREAD_RE.fullmatch(thread_id) is None
                or thread_id not in self._authorized_thread_ids
            ):
                raise CheckpointScopeError(
                    "checkpoint deletion is outside the bound tenant and deployment namespace"
                )

    def thread_authorized(self, thread_id: str) -> bool:
        with self._binding_lock:
            return bool(
                _STORAGE_THREAD_RE.fullmatch(thread_id) and thread_id in self._authorized_thread_ids
            )

    @staticmethod
    def _thread_listing_config(thread_id: str) -> RunnableConfig:
        return {"configurable": {"thread_id": thread_id}}

    def _verify_thread_deleted(self, thread_id: str) -> None:
        listing = iter(self._backend.list(self._thread_listing_config(thread_id), limit=1))
        try:
            remaining = next(listing, None)
        finally:
            close = getattr(listing, "close", None)
            if close is not None:
                close()
        if remaining is not None:
            raise CheckpointDeletionError("checkpoint backend still contains checkpoints")

    async def _averify_thread_deleted(self, thread_id: str) -> None:
        listing = self._backend.alist(
            self._thread_listing_config(thread_id),
            limit=1,
        ).__aiter__()
        try:
            try:
                await self._bounded(anext(listing))
            except StopAsyncIteration:
                return
            raise CheckpointDeletionError("checkpoint backend still contains checkpoints")
        finally:
            close = getattr(listing, "aclose", None)
            if close is not None:
                await self._bounded(close())

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        self._require_synchronous_operations()
        self._validate_config(config)
        return self._validate_tuple(self._backend.get_tuple(config))

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        self._require_synchronous_operations()
        if config is None:
            raise CheckpointScopeError("unscoped checkpoint listing is forbidden")
        self._validate_config(config)
        if before is not None:
            self._validate_config(before)
        for saved in self._backend.list(config, filter=filter, before=before, limit=limit):
            yield self._validate_tuple(saved)  # type: ignore[misc]

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        self._require_synchronous_operations()
        self._validate_config(config)
        self._validate_checkpoint(checkpoint)
        return self._backend.put(config, checkpoint, metadata, new_versions)

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        self._require_synchronous_operations()
        self._validate_config(config)
        self._validate_pending_write_channels([channel for channel, _value in writes])
        self._backend.put_writes(config, writes, task_id, task_path)

    def delete_thread(self, thread_id: str) -> None:
        self._require_synchronous_operations()
        self._validate_storage_thread_id(thread_id)
        self._backend.delete_thread(thread_id)
        self._verify_thread_deleted(thread_id)
        with self._binding_lock:
            self._authorized_thread_ids.discard(thread_id)

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        self._validate_config(config)
        return self._validate_tuple(await self._bounded(self._backend.aget_tuple(config)))

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        if config is None:
            raise CheckpointScopeError("unscoped checkpoint listing is forbidden")
        self._validate_config(config)
        if before is not None:
            self._validate_config(before)
        listing = self._backend.alist(
            config,
            filter=filter,
            before=before,
            limit=limit,
        ).__aiter__()
        while True:
            try:
                saved = await self._bounded(anext(listing))
            except StopAsyncIteration:
                return
            validated = self._validate_tuple(saved)
            assert validated is not None
            yield validated

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        self._validate_config(config)
        self._validate_checkpoint(checkpoint)
        return await self._bounded(self._backend.aput(config, checkpoint, metadata, new_versions))

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        self._validate_config(config)
        self._validate_pending_write_channels([channel for channel, _value in writes])
        await self._bounded(self._backend.aput_writes(config, writes, task_id, task_path))

    async def adelete_thread(self, thread_id: str) -> None:
        self._validate_storage_thread_id(thread_id)
        await self._bounded(self._backend.adelete_thread(thread_id))
        await self._averify_thread_deleted(thread_id)
        with self._binding_lock:
            self._authorized_thread_ids.discard(thread_id)

    async def aclear_thread(self, thread_id: str) -> None:
        """Delete persisted contents while retaining authority to rebuild the thread."""
        self._validate_storage_thread_id(thread_id)
        await self._bounded(self._backend.adelete_thread(thread_id))
        await self._averify_thread_deleted(thread_id)

    def get_next_version(self, current, channel):
        return self._backend.get_next_version(current, channel)


def build_checkpointer(
    backend: BaseCheckpointSaver | None = None,
    *,
    synchronous_operations: bool = True,
) -> SchemaValidatedCheckpointSaver:
    """Build the strict boundary over an in-memory or injected durable saver."""
    if backend is None:
        backend = InMemorySaver(serde=build_checkpoint_serializer())
    return SchemaValidatedCheckpointSaver(
        backend,
        synchronous_operations=synchronous_operations,
    )


def build_checkpoint_serializer() -> JsonPlusSerializer:
    """Build the allowlisted serializer shared by every checkpoint backend."""
    return JsonPlusSerializer(
        allowed_msgpack_modules=[*_CHECKPOINT_CHANNEL_DTOS, *_CHECKPOINT_NESTED_ENUMS]
    )
