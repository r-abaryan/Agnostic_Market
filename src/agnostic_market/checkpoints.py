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
    get_serializable_checkpoint_metadata,
)
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.serde.types import SCHEDULED
from langgraph.graph.state import CompiledStateGraph
from pydantic import TypeAdapter

from agnostic_market.dtos.orchestration import (
    ActiveInvocation,
    CancellableOrderScope,
    CapabilityDispatchEnvelope,
    CapabilityId,
    RouterNoActionEnvelope,
)
from agnostic_market.dtos.recovery import ExceptionAction, PendingRecovery
from agnostic_market.dtos.session import AuthorityIdentifier
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
from agnostic_market.durability.checkpoint_encryption import (
    CheckpointCipherScope,
    EncryptedCheckpointCodec,
)
from agnostic_market.durability.encryption import AesGcmSessionCipher

CHECKPOINT_EXECUTION_CONTRACT_VERSION = "1"
_STORAGE_THREAD_RE = re.compile(r"cp_[0-9a-f]{64}\Z")
_AUTHORITY_IDENTIFIER = TypeAdapter(AuthorityIdentifier)
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
    logical_session_id: str | None = None

    def __post_init__(self) -> None:
        values = {
            "tenant_id": self.tenant_id,
            "deployment_id": self.deployment_id,
            "graph_contract": self.graph_contract,
            "thread_id": self.thread_id,
        }
        if self.logical_session_id is not None:
            values["logical_session_id"] = self.logical_session_id
        for name, value in values.items():
            try:
                _AUTHORITY_IDENTIFIER.validate_python(value, strict=True)
            except ValueError as exc:
                raise ValueError(f"checkpoint binding has an invalid {name}") from exc

    def encryption_scope(self, langgraph_checkpoint_namespace: str) -> CheckpointCipherScope:
        if self.logical_session_id is None:
            raise ValueError("encrypted checkpoints require an explicit logical session id")
        return CheckpointCipherScope(
            tenant_id=self.tenant_id,
            logical_session_id=self.logical_session_id,
            checkpoint_generation_namespace=self.thread_id,
            langgraph_checkpoint_namespace=langgraph_checkpoint_namespace,
            deployment_id=self.deployment_id,
            graph_contract=self.graph_contract,
        )

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
            logical_session_id=self.logical_session_id,
        )


class SchemaValidatedCheckpointSaver(BaseCheckpointSaver):
    """Validate one graph contract while delegating storage to any LangGraph saver."""

    def __init__(
        self,
        backend: BaseCheckpointSaver,
        *,
        synchronous_operations: bool = True,
        cipher: AesGcmSessionCipher | None = None,
    ) -> None:
        super().__init__(serde=backend.serde)
        self._backend = backend
        self._synchronous_operations = synchronous_operations
        self._checkpoint_codec = (
            EncryptedCheckpointCodec(self.serde, cipher) if cipher is not None else None
        )
        self._allowed_checkpoint_channels: frozenset[str] | None = None
        self._graph_contract: str | None = None
        self._bindings: dict[str, CheckpointBinding] = {}
        self._io_timeout_seconds: float | None = None
        self._binding_lock = threading.RLock()

    @property
    def config_specs(self) -> list:
        return self._backend.config_specs

    @property
    def encryption_enabled(self) -> bool:
        return self._checkpoint_codec is not None

    def uses_storage_backend(self, backend_type: type[BaseCheckpointSaver]) -> bool:
        return isinstance(self._backend, backend_type)

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
            if self._checkpoint_codec is not None and binding.logical_session_id is None:
                raise ValueError("encrypted checkpoints require an explicit logical session id")
            self._allowed_checkpoint_channels = allowed
            self._graph_contract = binding.graph_contract
            self._io_timeout_seconds = io_timeout_seconds
            prior_binding = self._bindings.get(binding.storage_thread_id)
            if prior_binding is not None and prior_binding != binding:
                raise ValueError("checkpoint storage key is already bound to another context")
            self._bindings[binding.storage_thread_id] = binding

    async def _bounded(self, operation):
        if self._io_timeout_seconds is None:
            raise CheckpointScopeError("checkpoint contract is not bound")
        async with asyncio.timeout(self._io_timeout_seconds):
            return await operation

    def _binding_for_config(self, config: RunnableConfig) -> CheckpointBinding:
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
                or thread_id not in self._bindings
            ):
                raise CheckpointScopeError(
                    "checkpoint thread is outside the bound tenant and deployment namespace"
                )
            return self._bindings[thread_id]

    def _validate_config(self, config: RunnableConfig) -> None:
        self._binding_for_config(config)

    @staticmethod
    def _configured_checkpoint_namespace(config: RunnableConfig) -> str | None:
        configurable = config.get("configurable")
        if not isinstance(configurable, Mapping):
            raise CheckpointScopeError("checkpoint configuration has no namespace")
        checkpoint_ns = configurable.get("checkpoint_ns")
        if checkpoint_ns is not None and not isinstance(checkpoint_ns, str):
            raise CheckpointScopeError("checkpoint storage configuration is malformed")
        return checkpoint_ns

    @classmethod
    def _required_checkpoint_namespace(cls, config: RunnableConfig) -> str:
        checkpoint_ns = cls._configured_checkpoint_namespace(config)
        if checkpoint_ns is None:
            raise CheckpointScopeError("checkpoint storage configuration is malformed")
        return checkpoint_ns

    @classmethod
    def _storage_config(cls, config: RunnableConfig) -> RunnableConfig:
        configurable = config.get("configurable")
        if not isinstance(configurable, Mapping):
            raise CheckpointScopeError("checkpoint configuration has no namespace")
        thread_id = configurable.get("thread_id")
        checkpoint_ns = cls._required_checkpoint_namespace(config)
        if not isinstance(thread_id, str):
            raise CheckpointScopeError("checkpoint storage configuration is malformed")
        storage_values = {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
        }
        checkpoint_id = configurable.get("checkpoint_id")
        if checkpoint_id is not None:
            if not isinstance(checkpoint_id, str) or not checkpoint_id:
                raise CheckpointScopeError("checkpoint storage configuration is malformed")
            storage_values["checkpoint_id"] = checkpoint_id
        return {"configurable": storage_values}

    def _prepare_checkpoint_put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
        binding: CheckpointBinding,
    ) -> tuple[RunnableConfig, Checkpoint, CheckpointMetadata]:
        if self._checkpoint_codec is None:
            return config, checkpoint, metadata
        storage_config = self._storage_config(config)
        encryption_scope = binding.encryption_scope(
            self._required_checkpoint_namespace(storage_config)
        )
        effective_metadata = get_serializable_checkpoint_metadata(config, metadata)
        encrypted_checkpoint = self._checkpoint_codec.encrypt_checkpoint(
            checkpoint,
            encryption_scope,
            parent_config=storage_config,
            new_versions=new_versions,
        )
        encrypted_metadata = self._checkpoint_codec.encrypt_metadata(
            effective_metadata,
            encrypted_checkpoint["id"],
            encryption_scope,
        )
        return storage_config, encrypted_checkpoint, encrypted_metadata

    def _prepare_pending_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        binding: CheckpointBinding,
    ) -> Sequence[tuple[str, Any]]:
        self._validate_pending_write_channels([channel for channel, _value in writes])
        if self._checkpoint_codec is None:
            return writes
        return self._checkpoint_codec.encrypt_writes(
            config,
            writes,
            task_id,
            binding.encryption_scope(self._required_checkpoint_namespace(config)),
        )

    def _decrypt_saved_tuple(
        self,
        saved: CheckpointTuple | None,
        binding: CheckpointBinding,
        *,
        expected_namespace: str | None,
    ) -> CheckpointTuple | None:
        if self._checkpoint_codec is None or saved is None:
            return saved
        saved_namespace = self._configured_checkpoint_namespace(saved.config)
        if saved_namespace is None:
            if expected_namespace is None:
                raise CheckpointScopeError("checkpoint backend returned no storage namespace")
            saved_namespace = expected_namespace
        if expected_namespace is not None and saved_namespace != expected_namespace:
            raise CheckpointScopeError("checkpoint backend returned an unexpected namespace")
        return self._checkpoint_codec.decrypt_tuple(
            saved,
            binding.encryption_scope(saved_namespace),
        )

    def _reject_unsupported_filter(self, filter: dict[str, Any] | None) -> None:
        if self._checkpoint_codec is not None and filter is not None:
            raise NotImplementedError("metadata filtering is unavailable for encrypted checkpoints")

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
            if _STORAGE_THREAD_RE.fullmatch(thread_id) is None or thread_id not in self._bindings:
                raise CheckpointScopeError(
                    "checkpoint deletion is outside the bound tenant and deployment namespace"
                )

    def thread_authorized(self, thread_id: str) -> bool:
        with self._binding_lock:
            return bool(_STORAGE_THREAD_RE.fullmatch(thread_id) and thread_id in self._bindings)

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
        binding = self._binding_for_config(config)
        saved = self._backend.get_tuple(config)
        expected_namespace = self._configured_checkpoint_namespace(config)
        saved = self._decrypt_saved_tuple(
            saved,
            binding,
            expected_namespace="" if expected_namespace is None else expected_namespace,
        )
        return self._validate_tuple(saved)

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
        binding = self._binding_for_config(config)
        expected_namespace = self._configured_checkpoint_namespace(config)
        if before is not None:
            self._validate_config(before)
        self._reject_unsupported_filter(filter)
        for saved in self._backend.list(config, filter=filter, before=before, limit=limit):
            saved = self._decrypt_saved_tuple(
                saved,
                binding,
                expected_namespace=expected_namespace,
            )
            assert saved is not None
            yield self._validate_tuple(saved)  # type: ignore[misc]

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        self._require_synchronous_operations()
        binding = self._binding_for_config(config)
        self._validate_checkpoint(checkpoint)
        config, checkpoint, metadata = self._prepare_checkpoint_put(
            config,
            checkpoint,
            metadata,
            new_versions,
            binding,
        )
        return self._backend.put(config, checkpoint, metadata, new_versions)

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        self._require_synchronous_operations()
        binding = self._binding_for_config(config)
        writes = self._prepare_pending_writes(config, writes, task_id, binding)
        self._backend.put_writes(config, writes, task_id, task_path)

    def delete_thread(self, thread_id: str) -> None:
        self._require_synchronous_operations()
        self._validate_storage_thread_id(thread_id)
        self._backend.delete_thread(thread_id)
        self._verify_thread_deleted(thread_id)
        with self._binding_lock:
            self._bindings.pop(thread_id, None)

    async def _aget_decrypted_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        binding = self._binding_for_config(config)
        saved = await self._bounded(self._backend.aget_tuple(config))
        expected_namespace = self._configured_checkpoint_namespace(config)
        return self._decrypt_saved_tuple(
            saved,
            binding,
            expected_namespace="" if expected_namespace is None else expected_namespace,
        )

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return self._validate_tuple(await self._aget_decrypted_tuple(config))

    async def acheckpoint_has_pending_interrupt(self, config: RunnableConfig) -> bool:
        """Read the persisted LangGraph interrupt signal through this saver authority."""
        saved = await self._aget_decrypted_tuple(config)
        if saved is not None:
            self._validate_pending_write_channels(
                [write[1] for write in saved.pending_writes or ()]
            )
        return bool(
            saved is not None
            and any(
                channel == INTERRUPT for _task_id, channel, _value in saved.pending_writes or ()
            )
        )

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
        binding = self._binding_for_config(config)
        expected_namespace = self._configured_checkpoint_namespace(config)
        if before is not None:
            self._validate_config(before)
        self._reject_unsupported_filter(filter)
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
            saved = self._decrypt_saved_tuple(
                saved,
                binding,
                expected_namespace=expected_namespace,
            )
            assert saved is not None
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
        binding = self._binding_for_config(config)
        self._validate_checkpoint(checkpoint)
        config, checkpoint, metadata = self._prepare_checkpoint_put(
            config,
            checkpoint,
            metadata,
            new_versions,
            binding,
        )
        return await self._bounded(self._backend.aput(config, checkpoint, metadata, new_versions))

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        binding = self._binding_for_config(config)
        writes = self._prepare_pending_writes(config, writes, task_id, binding)
        await self._bounded(self._backend.aput_writes(config, writes, task_id, task_path))

    async def adelete_thread(self, thread_id: str) -> None:
        self._validate_storage_thread_id(thread_id)
        await self._bounded(self._backend.adelete_thread(thread_id))
        await self._averify_thread_deleted(thread_id)
        with self._binding_lock:
            self._bindings.pop(thread_id, None)

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
    cipher: AesGcmSessionCipher | None = None,
) -> SchemaValidatedCheckpointSaver:
    """Build the strict boundary over an in-memory or injected durable saver."""
    if backend is None:
        backend = InMemorySaver(serde=build_checkpoint_serializer())
    return SchemaValidatedCheckpointSaver(
        backend,
        synchronous_operations=synchronous_operations,
        cipher=cipher,
    )


def build_checkpoint_serializer() -> JsonPlusSerializer:
    """Build the allowlisted serializer shared by every checkpoint backend."""
    return JsonPlusSerializer(
        allowed_msgpack_modules=[*_CHECKPOINT_CHANNEL_DTOS, *_CHECKPOINT_NESTED_ENUMS]
    )
