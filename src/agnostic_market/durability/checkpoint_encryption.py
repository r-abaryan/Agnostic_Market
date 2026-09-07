"""Authenticated encryption codec for LangGraph checkpoint payload surfaces."""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata, CheckpointTuple
from langgraph.checkpoint.serde.base import SerializerProtocol
from pydantic import BaseModel, ConfigDict, Field

from agnostic_market.dtos.state import CheckpointSchemaError
from agnostic_market.durability.encryption import (
    AesGcmSessionCipher,
    CheckpointEnvelopeContext,
    CheckpointPayloadSurface,
    SessionEnvelope,
)

CHECKPOINT_PAYLOAD_ENVELOPE_SCHEMA_VERSION = 1
_ENCRYPTED_PAYLOAD_KEY = "__agnostic_checkpoint_payload__"
_ENCRYPTED_CHECKPOINT_KEY = "__agnostic_checkpoint_header__"
_CHECKPOINT_HEADER_KEYS = frozenset({"checkpoint", "parent_checkpoint_id", "channel_value_keys"})
_ENVELOPE_KEYS = frozenset(
    {
        "format",
        "key_version",
        "payload_schema_version",
        "nonce",
        "ciphertext",
    }
)
_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True)


class PendingWriteManifestEntry(BaseModel):
    model_config = _STRICT

    task_id: str = Field(min_length=1)
    task_path: str
    index: int
    channel: str = Field(min_length=1)
    type_tag: str = Field(min_length=1)
    blob_digest: str = Field(pattern=r"^[0-9a-f]{64}$")


class PendingWriteManifest(BaseModel):
    model_config = _STRICT

    entries: tuple[PendingWriteManifestEntry, ...]


@dataclass(frozen=True, slots=True)
class CheckpointCipherScope:
    tenant_id: str
    logical_session_id: str
    checkpoint_generation_namespace: str
    langgraph_checkpoint_namespace: str
    deployment_id: str
    graph_contract: str


@dataclass(frozen=True, slots=True)
class EncryptedCheckpointCodec:
    serde: SerializerProtocol
    cipher: AesGcmSessionCipher

    @staticmethod
    def _payload_reference(*parts: object) -> str:
        encoded = json.dumps(
            [str(part) for part in parts],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _envelope_value(envelope: SessionEnvelope) -> dict[str, object]:
        return {
            _ENCRYPTED_PAYLOAD_KEY: {
                "format": envelope.format,
                "key_version": envelope.key_version,
                "payload_schema_version": envelope.payload_schema_version,
                "nonce": base64.b64encode(envelope.nonce).decode("ascii"),
                "ciphertext": base64.b64encode(envelope.ciphertext).decode("ascii"),
            }
        }

    @staticmethod
    def _parse_envelope(value: object) -> SessionEnvelope:
        if not isinstance(value, Mapping) or set(value) != {_ENCRYPTED_PAYLOAD_KEY}:
            raise CheckpointSchemaError("persisted checkpoint payload is not encrypted")
        fields = value[_ENCRYPTED_PAYLOAD_KEY]
        if not isinstance(fields, Mapping) or set(fields) != _ENVELOPE_KEYS:
            raise CheckpointSchemaError("persisted checkpoint envelope is malformed")
        try:
            nonce = base64.b64decode(fields["nonce"], validate=True)
            ciphertext = base64.b64decode(fields["ciphertext"], validate=True)
            return SessionEnvelope.model_validate(
                {
                    **fields,
                    "nonce": nonce,
                    "ciphertext": ciphertext,
                }
            )
        except (TypeError, ValueError) as exc:
            raise CheckpointSchemaError("persisted checkpoint envelope is malformed") from exc

    @staticmethod
    def _context(
        scope: CheckpointCipherScope,
        *,
        surface: CheckpointPayloadSurface,
        reference: str,
    ) -> CheckpointEnvelopeContext:
        return CheckpointEnvelopeContext(
            tenant_id=scope.tenant_id,
            logical_session_id=scope.logical_session_id,
            checkpoint_namespace=scope.checkpoint_generation_namespace,
            deployment_id=scope.deployment_id,
            graph_contract=scope.graph_contract,
            langgraph_checkpoint_namespace=scope.langgraph_checkpoint_namespace,
            payload_schema_version=CHECKPOINT_PAYLOAD_ENVELOPE_SCHEMA_VERSION,
            payload_surface=surface,
            payload_reference=reference,
        )

    @staticmethod
    def _checkpoint_id(config: RunnableConfig | None) -> str | None:
        if config is None:
            return None
        configurable = config.get("configurable")
        if not isinstance(configurable, Mapping):
            raise CheckpointSchemaError("checkpoint parent configuration is malformed")
        checkpoint_id = configurable.get("checkpoint_id")
        if checkpoint_id is None:
            return None
        if not isinstance(checkpoint_id, str) or not checkpoint_id:
            raise CheckpointSchemaError("checkpoint parent configuration is malformed")
        return checkpoint_id

    def _encrypt_value(
        self,
        value: object,
        scope: CheckpointCipherScope,
        *,
        surface: CheckpointPayloadSurface,
        reference: str,
    ) -> object:
        type_tag, blob = self.serde.dumps_typed(value)
        frame = type_tag.encode("utf-8") + b"\0" + blob
        envelope = self.cipher.encrypt(
            frame,
            self._context(scope, surface=surface, reference=reference),
        )
        return self._envelope_value(envelope)

    def _decrypt_value(
        self,
        value: object,
        scope: CheckpointCipherScope,
        *,
        surface: CheckpointPayloadSurface,
        reference: str,
    ) -> object:
        try:
            frame = self.cipher.decrypt(
                self._parse_envelope(value),
                self._context(scope, surface=surface, reference=reference),
            )
            type_bytes, blob = frame.split(b"\0", 1)
            if not type_bytes:
                raise ValueError("empty serializer type")
            return self.serde.loads_typed((type_bytes.decode("utf-8"), blob))
        except CheckpointSchemaError:
            raise
        except Exception as exc:
            raise CheckpointSchemaError(
                "persisted checkpoint payload could not be authenticated"
            ) from exc

    def encrypt_checkpoint(
        self,
        checkpoint: Checkpoint,
        scope: CheckpointCipherScope,
        *,
        parent_config: RunnableConfig | None,
        new_versions: Mapping[str, object],
    ) -> Checkpoint:
        checkpoint_id = checkpoint.get("id")
        checkpoint_version = checkpoint.get("v")
        channel_versions = checkpoint.get("channel_versions")
        channel_values = checkpoint.get("channel_values")
        if (
            not isinstance(checkpoint_id, str)
            or not checkpoint_id
            or not isinstance(channel_versions, Mapping)
            or not isinstance(channel_values, Mapping)
        ):
            raise CheckpointSchemaError("checkpoint encryption input is malformed")
        if type(checkpoint_version) is not int or checkpoint_version <= 0:
            raise CheckpointSchemaError("checkpoint version is malformed")
        if any(not isinstance(channel, str) for channel in channel_versions):
            raise CheckpointSchemaError("checkpoint channel versions are malformed")
        invalid_channels = {
            channel
            for channel in (*channel_values, *new_versions)
            if not isinstance(channel, str) or channel not in channel_versions
        }
        if invalid_channels:
            raise CheckpointSchemaError(
                "checkpoint channel versions do not cover the encrypted values"
            )
        if any(channel_versions[channel] != version for channel, version in new_versions.items()):
            raise CheckpointSchemaError("checkpoint new versions do not match the current index")
        header = {key: value for key, value in checkpoint.items() if key != "channel_values"}
        authenticated_header = {
            "checkpoint": header,
            "parent_checkpoint_id": self._checkpoint_id(parent_config),
            "channel_value_keys": sorted(channel_values),
        }
        encrypted_values = {
            channel: self._encrypt_value(
                value,
                scope,
                surface="channel_value",
                reference=self._payload_reference(channel, channel_versions[channel]),
            )
            for channel, value in channel_values.items()
            if channel in new_versions
        }
        return cast(
            "Checkpoint",
            {
                "v": checkpoint_version,
                "id": checkpoint_id,
                "channel_versions": channel_versions,
                "channel_values": encrypted_values,
                _ENCRYPTED_CHECKPOINT_KEY: self._encrypt_value(
                    authenticated_header,
                    scope,
                    surface="checkpoint",
                    reference=self._payload_reference(checkpoint_id),
                ),
            },
        )

    def decrypt_checkpoint(
        self,
        checkpoint: Checkpoint,
        scope: CheckpointCipherScope,
        *,
        parent_config: RunnableConfig | None,
    ) -> Checkpoint:
        checkpoint_id = checkpoint.get("id")
        channel_versions = checkpoint.get("channel_versions")
        encrypted_header = checkpoint.get(_ENCRYPTED_CHECKPOINT_KEY)
        if not isinstance(checkpoint_id, str) or not isinstance(channel_versions, Mapping):
            raise CheckpointSchemaError("persisted encrypted checkpoint index is malformed")
        authenticated_header = self._decrypt_value(
            encrypted_header,
            scope,
            surface="checkpoint",
            reference=self._payload_reference(checkpoint_id),
        )
        if (
            not isinstance(authenticated_header, Mapping)
            or set(authenticated_header) != _CHECKPOINT_HEADER_KEYS
            or authenticated_header.get("parent_checkpoint_id")
            != self._checkpoint_id(parent_config)
        ):
            raise CheckpointSchemaError(
                "persisted encrypted checkpoint parent does not authenticate"
            )
        header = authenticated_header.get("checkpoint")
        expected_channels = authenticated_header.get("channel_value_keys")
        if not isinstance(header, Mapping):
            raise CheckpointSchemaError("persisted encrypted checkpoint header is malformed")
        if (
            not isinstance(expected_channels, list)
            or any(not isinstance(channel, str) for channel in expected_channels)
            or expected_channels != sorted(set(expected_channels))
        ):
            raise CheckpointSchemaError("persisted encrypted checkpoint manifest is malformed")
        if (
            header.get("id") != checkpoint_id
            or header.get("v") != checkpoint.get("v")
            or header.get("channel_versions") != channel_versions
        ):
            raise CheckpointSchemaError(
                "persisted encrypted checkpoint index does not authenticate"
            )
        channel_values = checkpoint.get("channel_values")
        if not isinstance(channel_values, Mapping):
            raise CheckpointSchemaError("persisted encrypted checkpoint channels are malformed")
        if set(channel_values) != set(expected_channels):
            raise CheckpointSchemaError(
                "persisted encrypted checkpoint channel manifest does not authenticate"
            )
        decrypted_values: dict[str, object] = {}
        for channel, value in channel_values.items():
            if not isinstance(channel, str) or channel not in channel_versions:
                raise CheckpointSchemaError("persisted encrypted checkpoint channel is malformed")
            decrypted_values[channel] = self._decrypt_value(
                value,
                scope,
                surface="channel_value",
                reference=self._payload_reference(channel, channel_versions[channel]),
            )
        return cast("Checkpoint", {**header, "channel_values": decrypted_values})

    def encrypt_metadata(
        self,
        metadata: CheckpointMetadata,
        checkpoint_id: str,
        scope: CheckpointCipherScope,
    ) -> CheckpointMetadata:
        return cast(
            "CheckpointMetadata",
            self._encrypt_value(
                metadata,
                scope,
                surface="metadata",
                reference=self._payload_reference(checkpoint_id),
            ),
        )

    def decrypt_metadata(
        self,
        metadata: CheckpointMetadata,
        checkpoint_id: str,
        scope: CheckpointCipherScope,
    ) -> CheckpointMetadata:
        value = self._decrypt_value(
            metadata,
            scope,
            surface="metadata",
            reference=self._payload_reference(checkpoint_id),
        )
        if not isinstance(value, dict):
            raise CheckpointSchemaError("persisted encrypted checkpoint metadata is malformed")
        return cast("CheckpointMetadata", value)

    def encrypt_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        scope: CheckpointCipherScope,
    ) -> tuple[tuple[str, object], ...]:
        checkpoint_id = config["configurable"].get("checkpoint_id")
        if not isinstance(checkpoint_id, str):
            raise CheckpointSchemaError("pending writes require a checkpoint id")
        return tuple(
            (
                channel,
                self._encrypt_value(
                    value,
                    scope,
                    surface="pending_write",
                    reference=self._payload_reference(checkpoint_id, task_id, channel),
                ),
            )
            for channel, value in writes
        )

    def decrypt_tuple(
        self,
        saved: CheckpointTuple | None,
        scope: CheckpointCipherScope,
    ) -> CheckpointTuple | None:
        if saved is None:
            return None
        checkpoint = self.decrypt_checkpoint(
            saved.checkpoint,
            scope,
            parent_config=saved.parent_config,
        )
        checkpoint_id = checkpoint["id"]
        pending_writes = tuple(
            (
                task_id,
                channel,
                self._decrypt_value(
                    value,
                    scope,
                    surface="pending_write",
                    reference=self._payload_reference(checkpoint_id, task_id, channel),
                ),
            )
            for task_id, channel, value in saved.pending_writes or ()
        )
        return CheckpointTuple(
            saved.config,
            checkpoint,
            self.decrypt_metadata(saved.metadata, checkpoint_id, scope),
            saved.parent_config,
            pending_writes,
        )


def seal_pending_write_manifest(
    manifest: PendingWriteManifest,
    scope: CheckpointCipherScope,
    checkpoint_id: str,
    cipher: AesGcmSessionCipher,
) -> SessionEnvelope:
    if not checkpoint_id:
        raise CheckpointSchemaError("pending-write manifest requires a checkpoint id")
    return cipher.encrypt(
        manifest.model_dump_json().encode("utf-8"),
        EncryptedCheckpointCodec._context(
            scope,
            surface="pending_write_manifest",
            reference=EncryptedCheckpointCodec._payload_reference(checkpoint_id),
        ),
    )


def verify_pending_write_manifest(
    envelope: SessionEnvelope,
    observed: PendingWriteManifest,
    scope: CheckpointCipherScope,
    checkpoint_id: str,
    cipher: AesGcmSessionCipher,
) -> None:
    try:
        plaintext = cipher.decrypt(
            envelope,
            EncryptedCheckpointCodec._context(
                scope,
                surface="pending_write_manifest",
                reference=EncryptedCheckpointCodec._payload_reference(checkpoint_id),
            ),
        )
        expected = PendingWriteManifest.model_validate_json(plaintext)
    except Exception as exc:
        raise CheckpointSchemaError(
            "persisted pending-write manifest could not be authenticated"
        ) from exc
    if expected != observed:
        raise CheckpointSchemaError(
            "persisted pending-write collection does not match its manifest"
        )
