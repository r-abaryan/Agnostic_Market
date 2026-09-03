"""Authenticated encryption for durable platform-session payloads."""

from __future__ import annotations

import json
import secrets
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from agnostic_market.dtos.platform import ConfigIdentifier
from agnostic_market.dtos.session import AuthorityIdentifier

_ENVELOPE_FORMAT = "aes_256_gcm_v1"
_NONCE_BYTES = 12
_KEY_BYTES = 32
_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True)
_KEY_VERSION = TypeAdapter(ConfigIdentifier)


class SessionEnvelopeError(RuntimeError):
    """A session envelope cannot be safely opened."""


class SessionEnvelopeContext(BaseModel):
    model_config = _STRICT

    tenant_id: AuthorityIdentifier
    logical_session_id: AuthorityIdentifier
    checkpoint_namespace: AuthorityIdentifier
    payload_schema_version: int = Field(ge=1)


class SessionEnvelope(BaseModel):
    model_config = _STRICT

    format: Literal["aes_256_gcm_v1"]
    key_version: ConfigIdentifier
    payload_schema_version: int = Field(ge=1)
    nonce: bytes = Field(min_length=_NONCE_BYTES, max_length=_NONCE_BYTES)
    ciphertext: bytes = Field(min_length=16)


def _associated_data(
    context: SessionEnvelopeContext,
    key_version: str,
    payload_schema_version: int,
) -> bytes:
    return json.dumps(
        {
            "checkpoint_namespace": context.checkpoint_namespace,
            "envelope_format": _ENVELOPE_FORMAT,
            "key_version": key_version,
            "logical_session_id": context.logical_session_id,
            "payload_schema_version": payload_schema_version,
            "tenant_id": context.tenant_id,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class AesGcmSessionCipher:
    """Encrypt with the active key and decrypt any retained key version."""

    active_key_version: str
    keys: Mapping[str, bytes] = field(repr=False)

    def __post_init__(self) -> None:
        active_key_version = _KEY_VERSION.validate_python(self.active_key_version)
        keys = dict(self.keys)
        if active_key_version not in keys:
            raise ValueError("active key version is unavailable")
        for key_version, key in keys.items():
            _KEY_VERSION.validate_python(key_version)
            if not isinstance(key, bytes) or len(key) != _KEY_BYTES:
                raise ValueError("AES-256-GCM keys must contain exactly 32 bytes")
        object.__setattr__(self, "active_key_version", active_key_version)
        object.__setattr__(self, "keys", MappingProxyType(keys))

    def encrypt(
        self,
        plaintext: bytes,
        context: SessionEnvelopeContext,
    ) -> SessionEnvelope:
        if not isinstance(plaintext, bytes) or not plaintext:
            raise ValueError("session plaintext must be non-empty bytes")
        nonce = secrets.token_bytes(_NONCE_BYTES)
        key = self.keys[self.active_key_version]
        ciphertext = AESGCM(key).encrypt(
            nonce,
            plaintext,
            _associated_data(
                context,
                self.active_key_version,
                context.payload_schema_version,
            ),
        )
        return SessionEnvelope(
            format=_ENVELOPE_FORMAT,
            key_version=self.active_key_version,
            payload_schema_version=context.payload_schema_version,
            nonce=nonce,
            ciphertext=ciphertext,
        )

    def decrypt(
        self,
        envelope: SessionEnvelope,
        context: SessionEnvelopeContext,
    ) -> bytes:
        key = self.keys.get(envelope.key_version)
        if key is None:
            raise SessionEnvelopeError("session envelope key version is unavailable")
        if envelope.payload_schema_version != context.payload_schema_version:
            raise SessionEnvelopeError("session envelope could not be authenticated")
        try:
            return AESGCM(key).decrypt(
                envelope.nonce,
                envelope.ciphertext,
                _associated_data(
                    context,
                    envelope.key_version,
                    envelope.payload_schema_version,
                ),
            )
        except InvalidTag as exc:
            raise SessionEnvelopeError("session envelope could not be authenticated") from exc
