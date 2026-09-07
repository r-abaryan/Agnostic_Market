"""Authenticated session-payload envelope contracts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agnostic_market.durability.encryption import (
    AesGcmSessionCipher,
    SessionEnvelope,
    SessionEnvelopeContext,
    SessionEnvelopeError,
)
from agnostic_market.durability.session_payload import SESSION_PAYLOAD_SCHEMA_VERSION

_KEY_V1 = bytes(range(32))
_KEY_V2 = bytes(reversed(range(32)))


def _context(**changes: object) -> SessionEnvelopeContext:
    values = {
        "tenant_id": "acme_store",
        "logical_session_id": "AD_session",
        "checkpoint_namespace": "cp_generation_7",
        "payload_schema_version": SESSION_PAYLOAD_SCHEMA_VERSION,
        "payload_purpose": "session_projection",
        "session_revision": 7,
    }
    values.update(changes)
    return SessionEnvelopeContext.model_validate(values)


def test_session_envelope_round_trip_hides_plaintext() -> None:
    plaintext = b'{"sentinel":"never-store-this-in-plaintext"}'
    cipher = AesGcmSessionCipher(active_key_version="key-v1", keys={"key-v1": _KEY_V1})

    envelope = cipher.encrypt(plaintext, _context())

    assert envelope.format == "aes_256_gcm_v1"
    assert envelope.key_version == "key-v1"
    assert envelope.payload_schema_version == SESSION_PAYLOAD_SCHEMA_VERSION
    assert plaintext not in envelope.ciphertext
    assert cipher.decrypt(envelope, _context()) == plaintext


@pytest.mark.parametrize(
    "field_name, replacement",
    (
        ("tenant_id", "other_store"),
        ("logical_session_id", "AD_other"),
        ("checkpoint_namespace", "cp_generation_8"),
        ("payload_schema_version", 1),
        ("session_revision", 8),
    ),
)
def test_session_envelope_authenticates_every_context_dimension(
    field_name: str,
    replacement: object,
) -> None:
    cipher = AesGcmSessionCipher(active_key_version="key-v1", keys={"key-v1": _KEY_V1})
    envelope = cipher.encrypt(b"session payload", _context())

    with pytest.raises(SessionEnvelopeError, match="authenticate"):
        cipher.decrypt(envelope, _context(**{field_name: replacement}))


def test_operation_result_context_requires_and_authenticates_operation_authority() -> None:
    cipher = AesGcmSessionCipher(active_key_version="key-v1", keys={"key-v1": _KEY_V1})
    context = _context(
        payload_purpose="operation_result",
        operation_id="operation-7",
        request_fingerprint="a" * 64,
    )
    envelope = cipher.encrypt(b"operation result", context)

    assert cipher.decrypt(envelope, context) == b"operation result"
    projection_envelope = cipher.encrypt(b"session projection", _context())
    with pytest.raises(SessionEnvelopeError, match="authenticate"):
        cipher.decrypt(projection_envelope, context)
    with pytest.raises(SessionEnvelopeError, match="authenticate"):
        cipher.decrypt(envelope, context.model_copy(update={"session_revision": 8}))
    with pytest.raises(SessionEnvelopeError, match="authenticate"):
        cipher.decrypt(envelope, context.model_copy(update={"operation_id": "operation-8"}))
    with pytest.raises(ValidationError, match="operation authority"):
        _context(payload_purpose="operation_result")


def test_session_envelope_rejects_ciphertext_or_nonce_tampering() -> None:
    cipher = AesGcmSessionCipher(active_key_version="key-v1", keys={"key-v1": _KEY_V1})
    envelope = cipher.encrypt(b"session payload", _context())
    tampered_ciphertext = bytes([envelope.ciphertext[0] ^ 1]) + envelope.ciphertext[1:]

    with pytest.raises(SessionEnvelopeError, match="authenticate"):
        cipher.decrypt(envelope.model_copy(update={"ciphertext": tampered_ciphertext}), _context())
    with pytest.raises(SessionEnvelopeError, match="authenticate"):
        cipher.decrypt(
            envelope.model_copy(
                update={"nonce": bytes([envelope.nonce[0] ^ 1]) + envelope.nonce[1:]}
            ),
            _context(),
        )


def test_session_envelope_key_rotation_reads_old_and_writes_active_version() -> None:
    first = AesGcmSessionCipher(active_key_version="key-v1", keys={"key-v1": _KEY_V1})
    old_envelope = first.encrypt(b"old payload", _context())
    rotated = AesGcmSessionCipher(
        active_key_version="key-v2",
        keys={"key-v1": _KEY_V1, "key-v2": _KEY_V2},
    )

    assert rotated.decrypt(old_envelope, _context()) == b"old payload"
    assert rotated.encrypt(b"new payload", _context()).key_version == "key-v2"


def test_current_registry_contexts_reject_frozen_version_1_ciphertext() -> None:
    legacy_envelope = SessionEnvelope(
        format="aes_256_gcm_v1",
        key_version="key-v1",
        payload_schema_version=1,
        nonce=bytes.fromhex("000102030405060708090a0b"),
        ciphertext=bytes.fromhex(
            "2b67b17aa69ce268e832e4e2de87581de2afeb5b911fa96ef2be2a9ab19e0cbe133868c79851"
        ),
    )
    cipher = AesGcmSessionCipher(active_key_version="key-v1", keys={"key-v1": _KEY_V1})

    contexts = (
        _context(),
        _context(
            payload_purpose="operation_result",
            operation_id="legacy-operation",
            request_fingerprint="f" * 64,
        ),
    )
    for context in contexts:
        with pytest.raises(SessionEnvelopeError, match="authenticate"):
            cipher.decrypt(legacy_envelope, context)


def test_session_envelope_fails_closed_for_unknown_or_invalid_keys() -> None:
    with pytest.raises(ValueError, match="32 bytes"):
        AesGcmSessionCipher(active_key_version="key-v1", keys={"key-v1": b"short"})
    with pytest.raises(ValueError, match="active key"):
        AesGcmSessionCipher(active_key_version="key-v2", keys={"key-v1": _KEY_V1})

    cipher = AesGcmSessionCipher(active_key_version="key-v1", keys={"key-v1": _KEY_V1})
    envelope = cipher.encrypt(b"session payload", _context())
    missing_key = AesGcmSessionCipher(active_key_version="key-v2", keys={"key-v2": _KEY_V2})

    with pytest.raises(SessionEnvelopeError, match="unavailable"):
        missing_key.decrypt(envelope, _context())

    with pytest.raises(ValidationError):
        SessionEnvelope.model_validate(
            {
                "format": "aes_256_gcm_v1",
                "key_version": "key-v1",
                "payload_schema_version": 1,
                "nonce": b"short",
                "ciphertext": b"too short",
            }
        )
