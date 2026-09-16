"""Closed release identity for durable runtime activation."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from agnostic_market.dtos.platform import ConfigIdentifier

_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_OCI_SHA256_PATTERN = r"^sha256:[0-9a-f]{64}$"


class ActivationEvidenceKind(StrEnum):
    CONTROLLED_CRASH = "controlled_crash"
    VOICE_LATENCY = "voice_latency"
    LIVE_TRANSPORT = "live_transport"
    OPERATIONAL_REAPER = "operational_reaper"


class ActivationEvidenceIdentity(BaseModel):
    """Canonical identity of evidence already accepted by its owning validator."""

    model_config = _STRICT

    kind: ActivationEvidenceKind
    schema_version: ConfigIdentifier
    canonical_fingerprint: str = Field(pattern=_SHA256_PATTERN)


class DurableRuntimeActivationManifest(BaseModel):
    """Complete release decision; an individual evidence report is never sufficient."""

    model_config = _STRICT

    schema_version: Literal[1]
    activated_at: datetime
    build_artifact_digest: str = Field(pattern=_OCI_SHA256_PATTERN)
    deployment_id: ConfigIdentifier
    runtime_contract_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    tenant_inventory_revision: int = Field(ge=1)
    tenant_inventory_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    transport_posture: Literal["terminate_call"]
    controlled_crash_evidence: ActivationEvidenceIdentity
    voice_latency_evidence: ActivationEvidenceIdentity
    live_transport_evidence: ActivationEvidenceIdentity
    operational_reaper_evidence: ActivationEvidenceIdentity

    @model_validator(mode="after")
    def validate_release_binding(self) -> Self:
        if self.activated_at.tzinfo is None:
            raise ValueError("activation timestamp must be timezone-aware")
        expected_kinds = {
            "controlled crash": (
                self.controlled_crash_evidence,
                ActivationEvidenceKind.CONTROLLED_CRASH,
            ),
            "voice latency": (
                self.voice_latency_evidence,
                ActivationEvidenceKind.VOICE_LATENCY,
            ),
            "live transport": (
                self.live_transport_evidence,
                ActivationEvidenceKind.LIVE_TRANSPORT,
            ),
            "operational reaper": (
                self.operational_reaper_evidence,
                ActivationEvidenceKind.OPERATIONAL_REAPER,
            ),
        }
        for label, (identity, expected_kind) in expected_kinds.items():
            if identity.kind is not expected_kind:
                raise ValueError(f"{label} evidence identity has the wrong kind")
        return self


class DurableRuntimeActivationManifestError(RuntimeError):
    """The durable runtime activation manifest is missing or invalid."""


def activation_manifest_fingerprint(manifest: DurableRuntimeActivationManifest) -> str:
    """Derive the canonical identity of a fully validated activation decision."""
    try:
        validated = DurableRuntimeActivationManifest.model_validate_json(manifest.model_dump_json())
    except ValidationError as exc:
        raise DurableRuntimeActivationManifestError("activation manifest is invalid") from exc
    canonical = json.dumps(
        validated.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def load_activation_manifest(path: Path) -> DurableRuntimeActivationManifest:
    """Load one complete activation decision without trusting its path or filename."""
    try:
        return DurableRuntimeActivationManifest.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except (OSError, ValidationError) as exc:
        raise DurableRuntimeActivationManifestError(
            "durable runtime activation manifest is missing or invalid"
        ) from exc
