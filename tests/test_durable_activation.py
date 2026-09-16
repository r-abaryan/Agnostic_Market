"""Fail-closed durable runtime activation-manifest contracts."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from agnostic_market.durability.activation import (
    ActivationEvidenceIdentity,
    ActivationEvidenceKind,
    DurableRuntimeActivationManifest,
    DurableRuntimeActivationManifestError,
    activation_manifest_fingerprint,
    load_activation_manifest,
)

_ACTIVATED_AT = datetime(2026, 9, 12, 12, 0, tzinfo=UTC)


def _evidence(kind: ActivationEvidenceKind, value: str) -> ActivationEvidenceIdentity:
    return ActivationEvidenceIdentity(
        kind=kind,
        schema_version="1",
        canonical_fingerprint=value * 64,
    )


def _manifest() -> DurableRuntimeActivationManifest:
    return DurableRuntimeActivationManifest(
        schema_version=1,
        activated_at=_ACTIVATED_AT,
        build_artifact_digest=f"sha256:{'a' * 64}",
        deployment_id="deployment-a",
        runtime_contract_fingerprint="b" * 64,
        tenant_inventory_revision=3,
        tenant_inventory_fingerprint="c" * 64,
        transport_posture="terminate_call",
        controlled_crash_evidence=_evidence(ActivationEvidenceKind.CONTROLLED_CRASH, "d"),
        voice_latency_evidence=_evidence(ActivationEvidenceKind.VOICE_LATENCY, "e"),
        live_transport_evidence=_evidence(ActivationEvidenceKind.LIVE_TRANSPORT, "f"),
        operational_reaper_evidence=_evidence(ActivationEvidenceKind.OPERATIONAL_REAPER, "0"),
    )


def test_manifest_requires_every_bound_evidence_family_and_canonical_identity() -> None:
    manifest = _manifest()

    assert manifest.transport_posture == "terminate_call"
    assert activation_manifest_fingerprint(manifest) == activation_manifest_fingerprint(manifest)
    assert set(DurableRuntimeActivationManifest.model_fields) == {
        "schema_version",
        "activated_at",
        "build_artifact_digest",
        "deployment_id",
        "runtime_contract_fingerprint",
        "tenant_inventory_revision",
        "tenant_inventory_fingerprint",
        "transport_posture",
        "controlled_crash_evidence",
        "voice_latency_evidence",
        "live_transport_evidence",
        "operational_reaper_evidence",
    }

    for field in (
        "controlled_crash_evidence",
        "voice_latency_evidence",
        "live_transport_evidence",
        "operational_reaper_evidence",
    ):
        payload = manifest.model_dump()
        payload.pop(field)
        with pytest.raises(ValidationError, match=field):
            DurableRuntimeActivationManifest.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("build_artifact_digest", f"sha256:{'1' * 64}"),
        ("deployment_id", "deployment-b"),
        ("runtime_contract_fingerprint", "1" * 64),
        ("tenant_inventory_revision", 4),
        ("tenant_inventory_fingerprint", "2" * 64),
        (
            "controlled_crash_evidence",
            _evidence(ActivationEvidenceKind.CONTROLLED_CRASH, "3"),
        ),
        ("voice_latency_evidence", _evidence(ActivationEvidenceKind.VOICE_LATENCY, "4")),
        ("live_transport_evidence", _evidence(ActivationEvidenceKind.LIVE_TRANSPORT, "5")),
        (
            "operational_reaper_evidence",
            _evidence(ActivationEvidenceKind.OPERATIONAL_REAPER, "6"),
        ),
    ),
)
def test_manifest_fingerprint_binds_every_release_authority(
    field: str,
    replacement: object,
) -> None:
    manifest = _manifest()
    changed = manifest.model_copy(update={field: replacement})

    assert activation_manifest_fingerprint(changed) != activation_manifest_fingerprint(manifest)


def test_manifest_rejects_non_immutable_build_labels_and_future_transport_postures() -> None:
    payload = _manifest().model_dump()
    payload["build_artifact_digest"] = "release-candidate"
    with pytest.raises(ValidationError, match="build_artifact_digest"):
        DurableRuntimeActivationManifest.model_validate(payload)

    payload = _manifest().model_dump()
    payload["transport_posture"] = "takeover"
    with pytest.raises(ValidationError, match="transport_posture"):
        DurableRuntimeActivationManifest.model_validate(payload)


def test_manifest_rejects_mislabelled_evidence_and_naive_time() -> None:
    payload = _manifest().model_dump()
    payload["live_transport_evidence"] = _evidence(
        ActivationEvidenceKind.OPERATIONAL_REAPER,
        "f",
    )
    with pytest.raises(ValidationError, match="live transport evidence identity"):
        DurableRuntimeActivationManifest.model_validate(payload)

    payload = _manifest().model_dump()
    payload["activated_at"] = _ACTIVATED_AT.replace(tzinfo=None)
    with pytest.raises(ValidationError, match="timezone-aware"):
        DurableRuntimeActivationManifest.model_validate(payload)


def test_manifest_loader_rejects_missing_invalid_and_extra_content(tmp_path: Path) -> None:
    path = tmp_path / "activation.json"
    with pytest.raises(DurableRuntimeActivationManifestError, match="missing or invalid"):
        load_activation_manifest(path)

    path.write_text('{"schema_version": 1}', encoding="utf-8")
    with pytest.raises(DurableRuntimeActivationManifestError, match="missing or invalid"):
        load_activation_manifest(path)

    payload = _manifest().model_dump_json()
    path.write_text(payload[:-1] + ',"operator_label":"trusted"}', encoding="utf-8")
    with pytest.raises(DurableRuntimeActivationManifestError, match="missing or invalid"):
        load_activation_manifest(path)

    path.write_text(_manifest().model_dump_json(), encoding="utf-8")
    assert load_activation_manifest(path) == _manifest()
