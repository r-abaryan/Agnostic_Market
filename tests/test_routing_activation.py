"""Fail-closed semantic recognizer activation contracts."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from llm_fakes import FakeChatModel, RecordingResolver
from pydantic import ValidationError

from agnostic_market.agents.capabilities import CapabilityRegistry
from agnostic_market.agents.routing import (
    CONTEXT_PROJECTOR_VERSION,
    ROUTE_SCHEMA_FINGERPRINT,
    ROUTER_PROMPT_FINGERPRINT,
    SemanticRouter,
    registry_fingerprint,
)
from agnostic_market.agents.routing_activation import (
    SEMANTIC_ROUTING_QUALIFICATION_SCHEMA_VERSION,
    ConfiguredSemanticRouterFactory,
    QualifiedSemanticRouterFactory,
    RoutingActivationError,
    SemanticRoutingReleaseEvidence,
    load_semantic_routing_release_evidence,
    semantic_routing_release_evidence_fingerprint,
    semantic_routing_runtime_contract,
    semantic_routing_runtime_contract_fingerprint,
)
from agnostic_market.dtos.config import ProviderModel
from agnostic_market.llm.gateway import load_provider_credentials


def _report(
    registry: CapabilityRegistry,
    *,
    corpus_role: str = "regression",
    corpus_fingerprint: str = "a" * 64,
    gate: str = "cutover",
    passed: bool | None = True,
    run_at: datetime | None = None,
) -> dict[str, object]:
    return {
        "schema_version": SEMANTIC_ROUTING_QUALIFICATION_SCHEMA_VERSION,
        "qualification_run_id": "qualification-run",
        "corpus_role": corpus_role,
        "run_at": (run_at or datetime.now(tz=UTC)).isoformat(),
        "corpus_fingerprint": corpus_fingerprint,
        "gate": {"mode": gate, "passed": passed, "failures": []},
        "projection": {"exact": True},
        "models": {
            "candidate": {
                "provider": "fake",
                "model": "qualified-router",
                "reasoning_effort": None,
                "structured_output_method": "function_calling",
                "route_schema_fingerprint": ROUTE_SCHEMA_FINGERPRINT,
                "prompt_fingerprint": ROUTER_PROMPT_FINGERPRINT,
                "registry_fingerprint": registry_fingerprint(registry),
                "input_max_chars": 2048,
                "timeout_seconds": 2.0,
                "projector_version": CONTEXT_PROJECTOR_VERSION,
            }
        },
    }


def _release_evidence(
    registry: CapabilityRegistry,
    *,
    gate: str = "cutover",
    passed: bool | None = True,
    run_at: datetime | None = None,
) -> dict[str, object]:
    report_time = run_at or datetime.now(tz=UTC)
    return {
        "schema_version": 1,
        "packaged_at": datetime.now(tz=UTC).isoformat(),
        "holdout_id": "routing-readiness-v1",
        "holdout_steward_id": "routing-steward",
        "holdout_frozen_at": (report_time - timedelta(seconds=1)).isoformat(),
        "holdout_lineage_reference": "lineage:routing-readiness-v1",
        "holdout_generation_recipe_fingerprint": "c" * 64,
        "holdout_source_disjoint_from_corpus_fingerprint": "a" * 64,
        "regression_report_sha256": "d" * 64,
        "readiness_report_sha256": "e" * 64,
        "regression": _report(
            registry,
            gate=gate,
            passed=passed,
            run_at=report_time,
        ),
        "readiness_holdout": _report(
            registry,
            corpus_role="readiness_holdout",
            corpus_fingerprint="b" * 64,
            gate=gate,
            passed=passed,
            run_at=report_time,
        ),
    }


def _factory(
    config_root: Path,
    report_path: Path,
    *,
    expected_evidence_fingerprint: str | None = None,
) -> QualifiedSemanticRouterFactory:
    if expected_evidence_fingerprint is None:
        evidence = load_semantic_routing_release_evidence(report_path)
        expected_evidence_fingerprint = semantic_routing_release_evidence_fingerprint(evidence)
    return QualifiedSemanticRouterFactory(
        qualification_path=report_path,
        selection=ProviderModel(provider="fake", model="qualified-router"),
        credentials=load_provider_credentials(config_root / "base" / "providers.yaml"),
        secrets=RecordingResolver(),
        structured_output_method="function_calling",
        timeout_seconds=2.0,
        input_max_chars=2048,
        max_report_age_days=30,
        expected_corpus_fingerprint="a" * 64,
        expected_qualification_evidence_fingerprint=expected_evidence_fingerprint,
    )


def _write_report(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_runtime_contract_fingerprint_binds_every_semantic_router_input() -> None:
    contract = semantic_routing_runtime_contract(
        CapabilityRegistry(()),
        selection=ProviderModel(
            provider="fake",
            model="qualified-router",
            reasoning_effort=None,
        ),
        structured_output_method="function_calling",
        timeout_seconds=2.0,
        input_max_chars=2048,
        corpus_fingerprint="a" * 64,
        qualification_evidence_fingerprint="f" * 64,
    )
    baseline = semantic_routing_runtime_contract_fingerprint(contract)

    for field, value in {
        "model": "changed-router",
        "prompt_fingerprint": "changed-prompt",
        "registry_fingerprint": "changed-registry",
        "timeout_seconds": 3.0,
        "corpus_fingerprint": "b" * 64,
        "qualification_evidence_fingerprint": "e" * 64,
    }.items():
        changed = contract.model_copy(update={field: value})
        assert semantic_routing_runtime_contract_fingerprint(changed) != baseline


def test_diagnostic_report_cannot_activate_routing(config_root: Path, tmp_path: Path) -> None:
    registry = CapabilityRegistry(())
    path = tmp_path / "qualification.json"
    _write_report(path, _release_evidence(registry, gate="diagnostic", passed=None))

    with pytest.raises(RoutingActivationError, match="missing or invalid"):
        _factory(config_root, path, expected_evidence_fingerprint="f" * 64)(registry)


def test_failed_cutover_release_cannot_activate_routing(
    config_root: Path,
    tmp_path: Path,
) -> None:
    registry = CapabilityRegistry(())
    path = tmp_path / "qualification.json"
    payload = _release_evidence(registry, passed=False)
    payload["regression"]["gate"]["failures"] = [  # type: ignore[index]
        "candidate produced an unsafe executable misroute"
    ]
    _write_report(path, payload)

    with pytest.raises(RoutingActivationError, match="missing or invalid"):
        _factory(config_root, path, expected_evidence_fingerprint="f" * 64)(registry)


def test_inexact_projection_reports_its_actual_activation_failure(
    config_root: Path,
    tmp_path: Path,
) -> None:
    registry = CapabilityRegistry(())
    path = tmp_path / "qualification.json"
    payload = _release_evidence(registry)
    payload["regression"]["projection"]["exact"] = False  # type: ignore[index]
    _write_report(path, payload)

    with pytest.raises(RoutingActivationError, match="missing or invalid"):
        _factory(config_root, path, expected_evidence_fingerprint="f" * 64)(registry)


def test_previous_report_schema_cannot_activate_routing(
    config_root: Path,
    tmp_path: Path,
) -> None:
    registry = CapabilityRegistry(())
    path = tmp_path / "qualification.json"
    payload = _release_evidence(registry)
    payload["regression"]["schema_version"] = "9"  # type: ignore[index]
    _write_report(path, payload)

    with pytest.raises(RoutingActivationError, match="missing or invalid"):
        _factory(config_root, path, expected_evidence_fingerprint="f" * 64)(registry)


def test_stale_report_cannot_activate_routing(config_root: Path, tmp_path: Path) -> None:
    registry = CapabilityRegistry(())
    path = tmp_path / "qualification.json"
    _write_report(
        path,
        _release_evidence(registry, run_at=datetime.now(tz=UTC) - timedelta(days=31)),
    )

    with pytest.raises(RoutingActivationError, match="not current"):
        _factory(config_root, path)(registry)


def test_runtime_contract_mismatch_cannot_activate_routing(
    config_root: Path, tmp_path: Path
) -> None:
    registry = CapabilityRegistry(())
    path = tmp_path / "qualification.json"
    payload = _release_evidence(registry)
    payload["regression"]["models"]["candidate"]["prompt_fingerprint"] = "stale"  # type: ignore[index]
    payload["readiness_holdout"]["models"]["candidate"]["prompt_fingerprint"] = "stale"  # type: ignore[index]
    _write_report(path, payload)

    with pytest.raises(RoutingActivationError, match="prompt_fingerprint"):
        _factory(config_root, path)(registry)


def test_reasoning_effort_mismatch_cannot_activate_routing(
    config_root: Path, tmp_path: Path
) -> None:
    registry = CapabilityRegistry(())
    path = tmp_path / "qualification.json"
    payload = _release_evidence(registry)
    payload["regression"]["models"]["candidate"]["reasoning_effort"] = "none"  # type: ignore[index]
    payload["readiness_holdout"]["models"]["candidate"]["reasoning_effort"] = "none"  # type: ignore[index]
    _write_report(path, payload)

    with pytest.raises(RoutingActivationError, match="reasoning_effort"):
        _factory(config_root, path)(registry)


def test_stale_corpus_cannot_activate_routing(config_root: Path, tmp_path: Path) -> None:
    registry = CapabilityRegistry(())
    path = tmp_path / "qualification.json"
    payload = _release_evidence(registry)
    payload["regression"]["corpus_fingerprint"] = "f" * 64  # type: ignore[index]
    payload["holdout_source_disjoint_from_corpus_fingerprint"] = "f" * 64
    _write_report(path, payload)

    with pytest.raises(RoutingActivationError, match="corpus_fingerprint"):
        _factory(config_root, path)(registry)


def test_exact_current_cutover_report_activates_configured_router(
    config_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agnostic_market.agents import routing_activation

    registry = CapabilityRegistry(())
    path = tmp_path / "qualification.json"
    _write_report(path, _release_evidence(registry))

    class FakeGateway:
        def __init__(self, *_args: object) -> None:
            pass

        def chat_model(self, selection: ProviderModel) -> FakeChatModel:
            assert selection == ProviderModel(provider="fake", model="qualified-router")
            return FakeChatModel()

    monkeypatch.setattr(routing_activation, "LLMGateway", FakeGateway)

    recognizer = _factory(config_root, path)(registry)

    assert isinstance(recognizer, SemanticRouter)


def test_release_evidence_requires_one_run_and_two_distinct_passing_corpora() -> None:
    registry = CapabilityRegistry(())
    payload = _release_evidence(registry)
    payload["readiness_holdout"]["qualification_run_id"] = "different-run"  # type: ignore[index]
    with pytest.raises(ValidationError, match="one qualification run"):
        SemanticRoutingReleaseEvidence.model_validate_json(json.dumps(payload))

    payload = _release_evidence(registry)
    payload["readiness_holdout"]["corpus_fingerprint"] = "a" * 64  # type: ignore[index]
    with pytest.raises(ValidationError, match="corpora must be distinct"):
        SemanticRoutingReleaseEvidence.model_validate_json(json.dumps(payload))


def test_prepared_factory_rejects_release_evidence_replaced_on_disk(
    config_root: Path,
    tmp_path: Path,
) -> None:
    registry = CapabilityRegistry(())
    path = tmp_path / "qualification.json"
    payload = _release_evidence(registry)
    _write_report(path, payload)
    factory = _factory(config_root, path)

    payload["regression_report_sha256"] = "f" * 64
    _write_report(path, payload)

    with pytest.raises(RoutingActivationError, match="changed after preparation"):
        factory(registry)


def test_configured_router_factory_does_not_claim_or_require_qualification(
    config_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agnostic_market.agents import routing_activation

    registry = CapabilityRegistry(())

    class FakeGateway:
        def __init__(self, *_args: object) -> None:
            pass

        def chat_model(self, selection: ProviderModel) -> FakeChatModel:
            assert selection == ProviderModel(provider="fake", model="development-router")
            return FakeChatModel()

    monkeypatch.setattr(routing_activation, "LLMGateway", FakeGateway)
    factory = ConfiguredSemanticRouterFactory(
        selection=ProviderModel(provider="fake", model="development-router"),
        credentials=load_provider_credentials(config_root / "base" / "providers.yaml"),
        secrets=RecordingResolver(),
        structured_output_method="function_calling",
        timeout_seconds=2.0,
        input_max_chars=2048,
    )

    assert isinstance(factory(registry), SemanticRouter)
