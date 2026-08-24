"""Fail-closed semantic recognizer activation contracts."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from llm_fakes import FakeChatModel, RecordingResolver

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
    QualifiedSemanticRouterFactory,
    RoutingActivationError,
)
from agnostic_market.dtos.config import ProviderModel
from agnostic_market.llm.gateway import load_provider_credentials


def _report(
    registry: CapabilityRegistry,
    *,
    gate: str = "cutover",
    passed: bool | None = True,
    run_at: datetime | None = None,
) -> dict[str, object]:
    return {
        "schema_version": SEMANTIC_ROUTING_QUALIFICATION_SCHEMA_VERSION,
        "run_at": (run_at or datetime.now(tz=UTC)).isoformat(),
        "corpus_fingerprint": "reviewed-corpus",
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


def _factory(config_root: Path, report_path: Path) -> QualifiedSemanticRouterFactory:
    return QualifiedSemanticRouterFactory(
        qualification_path=report_path,
        selection=ProviderModel(provider="fake", model="qualified-router"),
        credentials=load_provider_credentials(config_root / "base" / "providers.yaml"),
        secrets=RecordingResolver(),
        structured_output_method="function_calling",
        timeout_seconds=2.0,
        input_max_chars=2048,
        max_report_age_days=30,
        expected_corpus_fingerprint="reviewed-corpus",
    )


def _write_report(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_diagnostic_report_cannot_activate_routing(config_root: Path, tmp_path: Path) -> None:
    registry = CapabilityRegistry(())
    path = tmp_path / "qualification.json"
    _write_report(path, _report(registry, gate="diagnostic", passed=None))

    with pytest.raises(RoutingActivationError, match="missing or invalid"):
        _factory(config_root, path)(registry)


def test_previous_report_schema_cannot_activate_routing(
    config_root: Path,
    tmp_path: Path,
) -> None:
    registry = CapabilityRegistry(())
    path = tmp_path / "qualification.json"
    payload = _report(registry)
    payload["schema_version"] = "6"
    _write_report(path, payload)

    with pytest.raises(RoutingActivationError, match="missing or invalid"):
        _factory(config_root, path)(registry)


def test_stale_report_cannot_activate_routing(config_root: Path, tmp_path: Path) -> None:
    registry = CapabilityRegistry(())
    path = tmp_path / "qualification.json"
    _write_report(path, _report(registry, run_at=datetime.now(tz=UTC) - timedelta(days=31)))

    with pytest.raises(RoutingActivationError, match="not current"):
        _factory(config_root, path)(registry)


def test_runtime_contract_mismatch_cannot_activate_routing(
    config_root: Path, tmp_path: Path
) -> None:
    registry = CapabilityRegistry(())
    path = tmp_path / "qualification.json"
    payload = _report(registry)
    payload["models"]["candidate"]["prompt_fingerprint"] = "stale"  # type: ignore[index]
    _write_report(path, payload)

    with pytest.raises(RoutingActivationError, match="prompt_fingerprint"):
        _factory(config_root, path)(registry)


def test_reasoning_effort_mismatch_cannot_activate_routing(
    config_root: Path, tmp_path: Path
) -> None:
    registry = CapabilityRegistry(())
    path = tmp_path / "qualification.json"
    payload = _report(registry)
    payload["models"]["candidate"]["reasoning_effort"] = "none"  # type: ignore[index]
    _write_report(path, payload)

    with pytest.raises(RoutingActivationError, match="reasoning_effort"):
        _factory(config_root, path)(registry)


def test_stale_corpus_cannot_activate_routing(config_root: Path, tmp_path: Path) -> None:
    registry = CapabilityRegistry(())
    path = tmp_path / "qualification.json"
    payload = _report(registry)
    payload["corpus_fingerprint"] = "stale-corpus"
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
    _write_report(path, _report(registry))

    class FakeGateway:
        def __init__(self, *_args: object) -> None:
            pass

        def chat_model(self, selection: ProviderModel) -> FakeChatModel:
            assert selection == ProviderModel(provider="fake", model="qualified-router")
            return FakeChatModel()

    monkeypatch.setattr(routing_activation, "LLMGateway", FakeGateway)

    recognizer = _factory(config_root, path)(registry)

    assert isinstance(recognizer, SemanticRouter)
