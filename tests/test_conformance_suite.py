"""Conformance suite classification + fail-closed registry/gate (zero network)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from llm_fakes import BROKEN_QUOTE_ARGS, FakeChatModel
from pydantic import ValidationError

from agnostic_market.config.loader import ConfigError
from agnostic_market.dtos.config import MerchantConfig, ProviderModel
from agnostic_market.dtos.llm import ConformanceCheck, ConformanceReport
from agnostic_market.llm.providers import (
    SUITE_VERSION,
    ChatOnlyModelError,
    ConformanceRegistry,
    ConformanceRunError,
    check_llm_certification,
    load_conformance_targets,
    run_conformance,
)

_MAX_AGE_DAYS = 30


def _failed_checks(report: ConformanceReport) -> dict[str, str]:
    return {c.name: c.detail for c in report.checks if not c.passed}


# --- suite classification -------------------------------------------------------------


async def test_conformant_model_is_commerce_ready() -> None:
    report = await run_conformance(FakeChatModel(), provider="fake", model="good")
    assert report.verdict == "commerce-ready", _failed_checks(report)
    assert report.suite_version == SUITE_VERSION
    assert [c.name for c in report.checks] == ["tool_call", "structured_output", "streaming"]


async def test_no_tool_calls_flagged_chat_only() -> None:
    report = await run_conformance(
        FakeChatModel(emit_tool_calls=False), provider="fake", model="no-tools"
    )
    assert report.verdict == "chat-only"
    assert "no tool_calls" in _failed_checks(report)["tool_call"]


async def test_wrong_tool_selection_flagged_chat_only() -> None:
    report = await run_conformance(
        FakeChatModel(pick_wrong_tool=True), provider="fake", model="wrong-tool"
    )
    failed = _failed_checks(report)
    assert report.verdict == "chat-only"
    assert "wrong tool" in failed["tool_call"]


async def test_invalid_structured_output_flagged_chat_only() -> None:
    report = await run_conformance(
        FakeChatModel(canned_args=BROKEN_QUOTE_ARGS), provider="fake", model="bad-schema"
    )
    failed = _failed_checks(report)
    assert report.verdict == "chat-only"
    assert "tool_call" not in failed  # tool calling itself still works
    assert "structured_output" in failed


async def test_non_streaming_flagged_chat_only() -> None:
    report = await run_conformance(
        FakeChatModel(stream_chunks=1), provider="fake", model="no-stream"
    )
    failed = _failed_checks(report)
    assert report.verdict == "chat-only"
    assert list(failed) == ["streaming"]


async def test_transport_error_yields_no_verdict() -> None:
    with pytest.raises(ConformanceRunError, match="no verdict"):
        await run_conformance(FakeChatModel(raise_transport=True), provider="fake", model="flaky")


# --- registry: persistence + three-way fail-closed gate --------------------------------


def _report(
    *,
    provider: str = "fake",
    model: str = "good",
    verdict: str = "commerce-ready",
    suite_version: str = SUITE_VERSION,
    age_days: int = 0,
) -> ConformanceReport:
    return ConformanceReport(
        provider=provider,
        model=model,
        suite_version=suite_version,
        run_at=datetime.now(tz=UTC) - timedelta(days=age_days),
        checks=[ConformanceCheck(name="tool_call", passed=verdict == "commerce-ready", detail="")],
        verdict=verdict,  # type: ignore[arg-type]
    )


def _registry(tmp_path: Path) -> ConformanceRegistry:
    return ConformanceRegistry(tmp_path / "reports.json", max_report_age_days=_MAX_AGE_DAYS)


def test_registry_roundtrip(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    registry.record(_report())
    registry.save()
    reloaded = _registry(tmp_path)
    assert reloaded.is_commerce_ready("fake", "good")


def test_unknown_model_fails_closed(tmp_path: Path) -> None:
    assert not _registry(tmp_path).is_commerce_ready("fake", "never-certified")


def test_expired_report_fails_closed(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    registry.record(_report(age_days=_MAX_AGE_DAYS + 1))
    assert not registry.is_commerce_ready("fake", "good")


def test_suite_version_mismatch_fails_closed(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    registry.record(_report(suite_version="0"))
    assert not registry.is_commerce_ready("fake", "good")


def test_chat_only_verdict_fails_closed(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    registry.record(_report(verdict="chat-only"))
    assert not registry.is_commerce_ready("fake", "good")


def test_corrupt_reports_file_fails_loudly(tmp_path: Path) -> None:
    (tmp_path / "reports.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigError, match="corrupt or has an invalid shape"):
        _registry(tmp_path)


def test_wrong_shape_reports_file_fails_loudly(tmp_path: Path) -> None:
    (tmp_path / "reports.json").write_text('{"fake:good": {"provider": "fake"}}', encoding="utf-8")
    with pytest.raises(ConfigError, match="corrupt or has an invalid shape"):
        _registry(tmp_path)


def test_empty_targets_rejected_at_load(tmp_path: Path) -> None:
    targets = tmp_path / "targets.yaml"
    targets.write_text("max_report_age_days: 30\nmax_retries: 3\ntargets: []\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_conformance_targets(targets)


def test_require_commerce_ready_gate(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    registry.record(_report())
    registry.require_commerce_ready(ProviderModel(provider="fake", model="good"))  # no raise
    with pytest.raises(ChatOnlyModelError, match="not certified"):
        registry.require_commerce_ready(ProviderModel(provider="fake", model="uncertified"))


# --- config-time warn hook --------------------------------------------------------------


def _merchant_config(provider: str, model: str) -> MerchantConfig:
    return MerchantConfig.model_validate(
        {
            "schema_version": "0.2",
            "merchant_id": "m1",
            "extends_template": "fashion",
            "display_name": "M One",
            "locale": "en-US",
            "llm": {
                "routing": {"provider": provider, "model": model},
                "reasoning": {"provider": provider, "model": model},
            },
            "voice": {
                "stt": {"provider": "deepgram", "model": "nova-3"},
                "tts": {"provider": "cartesia", "voice_id": "v1"},
            },
            "telephony": {"provider": "telnyx", "inbound_number": "+15550000000"},
            "policies": {
                "max_order_value_usd": 1500,
                "refunds": {"auto_approve_under_usd": 50, "require_human_above_usd": 200},
                "allow_ai_merchant_handoff": True,
            },
            "prompts": {"persona_ref": "prompt://m1/persona@sha256-abc"},
            "integration": {
                "order_sor": {"type": "api", "ref": "vault://m1/order", "idempotency": "supported"},
                "catalog": {"source": "ingested", "freshness_sla_min": 15},
            },
            "isolation": {"tier": "shared"},
            "vector_namespace": "m1",
            "secrets_ref": "vault://m1",
        }
    )


def test_check_llm_certification_warns_on_uncertified(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    warnings = check_llm_certification(_merchant_config("fake", "uncertified"), registry)
    assert len(warnings) == 2  # routing + reasoning
    assert all("not certified commerce-ready" in w for w in warnings)


def test_check_llm_certification_silent_when_certified(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    registry.record(_report())
    assert check_llm_certification(_merchant_config("fake", "good"), registry) == []
