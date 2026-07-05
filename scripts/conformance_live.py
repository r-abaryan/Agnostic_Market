"""Phase-1 live conformance run — certify targets across real providers.

Run: uv run python scripts/conformance_live.py
Needs ANTHROPIC_API_KEY + OPENAI_API_KEY in .env (refs in config/base/providers.yaml).
Writes config/conformance/reports.json and prints certification warnings for the
merchant fixtures. ASCII-only output (Windows console).

Decision rule (explicit): a genuine check failure -> a chat-only verdict IS recorded
(that is the suite working; swap the target row in targets.yaml or accept the verdict).
A transport/auth failure -> NO verdict; fix and re-run (exit code 1).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from agnostic_market.config.registry import ConfigRegistry
from agnostic_market.llm.gateway import LLMGateway, load_provider_credentials
from agnostic_market.llm.providers import (
    ConformanceRegistry,
    ConformanceRunError,
    check_llm_certification,
    load_conformance_targets,
    run_conformance,
)
from agnostic_market.secrets.env_resolver import EnvSecretResolver

_CONFIG_ROOT = Path(__file__).resolve().parents[1] / "config"
_REPORTS_PATH = _CONFIG_ROOT / "conformance" / "reports.json"


async def _run() -> int:
    credentials = load_provider_credentials(_CONFIG_ROOT / "base" / "providers.yaml")
    targets_config = load_conformance_targets(_CONFIG_ROOT / "conformance" / "targets.yaml")
    gateway = LLMGateway(credentials, EnvSecretResolver())
    registry = ConformanceRegistry(
        _REPORTS_PATH, max_report_age_days=targets_config.max_report_age_days
    )

    run_errors: list[str] = []
    for target in targets_config.targets:
        label = f"{target.provider}:{target.model}"
        print(f"-> {label}")
        chat_model = gateway.chat_model(target, max_retries=targets_config.max_retries)
        try:
            report = await run_conformance(chat_model, provider=target.provider, model=target.model)
        except ConformanceRunError as exc:
            run_errors.append(label)
            print(f"   RUN ERROR (no verdict): {exc}")
            continue
        registry.record(report)
        for check in report.checks:
            status = "pass" if check.passed else f"FAIL - {check.detail}"
            print(f"   {check.name:<18} {status}")
        print(f"   verdict: {report.verdict}")

    registry.save()
    print(f"\nreports written: {_REPORTS_PATH}")

    # Config-time certification check (warn-only in Phase 1; Phase 4 gate makes it blocking).
    merchants = ConfigRegistry(_CONFIG_ROOT).load()
    warnings = [
        warning
        for merchant_id in sorted(merchants.merchant_ids)
        for warning in check_llm_certification(merchants.get(merchant_id).config, registry)
    ]
    for warning in warnings:
        print(f"WARNING: {warning}")
    if not warnings:
        print("merchant fixture LLM selections: all certified commerce-ready")

    if run_errors:
        print(f"\nRUN ERRORS (no verdict written) for: {', '.join(run_errors)} [RE-RUN NEEDED]")
        return 1
    print("\nPhase-1 live conformance complete. [PASS]")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_run()))
