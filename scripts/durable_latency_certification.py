"""Run one pre-registered deployment-shaped durable latency certification."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import selectors
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from agnostic_market.durability.latency import (
    LatencyCertificationOutcome,
    LatencyEnvironment,
    deployment_runtime_contract_fingerprint,
    load_latency_journey_corpus,
    load_latency_methodology,
    run_latency_certification,
    write_latency_certification_run,
)
from agnostic_market.durability.latency_probe import DeploymentLatencyProbeFactory
from agnostic_market.durability.platform_runtime import load_platform_runtime_config
from agnostic_market.secrets.env_resolver import EnvSecretResolver

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_ROOT = _REPOSITORY_ROOT / "config"
_DEFAULT_JOURNEYS = _CONFIG_ROOT / "eval" / "durable_latency_journeys.yaml"

logger = logging.getLogger("durable_latency_certification")


def _configured_path(variable: str) -> Path | None:
    value = os.environ.get(variable, "").strip()
    return Path(value) if value else None


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--platform-config",
        type=Path,
        default=_configured_path("VOICE_AGENT_PLATFORM_CONFIG"),
        required=_configured_path("VOICE_AGENT_PLATFORM_CONFIG") is None,
        help="deployment-owned PlatformRuntimeConfig YAML",
    )
    parser.add_argument(
        "--methodology",
        type=Path,
        default=_configured_path("VOICE_AGENT_LATENCY_METHODOLOGY"),
        required=_configured_path("VOICE_AGENT_LATENCY_METHODOLOGY") is None,
        help="pre-registered deployment latency methodology YAML",
    )
    parser.add_argument(
        "--journeys",
        type=Path,
        default=_DEFAULT_JOURNEYS,
        help="frozen synthetic latency journey corpus",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=_configured_path("VOICE_AGENT_LATENCY_REPORT"),
        required=_configured_path("VOICE_AGENT_LATENCY_REPORT") is None,
        help="new immutable certification result path",
    )
    parser.add_argument(
        "--deployment-id",
        default=os.environ.get("VOICE_AGENT_DEPLOYMENT_ID", "").strip(),
        required=not os.environ.get("VOICE_AGENT_DEPLOYMENT_ID", "").strip(),
        help="immutable deployment artifact identifier",
    )
    parser.add_argument(
        "--config-root",
        type=Path,
        default=_CONFIG_ROOT,
        help="merchant and provider configuration root",
    )
    return parser.parse_args()


async def _run(arguments: argparse.Namespace) -> int:
    methodology = load_latency_methodology(arguments.methodology)
    if methodology.environment is not LatencyEnvironment.DEPLOYMENT:
        raise ValueError("deployment certification requires a deployment-shaped methodology")
    corpus = load_latency_journey_corpus(arguments.journeys)
    platform_config = load_platform_runtime_config(arguments.platform_config)
    secrets = EnvSecretResolver()
    application_dsn = secrets.resolve(platform_config.database.application_dsn_ref.uri)
    runtime_fingerprint = deployment_runtime_contract_fingerprint(
        platform_config,
        application_dsn=application_dsn,
    )
    if methodology.runtime_contract_fingerprint != runtime_fingerprint:
        raise ValueError("latency methodology does not match the deployment runtime")
    probes = DeploymentLatencyProbeFactory(
        config_root=arguments.config_root,
        platform_config=platform_config,
        application_dsn=application_dsn,
        deployment_id=arguments.deployment_id,
        methodology=methodology,
        corpus=corpus,
        secrets=secrets,
    )
    run = await run_latency_certification(
        methodology,
        deployment_id=arguments.deployment_id,
        startup_probe=probes.startup_probe,
        turn_probe=probes.turn_probe,
        run_at=datetime.now(tz=UTC),
    )
    await asyncio.to_thread(write_latency_certification_run, arguments.report, run)
    if run.outcome is LatencyCertificationOutcome.ABORTED:
        logger.error(
            "durable latency certification aborted; evidence written to %s",
            arguments.report,
        )
        return 1
    assert run.report is not None
    if not run.report.gate.passed:
        logger.error("durable latency gate failed; evidence written to %s", arguments.report)
        return 1
    logger.info("durable latency gate passed; evidence written to %s", arguments.report)
    return 0


def _event_loop_factory() -> asyncio.AbstractEventLoop:
    if sys.platform == "win32":
        return asyncio.SelectorEventLoop(selectors.SelectSelector())
    return asyncio.new_event_loop()


def main() -> int:
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    return asyncio.run(_run(_arguments()), loop_factory=_event_loop_factory)


if __name__ == "__main__":
    raise SystemExit(main())
