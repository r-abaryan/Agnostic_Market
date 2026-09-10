"""Serve one configured merchant through the qualified voice pipeline.

Run:
    uv run python scripts/voice_agent.py console   # local mic/speaker dev loop
    uv run python scripts/voice_agent.py dev       # LiveKit Cloud -> Playground (the live call)

Needs in .env: the provider keys referenced by merchant config plus
LIVEKIT_URL/LIVEKIT_API_KEY/LIVEKIT_API_SECRET for dev mode.
VOICE_AGENT_DEPLOYMENT_ID must identify the immutable deployed artifact.
Console mode also requires an explicit VOICE_AGENT_MERCHANT_ID. Network jobs require strict
server-side dispatch metadata and an absolute VOICE_AGENT_PLATFORM_CONFIG path; SIP jobs
cross-check admission against the inbound number.
VOICE_AGENT_NAME must match the explicit LiveKit dispatch target for network worker commands.

Startup requires current LLM conformance and semantic-routing qualification reports. A qualified
session opens with the configured disclosure and logs per-turn latency through the voice pipeline.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import NoReturn

from dotenv import load_dotenv
from livekit import agents
from livekit.agents.voice import room_io

from agnostic_market.agents.routing_activation import build_qualified_semantic_router_factory
from agnostic_market.agents.telemetry import (
    DisabledTelemetrySink,
    InMemoryTelemetrySink,
    TelemetryStore,
    TenantTelemetry,
)
from agnostic_market.application import (
    build_fixture_tenant_services,
    prepare_application_routing,
)
from agnostic_market.config.registry import ConfigRegistry
from agnostic_market.durability.latency import (
    LatencyMeasurementSurface,
    deployment_runtime_contract_fingerprint,
    load_latency_journey_corpus,
    require_deployment_latency_evidence,
)
from agnostic_market.durability.platform_runtime import (
    DurablePlatformResources,
    DurableSessionResources,
    load_platform_runtime_config,
)
from agnostic_market.llm.gateway import LLMGateway, load_provider_credentials
from agnostic_market.llm.providers import (
    ConformanceRegistry,
    load_conformance_targets,
    require_llm_certification,
)
from agnostic_market.secrets.env_resolver import EnvSecretResolver
from agnostic_market.tenancy.resolver import TenantResolutionError
from agnostic_market.voice.admission import (
    NetworkVoiceAdmissionPreflight,
    NetworkVoiceTenantAdmission,
    VoiceJobAdmission,
    VoiceTenantAdmission,
)
from agnostic_market.voice.pipeline import VoiceLoop, build_voice_loop

if __package__:
    from .close_evidence_recorder import (
        CloseEvidenceRecorder,
        load_close_certification_request,
    )
else:
    from close_evidence_recorder import (
        CloseEvidenceRecorder,
        load_close_certification_request,
    )

# .env must be in the process env BEFORE the LiveKit worker starts (it reads LIVEKIT_URL
# at startup) and BEFORE the module-level env read below — so load at import, not in main
# (job subprocesses import this module without executing the __main__ block).
load_dotenv()

_CONFIG_ROOT = Path(__file__).resolve().parents[1] / "config"
_DEVELOPMENT_MERCHANT_ID_ENV = "VOICE_AGENT_MERCHANT_ID"
_DEPLOYMENT_ID_ENV = "VOICE_AGENT_DEPLOYMENT_ID"
_AGENT_NAME_ENV = "VOICE_AGENT_NAME"
_PLATFORM_CONFIG_ENV = "VOICE_AGENT_PLATFORM_CONFIG"
_LATENCY_METHODOLOGY_ENV = "VOICE_AGENT_LATENCY_METHODOLOGY"
_LATENCY_REPORT_ENV = "VOICE_AGENT_LATENCY_REPORT"

logger = logging.getLogger("voice_agent")


def _prewarm(_process: agents.JobProcess) -> None:
    """Select the psycopg-compatible loop before LiveKit creates the job loop."""
    if sys.platform == "win32":
        # LiveKit 1.6 creates its loop after prewarm and exposes no loop_factory.
        # Replace this policy bridge when the SDK offers explicit loop selection.
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def _noop_prewarm(_process: agents.JobProcess) -> None:
    """Leave console execution on the platform-default event loop."""


def _prewarm_for_arguments(arguments: Sequence[str]) -> Callable[[agents.JobProcess], None]:
    if arguments and arguments[0] == "console":
        return _noop_prewarm
    return _prewarm


def _deployment_id() -> str:
    value = os.environ.get(_DEPLOYMENT_ID_ENV, "").strip()
    if not value:
        raise RuntimeError(f"{_DEPLOYMENT_ID_ENV} must identify the immutable deployment artifact")
    return value


def _agent_name(arguments: Sequence[str]) -> str:
    value = os.environ.get(_AGENT_NAME_ENV, "").strip()
    if (
        not arguments
        or any(argument in {"-h", "--help"} for argument in arguments)
        or arguments[0] in {"console", "download-files"}
    ):
        return value
    if not value:
        raise RuntimeError(f"{_AGENT_NAME_ENV} must identify the LiveKit dispatch target")
    return value


def _required_absolute_path(
    variable: str,
    environ: Mapping[str, str] | None = None,
) -> Path:
    environment = os.environ if environ is None else environ
    raw_path = environment.get(variable, "").strip()
    if not raw_path:
        raise RuntimeError(f"{variable} must identify an absolute file path")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        raise RuntimeError(f"{variable} must be an absolute path")
    return path


def _platform_config_path(environ: Mapping[str, str] | None = None) -> Path:
    return _required_absolute_path(_PLATFORM_CONFIG_ENV, environ)


def _latency_evidence_paths(
    environ: Mapping[str, str] | None = None,
) -> tuple[Path, Path]:
    return (
        _required_absolute_path(_LATENCY_METHODOLOGY_ENV, environ),
        _required_absolute_path(_LATENCY_REPORT_ENV, environ),
    )


async def _raise_after_durable_startup_cleanup(
    failure: BaseException,
    resources: DurablePlatformResources,
    durable_session: DurableSessionResources | None,
    loop: VoiceLoop | None,
) -> NoReturn:
    cleanup_failures: list[BaseException] = []
    try:
        if loop is not None:
            await loop.aclose()
        elif durable_session is not None:
            await durable_session.closer.begin_close()
            await durable_session.closer.finalize_close()
    except BaseException as cleanup_failure:
        cleanup_failures.append(cleanup_failure)
    try:
        await resources.aclose()
    except BaseException as cleanup_failure:
        cleanup_failures.append(cleanup_failure)
    if cleanup_failures:
        cleanup_cause: BaseException = cleanup_failures[0]
        if len(cleanup_failures) > 1:
            cleanup_cause = BaseExceptionGroup(
                "durable voice startup cleanup failed",
                cleanup_failures,
            )
        raise failure from cleanup_cause
    raise failure


async def _start_admitted_session(
    ctx: agents.JobContext,
    loop: VoiceLoop,
    admission: VoiceTenantAdmission,
) -> None:
    room_options = room_io.RoomOptions()
    if isinstance(admission, NetworkVoiceTenantAdmission):
        room_options = room_io.RoomOptions(participant_identity=admission.participant_identity)
    await loop.session.start(loop.agent, room=ctx.room, room_options=room_options)


async def _certification_participant_identity(
    ctx: agents.JobContext,
    admission: VoiceTenantAdmission,
    *,
    timeout_seconds: float,
) -> str:
    if isinstance(admission, NetworkVoiceTenantAdmission):
        return admission.participant_identity
    try:
        async with asyncio.timeout(timeout_seconds):
            participant = await ctx.wait_for_participant()
    except TimeoutError as exc:
        raise TenantResolutionError(
            "close certification timed out while waiting for the participant"
        ) from exc
    return participant.identity


async def entrypoint(ctx: agents.JobContext) -> None:
    registry = ConfigRegistry(_CONFIG_ROOT).load()
    admission_boundary = VoiceJobAdmission(
        registry,
        development_merchant_id=os.environ.get(_DEVELOPMENT_MERCHANT_ID_ENV),
    )
    preflight = admission_boundary.preflight(ctx)
    tenant = preflight.tenant
    resolved = preflight.resolved

    close_certification = load_close_certification_request(_CONFIG_ROOT)
    secrets = EnvSecretResolver()
    credentials = load_provider_credentials(_CONFIG_ROOT / "base" / "providers.yaml")

    targets = load_conformance_targets(_CONFIG_ROOT / "conformance" / "targets.yaml")
    conformance = ConformanceRegistry(
        _CONFIG_ROOT / "conformance" / "reports.json",
        max_report_age_days=targets.max_report_age_days,
    )
    require_llm_certification(resolved.config, conformance)

    gateway = LLMGateway(credentials, secrets)
    routing_factory = build_qualified_semantic_router_factory(
        _CONFIG_ROOT,
        selection=resolved.config.llm.routing,
        credentials=credentials,
        secrets=secrets,
        structured_output_method=gateway.structured_output_method(resolved.config.llm.routing),
        timeout_seconds=resolved.config.runtime.semantic_router_timeout_seconds,
        input_max_chars=resolved.config.runtime.semantic_router_input_max_chars,
        max_report_age_days=targets.max_report_age_days,
    )
    prepared_routing = prepare_application_routing(routing_factory)
    operational_telemetry: TelemetryStore = (
        InMemoryTelemetrySink() if close_certification is not None else DisabledTelemetrySink()
    )
    routing_evidence = DisabledTelemetrySink()
    tenant_telemetry = TenantTelemetry(
        tenant.tenant_id,
        operational_telemetry,
        routing_evidence,
    )
    deployment_id = _deployment_id()
    platform_resources: DurablePlatformResources | None = None
    durable_session: DurableSessionResources | None = None
    loop: VoiceLoop | None = None
    if isinstance(preflight, NetworkVoiceAdmissionPreflight):
        methodology_path, latency_report_path = _latency_evidence_paths()
        platform_config = load_platform_runtime_config(_platform_config_path())
        application_dsn = secrets.resolve(platform_config.database.application_dsn_ref.uri)
        journey_corpus = load_latency_journey_corpus(
            _CONFIG_ROOT / "eval" / "durable_latency_journeys.yaml"
        )
        require_deployment_latency_evidence(
            methodology_path,
            latency_report_path,
            expected_deployment_id=deployment_id,
            expected_journey_corpus=journey_corpus,
            expected_runtime_contract_fingerprint=deployment_runtime_contract_fingerprint(
                platform_config,
                application_dsn=application_dsn,
            ),
            required_measurement_surface=LatencyMeasurementSurface.VOICE_PROCESSING,
        )
        platform_resources = await DurablePlatformResources.open(
            platform_config,
            secrets,
            application_dsn=application_dsn,
        )
    try:
        admission = await admission_boundary.complete(
            ctx,
            preflight,
            timeout_seconds=resolved.config.runtime.voice_admission_timeout_seconds,
        )
        if platform_resources is not None:
            if not isinstance(admission, NetworkVoiceTenantAdmission):
                raise RuntimeError("network durable resources require network admission")
            durable_session = await platform_resources.acquire_fresh_session(
                tenant_id=tenant.tenant_id,
                admitted_authority=admission.session_authority,
                deployment_id=deployment_id,
                config_version=tenant.config_version,
            )
        tenant_services = build_fixture_tenant_services(
            _CONFIG_ROOT,
            tenant,
            telemetry=tenant_telemetry,
            checkpointer=(durable_session.checkpointer if durable_session is not None else None),
        )
        loop = await build_voice_loop(
            tenant,
            resolved,
            credentials,
            secrets,
            deployment_id=deployment_id,
            tenant_services=tenant_services,
            routing_recognizer_factory=prepared_routing,
            durable_session=durable_session,
        )
        loop.register_shutdown(
            ctx,
            release_job_resources=(
                platform_resources.aclose if platform_resources is not None else None
            ),
        )
        if platform_resources is not None:
            loop.start_lease_supervision(
                ctx.room,
                transport_retirement_timeout_seconds=(
                    platform_resources.config.sessions.transport_retirement_timeout_seconds
                ),
            )
        logger.info(
            "serving merchant %s (config_version %s)",
            tenant.tenant_id,
            tenant.config_version[:12],
        )

        if close_certification is not None:
            # Certification is intentionally single-participant and opt-in. Resolve the linked
            # participant before AgentSession.start so disconnect evidence never depends on
            # LiveKit's set-backed close-listener ordering.
            linked_participant_identity = await _certification_participant_identity(
                ctx,
                admission,
                timeout_seconds=resolved.config.runtime.voice_admission_timeout_seconds,
            )
            close_recorder = CloseEvidenceRecorder(
                close_certification,
                merchant_id=tenant.tenant_id,
                telemetry=operational_telemetry,
            )
            close_recorder.attach(
                session=loop.session,
                room=ctx.room,
                engine=loop.engine,
                effect_source=loop.application.services.order_store,
                linked_participant_identity=linked_participant_identity,
            )
            ctx.add_shutdown_callback(close_recorder.wait_for_completion)
        # The disclosure (COMPLIANCE 2 / EU AI Act Art. 50(1)) plays via the agent's own
        # on_enter hook - structurally first, before any user turn can be answered.
        await _start_admitted_session(ctx, loop, admission)
        # The thinking-sound earcon needs the room (a runtime concern); start it after the
        # session. Auto-plays while the agent is 'thinking', stops when it speaks (no overlap with
        # the answer or a readback). No-op/warn in console mode (LiveKit-managed).
        await loop.background_audio.start(room=ctx.room, agent_session=loop.session)
    except BaseException as failure:
        if platform_resources is not None:
            await _raise_after_durable_startup_cleanup(
                failure,
                platform_resources,
                durable_session,
                loop,
            )
        raise


if __name__ == "__main__":
    cli_arguments = tuple(sys.argv[1:])
    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=_prewarm_for_arguments(cli_arguments),
            agent_name=_agent_name(cli_arguments),
        )
    )
