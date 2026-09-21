"""Serve one configured merchant through the qualified voice pipeline.

Run:
    uv run python scripts/voice_agent.py console   # local mic/speaker dev loop
    uv run python scripts/voice_agent.py dev       # LiveKit Cloud -> Playground (the live call)

Needs in .env: the provider keys referenced by merchant config plus
LIVEKIT_URL/LIVEKIT_API_KEY/LIVEKIT_API_SECRET for dev mode.
VOICE_AGENT_DEPLOYMENT_ID must identify the immutable deployed artifact.
VOICE_AGENT_BUILD_ARTIFACT_DIGEST must identify the immutable OCI image for network jobs.
Console mode also requires an explicit VOICE_AGENT_MERCHANT_ID. Network jobs require strict
server-side dispatch metadata and an absolute VOICE_AGENT_PLATFORM_CONFIG path; SIP jobs
cross-check admission against the inbound number.
VOICE_AGENT_CERTIFICATION_CONFIG owns the production and certification dispatch names.

Startup requires current LLM conformance, an immutable semantic-routing release package, and the
voice evidence bound to that package. A qualified session opens with the configured disclosure and
logs per-turn latency through the voice pipeline.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from contextvars import ContextVar
from pathlib import Path
from typing import NoReturn

from dotenv import load_dotenv
from livekit import agents
from livekit.agents.voice import room_io

from agnostic_market.agents.routing_activation import (
    ConfiguredSemanticRouterFactory,
    QualifiedSemanticRouterFactory,
    build_qualified_semantic_router_factory,
    semantic_routing_runtime_contract,
)
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
    LatencyJourneyContract,
    LatencyJourneyCorpus,
    LatencyMeasurementSurface,
    LatencyPhase,
    VoiceApplicationContract,
    VoiceTransportSurface,
    deployment_runtime_contract_fingerprint,
    latency_evidence_schema_version_for_methodology,
    latency_journey_corpus_fingerprint,
    load_latency_journey_corpus,
    require_deployment_latency_evidence,
)
from agnostic_market.durability.platform_runtime import (
    DurablePlatformResources,
    DurableSessionResources,
    load_platform_runtime_config,
)
from agnostic_market.durability.voice_certification import (
    VoiceCertificationMeasurements,
    VoiceCertificationSampleAssignment,
    VoiceCertificationTarget,
    announce_voice_certification_progress,
    announce_voice_certification_ready,
    capture_voice_certification_application_state,
    load_voice_certification_target,
    load_voice_sample_assignment,
    resolve_voice_journey_contract,
    submit_voice_certification_sample,
    validate_certification_admission,
    validate_certification_job,
    voice_certification_target_fingerprint,
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
    DevelopmentStandardVoiceJobAdmission,
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
_PLATFORM_CONFIG_ENV = "VOICE_AGENT_PLATFORM_CONFIG"
_LATENCY_METHODOLOGY_ENV = "VOICE_AGENT_LATENCY_METHODOLOGY"
_LATENCY_REPORT_ENV = "VOICE_AGENT_LATENCY_REPORT"
_CERTIFICATION_CONFIG_ENV = "VOICE_AGENT_CERTIFICATION_CONFIG"
_BUILD_ARTIFACT_DIGEST_ENV = "VOICE_AGENT_BUILD_ARTIFACT_DIGEST"

logger = logging.getLogger("voice_agent")
_startup_stage: ContextVar[str] = ContextVar("voice_startup_stage", default="received")


def _mark_startup_stage(stage: str) -> None:
    _startup_stage.set(stage)
    logger.info(
        "voice startup stage=%s",
        stage,
        extra={"event": "voice_startup_stage", "startup_stage": stage},
    )


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


def _build_artifact_digest() -> str:
    value = os.environ.get(_BUILD_ARTIFACT_DIGEST_ENV, "").strip()
    if not value:
        raise RuntimeError(
            f"{_BUILD_ARTIFACT_DIGEST_ENV} must identify the immutable OCI build artifact"
        )
    return value


def _agent_name(arguments: Sequence[str]) -> str:
    if (
        not arguments
        or any(argument in {"-h", "--help"} for argument in arguments)
        or arguments[0] in {"console", "download-files"}
    ):
        return ""
    target_path = _required_absolute_path(_CERTIFICATION_CONFIG_ENV)
    return load_voice_certification_target(target_path).production_agent_name


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


async def _run_entrypoint(
    ctx: agents.JobContext,
    *,
    certification_target: VoiceCertificationTarget | None = None,
    development_network: bool = False,
) -> None:
    if development_network and certification_target is not None:
        raise RuntimeError("development network jobs cannot produce certification evidence")
    _mark_startup_stage("configuration_load")
    certification_directive = None
    if certification_target is not None:
        certification_directive = validate_certification_job(ctx.job, certification_target)
    certification_measurements = (
        VoiceCertificationMeasurements(
            evidence_schema_version=latency_evidence_schema_version_for_methodology(
                certification_directive.methodology_schema_version
            )
        )
        if certification_directive is not None
        else None
    )
    registry = ConfigRegistry(_CONFIG_ROOT).load()
    _mark_startup_stage("admission_preflight")
    admission_boundary = (
        DevelopmentStandardVoiceJobAdmission(
            registry,
            merchant_id=os.environ.get(_DEVELOPMENT_MERCHANT_ID_ENV, ""),
        )
        if development_network
        else VoiceJobAdmission(
            registry,
            development_merchant_id=os.environ.get(_DEVELOPMENT_MERCHANT_ID_ENV),
        )
    )
    preflight = admission_boundary.preflight(ctx)
    _mark_startup_stage("preflight_complete")
    tenant = preflight.tenant
    resolved = preflight.resolved
    if certification_target is not None:
        if not isinstance(preflight, NetworkVoiceAdmissionPreflight):
            raise RuntimeError("voice certification requires network admission")
        validate_certification_admission(
            certification_target,
            merchant_id=tenant.tenant_id,
            participant_kind=preflight.participant_kind,
            participant_identity=preflight.participant_identity,
        )

    close_certification = load_close_certification_request(_CONFIG_ROOT)
    secrets = EnvSecretResolver()
    credentials = load_provider_credentials(_CONFIG_ROOT / "base" / "providers.yaml")

    targets = load_conformance_targets(_CONFIG_ROOT / "conformance" / "targets.yaml")
    conformance = ConformanceRegistry(
        _CONFIG_ROOT / "conformance" / "reports.json",
        max_report_age_days=targets.max_report_age_days,
    )
    require_llm_certification(resolved.config, conformance)
    _mark_startup_stage("llm_certification_complete")

    gateway = LLMGateway(credentials, secrets)
    routing_structured_output_method = gateway.structured_output_method(resolved.config.llm.routing)
    _mark_startup_stage("routing_qualification")
    qualified_routing_factory: QualifiedSemanticRouterFactory | None = None
    if development_network:
        logger.warning(
            "using non-authorizing development semantic routing",
            extra={"event": "development_routing_non_authorizing"},
        )
        routing_factory = ConfiguredSemanticRouterFactory(
            selection=resolved.config.llm.routing,
            credentials=credentials,
            secrets=secrets,
            structured_output_method=routing_structured_output_method,
            timeout_seconds=resolved.config.runtime.semantic_router_timeout_seconds,
            input_max_chars=resolved.config.runtime.semantic_router_input_max_chars,
        )
    else:
        qualified_routing_factory = build_qualified_semantic_router_factory(
            _CONFIG_ROOT,
            selection=resolved.config.llm.routing,
            credentials=credentials,
            secrets=secrets,
            structured_output_method=routing_structured_output_method,
            timeout_seconds=resolved.config.runtime.semantic_router_timeout_seconds,
            input_max_chars=resolved.config.runtime.semantic_router_input_max_chars,
            max_report_age_days=targets.max_report_age_days,
        )
        routing_factory = qualified_routing_factory
    prepared_routing = prepare_application_routing(routing_factory)
    _mark_startup_stage("routing_qualification_complete")
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
    certification_assignment: VoiceCertificationSampleAssignment | None = None
    certification_contract: LatencyJourneyContract | None = None
    journey_corpus: LatencyJourneyCorpus | None = None
    journey_corpus_fingerprint_value: str | None = None
    runtime_contract_fingerprint: str | None = None
    application_contract: VoiceApplicationContract | None = None
    loop: VoiceLoop | None = None
    if isinstance(preflight, NetworkVoiceAdmissionPreflight) and not development_network:
        if qualified_routing_factory is None:
            raise RuntimeError("production network jobs require qualified semantic routing")
        _mark_startup_stage("platform_contract_load")
        platform_config = load_platform_runtime_config(_platform_config_path())
        application_dsn = secrets.resolve(platform_config.database.application_dsn_ref.uri)
        journey_corpus = load_latency_journey_corpus(
            _CONFIG_ROOT / "eval" / "durable_latency_journeys.yaml"
        )
        journey_corpus_fingerprint_value = latency_journey_corpus_fingerprint(journey_corpus)
        runtime_contract_fingerprint = deployment_runtime_contract_fingerprint(
            platform_config,
            application_dsn=application_dsn,
        )
        application_identity_target = certification_target or load_voice_certification_target(
            _required_absolute_path(_CERTIFICATION_CONFIG_ENV)
        )
        _mark_startup_stage("application_contract_validation")
        application_contract = VoiceApplicationContract(
            durable_platform_fingerprint=runtime_contract_fingerprint,
            build_artifact_digest=_build_artifact_digest(),
            tenant_config_version=tenant.config_version,
            semantic_routing=semantic_routing_runtime_contract(
                prepared_routing.capability_registry,
                selection=resolved.config.llm.routing,
                structured_output_method=routing_structured_output_method,
                timeout_seconds=resolved.config.runtime.semantic_router_timeout_seconds,
                input_max_chars=resolved.config.runtime.semantic_router_input_max_chars,
                corpus_fingerprint=qualified_routing_factory.expected_corpus_fingerprint,
                qualification_evidence_fingerprint=(
                    qualified_routing_factory.expected_qualification_evidence_fingerprint
                ),
            ),
            certification_target_fingerprint=voice_certification_target_fingerprint(
                application_identity_target
            ),
        )
        if certification_target is None:
            methodology_path, latency_report_path = _latency_evidence_paths()
            require_deployment_latency_evidence(
                methodology_path,
                latency_report_path,
                expected_deployment_id=deployment_id,
                expected_journey_corpus=journey_corpus,
                expected_runtime_contract_fingerprint=runtime_contract_fingerprint,
                expected_application_contract=application_contract,
                required_measurement_surface=LatencyMeasurementSurface.VOICE_PROCESSING,
                required_transport_surface=VoiceTransportSurface(preflight.participant_kind),
            )
            _mark_startup_stage("latency_evidence_authorized")
        platform_resources = await DurablePlatformResources.open(
            platform_config,
            secrets,
            durability_timing=(
                certification_measurements.durability_timing
                if certification_measurements is not None
                else None
            ),
            application_dsn=application_dsn,
        )
        _mark_startup_stage("platform_resources_opened")
    try:
        _mark_startup_stage("transport_admission")
        admission = await admission_boundary.complete(
            ctx,
            preflight,
            timeout_seconds=resolved.config.runtime.voice_admission_timeout_seconds,
        )
        _mark_startup_stage("transport_admitted")
        if certification_target is not None:
            if (
                certification_measurements is None
                or not isinstance(admission, NetworkVoiceTenantAdmission)
                or journey_corpus_fingerprint_value is None
                or runtime_contract_fingerprint is None
                or application_contract is None
            ):
                raise RuntimeError("voice certification requires network measurement authority")
            certification_assignment = load_voice_sample_assignment(
                ctx.job,
                certification_target,
                expected_deployment_id=deployment_id,
                expected_journey_corpus_fingerprint=journey_corpus_fingerprint_value,
                expected_runtime_contract_fingerprint=runtime_contract_fingerprint,
                expected_application_contract=application_contract,
                worker_id=ctx.worker_id,
                worker_participant_identity=ctx.agent.identity,
            )
            if certification_assignment.sample.phase is LatencyPhase.TURN:
                if journey_corpus is None:
                    raise RuntimeError("voice certification requires its journey corpus")
                certification_contract = resolve_voice_journey_contract(
                    journey_corpus,
                    certification_assignment.sample,
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
            _mark_startup_stage("durable_session_acquired")
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
            turn_latency_observer=(
                certification_measurements.observe_turn_latency
                if certification_measurements is not None
                else None
            ),
            turn_started_observer=(
                certification_measurements.begin_turn
                if certification_measurements is not None
                else None
            ),
            turn_event_observer=(
                certification_measurements.observe_event
                if certification_measurements is not None
                else None
            ),
            turn_observer_failure_observer=(
                certification_measurements.record_observer_failure
                if certification_measurements is not None
                else None
            ),
        )
        _mark_startup_stage("voice_loop_built")
        if certification_contract is not None:
            if certification_measurements is None:
                raise RuntimeError("voice certification has no measurement collector")
            certification_measurements.bind_turn_contract(
                certification_contract,
                lambda: capture_voice_certification_application_state(
                    loop.application.state.cart_store.snapshot(),
                    tenant_services.order_store,
                ),
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
        _mark_startup_stage("voice_session_started")
        # The thinking-sound earcon needs the room (a runtime concern); start it after the
        # session. Auto-plays while the agent is 'thinking', stops when it speaks (no overlap with
        # the answer or a readback). No-op/warn in console mode (LiveKit-managed).
        await loop.background_audio.start(room=ctx.room, agent_session=loop.session)
        _mark_startup_stage("ready")
        if certification_assignment is not None and certification_measurements is not None:
            observation = certification_measurements.mark_ready(certification_assignment)
            await announce_voice_certification_ready(ctx.agent, certification_assignment)
            if observation is None:

                async def report_setup_progress(completed_setup_turns: int) -> None:
                    await announce_voice_certification_progress(
                        ctx.agent,
                        certification_assignment,
                        completed_setup_turns=completed_setup_turns,
                    )

                observation = (
                    await certification_measurements.wait_for_turn(
                        certification_assignment,
                        on_setup_progress=report_setup_progress,
                    )
                ).observation
            await submit_voice_certification_sample(
                ctx.agent,
                certification_assignment,
                observation,
            )
    except BaseException as failure:
        if platform_resources is not None:
            await _raise_after_durable_startup_cleanup(
                failure,
                platform_resources,
                durable_session,
                loop,
            )
        raise


async def _entrypoint_with_diagnostics(
    ctx: agents.JobContext,
    *,
    certification_target: VoiceCertificationTarget | None = None,
    development_network: bool = False,
) -> None:
    """Run one job with privacy-safe, stage-correlated startup diagnostics."""

    token = _startup_stage.set("received")
    try:
        await _run_entrypoint(
            ctx,
            certification_target=certification_target,
            development_network=development_network,
        )
    except asyncio.CancelledError:
        stage = _startup_stage.get()
        logger.info(
            "voice job cancelled stage=%s",
            stage,
            extra={"event": "voice_startup_cancelled", "startup_stage": stage},
        )
        raise
    except BaseException as exc:
        stage = _startup_stage.get()
        logger.error(
            "voice job failed stage=%s exception_type=%s",
            stage,
            type(exc).__name__,
            extra={
                "event": "voice_startup_failed",
                "startup_stage": stage,
                "exception_type": type(exc).__name__,
            },
        )
        raise
    finally:
        _startup_stage.reset(token)


async def entrypoint(
    ctx: agents.JobContext,
    *,
    certification_target: VoiceCertificationTarget | None = None,
) -> None:
    """Run one production or certification job with stage-correlated diagnostics."""

    await _entrypoint_with_diagnostics(ctx, certification_target=certification_target)


async def development_network_entrypoint(ctx: agents.JobContext) -> None:
    """Run one non-authorizing LiveKit development job with in-memory state."""

    logger.warning(
        "starting non-authorizing development network job",
        extra={"event": "development_network_non_authorizing"},
    )
    await _entrypoint_with_diagnostics(ctx, development_network=True)


if __name__ == "__main__":
    cli_arguments = tuple(sys.argv[1:])
    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=_prewarm_for_arguments(cli_arguments),
            agent_name=_agent_name(cli_arguments),
        )
    )
