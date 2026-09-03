"""Serve one configured merchant through the qualified voice pipeline.

Run:
    uv run python scripts/voice_agent.py console   # local mic/speaker dev loop
    uv run python scripts/voice_agent.py dev       # LiveKit Cloud -> Playground (the live call)

Needs in .env: the provider keys referenced by merchant config plus
LIVEKIT_URL/LIVEKIT_API_KEY/LIVEKIT_API_SECRET for dev mode.
VOICE_AGENT_DEPLOYMENT_ID must identify the immutable deployed artifact.
Console mode also requires an explicit VOICE_AGENT_MERCHANT_ID. Network jobs require strict
server-side dispatch metadata; SIP jobs cross-check it against the inbound number.
VOICE_AGENT_NAME must match the explicit LiveKit dispatch target for network worker commands.

Startup requires current LLM conformance and semantic-routing qualification reports. A qualified
session opens with the configured disclosure and logs per-turn latency through the voice pipeline.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from dotenv import load_dotenv
from livekit import agents
from livekit.agents.voice import room_io

from agnostic_market.agents.routing_activation import QualifiedSemanticRouterFactory
from agnostic_market.agents.telemetry import (
    DisabledTelemetrySink,
    InMemoryTelemetrySink,
    TelemetryStore,
    TenantTelemetry,
)
from agnostic_market.application import build_fixture_tenant_services
from agnostic_market.config.loader import load_yaml_layer
from agnostic_market.config.registry import ConfigRegistry
from agnostic_market.llm.gateway import LLMGateway, load_provider_credentials
from agnostic_market.llm.providers import (
    ConformanceRegistry,
    load_conformance_targets,
    require_llm_certification,
)
from agnostic_market.secrets.env_resolver import EnvSecretResolver
from agnostic_market.tenancy.resolver import TenantResolutionError
from agnostic_market.voice.admission import VoiceJobAdmission, VoiceTenantAdmission
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

logger = logging.getLogger("voice_agent")


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


async def _start_admitted_session(
    ctx: agents.JobContext,
    loop: VoiceLoop,
    admission: VoiceTenantAdmission,
) -> None:
    room_options = (
        room_io.RoomOptions()
        if admission.participant_identity is None
        else room_io.RoomOptions(participant_identity=admission.participant_identity)
    )
    await loop.session.start(loop.agent, room=ctx.room, room_options=room_options)


async def _certification_participant_identity(
    ctx: agents.JobContext,
    admission: VoiceTenantAdmission,
    *,
    timeout_seconds: float,
) -> str:
    if admission.participant_identity is not None:
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
    routing_contract = load_yaml_layer(
        _CONFIG_ROOT / "eval" / "frontline_semantic_route_structural.yaml"
    )
    expected_corpus_fingerprint = routing_contract.get("frozen_corpus_fingerprint")
    if not isinstance(expected_corpus_fingerprint, str) or not expected_corpus_fingerprint.strip():
        raise RuntimeError("semantic routing corpus contract has no frozen fingerprint")
    routing_factory = QualifiedSemanticRouterFactory(
        qualification_path=_CONFIG_ROOT / "telemetry" / "semantic_routing_report.json",
        selection=resolved.config.llm.routing,
        credentials=credentials,
        secrets=secrets,
        structured_output_method=gateway.structured_output_method(resolved.config.llm.routing),
        timeout_seconds=resolved.config.runtime.semantic_router_timeout_seconds,
        input_max_chars=resolved.config.runtime.semantic_router_input_max_chars,
        max_report_age_days=targets.max_report_age_days,
        expected_corpus_fingerprint=expected_corpus_fingerprint,
    )
    operational_telemetry: TelemetryStore = (
        InMemoryTelemetrySink() if close_certification is not None else DisabledTelemetrySink()
    )
    routing_evidence = DisabledTelemetrySink()

    loop = build_voice_loop(
        tenant,
        resolved,
        credentials,
        secrets,
        deployment_id=_deployment_id(),
        tenant_services=build_fixture_tenant_services(
            _CONFIG_ROOT,
            tenant,
            telemetry=TenantTelemetry(
                tenant.tenant_id,
                operational_telemetry,
                routing_evidence,
            ),
        ),
        routing_recognizer_factory=routing_factory,
    )
    loop.register_shutdown(ctx)

    admission = await admission_boundary.complete(
        ctx,
        preflight,
        timeout_seconds=resolved.config.runtime.voice_admission_timeout_seconds,
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
    # Stop the earcon's mixer on shutdown, or its background task throws
    # "Event loop is closed" when the loop tears down (a dangling task on disconnect).
    ctx.add_shutdown_callback(loop.background_audio.aclose)


if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name=_agent_name(sys.argv[1:]),
        )
    )
