"""Run one pre-registered LiveKit voice-processing latency certification."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv
from livekit import api, rtc

from agnostic_market.agents.frontline.graph import build_frontline_capability_registry
from agnostic_market.agents.routing_activation import (
    load_semantic_routing_release_evidence,
    semantic_routing_corpus_fingerprint,
    semantic_routing_release_evidence_fingerprint,
    semantic_routing_runtime_contract,
)
from agnostic_market.config.registry import ConfigRegistry
from agnostic_market.durability.latency import (
    DurableLatencyCertificationRun,
    LatencyCertificationOutcome,
    LatencyEnvironment,
    LatencyMeasurementSurface,
    LatencyObservation,
    VoiceApplicationContract,
    VoiceAudioTreatment,
    VoiceTransportSurface,
    bind_latency_journey_contracts,
    deployment_runtime_contract_fingerprint,
    load_latency_certification_run,
    load_latency_journey_corpus,
    load_latency_methodology,
    require_deployment_latency_report,
    require_voice_application_contract,
    write_latency_certification_run,
)
from agnostic_market.durability.platform_runtime import load_platform_runtime_config
from agnostic_market.durability.voice_certification import (
    VOICE_CERTIFICATION_SAMPLE_ATTRIBUTE,
    PreparedVoiceAudio,
    VoiceCertificationController,
    VoiceCertificationDispatchDirective,
    VoiceCertificationProtocolError,
    VoiceCertificationTarget,
    abort_voice_controller_cleanup,
    load_voice_audio_asset,
    load_voice_certification_target,
    publish_prepared_voice_audio,
    register_voice_certification_controller,
    run_voice_certification_controller,
    run_voice_certification_smoke,
    voice_certification_target_fingerprint,
)
from agnostic_market.llm.gateway import LLMGateway, load_provider_credentials
from agnostic_market.secrets.env_resolver import EnvSecretResolver
from agnostic_market.voice.admission import VoiceJobMetadata

_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_ROOT = _ROOT / "config"
_DEFAULT_JOURNEYS = _CONFIG_ROOT / "eval" / "durable_latency_journeys.yaml"

logger = logging.getLogger("durable_voice_certification")


def _configured_path(variable: str) -> Path | None:
    value = os.environ.get(variable, "").strip()
    return Path(value) if value else None


def _arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--platform-config",
        type=Path,
        default=_configured_path("VOICE_AGENT_PLATFORM_CONFIG"),
        required=_configured_path("VOICE_AGENT_PLATFORM_CONFIG") is None,
    )
    parser.add_argument(
        "--methodology",
        type=Path,
        default=_configured_path("VOICE_AGENT_LATENCY_METHODOLOGY"),
        required=_configured_path("VOICE_AGENT_LATENCY_METHODOLOGY") is None,
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=_configured_path("VOICE_AGENT_CERTIFICATION_CONFIG"),
        required=_configured_path("VOICE_AGENT_CERTIFICATION_CONFIG") is None,
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=_configured_path("VOICE_AGENT_CERTIFICATION_AUDIO_ROOT"),
        required=_configured_path("VOICE_AGENT_CERTIFICATION_AUDIO_ROOT") is None,
    )
    parser.add_argument("--journeys", type=Path, default=_DEFAULT_JOURNEYS)
    parser.add_argument(
        "--report",
        type=Path,
        help="new immutable certification result; forbidden for a non-authorizing smoke run",
    )
    parser.add_argument(
        "--smoke-journey",
        help="run one methodology journey without writing certification evidence",
    )
    parser.add_argument(
        "--deployment-id",
        default=os.environ.get("VOICE_AGENT_DEPLOYMENT_ID", "").strip(),
        required=not os.environ.get("VOICE_AGENT_DEPLOYMENT_ID", "").strip(),
    )
    parser.add_argument(
        "--build-artifact-digest",
        default=os.environ.get("VOICE_AGENT_BUILD_ARTIFACT_DIGEST", "").strip(),
        required=not os.environ.get("VOICE_AGENT_BUILD_ARTIFACT_DIGEST", "").strip(),
    )
    parsed = parser.parse_args(arguments)
    if parsed.smoke_journey is None:
        parsed.report = parsed.report or _configured_path("VOICE_AGENT_LATENCY_REPORT")
        if parsed.report is None:
            parser.error(
                "--report or VOICE_AGENT_LATENCY_REPORT is required for full certification"
            )
    elif parsed.report is not None:
        parser.error("--report cannot be used with --smoke-journey")
    return parsed


def _require_livekit_credentials() -> tuple[str, str, str]:
    values = tuple(
        os.environ.get(name, "").strip()
        for name in (
            "LIVEKIT_URL",
            "LIVEKIT_API_KEY",
            "LIVEKIT_API_SECRET",
        )
    )
    if any(not value for value in values):
        raise RuntimeError("LIVEKIT_URL, LIVEKIT_API_KEY, and LIVEKIT_API_SECRET are required")
    return values  # type: ignore[return-value]


def _prepare_audio_assets(
    methodology,
    audio_root: Path,
) -> tuple[Mapping[str, PreparedVoiceAudio], int, int]:
    treatments: list[VoiceAudioTreatment] = []
    for journey in methodology.journeys:
        treatments.extend(journey.setup_audio_treatments)
        assert journey.audio_treatment is not None
        treatments.append(journey.audio_treatment)
    prepared = {
        treatment.asset_sha256: load_voice_audio_asset(audio_root, treatment)
        for treatment in treatments
    }
    formats = {(item.sample_rate_hz, item.channel_count) for item in prepared.values()}
    if len(formats) != 1:
        raise VoiceCertificationProtocolError(
            "single-track voice certification requires one PCM sample rate and channel count"
        )
    sample_rate_hz, channel_count = formats.pop()
    return prepared, sample_rate_hz, channel_count


class _LiveKitVoiceTransport:
    def __init__(
        self,
        *,
        room: rtc.Room,
        client: api.LiveKitAPI,
        target: VoiceCertificationTarget,
        source: rtc.AudioSource,
    ) -> None:
        self._room = room
        self._client = client
        self._target = target
        self._source = source
        self._departures: dict[str, asyncio.Event] = {}
        room.on("participant_disconnected", self._participant_disconnected)

    def _participant_disconnected(self, participant: rtc.RemoteParticipant) -> None:
        event = self._departures.get(participant.identity)
        if event is not None:
            event.set()

    def _dispatch_request(
        self,
        directive: VoiceCertificationDispatchDirective,
    ) -> api.CreateAgentDispatchRequest:
        metadata = VoiceJobMetadata(
            schema_version=1,
            merchant_id=self._target.merchant_id,
            participant_kind="standard",
            participant_identity=self._target.controller_participant_identity,
        )
        return api.CreateAgentDispatchRequest(
            agent_name=self._target.certification_agent_name,
            room=self._target.room_name,
            metadata=metadata.model_dump_json(),
            restart_policy=api.JobRestartPolicy.JRP_NEVER,
            attributes={VOICE_CERTIFICATION_SAMPLE_ATTRIBUTE: directive.model_dump_json()},
        )

    async def create_dispatch(self, directive: VoiceCertificationDispatchDirective) -> str:
        dispatch = await self._client.agent_dispatch.create_dispatch(
            self._dispatch_request(directive)
        )
        return dispatch.id

    async def publish_audio(self, prepared: PreparedVoiceAudio) -> None:
        await publish_prepared_voice_audio(self._source, prepared)

    async def close_dispatch(
        self,
        directive: VoiceCertificationDispatchDirective,
        assignment_id: str | None,
        worker_participant_identity: str | None,
    ) -> None:
        dispatch_ids: list[str]
        participant_identities = set(self._room.remote_participants)
        if worker_participant_identity is not None:
            participant_identities.add(worker_participant_identity)
        duplicate_dispatches = False
        if assignment_id is None:
            expected = self._dispatch_request(directive)
            dispatches = await self._client.agent_dispatch.list_dispatch(self._target.room_name)
            matches = tuple(
                dispatch
                for dispatch in dispatches
                if dispatch.agent_name == expected.agent_name
                and dispatch.room == expected.room
                and dispatch.metadata == expected.metadata
                and dispatch.attributes == expected.attributes
            )
            if not matches:
                if participant_identities:
                    raise VoiceCertificationProtocolError(
                        "voice certification room is not empty after an ambiguous dispatch"
                    )
                return
            duplicate_dispatches = len(matches) != 1
            dispatch_ids = [dispatch.id for dispatch in matches]
            for dispatch in matches:
                participant_identities.update(
                    job.state.participant_identity
                    for job in dispatch.state.jobs
                    if job.state.participant_identity
                )
        else:
            dispatch_ids = [assignment_id]

        departures = {identity: asyncio.Event() for identity in participant_identities}
        self._departures.update(departures)
        try:
            deletion_failures: list[Exception] = []
            for dispatch_id in dispatch_ids:
                try:
                    await self._client.agent_dispatch.delete_dispatch(
                        dispatch_id,
                        self._target.room_name,
                    )
                except Exception as exc:
                    deletion_failures.append(exc)
            if deletion_failures:
                raise ExceptionGroup(
                    "voice certification dispatch deletion failed",
                    deletion_failures,
                )
            for identity, departure in departures.items():
                if identity in self._room.remote_participants:
                    await departure.wait()
            if self._room.remote_participants:
                raise VoiceCertificationProtocolError(
                    "voice certification room retained a participant after dispatch cleanup"
                )
            if duplicate_dispatches:
                raise VoiceCertificationProtocolError(
                    "voice certification found duplicate dispatches for one sample"
                )
        finally:
            for identity, departure in departures.items():
                if self._departures.get(identity) is departure:
                    del self._departures[identity]


async def _close_controller_resources(
    room: rtc.Room,
    client: api.LiveKitAPI,
    publication: rtc.LocalTrackPublication | None,
) -> BaseException | None:
    failures: list[BaseException] = []
    if publication is not None:
        try:
            await room.local_participant.unpublish_track(publication.sid)
        except BaseException as exc:
            failures.append(exc)
    try:
        await room.disconnect()
    except BaseException as exc:
        failures.append(exc)
    try:
        await client.aclose()
    except BaseException as exc:
        failures.append(exc)
    if not failures:
        return None
    if len(failures) == 1:
        return failures[0]
    return BaseExceptionGroup("voice certification controller cleanup failed", failures)


async def _persist_controller_outcome(
    path: Path,
    run: DurableLatencyCertificationRun,
    cleanup_failure: BaseException | None,
) -> DurableLatencyCertificationRun:
    retained = (
        abort_voice_controller_cleanup(run, cleanup_failure) if cleanup_failure is not None else run
    )
    await asyncio.to_thread(write_latency_certification_run, path, retained)
    loaded = await asyncio.to_thread(load_latency_certification_run, path)
    if loaded != retained:
        raise VoiceCertificationProtocolError(
            "written voice certification evidence did not reload identically"
        )
    return retained


async def _run(arguments: argparse.Namespace) -> int:
    livekit_url, api_key, api_secret = _require_livekit_credentials()
    methodology = load_latency_methodology(arguments.methodology)
    if (
        methodology.environment is not LatencyEnvironment.DEPLOYMENT
        or methodology.schema_version != "5"
        or methodology.measurement_surface is not LatencyMeasurementSurface.VOICE_PROCESSING
        or methodology.transport_surface is not VoiceTransportSurface.STANDARD
    ):
        raise ValueError("voice certification requires deployment schema-5 standard methodology")

    target = load_voice_certification_target(arguments.target)
    corpus = load_latency_journey_corpus(arguments.journeys)
    bind_latency_journey_contracts(methodology, corpus)
    if target.merchant_id != corpus.merchant_id:
        raise ValueError("voice certification target and journey corpus use different merchants")
    platform_config = load_platform_runtime_config(arguments.platform_config)
    secrets = EnvSecretResolver()
    application_dsn = secrets.resolve(platform_config.database.application_dsn_ref.uri)
    runtime_fingerprint = deployment_runtime_contract_fingerprint(
        platform_config,
        application_dsn=application_dsn,
    )
    if methodology.runtime_contract_fingerprint != runtime_fingerprint:
        raise ValueError("voice methodology does not match the deployment runtime")
    resolved = ConfigRegistry(_CONFIG_ROOT).load().get(target.merchant_id)
    credentials = load_provider_credentials(_CONFIG_ROOT / "base" / "providers.yaml")
    routing_method = LLMGateway(credentials, secrets).structured_output_method(
        resolved.config.llm.routing
    )
    routing_release_evidence = load_semantic_routing_release_evidence(
        _CONFIG_ROOT / "qualification" / "semantic_routing_release.json"
    )
    application_contract = VoiceApplicationContract(
        durable_platform_fingerprint=runtime_fingerprint,
        build_artifact_digest=arguments.build_artifact_digest,
        tenant_config_version=resolved.config_version,
        semantic_routing=semantic_routing_runtime_contract(
            build_frontline_capability_registry(),
            selection=resolved.config.llm.routing,
            structured_output_method=routing_method,
            timeout_seconds=resolved.config.runtime.semantic_router_timeout_seconds,
            input_max_chars=resolved.config.runtime.semantic_router_input_max_chars,
            corpus_fingerprint=semantic_routing_corpus_fingerprint(_CONFIG_ROOT),
            qualification_evidence_fingerprint=(
                semantic_routing_release_evidence_fingerprint(routing_release_evidence)
            ),
        ),
        certification_target_fingerprint=voice_certification_target_fingerprint(target),
    )
    require_voice_application_contract(methodology, application_contract)
    assets, sample_rate_hz, channel_count = _prepare_audio_assets(
        methodology,
        arguments.audio_root,
    )

    room = rtc.Room()
    client = api.LiveKitAPI(livekit_url, api_key, api_secret)
    publication: rtc.LocalTrackPublication | None = None
    run: DurableLatencyCertificationRun | None = None
    smoke_observation: LatencyObservation | None = None
    execution_failure: BaseException | None = None
    try:
        token = (
            api.AccessToken(api_key, api_secret)
            .with_identity(target.controller_participant_identity)
            .with_grants(
                api.VideoGrants(
                    room_join=True,
                    room=target.room_name,
                    can_publish=True,
                    can_subscribe=True,
                )
            )
            .to_jwt()
        )
        await room.connect(
            livekit_url,
            token,
            rtc.RoomOptions(connect_timeout=methodology.sample_timeout_seconds),
        )
        controller = VoiceCertificationController(
            methodology=methodology,
            run_id=f"voice-{uuid.uuid4().hex}",
            deployment_id=arguments.deployment_id,
            room_id=await room.sid,
            controller_identity=target.controller_participant_identity,
        )
        register_voice_certification_controller(room.local_participant, controller)
        source = rtc.AudioSource(sample_rate_hz, channel_count)
        track = rtc.LocalAudioTrack.create_audio_track("voice-certification-input", source)
        publication = await room.local_participant.publish_track(
            track,
            rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
        )
        transport = _LiveKitVoiceTransport(
            room=room,
            client=client,
            target=target,
            source=source,
        )
        if arguments.smoke_journey is None:
            run = await run_voice_certification_controller(
                controller,
                transport,
                assets,
                run_at=datetime.now(tz=UTC),
            )
        else:
            smoke_observation = await run_voice_certification_smoke(
                controller,
                transport,
                assets,
                journey_id=arguments.smoke_journey,
                run_at=datetime.now(tz=UTC),
            )
    except BaseException as exc:
        execution_failure = exc

    cleanup_failure = await _close_controller_resources(room, client, publication)
    if run is not None:
        assert arguments.report is not None
        run = await _persist_controller_outcome(arguments.report, run, cleanup_failure)

    if execution_failure is not None:
        if cleanup_failure is not None:
            raise execution_failure from cleanup_failure
        raise execution_failure
    if cleanup_failure is not None:
        raise cleanup_failure
    if arguments.smoke_journey is not None:
        if smoke_observation is None:
            raise RuntimeError("voice smoke produced no observation")
        logger.info(
            "non-authorizing voice smoke passed for %s; no evidence was written",
            arguments.smoke_journey,
        )
        return 0
    if run is None:
        raise RuntimeError("voice certification produced no run evidence")

    if run.outcome is LatencyCertificationOutcome.ABORTED:
        assert arguments.report is not None
        logger.error("voice certification aborted; evidence written to %s", arguments.report)
        return 1
    assert run.report is not None
    try:
        require_deployment_latency_report(
            run.report,
            expected_methodology_fingerprint=run.methodology_fingerprint,
            expected_deployment_id=arguments.deployment_id,
        )
    except Exception:
        assert arguments.report is not None
        logger.exception("voice latency gate failed; evidence written to %s", arguments.report)
        return 1
    assert arguments.report is not None
    logger.info("voice latency gate passed; evidence written to %s", arguments.report)
    return 0


def main() -> int:
    load_dotenv(_ROOT / ".env")
    logging.basicConfig(level=logging.INFO)
    return asyncio.run(_run(_arguments()))


if __name__ == "__main__":
    raise SystemExit(main())
