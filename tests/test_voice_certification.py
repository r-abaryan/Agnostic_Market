"""Isolation contracts for the explicitly launched voice-certification worker."""

from __future__ import annotations

import asyncio
import hashlib
import wave
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from livekit import api, rtc
from livekit.protocol import agent, models
from pydantic import ValidationError

from agnostic_market.agents.routing_activation import SemanticRoutingRuntimeContract
from agnostic_market.dtos.events import InterruptEvent, SpokenMessageEvent
from agnostic_market.durability.latency import (
    DurabilityComponentObservation,
    DurableLatencyActivationError,
    DurableLatencyMethodology,
    LatencyAbortStage,
    LatencyCartLine,
    LatencyCertificationOutcome,
    LatencyEnvironment,
    LatencyJourney,
    LatencyJourneyContract,
    LatencyMeasurementSurface,
    LatencyObservation,
    LatencyObservationOutcome,
    LatencyPhase,
    LatencyTier,
    ThermalState,
    VoiceApplicationContract,
    VoiceAudioTreatment,
    VoiceLatencyMetrics,
    VoiceTransportSurface,
    require_voice_application_contract,
)
from agnostic_market.durability.timing import (
    DurabilityOperation,
    DurabilityTimingOutcome,
    DurabilityTimingSample,
)
from agnostic_market.durability.voice_certification import (
    VOICE_CERTIFICATION_PROGRESS_RPC,
    VOICE_CERTIFICATION_READY_RPC,
    VOICE_CERTIFICATION_RESULT_RPC,
    VOICE_CERTIFICATION_SAMPLE_ATTRIBUTE,
    PreparedVoiceAudio,
    VoiceCertificationApplicationSnapshot,
    VoiceCertificationController,
    VoiceCertificationControllerTransport,
    VoiceCertificationDispatchDirective,
    VoiceCertificationJobBinding,
    VoiceCertificationMeasurements,
    VoiceCertificationProgressAck,
    VoiceCertificationProgressEnvelope,
    VoiceCertificationProtocolError,
    VoiceCertificationReadyAck,
    VoiceCertificationSampleAssignment,
    VoiceCertificationSampleEnvelope,
    VoiceCertificationTarget,
    VoiceCertificationTargetError,
    abort_voice_controller_cleanup,
    announce_voice_certification_progress,
    announce_voice_certification_ready,
    answer_certification_request,
    build_voice_certification_schedule,
    build_voice_certification_warmup_schedule,
    build_voice_turn_observation,
    load_voice_audio_asset,
    load_voice_certification_target,
    load_voice_sample_assignment,
    publish_prepared_voice_audio,
    register_voice_certification_controller,
    run_voice_certification_controller,
    run_voice_certification_smoke,
    submit_voice_certification_sample,
    validate_certification_admission,
    validate_certification_job,
    voice_certification_target_fingerprint,
)
from agnostic_market.voice.admission import VoiceJobMetadata
from agnostic_market.voice.pipeline import TurnLatencyMeasurement


def _target() -> VoiceCertificationTarget:
    return VoiceCertificationTarget(
        schema_version=1,
        environment="synthetic",
        production_agent_name="agnostic-market",
        certification_agent_name="agnostic-market-certification",
        room_name="certification-room-01",
        controller_participant_identity="certification-controller",
        merchant_id="acme_store",
    )


def _job(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "agent_name": "agnostic-market-certification",
        "room": SimpleNamespace(name="certification-room-01"),
        # Explicit room dispatches are JT_ROOM jobs. LiveKit populates
        # Job.participant only for publisher-scoped jobs.
        "participant": None,
        "attributes": {
            VOICE_CERTIFICATION_SAMPLE_ATTRIBUTE: _controller()
            .directive("sample-startup-0001")
            .model_dump_json()
        },
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _application_contract() -> VoiceApplicationContract:
    return VoiceApplicationContract(
        durable_platform_fingerprint="2" * 64,
        build_artifact_digest=f"sha256:{'6' * 64}",
        tenant_config_version="7" * 64,
        semantic_routing=SemanticRoutingRuntimeContract(
            provider="fake",
            model="router",
            reasoning_effort=None,
            structured_output_method="function_calling",
            route_schema_fingerprint="route-schema",
            prompt_fingerprint="prompt",
            registry_fingerprint="registry",
            projector_version="projector",
            input_max_chars=4_000,
            timeout_seconds=2.0,
            corpus_fingerprint="8" * 64,
            qualification_evidence_fingerprint="a" * 64,
        ),
        certification_target_fingerprint="9" * 64,
    )


def _methodology() -> DurableLatencyMethodology:
    return DurableLatencyMethodology(
        schema_version="5",
        environment=LatencyEnvironment.DEPLOYMENT,
        backend_location="deployment-region-a",
        journey_corpus_fingerprint="1" * 64,
        runtime_contract_fingerprint="2" * 64,
        application_contract=_application_contract(),
        measurement_surface=LatencyMeasurementSurface.VOICE_PROCESSING,
        transport_surface=VoiceTransportSurface.STANDARD,
        result_rpc_timeout_seconds=3.0,
        result_rpc_retries=1,
        result_rpc_retry_backoff_seconds=0.01,
        job_ready_timeout_seconds=1.0,
        dispatch_cleanup_timeout_seconds=1.0,
        startup_treatment="fresh_job_resources",
        concurrency=1,
        warmup_runs=1,
        startup_samples=20,
        startup_p95_limit_seconds=1.5,
        required_startup_operations=(
            DurabilityOperation.POOL_OPEN,
            DurabilityOperation.REGISTRY_CHECKPOINT_GENERATIONS,
            DurabilityOperation.CHECKPOINT_READ,
        ),
        samples_per_journey=20,
        sample_timeout_seconds=10.0,
        reported_statistics=("p50", "p95", "maximum"),
        journeys=(
            LatencyJourney(
                journey_id="simple-cart-read",
                journey_contract_fingerprint="3" * 64,
                tier=LatencyTier.SIMPLE,
                required_operations=(DurabilityOperation.CHECKPOINT_READ,),
                audio_treatment=VoiceAudioTreatment(
                    asset_path="assets/audio/latency/simple-cart-read.wav",
                    asset_sha256="3" * 64,
                    container="wav",
                    pcm_format="s16le",
                    channel_count=1,
                    sample_rate_hz=16_000,
                    playback_rate=1.0,
                    pre_speech_silence_seconds=0.25,
                    post_speech_silence_seconds=0.5,
                ),
            ),
            LatencyJourney(
                journey_id="commerce-cart-confirmation",
                journey_contract_fingerprint="4" * 64,
                tier=LatencyTier.NON_CHECKOUT_COMMERCE,
                required_operations=(DurabilityOperation.CHECKPOINT_READ,),
                audio_treatment=VoiceAudioTreatment(
                    asset_path="assets/audio/latency/commerce-cart-confirmation.wav",
                    asset_sha256="4" * 64,
                    container="wav",
                    pcm_format="s16le",
                    channel_count=1,
                    sample_rate_hz=16_000,
                    playback_rate=1.0,
                    pre_speech_silence_seconds=0.25,
                    post_speech_silence_seconds=0.5,
                ),
            ),
            LatencyJourney(
                journey_id="checkout-placement-readback",
                journey_contract_fingerprint="5" * 64,
                tier=LatencyTier.CHECKOUT,
                required_operations=(DurabilityOperation.CHECKPOINT_READ,),
                setup_audio_treatments=(
                    VoiceAudioTreatment(
                        asset_path="assets/audio/latency/checkout-setup-add.wav",
                        asset_sha256="a" * 64,
                        container="wav",
                        pcm_format="s16le",
                        channel_count=1,
                        sample_rate_hz=16_000,
                        playback_rate=1.0,
                        pre_speech_silence_seconds=0.25,
                        post_speech_silence_seconds=0.5,
                    ),
                    VoiceAudioTreatment(
                        asset_path="assets/audio/latency/checkout-setup-confirm.wav",
                        asset_sha256="b" * 64,
                        container="wav",
                        pcm_format="s16le",
                        channel_count=1,
                        sample_rate_hz=16_000,
                        playback_rate=1.0,
                        pre_speech_silence_seconds=0.25,
                        post_speech_silence_seconds=0.5,
                    ),
                ),
                audio_treatment=VoiceAudioTreatment(
                    asset_path="assets/audio/latency/checkout-placement-readback.wav",
                    asset_sha256="5" * 64,
                    container="wav",
                    pcm_format="s16le",
                    channel_count=1,
                    sample_rate_hz=16_000,
                    playback_rate=1.0,
                    pre_speech_silence_seconds=0.25,
                    post_speech_silence_seconds=0.5,
                ),
            ),
        ),
    )


def _binding(sample_id: str) -> VoiceCertificationJobBinding:
    return VoiceCertificationJobBinding(
        sample_id=sample_id,
        room_id="RM_certification_01",
        assignment_id=f"dispatch-{sample_id}",
        job_id=f"job-{sample_id}",
        worker_id=f"worker-{sample_id}",
        worker_participant_identity=f"agent-{sample_id}",
    )


def _contract(journey_id: str) -> LatencyJourneyContract:
    if journey_id == "checkout-placement-readback":
        return LatencyJourneyContract(
            journey_id=journey_id,
            setup_turns=("add one waterproof rain jacket to my cart", "yes"),
            utterance="place my order",
            initial_cart=(LatencyCartLine(sku="SKU-BLU-07", quantity=1),),
            expected_cart=(LatencyCartLine(sku="SKU-BLU-07", quantity=1),),
            expected_event_kind="interrupt",
            expected_event_text_contains=("$129.00",),
        )
    return LatencyJourneyContract(
        journey_id=journey_id,
        utterance="what is in my cart?",
        initial_cart=(),
        expected_cart=(),
        expected_event_kind="spoken_message",
        expected_event_node="cart_view_render",
        expected_event_text_contains=("empty",),
    )


def _state(
    *cart: LatencyCartLine,
    placed_order_count: int = 0,
) -> VoiceCertificationApplicationSnapshot:
    return VoiceCertificationApplicationSnapshot(
        cart=cart,
        placed_order_count=placed_order_count,
    )


def _assigned_job(directive: VoiceCertificationDispatchDirective) -> agent.Job:
    return agent.Job(
        id="AJ_job_01",
        dispatch_id="AD_dispatch_01",
        room=models.Room(name="certification-room-01", sid="RM_certification_01"),
        agent_name="agnostic-market-certification",
        attributes={
            VOICE_CERTIFICATION_SAMPLE_ATTRIBUTE: directive.model_dump_json(),
        },
    )


def _controller(
    methodology: DurableLatencyMethodology | None = None,
) -> VoiceCertificationController:
    return VoiceCertificationController(
        methodology=methodology or _methodology(),
        run_id="voice-certification-run-01",
        deployment_id="deployment-a",
        room_id="RM_certification_01",
        controller_identity="certification-controller",
    )


def _observation(sample_id: str) -> LatencyObservation:
    if sample_id.startswith("sample-startup-"):
        return LatencyObservation(
            schema_version="3",
            sample_id=sample_id,
            phase=LatencyPhase.STARTUP,
            thermal_state=ThermalState.COLD,
            elapsed_seconds=0.5,
            components=tuple(
                DurabilityComponentObservation(
                    operation=operation,
                    elapsed_seconds=0.05,
                    outcome=DurabilityTimingOutcome.SUCCESS,
                )
                for operation in _methodology().required_startup_operations
            ),
        )
    journey_id = sample_id.removeprefix("sample-turn-").rsplit("-", 1)[0]
    journey = next(item for item in _methodology().journeys if item.journey_id == journey_id)
    return LatencyObservation(
        schema_version="3",
        sample_id=sample_id,
        phase=LatencyPhase.TURN,
        thermal_state=ThermalState.WARM,
        elapsed_seconds=0.5,
        journey_id=journey_id,
        tier=journey.tier,
        voice_metrics=VoiceLatencyMetrics(
            end_to_end_seconds=0.75,
            endpointing_seconds=0.25,
            processing_seconds=0.5,
        ),
        components=(
            DurabilityComponentObservation(
                operation=DurabilityOperation.CHECKPOINT_READ,
                elapsed_seconds=0.05,
                outcome=DurabilityTimingOutcome.SUCCESS,
            ),
        ),
    )


def _envelope(
    controller: VoiceCertificationController,
    binding: VoiceCertificationJobBinding,
) -> VoiceCertificationSampleEnvelope:
    return VoiceCertificationSampleEnvelope.from_assignment(
        controller.assignment(binding.sample_id),
        _observation(binding.sample_id),
    )


class _JobRequest:
    def __init__(self, job: SimpleNamespace) -> None:
        self.job = job
        self.accepted = False
        self.rejected = False
        self.terminate: bool | None = None

    async def accept(self) -> None:
        self.accepted = True

    async def reject(self, *, terminate: bool = True) -> None:
        self.rejected = True
        self.terminate = terminate


def test_certification_target_reserves_a_disjoint_worker_namespace() -> None:
    target = _target()
    assert target.certification_agent_name == f"{target.production_agent_name}-certification"

    with pytest.raises(ValidationError, match="reserved suffix"):
        VoiceCertificationTarget.model_validate(
            target.model_dump() | {"certification_agent_name": target.production_agent_name}
        )
    with pytest.raises(ValidationError, match="reserved certification namespace"):
        VoiceCertificationTarget.model_validate(
            target.model_dump()
            | {
                "production_agent_name": "agnostic-market-certification",
                "certification_agent_name": "agnostic-market-certification-certification",
            }
        )
    with pytest.raises(ValidationError, match="environment"):
        VoiceCertificationTarget.model_validate(target.model_dump() | {"environment": "live"})


def test_certification_target_loader_is_strict(tmp_path: Path) -> None:
    path = tmp_path / "voice-certification.yaml"
    path.write_text(
        "schema_version: 1\n"
        "environment: synthetic\n"
        "production_agent_name: agnostic-market\n"
        "certification_agent_name: agnostic-market-certification\n"
        "room_name: certification-room-01\n"
        "controller_participant_identity: certification-controller\n"
        "merchant_id: acme_store\n",
        encoding="utf-8",
    )
    assert load_voice_certification_target(path) == _target()

    path.write_text(path.read_text(encoding="utf-8") + "unknown: true\n", encoding="utf-8")
    with pytest.raises(VoiceCertificationTargetError, match="cannot be loaded"):
        load_voice_certification_target(path)


@pytest.mark.parametrize(
    "job",
    (
        _job(agent_name="agnostic-market"),
        _job(room=SimpleNamespace(name="other-room")),
        _job(
            participant=SimpleNamespace(
                identity="other-controller",
                kind=models.ParticipantInfo.STANDARD,
            )
        ),
        _job(
            participant=SimpleNamespace(
                identity="certification-controller",
                kind=models.ParticipantInfo.SIP,
            )
        ),
        _job(attributes={}),
        _job(attributes={VOICE_CERTIFICATION_SAMPLE_ATTRIBUTE: "not-json"}),
    ),
)
async def test_certification_request_rejects_every_foreign_job(job: SimpleNamespace) -> None:
    request = _JobRequest(job)

    accepted = await answer_certification_request(request, _target())  # type: ignore[arg-type]

    assert not accepted
    assert request.rejected
    assert request.terminate is True
    assert not request.accepted


async def test_certification_request_accepts_only_the_exact_server_observed_job() -> None:
    request = _JobRequest(_job())

    accepted = await answer_certification_request(request, _target())  # type: ignore[arg-type]

    assert accepted
    assert request.accepted
    assert not request.rejected


async def test_certification_request_accepts_room_dispatch_without_job_participant() -> None:
    request = _JobRequest(_job(participant=None))

    accepted = await answer_certification_request(request, _target())  # type: ignore[arg-type]

    assert accepted
    assert request.accepted


async def test_dedicated_worker_requires_its_target_and_distinct_agent_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import durable_voice_certification_worker

    path = tmp_path / "voice-certification.yaml"
    path.write_text(_target().model_dump_json(), encoding="utf-8")
    monkeypatch.setenv("VOICE_AGENT_CERTIFICATION_CONFIG", str(path))

    options = durable_voice_certification_worker._worker_options(("dev",))
    assert options.agent_name == _target().certification_agent_name
    request = _JobRequest(_job())
    await options.request_fnc(request)  # type: ignore[arg-type]
    assert request.accepted

    from scripts import voice_agent

    assert voice_agent._agent_name(("dev",)) == _target().production_agent_name


async def test_livekit_controller_dispatch_carries_exact_admission_and_sample_authority() -> None:
    from scripts import durable_voice_certification

    class DispatchService:
        def __init__(self) -> None:
            self.request: object | None = None

        async def create_dispatch(self, request: object) -> SimpleNamespace:
            self.request = request
            return SimpleNamespace(id="AD_server_created")

        async def delete_dispatch(self, _dispatch_id: str, _room_name: str) -> None:
            return None

    class Room:
        def __init__(self) -> None:
            self.remote_participants: dict[str, object] = {}

        def on(self, event: str, _callback: object) -> None:
            assert event == "participant_disconnected"

    service = DispatchService()
    client = SimpleNamespace(agent_dispatch=service)
    transport = durable_voice_certification._LiveKitVoiceTransport(  # type: ignore[attr-defined]
        room=cast(rtc.Room, Room()),
        client=client,  # type: ignore[arg-type]
        target=_target(),
        source=cast(rtc.AudioSource, object()),
    )
    directive = _controller().directive("sample-startup-0001")

    dispatch_id = await transport.create_dispatch(directive)

    assert dispatch_id == "AD_server_created"
    assert service.request is not None
    request = cast(api.CreateAgentDispatchRequest, service.request)
    assert request.agent_name == _target().certification_agent_name
    assert request.room == _target().room_name
    assert request.restart_policy == api.JobRestartPolicy.JRP_NEVER
    assert request.attributes == {VOICE_CERTIFICATION_SAMPLE_ATTRIBUTE: directive.model_dump_json()}
    assert VoiceJobMetadata.model_validate_json(request.metadata) == VoiceJobMetadata(
        schema_version=1,
        merchant_id=_target().merchant_id,
        participant_kind="standard",
        participant_identity=_target().controller_participant_identity,
    )


async def test_livekit_transport_reconciles_and_deletes_an_ambiguously_created_dispatch() -> None:
    from scripts import durable_voice_certification

    directive = _controller().directive("sample-startup-0001")
    expected_metadata = VoiceJobMetadata(
        schema_version=1,
        merchant_id=_target().merchant_id,
        participant_kind="standard",
        participant_identity=_target().controller_participant_identity,
    ).model_dump_json()

    class DispatchService:
        def __init__(self) -> None:
            self.deleted: list[tuple[str, str]] = []

        async def list_dispatch(self, room_name: str) -> list[api.AgentDispatch]:
            assert room_name == _target().room_name
            return [
                api.AgentDispatch(
                    id="AD_ambiguous",
                    agent_name=_target().certification_agent_name,
                    room=_target().room_name,
                    metadata=expected_metadata,
                    attributes={VOICE_CERTIFICATION_SAMPLE_ATTRIBUTE: directive.model_dump_json()},
                )
            ]

        async def delete_dispatch(self, dispatch_id: str, room_name: str) -> None:
            self.deleted.append((dispatch_id, room_name))

    class Room:
        def __init__(self) -> None:
            self.remote_participants: dict[str, object] = {}

        def on(self, event: str, _callback: object) -> None:
            assert event == "participant_disconnected"

    service = DispatchService()
    transport = durable_voice_certification._LiveKitVoiceTransport(  # type: ignore[attr-defined]
        room=cast(rtc.Room, Room()),
        client=SimpleNamespace(agent_dispatch=service),  # type: ignore[arg-type]
        target=_target(),
        source=cast(rtc.AudioSource, object()),
    )

    await transport.close_dispatch(directive, None, None)

    assert service.deleted == [("AD_ambiguous", _target().room_name)]


async def test_livekit_transport_waits_for_each_reused_participant_identity_departure() -> None:
    from scripts import durable_voice_certification

    class DispatchService:
        def __init__(self) -> None:
            self.deleted: asyncio.Queue[str] = asyncio.Queue()

        async def delete_dispatch(self, dispatch_id: str, room_name: str) -> None:
            assert room_name == _target().room_name
            await self.deleted.put(dispatch_id)

    class Room:
        def __init__(self) -> None:
            self.remote_participants: dict[str, object] = {}
            self.disconnected = None

        def on(self, event: str, callback: object) -> None:
            assert event == "participant_disconnected"
            self.disconnected = callback

    room = Room()
    service = DispatchService()
    transport = durable_voice_certification._LiveKitVoiceTransport(  # type: ignore[attr-defined]
        room=cast(rtc.Room, room),
        client=SimpleNamespace(agent_dispatch=service),  # type: ignore[arg-type]
        target=_target(),
        source=cast(rtc.AudioSource, object()),
    )
    directive = _controller().directive("sample-startup-0001")

    for dispatch_id in ("AD_first", "AD_second"):
        participant = SimpleNamespace(identity="reused-worker")
        room.remote_participants[participant.identity] = participant
        cleanup = asyncio.create_task(
            transport.close_dispatch(directive, dispatch_id, participant.identity)
        )
        assert await service.deleted.get() == dispatch_id
        assert not cleanup.done()

        del room.remote_participants[participant.identity]
        disconnected = cast(Callable[[object], None], room.disconnected)
        disconnected(participant)
        await cleanup


@pytest.mark.parametrize("cancel_phase", ["dispatch-deletion", "participant-departure"])
async def test_livekit_cleanup_defers_cancellation_and_clears_departure_state(
    cancel_phase: str,
) -> None:
    from agnostic_market.durability import voice_certification
    from scripts import durable_voice_certification

    class DispatchService:
        def __init__(self) -> None:
            self.delete_started = asyncio.Event()
            self.release_delete = asyncio.Event()

        async def delete_dispatch(self, dispatch_id: str, room_name: str) -> None:
            assert dispatch_id == "AD_cleanup"
            assert room_name == _target().room_name
            self.delete_started.set()
            await self.release_delete.wait()

    class Room:
        def __init__(self) -> None:
            participant = SimpleNamespace(identity="worker-cleanup")
            self.remote_participants = {participant.identity: participant}
            self.disconnected = None
            self.departure_observer_ready = asyncio.Event()

        def on(self, event: str, callback: object) -> None:
            assert event == "participant_disconnected"
            self.disconnected = callback
            self.departure_observer_ready.set()

    room = Room()
    service = DispatchService()
    transport = durable_voice_certification._LiveKitVoiceTransport(  # type: ignore[attr-defined]
        room=cast(rtc.Room, room),
        client=SimpleNamespace(agent_dispatch=service),  # type: ignore[arg-type]
        target=_target(),
        source=cast(rtc.AudioSource, object()),
    )
    directive = _controller().directive("sample-startup-0001")
    cleanup = asyncio.create_task(
        transport.close_dispatch(directive, "AD_cleanup", "worker-cleanup")
    )
    finisher = asyncio.create_task(
        voice_certification._finish_voice_cleanup(  # type: ignore[attr-defined]
            cleanup,
            timeout_seconds=1.0,
        )
    )
    await service.delete_started.wait()
    await asyncio.sleep(0)
    if cancel_phase == "dispatch-deletion":
        finisher.cancel()
    service.release_delete.set()
    await asyncio.wait_for(room.departure_observer_ready.wait(), timeout=0.5)
    if cancel_phase == "participant-departure":
        finisher.cancel()
    participant = room.remote_participants.pop("worker-cleanup")
    disconnected = cast(Callable[[object], None], room.disconnected)
    disconnected(participant)

    deferred = await finisher
    assert isinstance(deferred, asyncio.CancelledError)
    assert cleanup.done()
    assert transport._departures == {}  # type: ignore[attr-defined]


async def test_controller_resource_cleanup_attempts_every_owned_resource() -> None:
    from scripts import durable_voice_certification

    calls: list[str] = []

    class LocalParticipant:
        async def unpublish_track(self, sid: str) -> None:
            calls.append(f"unpublish:{sid}")
            raise RuntimeError("track cleanup failed")

    class Room:
        local_participant = LocalParticipant()

        async def disconnect(self) -> None:
            calls.append("disconnect")
            raise ValueError("room cleanup failed")

    class Client:
        async def aclose(self) -> None:
            calls.append("client-close")

    failure = await durable_voice_certification._close_controller_resources(  # type: ignore[attr-defined]
        cast(rtc.Room, Room()),
        cast(api.LiveKitAPI, Client()),
        cast(rtc.LocalTrackPublication, SimpleNamespace(sid="TR_certification")),
    )

    assert calls == ["unpublish:TR_certification", "disconnect", "client-close"]
    assert isinstance(failure, BaseExceptionGroup)
    assert [type(item).__name__ for item in failure.exceptions] == ["RuntimeError", "ValueError"]


def test_voice_smoke_cli_cannot_write_a_certification_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import durable_voice_certification

    monkeypatch.setenv("VOICE_AGENT_LATENCY_REPORT", "ambient-certification-result.json")
    common = (
        "--platform-config",
        "platform.yaml",
        "--methodology",
        "methodology.yaml",
        "--target",
        "target.yaml",
        "--audio-root",
        "audio",
        "--deployment-id",
        "deployment-a",
        "--build-artifact-digest",
        f"sha256:{'6' * 64}",
    )

    arguments = durable_voice_certification._arguments(  # type: ignore[attr-defined]
        (*common, "--smoke-journey", "simple-cart-read")
    )

    assert arguments.smoke_journey == "simple-cart-read"
    assert arguments.report is None
    with pytest.raises(SystemExit):
        durable_voice_certification._arguments(  # type: ignore[attr-defined]
            (
                *common,
                "--smoke-journey",
                "simple-cart-read",
                "--report",
                "must-not-exist.json",
            )
        )


def test_dedicated_worker_help_does_not_require_runtime_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import durable_voice_certification_worker

    monkeypatch.delenv("VOICE_AGENT_CERTIFICATION_CONFIG", raising=False)

    options = durable_voice_certification_worker._worker_options(("--help",))

    assert options.agent_name == ""


def test_assigned_job_and_normal_admission_are_both_bound_to_the_target() -> None:
    target = _target()
    validate_certification_job(_job(), target)  # type: ignore[arg-type]
    validate_certification_admission(
        target,
        merchant_id="acme_store",
        participant_kind="standard",
        participant_identity="certification-controller",
    )

    with pytest.raises(VoiceCertificationTargetError, match="merchant"):
        validate_certification_admission(
            target,
            merchant_id="demo_shop",
            participant_kind="standard",
            participant_identity="certification-controller",
        )
    with pytest.raises(VoiceCertificationTargetError, match="standard participant"):
        validate_certification_admission(
            target,
            merchant_id="acme_store",
            participant_kind="sip",
            participant_identity=None,
        )


def test_voice_schedule_is_exactly_derived_from_the_frozen_methodology() -> None:
    schedule = build_voice_certification_schedule(_methodology())
    warmups = build_voice_certification_warmup_schedule(_methodology())

    assert len(schedule) == 80
    assert schedule[0].sample_id == "sample-startup-0001"
    assert schedule[19].sample_id == "sample-startup-0020"
    assert schedule[20].sample_id == "sample-turn-simple-cart-read-0001"
    assert schedule[-1].sample_id == "sample-turn-checkout-placement-readback-0020"
    assert schedule[0].phase is LatencyPhase.STARTUP
    assert schedule[0].journey_id is None
    assert schedule[20].phase is LatencyPhase.TURN
    assert schedule[20].journey_id == "simple-cart-read"
    assert schedule[20].tier is LatencyTier.SIMPLE
    assert len(schedule[-1].setup_audio_treatments) == 2
    assert tuple(sample.sample_id for sample in warmups) == (
        "warmup-turn-simple-cart-read-0001",
        "warmup-turn-commerce-cart-confirmation-0001",
        "warmup-turn-checkout-placement-readback-0001",
    )

    with pytest.raises(VoiceCertificationProtocolError, match="deployment voice-processing"):
        build_voice_certification_schedule(
            _methodology().model_copy(
                update={"measurement_surface": LatencyMeasurementSurface.REASONING_GRAPH}
            )
        )


def test_controller_accepts_only_a_server_bound_sample_and_reuses_exact_ack() -> None:
    controller = _controller()
    sample_id = "sample-startup-0001"
    binding = _binding(sample_id)
    controller.bind_job(binding)
    envelope = _envelope(controller, binding)
    payload = envelope.model_dump_json()
    invocation = rtc.RpcInvocationData(
        request_id="rpc-01",
        caller_identity=binding.worker_participant_identity,
        payload=payload,
        response_timeout=3.0,
    )

    first_ack = controller.handle_rpc(invocation)
    retry_ack = controller.handle_rpc(invocation)

    assert first_ack == retry_ack
    assert controller.received_count == 1
    assert controller.pending_sample_ids[0] == "sample-startup-0002"
    assert controller.observation(sample_id) == envelope.observation

    divergent_payload = envelope.model_copy(
        update={"observation": envelope.observation.model_copy(update={"elapsed_seconds": 0.6})}
    ).model_dump_json()
    with pytest.raises(VoiceCertificationProtocolError, match="divergent duplicate"):
        controller.handle_rpc(
            rtc.RpcInvocationData(
                request_id="rpc-02",
                caller_identity=binding.worker_participant_identity,
                payload=divergent_payload,
                response_timeout=3.0,
            )
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ({"run_id": "other-run"}, "run binding"),
        ({"room_id": "RM_other"}, "job binding"),
        ({"job_id": "other-job"}, "job binding"),
        ({"measurement_surface": LatencyMeasurementSurface.REASONING_GRAPH}, "surface"),
    ),
)
def test_controller_rejects_foreign_or_replayed_envelopes(
    mutation: dict[str, object],
    message: str,
) -> None:
    controller = _controller()
    binding = _binding("sample-startup-0001")
    controller.bind_job(binding)
    envelope = _envelope(controller, binding).model_copy(update=mutation)

    with pytest.raises(VoiceCertificationProtocolError, match=message):
        controller.submit(
            envelope.model_dump_json(),
            caller_identity=binding.worker_participant_identity,
        )


def test_controller_requires_the_authenticated_worker_and_trusted_job_binding() -> None:
    controller = _controller()
    binding = _binding("sample-startup-0001")
    envelope_source = _controller()
    envelope_source.bind_job(binding)
    envelope = _envelope(envelope_source, binding)

    with pytest.raises(VoiceCertificationProtocolError, match="not assigned"):
        controller.submit(
            envelope.model_dump_json(),
            caller_identity=binding.worker_participant_identity,
        )

    controller.bind_job(binding)
    with pytest.raises(VoiceCertificationProtocolError, match="authenticated participant"):
        controller.submit(envelope.model_dump_json(), caller_identity="foreign-agent")

    with pytest.raises(VoiceCertificationProtocolError, match="already bound"):
        controller.bind_job(binding.model_copy(update={"job_id": "other-job"}))


def test_job_assignment_carries_one_complete_controller_owned_binding() -> None:
    controller = _controller()
    binding = _binding("sample-turn-simple-cart-read-0001")
    controller.bind_job(binding)

    assignment = controller.assignment(binding.sample_id)

    assert assignment.sample == controller.schedule[20]
    assert assignment.job == binding
    assert assignment.run_id == controller.run_id
    assert assignment.methodology_fingerprint == controller.methodology_fingerprint
    assert assignment.runtime_contract_fingerprint == controller.runtime_contract_fingerprint
    assert (
        assignment.application_contract_fingerprint == controller.application_contract_fingerprint
    )
    with pytest.raises(ValidationError, match="must match"):
        VoiceCertificationSampleAssignment.model_validate(
            assignment.model_dump()
            | {"job": binding.model_copy(update={"sample_id": "sample-startup-0001"})}
        )


def test_controller_registers_both_versioned_livekit_rpc_methods() -> None:
    class LocalParticipant:
        def __init__(self) -> None:
            self.methods: dict[str, object] = {}

        def register_rpc_method(self, method_name: str, handler: object) -> None:
            self.methods[method_name] = handler

    participant = LocalParticipant()
    controller = _controller()

    register_voice_certification_controller(participant, controller)  # type: ignore[arg-type]

    assert participant.methods == {
        VOICE_CERTIFICATION_READY_RPC: controller.handle_ready_rpc,
        VOICE_CERTIFICATION_PROGRESS_RPC: controller.handle_progress_rpc,
        VOICE_CERTIFICATION_RESULT_RPC: controller.handle_rpc,
    }


def test_controller_rejects_out_of_methodology_observation_shape() -> None:
    controller = _controller()
    binding = _binding("sample-turn-simple-cart-read-0001")
    controller.bind_job(binding)
    envelope = _envelope(controller, binding)
    wrong_tier = envelope.observation.model_copy(update={"tier": LatencyTier.CHECKOUT})

    with pytest.raises(VoiceCertificationProtocolError, match="sample schedule"):
        controller.submit(
            envelope.model_copy(update={"observation": wrong_tier}).model_dump_json(),
            caller_identity=binding.worker_participant_identity,
        )

    unknown = binding.model_copy(update={"sample_id": "sample-turn-unknown-0001"})
    with pytest.raises(VoiceCertificationProtocolError, match="frozen schedule"):
        controller.bind_job(unknown)


def test_controller_records_timeout_and_refuses_late_submission() -> None:
    controller = _controller()
    sample_id = "sample-turn-simple-cart-read-0001"
    binding = _binding(sample_id)
    controller.bind_job(binding)

    controller.record_timeout(sample_id)

    observation = controller.observation(sample_id)
    assert observation.outcome is LatencyObservationOutcome.ERROR
    assert observation.error_type == "TimeoutError"
    assert observation.elapsed_seconds is None
    with pytest.raises(VoiceCertificationProtocolError, match="timed out"):
        controller.submit(
            _envelope(controller, binding).model_dump_json(),
            caller_identity=binding.worker_participant_identity,
        )


def test_controller_keeps_an_accepted_result_when_the_watchdog_finishes_late() -> None:
    controller = _controller()
    sample_id = "sample-turn-simple-cart-read-0001"
    binding = _binding(sample_id)
    controller.bind_job(binding)
    envelope = _envelope(controller, binding)
    payload = envelope.model_dump_json()

    acknowledgement = controller.submit(
        payload,
        caller_identity=binding.worker_participant_identity,
    )

    retained = controller.observation(sample_id)
    assert retained == envelope.observation
    assert controller.record_timeout(sample_id) is retained
    assert controller.record_failure(sample_id, RuntimeError("late watchdog")) is retained
    assert controller.received_count == 1
    assert controller.resolved_count == 1
    assert (
        controller.submit(payload, caller_identity=binding.worker_participant_identity)
        == acknowledgement
    )


async def test_controller_turns_a_lost_job_into_typed_timeout_evidence() -> None:
    methodology = _methodology().model_copy(update={"sample_timeout_seconds": 0.001})
    controller = _controller(methodology)
    sample_id = "sample-startup-0001"
    controller.bind_job(_binding(sample_id))

    async with asyncio.timeout(0.5):
        observation = await controller.wait_for_sample(sample_id)

    assert observation == controller.observation(sample_id)
    assert observation.outcome is LatencyObservationOutcome.ERROR
    assert observation.error_type == "TimeoutError"
    assert controller.received_count == 0
    assert controller.resolved_count == 1


async def test_controller_wait_returns_the_submitted_observation_without_retiming_it() -> None:
    controller = _controller()
    binding = _binding("sample-startup-0001")
    controller.bind_job(binding)
    envelope = _envelope(controller, binding)
    controller.submit(
        envelope.model_dump_json(),
        caller_identity=binding.worker_participant_identity,
    )

    observation = await controller.wait_for_sample(binding.sample_id)

    assert observation is controller.observation(binding.sample_id)
    assert observation == envelope.observation


def test_controller_builds_only_a_complete_report_through_the_existing_builder() -> None:
    controller = _controller()
    with pytest.raises(VoiceCertificationProtocolError, match="incomplete"):
        controller.build_report(run_at=datetime(2026, 9, 12, tzinfo=UTC))

    last_binding: VoiceCertificationJobBinding | None = None
    for sample in controller.schedule:
        last_binding = _binding(sample.sample_id)
        controller.bind_job(last_binding)
        controller.submit(
            _envelope(controller, last_binding).model_dump_json(),
            caller_identity=last_binding.worker_participant_identity,
        )
    assert last_binding is not None

    run_at = datetime(2026, 9, 12, tzinfo=UTC)
    run = controller.build_run(run_at=run_at)
    report = run.report

    assert report is not None
    assert report.deployment_id == controller.deployment_id
    assert report.methodology_fingerprint == controller.methodology_fingerprint
    assert report.gate.passed
    assert report.observations == tuple(
        controller.observation(sample.sample_id) for sample in controller.schedule
    )
    assert controller.build_run(run_at=run_at) == run
    cleanup_abort = abort_voice_controller_cleanup(run, RuntimeError("redacted"))
    assert cleanup_abort.outcome is LatencyCertificationOutcome.ABORTED
    assert cleanup_abort.report is None
    assert cleanup_abort.abort is not None
    assert cleanup_abort.abort.stage is LatencyAbortStage.VOICE_CONTROLLER_CLEANUP
    assert cleanup_abort.abort.sample_id is None
    assert cleanup_abort.abort.journey_id is None
    assert cleanup_abort.abort.error_type == "RuntimeError"
    assert controller.submit(
        _envelope(controller, last_binding).model_dump_json(),
        caller_identity=last_binding.worker_participant_identity,
    ) == controller.submit(
        _envelope(controller, last_binding).model_dump_json(),
        caller_identity=last_binding.worker_participant_identity,
    )
    retained = controller.observation("sample-startup-0001")
    assert controller.record_timeout("sample-startup-0001") is retained
    assert controller.record_failure("sample-startup-0001", RuntimeError("late")) is retained


async def test_controller_cleanup_failure_is_persisted_as_an_abort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import durable_voice_certification

    controller = _controller()
    for sample in controller.schedule:
        binding = _binding(sample.sample_id)
        controller.bind_job(binding)
        controller.submit(
            _envelope(controller, binding).model_dump_json(),
            caller_identity=binding.worker_participant_identity,
        )
    completed = controller.build_run(run_at=datetime(2026, 9, 12, tzinfo=UTC))
    written: list[object] = []

    def write_run(path: Path, run: object) -> None:
        assert path == tmp_path / "voice-result.json"
        written.append(run)

    def load_run(path: Path) -> object:
        assert path == tmp_path / "voice-result.json"
        return written[-1]

    monkeypatch.setattr(durable_voice_certification, "write_latency_certification_run", write_run)
    monkeypatch.setattr(durable_voice_certification, "load_latency_certification_run", load_run)

    retained = await durable_voice_certification._persist_controller_outcome(  # type: ignore[attr-defined]
        tmp_path / "voice-result.json",
        completed,
        RuntimeError("controller cleanup failed"),
    )

    assert written == [retained]
    assert retained.outcome is LatencyCertificationOutcome.ABORTED
    assert retained.report is None
    assert retained.abort is not None
    assert retained.abort.stage is LatencyAbortStage.VOICE_CONTROLLER_CLEANUP
    assert retained.abort.error_type == "RuntimeError"


def test_controller_validates_its_operator_owned_run_binding() -> None:
    with pytest.raises(VoiceCertificationProtocolError, match="run binding"):
        VoiceCertificationController(
            methodology=_methodology(),
            run_id="invalid run id",
            deployment_id="deployment-a",
            room_id="RM_certification_01",
            controller_identity="certification-controller",
        )


def test_controller_rejects_historical_schema_4_voice_methodology() -> None:
    historical = _methodology().model_copy(
        update={"schema_version": "4", "application_contract": None}
    )

    with pytest.raises(VoiceCertificationProtocolError, match="schema-5"):
        _controller(historical)


def test_schema_5_methodology_requires_the_observed_startup_operations() -> None:
    payload = _methodology().model_dump()
    payload["required_startup_operations"] = (
        DurabilityOperation.POOL_OPEN,
        DurabilityOperation.REGISTRY_GET,
    )

    with pytest.raises(ValidationError, match="generation lookup and checkpoint read"):
        DurableLatencyMethodology.model_validate(payload)


def test_application_contract_fingerprint_binds_build_tenant_routing_and_target() -> None:
    from agnostic_market.durability.latency import voice_application_contract_fingerprint

    contract = _application_contract()
    baseline = voice_application_contract_fingerprint(contract)
    mutations = (
        contract.model_copy(update={"build_artifact_digest": f"sha256:{'a' * 64}"}),
        contract.model_copy(update={"tenant_config_version": "b" * 64}),
        contract.model_copy(
            update={
                "semantic_routing": contract.semantic_routing.model_copy(
                    update={"model": "different-router"}
                )
            }
        ),
        contract.model_copy(update={"certification_target_fingerprint": "c" * 64}),
    )

    assert all(
        voice_application_contract_fingerprint(mutation) != baseline for mutation in mutations
    )
    changed_target = _target().model_copy(update={"room_name": "another-certification-room"})
    assert voice_certification_target_fingerprint(
        _target()
    ) != voice_certification_target_fingerprint(changed_target)

    with pytest.raises(DurableLatencyActivationError, match="application contract"):
        require_voice_application_contract(_methodology(), mutations[0])


def test_job_assignment_combines_controller_directive_with_server_authority() -> None:
    controller = _controller()
    directive = controller.directive("sample-turn-simple-cart-read-0001")
    job = _assigned_job(directive)

    assignment = load_voice_sample_assignment(
        job,
        _target(),
        expected_deployment_id="deployment-a",
        expected_journey_corpus_fingerprint="1" * 64,
        expected_runtime_contract_fingerprint="2" * 64,
        expected_application_contract=_application_contract(),
        worker_id="AW_worker_01",
        worker_participant_identity="agent-participant-01",
    )

    assert assignment.sample == controller.schedule[20]
    assert assignment.job == VoiceCertificationJobBinding(
        sample_id=assignment.sample.sample_id,
        room_id="RM_certification_01",
        assignment_id="AD_dispatch_01",
        job_id="AJ_job_01",
        worker_id="AW_worker_01",
        worker_participant_identity="agent-participant-01",
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"deployment_id": "deployment-b"}, "deployment"),
        ({"runtime_contract_fingerprint": "9" * 64}, "runtime"),
        ({"application_contract_fingerprint": "0" * 64}, "application"),
    ],
)
def test_job_assignment_rejects_divergent_controller_authority(
    mutation: dict[str, object],
    message: str,
) -> None:
    directive = _controller().directive("sample-startup-0001")
    payload = directive.model_dump() | mutation
    job = _assigned_job(VoiceCertificationDispatchDirective.model_validate(payload))

    with pytest.raises(VoiceCertificationProtocolError, match=message):
        load_voice_sample_assignment(
            job,
            _target(),
            expected_deployment_id="deployment-a",
            expected_journey_corpus_fingerprint="1" * 64,
            expected_runtime_contract_fingerprint="2" * 64,
            expected_application_contract=_application_contract(),
            worker_id="AW_worker_01",
            worker_participant_identity="agent-participant-01",
        )


def test_job_assignment_requires_one_valid_server_delivered_directive() -> None:
    job = _assigned_job(_controller().directive("sample-startup-0001"))
    del job.attributes[VOICE_CERTIFICATION_SAMPLE_ATTRIBUTE]

    with pytest.raises(VoiceCertificationProtocolError, match="directive"):
        load_voice_sample_assignment(
            job,
            _target(),
            expected_deployment_id="deployment-a",
            expected_journey_corpus_fingerprint="1" * 64,
            expected_runtime_contract_fingerprint="2" * 64,
            expected_application_contract=_application_contract(),
            worker_id="AW_worker_01",
            worker_participant_identity="agent-participant-01",
        )


def test_voice_audio_loader_authenticates_exact_pcm_asset_and_treatment(tmp_path: Path) -> None:
    asset = tmp_path / "journey.wav"
    with wave.open(str(asset), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16_000)
        stream.writeframes(b"\x01\x00" * 160)
    treatment = VoiceAudioTreatment(
        asset_path="journey.wav",
        asset_sha256=hashlib.sha256(asset.read_bytes()).hexdigest(),
        container="wav",
        pcm_format="s16le",
        channel_count=1,
        sample_rate_hz=16_000,
        playback_rate=1.0,
        pre_speech_silence_seconds=0.01,
        post_speech_silence_seconds=0.02,
    )

    prepared = load_voice_audio_asset(tmp_path, treatment)

    assert prepared.sample_rate_hz == 16_000
    assert prepared.channel_count == 1
    assert prepared.samples_per_channel == 640
    assert len(prepared.pcm_bytes) == 1_280
    assert prepared.pcm_bytes[:320] == bytes(320)

    asset.write_bytes(asset.read_bytes() + b"tampered")
    with pytest.raises(VoiceCertificationProtocolError, match="digest"):
        load_voice_audio_asset(tmp_path, treatment)


def test_voice_audio_loader_decodes_the_exact_authenticated_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asset = tmp_path / "journey.wav"
    with wave.open(str(asset), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16_000)
        stream.writeframes(b"\x01\x00" * 160)
    authenticated = asset.read_bytes()
    treatment = VoiceAudioTreatment(
        asset_path="journey.wav",
        asset_sha256=hashlib.sha256(authenticated).hexdigest(),
        container="wav",
        pcm_format="s16le",
        channel_count=1,
        sample_rate_hz=16_000,
        playback_rate=1.0,
        pre_speech_silence_seconds=0,
        post_speech_silence_seconds=0,
    )
    original_read_bytes = Path.read_bytes

    def replace_after_read(path: Path) -> bytes:
        payload = original_read_bytes(path)
        if path == asset:
            with wave.open(str(asset), "wb") as stream:
                stream.setnchannels(1)
                stream.setsampwidth(2)
                stream.setframerate(16_000)
                stream.writeframes(b"\x02\x00" * 320)
        return payload

    monkeypatch.setattr(Path, "read_bytes", replace_after_read)

    prepared = load_voice_audio_asset(tmp_path, treatment)

    assert prepared.samples_per_channel == 160
    assert prepared.pcm_bytes == b"\x01\x00" * 160


async def test_voice_audio_publisher_preserves_pcm_and_awaits_playout() -> None:
    class AudioSource:
        def __init__(self) -> None:
            self.frames: list[rtc.AudioFrame] = []
            self.playout_waited = False

        async def capture_frame(self, frame: rtc.AudioFrame) -> None:
            self.frames.append(frame)

        async def wait_for_playout(self) -> None:
            self.playout_waited = True

    source = AudioSource()
    prepared = PreparedVoiceAudio(
        pcm_bytes=b"\x01\x00" * 500,
        sample_rate_hz=16_000,
        channel_count=1,
        samples_per_channel=500,
    )

    await publish_prepared_voice_audio(source, prepared)

    assert [frame.samples_per_channel for frame in source.frames] == [320, 180]
    assert b"".join(bytes(frame.data) for frame in source.frames) == prepared.pcm_bytes
    assert source.playout_waited


def test_voice_audio_treatment_rejects_nonportable_or_unimplemented_playback() -> None:
    payload = VoiceAudioTreatment(
        asset_path="assets/audio/latency/journey.wav",
        asset_sha256="a" * 64,
        container="wav",
        pcm_format="s16le",
        channel_count=1,
        sample_rate_hz=16_000,
        playback_rate=1.0,
        pre_speech_silence_seconds=0.25,
        post_speech_silence_seconds=0.5,
    ).model_dump()

    with pytest.raises(ValidationError, match="relative POSIX"):
        VoiceAudioTreatment.model_validate(payload | {"asset_path": "../journey.wav"})
    with pytest.raises(ValidationError, match="playback_rate"):
        VoiceAudioTreatment.model_validate(payload | {"playback_rate": 1.25})


def test_voice_turn_observation_preserves_existing_metrics_without_retiming() -> None:
    sample = _controller().schedule[20]
    components = (
        DurabilityTimingSample(
            operation=DurabilityOperation.CHECKPOINT_READ,
            elapsed_seconds=0.05,
            outcome=DurabilityTimingOutcome.SUCCESS,
        ),
    )
    measurement = TurnLatencyMeasurement(
        end_to_end_seconds=0.75,
        endpointing_seconds=0.25,
        processing_seconds=0.5,
        interrupted=False,
    )

    observation = build_voice_turn_observation(
        sample,
        measurements=(measurement,),
        component_samples=components,
    )

    assert observation.outcome is LatencyObservationOutcome.SUCCESS
    assert observation.elapsed_seconds == 0.5
    assert observation.voice_metrics == VoiceLatencyMetrics(
        end_to_end_seconds=0.75,
        endpointing_seconds=0.25,
        processing_seconds=0.5,
    )

    missing = build_voice_turn_observation(
        sample,
        measurements=(),
        component_samples=components,
    )
    ambiguous = build_voice_turn_observation(
        sample,
        measurements=(measurement, measurement),
        component_samples=components,
    )
    assert missing.error_type == "MissingVoiceLatencyEvidenceError"
    assert ambiguous.error_type == "AmbiguousVoiceLatencyEvidenceError"


def test_voice_measurements_keep_startup_and_turn_components_separate() -> None:
    controller = _controller()
    startup_binding = _binding("sample-startup-0001")
    controller.bind_job(startup_binding)
    startup = controller.assignment(startup_binding.sample_id)
    startup_times = iter((10.0, 10.75))
    startup_measurements = VoiceCertificationMeasurements(time_source=lambda: next(startup_times))
    startup_measurements.durability_timing.observe(
        DurabilityTimingSample(
            operation=DurabilityOperation.POOL_OPEN,
            elapsed_seconds=0.25,
            outcome=DurabilityTimingOutcome.SUCCESS,
        )
    )

    startup_observation = startup_measurements.mark_ready(startup)

    assert startup_observation is not None
    assert startup_observation.elapsed_seconds == 0.75
    assert tuple(item.operation for item in startup_observation.components) == (
        DurabilityOperation.POOL_OPEN,
    )

    turn_binding = _binding("sample-turn-simple-cart-read-0001")
    controller.bind_job(turn_binding)
    turn = controller.assignment(turn_binding.sample_id)
    turn_measurements = VoiceCertificationMeasurements(time_source=lambda: 20.0)
    turn_measurements.bind_turn_contract(
        _contract("simple-cart-read"),
        lambda: _state(),
    )
    assert turn_measurements.mark_ready(turn) is None
    turn_measurements.begin_turn()
    turn_measurements.durability_timing.observe(
        DurabilityTimingSample(
            operation=DurabilityOperation.CHECKPOINT_READ,
            elapsed_seconds=0.05,
            outcome=DurabilityTimingOutcome.SUCCESS,
        )
    )
    event = SpokenMessageEvent(text="Your cart is empty.", node="cart_view_render")
    turn_measurements.observe_event(event)
    turn_measurements.observe_turn_latency(
        TurnLatencyMeasurement(
            end_to_end_seconds=0.75,
            endpointing_seconds=0.25,
            processing_seconds=0.5,
            interrupted=False,
        )
    )

    evidence = turn_measurements.completed_turn(turn)

    assert evidence.events == (event,)
    assert evidence.observation.elapsed_seconds == 0.5
    assert tuple(item.operation for item in evidence.observation.components) == (
        DurabilityOperation.CHECKPOINT_READ,
    )


def test_voice_measurements_partition_setup_responses_from_the_measured_turn() -> None:
    controller = _controller()
    binding = _binding("sample-turn-checkout-placement-readback-0001")
    controller.bind_job(binding)
    assignment = controller.assignment(binding.sample_id)
    measurements = VoiceCertificationMeasurements(time_source=lambda: 1.0)
    snapshots = iter(
        (
            _state(),
            _state(),
            _state(LatencyCartLine(sku="SKU-BLU-07", quantity=1)),
            _state(LatencyCartLine(sku="SKU-BLU-07", quantity=1)),
        )
    )
    measurements.bind_turn_contract(
        _contract("checkout-placement-readback"),
        lambda: next(snapshots),
    )
    measurements.mark_ready(assignment)

    setup_events = (
        SpokenMessageEvent(text="Confirm adding the jacket.", node="cart_add_readback"),
        SpokenMessageEvent(text="Added.", node="cart_add_place"),
    )
    measured_event = InterruptEvent(prompt="Confirm the $129.00 order")
    for event in (*setup_events, measured_event):
        measurements.begin_turn()
        measurements.observe_event(event)
        measurements.observe_turn_latency(
            TurnLatencyMeasurement(
                end_to_end_seconds=0.75,
                endpointing_seconds=0.25,
                processing_seconds=0.5,
                interrupted=False,
            )
        )

    evidence = measurements.completed_turn(assignment)

    assert evidence.setup_events == ((setup_events[0],), (setup_events[1],))
    assert evidence.events == (measured_event,)


def test_voice_measurements_do_not_merge_a_missing_callback_into_the_next_turn() -> None:
    controller = _controller()
    binding = _binding("sample-turn-checkout-placement-readback-0001")
    controller.bind_job(binding)
    assignment = controller.assignment(binding.sample_id)
    line = LatencyCartLine(sku="SKU-BLU-07", quantity=1)
    snapshots = iter((_state(), _state(), _state(line), _state(line)))
    measurements = VoiceCertificationMeasurements(time_source=lambda: 1.0)
    measurements.bind_turn_contract(
        _contract("checkout-placement-readback"),
        lambda: next(snapshots),
    )
    measurements.mark_ready(assignment)
    first = SpokenMessageEvent(text="Confirm adding the jacket.", node="cart_add_readback")
    second = SpokenMessageEvent(text="Added.", node="cart_add_place")
    measured = InterruptEvent(prompt="Confirm the $129.00 order")

    measurements.begin_turn()
    measurements.observe_event(first)
    measurements.begin_turn()
    measurements.observe_event(second)
    measurements.observe_turn_latency(
        TurnLatencyMeasurement(
            end_to_end_seconds=0.75,
            endpointing_seconds=0.25,
            processing_seconds=0.5,
            interrupted=False,
        )
    )
    measurements.begin_turn()
    measurements.observe_event(measured)
    measurements.observe_turn_latency(
        TurnLatencyMeasurement(
            end_to_end_seconds=0.75,
            endpointing_seconds=0.25,
            processing_seconds=0.5,
            interrupted=False,
        )
    )

    evidence = measurements.completed_turn(assignment)

    assert evidence.observation.error_type == "MissingVoiceLatencyEvidenceError"
    assert evidence.setup_events == ((first,), (second,))
    assert evidence.events == (measured,)


def test_voice_measurements_retain_observer_failure_as_sample_evidence() -> None:
    controller = _controller()
    binding = _binding("sample-turn-simple-cart-read-0001")
    controller.bind_job(binding)
    assignment = controller.assignment(binding.sample_id)
    measurements = VoiceCertificationMeasurements(time_source=lambda: 1.0)
    measurements.bind_turn_contract(_contract("simple-cart-read"), lambda: _state())
    measurements.mark_ready(assignment)

    measurements.begin_turn()
    measurements.observe_event(
        SpokenMessageEvent(text="Your cart is empty.", node="cart_view_render")
    )
    measurements.record_observer_failure(RuntimeError("diagnostic observer failed"))
    measurements.observe_turn_latency(
        TurnLatencyMeasurement(
            end_to_end_seconds=0.75,
            endpointing_seconds=0.25,
            processing_seconds=0.5,
            interrupted=False,
        )
    )

    evidence = measurements.completed_turn(assignment)

    assert evidence.observation.outcome is LatencyObservationOutcome.ERROR
    assert evidence.observation.error_type == "RuntimeError"


def test_voice_measurements_reject_missing_or_extra_turn_evidence() -> None:
    controller = _controller()
    binding = _binding("sample-turn-simple-cart-read-0001")
    controller.bind_job(binding)
    assignment = controller.assignment(binding.sample_id)
    measurements = VoiceCertificationMeasurements(time_source=lambda: 1.0)
    measurements.bind_turn_contract(
        _contract("simple-cart-read"),
        lambda: _state(),
    )
    measurements.mark_ready(assignment)

    missing = measurements.completed_turn(assignment)
    assert missing.observation.error_type == "MissingScheduledTurnEvidenceError"

    event = SpokenMessageEvent(text="Your cart is empty.", node="cart_view_render")
    for _ in range(2):
        measurements.begin_turn()
        measurements.observe_event(event)
        measurements.observe_turn_latency(
            TurnLatencyMeasurement(
                end_to_end_seconds=0.75,
                endpointing_seconds=0.25,
                processing_seconds=0.5,
                interrupted=False,
            )
        )
    extra = measurements.completed_turn(assignment)
    assert extra.observation.error_type == "AmbiguousScheduledTurnEvidenceError"


def test_voice_measurements_fail_interrupted_or_semantically_wrong_responses() -> None:
    controller = _controller()
    binding = _binding("sample-turn-simple-cart-read-0001")
    controller.bind_job(binding)
    assignment = controller.assignment(binding.sample_id)

    interrupted = VoiceCertificationMeasurements(time_source=lambda: 1.0)
    interrupted.bind_turn_contract(_contract("simple-cart-read"), lambda: _state())
    interrupted.mark_ready(assignment)
    interrupted.begin_turn()
    interrupted.observe_event(
        SpokenMessageEvent(text="Your cart is empty.", node="cart_view_render")
    )
    interrupted.observe_turn_latency(
        TurnLatencyMeasurement(
            end_to_end_seconds=0.75,
            endpointing_seconds=0.25,
            processing_seconds=0.5,
            interrupted=True,
        )
    )
    assert (
        interrupted.completed_turn(assignment).observation.error_type
        == "InterruptedVoiceResponseError"
    )

    wrong_state = VoiceCertificationMeasurements(time_source=lambda: 1.0)
    states = iter(
        (
            _state(),
            _state(LatencyCartLine(sku="SKU-BLU-07", quantity=1)),
        )
    )
    wrong_state.bind_turn_contract(_contract("simple-cart-read"), lambda: next(states))
    wrong_state.mark_ready(assignment)
    wrong_state.begin_turn()
    wrong_state.durability_timing.observe(
        DurabilityTimingSample(
            operation=DurabilityOperation.CHECKPOINT_READ,
            elapsed_seconds=0.05,
            outcome=DurabilityTimingOutcome.SUCCESS,
        )
    )
    wrong_state.observe_event(
        SpokenMessageEvent(text="Your cart is empty.", node="cart_view_render")
    )
    wrong_state.observe_turn_latency(
        TurnLatencyMeasurement(
            end_to_end_seconds=0.75,
            endpointing_seconds=0.25,
            processing_seconds=0.5,
            interrupted=False,
        )
    )
    assert (
        wrong_state.completed_turn(assignment).observation.error_type == "JourneyPostconditionError"
    )


async def test_job_retries_only_the_same_result_payload_after_lost_ack() -> None:
    controller = _controller()
    binding = _binding("sample-startup-0001")
    controller.bind_job(binding)
    assignment = controller.assignment(binding.sample_id)
    observation = _observation(binding.sample_id)

    class LocalParticipant:
        def __init__(self) -> None:
            self.payloads: list[str] = []

        async def perform_rpc(self, **arguments: object) -> str:
            payload = arguments["payload"]
            assert isinstance(payload, str)
            self.payloads.append(payload)
            assert arguments["destination_identity"] == controller.controller_identity
            assert arguments["method"] == VOICE_CERTIFICATION_RESULT_RPC
            assert arguments["response_timeout"] == 3.0
            if len(self.payloads) == 1:
                raise rtc.RpcError(rtc.RpcError.ErrorCode.RESPONSE_TIMEOUT, "timeout")
            return controller.submit(
                payload,
                caller_identity=binding.worker_participant_identity,
            )

    participant = LocalParticipant()

    ack = await submit_voice_certification_sample(
        cast(rtc.LocalParticipant, participant),
        assignment,
        observation,
    )

    assert ack.sample_id == binding.sample_id
    assert participant.payloads[0] == participant.payloads[1]
    assert controller.received_count == 1


async def test_job_announces_readiness_with_its_exact_server_binding() -> None:
    controller = _controller()
    binding = _binding("sample-turn-simple-cart-read-0001")
    controller.bind_job(binding)
    assignment = controller.assignment(binding.sample_id)

    class LocalParticipant:
        def __init__(self) -> None:
            self.payloads: list[str] = []

        async def perform_rpc(self, **arguments: object) -> str:
            payload = arguments["payload"]
            assert isinstance(payload, str)
            self.payloads.append(payload)
            assert arguments["method"] == VOICE_CERTIFICATION_READY_RPC
            if len(self.payloads) == 1:
                raise rtc.RpcError(rtc.RpcError.ErrorCode.RESPONSE_TIMEOUT, "timeout")
            return controller.submit_ready(
                payload,
                caller_identity=binding.worker_participant_identity,
            )

    participant = LocalParticipant()

    ack = await announce_voice_certification_ready(
        cast(rtc.LocalParticipant, participant),
        assignment,
    )

    assert ack == VoiceCertificationReadyAck(
        run_id=controller.run_id,
        sample_id=binding.sample_id,
    )
    assert participant.payloads[0] == participant.payloads[1]
    assert controller.ready_sample_ids == (binding.sample_id,)


def test_controller_rejects_forged_or_divergent_readiness() -> None:
    controller = _controller()
    binding = _binding("sample-startup-0001")
    controller.bind_job(binding)
    assignment = controller.assignment(binding.sample_id)
    payload = assignment.model_dump_json()

    with pytest.raises(VoiceCertificationProtocolError, match="authenticated participant"):
        controller.submit_ready(payload, caller_identity="foreign-worker")

    assert (
        controller.submit_ready(
            payload,
            caller_identity=binding.worker_participant_identity,
        )
        == VoiceCertificationReadyAck(
            run_id=controller.run_id,
            sample_id=binding.sample_id,
        ).model_dump_json()
    )

    divergent = assignment.model_copy(
        update={"result_rpc_retries": assignment.result_rpc_retries + 1}
    ).model_dump_json()
    with pytest.raises(VoiceCertificationProtocolError, match="divergent"):
        controller.submit_ready(
            divergent,
            caller_identity=binding.worker_participant_identity,
        )


async def test_controller_binds_the_ready_job_to_its_server_created_dispatch() -> None:
    controller = _controller()
    binding = _binding("sample-startup-0001")
    controller.bind_dispatch(binding.sample_id, binding.assignment_id)
    assignment = VoiceCertificationSampleAssignment.from_directive(
        controller.directive(binding.sample_id),
        binding,
    )

    foreign = assignment.model_copy(
        update={"job": binding.model_copy(update={"assignment_id": "foreign-dispatch"})}
    )
    with pytest.raises(VoiceCertificationProtocolError, match="dispatch"):
        controller.submit_ready(
            foreign.model_dump_json(),
            caller_identity=binding.worker_participant_identity,
        )

    controller.submit_ready(
        assignment.model_dump_json(),
        caller_identity=binding.worker_participant_identity,
    )

    assert controller.assignment(binding.sample_id) == assignment
    assert await controller.wait_for_ready(binding.sample_id) == assignment


def test_readiness_is_refused_after_the_controller_records_a_sample_failure() -> None:
    # A terminally failed sample must not be reopened by a late readiness RPC, which
    # would consume a dispatch for work whose outcome is already recorded.
    controller = _controller()
    binding = _binding("sample-turn-checkout-placement-readback-0001")
    controller.bind_job(binding)
    assignment = VoiceCertificationSampleAssignment.from_directive(
        controller.directive(binding.sample_id),
        binding,
    )
    controller.record_failure(binding.sample_id, RuntimeError("provider failure"))

    with pytest.raises(VoiceCertificationProtocolError, match="already failed"):
        controller.submit_ready(
            assignment.model_dump_json(),
            caller_identity=binding.worker_participant_identity,
        )


async def test_setup_progress_is_ordered_authenticated_and_idempotent() -> None:
    controller = _controller()
    binding = _binding("sample-turn-checkout-placement-readback-0001")
    controller.bind_job(binding)
    assignment = controller.assignment(binding.sample_id)
    ready_payload = assignment.model_dump_json()
    controller.submit_ready(
        ready_payload,
        caller_identity=binding.worker_participant_identity,
    )

    class LocalParticipant:
        def __init__(self) -> None:
            self.payloads: list[str] = []

        async def perform_rpc(self, **arguments: object) -> str:
            payload = arguments["payload"]
            assert isinstance(payload, str)
            self.payloads.append(payload)
            assert arguments["method"] == VOICE_CERTIFICATION_PROGRESS_RPC
            if len(self.payloads) == 1:
                raise rtc.RpcError(rtc.RpcError.ErrorCode.RESPONSE_TIMEOUT, "timeout")
            return controller.submit_progress(
                payload,
                caller_identity=binding.worker_participant_identity,
            )

    participant = LocalParticipant()
    ack = await announce_voice_certification_progress(
        cast(rtc.LocalParticipant, participant),
        assignment,
        completed_setup_turns=1,
    )

    assert ack == VoiceCertificationProgressAck(
        run_id=controller.run_id,
        sample_id=binding.sample_id,
        completed_setup_turns=1,
    )
    assert participant.payloads[0] == participant.payloads[1]
    await asyncio.wait_for(controller.wait_for_setup_progress(binding.sample_id, 1), 0.1)

    second_payload = VoiceCertificationProgressEnvelope(
        assignment=assignment,
        completed_setup_turns=2,
    ).model_dump_json()
    with pytest.raises(VoiceCertificationProtocolError, match="authenticated participant"):
        controller.submit_progress(second_payload, caller_identity="foreign-worker")


def test_setup_progress_rejects_a_skipped_boundary() -> None:
    controller = _controller()
    binding = _binding("sample-turn-checkout-placement-readback-0001")
    controller.bind_job(binding)
    assignment = controller.assignment(binding.sample_id)
    controller.submit_ready(
        assignment.model_dump_json(),
        caller_identity=binding.worker_participant_identity,
    )
    payload = VoiceCertificationProgressEnvelope(
        assignment=assignment,
        completed_setup_turns=2,
    ).model_dump_json()

    with pytest.raises(VoiceCertificationProtocolError, match="next scheduled"):
        controller.submit_progress(
            payload,
            caller_identity=binding.worker_participant_identity,
        )


async def test_job_rejects_a_wrong_result_acknowledgement_without_rewriting_evidence() -> None:
    controller = _controller()
    binding = _binding("sample-startup-0001")
    controller.bind_job(binding)

    class LocalParticipant:
        async def perform_rpc(self, **_arguments: object) -> str:
            return (
                '{"schema_version":"1","run_id":"other-run",'
                '"sample_id":"sample-startup-0001","accepted":true}'
            )

    with pytest.raises(VoiceCertificationProtocolError, match="wrong binding"):
        await submit_voice_certification_sample(
            cast(rtc.LocalParticipant, LocalParticipant()),
            controller.assignment(binding.sample_id),
            _observation(binding.sample_id),
        )


async def test_single_room_controller_executes_the_exact_warmup_and_sample_schedule() -> None:
    methodology = _methodology()
    controller = _controller(methodology)
    prepared = PreparedVoiceAudio(
        pcm_bytes=b"\x00\x00" * 160,
        sample_rate_hz=16_000,
        channel_count=1,
        samples_per_channel=160,
    )
    assets = {
        treatment.asset_sha256: prepared
        for journey in methodology.journeys
        for treatment in (*journey.setup_audio_treatments, journey.audio_treatment)
        if treatment is not None
    }

    class Transport:
        def __init__(self) -> None:
            self.created: list[str] = []
            self.closed: list[str] = []
            self._played = 0
            self._played_changed = asyncio.Condition()
            self._workers: dict[str, asyncio.Task[None]] = {}

        async def create_dispatch(self, directive: VoiceCertificationDispatchDirective) -> str:
            sample = directive.sample
            assignment_id = f"dispatch-{sample.sample_id}"
            self.created.append(sample.sample_id)
            played_before = self._played

            async def worker() -> None:
                await asyncio.sleep(0)
                binding = VoiceCertificationJobBinding(
                    sample_id=sample.sample_id,
                    room_id="RM_certification_01",
                    assignment_id=assignment_id,
                    job_id=f"job-{sample.sample_id}",
                    worker_id=f"worker-{sample.sample_id}",
                    worker_participant_identity=f"agent-{sample.sample_id}",
                )
                assignment = VoiceCertificationSampleAssignment.from_directive(
                    directive,
                    binding,
                )
                controller.submit_ready(
                    assignment.model_dump_json(),
                    caller_identity=binding.worker_participant_identity,
                )
                expected_audio = len(sample.setup_audio_treatments)
                if sample.audio_treatment is not None:
                    expected_audio += 1
                for completed_setup_turns in range(1, len(sample.setup_audio_treatments) + 1):
                    async with self._played_changed:
                        await self._played_changed.wait_for(
                            lambda completed=completed_setup_turns: (
                                self._played >= played_before + completed
                            )
                        )
                    controller.submit_progress(
                        VoiceCertificationProgressEnvelope(
                            assignment=assignment,
                            completed_setup_turns=completed_setup_turns,
                        ).model_dump_json(),
                        caller_identity=binding.worker_participant_identity,
                    )
                if expected_audio:
                    async with self._played_changed:
                        await self._played_changed.wait_for(
                            lambda: self._played >= played_before + expected_audio
                        )
                components = tuple(
                    DurabilityComponentObservation(
                        operation=operation,
                        elapsed_seconds=0.01,
                        outcome=DurabilityTimingOutcome.SUCCESS,
                    )
                    for operation in sample.required_operations
                )
                observation = LatencyObservation(
                    schema_version="3",
                    sample_id=sample.sample_id,
                    phase=sample.phase,
                    thermal_state=(
                        ThermalState.COLD
                        if sample.phase is LatencyPhase.STARTUP
                        else ThermalState.WARM
                    ),
                    elapsed_seconds=0.2,
                    journey_id=sample.journey_id,
                    tier=sample.tier,
                    voice_metrics=(
                        None
                        if sample.phase is LatencyPhase.STARTUP
                        else VoiceLatencyMetrics(
                            end_to_end_seconds=0.3,
                            endpointing_seconds=0.1,
                            processing_seconds=0.2,
                        )
                    ),
                    components=components,
                )
                controller.submit(
                    VoiceCertificationSampleEnvelope.from_assignment(
                        assignment,
                        observation,
                    ).model_dump_json(),
                    caller_identity=binding.worker_participant_identity,
                )

            self._workers[assignment_id] = asyncio.create_task(worker())
            return assignment_id

        async def publish_audio(self, prepared: PreparedVoiceAudio) -> None:
            assert prepared.samples_per_channel > 0
            async with self._played_changed:
                self._played += 1
                self._played_changed.notify_all()

        async def close_dispatch(
            self,
            directive: VoiceCertificationDispatchDirective,
            assignment_id: str | None,
            worker_participant_identity: str | None,
        ) -> None:
            assert assignment_id == f"dispatch-{directive.sample.sample_id}"
            assert worker_participant_identity is not None
            assert assignment_id is not None
            await self._workers[assignment_id]
            self.closed.append(assignment_id)

    transport = Transport()
    run = await run_voice_certification_controller(
        controller,
        transport,
        assets,
        run_at=datetime(2026, 9, 13, tzinfo=UTC),
    )

    expected_ids = [
        sample.sample_id for sample in (*controller.warmup_schedule, *controller.schedule)
    ]
    assert transport.created == expected_ids
    assert len(transport.closed) == len(expected_ids)
    assert run.outcome is LatencyCertificationOutcome.COMPLETED
    assert run.report is not None
    assert run.report.gate.passed


async def test_smoke_executes_one_real_journey_without_finalizing_evidence() -> None:
    methodology = _methodology()
    controller = _controller(methodology)
    prepared = PreparedVoiceAudio(
        pcm_bytes=b"\x00\x00" * 160,
        sample_rate_hz=16_000,
        channel_count=1,
        samples_per_channel=160,
    )
    assets = {
        treatment.asset_sha256: prepared
        for journey in methodology.journeys
        for treatment in (*journey.setup_audio_treatments, journey.audio_treatment)
        if treatment is not None
    }

    class Transport:
        def __init__(self) -> None:
            self.audio_published = asyncio.Event()
            self.created: list[str] = []
            self.closed: list[str] = []
            self.worker: asyncio.Task[None] | None = None

        async def create_dispatch(self, directive: VoiceCertificationDispatchDirective) -> str:
            sample = directive.sample
            assignment_id = f"dispatch-{sample.sample_id}"
            self.created.append(sample.sample_id)

            async def worker() -> None:
                await asyncio.sleep(0)
                binding = VoiceCertificationJobBinding(
                    sample_id=sample.sample_id,
                    room_id="RM_certification_01",
                    assignment_id=assignment_id,
                    job_id=f"job-{sample.sample_id}",
                    worker_id=f"worker-{sample.sample_id}",
                    worker_participant_identity=f"agent-{sample.sample_id}",
                )
                assignment = VoiceCertificationSampleAssignment.from_directive(
                    directive,
                    binding,
                )
                controller.submit_ready(
                    assignment.model_dump_json(),
                    caller_identity=binding.worker_participant_identity,
                )
                await self.audio_published.wait()
                controller.submit(
                    VoiceCertificationSampleEnvelope.from_assignment(
                        assignment,
                        _observation(sample.sample_id),
                    ).model_dump_json(),
                    caller_identity=binding.worker_participant_identity,
                )

            self.worker = asyncio.create_task(worker())
            return assignment_id

        async def publish_audio(self, prepared: PreparedVoiceAudio) -> None:
            assert prepared.samples_per_channel > 0
            self.audio_published.set()

        async def close_dispatch(
            self,
            directive: VoiceCertificationDispatchDirective,
            assignment_id: str | None,
            worker_participant_identity: str | None,
        ) -> None:
            assert assignment_id == f"dispatch-{directive.sample.sample_id}"
            assert worker_participant_identity is not None
            assert self.worker is not None
            await self.worker
            self.closed.append(assignment_id)

    transport = Transport()
    observation = await run_voice_certification_smoke(
        controller,
        transport,
        assets,
        journey_id="simple-cart-read",
        run_at=datetime(2026, 9, 13, tzinfo=UTC),
    )

    assert observation.outcome is LatencyObservationOutcome.SUCCESS
    assert transport.created == ["sample-turn-simple-cart-read-0001"]
    assert transport.closed == ["dispatch-sample-turn-simple-cart-read-0001"]
    with pytest.raises(VoiceCertificationProtocolError, match="incomplete"):
        controller.build_run(run_at=datetime(2026, 9, 13, tzinfo=UTC))


async def test_smoke_rejects_a_journey_outside_the_frozen_methodology() -> None:
    with pytest.raises(VoiceCertificationProtocolError, match="outside the frozen methodology"):
        await run_voice_certification_smoke(
            _controller(_methodology()),
            cast(VoiceCertificationControllerTransport, object()),
            {},
            journey_id="not-a-journey",
            run_at=datetime(2026, 9, 13, tzinfo=UTC),
        )


def test_controller_rejects_parallel_concurrency_smuggled_past_the_validator() -> None:
    # model_copy does not run validators, so the model-level rule alone is bypassable.
    # The controller constructor is the chokepoint every voice execution passes through.
    smuggled = _methodology().model_copy(update={"concurrency": 4})
    assert smuggled.concurrency == 4

    with pytest.raises(VoiceCertificationProtocolError, match="concurrency 1"):
        _controller(smuggled)


def test_voice_methodology_rejects_parallel_concurrency() -> None:
    # The single-room voice path is strictly sequential, so the frozen contract is
    # refused at validation rather than at run time after it has been fingerprinted.
    with pytest.raises(ValidationError, match="concurrency 1"):
        DurableLatencyMethodology.model_validate(_methodology().model_dump() | {"concurrency": 2})


async def test_voice_controller_aborts_before_dispatch_when_audio_is_missing() -> None:
    methodology = _methodology()
    controller = _controller(methodology)

    class Transport:
        async def create_dispatch(self, directive: VoiceCertificationDispatchDirective) -> str:
            raise AssertionError(f"unexpected dispatch for {directive.sample.sample_id}")

        async def publish_audio(self, prepared: PreparedVoiceAudio) -> None:
            raise AssertionError(f"unexpected audio with {prepared.samples_per_channel} samples")

        async def close_dispatch(
            self,
            directive: VoiceCertificationDispatchDirective,
            assignment_id: str | None,
            worker_participant_identity: str | None,
        ) -> None:
            raise AssertionError(
                "unexpected cleanup for "
                f"{directive.sample.sample_id}/{assignment_id}/{worker_participant_identity}"
            )

    run = await run_voice_certification_controller(
        controller,
        Transport(),
        {},
        run_at=datetime(2026, 9, 13, tzinfo=UTC),
    )

    assert run.outcome is LatencyCertificationOutcome.ABORTED
    assert run.abort is not None
    assert run.abort.stage is LatencyAbortStage.VOICE_PREPARE
    assert run.abort.error_type == "VoiceCertificationProtocolError"


async def test_voice_controller_reconciles_an_unknown_dispatch_before_aborting() -> None:
    methodology = _methodology()
    controller = _controller(methodology)
    prepared = PreparedVoiceAudio(
        pcm_bytes=b"\x00\x00" * 160,
        sample_rate_hz=16_000,
        channel_count=1,
        samples_per_channel=160,
    )
    assets = {
        treatment.asset_sha256: prepared
        for journey in methodology.journeys
        for treatment in (*journey.setup_audio_treatments, journey.audio_treatment)
        if treatment is not None
    }

    class Transport:
        def __init__(self) -> None:
            self.created: list[str] = []
            self.cleaned: list[tuple[str, str | None]] = []

        async def create_dispatch(self, directive: VoiceCertificationDispatchDirective) -> str:
            self.created.append(directive.sample.sample_id)
            raise RuntimeError("response lost after server commit")

        async def publish_audio(self, prepared: PreparedVoiceAudio) -> None:
            raise AssertionError(f"unexpected audio with {prepared.samples_per_channel} samples")

        async def close_dispatch(
            self,
            directive: VoiceCertificationDispatchDirective,
            assignment_id: str | None,
            worker_participant_identity: str | None,
        ) -> None:
            assert worker_participant_identity is None
            self.cleaned.append((directive.sample.sample_id, assignment_id))

    transport = Transport()
    run = await run_voice_certification_controller(
        controller,
        transport,
        assets,
        run_at=datetime(2026, 9, 13, tzinfo=UTC),
    )

    first_sample = controller.warmup_schedule[0].sample_id
    assert transport.created == [first_sample]
    assert transport.cleaned == [(first_sample, None)]
    assert run.outcome is LatencyCertificationOutcome.ABORTED
    assert run.abort is not None
    assert run.abort.stage is LatencyAbortStage.VOICE_RUN


async def test_voice_controller_finishes_dispatch_cleanup_before_propagating_cancellation() -> None:
    methodology = _methodology()
    controller = _controller(methodology)
    prepared = PreparedVoiceAudio(
        pcm_bytes=b"\x00\x00" * 160,
        sample_rate_hz=16_000,
        channel_count=1,
        samples_per_channel=160,
    )
    assets = {
        treatment.asset_sha256: prepared
        for journey in methodology.journeys
        for treatment in (*journey.setup_audio_treatments, journey.audio_treatment)
        if treatment is not None
    }

    class Transport:
        def __init__(self) -> None:
            self.dispatched = asyncio.Event()
            self.cleanup_started = asyncio.Event()
            self.release_cleanup = asyncio.Event()
            self.cleanup_finished = False

        async def create_dispatch(self, directive: VoiceCertificationDispatchDirective) -> str:
            self.dispatched.set()
            return f"dispatch-{directive.sample.sample_id}"

        async def publish_audio(self, prepared: PreparedVoiceAudio) -> None:
            raise AssertionError(f"unexpected audio with {prepared.samples_per_channel} samples")

        async def close_dispatch(
            self,
            directive: VoiceCertificationDispatchDirective,
            assignment_id: str | None,
            worker_participant_identity: str | None,
        ) -> None:
            assert assignment_id == f"dispatch-{directive.sample.sample_id}"
            assert assignment_id is not None
            assert assignment_id.startswith("dispatch-")
            assert worker_participant_identity is None
            self.cleanup_started.set()
            await self.release_cleanup.wait()
            self.cleanup_finished = True

    transport = Transport()
    task = asyncio.create_task(
        run_voice_certification_controller(
            controller,
            transport,
            assets,
            run_at=datetime(2026, 9, 13, tzinfo=UTC),
        )
    )
    await transport.dispatched.wait()
    task.cancel()
    await transport.cleanup_started.wait()
    task.cancel()
    transport.release_cleanup.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert transport.cleanup_finished


async def test_voice_cleanup_timeout_awaits_owned_task_cancellation() -> None:
    from agnostic_market.durability import voice_certification

    started = asyncio.Event()
    finalized = asyncio.Event()

    async def cleanup() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            finalized.set()

    task = asyncio.create_task(cleanup())
    await started.wait()

    with pytest.raises(TimeoutError, match="cleanup timed out"):
        await voice_certification._finish_voice_cleanup(  # type: ignore[attr-defined]
            task,
            timeout_seconds=0.001,
        )

    assert finalized.is_set()
    assert task.done()
    assert task.cancelled()


async def test_result_rpc_stops_immediately_on_a_non_retryable_error() -> None:
    """A protocol rejection is not transient, so it must not consume the retry budget."""
    controller = _controller()
    binding = _binding("sample-startup-0001")
    controller.bind_job(binding)
    assignment = controller.assignment(binding.sample_id)
    observation = _observation(binding.sample_id)

    class LocalParticipant:
        def __init__(self) -> None:
            self.attempts = 0

        async def perform_rpc(self, **_arguments: object) -> str:
            self.attempts += 1
            raise rtc.RpcError(rtc.RpcError.ErrorCode.UNSUPPORTED_METHOD, "unknown method")

    participant = LocalParticipant()

    with pytest.raises(VoiceCertificationProtocolError, match="RPC failed"):
        await submit_voice_certification_sample(
            cast(rtc.LocalParticipant, participant),
            assignment,
            observation,
        )

    assert participant.attempts == 1


async def test_result_rpc_exhausts_its_frozen_retry_budget_then_fails_closed() -> None:
    """A transient failure that never clears fails the sample instead of retrying forever."""
    controller = _controller()
    binding = _binding("sample-startup-0001")
    controller.bind_job(binding)
    assignment = controller.assignment(binding.sample_id)
    observation = _observation(binding.sample_id)

    class LocalParticipant:
        def __init__(self) -> None:
            self.attempts = 0

        async def perform_rpc(self, **_arguments: object) -> str:
            self.attempts += 1
            raise rtc.RpcError(rtc.RpcError.ErrorCode.RECIPIENT_NOT_FOUND, "controller absent")

    participant = LocalParticipant()

    with pytest.raises(VoiceCertificationProtocolError, match="RPC failed"):
        await submit_voice_certification_sample(
            cast(rtc.LocalParticipant, participant),
            assignment,
            observation,
        )

    assert participant.attempts == assignment.result_rpc_retries + 1


async def test_readiness_deadline_aborts_without_consuming_the_sample_budget() -> None:
    """Readiness and sample results own separate deadlines.

    A worker that never reports ready must fail on the readiness budget rather than the
    far larger result budget, which also keeps the worker's own timeout evidence viable.
    """
    methodology = _methodology().model_copy(update={"job_ready_timeout_seconds": 0.05})
    controller = _controller(methodology)
    prepared = PreparedVoiceAudio(
        pcm_bytes=b"\x00\x00" * 160,
        sample_rate_hz=16_000,
        channel_count=1,
        samples_per_channel=160,
    )
    assets = {
        treatment.asset_sha256: prepared
        for journey in methodology.journeys
        for treatment in (*journey.setup_audio_treatments, journey.audio_treatment)
        if treatment is not None
    }

    class Transport:
        def __init__(self) -> None:
            self.created: list[str] = []
            self.cleaned: list[str] = []

        async def create_dispatch(self, directive: VoiceCertificationDispatchDirective) -> str:
            self.created.append(directive.sample.sample_id)
            return f"dispatch-{directive.sample.sample_id}"

        async def publish_audio(self, prepared: PreparedVoiceAudio) -> None:
            raise AssertionError(f"unexpected audio with {prepared.samples_per_channel} samples")

        async def close_dispatch(
            self,
            directive: VoiceCertificationDispatchDirective,
            assignment_id: str | None,
            worker_participant_identity: str | None,
        ) -> None:
            assert assignment_id == f"dispatch-{directive.sample.sample_id}"
            assert worker_participant_identity is None
            self.cleaned.append(directive.sample.sample_id)

    transport = Transport()
    started = asyncio.get_running_loop().time()
    run = await run_voice_certification_controller(
        controller,
        transport,
        assets,
        run_at=datetime(2026, 9, 13, tzinfo=UTC),
    )
    elapsed = asyncio.get_running_loop().time() - started

    first_sample = controller.warmup_schedule[0].sample_id
    assert transport.created == [first_sample]
    assert transport.cleaned == [first_sample]
    assert run.outcome is LatencyCertificationOutcome.ABORTED
    assert run.abort is not None
    assert run.abort.stage is LatencyAbortStage.VOICE_RUN
    assert run.abort.error_type == "TimeoutError"
    # The result deadline covers the sample budget plus the worst-case RPC policy.
    # Aborting far inside it proves the readiness deadline governed this failure.
    assert elapsed < controller.sample_result_timeout_seconds / 4
