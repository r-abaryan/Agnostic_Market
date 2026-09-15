"""Fail-closed dispatch boundary for the synthetic voice-certification worker."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import time
import wave
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Protocol, Self

from livekit import rtc
from livekit.agents import JobRequest
from livekit.protocol import agent, models
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from agnostic_market.config.loader import ConfigError, load_yaml_layer
from agnostic_market.dtos.events import TurnEvent
from agnostic_market.dtos.platform import ConfigIdentifier
from agnostic_market.dtos.session import AuthorityIdentifier
from agnostic_market.durability.latency import (
    DurabilityComponentObservation,
    DurableLatencyCertificationRun,
    DurableLatencyMethodology,
    DurableLatencyReport,
    LatencyAbortStage,
    LatencyCartLine,
    LatencyCertificationAbort,
    LatencyCertificationOutcome,
    LatencyEnvironment,
    LatencyJourneyContract,
    LatencyJourneyCorpus,
    LatencyJourneyPostconditionError,
    LatencyMeasurementSurface,
    LatencyObservation,
    LatencyObservationOutcome,
    LatencyPhase,
    LatencyTier,
    ThermalState,
    VoiceApplicationContract,
    VoiceAudioTreatment,
    VoiceLatencyMetrics,
    build_latency_report,
    latency_evidence_schema_version,
    latency_journey_contract_fingerprint,
    methodology_fingerprint,
    require_latency_journey_postconditions,
    voice_application_contract_fingerprint,
)
from agnostic_market.durability.timing import (
    DurabilityOperation,
    DurabilityTimingSample,
    InMemoryDurabilityTimingObserver,
)

_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)
VOICE_CERTIFICATION_RESULT_RPC = "agnostic_market.certification.sample_result.v1"
VOICE_CERTIFICATION_READY_RPC = "agnostic_market.certification.sample_ready.v1"
VOICE_CERTIFICATION_PROGRESS_RPC = "agnostic_market.certification.setup_progress.v1"
VOICE_CERTIFICATION_SAMPLE_ATTRIBUTE = "agnostic_market.certification.sample.v1"
_RETRYABLE_RPC_ERRORS = {
    rtc.RpcError.ErrorCode.CONNECTION_TIMEOUT,
    rtc.RpcError.ErrorCode.RESPONSE_TIMEOUT,
    rtc.RpcError.ErrorCode.RECIPIENT_DISCONNECTED,
    rtc.RpcError.ErrorCode.SEND_FAILED,
    rtc.RpcError.ErrorCode.RECIPIENT_NOT_FOUND,
}


class VoiceCertificationProtocolError(RuntimeError):
    """A sample is not authorized by the frozen certification run."""


@dataclass(frozen=True, slots=True)
class _VoiceSampleResolution:
    kind: Literal["accepted", "timeout", "failure"]
    observation: LatencyObservation
    payload: str | None = None


@dataclass(frozen=True, slots=True)
class PreparedVoiceAudio:
    """Authenticated PCM bytes including the frozen leading and trailing silence."""

    pcm_bytes: bytes
    sample_rate_hz: int
    channel_count: int
    samples_per_channel: int


class VoiceCertificationAudioSource(Protocol):
    async def capture_frame(self, frame: rtc.AudioFrame) -> None: ...

    async def wait_for_playout(self) -> None: ...


def load_voice_audio_asset(
    asset_root: Path,
    treatment: VoiceAudioTreatment,
) -> PreparedVoiceAudio:
    """Authenticate and decode one pre-generated PCM WAV without resampling it."""
    try:
        root = asset_root.resolve(strict=True)
        asset = (root / treatment.asset_path).resolve(strict=True)
        if not asset.is_relative_to(root) or not asset.is_file():
            raise VoiceCertificationProtocolError(
                "voice certification audio asset is outside its configured root"
            )
        encoded = asset.read_bytes()
    except VoiceCertificationProtocolError:
        raise
    except OSError as exc:
        raise VoiceCertificationProtocolError(
            "voice certification audio asset cannot be read"
        ) from exc
    if hashlib.sha256(encoded).hexdigest() != treatment.asset_sha256:
        raise VoiceCertificationProtocolError("voice certification audio asset digest mismatch")
    try:
        # Decode the authenticated bytes, not a second read of the path. Otherwise an
        # asset replacement between the digest and WAV reads could measure different
        # audio from the bytes bound into the methodology.
        with wave.open(io.BytesIO(encoded), "rb") as stream:
            if (
                stream.getcomptype() != "NONE"
                or stream.getsampwidth() != 2
                or stream.getnchannels() != treatment.channel_count
                or stream.getframerate() != treatment.sample_rate_hz
            ):
                raise VoiceCertificationProtocolError(
                    "voice certification audio asset has the wrong PCM format"
                )
            speech_samples = stream.getnframes()
            speech = stream.readframes(speech_samples)
    except VoiceCertificationProtocolError:
        raise
    except (OSError, EOFError, wave.Error) as exc:
        raise VoiceCertificationProtocolError(
            "voice certification audio asset is not a valid WAV file"
        ) from exc
    if speech_samples <= 0:
        raise VoiceCertificationProtocolError("voice certification audio asset is empty")
    bytes_per_sample = treatment.channel_count * 2
    leading_samples = round(treatment.pre_speech_silence_seconds * treatment.sample_rate_hz)
    trailing_samples = round(treatment.post_speech_silence_seconds * treatment.sample_rate_hz)
    pcm = (
        bytes(leading_samples * bytes_per_sample)
        + speech
        + bytes(trailing_samples * bytes_per_sample)
    )
    return PreparedVoiceAudio(
        pcm_bytes=pcm,
        sample_rate_hz=treatment.sample_rate_hz,
        channel_count=treatment.channel_count,
        samples_per_channel=leading_samples + speech_samples + trailing_samples,
    )


async def publish_prepared_voice_audio(
    source: VoiceCertificationAudioSource,
    prepared: PreparedVoiceAudio,
) -> None:
    """Publish authenticated PCM in real-time 20 ms frames and await playout."""
    if prepared.sample_rate_hz % 50 != 0:
        raise VoiceCertificationProtocolError(
            "voice certification sample rate cannot form exact 20 ms frames"
        )
    frame_samples = prepared.sample_rate_hz // 50
    bytes_per_sample = prepared.channel_count * 2
    frame_bytes = frame_samples * bytes_per_sample
    for offset in range(0, len(prepared.pcm_bytes), frame_bytes):
        payload = prepared.pcm_bytes[offset : offset + frame_bytes]
        samples = len(payload) // bytes_per_sample
        await source.capture_frame(
            rtc.AudioFrame(
                data=payload,
                sample_rate=prepared.sample_rate_hz,
                num_channels=prepared.channel_count,
                samples_per_channel=samples,
            )
        )
    await source.wait_for_playout()


class VoiceTurnLatencyMeasurement(Protocol):
    @property
    def end_to_end_seconds(self) -> float: ...

    @property
    def endpointing_seconds(self) -> float: ...

    @property
    def processing_seconds(self) -> float: ...

    @property
    def interrupted(self) -> bool: ...


class VoiceCertificationCartLine(Protocol):
    sku: str
    quantity: int


def _component_observations(
    samples: tuple[DurabilityTimingSample, ...],
) -> tuple[DurabilityComponentObservation, ...]:
    return tuple(
        DurabilityComponentObservation(
            operation=sample.operation,
            elapsed_seconds=sample.elapsed_seconds,
            outcome=sample.outcome,
        )
        for sample in samples
    )


def _failed_voice_observation(
    sample: VoiceCertificationSampleSpec,
    error_type: str,
    component_samples: tuple[DurabilityTimingSample, ...],
    *,
    evidence_schema_version: Literal["2", "3"] = "2",
) -> LatencyObservation:
    return LatencyObservation(
        schema_version=evidence_schema_version,
        sample_id=sample.sample_id,
        phase=sample.phase,
        thermal_state=(
            ThermalState.COLD if sample.phase is LatencyPhase.STARTUP else ThermalState.WARM
        ),
        outcome=LatencyObservationOutcome.ERROR,
        error_type=error_type,
        journey_id=sample.journey_id,
        tier=sample.tier,
        components=_component_observations(component_samples),
    )


def build_voice_turn_observation(
    sample: VoiceCertificationSampleSpec,
    *,
    measurements: tuple[VoiceTurnLatencyMeasurement, ...],
    component_samples: tuple[DurabilityTimingSample, ...],
    evidence_schema_version: Literal["2", "3"] = "2",
) -> LatencyObservation:
    """Convert the existing correlated voice metric into one typed sample result."""
    if sample.phase is not LatencyPhase.TURN:
        raise VoiceCertificationProtocolError("voice turn evidence requires a turn sample")
    error_type: str | None = None
    metrics: VoiceLatencyMetrics | None = None
    if len(measurements) == 0:
        error_type = "MissingVoiceLatencyEvidenceError"
    elif len(measurements) != 1:
        error_type = "AmbiguousVoiceLatencyEvidenceError"
    else:
        measurement = measurements[0]
        if measurement.interrupted:
            error_type = "InterruptedVoiceResponseError"
        else:
            try:
                metrics = VoiceLatencyMetrics(
                    end_to_end_seconds=measurement.end_to_end_seconds,
                    endpointing_seconds=measurement.endpointing_seconds,
                    processing_seconds=measurement.processing_seconds,
                )
            except ValidationError:
                error_type = "InvalidVoiceLatencyEvidenceError"
    if not component_samples and error_type is None:
        error_type = "MissingComponentEvidenceError"
    if error_type is not None:
        return _failed_voice_observation(
            sample,
            error_type,
            component_samples,
            evidence_schema_version=evidence_schema_version,
        )
    assert metrics is not None
    return LatencyObservation.from_timing(
        schema_version=evidence_schema_version,
        sample_id=sample.sample_id,
        phase=LatencyPhase.TURN,
        thermal_state=ThermalState.WARM,
        elapsed_seconds=metrics.processing_seconds,
        journey_id=sample.journey_id,
        tier=sample.tier,
        voice_metrics=metrics,
        component_samples=component_samples,
    )


@dataclass(frozen=True, slots=True)
class VoiceCertificationTurnEvidence:
    """One measured turn plus its ordered, unmeasured setup response events."""

    observation: LatencyObservation
    setup_events: tuple[tuple[TurnEvent, ...], ...]
    events: tuple[TurnEvent, ...]


@dataclass(slots=True)
class _VoiceCertificationTurnCapture:
    component_start: int
    events: list[TurnEvent]
    measurement: VoiceTurnLatencyMeasurement | None = None
    components: tuple[DurabilityTimingSample, ...] = ()
    state: VoiceCertificationApplicationSnapshot | None = None
    error_type: str | None = None
    complete: bool = False


@dataclass(frozen=True, slots=True)
class VoiceCertificationApplicationSnapshot:
    """The synthetic business state needed to prove one journey postcondition."""

    cart: tuple[LatencyCartLine, ...]
    placed_order_count: int

    def __post_init__(self) -> None:
        if self.placed_order_count < 0:
            raise ValueError("voice certification order count cannot be negative")


def capture_voice_certification_application_state(
    cart_lines: Sequence[VoiceCertificationCartLine],
    order_store: object,
) -> VoiceCertificationApplicationSnapshot:
    """Read the synthetic adapters through their established verification surfaces."""
    try:
        cart = tuple(
            LatencyCartLine(
                sku=line.sku,
                quantity=line.quantity,
            )
            for line in cart_lines
        )
    except (AttributeError, ValidationError) as exc:
        raise VoiceCertificationProtocolError(
            "voice certification cart state cannot be observed"
        ) from exc
    placed_count = getattr(order_store, "placed_count", None)
    if isinstance(placed_count, bool) or not isinstance(placed_count, int):
        raise VoiceCertificationProtocolError(
            "voice certification requires the synthetic order verification surface"
        )
    return VoiceCertificationApplicationSnapshot(
        cart=cart,
        placed_order_count=placed_count,
    )


def resolve_voice_journey_contract(
    corpus: LatencyJourneyCorpus,
    sample: VoiceCertificationSampleSpec,
) -> LatencyJourneyContract:
    """Resolve and authenticate the semantic contract carried by one turn sample."""
    if sample.phase is not LatencyPhase.TURN or sample.journey_id is None:
        raise VoiceCertificationProtocolError("startup sample has no journey contract")
    matches = tuple(
        contract for contract in corpus.journeys if contract.journey_id == sample.journey_id
    )
    if len(matches) != 1:
        raise VoiceCertificationProtocolError(
            "voice certification journey is outside the frozen corpus"
        )
    contract = matches[0]
    expected = latency_journey_contract_fingerprint(
        contract,
        merchant_id=corpus.merchant_id,
    )
    if sample.journey_contract_fingerprint != expected:
        raise VoiceCertificationProtocolError(
            "voice certification journey contract fingerprint mismatch"
        )
    if len(sample.setup_audio_treatments) != len(contract.setup_turns):
        raise VoiceCertificationProtocolError("voice certification journey setup schedule mismatch")
    return contract


class VoiceCertificationMeasurements:
    """Partition one job's existing voice and durability observers by turn."""

    def __init__(
        self,
        *,
        time_source: Callable[[], float] = time.perf_counter,
        evidence_schema_version: Literal["2", "3"] = "2",
    ) -> None:
        self.durability_timing = InMemoryDurabilityTimingObserver()
        self._time_source = time_source
        self._evidence_schema_version: Literal["2", "3"] = evidence_schema_version
        self._started_at = time_source()
        self._ready = False
        self._turn_contract: LatencyJourneyContract | None = None
        self._state_observer: Callable[[], VoiceCertificationApplicationSnapshot] | None = None
        self._initial_state: VoiceCertificationApplicationSnapshot | None = None
        self._turns: list[_VoiceCertificationTurnCapture] = []
        self._active_turn: _VoiceCertificationTurnCapture | None = None
        self._unbound_observer_error_type: str | None = None
        self._turn_completed = asyncio.Event()

    def bind_turn_contract(
        self,
        contract: LatencyJourneyContract,
        state_observer: Callable[[], VoiceCertificationApplicationSnapshot],
    ) -> None:
        """Bind the exact behavior contract before synthetic caller audio is admitted."""
        if self._ready or self._turn_contract is not None:
            raise VoiceCertificationProtocolError(
                "voice certification turn contract is already bound"
            )
        self._turn_contract = contract
        self._state_observer = state_observer

    def mark_ready(
        self,
        assignment: VoiceCertificationSampleAssignment,
    ) -> LatencyObservation | None:
        """Close startup timing before any synthetic caller audio is admitted."""
        if self._ready:
            raise VoiceCertificationProtocolError("voice certification job is already ready")
        if assignment.sample.phase is LatencyPhase.TURN:
            if self._turn_contract is None or self._state_observer is None:
                raise VoiceCertificationProtocolError(
                    "voice certification turn has no bound behavior contract"
                )
            if self._turn_contract.journey_id != assignment.sample.journey_id:
                raise VoiceCertificationProtocolError(
                    "voice certification turn contract has the wrong journey"
                )
            self._initial_state = self._state_observer()
        self._ready = True
        components = self.durability_timing.samples
        if assignment.sample.phase is LatencyPhase.TURN:
            return None
        return LatencyObservation.from_timing(
            schema_version=self._evidence_schema_version,
            sample_id=assignment.sample.sample_id,
            phase=LatencyPhase.STARTUP,
            thermal_state=ThermalState.COLD,
            elapsed_seconds=self._time_source() - self._started_at,
            component_samples=components,
        )

    def begin_turn(self) -> None:
        """Open one explicit graph turn before any graph-authored event is observed."""
        if not self._ready:
            raise VoiceCertificationProtocolError(
                "voice certification observed a turn before readiness"
            )
        if self._active_turn is not None:
            self._complete_active_turn(error_type="MissingVoiceLatencyEvidenceError")
        capture = _VoiceCertificationTurnCapture(
            component_start=len(self.durability_timing.samples),
            events=[],
            error_type=self._unbound_observer_error_type,
        )
        self._unbound_observer_error_type = None
        self._turns.append(capture)
        self._active_turn = capture

    def observe_event(self, event: TurnEvent) -> None:
        if not self._ready:
            raise VoiceCertificationProtocolError(
                "voice certification observed a turn before readiness"
            )
        if self._active_turn is None:
            raise VoiceCertificationProtocolError(
                "voice certification observed an event outside a turn"
            )
        self._active_turn.events.append(event)

    def observe_turn_latency(self, measurement: VoiceTurnLatencyMeasurement) -> None:
        if not self._ready:
            raise VoiceCertificationProtocolError(
                "voice certification observed latency before readiness"
            )
        if self._active_turn is None:
            raise VoiceCertificationProtocolError(
                "voice certification observed latency outside a turn"
            )
        self._complete_active_turn(measurement=measurement)

    def record_observer_failure(self, failure: Exception) -> None:
        """Retain diagnostic-observer failure without interrupting caller output."""
        error_type = type(failure).__name__
        if self._active_turn is None:
            self._unbound_observer_error_type = error_type
            return
        if self._active_turn.error_type is None:
            self._active_turn.error_type = error_type

    def _complete_active_turn(
        self,
        *,
        measurement: VoiceTurnLatencyMeasurement | None = None,
        error_type: str | None = None,
    ) -> None:
        capture = self._active_turn
        if capture is None:
            raise VoiceCertificationProtocolError("voice certification has no active turn")
        samples = self.durability_timing.samples
        capture.measurement = measurement
        capture.components = samples[capture.component_start :]
        if error_type is not None and capture.error_type is None:
            capture.error_type = error_type
        try:
            capture.state = self._state_observer() if self._state_observer is not None else None
        except Exception as exc:
            if capture.error_type is None:
                capture.error_type = type(exc).__name__
        capture.complete = True
        self._active_turn = None
        self._turn_completed.set()

    @property
    def _completed_turn_count(self) -> int:
        return sum(turn.complete for turn in self._turns)

    async def wait_for_turn(
        self,
        assignment: VoiceCertificationSampleAssignment,
        *,
        on_setup_progress: Callable[[int], Awaitable[None]] | None = None,
    ) -> VoiceCertificationTurnEvidence:
        expected = len(assignment.sample.setup_audio_treatments) + 1
        try:
            async with asyncio.timeout(assignment.sample_timeout_seconds):
                for completed_setup_turns in range(1, expected):
                    await self._wait_for_turn_count(completed_setup_turns)
                    if on_setup_progress is not None:
                        await on_setup_progress(completed_setup_turns)
                await self._wait_for_turn_count(expected)
        except TimeoutError:
            pass
        return self.completed_turn(assignment)

    async def _wait_for_turn_count(self, completed_turns: int) -> None:
        if completed_turns < 1:
            raise VoiceCertificationProtocolError(
                "voice certification completed-turn count must be positive"
            )
        while self._completed_turn_count < completed_turns:
            self._turn_completed.clear()
            await self._turn_completed.wait()

    def completed_turn(
        self,
        assignment: VoiceCertificationSampleAssignment,
    ) -> VoiceCertificationTurnEvidence:
        sample = assignment.sample
        if not self._ready:
            raise VoiceCertificationProtocolError("voice certification job is not ready")
        if sample.phase is not LatencyPhase.TURN:
            raise VoiceCertificationProtocolError("startup sample has no measured turn")
        if self._active_turn is not None:
            self._complete_active_turn(error_type="MissingVoiceLatencyEvidenceError")
        expected = len(sample.setup_audio_treatments) + 1
        if len(self._turns) != expected:
            error_type = (
                "MissingScheduledTurnEvidenceError"
                if len(self._turns) < expected
                else "AmbiguousScheduledTurnEvidenceError"
            )
            components = tuple(component for turn in self._turns for component in turn.components)
            setup_count = len(sample.setup_audio_treatments)
            return VoiceCertificationTurnEvidence(
                observation=_failed_voice_observation(
                    sample,
                    error_type,
                    components,
                    evidence_schema_version=self._evidence_schema_version,
                ),
                setup_events=tuple(tuple(turn.events) for turn in self._turns[:setup_count]),
                events=(
                    tuple(self._turns[setup_count].events) if len(self._turns) > setup_count else ()
                ),
            )
        setup = self._turns[:-1]
        measured = self._turns[-1]
        all_components = tuple(component for turn in self._turns for component in turn.components)
        evidence_error = next(
            (turn.error_type for turn in self._turns if turn.error_type is not None),
            self._unbound_observer_error_type,
        )
        if evidence_error is not None:
            observation = _failed_voice_observation(
                sample,
                evidence_error,
                all_components,
                evidence_schema_version=self._evidence_schema_version,
            )
        else:
            observation = build_voice_turn_observation(
                sample,
                measurements=(() if measured.measurement is None else (measured.measurement,)),
                component_samples=measured.components,
                evidence_schema_version=self._evidence_schema_version,
            )
        if observation.outcome is LatencyObservationOutcome.SUCCESS:
            try:
                self._require_postconditions(setup, tuple(measured.events), measured.state)
            except (LatencyJourneyPostconditionError, VoiceCertificationProtocolError):
                observation = LatencyObservation.model_validate(
                    observation.model_dump()
                    | {
                        "outcome": LatencyObservationOutcome.ERROR,
                        "error_type": "JourneyPostconditionError",
                    }
                )
        return VoiceCertificationTurnEvidence(
            observation=observation,
            setup_events=tuple(tuple(turn.events) for turn in setup),
            events=tuple(measured.events),
        )

    def _require_postconditions(
        self,
        setup: Sequence[_VoiceCertificationTurnCapture],
        events: tuple[TurnEvent, ...],
        final_state: VoiceCertificationApplicationSnapshot | None,
    ) -> None:
        contract = self._turn_contract
        initial_state = self._initial_state
        if contract is None or initial_state is None or final_state is None:
            raise VoiceCertificationProtocolError(
                "voice certification journey state evidence is incomplete"
            )
        prepared_state = initial_state if not setup else setup[-1].state
        if prepared_state is None or prepared_state.cart != contract.initial_cart:
            raise LatencyJourneyPostconditionError("journey cart state is invalid after setup")
        if prepared_state.placed_order_count != initial_state.placed_order_count:
            raise LatencyJourneyPostconditionError("latency journey setup committed an order")
        require_latency_journey_postconditions(
            contract,
            events=events,
            actual_cart=final_state.cart,
            placed_orders_before=initial_state.placed_order_count,
            placed_orders_after=final_state.placed_order_count,
        )


class _VoiceCertificationRunBinding(BaseModel):
    model_config = _STRICT

    run_id: AuthorityIdentifier
    deployment_id: ConfigIdentifier
    room_id: AuthorityIdentifier
    controller_identity: AuthorityIdentifier


class VoiceCertificationSampleSpec(BaseModel):
    """One exact sample in the controller-owned methodology schedule."""

    model_config = _STRICT

    schema_version: Literal["1"] = "1"
    sample_id: ConfigIdentifier
    phase: LatencyPhase
    journey_id: ConfigIdentifier | None = None
    journey_contract_fingerprint: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    tier: LatencyTier | None = None
    setup_audio_treatments: tuple[VoiceAudioTreatment, ...] = ()
    audio_treatment: VoiceAudioTreatment | None = None
    required_operations: tuple[DurabilityOperation, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_phase_shape(self) -> Self:
        if self.phase is LatencyPhase.STARTUP:
            if (
                self.journey_id is not None
                or self.journey_contract_fingerprint is not None
                or self.tier is not None
                or self.setup_audio_treatments
                or self.audio_treatment is not None
            ):
                raise ValueError("startup certification samples cannot name a journey")
        elif (
            self.journey_id is None
            or self.journey_contract_fingerprint is None
            or self.tier is None
            or self.audio_treatment is None
        ):
            raise ValueError("turn certification samples require a typed journey")
        return self


class VoiceCertificationDispatchDirective(BaseModel):
    """Controller-authored sample authority carried by the server-created dispatch."""

    model_config = _STRICT

    schema_version: Literal["3"] = "3"
    methodology_schema_version: Literal["5"]
    run_id: AuthorityIdentifier
    methodology_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    journey_corpus_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_contract_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    application_contract_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    deployment_id: ConfigIdentifier
    controller_identity: AuthorityIdentifier
    measurement_surface: Literal[LatencyMeasurementSurface.VOICE_PROCESSING]
    result_rpc_timeout_seconds: float = Field(gt=0)
    result_rpc_retries: int = Field(ge=0, le=3)
    result_rpc_retry_backoff_seconds: float = Field(gt=0)
    sample_timeout_seconds: float = Field(gt=0)
    sample: VoiceCertificationSampleSpec


class VoiceCertificationJobBinding(BaseModel):
    """Server-observed transport authority assigned to one sample."""

    model_config = _STRICT

    schema_version: Literal["1"] = "1"
    sample_id: ConfigIdentifier
    room_id: AuthorityIdentifier
    assignment_id: AuthorityIdentifier
    job_id: AuthorityIdentifier
    worker_id: AuthorityIdentifier
    worker_participant_identity: AuthorityIdentifier


class _VoiceCertificationDispatchReservation(BaseModel):
    model_config = _STRICT

    sample_id: ConfigIdentifier
    assignment_id: AuthorityIdentifier


class VoiceCertificationControllerTransport(Protocol):
    """Live transport operations required by the deterministic controller runner."""

    async def create_dispatch(self, directive: VoiceCertificationDispatchDirective) -> str: ...

    async def publish_audio(self, prepared: PreparedVoiceAudio) -> None: ...

    async def close_dispatch(
        self,
        directive: VoiceCertificationDispatchDirective,
        assignment_id: str | None,
        worker_participant_identity: str | None,
    ) -> None: ...


class VoiceCertificationSampleAssignment(BaseModel):
    """Controller-issued, job-readable authority for exactly one scheduled sample."""

    model_config = _STRICT

    schema_version: Literal["3"] = "3"
    methodology_schema_version: Literal["5"]
    run_id: AuthorityIdentifier
    methodology_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    journey_corpus_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_contract_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    application_contract_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    deployment_id: ConfigIdentifier
    controller_identity: AuthorityIdentifier
    measurement_surface: Literal[LatencyMeasurementSurface.VOICE_PROCESSING]
    result_rpc_timeout_seconds: float = Field(gt=0)
    result_rpc_retries: int = Field(ge=0, le=3)
    result_rpc_retry_backoff_seconds: float = Field(gt=0)
    sample_timeout_seconds: float = Field(gt=0)
    sample: VoiceCertificationSampleSpec
    job: VoiceCertificationJobBinding

    @model_validator(mode="after")
    def validate_sample_binding(self) -> Self:
        if self.sample.sample_id != self.job.sample_id:
            raise ValueError("sample assignment and job ids must match")
        return self

    @classmethod
    def from_directive(
        cls,
        directive: VoiceCertificationDispatchDirective,
        job: VoiceCertificationJobBinding,
    ) -> VoiceCertificationSampleAssignment:
        return cls(**directive.model_dump(), job=job)


class VoiceCertificationSampleEnvelope(BaseModel):
    """One job-owned observation and its complete certification binding."""

    model_config = _STRICT

    schema_version: Literal["2"] = "2"
    run_id: AuthorityIdentifier
    sample_id: ConfigIdentifier
    methodology_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    journey_corpus_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_contract_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    application_contract_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    deployment_id: ConfigIdentifier
    room_id: AuthorityIdentifier
    assignment_id: AuthorityIdentifier
    job_id: AuthorityIdentifier
    worker_id: AuthorityIdentifier
    controller_identity: AuthorityIdentifier
    worker_participant_identity: AuthorityIdentifier
    measurement_surface: LatencyMeasurementSurface
    observation: LatencyObservation

    @model_validator(mode="after")
    def validate_observation_identity(self) -> Self:
        if self.observation.sample_id != self.sample_id:
            raise ValueError("sample envelope and observation ids must match")
        return self

    @classmethod
    def from_assignment(
        cls,
        assignment: VoiceCertificationSampleAssignment,
        observation: LatencyObservation,
    ) -> VoiceCertificationSampleEnvelope:
        """Build the response without re-authoring any controller or transport binding."""
        return cls(
            run_id=assignment.run_id,
            sample_id=assignment.sample.sample_id,
            methodology_fingerprint=assignment.methodology_fingerprint,
            journey_corpus_fingerprint=assignment.journey_corpus_fingerprint,
            runtime_contract_fingerprint=assignment.runtime_contract_fingerprint,
            application_contract_fingerprint=assignment.application_contract_fingerprint,
            deployment_id=assignment.deployment_id,
            room_id=assignment.job.room_id,
            assignment_id=assignment.job.assignment_id,
            job_id=assignment.job.job_id,
            worker_id=assignment.job.worker_id,
            controller_identity=assignment.controller_identity,
            worker_participant_identity=assignment.job.worker_participant_identity,
            measurement_surface=assignment.measurement_surface,
            observation=observation,
        )


class VoiceCertificationSampleAck(BaseModel):
    """Deterministic acknowledgement returned for an accepted sample."""

    model_config = _STRICT

    schema_version: Literal["1"] = "1"
    run_id: AuthorityIdentifier
    sample_id: ConfigIdentifier
    accepted: Literal[True] = True


class VoiceCertificationReadyAck(BaseModel):
    """Deterministic acknowledgement that a sample job is ready for caller audio."""

    model_config = _STRICT

    schema_version: Literal["1"] = "1"
    run_id: AuthorityIdentifier
    sample_id: ConfigIdentifier
    ready: Literal[True] = True


class VoiceCertificationProgressEnvelope(BaseModel):
    """One authenticated setup-response boundary within an assigned turn sample."""

    model_config = _STRICT

    schema_version: Literal["1"] = "1"
    assignment: VoiceCertificationSampleAssignment
    completed_setup_turns: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_setup_boundary(self) -> Self:
        if self.assignment.sample.phase is not LatencyPhase.TURN:
            raise ValueError("startup samples cannot report setup progress")
        if self.completed_setup_turns > len(self.assignment.sample.setup_audio_treatments):
            raise ValueError("setup progress exceeds the assigned setup schedule")
        return self


class VoiceCertificationProgressAck(BaseModel):
    model_config = _STRICT

    schema_version: Literal["1"] = "1"
    run_id: AuthorityIdentifier
    sample_id: ConfigIdentifier
    completed_setup_turns: int = Field(ge=1)
    accepted: Literal[True] = True


async def _perform_certification_rpc(
    participant: rtc.LocalParticipant,
    assignment: VoiceCertificationSampleAssignment,
    *,
    method: str,
    payload: str,
) -> str:
    response: str | None = None
    for attempt in range(assignment.result_rpc_retries + 1):
        try:
            response = await participant.perform_rpc(
                destination_identity=assignment.controller_identity,
                method=method,
                payload=payload,
                response_timeout=assignment.result_rpc_timeout_seconds,
            )
        except rtc.RpcError as exc:
            if attempt >= assignment.result_rpc_retries or exc.code not in _RETRYABLE_RPC_ERRORS:
                raise VoiceCertificationProtocolError("voice certification RPC failed") from exc
            await asyncio.sleep(assignment.result_rpc_retry_backoff_seconds)
        else:
            break
    if response is None:
        raise VoiceCertificationProtocolError("voice certification RPC returned no response")
    return response


async def announce_voice_certification_ready(
    participant: rtc.LocalParticipant,
    assignment: VoiceCertificationSampleAssignment,
) -> VoiceCertificationReadyAck:
    """Tell the controller that the exact assigned job can now admit caller audio."""
    response = await _perform_certification_rpc(
        participant,
        assignment,
        method=VOICE_CERTIFICATION_READY_RPC,
        payload=assignment.model_dump_json(),
    )
    try:
        ack = VoiceCertificationReadyAck.model_validate_json(response)
    except ValidationError as exc:
        raise VoiceCertificationProtocolError(
            "voice certification readiness acknowledgement is invalid"
        ) from exc
    if ack.run_id != assignment.run_id or ack.sample_id != assignment.sample.sample_id:
        raise VoiceCertificationProtocolError(
            "voice certification readiness acknowledgement has the wrong binding"
        )
    return ack


async def announce_voice_certification_progress(
    participant: rtc.LocalParticipant,
    assignment: VoiceCertificationSampleAssignment,
    *,
    completed_setup_turns: int,
) -> VoiceCertificationProgressAck:
    envelope = VoiceCertificationProgressEnvelope(
        assignment=assignment,
        completed_setup_turns=completed_setup_turns,
    )
    response = await _perform_certification_rpc(
        participant,
        assignment,
        method=VOICE_CERTIFICATION_PROGRESS_RPC,
        payload=envelope.model_dump_json(),
    )
    try:
        ack = VoiceCertificationProgressAck.model_validate_json(response)
    except ValidationError as exc:
        raise VoiceCertificationProtocolError(
            "voice certification progress acknowledgement is invalid"
        ) from exc
    if (
        ack.run_id != assignment.run_id
        or ack.sample_id != assignment.sample.sample_id
        or ack.completed_setup_turns != completed_setup_turns
    ):
        raise VoiceCertificationProtocolError(
            "voice certification progress acknowledgement has the wrong binding"
        )
    return ack


async def submit_voice_certification_sample(
    participant: rtc.LocalParticipant,
    assignment: VoiceCertificationSampleAssignment,
    observation: LatencyObservation,
) -> VoiceCertificationSampleAck:
    """Send one immutable payload, retrying only bounded transient RPC failures."""
    envelope = VoiceCertificationSampleEnvelope.from_assignment(assignment, observation)
    payload = envelope.model_dump_json()
    response = await _perform_certification_rpc(
        participant,
        assignment,
        method=VOICE_CERTIFICATION_RESULT_RPC,
        payload=payload,
    )
    try:
        ack = VoiceCertificationSampleAck.model_validate_json(response)
    except ValidationError as exc:
        raise VoiceCertificationProtocolError(
            "voice certification result acknowledgement is invalid"
        ) from exc
    if ack.run_id != assignment.run_id or ack.sample_id != assignment.sample.sample_id:
        raise VoiceCertificationProtocolError(
            "voice certification result acknowledgement has the wrong binding"
        )
    return ack


def build_voice_certification_schedule(
    methodology: DurableLatencyMethodology,
) -> tuple[VoiceCertificationSampleSpec, ...]:
    """Derive the exact per-job schedule from deployment voice methodology."""
    if (
        methodology.environment is not LatencyEnvironment.DEPLOYMENT
        or methodology.measurement_surface is not LatencyMeasurementSurface.VOICE_PROCESSING
    ):
        raise VoiceCertificationProtocolError(
            "voice certification requires deployment voice-processing methodology"
        )
    schedule = [
        VoiceCertificationSampleSpec(
            sample_id=f"sample-startup-{index + 1:04d}",
            phase=LatencyPhase.STARTUP,
            required_operations=methodology.required_startup_operations,
        )
        for index in range(methodology.startup_samples)
    ]
    for journey in methodology.journeys:
        schedule.extend(
            VoiceCertificationSampleSpec(
                sample_id=f"sample-turn-{journey.journey_id}-{index + 1:04d}",
                phase=LatencyPhase.TURN,
                journey_id=journey.journey_id,
                journey_contract_fingerprint=journey.journey_contract_fingerprint,
                tier=journey.tier,
                setup_audio_treatments=journey.setup_audio_treatments,
                audio_treatment=journey.audio_treatment,
                required_operations=journey.required_operations,
            )
            for index in range(methodology.samples_per_journey)
        )
    return tuple(schedule)


def build_voice_certification_warmup_schedule(
    methodology: DurableLatencyMethodology,
) -> tuple[VoiceCertificationSampleSpec, ...]:
    """Derive the unreported warmup jobs required before measured samples."""
    build_voice_certification_schedule(methodology)
    return tuple(
        VoiceCertificationSampleSpec(
            sample_id=f"warmup-turn-{journey.journey_id}-{index + 1:04d}",
            phase=LatencyPhase.TURN,
            journey_id=journey.journey_id,
            journey_contract_fingerprint=journey.journey_contract_fingerprint,
            tier=journey.tier,
            setup_audio_treatments=journey.setup_audio_treatments,
            audio_treatment=journey.audio_treatment,
            required_operations=journey.required_operations,
        )
        for journey in methodology.journeys
        for index in range(methodology.warmup_runs)
    )


class VoiceCertificationController:
    """Collect one authenticated observation for every frozen scheduled sample."""

    def __init__(
        self,
        *,
        methodology: DurableLatencyMethodology,
        run_id: str,
        deployment_id: str,
        room_id: str,
        controller_identity: str,
    ) -> None:
        if methodology.schema_version != "5" or methodology.application_contract is None:
            raise VoiceCertificationProtocolError(
                "voice certification controller requires schema-5 application identity"
            )
        try:
            run = _VoiceCertificationRunBinding(
                run_id=run_id,
                deployment_id=deployment_id,
                room_id=room_id,
                controller_identity=controller_identity,
            )
        except ValidationError as exc:
            raise VoiceCertificationProtocolError(
                "voice certification run binding is invalid"
            ) from exc
        self._schedule = build_voice_certification_schedule(methodology)
        self._warmup_schedule = build_voice_certification_warmup_schedule(methodology)
        rpc_timeout = methodology.result_rpc_timeout_seconds
        rpc_retries = methodology.result_rpc_retries
        rpc_retry_backoff = methodology.result_rpc_retry_backoff_seconds
        job_ready_timeout = methodology.job_ready_timeout_seconds
        cleanup_timeout = methodology.dispatch_cleanup_timeout_seconds
        if (
            rpc_timeout is None
            or rpc_retries is None
            or rpc_retry_backoff is None
            or job_ready_timeout is None
            or cleanup_timeout is None
        ):
            raise VoiceCertificationProtocolError(
                "voice certification methodology has incomplete orchestration deadlines"
            )
        self._methodology = methodology
        self._methodology_fingerprint = methodology_fingerprint(methodology)
        self._application_contract_fingerprint = voice_application_contract_fingerprint(
            methodology.application_contract
        )
        evidence_schema_version = latency_evidence_schema_version(methodology)
        if evidence_schema_version == "1":
            raise VoiceCertificationProtocolError(
                "voice certification cannot use reasoning-graph evidence schema"
            )
        self._evidence_schema_version: Literal["2", "3"] = evidence_schema_version
        self._run_id = run.run_id
        self._deployment_id = run.deployment_id
        self._room_id = run.room_id
        self._controller_identity = run.controller_identity
        self._result_rpc_timeout_seconds = rpc_timeout
        self._result_rpc_retries = rpc_retries
        self._result_rpc_retry_backoff_seconds = rpc_retry_backoff
        self._job_ready_timeout_seconds = job_ready_timeout
        self._dispatch_cleanup_timeout_seconds = cleanup_timeout
        self._samples = {
            sample.sample_id: sample for sample in (*self._warmup_schedule, *self._schedule)
        }
        self._dispatch_ids: dict[str, str] = {}
        self._bindings: dict[str, VoiceCertificationJobBinding] = {}
        self._ready_payloads: dict[str, str] = {}
        self._progress_payloads: dict[tuple[str, int], str] = {}
        self._progress_counts: dict[str, int] = {}
        self._resolutions: dict[str, _VoiceSampleResolution] = {}
        self._sample_events = {sample.sample_id: asyncio.Event() for sample in self._schedule}
        self._sample_events.update(
            {sample.sample_id: asyncio.Event() for sample in self._warmup_schedule}
        )
        self._ready_events = {
            sample.sample_id: asyncio.Event()
            for sample in (*self._warmup_schedule, *self._schedule)
        }
        self._progress_events = {
            sample.sample_id: asyncio.Event()
            for sample in (*self._warmup_schedule, *self._schedule)
        }
        self._report: DurableLatencyReport | None = None

    @property
    def job_ready_timeout_seconds(self) -> float:
        return self._job_ready_timeout_seconds

    @property
    def dispatch_cleanup_timeout_seconds(self) -> float:
        return self._dispatch_cleanup_timeout_seconds

    @property
    def sample_result_timeout_seconds(self) -> float:
        """Bound sample execution plus the worker's worst-case result RPC policy."""
        return (
            self._methodology.sample_timeout_seconds
            + self._result_rpc_timeout_seconds * (self._result_rpc_retries + 1)
            + self._result_rpc_retry_backoff_seconds * self._result_rpc_retries
        )

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def deployment_id(self) -> str:
        return self._deployment_id

    @property
    def controller_identity(self) -> str:
        return self._controller_identity

    @property
    def methodology_fingerprint(self) -> str:
        return self._methodology_fingerprint

    @property
    def runtime_contract_fingerprint(self) -> str:
        return self._methodology.runtime_contract_fingerprint

    @property
    def application_contract_fingerprint(self) -> str:
        return self._application_contract_fingerprint

    @property
    def methodology(self) -> DurableLatencyMethodology:
        return self._methodology

    @property
    def schedule(self) -> tuple[VoiceCertificationSampleSpec, ...]:
        return self._schedule

    @property
    def warmup_schedule(self) -> tuple[VoiceCertificationSampleSpec, ...]:
        return self._warmup_schedule

    @property
    def received_count(self) -> int:
        return sum(resolution.kind == "accepted" for resolution in self._resolutions.values())

    @property
    def ready_sample_ids(self) -> tuple[str, ...]:
        return tuple(
            sample.sample_id
            for sample in (*self._warmup_schedule, *self._schedule)
            if sample.sample_id in self._ready_payloads
        )

    @property
    def resolved_count(self) -> int:
        return len(self._resolutions)

    @property
    def pending_sample_ids(self) -> tuple[str, ...]:
        return tuple(
            sample.sample_id
            for sample in self._schedule
            if sample.sample_id not in self._resolutions
        )

    def bind_job(self, binding: VoiceCertificationJobBinding) -> None:
        """Record the controller-observed assignment before accepting its RPC."""
        self._require_open()
        if binding.sample_id not in self._samples:
            raise VoiceCertificationProtocolError("job sample is outside the frozen schedule")
        if binding.room_id != self._room_id:
            raise VoiceCertificationProtocolError("job binding targets the wrong room")
        existing = self._bindings.get(binding.sample_id)
        if existing is not None and existing != binding:
            raise VoiceCertificationProtocolError("sample is already bound to another job")
        self._bindings[binding.sample_id] = binding

    def bind_dispatch(self, sample_id: str, assignment_id: str) -> None:
        """Bind one sample to the dispatch ID returned by the LiveKit service."""
        self._require_open()
        try:
            reservation = _VoiceCertificationDispatchReservation(
                sample_id=sample_id,
                assignment_id=assignment_id,
            )
        except ValidationError as exc:
            raise VoiceCertificationProtocolError(
                "voice certification dispatch binding is invalid"
            ) from exc
        if reservation.sample_id not in self._samples:
            raise VoiceCertificationProtocolError("dispatch is outside the frozen schedule")
        existing = self._dispatch_ids.get(reservation.sample_id)
        if existing is not None and existing != reservation.assignment_id:
            raise VoiceCertificationProtocolError("sample is already bound to another dispatch")
        self._dispatch_ids[reservation.sample_id] = reservation.assignment_id

    def directive(self, sample_id: str) -> VoiceCertificationDispatchDirective:
        """Build the immutable authority placed on one server-created dispatch."""
        sample = self._samples.get(sample_id)
        if sample is None:
            raise VoiceCertificationProtocolError("job sample is outside the frozen schedule")
        return VoiceCertificationDispatchDirective(
            methodology_schema_version=self._methodology.schema_version,
            run_id=self._run_id,
            methodology_fingerprint=self._methodology_fingerprint,
            journey_corpus_fingerprint=self._methodology.journey_corpus_fingerprint,
            runtime_contract_fingerprint=self._methodology.runtime_contract_fingerprint,
            application_contract_fingerprint=self._application_contract_fingerprint,
            deployment_id=self._deployment_id,
            controller_identity=self._controller_identity,
            measurement_surface=LatencyMeasurementSurface.VOICE_PROCESSING,
            result_rpc_timeout_seconds=self._result_rpc_timeout_seconds,
            result_rpc_retries=self._result_rpc_retries,
            result_rpc_retry_backoff_seconds=self._result_rpc_retry_backoff_seconds,
            sample_timeout_seconds=self._methodology.sample_timeout_seconds,
            sample=sample,
        )

    def assignment(self, sample_id: str) -> VoiceCertificationSampleAssignment:
        sample = self._samples.get(sample_id)
        binding = self._bindings.get(sample_id)
        if sample is None:
            raise VoiceCertificationProtocolError("job sample is outside the frozen schedule")
        if binding is None:
            raise VoiceCertificationProtocolError("sample is not assigned to a trusted job")
        return VoiceCertificationSampleAssignment.from_directive(
            self.directive(sample_id),
            binding,
        )

    def handle_rpc(self, invocation: rtc.RpcInvocationData) -> str:
        """Validate a LiveKit-authenticated RPC submission and return its acknowledgement."""
        return self.submit(invocation.payload, caller_identity=invocation.caller_identity)

    def handle_ready_rpc(self, invocation: rtc.RpcInvocationData) -> str:
        """Validate an authenticated worker-readiness announcement."""
        return self.submit_ready(
            invocation.payload,
            caller_identity=invocation.caller_identity,
        )

    def handle_progress_rpc(self, invocation: rtc.RpcInvocationData) -> str:
        """Validate one authenticated setup-response boundary."""
        return self.submit_progress(
            invocation.payload,
            caller_identity=invocation.caller_identity,
        )

    def submit_progress(self, payload: str, *, caller_identity: str) -> str:
        try:
            envelope = VoiceCertificationProgressEnvelope.model_validate_json(payload)
        except ValidationError as exc:
            raise VoiceCertificationProtocolError(
                "voice certification progress payload is invalid"
            ) from exc
        assignment = envelope.assignment
        sample_id = assignment.sample.sample_id
        binding = self._bindings.get(sample_id)
        if binding is None or sample_id not in self._ready_payloads:
            raise VoiceCertificationProtocolError("sample is not ready for setup progress")
        if caller_identity != binding.worker_participant_identity:
            raise VoiceCertificationProtocolError(
                "setup progress did not come from its authenticated participant"
            )
        if assignment != self.assignment(sample_id):
            raise VoiceCertificationProtocolError("setup progress has the wrong assignment binding")
        key = (sample_id, envelope.completed_setup_turns)
        existing_payload = self._progress_payloads.get(key)
        if existing_payload is not None:
            if existing_payload != payload:
                raise VoiceCertificationProtocolError(
                    "setup progress has a divergent duplicate payload"
                )
        else:
            self._require_open()
            expected = self._progress_counts.get(sample_id, 0) + 1
            if envelope.completed_setup_turns != expected:
                raise VoiceCertificationProtocolError(
                    "setup progress is outside the next scheduled boundary"
                )
            self._progress_payloads[key] = payload
            self._progress_counts[sample_id] = envelope.completed_setup_turns
            self._progress_events[sample_id].set()
        return VoiceCertificationProgressAck(
            run_id=self._run_id,
            sample_id=sample_id,
            completed_setup_turns=envelope.completed_setup_turns,
        ).model_dump_json()

    def submit_ready(self, payload: str, *, caller_identity: str) -> str:
        try:
            assignment = VoiceCertificationSampleAssignment.model_validate_json(payload)
        except ValidationError as exc:
            raise VoiceCertificationProtocolError(
                "voice certification readiness payload is invalid"
            ) from exc
        sample_id = assignment.sample.sample_id
        if sample_id not in self._samples:
            raise VoiceCertificationProtocolError("readiness is outside the frozen schedule")
        resolution = self._resolutions.get(sample_id)
        if resolution is not None and resolution.kind == "timeout":
            raise VoiceCertificationProtocolError("sample already timed out")
        binding = self._bindings.get(sample_id)
        if binding is None:
            expected_dispatch = self._dispatch_ids.get(sample_id)
            if expected_dispatch is None:
                raise VoiceCertificationProtocolError("sample is not assigned to a trusted job")
            if assignment.job.assignment_id != expected_dispatch:
                raise VoiceCertificationProtocolError(
                    "readiness came from the wrong server-created dispatch"
                )
            if assignment.job.room_id != self._room_id:
                raise VoiceCertificationProtocolError("readiness came from the wrong room")
            if caller_identity != assignment.job.worker_participant_identity:
                raise VoiceCertificationProtocolError(
                    "readiness did not come from its authenticated participant"
                )
            expected_assignment = VoiceCertificationSampleAssignment.from_directive(
                self.directive(sample_id),
                assignment.job,
            )
            if assignment != expected_assignment:
                raise VoiceCertificationProtocolError("readiness has the wrong assignment binding")
            self.bind_job(assignment.job)
            binding = assignment.job
        if caller_identity != binding.worker_participant_identity:
            raise VoiceCertificationProtocolError(
                "readiness did not come from its authenticated participant"
            )
        existing_payload = self._ready_payloads.get(sample_id)
        if existing_payload is not None:
            if existing_payload != payload:
                raise VoiceCertificationProtocolError("readiness has a divergent duplicate payload")
        else:
            if assignment != self.assignment(sample_id):
                raise VoiceCertificationProtocolError("readiness has the wrong assignment binding")
            self._require_open()
            self._ready_payloads[sample_id] = payload
            self._ready_events[sample_id].set()
        return VoiceCertificationReadyAck(
            run_id=self._run_id,
            sample_id=sample_id,
        ).model_dump_json()

    async def wait_for_ready(self, sample_id: str) -> VoiceCertificationSampleAssignment:
        if sample_id not in self._samples:
            raise VoiceCertificationProtocolError("readiness is outside the frozen schedule")
        if sample_id not in self._dispatch_ids and sample_id not in self._bindings:
            raise VoiceCertificationProtocolError("sample has no server-created dispatch")
        if sample_id not in self._ready_payloads:
            await self._ready_events[sample_id].wait()
        return self.assignment(sample_id)

    def submit(self, payload: str, *, caller_identity: str) -> str:
        try:
            envelope = VoiceCertificationSampleEnvelope.model_validate_json(payload)
        except ValidationError as exc:
            raise VoiceCertificationProtocolError(
                "voice certification sample envelope is invalid"
            ) from exc
        sample = self._samples.get(envelope.sample_id)
        if sample is None:
            raise VoiceCertificationProtocolError("sample is outside the frozen schedule")
        binding = self._bindings.get(envelope.sample_id)
        if binding is None:
            raise VoiceCertificationProtocolError("sample is not assigned to a trusted job")
        if caller_identity != envelope.worker_participant_identity:
            raise VoiceCertificationProtocolError(
                "sample did not come from its authenticated participant"
            )
        self._validate_run_binding(envelope)
        self._validate_job_binding(envelope, binding)
        self._validate_sample_shape(envelope.observation, sample)
        existing = self._resolutions.get(envelope.sample_id)
        if existing is not None:
            if existing.kind == "timeout":
                raise VoiceCertificationProtocolError("sample already timed out")
            if existing.kind == "failure":
                raise VoiceCertificationProtocolError("sample already failed")
            if existing.payload != payload:
                raise VoiceCertificationProtocolError("sample has a divergent duplicate envelope")
        else:
            self._require_open()
            self._resolutions[envelope.sample_id] = _VoiceSampleResolution(
                kind="accepted",
                observation=envelope.observation,
                payload=payload,
            )
            self._sample_events[envelope.sample_id].set()
        return VoiceCertificationSampleAck(
            run_id=self._run_id,
            sample_id=envelope.sample_id,
        ).model_dump_json()

    def record_timeout(self, sample_id: str) -> LatencyObservation:
        sample = self._samples.get(sample_id)
        if sample is None:
            raise VoiceCertificationProtocolError("timeout is outside the frozen schedule")
        existing = self._resolutions.get(sample_id)
        if existing is not None:
            return existing.observation
        self._require_open()
        observation = LatencyObservation(
            schema_version=self._evidence_schema_version,
            sample_id=sample_id,
            phase=sample.phase,
            thermal_state=(
                ThermalState.COLD if sample.phase is LatencyPhase.STARTUP else ThermalState.WARM
            ),
            outcome=LatencyObservationOutcome.ERROR,
            error_type="TimeoutError",
            journey_id=sample.journey_id,
            tier=sample.tier,
            components=(),
        )
        self._resolutions[sample_id] = _VoiceSampleResolution(
            kind="timeout",
            observation=observation,
        )
        self._sample_events[sample_id].set()
        return observation

    def record_failure(self, sample_id: str, failure: Exception) -> LatencyObservation:
        """Retain one controller-observed sample failure without fabricating timing."""
        sample = self._samples.get(sample_id)
        if sample is None:
            raise VoiceCertificationProtocolError("failure is outside the frozen schedule")
        existing = self._resolutions.get(sample_id)
        if existing is not None:
            return existing.observation
        self._require_open()
        observation = LatencyObservation(
            schema_version=self._evidence_schema_version,
            sample_id=sample_id,
            phase=sample.phase,
            thermal_state=(
                ThermalState.COLD if sample.phase is LatencyPhase.STARTUP else ThermalState.WARM
            ),
            outcome=LatencyObservationOutcome.ERROR,
            error_type=type(failure).__name__,
            journey_id=sample.journey_id,
            tier=sample.tier,
            components=(),
        )
        self._resolutions[sample_id] = _VoiceSampleResolution(
            kind="failure",
            observation=observation,
        )
        self._sample_events[sample_id].set()
        return observation

    async def wait_for_sample(
        self,
        sample_id: str,
        *,
        timeout_seconds: float | None = None,
    ) -> LatencyObservation:
        """Wait for an assigned job and retain a controller-side timeout."""
        if sample_id not in self._samples:
            raise VoiceCertificationProtocolError("sample is outside the frozen schedule")
        if sample_id not in self._bindings:
            raise VoiceCertificationProtocolError("sample is not assigned to a trusted job")
        existing = self._resolutions.get(sample_id)
        if existing is not None:
            return existing.observation
        timeout = self._methodology.sample_timeout_seconds
        if timeout_seconds is not None:
            timeout = timeout_seconds
        try:
            async with asyncio.timeout(timeout):
                await self._sample_events[sample_id].wait()
        except TimeoutError:
            existing = self._resolutions.get(sample_id)
            if existing is None:
                return self.record_timeout(sample_id)
        return self._resolutions[sample_id].observation

    async def wait_for_setup_progress(
        self,
        sample_id: str,
        completed_setup_turns: int,
    ) -> None:
        sample = self._samples.get(sample_id)
        if sample is None:
            raise VoiceCertificationProtocolError("progress is outside the frozen schedule")
        if not 1 <= completed_setup_turns <= len(sample.setup_audio_treatments):
            raise VoiceCertificationProtocolError("progress is outside the setup schedule")
        while self._progress_counts.get(sample_id, 0) < completed_setup_turns:
            self._progress_events[sample_id].clear()
            await self._progress_events[sample_id].wait()

    def observation(self, sample_id: str) -> LatencyObservation:
        try:
            return self._resolutions[sample_id].observation
        except KeyError as exc:
            raise VoiceCertificationProtocolError("sample has no observation") from exc

    def build_report(self, *, run_at: datetime) -> DurableLatencyReport:
        if self._report is not None:
            if self._report.run_at != run_at:
                raise VoiceCertificationProtocolError(
                    "finalized certification timestamp cannot change"
                )
            return self._report
        if self.pending_sample_ids:
            raise VoiceCertificationProtocolError("voice certification run is incomplete")
        self._report = build_latency_report(
            self._methodology,
            tuple(self._resolutions[sample.sample_id].observation for sample in self._schedule),
            run_at=run_at,
            deployment_id=self._deployment_id,
        )
        return self._report

    def build_run(self, *, run_at: datetime) -> DurableLatencyCertificationRun:
        report = self.build_report(run_at=run_at)
        return DurableLatencyCertificationRun(
            schema_version=self._evidence_schema_version,
            run_at=run_at,
            deployment_id=self._deployment_id,
            methodology_fingerprint=self._methodology_fingerprint,
            methodology=self._methodology,
            outcome=LatencyCertificationOutcome.COMPLETED,
            report=report,
        )

    def _require_open(self) -> None:
        if self._report is not None:
            raise VoiceCertificationProtocolError("voice certification run is finalized")

    def _validate_run_binding(self, envelope: VoiceCertificationSampleEnvelope) -> None:
        if (
            envelope.run_id != self._run_id
            or envelope.methodology_fingerprint != self._methodology_fingerprint
            or envelope.journey_corpus_fingerprint != self._methodology.journey_corpus_fingerprint
            or envelope.runtime_contract_fingerprint
            != self._methodology.runtime_contract_fingerprint
            or envelope.application_contract_fingerprint != self._application_contract_fingerprint
            or envelope.deployment_id != self._deployment_id
            or envelope.controller_identity != self._controller_identity
        ):
            raise VoiceCertificationProtocolError("sample has the wrong run binding")
        if envelope.measurement_surface is not LatencyMeasurementSurface.VOICE_PROCESSING:
            raise VoiceCertificationProtocolError("sample has the wrong measurement surface")

    @staticmethod
    def _validate_job_binding(
        envelope: VoiceCertificationSampleEnvelope,
        binding: VoiceCertificationJobBinding,
    ) -> None:
        if (
            envelope.room_id != binding.room_id
            or envelope.assignment_id != binding.assignment_id
            or envelope.job_id != binding.job_id
            or envelope.worker_id != binding.worker_id
            or envelope.worker_participant_identity != binding.worker_participant_identity
        ):
            raise VoiceCertificationProtocolError("sample has the wrong job binding")

    def _validate_sample_shape(
        self,
        observation: LatencyObservation,
        sample: VoiceCertificationSampleSpec,
    ) -> None:
        if (
            observation.schema_version != self._evidence_schema_version
            or observation.sample_id != sample.sample_id
            or observation.phase is not sample.phase
            or observation.journey_id != sample.journey_id
            or observation.tier is not sample.tier
        ):
            raise VoiceCertificationProtocolError("observation does not match the sample schedule")


def _voice_abort_run(
    controller: VoiceCertificationController,
    *,
    run_at: datetime,
    stage: LatencyAbortStage,
    sample: VoiceCertificationSampleSpec,
    error_type: str,
    cleanup_error_type: str | None = None,
) -> DurableLatencyCertificationRun:
    return DurableLatencyCertificationRun(
        schema_version=latency_evidence_schema_version(controller.methodology),
        run_at=run_at,
        deployment_id=controller.deployment_id,
        methodology_fingerprint=controller.methodology_fingerprint,
        methodology=controller.methodology,
        outcome=LatencyCertificationOutcome.ABORTED,
        abort=LatencyCertificationAbort(
            stage=stage,
            sample_id=sample.sample_id,
            journey_id=sample.journey_id,
            error_type=error_type,
            cleanup_error_type=cleanup_error_type,
        ),
    )


def abort_voice_controller_cleanup(
    run: DurableLatencyCertificationRun,
    failure: BaseException,
) -> DurableLatencyCertificationRun:
    """Retain a non-authorizing artifact when controller-owned teardown fails."""
    error_type = type(failure).__name__
    if run.outcome is LatencyCertificationOutcome.ABORTED:
        assert run.abort is not None
        abort = LatencyCertificationAbort.model_validate(
            run.abort.model_dump()
            | {
                "cleanup_error_type": (
                    error_type if run.abort.cleanup_error_type is None else "ExceptionGroup"
                )
            }
        )
    else:
        abort = LatencyCertificationAbort(
            stage=LatencyAbortStage.VOICE_CONTROLLER_CLEANUP,
            error_type=error_type,
        )
    return DurableLatencyCertificationRun(
        schema_version=run.schema_version,
        run_at=run.run_at,
        deployment_id=run.deployment_id,
        methodology_fingerprint=run.methodology_fingerprint,
        methodology=run.methodology,
        outcome=LatencyCertificationOutcome.ABORTED,
        abort=abort,
    )


async def run_voice_certification_controller(
    controller: VoiceCertificationController,
    transport: VoiceCertificationControllerTransport,
    audio_assets: Mapping[str, PreparedVoiceAudio],
    *,
    run_at: datetime,
) -> DurableLatencyCertificationRun:
    """Run the frozen single-room schedule and return completed or aborted evidence."""
    methodology = controller.methodology
    if methodology.concurrency != 1:
        raise VoiceCertificationProtocolError(
            "single-room voice certification requires concurrency 1"
        )

    ordered_samples = (*controller.warmup_schedule, *controller.schedule)
    warmup_ids = {sample.sample_id for sample in controller.warmup_schedule}
    aborted = await _execute_voice_samples(
        controller,
        transport,
        audio_assets,
        ordered_samples=ordered_samples,
        warmup_ids=warmup_ids,
        run_at=run_at,
    )
    if aborted is not None:
        return aborted
    return controller.build_run(run_at=run_at)


async def run_voice_certification_smoke(
    controller: VoiceCertificationController,
    transport: VoiceCertificationControllerTransport,
    audio_assets: Mapping[str, PreparedVoiceAudio],
    *,
    journey_id: str,
    run_at: datetime,
) -> LatencyObservation:
    """Run one journey through the real protocol without producing activation evidence."""
    methodology = controller.methodology
    if methodology.concurrency != 1:
        raise VoiceCertificationProtocolError(
            "single-room voice certification requires concurrency 1"
        )
    sample = next(
        (
            candidate
            for candidate in controller.schedule
            if candidate.phase is LatencyPhase.TURN and candidate.journey_id == journey_id
        ),
        None,
    )
    if sample is None:
        raise VoiceCertificationProtocolError(
            "voice smoke journey is outside the frozen methodology"
        )
    aborted = await _execute_voice_samples(
        controller,
        transport,
        audio_assets,
        ordered_samples=(sample,),
        warmup_ids=set(),
        run_at=run_at,
    )
    if aborted is not None:
        assert aborted.abort is not None
        raise VoiceCertificationProtocolError(
            f"voice smoke failed at {aborted.abort.stage.value}: {aborted.abort.error_type}"
        )
    observation = controller.observation(sample.sample_id)
    if observation.outcome is not LatencyObservationOutcome.SUCCESS:
        raise VoiceCertificationProtocolError(
            f"voice smoke journey failed with {observation.error_type or 'unknown error'}"
        )
    return observation


async def _execute_voice_samples(
    controller: VoiceCertificationController,
    transport: VoiceCertificationControllerTransport,
    audio_assets: Mapping[str, PreparedVoiceAudio],
    *,
    ordered_samples: Sequence[VoiceCertificationSampleSpec],
    warmup_ids: set[str],
    run_at: datetime,
) -> DurableLatencyCertificationRun | None:
    """Execute a selected schedule through the one dispatch and cleanup implementation."""
    methodology = controller.methodology
    for sample in ordered_samples:
        required_assets = (*sample.setup_audio_treatments, sample.audio_treatment)
        if any(
            treatment is not None and treatment.asset_sha256 not in audio_assets
            for treatment in required_assets
        ):
            return _voice_abort_run(
                controller,
                run_at=run_at,
                stage=LatencyAbortStage.VOICE_PREPARE,
                sample=sample,
                error_type="VoiceCertificationProtocolError",
            )

    for sample in ordered_samples:
        directive = controller.directive(sample.sample_id)
        assignment_id: str | None = None
        worker_identity: str | None = None
        failure: Exception | None = None
        cancellation: asyncio.CancelledError | None = None
        try:
            async with asyncio.timeout(controller.job_ready_timeout_seconds):
                assignment_id = await transport.create_dispatch(directive)
                controller.bind_dispatch(sample.sample_id, assignment_id)
                assignment = await controller.wait_for_ready(sample.sample_id)
                worker_identity = assignment.job.worker_participant_identity
            result_deadline = (
                asyncio.get_running_loop().time() + controller.sample_result_timeout_seconds
            )
            for index, treatment in enumerate(sample.setup_audio_treatments, start=1):
                async with asyncio.timeout_at(result_deadline):
                    await transport.publish_audio(audio_assets[treatment.asset_sha256])
                    await controller.wait_for_setup_progress(sample.sample_id, index)
            if sample.audio_treatment is not None:
                async with asyncio.timeout_at(result_deadline):
                    await transport.publish_audio(audio_assets[sample.audio_treatment.asset_sha256])
            remaining = result_deadline - asyncio.get_running_loop().time()
            observation = await controller.wait_for_sample(
                sample.sample_id,
                timeout_seconds=max(remaining, 1e-9),
            )
            if (
                sample.sample_id in warmup_ids
                and observation.outcome is not LatencyObservationOutcome.SUCCESS
            ):
                raise VoiceCertificationProtocolError(
                    f"voice warmup failed with {observation.error_type or 'unknown error'}"
                )
        except asyncio.CancelledError as exc:
            cancellation = exc
        except Exception as exc:
            failure = exc

        cleanup_failure: Exception | None = None
        deferred_cancellation: asyncio.CancelledError | None = None
        cleanup_task = asyncio.create_task(
            transport.close_dispatch(directive, assignment_id, worker_identity)
        )
        try:
            deferred_cancellation = await _finish_voice_cleanup(
                cleanup_task,
                timeout_seconds=controller.dispatch_cleanup_timeout_seconds,
            )
        except asyncio.CancelledError as exc:
            deferred_cancellation = exc
        except Exception as exc:
            cleanup_failure = exc

        cancellation = cancellation or deferred_cancellation
        if cancellation is not None:
            if cleanup_failure is not None:
                raise cancellation from cleanup_failure
            raise cancellation

        if cleanup_failure is not None:
            return _voice_abort_run(
                controller,
                run_at=run_at,
                stage=LatencyAbortStage.VOICE_CLEANUP,
                sample=sample,
                error_type=type(failure or cleanup_failure).__name__,
                cleanup_error_type=(
                    type(cleanup_failure).__name__ if failure is not None else None
                ),
            )
        if failure is None:
            continue
        if assignment_id is None or sample.sample_id in warmup_ids:
            return _voice_abort_run(
                controller,
                run_at=run_at,
                stage=LatencyAbortStage.VOICE_RUN,
                sample=sample,
                error_type=type(failure).__name__,
            )
        if isinstance(failure, TimeoutError):
            controller.record_timeout(sample.sample_id)
        else:
            controller.record_failure(sample.sample_id, failure)
    return None


async def _finish_voice_cleanup(
    task: asyncio.Task[None],
    *,
    timeout_seconds: float,
) -> asyncio.CancelledError | None:
    """Finish dispatch cleanup while retaining cancellation for the caller."""
    deferred: asyncio.CancelledError | None = None
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    while not task.done():
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            later_cancellation = await _cancel_and_await_voice_cleanup(task)
            cancellation = deferred or later_cancellation
            timeout = TimeoutError("voice certification dispatch cleanup timed out")
            if cancellation is not None:
                raise cancellation from timeout
            raise timeout
        try:
            async with asyncio.timeout(remaining):
                await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            if task.cancelled():
                raise
            if deferred is None:
                deferred = exc
        except TimeoutError:
            later_cancellation = await _cancel_and_await_voice_cleanup(task)
            cancellation = deferred or later_cancellation
            timeout = TimeoutError("voice certification dispatch cleanup timed out")
            if cancellation is not None:
                raise cancellation from timeout
            raise timeout from None
        except Exception:
            break
    task.result()
    return deferred


async def _cancel_and_await_voice_cleanup(
    task: asyncio.Task[None],
) -> asyncio.CancelledError | None:
    """Cancel owned cleanup and wait until the task reaches a terminal state."""
    task.cancel()
    deferred: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            if task.cancelled():
                break
            if deferred is None:
                deferred = exc
        except BaseException:
            break
    if not task.cancelled():
        task.result()
    return deferred


def register_voice_certification_controller(
    participant: rtc.LocalParticipant,
    controller: VoiceCertificationController,
) -> None:
    """Expose the controller's versioned readiness and result RPCs."""
    participant.register_rpc_method(
        VOICE_CERTIFICATION_READY_RPC,
        controller.handle_ready_rpc,
    )
    participant.register_rpc_method(
        VOICE_CERTIFICATION_PROGRESS_RPC,
        controller.handle_progress_rpc,
    )
    participant.register_rpc_method(
        VOICE_CERTIFICATION_RESULT_RPC,
        controller.handle_rpc,
    )


class VoiceCertificationTarget(BaseModel):
    """Deployment-owned production namespace and its isolated synthetic dispatch target."""

    model_config = _STRICT

    schema_version: Literal[1]
    environment: Literal["synthetic"]
    production_agent_name: ConfigIdentifier
    certification_agent_name: ConfigIdentifier
    room_name: AuthorityIdentifier
    controller_participant_identity: AuthorityIdentifier
    merchant_id: ConfigIdentifier

    @model_validator(mode="after")
    def validate_worker_isolation(self) -> Self:
        suffix = "-certification"
        if self.production_agent_name.endswith(suffix):
            raise ValueError("production agent name uses the reserved certification namespace")
        if self.certification_agent_name != f"{self.production_agent_name}{suffix}":
            raise ValueError(
                "certification agent name must use the production name plus the reserved suffix"
            )
        return self


def voice_certification_target_fingerprint(target: VoiceCertificationTarget) -> str:
    canonical = json.dumps(
        target.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


class VoiceCertificationTargetError(RuntimeError):
    """The worker or assigned job does not match its synthetic certification authority."""


def load_voice_certification_target(path: Path) -> VoiceCertificationTarget:
    try:
        payload = json.dumps(load_yaml_layer(path))
        return VoiceCertificationTarget.model_validate_json(payload)
    except (ConfigError, TypeError, ValueError) as exc:
        raise VoiceCertificationTargetError(
            f"voice certification target cannot be loaded: {path}"
        ) from exc


def _load_job_directive(
    job: agent.Job,
    target: VoiceCertificationTarget,
) -> VoiceCertificationDispatchDirective:
    try:
        raw_directive = job.attributes.get(VOICE_CERTIFICATION_SAMPLE_ATTRIBUTE, "")
        if not raw_directive:
            raise VoiceCertificationTargetError("voice certification job has no sample directive")
        directive = VoiceCertificationDispatchDirective.model_validate_json(raw_directive)
    except VoiceCertificationTargetError:
        raise
    except (AttributeError, ValidationError) as exc:
        raise VoiceCertificationTargetError(
            "voice certification job sample directive is invalid"
        ) from exc
    if directive.controller_identity != target.controller_participant_identity:
        raise VoiceCertificationTargetError(
            "voice certification directive has the wrong controller"
        )
    return directive


def validate_certification_job(
    job: agent.Job,
    target: VoiceCertificationTarget,
) -> VoiceCertificationDispatchDirective:
    """Bind certification to the server-observed room dispatch and its directive."""
    if job.agent_name != target.certification_agent_name:
        raise VoiceCertificationTargetError("voice certification job targets the wrong worker")
    if job.room.name != target.room_name:
        raise VoiceCertificationTargetError("voice certification job targets the wrong room")
    participant = job.participant
    try:
        participant_is_set = job.HasField("participant")
    except AttributeError:
        participant_is_set = participant is not None
    # Explicit API dispatch creates a room-scoped job. LiveKit supplies a Job participant
    # only for publisher-scoped jobs, so the controller identity is bound during normal
    # post-connect admission. If a participant is present, it must still match exactly.
    if (
        participant_is_set
        and participant is not None
        and (
            participant.kind != models.ParticipantInfo.STANDARD
            or participant.identity != target.controller_participant_identity
        )
    ):
        raise VoiceCertificationTargetError(
            "voice certification requires its addressed standard participant"
        )
    return _load_job_directive(job, target)


def validate_certification_admission(
    target: VoiceCertificationTarget,
    *,
    merchant_id: str,
    participant_kind: str,
    participant_identity: str | None,
) -> None:
    """Cross-check normal tenant admission against the operator-owned target."""
    if merchant_id != target.merchant_id:
        raise VoiceCertificationTargetError(
            "voice certification admission resolved the wrong merchant"
        )
    if (
        participant_kind != "standard"
        or participant_identity != target.controller_participant_identity
    ):
        raise VoiceCertificationTargetError(
            "voice certification admission did not bind its standard participant"
        )


def load_voice_sample_assignment(
    job: agent.Job,
    target: VoiceCertificationTarget,
    *,
    expected_deployment_id: str,
    expected_journey_corpus_fingerprint: str,
    expected_runtime_contract_fingerprint: str,
    expected_application_contract: VoiceApplicationContract,
    worker_id: str,
    worker_participant_identity: str,
) -> VoiceCertificationSampleAssignment:
    """Join a controller directive to the job authority observed by this worker."""
    try:
        directive = validate_certification_job(job, target)
        if directive.deployment_id != expected_deployment_id:
            raise VoiceCertificationProtocolError(
                "voice certification directive has the wrong deployment"
            )
        if directive.journey_corpus_fingerprint != expected_journey_corpus_fingerprint:
            raise VoiceCertificationProtocolError(
                "voice certification directive has the wrong journey corpus"
            )
        if directive.runtime_contract_fingerprint != expected_runtime_contract_fingerprint:
            raise VoiceCertificationProtocolError(
                "voice certification directive has the wrong runtime contract"
            )
        if directive.application_contract_fingerprint != voice_application_contract_fingerprint(
            expected_application_contract
        ):
            raise VoiceCertificationProtocolError(
                "voice certification directive has the wrong application contract"
            )
        binding = VoiceCertificationJobBinding(
            sample_id=directive.sample.sample_id,
            room_id=job.room.sid,
            assignment_id=job.dispatch_id,
            job_id=job.id,
            worker_id=worker_id,
            worker_participant_identity=worker_participant_identity,
        )
        return VoiceCertificationSampleAssignment.from_directive(directive, binding)
    except VoiceCertificationProtocolError:
        raise
    except (ValidationError, VoiceCertificationTargetError) as exc:
        raise VoiceCertificationProtocolError(
            "voice certification job sample directive is invalid"
        ) from exc


async def answer_certification_request(
    request: JobRequest,
    target: VoiceCertificationTarget,
) -> bool:
    """Accept only the configured synthetic job and terminate every foreign request."""
    try:
        validate_certification_job(request.job, target)
    except VoiceCertificationTargetError:
        await request.reject(terminate=True)
        return False
    await request.accept()
    return True
