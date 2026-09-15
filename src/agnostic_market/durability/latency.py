"""Versioned, transcript-free latency evidence for the durable runtime."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Awaitable, Callable, Sequence
from datetime import datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Literal, NoReturn, Protocol, Self

from psycopg import ProgrammingError
from psycopg.conninfo import conninfo_to_dict
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from agnostic_market.agents.routing_activation import SemanticRoutingRuntimeContract
from agnostic_market.config.loader import ConfigError, load_yaml_layer
from agnostic_market.dtos.events import InterruptEvent, SpokenMessageEvent, TurnEvent
from agnostic_market.dtos.platform import ConfigIdentifier, PlatformRuntimeConfig
from agnostic_market.durability.evidence import ExceptionTypeName, write_immutable_evidence
from agnostic_market.durability.timing import (
    DurabilityOperation,
    DurabilityTimingOutcome,
    DurabilityTimingSample,
    InMemoryDurabilityTimingObserver,
)

_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)
_REPORTED_STATISTICS = ("p50", "p95", "maximum")
_MINIMUM_P95_SAMPLE_COUNT = 20
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_OCI_SHA256_PATTERN = r"^sha256:[0-9a-f]{64}$"


class LatencyTier(StrEnum):
    SIMPLE = "simple"
    NON_CHECKOUT_COMMERCE = "non_checkout_commerce"
    CHECKOUT = "checkout"


class LatencyEnvironment(StrEnum):
    CONTROLLED = "controlled"
    DEPLOYMENT = "deployment"


class LatencyMeasurementSurface(StrEnum):
    REASONING_GRAPH = "reasoning_graph"
    VOICE_PROCESSING = "voice_processing"


class VoiceTransportSurface(StrEnum):
    STANDARD = "standard"
    SIP = "sip"


_P95_LIMIT_SECONDS = {
    LatencyTier.SIMPLE: 1.0,
    LatencyTier.NON_CHECKOUT_COMMERCE: 2.0,
    LatencyTier.CHECKOUT: 3.5,
}


class LatencyPhase(StrEnum):
    STARTUP = "startup"
    TURN = "turn"


class ThermalState(StrEnum):
    COLD = "cold"
    WARM = "warm"


class LatencyObservationOutcome(StrEnum):
    SUCCESS = "success"
    ERROR = "error"


class VoiceAudioTreatment(BaseModel):
    """Frozen synthetic audio asset and playback treatment for one voice journey."""

    model_config = _STRICT

    schema_version: Literal["1"] = "1"
    asset_path: str = Field(min_length=1)
    asset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    container: Literal["wav"]
    pcm_format: Literal["s16le"]
    channel_count: int = Field(ge=1, le=2)
    sample_rate_hz: int = Field(ge=8_000, le=48_000)
    playback_rate: float = Field(gt=0)
    pre_speech_silence_seconds: float = Field(ge=0, le=5)
    post_speech_silence_seconds: float = Field(ge=0, le=5)

    @field_validator("asset_path")
    @classmethod
    def validate_portable_asset_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or "\\" in value or path.as_posix() != value or ".." in path.parts:
            raise ValueError("voice audio asset path must be a normalized relative POSIX path")
        return value

    @field_validator("playback_rate")
    @classmethod
    def validate_supported_playback_rate(cls, value: float) -> float:
        if value != 1.0:
            raise ValueError("voice certification currently requires playback_rate 1.0")
        return value


class VoiceLatencyMetrics(BaseModel):
    """Correlated raw LiveKit metrics plus their processing-only derivation."""

    model_config = _STRICT

    schema_version: Literal["1"] = "1"
    end_to_end_seconds: float = Field(ge=0)
    endpointing_seconds: float = Field(ge=0)
    processing_seconds: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_processing_derivation(self) -> Self:
        expected = self.end_to_end_seconds - self.endpointing_seconds
        if expected < 0 or not math.isclose(
            self.processing_seconds,
            expected,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("voice processing latency must equal end-to-end minus endpointing")
        return self


class LatencyJourney(BaseModel):
    model_config = _STRICT

    journey_id: ConfigIdentifier
    journey_contract_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    tier: LatencyTier
    required_operations: tuple[DurabilityOperation, ...] = Field(min_length=1)
    setup_audio_treatments: tuple[VoiceAudioTreatment, ...] = ()
    audio_treatment: VoiceAudioTreatment | None = None

    @field_validator("required_operations")
    @classmethod
    def unique_operations(
        cls, value: tuple[DurabilityOperation, ...]
    ) -> tuple[DurabilityOperation, ...]:
        if len(set(value)) != len(value):
            raise ValueError("required operations must be unique")
        return value


class LatencyCartLine(BaseModel):
    model_config = _STRICT

    sku: ConfigIdentifier
    quantity: int = Field(ge=1)


class LatencyJourneyContract(BaseModel):
    """One frozen synthetic turn and its required semantic postcondition."""

    model_config = _STRICT

    schema_version: Literal["1"] = "1"
    journey_id: ConfigIdentifier
    setup_turns: tuple[str, ...] = ()
    utterance: str = Field(min_length=1)
    initial_cart: tuple[LatencyCartLine, ...]
    expected_cart: tuple[LatencyCartLine, ...]
    expected_event_kind: Literal["spoken_message", "interrupt"]
    expected_event_node: ConfigIdentifier | None = None
    expected_event_text_contains: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_expected_event(self) -> Self:
        if (self.expected_event_kind == "spoken_message") != (self.expected_event_node is not None):
            raise ValueError("only spoken-message journeys require an expected event node")
        for field_name in ("initial_cart", "expected_cart"):
            lines = getattr(self, field_name)
            if len({line.sku for line in lines}) != len(lines):
                raise ValueError(f"{field_name} cannot contain duplicate SKUs")
        text_values = (
            self.utterance,
            *self.setup_turns,
            *self.expected_event_text_contains,
        )
        if any(not value.strip() for value in text_values):
            raise ValueError("latency journey text cannot be blank")
        if bool(self.initial_cart) != bool(self.setup_turns):
            raise ValueError("latency journey setup turns must establish any initial cart")
        return self


class LatencyJourneyPostconditionError(RuntimeError):
    """A measured journey did not satisfy its frozen semantic contract."""


def require_latency_journey_postconditions(
    contract: LatencyJourneyContract,
    *,
    events: Sequence[TurnEvent],
    actual_cart: tuple[LatencyCartLine, ...],
    placed_orders_before: int,
    placed_orders_after: int,
) -> None:
    """Verify the shared graph and voice journey outcome without reclassifying it."""
    if contract.expected_event_kind == "interrupt":
        expected = [event for event in events if isinstance(event, InterruptEvent)]
        unexpected = [event for event in events if not isinstance(event, InterruptEvent)]
        text = expected[0].prompt if len(expected) == 1 else ""
    else:
        expected = [event for event in events if isinstance(event, SpokenMessageEvent)]
        unexpected = [event for event in events if not isinstance(event, SpokenMessageEvent)]
        text = expected[0].text if len(expected) == 1 else ""
        if len(expected) == 1 and expected[0].node != contract.expected_event_node:
            raise LatencyJourneyPostconditionError(
                "journey produced spoken message from "
                f"{expected[0].node!r}; expected {contract.expected_event_node!r}"
            )
    if len(expected) != 1 or unexpected:
        raise LatencyJourneyPostconditionError("journey produced the wrong caller-event shape")
    normalized = text.casefold()
    if any(
        fragment.casefold() not in normalized for fragment in contract.expected_event_text_contains
    ):
        raise LatencyJourneyPostconditionError("journey output did not satisfy its frozen contract")
    if actual_cart != contract.expected_cart:
        raise LatencyJourneyPostconditionError("journey cart state is invalid after measured turn")
    if placed_orders_after != placed_orders_before:
        raise LatencyJourneyPostconditionError("latency journey committed an order")


class LatencyJourneyCorpus(BaseModel):
    model_config = _STRICT

    schema_version: Literal["1"]
    merchant_id: ConfigIdentifier
    journeys: tuple[LatencyJourneyContract, ...] = Field(min_length=3)

    @model_validator(mode="after")
    def validate_unique_journeys(self) -> Self:
        if len({journey.journey_id for journey in self.journeys}) != len(self.journeys):
            raise ValueError("latency journey ids must be unique")
        return self


class VoiceApplicationContract(BaseModel):
    """Canonical release identity measured by deployment voice certification."""

    model_config = _STRICT

    schema_version: Literal[1] = 1
    durable_platform_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    build_artifact_digest: str = Field(pattern=_OCI_SHA256_PATTERN)
    tenant_config_version: str = Field(pattern=_SHA256_PATTERN)
    semantic_routing: SemanticRoutingRuntimeContract
    certification_target_fingerprint: str = Field(pattern=_SHA256_PATTERN)


def voice_application_contract_fingerprint(contract: VoiceApplicationContract) -> str:
    canonical = json.dumps(
        contract.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


class DurableLatencyMethodology(BaseModel):
    """Frozen inputs selected before a deployment-shaped certification run."""

    model_config = _STRICT

    schema_version: Literal["3", "4", "5"]
    environment: LatencyEnvironment
    backend_location: ConfigIdentifier
    journey_corpus_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_contract_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    application_contract: VoiceApplicationContract | None = None
    measurement_surface: LatencyMeasurementSurface
    transport_surface: VoiceTransportSurface | None = None
    result_rpc_timeout_seconds: float | None = Field(default=None, gt=0)
    result_rpc_retries: int | None = Field(default=None, ge=0, le=3)
    result_rpc_retry_backoff_seconds: float | None = Field(default=None, gt=0)
    job_ready_timeout_seconds: float | None = Field(default=None, gt=0)
    dispatch_cleanup_timeout_seconds: float | None = Field(default=None, gt=0)
    startup_treatment: Literal["fresh_job_resources"]
    concurrency: int = Field(ge=1)
    warmup_runs: int = Field(ge=1)
    startup_samples: int = Field(ge=_MINIMUM_P95_SAMPLE_COUNT)
    startup_p95_limit_seconds: float = Field(gt=0)
    required_startup_operations: tuple[DurabilityOperation, ...] = Field(min_length=1)
    samples_per_journey: int = Field(ge=_MINIMUM_P95_SAMPLE_COUNT)
    sample_timeout_seconds: float = Field(gt=0)
    reported_statistics: tuple[Literal["p50", "p95", "maximum"], ...]
    journeys: tuple[LatencyJourney, ...] = Field(min_length=3)

    @model_validator(mode="after")
    def validate_closed_methodology(self) -> Self:
        if len(set(self.required_startup_operations)) != len(self.required_startup_operations):
            raise ValueError("required startup operations must be unique")
        if self.reported_statistics != _REPORTED_STATISTICS:
            raise ValueError("latency methodology must report p50, p95, and maximum")
        journey_ids = tuple(journey.journey_id for journey in self.journeys)
        if len(set(journey_ids)) != len(journey_ids):
            raise ValueError("latency journey ids must be unique")
        contract_fingerprints = tuple(
            journey.journey_contract_fingerprint for journey in self.journeys
        )
        if len(set(contract_fingerprints)) != len(contract_fingerprints):
            raise ValueError("latency journey contracts must be unique")
        covered_tiers = {journey.tier for journey in self.journeys}
        if covered_tiers != set(LatencyTier):
            raise ValueError("latency methodology must cover every turn tier")
        audio_treatments = tuple(journey.audio_treatment for journey in self.journeys)
        setup_audio_treatments = tuple(
            treatment for journey in self.journeys for treatment in journey.setup_audio_treatments
        )
        if self.schema_version == "3":
            if (
                self.measurement_surface is not LatencyMeasurementSurface.REASONING_GRAPH
                or self.transport_surface is not None
                or self.result_rpc_timeout_seconds is not None
                or self.result_rpc_retries is not None
                or self.result_rpc_retry_backoff_seconds is not None
                or self.job_ready_timeout_seconds is not None
                or self.dispatch_cleanup_timeout_seconds is not None
                or any(treatment is not None for treatment in audio_treatments)
                or setup_audio_treatments
                or self.application_contract is not None
            ):
                raise ValueError("schema-3 latency methodology covers only the reasoning graph")
        elif self.schema_version in {"4", "5"} and (
            self.measurement_surface is not LatencyMeasurementSurface.VOICE_PROCESSING
            or self.transport_surface is None
            or self.result_rpc_timeout_seconds is None
            or self.result_rpc_retries is None
        ):
            raise ValueError("voice methodology requires a voice-processing transport surface")
        elif any(treatment is None for treatment in audio_treatments):
            raise ValueError("voice methodology requires every audio treatment")
        elif len(
            {
                treatment.asset_sha256
                for treatment in (*setup_audio_treatments, *audio_treatments)
                if treatment is not None
            }
        ) != len(setup_audio_treatments) + len(audio_treatments):
            raise ValueError("voice methodology audio assets must be unique by digest")
        if self.schema_version == "4" and (
            self.application_contract is not None
            or self.result_rpc_retry_backoff_seconds is not None
            or self.job_ready_timeout_seconds is not None
            or self.dispatch_cleanup_timeout_seconds is not None
        ):
            raise ValueError("schema-4 voice methodology cannot claim schema-5 contracts")
        if self.schema_version == "5":
            if (
                self.application_contract is None
                or self.result_rpc_retry_backoff_seconds is None
                or self.job_ready_timeout_seconds is None
                or self.dispatch_cleanup_timeout_seconds is None
            ):
                raise ValueError(
                    "schema-5 voice methodology requires application, readiness, cleanup, "
                    "and RPC retry contracts"
                )
            if (
                self.application_contract.durable_platform_fingerprint
                != self.runtime_contract_fingerprint
            ):
                raise ValueError("voice application and durable runtime fingerprints must match")
            required_startup = set(self.required_startup_operations)
            if (
                not {
                    DurabilityOperation.REGISTRY_CHECKPOINT_GENERATIONS,
                    DurabilityOperation.CHECKPOINT_READ,
                }
                <= required_startup
                or DurabilityOperation.REGISTRY_GET in required_startup
            ):
                raise ValueError(
                    "schema-5 startup must require generation lookup and checkpoint read, "
                    "not registry get"
                )
        return self


class DurabilityComponentObservation(BaseModel):
    model_config = _STRICT

    operation: DurabilityOperation
    elapsed_seconds: float = Field(ge=0)
    outcome: DurabilityTimingOutcome


class LatencyObservation(BaseModel):
    """One redacted startup or turn measurement and its durable attribution."""

    model_config = _STRICT

    schema_version: Literal["1", "2", "3"] = "1"
    sample_id: ConfigIdentifier
    phase: LatencyPhase
    thermal_state: ThermalState
    elapsed_seconds: float | None = Field(default=None, ge=0)
    outcome: LatencyObservationOutcome = LatencyObservationOutcome.SUCCESS
    error_type: ExceptionTypeName | None = None
    journey_id: ConfigIdentifier | None = None
    tier: LatencyTier | None = None
    voice_metrics: VoiceLatencyMetrics | None = None
    components: tuple[DurabilityComponentObservation, ...]

    @model_validator(mode="after")
    def validate_phase_shape(self) -> Self:
        if self.outcome is LatencyObservationOutcome.SUCCESS:
            if self.elapsed_seconds is None or self.error_type is not None or not self.components:
                raise ValueError(
                    "successful latency observations require elapsed and component evidence"
                )
        elif self.error_type is None:
            raise ValueError("failed latency observations require a redacted error type")
        if self.phase is LatencyPhase.STARTUP:
            if self.thermal_state is not ThermalState.COLD:
                raise ValueError("startup latency observations must be cold")
            if self.journey_id is not None or self.tier is not None:
                raise ValueError("startup latency observations cannot name a turn journey")
        elif (
            self.thermal_state is not ThermalState.WARM
            or self.journey_id is None
            or self.tier is None
        ):
            raise ValueError("turn latency observations require one warm typed journey")
        if self.schema_version == "1" and self.voice_metrics is not None:
            raise ValueError("schema-1 observations cannot contain voice metrics")
        if self.schema_version in {"2", "3"}:
            if self.phase is LatencyPhase.STARTUP and self.voice_metrics is not None:
                raise ValueError("startup observations cannot contain voice turn metrics")
            if (
                self.phase is LatencyPhase.TURN
                and self.outcome is LatencyObservationOutcome.SUCCESS
                and self.voice_metrics is None
            ):
                raise ValueError("successful voice turns require correlated voice metrics")
        if self.voice_metrics is not None and (
            self.elapsed_seconds is None
            or not math.isclose(
                self.elapsed_seconds,
                self.voice_metrics.processing_seconds,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("voice processing metric must equal observed latency")
        return self

    @classmethod
    def from_timing(
        cls,
        *,
        schema_version: Literal["1", "2", "3"] = "1",
        sample_id: str,
        phase: LatencyPhase,
        thermal_state: ThermalState,
        elapsed_seconds: float,
        component_samples: Sequence[DurabilityTimingSample],
        journey_id: str | None = None,
        tier: LatencyTier | None = None,
        voice_metrics: VoiceLatencyMetrics | None = None,
    ) -> LatencyObservation:
        return cls(
            schema_version=schema_version,
            sample_id=sample_id,
            phase=phase,
            thermal_state=thermal_state,
            elapsed_seconds=elapsed_seconds,
            outcome=LatencyObservationOutcome.SUCCESS,
            journey_id=journey_id,
            tier=tier,
            voice_metrics=voice_metrics,
            components=tuple(
                DurabilityComponentObservation(
                    operation=sample.operation,
                    elapsed_seconds=sample.elapsed_seconds,
                    outcome=sample.outcome,
                )
                for sample in component_samples
            ),
        )


class LatencyStatistics(BaseModel):
    model_config = _STRICT

    count: int = Field(ge=0)
    p50_seconds: float | None = Field(default=None, ge=0)
    p95_seconds: float | None = Field(default=None, ge=0)
    maximum_seconds: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_complete_statistics(self) -> Self:
        values = (self.p50_seconds, self.p95_seconds, self.maximum_seconds)
        if self.count == 0:
            if any(value is not None for value in values):
                raise ValueError("empty latency statistics cannot contain percentile values")
        elif any(value is None for value in values):
            raise ValueError("non-empty latency statistics require every percentile value")
        return self


class ComponentLatencyResult(BaseModel):
    model_config = _STRICT

    operation: DurabilityOperation
    statistics: LatencyStatistics


class StartupLatencyResult(BaseModel):
    model_config = _STRICT

    p95_limit_seconds: float = Field(gt=0)
    statistics: LatencyStatistics
    components: tuple[ComponentLatencyResult, ...]


class JourneyLatencyResult(BaseModel):
    model_config = _STRICT

    journey_id: ConfigIdentifier
    tier: LatencyTier
    p95_limit_seconds: float = Field(gt=0)
    statistics: LatencyStatistics
    components: tuple[ComponentLatencyResult, ...]


class LatencyGate(BaseModel):
    model_config = _STRICT

    passed: bool
    failures: tuple[str, ...]


def latency_evidence_schema_version(
    methodology: DurableLatencyMethodology,
) -> Literal["1", "2", "3"]:
    return latency_evidence_schema_version_for_methodology(methodology.schema_version)


def latency_evidence_schema_version_for_methodology(
    methodology_schema_version: Literal["3", "4", "5"],
) -> Literal["1", "2", "3"]:
    """Map a methodology schema to the only evidence envelope it can produce."""
    if methodology_schema_version == "3":
        return "1"
    if methodology_schema_version == "4":
        return "2"
    return "3"


class DurableLatencyReport(BaseModel):
    """Complete evidence artifact for one frozen deployment-shaped run."""

    model_config = _STRICT

    schema_version: Literal["1", "2", "3"] = "1"
    run_at: datetime
    deployment_id: ConfigIdentifier
    methodology_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    methodology: DurableLatencyMethodology
    observations: tuple[LatencyObservation, ...]
    startup: StartupLatencyResult
    journeys: tuple[JourneyLatencyResult, ...]
    gate: LatencyGate

    @model_validator(mode="after")
    def validate_report_binding(self) -> Self:
        if self.run_at.tzinfo is None:
            raise ValueError("latency report timestamp must be timezone-aware")
        if self.methodology_fingerprint != methodology_fingerprint(self.methodology):
            raise ValueError("latency report methodology fingerprint is invalid")
        if self.schema_version != latency_evidence_schema_version(self.methodology):
            raise ValueError("latency report schema does not match its methodology")
        if self.gate.passed != (not self.gate.failures):
            raise ValueError("latency report gate result is inconsistent")
        return self


class LatencyCertificationOutcome(StrEnum):
    COMPLETED = "completed"
    ABORTED = "aborted"


class LatencyAbortStage(StrEnum):
    WARMUP_PREPARE = "warmup_prepare"
    WARMUP_RUN = "warmup_run"
    WARMUP_CLEANUP = "warmup_cleanup"
    VOICE_PREPARE = "voice_prepare"
    VOICE_RUN = "voice_run"
    VOICE_CLEANUP = "voice_cleanup"
    VOICE_CONTROLLER_CLEANUP = "voice_controller_cleanup"


class LatencyCertificationAbort(BaseModel):
    """Redacted evidence for a run that could not complete its frozen execution."""

    model_config = _STRICT

    stage: LatencyAbortStage
    sample_id: ConfigIdentifier | None = None
    journey_id: ConfigIdentifier | None = None
    error_type: ExceptionTypeName
    cleanup_error_type: ExceptionTypeName | None = None

    @model_validator(mode="after")
    def validate_stage_binding(self) -> Self:
        warmup_stages = {
            LatencyAbortStage.WARMUP_PREPARE,
            LatencyAbortStage.WARMUP_RUN,
            LatencyAbortStage.WARMUP_CLEANUP,
        }
        sample_stages = warmup_stages | {
            LatencyAbortStage.VOICE_PREPARE,
            LatencyAbortStage.VOICE_RUN,
            LatencyAbortStage.VOICE_CLEANUP,
        }
        if self.stage in warmup_stages and self.journey_id is None:
            raise ValueError("latency warmup abort requires a journey")
        if self.stage in sample_stages and self.sample_id is None:
            raise ValueError("sample-scoped latency abort requires a sample")
        if self.stage is LatencyAbortStage.VOICE_CONTROLLER_CLEANUP and (
            self.sample_id is not None or self.journey_id is not None
        ):
            raise ValueError("voice controller cleanup abort cannot claim a sample")
        return self


class DurableLatencyCertificationRun(BaseModel):
    """Versioned outcome envelope for completed and aborted certification runs."""

    model_config = _STRICT

    schema_version: Literal["1", "2", "3"] = "1"
    run_at: datetime
    deployment_id: ConfigIdentifier
    methodology_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    methodology: DurableLatencyMethodology
    outcome: LatencyCertificationOutcome
    report: DurableLatencyReport | None = None
    abort: LatencyCertificationAbort | None = None

    @model_validator(mode="after")
    def validate_outcome_binding(self) -> Self:
        if self.run_at.tzinfo is None:
            raise ValueError("latency certification timestamp must be timezone-aware")
        if self.methodology_fingerprint != methodology_fingerprint(self.methodology):
            raise ValueError("latency certification methodology fingerprint is invalid")
        if self.schema_version != latency_evidence_schema_version(self.methodology):
            raise ValueError("latency certification schema does not match its methodology")
        if self.outcome is LatencyCertificationOutcome.COMPLETED:
            if self.report is None or self.abort is not None:
                raise ValueError("completed latency certification requires only a report")
            if (
                self.report.run_at != self.run_at
                or self.report.deployment_id != self.deployment_id
                or self.report.methodology != self.methodology
                or self.report.methodology_fingerprint != self.methodology_fingerprint
            ):
                raise ValueError("completed latency report does not match its run envelope")
        elif self.report is not None or self.abort is None:
            raise ValueError("aborted latency certification requires only abort evidence")
        return self


class DurableLatencyActivationError(RuntimeError):
    """The supplied latency evidence cannot authorize durable activation."""


def require_voice_application_contract(
    methodology: DurableLatencyMethodology,
    expected: VoiceApplicationContract,
) -> None:
    """Require schema-5 evidence to name the exact observed application identity."""
    if methodology.schema_version != "5" or methodology.application_contract != expected:
        raise DurableLatencyActivationError(
            "voice latency evidence does not match the application contract"
        )


class InvalidLatencyMeasurementError(ValueError):
    pass


class MissingComponentEvidenceError(RuntimeError):
    pass


class ClosableLatencyExecution(Protocol):
    async def aclose(self) -> None: ...


class StartedLatencyExecution(ClosableLatencyExecution, Protocol):
    @property
    def startup_elapsed_seconds(self) -> float: ...


class PreparedTurnLatencyExecution(ClosableLatencyExecution, Protocol):
    async def run(self) -> float: ...


type StartupLatencyProbe = Callable[
    [str, InMemoryDurabilityTimingObserver],
    Awaitable[StartedLatencyExecution],
]
type TurnLatencyProbe = Callable[
    [LatencyJourney, str, InMemoryDurabilityTimingObserver],
    Awaitable[PreparedTurnLatencyExecution],
]


def methodology_fingerprint(methodology: DurableLatencyMethodology) -> str:
    canonical = json.dumps(
        methodology.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def latency_journey_contract_fingerprint(
    contract: LatencyJourneyContract,
    *,
    merchant_id: str,
) -> str:
    canonical = json.dumps(
        {
            "merchant_id": merchant_id,
            "contract": contract.model_dump(mode="json"),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def latency_journey_corpus_fingerprint(corpus: LatencyJourneyCorpus) -> str:
    canonical = json.dumps(
        corpus.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def deployment_runtime_contract_fingerprint(
    platform: PlatformRuntimeConfig,
    *,
    application_dsn: str,
) -> str:
    """Hash the non-secret runtime identity that deployment latency must measure."""
    try:
        connection_parameters = conninfo_to_dict(application_dsn)
    except ProgrammingError as exc:
        raise DurableLatencyActivationError(
            "deployment latency runtime has invalid database connection information"
        ) from exc
    redacted_connection = {
        key: value
        for key, value in connection_parameters.items()
        if key not in {"password", "passfile", "sslpassword"}
    }
    canonical = json.dumps(
        {
            "schema_version": "1",
            "platform": platform.model_dump(mode="json"),
            "database_connection": redacted_connection,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def load_latency_journey_corpus(path: Path) -> LatencyJourneyCorpus:
    """Load the frozen synthetic behavior contracts used by the deployment probe."""
    try:
        payload = json.dumps(load_yaml_layer(path))
        return LatencyJourneyCorpus.model_validate_json(payload)
    except (ConfigError, TypeError, ValueError) as exc:
        raise DurableLatencyActivationError("durable latency journey corpus is invalid") from exc


def bind_latency_journey_contracts(
    methodology: DurableLatencyMethodology,
    corpus: LatencyJourneyCorpus,
) -> dict[str, LatencyJourneyContract]:
    """Require exact journey identity and content binding before a run begins."""
    if methodology.journey_corpus_fingerprint != latency_journey_corpus_fingerprint(corpus):
        raise DurableLatencyActivationError("latency journey corpus fingerprint mismatch")
    contracts = {journey.journey_id: journey for journey in corpus.journeys}
    configured = {journey.journey_id: journey for journey in methodology.journeys}
    if contracts.keys() != configured.keys():
        raise DurableLatencyActivationError(
            "latency methodology and journey corpus do not cover the same journeys"
        )
    for journey_id, journey in configured.items():
        expected = latency_journey_contract_fingerprint(
            contracts[journey_id],
            merchant_id=corpus.merchant_id,
        )
        if journey.journey_contract_fingerprint != expected:
            raise DurableLatencyActivationError(
                f"latency journey contract fingerprint mismatch: {journey_id}"
            )
        if methodology.schema_version in {"4", "5"} and len(journey.setup_audio_treatments) != len(
            contracts[journey_id].setup_turns
        ):
            raise DurableLatencyActivationError(
                f"latency journey setup audio count mismatch: {journey_id}"
            )
    return contracts


def load_latency_methodology(path: Path) -> DurableLatencyMethodology:
    """Load the deployment-owned, pre-registered latency contract."""
    try:
        payload = json.dumps(load_yaml_layer(path))
        return DurableLatencyMethodology.model_validate_json(payload)
    except (ConfigError, TypeError, ValueError) as exc:
        raise DurableLatencyActivationError(
            "durable latency methodology is missing or invalid"
        ) from exc


def load_latency_certification_run(path: Path) -> DurableLatencyCertificationRun:
    """Load one strict certification outcome without accepting partial evidence."""
    try:
        payload = path.read_text(encoding="utf-8")
        return DurableLatencyCertificationRun.model_validate_json(payload)
    except (OSError, ValidationError) as exc:
        raise DurableLatencyActivationError("durable latency report is missing or invalid") from exc


def write_latency_certification_run(
    path: Path,
    run: DurableLatencyCertificationRun,
) -> None:
    """Publish one immutable evidence artifact atomically without replacement."""
    write_immutable_evidence(path, run)


def require_deployment_latency_report(
    report: DurableLatencyReport,
    *,
    expected_methodology_fingerprint: str,
    expected_deployment_id: str,
) -> None:
    """Authorize only an exact passing report from the deployment-shaped environment."""
    if report.methodology.environment is not LatencyEnvironment.DEPLOYMENT:
        raise DurableLatencyActivationError(
            "durable activation requires deployment-shaped latency evidence"
        )
    if (
        report.methodology_fingerprint != expected_methodology_fingerprint
        or report.deployment_id != expected_deployment_id
    ):
        raise DurableLatencyActivationError("latency evidence runtime contract mismatch")
    try:
        derived = build_latency_report(
            report.methodology,
            report.observations,
            run_at=report.run_at,
            deployment_id=report.deployment_id,
        )
    except (TypeError, ValueError) as exc:
        raise DurableLatencyActivationError("latency evidence derived results are invalid") from exc
    if (
        report.startup != derived.startup
        or report.journeys != derived.journeys
        or report.gate != derived.gate
    ):
        raise DurableLatencyActivationError("latency evidence derived results are inconsistent")
    if not report.gate.passed:
        raise DurableLatencyActivationError("deployment latency gate did not pass")


def require_deployment_latency_evidence(
    methodology_path: Path,
    report_path: Path,
    *,
    expected_deployment_id: str,
    expected_journey_corpus: LatencyJourneyCorpus,
    expected_runtime_contract_fingerprint: str,
    expected_application_contract: VoiceApplicationContract | None = None,
    required_measurement_surface: LatencyMeasurementSurface,
    required_transport_surface: VoiceTransportSurface | None = None,
) -> DurableLatencyReport:
    """Require one completed run matching the pre-registered deployment contract."""
    methodology = load_latency_methodology(methodology_path)
    if methodology.environment is not LatencyEnvironment.DEPLOYMENT:
        raise DurableLatencyActivationError(
            "durable activation requires a deployment-shaped latency methodology"
        )
    bind_latency_journey_contracts(methodology, expected_journey_corpus)
    if methodology.runtime_contract_fingerprint != expected_runtime_contract_fingerprint:
        raise DurableLatencyActivationError(
            "durable latency evidence does not match the deployment runtime"
        )
    if methodology.measurement_surface is not required_measurement_surface:
        raise DurableLatencyActivationError(
            "durable latency evidence does not cover the required measurement surface"
        )
    if required_measurement_surface is LatencyMeasurementSurface.VOICE_PROCESSING:
        if expected_application_contract is None:
            raise DurableLatencyActivationError(
                "voice activation requires an expected application contract"
            )
        require_voice_application_contract(methodology, expected_application_contract)
        if (
            required_transport_surface is None
            or methodology.transport_surface is not required_transport_surface
        ):
            raise DurableLatencyActivationError(
                "durable latency evidence does not cover the required transport surface"
            )
    elif required_transport_surface is not None:
        raise DurableLatencyActivationError(
            "reasoning-graph latency evidence cannot require a transport surface"
        )
    run = load_latency_certification_run(report_path)
    if (
        run.outcome is not LatencyCertificationOutcome.COMPLETED
        or run.report is None
        or run.methodology != methodology
    ):
        raise DurableLatencyActivationError(
            "durable latency evidence does not match the deployment methodology"
        )
    require_deployment_latency_report(
        run.report,
        expected_methodology_fingerprint=methodology_fingerprint(methodology),
        expected_deployment_id=expected_deployment_id,
    )
    return run.report


def _nearest_rank(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    index = max(math.ceil(len(ordered) * quantile) - 1, 0)
    return ordered[index]


def _statistics(values: Sequence[float]) -> LatencyStatistics:
    if not values:
        return LatencyStatistics(count=0)
    return LatencyStatistics(
        count=len(values),
        p50_seconds=_nearest_rank(values, 0.50),
        p95_seconds=_nearest_rank(values, 0.95),
        maximum_seconds=max(values),
    )


def _validated_elapsed_seconds(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise InvalidLatencyMeasurementError
    elapsed_seconds = float(value)
    if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
        raise InvalidLatencyMeasurementError
    return elapsed_seconds


async def _close_preserving_failure(
    execution: ClosableLatencyExecution,
    failure: BaseException,
    *,
    timeout_seconds: float,
) -> NoReturn:
    try:
        await _await_with_timeout(execution.aclose(), timeout_seconds)
    except asyncio.CancelledError as cleanup_cancellation:
        if isinstance(failure, asyncio.CancelledError):
            raise failure from cleanup_cancellation
        raise cleanup_cancellation from failure
    except BaseException as cleanup_failure:
        raise failure from cleanup_failure
    raise failure


async def _await_with_timeout[T](
    awaitable: Awaitable[T],
    timeout_seconds: float,
) -> T:
    async with asyncio.timeout(timeout_seconds):
        return await awaitable


def _component_results(
    observations: Sequence[LatencyObservation],
) -> tuple[ComponentLatencyResult, ...]:
    elapsed_by_operation: dict[DurabilityOperation, list[float]] = defaultdict(list)
    for observation in observations:
        for component in observation.components:
            elapsed_by_operation[component.operation].append(component.elapsed_seconds)
    return tuple(
        ComponentLatencyResult(
            operation=operation,
            statistics=_statistics(elapsed_by_operation[operation]),
        )
        for operation in sorted(elapsed_by_operation, key=lambda item: item.value)
    )


def _component_observations(
    samples: Sequence[DurabilityTimingSample],
) -> tuple[DurabilityComponentObservation, ...]:
    return tuple(
        DurabilityComponentObservation(
            operation=sample.operation,
            elapsed_seconds=sample.elapsed_seconds,
            outcome=sample.outcome,
        )
        for sample in samples
    )


def build_latency_report(
    methodology: DurableLatencyMethodology,
    observations: Sequence[LatencyObservation],
    *,
    run_at: datetime,
    deployment_id: str,
) -> DurableLatencyReport:
    """Validate exact evidence coverage, retain misses, and evaluate the frozen gate."""
    evidence = tuple(observations)
    expected_schema = latency_evidence_schema_version(methodology)
    if any(observation.schema_version != expected_schema for observation in evidence):
        raise ValueError("latency observation schema does not match its methodology")
    sample_ids = tuple(observation.sample_id for observation in evidence)
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("latency sample ids must be unique")
    startup_observations = tuple(
        observation for observation in evidence if observation.phase is LatencyPhase.STARTUP
    )
    if len(startup_observations) != methodology.startup_samples:
        raise ValueError(
            f"latency evidence requires exactly {methodology.startup_samples} startup samples"
        )
    startup_statistics = _statistics(
        tuple(
            observation.elapsed_seconds
            for observation in startup_observations
            if observation.outcome is LatencyObservationOutcome.SUCCESS
            and observation.elapsed_seconds is not None
        )
    )
    failures: list[str] = []
    for observation in evidence:
        if observation.outcome is LatencyObservationOutcome.ERROR:
            failures.append(f"sample {observation.sample_id} failed with {observation.error_type}")
        for component in observation.components:
            if component.outcome is not DurabilityTimingOutcome.SUCCESS:
                failures.append(
                    f"sample {observation.sample_id} contains non-success durable component "
                    f"{component.operation.value}"
                )
    if (
        startup_statistics.p95_seconds is not None
        and startup_statistics.p95_seconds > methodology.startup_p95_limit_seconds
    ):
        failures.append(
            "startup p95 "
            f"{startup_statistics.p95_seconds:.6f}s exceeds "
            f"{methodology.startup_p95_limit_seconds:.6f}s"
        )

    journey_by_id = {journey.journey_id: journey for journey in methodology.journeys}
    unknown_journeys = sorted(
        {
            observation.journey_id
            for observation in evidence
            if observation.phase is LatencyPhase.TURN
            and observation.journey_id is not None
            and observation.journey_id not in journey_by_id
        }
    )
    if unknown_journeys:
        raise ValueError(
            "latency evidence contains unknown journeys: " + ", ".join(unknown_journeys)
        )

    for observation in evidence:
        if observation.phase is LatencyPhase.STARTUP:
            required = methodology.required_startup_operations
        else:
            assert observation.journey_id is not None
            required = journey_by_id[observation.journey_id].required_operations
        observed = {component.operation for component in observation.components}
        missing = sorted(set(required) - observed)
        if missing:
            failures.append(
                f"sample {observation.sample_id} missing required operations: " + ", ".join(missing)
            )

    journey_results: list[JourneyLatencyResult] = []
    for journey in methodology.journeys:
        journey_observations = tuple(
            observation
            for observation in evidence
            if observation.phase is LatencyPhase.TURN
            and observation.journey_id == journey.journey_id
        )
        if len(journey_observations) != methodology.samples_per_journey:
            raise ValueError(
                f"latency evidence requires exactly {methodology.samples_per_journey} samples "
                f"for journey {journey.journey_id}"
            )
        if any(observation.tier is not journey.tier for observation in journey_observations):
            raise ValueError(f"latency evidence tier mismatches journey {journey.journey_id}")
        statistics = _statistics(
            tuple(
                observation.elapsed_seconds
                for observation in journey_observations
                if observation.outcome is LatencyObservationOutcome.SUCCESS
                and observation.elapsed_seconds is not None
            )
        )
        limit = _P95_LIMIT_SECONDS[journey.tier]
        if statistics.p95_seconds is not None and statistics.p95_seconds > limit:
            failures.append(
                f"journey {journey.journey_id} p95 {statistics.p95_seconds:.6f}s "
                f"exceeds {limit:.6f}s"
            )
        journey_results.append(
            JourneyLatencyResult(
                journey_id=journey.journey_id,
                tier=journey.tier,
                p95_limit_seconds=limit,
                statistics=statistics,
                components=_component_results(journey_observations),
            )
        )

    expected_observation_count = methodology.startup_samples + (
        methodology.samples_per_journey * len(methodology.journeys)
    )
    if len(evidence) != expected_observation_count:
        raise ValueError("latency evidence contains observations outside the frozen methodology")

    return DurableLatencyReport(
        schema_version=expected_schema,
        run_at=run_at,
        deployment_id=deployment_id,
        methodology_fingerprint=methodology_fingerprint(methodology),
        methodology=methodology,
        observations=evidence,
        startup=StartupLatencyResult(
            p95_limit_seconds=methodology.startup_p95_limit_seconds,
            statistics=startup_statistics,
            components=_component_results(startup_observations),
        ),
        journeys=tuple(journey_results),
        gate=LatencyGate(passed=not failures, failures=tuple(failures)),
    )


async def run_latency_certification(
    methodology: DurableLatencyMethodology,
    *,
    deployment_id: str,
    startup_probe: StartupLatencyProbe,
    turn_probe: TurnLatencyProbe,
    run_at: datetime,
) -> DurableLatencyCertificationRun:
    """Execute the frozen warmup, sample count, and concurrency contract."""
    if (
        methodology.schema_version != "3"
        or methodology.measurement_surface is not LatencyMeasurementSurface.REASONING_GRAPH
    ):
        raise DurableLatencyActivationError(
            "the in-process latency runner supports only schema-3 reasoning-graph methodology"
        )
    frozen_fingerprint = methodology_fingerprint(methodology)

    def aborted(
        *,
        stage: LatencyAbortStage,
        sample_id: str,
        journey_id: str,
        failure: Exception,
        cleanup_failure: Exception | None = None,
    ) -> DurableLatencyCertificationRun:
        return DurableLatencyCertificationRun(
            schema_version=latency_evidence_schema_version(methodology),
            run_at=run_at,
            deployment_id=deployment_id,
            methodology_fingerprint=frozen_fingerprint,
            methodology=methodology,
            outcome=LatencyCertificationOutcome.ABORTED,
            abort=LatencyCertificationAbort(
                stage=stage,
                sample_id=sample_id,
                journey_id=journey_id,
                error_type=type(failure).__name__,
                cleanup_error_type=(
                    type(cleanup_failure).__name__ if cleanup_failure is not None else None
                ),
            ),
        )

    for journey in methodology.journeys:
        for index in range(methodology.warmup_runs):
            sample_id = f"warmup-turn-{journey.journey_id}-{index + 1:04d}"
            observer = InMemoryDurabilityTimingObserver()
            try:
                execution = await _await_with_timeout(
                    turn_probe(journey, sample_id, observer),
                    methodology.sample_timeout_seconds,
                )
            except asyncio.CancelledError:
                raise
            except Exception as failure:
                return aborted(
                    stage=LatencyAbortStage.WARMUP_PREPARE,
                    sample_id=sample_id,
                    journey_id=journey.journey_id,
                    failure=failure,
                )
            try:
                _validated_elapsed_seconds(
                    await _await_with_timeout(
                        execution.run(),
                        methodology.sample_timeout_seconds,
                    )
                )
            except asyncio.CancelledError as cancellation:
                await _close_preserving_failure(
                    execution,
                    cancellation,
                    timeout_seconds=methodology.sample_timeout_seconds,
                )
            except Exception as failure:
                try:
                    await _await_with_timeout(
                        execution.aclose(),
                        methodology.sample_timeout_seconds,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as cleanup_failure:
                    return aborted(
                        stage=LatencyAbortStage.WARMUP_RUN,
                        sample_id=sample_id,
                        journey_id=journey.journey_id,
                        failure=failure,
                        cleanup_failure=cleanup_failure,
                    )
                return aborted(
                    stage=LatencyAbortStage.WARMUP_RUN,
                    sample_id=sample_id,
                    journey_id=journey.journey_id,
                    failure=failure,
                )
            else:
                try:
                    await _await_with_timeout(
                        execution.aclose(),
                        methodology.sample_timeout_seconds,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as failure:
                    return aborted(
                        stage=LatencyAbortStage.WARMUP_CLEANUP,
                        sample_id=sample_id,
                        journey_id=journey.journey_id,
                        failure=failure,
                    )

    semaphore = asyncio.Semaphore(methodology.concurrency)

    async def observe_startup(index: int) -> LatencyObservation:
        sample_id = f"sample-startup-{index + 1:04d}"
        observer = InMemoryDurabilityTimingObserver()
        async with semaphore:
            execution: StartedLatencyExecution | None = None
            probe_error: Exception | None = None
            elapsed_seconds: float | None = None
            try:
                execution = await _await_with_timeout(
                    startup_probe(sample_id, observer),
                    methodology.sample_timeout_seconds,
                )
                elapsed_seconds = _validated_elapsed_seconds(execution.startup_elapsed_seconds)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                probe_error = exc
            components = _component_observations(observer.samples)
            if execution is not None:
                try:
                    await _await_with_timeout(
                        execution.aclose(),
                        methodology.sample_timeout_seconds,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    if probe_error is None:
                        probe_error = exc
            if probe_error is None and not components:
                probe_error = MissingComponentEvidenceError()
            return LatencyObservation(
                sample_id=sample_id,
                phase=LatencyPhase.STARTUP,
                thermal_state=ThermalState.COLD,
                elapsed_seconds=elapsed_seconds,
                outcome=(
                    LatencyObservationOutcome.ERROR
                    if probe_error is not None
                    else LatencyObservationOutcome.SUCCESS
                ),
                error_type=type(probe_error).__name__ if probe_error is not None else None,
                components=components,
            )

    async def observe_turn(journey: LatencyJourney, index: int) -> LatencyObservation:
        sample_id = f"sample-turn-{journey.journey_id}-{index + 1:04d}"
        observer = InMemoryDurabilityTimingObserver()
        async with semaphore:
            try:
                execution = await _await_with_timeout(
                    turn_probe(journey, sample_id, observer),
                    methodology.sample_timeout_seconds,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                return LatencyObservation(
                    sample_id=sample_id,
                    phase=LatencyPhase.TURN,
                    thermal_state=ThermalState.WARM,
                    elapsed_seconds=None,
                    outcome=LatencyObservationOutcome.ERROR,
                    error_type=type(exc).__name__,
                    journey_id=journey.journey_id,
                    tier=journey.tier,
                    components=(),
                )
            component_start = len(observer.samples)
            probe_error: Exception | None = None
            elapsed_seconds: float | None = None
            try:
                elapsed_seconds = _validated_elapsed_seconds(
                    await _await_with_timeout(
                        execution.run(),
                        methodology.sample_timeout_seconds,
                    )
                )
            except asyncio.CancelledError as cancellation:
                await _close_preserving_failure(
                    execution,
                    cancellation,
                    timeout_seconds=methodology.sample_timeout_seconds,
                )
            except Exception as exc:
                probe_error = exc
            components = _component_observations(observer.samples[component_start:])
            try:
                await _await_with_timeout(
                    execution.aclose(),
                    methodology.sample_timeout_seconds,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if probe_error is None:
                    probe_error = exc
            if probe_error is None and not components:
                probe_error = MissingComponentEvidenceError()
            if probe_error is not None:
                return LatencyObservation(
                    sample_id=sample_id,
                    phase=LatencyPhase.TURN,
                    thermal_state=ThermalState.WARM,
                    elapsed_seconds=elapsed_seconds,
                    outcome=LatencyObservationOutcome.ERROR,
                    error_type=type(probe_error).__name__,
                    journey_id=journey.journey_id,
                    tier=journey.tier,
                    components=components,
                )
            return LatencyObservation(
                sample_id=sample_id,
                phase=LatencyPhase.TURN,
                thermal_state=ThermalState.WARM,
                elapsed_seconds=elapsed_seconds,
                journey_id=journey.journey_id,
                tier=journey.tier,
                components=components,
            )

    startup_tasks: list[asyncio.Task[LatencyObservation]] = []
    async with asyncio.TaskGroup() as startup_group:
        for index in range(methodology.startup_samples):
            startup_tasks.append(startup_group.create_task(observe_startup(index)))

    turn_tasks: list[asyncio.Task[LatencyObservation]] = []
    async with asyncio.TaskGroup() as turn_group:
        for journey in methodology.journeys:
            for index in range(methodology.samples_per_journey):
                turn_tasks.append(turn_group.create_task(observe_turn(journey, index)))

    report = build_latency_report(
        methodology,
        tuple(task.result() for task in (*startup_tasks, *turn_tasks)),
        run_at=run_at,
        deployment_id=deployment_id,
    )
    return DurableLatencyCertificationRun(
        schema_version=latency_evidence_schema_version(methodology),
        run_at=run_at,
        deployment_id=deployment_id,
        methodology_fingerprint=frozen_fingerprint,
        methodology=methodology,
        outcome=LatencyCertificationOutcome.COMPLETED,
        report=report,
    )
