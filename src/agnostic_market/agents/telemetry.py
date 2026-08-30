"""Session-scoped, privacy-filtered runtime telemetry boundaries."""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from enum import StrEnum, auto
from pathlib import Path
from typing import Literal, Protocol

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

from agnostic_market.commerce.spoken import redact_contact

logger = logging.getLogger("agnostic_market.agents.telemetry")

_SENSITIVE_UTTERANCE_REASONS = frozenset({"address_change", "contact_change"})
_REDACTED_UTTERANCE = "[redacted]"
_ENVELOPE_FIELDS = frozenset({"schema_version", "purpose", "tenant_id", "session_id", "event"})


def _validate_scope_id(scope: str, value: str) -> None:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"telemetry {scope} id must be non-empty and normalized")


class TelemetryPurpose(StrEnum):
    OPERATIONAL = "operational"
    ROUTING_EVIDENCE = "routing_evidence"


class OperationalTelemetryEvent(StrEnum):
    AUTOMATION_TERMINAL_RESPONSE = auto()
    CALLER_CONTEXT_CLOSED = auto()
    CANCEL_BATCH_OVER_CAP = auto()
    CANCEL_CONFIRMED = auto()
    CANCEL_DECLINED = auto()
    CANCEL_DENIED = auto()
    CANCEL_EXPIRED = auto()
    CANCEL_RESOLVE_DECLINED = auto()
    CANCEL_RESOLVE_OVER_CAP = auto()
    CANCEL_RESOLVED_FROM_SCOPE = auto()
    CANCEL_STEPUP_TO_HUMAN = auto()
    CAPABILITY_DISPATCH_REJECTED = auto()
    CART_ITEM_ADDED = auto()
    CART_ITEM_REMOVED = auto()
    CART_LEFT = auto()
    CART_MUTATION_CANCELLED = auto()
    CART_MUTATION_EXPIRED = auto()
    CART_QUANTITY_SET = auto()
    CHECKOUT_CANCELLED = auto()
    CHECKOUT_CONFIRMED = auto()
    CHECKOUT_DENIED = auto()
    CHECKOUT_DUPLICATE_FLAGGED = auto()
    CHECKOUT_EXPIRED = auto()
    CLARIFICATION_EXHAUSTED = auto()
    CROSS_THREAD_TURN_CONSUMED = auto()
    DUPLICATE_TURN_IGNORED = auto()
    FLOW_ABANDONED = auto()
    FLOW_ABANDONMENT_OBSERVATION_FAILED = auto()
    HUMAN_ONRAMP = auto()
    IDENTITY_BOUND = auto()
    IDENTITY_BOUND_FOR_ACTION = auto()
    IDENTITY_LEFT = auto()
    IDENTITY_REASK = auto()
    IDENTITY_STEPUP_FAILED = auto()
    IDENTITY_STEPUP_OK = auto()
    IDENTITY_STEPUP_REQUIRED = auto()
    INGRESS_TURN_REJECTED = auto()
    ORDER_LIST_RENDERED = auto()
    ORDER_READ_DENIED = auto()
    ORDER_READ_GRANTED = auto()
    PRINCIPAL_TRANSITION_RECONCILED = auto()
    PRINCIPAL_TRANSITION_SKIPPED = auto()
    PRINCIPAL_TRANSITIONED = auto()
    PROFILE_CHANGE_CANCELLED = auto()
    PROFILE_CHANGE_CONFIRMED = auto()
    PROFILE_CHANGE_DENIED = auto()
    PROFILE_EXPIRED = auto()
    PROFILE_STEPUP_FAILED = auto()
    PROFILE_STEPUP_OK = auto()
    PROFILE_STEPUP_REQUIRED = auto()
    REASONING_CONTEXT_ROTATED = auto()
    REFUND_CANCELLED = auto()
    REFUND_CONFIRMED = auto()
    REFUND_DENIED = auto()
    REFUND_DESTINATION_UNAVAILABLE = auto()
    REFUND_EXPIRED = auto()
    REFUND_NEEDS_HUMAN = auto()
    REFUND_NEEDS_RETURN = auto()
    REFUND_STEERED_TO_CANCEL = auto()
    REFUND_STEERED_TO_RETURN = auto()
    REFUND_STEPUP_FAILED = auto()
    REFUND_STEPUP_OK = auto()
    REFUND_STEPUP_REQUIRED = auto()
    RETURN_CANCELLED = auto()
    RETURN_CONFIRMED = auto()
    RETURN_DENIED = auto()
    RETURN_EXPIRED = auto()
    RETURN_NEEDS_HUMAN = auto()
    RETURN_STEERED_TO_CANCEL = auto()
    ROUTER_NO_ACTION_REJECTED = auto()
    SUPPORT_ACTION_AUTHORIZED = auto()
    SUPPORT_ACTION_NEEDS_IDENTITY = auto()
    SUPPORT_AUTH_DENIED = auto()
    SUPPORT_AUTH_NEEDS_IDENTITY = auto()
    SUPPORT_LEFT = auto()
    TURN_FAILED = auto()
    TURN_RECOVERY_SEEDED = auto()


class RoutingEvidenceTelemetryEvent(StrEnum):
    CAPABILITY_ANSWERED = auto()
    CAPABILITY_OWNER_DECLINED = auto()
    SEMANTIC_HUMAN_REQUESTED = auto()
    SEMANTIC_REQUEST_ABORTED = auto()
    SEMANTIC_ROUTE = auto()
    SEMANTIC_ROUTE_NO_ACTION = auto()


type TelemetryEvent = OperationalTelemetryEvent | RoutingEvidenceTelemetryEvent


class TelemetryRecord(BaseModel):
    """Validated event envelope emitted by one application session."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    purpose: TelemetryPurpose
    tenant_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    event: TelemetryEvent
    attributes: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("tenant_id", "session_id")
    @classmethod
    def validate_scope_id(cls, value: str, info: ValidationInfo) -> str:
        _validate_scope_id(info.field_name.removesuffix("_id"), value)
        return value

    @model_validator(mode="after")
    def event_matches_purpose(self) -> TelemetryRecord:
        expected_type = (
            OperationalTelemetryEvent
            if self.purpose is TelemetryPurpose.OPERATIONAL
            else RoutingEvidenceTelemetryEvent
        )
        if not isinstance(self.event, expected_type):
            raise ValueError(
                f"telemetry event {self.event.value!r} does not belong to purpose "
                f"{self.purpose.value!r}"
            )
        return self

    def flattened(self) -> dict[str, JsonValue]:
        return {
            **self.attributes,
            "schema_version": self.schema_version,
            "purpose": self.purpose.value,
            "tenant_id": self.tenant_id,
            "session_id": self.session_id,
            "event": self.event.value,
        }


class TelemetrySink(Protocol):
    def emit(self, record: TelemetryRecord) -> None: ...


@dataclass(frozen=True, slots=True)
class TelemetryRead:
    records: tuple[TelemetryRecord, ...]
    complete: bool


class TelemetryEvidenceSource(Protocol):
    def mark(self) -> int: ...

    def read_since(self, mark: int) -> TelemetryRead: ...


class TelemetryStore(TelemetrySink, TelemetryEvidenceSource, Protocol):
    pass


class DisabledTelemetrySink:
    """Explicit policy for deployments where runtime telemetry is disabled."""

    def emit(self, record: TelemetryRecord) -> None:
        del record

    def mark(self) -> int:
        return 0

    def read_since(self, mark: int) -> TelemetryRead:
        return TelemetryRead(records=(), complete=mark == 0)


class InMemoryTelemetrySink:
    """Thread-safe telemetry capture for tests and offline evaluation."""

    def __init__(self) -> None:
        self._records: list[TelemetryRecord] = []
        self._lock = threading.Lock()

    def emit(self, record: TelemetryRecord) -> None:
        with self._lock:
            self._records.append(record)

    @property
    def records(self) -> tuple[TelemetryRecord, ...]:
        with self._lock:
            return tuple(self._records)

    def mark(self) -> int:
        with self._lock:
            return len(self._records)

    def read_since(self, mark: int) -> TelemetryRead:
        with self._lock:
            if mark < 0 or mark > len(self._records):
                return TelemetryRead(records=(), complete=False)
            return TelemetryRead(records=tuple(self._records[mark:]), complete=True)


class JsonlTelemetrySink:
    """Best-effort, thread-safe development sink for one configured file."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    def emit(self, record: TelemetryRecord) -> None:
        try:
            with self._lock:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with self._path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record.flattened()) + "\n")
        except OSError as exc:
            logger.warning("telemetry write failed: %s", exc)

    def mark(self) -> int:
        with self._lock:
            return self._path.stat().st_size if self._path.is_file() else 0

    def read_since(self, mark: int) -> TelemetryRead:
        try:
            with self._lock:
                if not self._path.is_file() or self._path.stat().st_size < mark:
                    return TelemetryRead(records=(), complete=False)
                records: list[TelemetryRecord] = []
                with self._path.open("rb") as stream:
                    stream.seek(mark)
                    for raw_line in stream:
                        flattened = json.loads(raw_line.decode("utf-8"))
                        if not isinstance(flattened, dict):
                            return TelemetryRead(records=(), complete=False)
                        metadata = {
                            key: flattened.pop(key, None)
                            for key in (
                                "schema_version",
                                "purpose",
                                "tenant_id",
                                "session_id",
                                "event",
                            )
                        }
                        records.append(
                            TelemetryRecord.model_validate({**metadata, "attributes": flattened})
                        )
                return TelemetryRead(records=tuple(records), complete=True)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValidationError):
            return TelemetryRead(records=(), complete=False)


@dataclass(frozen=True, slots=True)
class TelemetryRecorder:
    """Binds event emission to one tenant, session, and purpose."""

    tenant_id: str
    session_id: str
    purpose: TelemetryPurpose
    sink: TelemetrySink
    _recorded_once_keys: set[str] = field(
        default_factory=set,
        init=False,
        repr=False,
        compare=False,
    )
    _record_once_lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        _validate_scope_id("tenant", self.tenant_id)
        _validate_scope_id("session", self.session_id)

    def _validated_record(self, event: dict[str, object]) -> TelemetryRecord:
        event_name = event.get("event")
        if not isinstance(event_name, str):
            raise ValueError("telemetry event requires a string event name")
        attributes = {key: value for key, value in event.items() if key != "event"}
        conflicts = sorted(_ENVELOPE_FIELDS & attributes.keys())
        if conflicts:
            raise ValueError(
                "telemetry attributes cannot replace envelope fields: " + ", ".join(conflicts)
            )
        utterance = attributes.get("utterance")
        if isinstance(utterance, str):
            attributes["utterance"] = (
                _REDACTED_UTTERANCE
                if attributes.get("reason_code") in _SENSITIVE_UTTERANCE_REASONS
                else redact_contact(utterance)
            )
        return TelemetryRecord.model_validate(
            {
                "purpose": self.purpose,
                "tenant_id": self.tenant_id,
                "session_id": self.session_id,
                "event": event_name,
                "attributes": attributes,
            }
        )

    def record(self, event: dict[str, object]) -> None:
        self.sink.emit(self._validated_record(event))

    def record_once(self, key: str, event: dict[str, object]) -> None:
        """Emit one replay-safe event per recorder-local idempotency key."""
        normalized_key = key.strip()
        if not normalized_key:
            raise ValueError("replay-safe telemetry requires a non-empty key")
        record = self._validated_record(event)
        with self._record_once_lock:
            if normalized_key in self._recorded_once_keys:
                return
            self.sink.emit(record)
            self._recorded_once_keys.add(normalized_key)


@dataclass(frozen=True, slots=True)
class SessionTelemetry:
    operational: TelemetryRecorder
    routing_evidence: TelemetryRecorder

    def __post_init__(self) -> None:
        if (
            self.operational.tenant_id != self.routing_evidence.tenant_id
            or self.operational.session_id != self.routing_evidence.session_id
        ):
            raise ValueError("session telemetry scope is split")
        if (
            self.operational.purpose is not TelemetryPurpose.OPERATIONAL
            or self.routing_evidence.purpose is not TelemetryPurpose.ROUTING_EVIDENCE
        ):
            raise ValueError("session telemetry purpose is invalid")

    @property
    def tenant_id(self) -> str:
        return self.operational.tenant_id

    @property
    def session_id(self) -> str:
        return self.operational.session_id


@dataclass(frozen=True, slots=True)
class TenantTelemetry:
    tenant_id: str
    operational_sink: TelemetrySink
    routing_evidence_sink: TelemetrySink

    def __post_init__(self) -> None:
        _validate_scope_id("tenant", self.tenant_id)

    def bind_session(self, session_id: str) -> SessionTelemetry:
        _validate_scope_id("session", session_id)
        return SessionTelemetry(
            operational=TelemetryRecorder(
                tenant_id=self.tenant_id,
                session_id=session_id,
                purpose=TelemetryPurpose.OPERATIONAL,
                sink=self.operational_sink,
            ),
            routing_evidence=TelemetryRecorder(
                tenant_id=self.tenant_id,
                session_id=session_id,
                purpose=TelemetryPurpose.ROUTING_EVIDENCE,
                sink=self.routing_evidence_sink,
            ),
        )


type CapabilityAnswerSource = Literal[
    "code_authored_read",
    "grounded_model_response",
    "general_model_response",
]


def record_capability_answered(
    telemetry: TelemetryRecorder,
    utterance: str,
    capability: str,
    *,
    answer_source: CapabilityAnswerSource,
) -> None:
    """Record one capability answer after its caller-visible line exists."""
    if not utterance.strip():
        return
    telemetry.record(
        {
            "event": "capability_answered",
            "utterance": utterance,
            "outcome": "answered",
            "outcome_detail": "capability_answer",
            "capability": capability,
            "answer_source": answer_source,
        }
    )
