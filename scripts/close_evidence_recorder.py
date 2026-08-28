"""Opt-in, scripts-only evidence recorder for Milestone 6E-c live close certification.

The normal worker never constructs this recorder. Certification requires an explicit case and an
absolute report path, validates both before the session starts, and permits one active recorder per
worker process. Reports contain IDs, timestamps, closed slugs, and aggregate state only—never
transcripts, contact values, OTPs, provider payloads, or raw exception text.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import threading
import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Literal, Protocol

from livekit import rtc
from livekit.agents import (
    AgentSession,
    CloseEvent,
    CloseReason,
    ConversationItemAddedEvent,
)
from livekit.agents.llm import ChatMessage
from livekit.agents.voice.room_io.types import DEFAULT_CLOSE_ON_DISCONNECT_REASONS
from pydantic import BaseModel, ConfigDict, Field, model_validator

from agnostic_market.agents import telemetry
from agnostic_market.agents.engine import ReasoningEngine
from agnostic_market.agents.recovery import NodeExecutionTracker
from agnostic_market.checkpoints import SchemaValidatedCheckpointSaver
from agnostic_market.config.loader import load_yaml_layer
from agnostic_market.session import CallerContext

logger = logging.getLogger("close_evidence_recorder")

CLOSE_CERTIFICATION_CONTRACT_VERSION = "2"
CLOSE_CERTIFICATION_CASE_ENV = "VOICE_CLOSE_CERTIFICATION_CASE"
CLOSE_CERTIFICATION_REPORT_ENV = "VOICE_CLOSE_CERTIFICATION_REPORT"
_SUITE_PATH = Path("eval") / "close_lifecycle.yaml"
_STRICT = ConfigDict(extra="forbid", frozen=True)
_ALLOWED_DISCONNECT_REASONS = frozenset(
    rtc.DisconnectReason.Name(reason) for reason in DEFAULT_CLOSE_ON_DISCONNECT_REASONS
)
_TELEMETRY_FIELDS = (
    "caller_context_closed",
    "flow_abandoned",
    "turn_failed",
    "ingress_turn_rejected",
    "checkout_confirmed",
    "refund_confirmed",
    "cancel_confirmed",
    "return_confirmed",
    "profile_change_confirmed",
    "principal_transitioned",
    "reasoning_context_rotated",
    "automation_terminal_response",
)
_ACTIVE_LOCK = threading.Lock()
_ACTIVE_RECORDER: int | None = None

ShutdownMode = Literal["participant_disconnect", "graceful_drain"]
GracefulTrigger = Literal["first_thinking_after_user"]
FailureReason = Literal[
    "lifecycle_close_failed",
    "close_event_missing",
    "duplicate_close_event",
    "unexpected_close_reason",
    "disconnect_event_missing",
    "unexpected_disconnect_reason",
    "caller_context_not_closed",
    "cart_not_cleared",
    "authority_not_cleared",
    "principal_transition_not_cleared",
    "checkpoint_authorization_inspection_failed",
    "reasoning_thread_not_retired",
    "telemetry_unavailable",
    "caller_context_close_count_invalid",
]


class CloseCertificationError(RuntimeError):
    """A certification report could not be completed or failed its structural evidence gate."""


class CloseCaseConfig(BaseModel):
    model_config = _STRICT

    shutdown_mode: ShutdownMode
    graceful_trigger: GracefulTrigger | None = None

    @model_validator(mode="after")
    def trigger_matches_shutdown_mode(self) -> CloseCaseConfig:
        required = self.shutdown_mode == "graceful_drain"
        if required != (self.graceful_trigger is not None):
            raise ValueError("graceful_trigger is required only for graceful_drain cases")
        return self


class CloseCertificationSuite(BaseModel):
    model_config = _STRICT

    contract_version: str = Field(min_length=1)
    finalization_timeout_seconds: float = Field(gt=0, le=60)
    cases: dict[str, CloseCaseConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def contract_is_supported(self) -> CloseCertificationSuite:
        if self.contract_version != CLOSE_CERTIFICATION_CONTRACT_VERSION:
            raise ValueError(
                "unsupported close-certification contract version "
                f"{self.contract_version!r}; expected {CLOSE_CERTIFICATION_CONTRACT_VERSION!r}"
            )
        if any(not name.strip() for name in self.cases):
            raise ValueError("close-certification case names must not be blank")
        return self


@dataclass(frozen=True)
class CloseCertificationRequest:
    case_name: str
    case: CloseCaseConfig
    report_path: Path
    finalization_timeout_seconds: float


class EffectCounts(BaseModel):
    model_config = _STRICT

    placements: int = Field(ge=0)
    refunds: int = Field(ge=0)
    cancellations: int = Field(ge=0)
    returns: int = Field(ge=0)


class CallerStateEvidence(BaseModel):
    model_config = _STRICT

    cart_line_count: int = Field(ge=0)
    identity_bound: bool
    verification_level: int = Field(ge=0)
    verification_grant_count: int = Field(ge=0)
    principal_transition_disposition: Literal["none", "coherent", "inconsistent"]


class MessageEvidence(BaseModel):
    model_config = _STRICT

    message_id: str = Field(min_length=1)
    role: Literal["user", "assistant"]
    created_at: float
    interrupted: bool | None
    text_present: bool


class DisconnectEvidence(BaseModel):
    model_config = _STRICT

    reason: str = Field(min_length=1)
    observed_at: float


class CloseEventEvidence(BaseModel):
    model_config = _STRICT

    reason: str = Field(min_length=1)
    created_at: float
    delivery_count: int = Field(ge=1)


class TelemetryCounts(BaseModel):
    model_config = _STRICT

    caller_context_closed: int = Field(default=0, ge=0)
    flow_abandoned: int = Field(default=0, ge=0)
    turn_failed: int = Field(default=0, ge=0)
    ingress_turn_rejected: int = Field(default=0, ge=0)
    checkout_confirmed: int = Field(default=0, ge=0)
    refund_confirmed: int = Field(default=0, ge=0)
    cancel_confirmed: int = Field(default=0, ge=0)
    return_confirmed: int = Field(default=0, ge=0)
    profile_change_confirmed: int = Field(default=0, ge=0)
    principal_transitioned: int = Field(default=0, ge=0)
    reasoning_context_rotated: int = Field(default=0, ge=0)
    automation_terminal_response: int = Field(default=0, ge=0)


class CloseEvidenceReport(BaseModel):
    model_config = _STRICT

    close_contract_version: str
    installed_livekit_agents_version: str
    merchant_id: str = Field(min_length=1)
    case_name: str = Field(min_length=1)
    shutdown_mode: ShutdownMode
    generated_at: float
    status: Literal["complete", "failed"]
    failure_reasons: tuple[FailureReason, ...]
    disconnect: DisconnectEvidence | None
    close: CloseEventEvidence | None
    messages: tuple[MessageEvidence, ...]
    telemetry: TelemetryCounts
    before_effects: EffectCounts
    after_effects: EffectCounts
    before_caller_state: CallerStateEvidence
    after_caller_state: CallerStateEvidence
    observed_reasoning_thread_count: int = Field(ge=1)
    all_observed_reasoning_threads_retired: bool


def load_close_certification_request(
    config_root: Path,
    environ: Mapping[str, str] | None = None,
) -> CloseCertificationRequest | None:
    """Resolve the opt-in request; incomplete/unknown configuration fails before session start."""
    environment = os.environ if environ is None else environ
    raw_case = environment.get(CLOSE_CERTIFICATION_CASE_ENV)
    raw_report = environment.get(CLOSE_CERTIFICATION_REPORT_ENV)
    if raw_case is None and raw_report is None:
        return None
    if not raw_case or not raw_case.strip() or not raw_report or not raw_report.strip():
        raise CloseCertificationError(
            f"{CLOSE_CERTIFICATION_CASE_ENV} and {CLOSE_CERTIFICATION_REPORT_ENV} "
            "must be set together"
        )

    suite = CloseCertificationSuite.model_validate(load_yaml_layer(config_root / _SUITE_PATH))
    case_name = raw_case.strip()
    case = suite.cases.get(case_name)
    if case is None:
        raise CloseCertificationError(f"unknown close-certification case {case_name!r}")

    report_path = Path(raw_report).expanduser()
    if not report_path.is_absolute():
        raise CloseCertificationError(f"{CLOSE_CERTIFICATION_REPORT_ENV} must be an absolute path")
    if report_path.suffix.lower() != ".json":
        raise CloseCertificationError(f"{CLOSE_CERTIFICATION_REPORT_ENV} must end in .json")
    if report_path.exists():
        raise CloseCertificationError("close-certification report path already exists")
    return CloseCertificationRequest(
        case_name=case_name,
        case=case,
        report_path=report_path,
        finalization_timeout_seconds=suite.finalization_timeout_seconds,
    )


def _claim_active_recorder(token: int) -> None:
    global _ACTIVE_RECORDER
    with _ACTIVE_LOCK:
        if _ACTIVE_RECORDER is not None:
            raise CloseCertificationError(
                "another close-certification recorder is active in this worker"
            )
        _ACTIVE_RECORDER = token


def _release_active_recorder(token: int) -> None:
    global _ACTIVE_RECORDER
    with _ACTIVE_LOCK:
        if token == _ACTIVE_RECORDER:
            _ACTIVE_RECORDER = None


class EffectCountSource(Protocol):
    @property
    def placed_count(self) -> int: ...

    @property
    def refund_count(self) -> int: ...

    @property
    def cancel_count(self) -> int: ...

    @property
    def return_count(self) -> int: ...


def _effect_counts(source: EffectCountSource) -> EffectCounts:
    return EffectCounts(
        placements=source.placed_count,
        refunds=source.refund_count,
        cancellations=source.cancel_count,
        returns=source.return_count,
    )


def _caller_state(context: CallerContext) -> CallerStateEvidence:
    return CallerStateEvidence(
        cart_line_count=len(context.cart_store.snapshot()),
        identity_bound=context.identity_store.current() is not None,
        verification_level=context.verification_store.current_level(),
        verification_grant_count=len(context.verification_store.grants),
        principal_transition_disposition=context.inspect_principal_transition().outcome,
    )


def _telemetry_offset() -> tuple[Path, int]:
    path = telemetry._TELEMETRY_PATH
    return path, path.stat().st_size if path.is_file() else 0


def _telemetry_counts(path: Path, offset: int) -> tuple[TelemetryCounts, bool]:
    try:
        if not path.is_file():
            return TelemetryCounts(), False
        if path.stat().st_size < offset:
            return TelemetryCounts(), False
        counts: Counter[str] = Counter()
        with path.open("rb") as stream:
            stream.seek(offset)
            for raw_line in stream:
                record = json.loads(raw_line.decode("utf-8"))
                event = record.get("event")
                if event in _TELEMETRY_FIELDS:
                    counts[event] += 1
        return (
            TelemetryCounts.model_validate({field: counts[field] for field in _TELEMETRY_FIELDS}),
            True,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return TelemetryCounts(), False


def _write_report_atomic(path: Path, report: CloseEvidenceReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError("close-certification report path already exists")
    descriptor, raw_temp = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temp_path = Path(raw_temp)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(report.model_dump(mode="json"), stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            raise FileExistsError("close-certification report path already exists")
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


class CloseEvidenceRecorder:
    """Collect one redacted live close report after the existing lifecycle reaches full idle."""

    def __init__(
        self,
        request: CloseCertificationRequest,
        *,
        merchant_id: str,
    ) -> None:
        self._request = request
        self._merchant_id = merchant_id
        self._token = id(self)
        self._attached = False
        self._released = False
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._completion: asyncio.Future[Path] | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._finalize_task: asyncio.Task[None] | None = None
        self._session: AgentSession | None = None
        self._room: rtc.Room | None = None
        self._engine: ReasoningEngine | None = None
        self._context: CallerContext | None = None
        self._effect_source: EffectCountSource | None = None
        self._linked_participant_identity: str | None = None
        self._telemetry_path: Path | None = None
        self._telemetry_start = 0
        self._before_effects: EffectCounts | None = None
        self._before_caller_state: CallerStateEvidence | None = None
        self._observed_thread_ids: set[str] = set()
        self._disconnect: DisconnectEvidence | None = None
        self._close_event: CloseEvent | None = None
        self._close_delivery_count = 0
        self._saw_user_item = False
        self._graceful_triggered = False
        self._close_handled = False
        self._lifecycle_close_failed = False

    def attach(
        self,
        *,
        session: AgentSession,
        room: rtc.Room,
        engine: ReasoningEngine,
        effect_source: EffectCountSource,
        linked_participant_identity: str,
    ) -> None:
        """Attach before AgentSession.start; private inspection is intentionally version-pinned."""
        if self._attached:
            raise CloseCertificationError("close-certification recorder is already attached")
        if self._request.report_path.exists():
            raise CloseCertificationError("close-certification report path already exists")
        _claim_active_recorder(self._token)
        try:
            lifecycle = engine._lifecycle
            if not isinstance(lifecycle, CallerContext):
                raise CloseCertificationError(
                    "reasoning engine does not expose the expected CallerContext lifecycle"
                )
            tracker = lifecycle._execution_quiescence
            if not isinstance(tracker, NodeExecutionTracker):
                raise CloseCertificationError(
                    "caller lifecycle does not expose the expected NodeExecutionTracker"
                )
            if engine._graph.checkpointer is None:
                raise CloseCertificationError("reasoning engine has no checkpointer")

            self._event_loop = asyncio.get_running_loop()
            self._completion = self._event_loop.create_future()
            self._session = session
            self._room = room
            self._engine = engine
            self._context = lifecycle
            self._effect_source = effect_source
            self._linked_participant_identity = linked_participant_identity
            self._telemetry_path, self._telemetry_start = _telemetry_offset()
            self._before_effects = _effect_counts(effect_source)
            self._before_caller_state = _caller_state(lifecycle)
            self._observe_current_thread()

            session.on("conversation_item_added", self._on_conversation_item)
            session.on("agent_state_changed", self._on_agent_state_changed)
            session.on("close", self._on_close)
            room.on("participant_disconnected", self._on_participant_disconnected)
            self._attached = True
        except Exception:
            _release_active_recorder(self._token)
            raise

    def _observe_current_thread(self) -> None:
        if self._engine is not None:
            self._observed_thread_ids.add(self._engine._checkpoint_binding.storage_thread_id)

    def _on_conversation_item(self, event: ConversationItemAddedEvent) -> None:
        item = event.item
        if isinstance(item, ChatMessage) and item.role == "user":
            self._saw_user_item = True
        self._observe_current_thread()

    def _on_agent_state_changed(self, event: object) -> None:
        if (
            self._request.case.graceful_trigger == "first_thinking_after_user"
            and self._saw_user_item
            and not self._graceful_triggered
            and getattr(event, "new_state", None) == "thinking"
        ):
            self._graceful_triggered = True
            assert self._session is not None
            self._session.shutdown(drain=True)

    def _on_participant_disconnected(self, participant: rtc.RemoteParticipant) -> None:
        if (
            participant.identity != self._linked_participant_identity
            or self._disconnect is not None
        ):
            return
        reason = participant.disconnect_reason
        reason_name = (
            rtc.DisconnectReason.Name(reason)
            if reason is not None
            else rtc.DisconnectReason.Name(rtc.DisconnectReason.UNKNOWN_REASON)
        )
        self._disconnect = DisconnectEvidence(reason=reason_name, observed_at=time.time())

    def _on_close(self, event: CloseEvent) -> None:
        self._close_delivery_count += 1
        if self._close_event is None:
            self._close_event = event
        self._observe_current_thread()
        if self._close_handled:
            return
        self._close_handled = True
        assert self._context is not None
        self._close_task = asyncio.create_task(self._close_and_notify())

    async def _close_and_notify(self) -> None:
        assert self._context is not None
        try:
            await self._context.aclose_session()
        except Exception:
            self._lifecycle_close_failed = True
            logger.exception("close certification lifecycle teardown failed")
        finally:
            self._on_fully_idle()

    def _on_fully_idle(self) -> None:
        assert self._event_loop is not None
        self._event_loop.call_soon_threadsafe(self._start_finalize_task)

    def _start_finalize_task(self) -> None:
        if self._finalize_task is None:
            self._finalize_task = asyncio.create_task(self._finalize())

    def _message_evidence(self) -> tuple[MessageEvidence, ...]:
        assert self._session is not None
        evidence: list[MessageEvidence] = []
        for item in self._session.history.items:
            if not isinstance(item, ChatMessage) or item.role not in {"user", "assistant"}:
                continue
            evidence.append(
                MessageEvidence(
                    message_id=item.id,
                    role=item.role,
                    created_at=item.created_at,
                    interrupted=item.interrupted if item.role == "assistant" else None,
                    text_present=bool(item.text_content),
                )
            )
        return tuple(evidence)

    def _checkpoint_evidence(self) -> tuple[bool, bool]:
        assert self._engine is not None
        try:
            checkpointer = self._engine._graph.checkpointer
            if not isinstance(checkpointer, SchemaValidatedCheckpointSaver):
                return False, False
            all_retired = all(
                not checkpointer.thread_authorized(thread_id)
                for thread_id in self._observed_thread_ids
            )
        except Exception:
            return False, False
        return all_retired, True

    async def _build_report(self) -> CloseEvidenceReport:
        assert self._context is not None
        assert self._effect_source is not None
        assert self._telemetry_path is not None
        assert self._before_effects is not None
        assert self._before_caller_state is not None

        after_effects = _effect_counts(self._effect_source)
        after_state = _caller_state(self._context)
        telemetry_counts, telemetry_available = _telemetry_counts(
            self._telemetry_path,
            self._telemetry_start,
        )
        threads_retired, checkpoint_authorization_available = self._checkpoint_evidence()
        failures: list[FailureReason] = []
        if self._lifecycle_close_failed:
            failures.append("lifecycle_close_failed")
        if self._close_event is None:
            failures.append("close_event_missing")
        if self._close_delivery_count > 1:
            failures.append("duplicate_close_event")
        if self._close_event is not None:
            expected_close = (
                CloseReason.PARTICIPANT_DISCONNECTED
                if self._request.case.shutdown_mode == "participant_disconnect"
                else CloseReason.USER_INITIATED
            )
            if self._close_event.reason != expected_close:
                failures.append("unexpected_close_reason")
        if self._request.case.shutdown_mode == "participant_disconnect":
            if self._disconnect is None:
                failures.append("disconnect_event_missing")
            elif self._disconnect.reason not in _ALLOWED_DISCONNECT_REASONS:
                failures.append("unexpected_disconnect_reason")
        if not self._context._closed:
            failures.append("caller_context_not_closed")
        if after_state.cart_line_count:
            failures.append("cart_not_cleared")
        if (
            after_state.identity_bound
            or after_state.verification_level != 1
            or after_state.verification_grant_count
        ):
            failures.append("authority_not_cleared")
        if after_state.principal_transition_disposition != "none":
            failures.append("principal_transition_not_cleared")
        if not checkpoint_authorization_available:
            failures.append("checkpoint_authorization_inspection_failed")
        elif not threads_retired:
            failures.append("reasoning_thread_not_retired")
        if not telemetry_available:
            failures.append("telemetry_unavailable")
        elif telemetry_counts.caller_context_closed != 1:
            failures.append("caller_context_close_count_invalid")

        close_evidence = (
            CloseEventEvidence(
                reason=self._close_event.reason.value,
                created_at=self._close_event.created_at,
                delivery_count=self._close_delivery_count,
            )
            if self._close_event is not None
            else None
        )
        return CloseEvidenceReport(
            close_contract_version=CLOSE_CERTIFICATION_CONTRACT_VERSION,
            installed_livekit_agents_version=version("livekit-agents"),
            merchant_id=self._merchant_id,
            case_name=self._request.case_name,
            shutdown_mode=self._request.case.shutdown_mode,
            generated_at=time.time(),
            status="failed" if failures else "complete",
            failure_reasons=tuple(failures),
            disconnect=self._disconnect,
            close=close_evidence,
            messages=self._message_evidence(),
            telemetry=telemetry_counts,
            before_effects=self._before_effects,
            after_effects=after_effects,
            before_caller_state=self._before_caller_state,
            after_caller_state=after_state,
            observed_reasoning_thread_count=len(self._observed_thread_ids),
            all_observed_reasoning_threads_retired=threads_retired,
        )

    async def _finalize(self) -> None:
        assert self._completion is not None
        try:
            report = await self._build_report()
            await asyncio.to_thread(_write_report_atomic, self._request.report_path, report)
            if report.status == "failed":
                self._completion.set_exception(
                    CloseCertificationError(
                        "close-certification evidence report failed its structural gate"
                    )
                )
            else:
                self._completion.set_result(self._request.report_path)
        except Exception:
            logger.exception("close-certification report finalization failed")
            if not self._completion.done():
                self._completion.set_exception(
                    CloseCertificationError(
                        "close-certification evidence report could not be finalized"
                    )
                )
        finally:
            self._release()

    def _release(self) -> None:
        if not self._released:
            self._released = True
            _release_active_recorder(self._token)

    async def wait_for_completion(self, _shutdown_reason: str | None = None) -> Path:
        """Job-shutdown callback: wait for the same one-shot finalization task, never a sleep."""
        if self._completion is None:
            raise CloseCertificationError("close-certification recorder was not attached")
        try:
            return await asyncio.wait_for(
                asyncio.shield(self._completion),
                timeout=self._request.finalization_timeout_seconds,
            )
        except TimeoutError as exc:
            self._release()
            raise CloseCertificationError("close-certification finalization timed out") from exc
