"""Shared risk, dispatch, and collection nodes for subject-bound step-up flows.

Dispatch replays the same provider challenge. Collection retries that challenge, while the
provider owns expiry and the attempt budget. A protected action may reuse only a fresh,
same-factor proof after its own risk check; identity collection still requires its exact proof.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal, Protocol, Self

from langgraph.types import interrupt

from agnostic_market.agents.telemetry import OperationalTelemetryEvent, TelemetryRecorder
from agnostic_market.commerce.verification import (
    OtpVerificationStatus,
    RiskDecision,
    RiskPort,
    VerificationStore,
    VerificationSubject,
)
from agnostic_market.dtos.state import HandoffRequest, HandoffSource, ReasoningState


class _SteppablePending(Protocol):
    """The challenge state shared by each step-up family."""

    customer_ref: str | None
    factor_ref: str | None
    attempt_key: str
    challenge_id: str | None

    def model_copy(self, *, update: dict[str, object]) -> Self: ...


@dataclass(frozen=True)
class StepupNodes:
    """One family's step-up chain instances + its decision router."""

    risk_check: Callable[[ReasoningState], Awaitable[dict[str, object]]]
    dispatch: Callable[[ReasoningState], Awaitable[dict[str, object]]]
    collect: Callable[[ReasoningState], Awaitable[dict[str, object]]]
    route_after_risk: Callable[[ReasoningState], str]
    # "confirm" (raised to the required level) | "collect" (retry) | "handover".
    route_after_collect: Callable[[ReasoningState], str]


type StepupEventFamily = Literal["identity", "profile", "refund"]


@dataclass(frozen=True)
class _StepupTelemetryEvents:
    succeeded: OperationalTelemetryEvent
    failed: OperationalTelemetryEvent


_STEPUP_TELEMETRY_EVENTS: dict[StepupEventFamily, _StepupTelemetryEvents] = {
    "identity": _StepupTelemetryEvents(
        succeeded=OperationalTelemetryEvent.IDENTITY_STEPUP_OK,
        failed=OperationalTelemetryEvent.IDENTITY_STEPUP_FAILED,
    ),
    "profile": _StepupTelemetryEvents(
        succeeded=OperationalTelemetryEvent.PROFILE_STEPUP_OK,
        failed=OperationalTelemetryEvent.PROFILE_STEPUP_FAILED,
    ),
    "refund": _StepupTelemetryEvents(
        succeeded=OperationalTelemetryEvent.REFUND_STEPUP_OK,
        failed=OperationalTelemetryEvent.REFUND_STEPUP_FAILED,
    ),
}


def build_stepup_nodes[PendingT: _SteppablePending](
    verification_store: VerificationStore,
    risk: RiskPort,
    *,
    pending_field: str,
    pending_type: type[PendingT],
    required_level: Callable[[PendingT], int],
    subject: Callable[[PendingT], VerificationSubject],
    event_prefix: StepupEventFamily,
    reuse_existing_proof: bool,
    max_otp_attempts: int,
    telemetry: TelemetryRecorder,
) -> StepupNodes:
    """Build one family's chain, closed over the stores + the family's state field.

    `pending_field` names the ReasoningState field the chain reads or updates.
    `required_level` computes the platform floor, `subject` binds the complete authority scope,
    and `event_prefix` selects the closed telemetry event family. `reuse_existing_proof` controls
    whether the post-risk router may accept another fresh proof for the same authority dimensions.
    `max_otp_attempts` is the committed-miss budget before the human handover (§A9,
    merchant-tuned within the platform ceiling — was the hardcoded `_MAX_OTP_ATTEMPTS`).
    """

    telemetry_events = _STEPUP_TELEMETRY_EVENTS[event_prefix]

    def _pending(state: ReasoningState) -> PendingT | None:
        pending = getattr(state, pending_field)
        if pending is None:
            return None
        if not isinstance(pending, pending_type):
            raise TypeError(f"{pending_field} has an unexpected pending-state type")
        return pending

    async def risk_check_node(state: ReasoningState) -> dict[str, object]:
        """Escalate when the exact verification subject is blocked by risk policy."""
        pending = _pending(state)
        assert pending is not None
        if await risk.assess(subject(pending)) is RiskDecision.BLOCKED:
            telemetry.record({"event": telemetry_events.failed, "reason": "sim_swap_risk"})
            return {
                pending_field: None,
                "execution_owner": None,
                "handover": HandoffRequest(
                    destination="human",
                    reason_code="verification_required",
                    source=HandoffSource.DETERMINISTIC_POLICY,
                ),
            }
        return {}

    def route_after_risk(state: ReasoningState) -> str:
        if state.handover is not None:
            return "handover"
        pending = _pending(state)
        assert pending is not None
        if reuse_existing_proof and verification_store.authorization_satisfies(
            subject(pending),
            required_level(pending),
        ):
            return "confirm"
        return "dispatch"

    async def dispatch_node(state: ReasoningState) -> dict[str, object]:
        """Idempotently dispatch or recover this pending action's challenge."""
        pending = _pending(state)
        assert pending is not None
        challenge = await verification_store.dispatch_otp(
            subject=subject(pending),
            dispatch_idempotency_key=pending.attempt_key,
            max_attempts=max_otp_attempts,
        )
        if pending.challenge_id is not None and pending.challenge_id != challenge.challenge_id:
            raise RuntimeError("OTP dispatch replay returned a different challenge")
        if pending.challenge_id == challenge.challenge_id:
            return {}
        return {pending_field: pending.model_copy(update={"challenge_id": challenge.challenge_id})}

    async def collect_node(state: ReasoningState) -> dict[str, object]:
        """Collect one committed code and verify the existing provider challenge."""
        pending = _pending(state)
        assert pending is not None
        if pending.challenge_id is None:
            raise RuntimeError("OTP collect requires a dispatched challenge")
        answer = interrupt("For security, please read me the 6-digit code we just sent you.")
        outcome = await verification_store.verify_otp(
            pending.challenge_id,
            str(answer.get("text", "")),
        )
        if outcome is OtpVerificationStatus.VERIFIED:
            telemetry.record({"event": telemetry_events.succeeded, "raised_to": 2})
            return {}
        if outcome is OtpVerificationStatus.MISMATCHED:
            return {}
        telemetry.record({"event": telemetry_events.failed, "reason": f"otp_{outcome.value}"})
        return {
            pending_field: None,
            "execution_owner": None,
            "handover": HandoffRequest(
                destination="human",
                reason_code="verification_required",
                source=HandoffSource.DETERMINISTIC_POLICY,
            ),
        }

    def route_after_collect(state: ReasoningState) -> str:
        if state.handover is not None:
            return "handover"  # attempts exhausted -> human
        pending = _pending(state)
        assert pending is not None  # collect clears only via handover
        if pending.challenge_id is None:
            raise RuntimeError("OTP collect completed without a dispatched challenge")
        # Only this challenge may authorize the protected action.
        return (
            "confirm"
            if verification_store.challenge_satisfies(
                pending.challenge_id,
                subject(pending),
                required_level(pending),
            )
            else "collect"
        )

    return StepupNodes(
        risk_check=risk_check_node,
        dispatch=dispatch_node,
        collect=collect_node,
        route_after_risk=route_after_risk,
        route_after_collect=route_after_collect,
    )
