"""The step-up verification chain (risk_check -> dispatch -> collect), family-parametrized.

Extracted VERBATIM from the refund flow's T3 nodes (Group C) so the profile-change flow can
run the SAME chain over its own pending without a drifting second copy. The factory yields
per-family node INSTANCES with zero runtime branching (everything closed at build, §A5) —
a refund test can still pin that its chain never touches profile state, and vice versa.

The factory owns the chain's BEHAVIOR only. Graph node NAMES and the confirm target stay in
graph.py: `route_after_collect` returns the DECISION ("confirm" | "dispatch" | "handover"),
and each family's graph wrapper maps "confirm" to ITS confirm node (refund -> support_confirm,
profile -> support_profile_confirm) — one collect node name per family, never shared.

A10a invariants carried over unchanged: dispatch is idempotent-pre-interrupt (keyed by
attempt — a replay's re-send is a no-op, S3); collect verifies AFTER its only interrupt and
raises the store level on RETURN (committed before the next node's interrupt); the re-collect
loop is bounded by counted committed misses, never a raw loop (rule 3).
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from langgraph.types import interrupt

from agnostic_market.agents.telemetry import write_event
from agnostic_market.commerce.verification import OtpProvider, RiskProvider, VerificationStore
from agnostic_market.dtos.state import HandoffRequest, HandoffSource, ReasoningState


class _SteppablePending(Protocol):
    """What the chain needs from a pending: the idempotent-dispatch key + the try counter."""

    attempt_key: str
    otp_tries: int

    def model_copy(self, *, update: dict[str, object]) -> object: ...


@dataclass(frozen=True)
class StepupNodes:
    """One family's step-up chain instances + its decision router."""

    risk_check: Callable[[ReasoningState], dict[str, object]]
    dispatch: Callable[[ReasoningState], dict[str, object]]
    collect: Callable[[ReasoningState], dict[str, object]]
    # "confirm" (raised to the required level) | "dispatch" (re-collect) | "handover".
    route_after_collect: Callable[[ReasoningState], str]


def build_stepup_nodes(
    verification_store: VerificationStore,
    otp: OtpProvider,
    risk: RiskProvider,
    *,
    pending_field: str,
    required_level: Callable[[object], int],
    event_prefix: str,
    max_otp_attempts: int,
) -> StepupNodes:
    """Build one family's chain, closed over the stores + the family's state field.

    `pending_field` names the ReasoningState field the chain reads/clears/updates
    ("pending_refund" | "pending_profile_change"); `required_level` computes the level the
    pending demands (the platform floor functions); `event_prefix` keys the telemetry
    family (f"{event_prefix}_stepup_*" — refund events stay byte-identical to pre-factory).
    `max_otp_attempts` is the committed-miss budget before the human handover (§A9,
    merchant-tuned within the platform ceiling — was the hardcoded `_MAX_OTP_ATTEMPTS`).
    """

    def _pending(state: ReasoningState) -> _SteppablePending | None:
        return getattr(state, pending_field)

    def risk_check_node(state: ReasoningState) -> dict[str, object]:
        """SIM-swap / port-out check on the number-on-file (§A4a). Flagged -> do NOT trust an
        OTP; escalate to a person. ANI is never the authenticator."""
        if risk.check_sim_swap():
            write_event({"event": f"{event_prefix}_stepup_failed", "reason": "sim_swap_risk"})
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

    def dispatch_node(state: ReasoningState) -> dict[str, object]:
        """Dispatch the OTP to the number-on-file — IDEMPOTENT per step-up attempt (S3: a
        pre-interrupt effect re-runs on replay; the attempt key makes the re-send a no-op)."""
        pending = _pending(state)
        assert pending is not None
        otp.dispatch(pending.attempt_key)
        return {}

    def collect_node(state: ReasoningState) -> dict[str, object]:
        """HITL interrupt #1: collect the committed OTP, then verify (CODE seam).

        The verify sits AFTER this node's only interrupt — a replay re-runs the node from the
        top, re-hits the interrupt, and re-verifies the freshly-collected code (harmless,
        idempotent: verify_otp just re-checks). It does NOT precede a LATER interrupt, so it
        is not the S3 hazard (that was a dispatch above an interrupt). On match: raise the
        store to L2 and RETURN (commits before the readback interrupt in the next node). On
        miss: re-collect until `max_otp_attempts` is spent (each retry a new attempt key ->
        legit re-dispatch), then human (§A9)."""
        pending = _pending(state)
        assert pending is not None
        answer = interrupt("For security, please read me the 6-digit code we just sent you.")
        if verification_store.verify_otp(str(answer.get("text", ""))):
            write_event({"event": f"{event_prefix}_stepup_ok", "raised_to": 2})
            return {}  # level now L2 in the store; router -> confirm
        tries = pending.otp_tries + 1
        if tries >= max_otp_attempts:
            write_event({"event": f"{event_prefix}_stepup_failed", "reason": "otp_exhausted"})
            return {
                pending_field: None,
                "execution_owner": None,
                "handover": HandoffRequest(
                    destination="human",
                    reason_code="verification_required",
                    source=HandoffSource.DETERMINISTIC_POLICY,
                ),
            }
        # Re-collect: bump tries + a NEW attempt key so the re-dispatch is a legitimate send.
        return {
            pending_field: pending.model_copy(
                update={"otp_tries": tries, "attempt_key": uuid.uuid4().hex}
            )
        }

    def route_after_collect(state: ReasoningState) -> str:
        if state.handover is not None:
            return "handover"  # attempts exhausted -> human
        pending = _pending(state)
        assert pending is not None  # collect clears only via handover
        # Level raised to the requirement -> proceed; otherwise a re-collect was requested.
        return (
            "confirm"
            if verification_store.current_level() >= required_level(pending)
            else "dispatch"
        )

    return StepupNodes(
        risk_check=risk_check_node,
        dispatch=dispatch_node,
        collect=collect_node,
        route_after_collect=route_after_collect,
    )
