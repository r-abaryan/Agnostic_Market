"""Closed failure-lifecycle policy and checkpoint state."""

from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

_FROZEN = ConfigDict(extra="forbid", frozen=True)


class ExceptionAction(StrEnum):
    SAFE_ABORT = "safe_abort"
    CART_REVIEW = "cart_review"
    RECONCILE_PLACEMENT = "reconcile_placement"
    RECONCILE_REFUND = "reconcile_refund"
    RECONCILE_CANCEL = "reconcile_cancel"
    RECONCILE_RETURN = "reconcile_return"
    RECONCILE_PROFILE_CHANGE = "reconcile_profile_change"
    ABORT_PRINCIPAL_WARNING = "abort_principal_warning"
    ABORT_PLACEMENT_CONFIRMATION = "abort_placement_confirmation"
    ABORT_REFUND_VERIFICATION = "abort_refund_verification"
    ABORT_REFUND_CONFIRMATION = "abort_refund_confirmation"
    ABORT_CANCEL_CONFIRMATION = "abort_cancel_confirmation"
    ABORT_RETURN_CONFIRMATION = "abort_return_confirmation"
    ABORT_PROFILE_VERIFICATION = "abort_profile_verification"
    ABORT_PROFILE_CONFIRMATION = "abort_profile_confirmation"
    ABORT_IDENTITY_VERIFICATION = "abort_identity_verification"
    RECONCILE_PRINCIPAL_TRANSITION = "reconcile_principal_transition"
    TERMINAL = "terminal"
    ENGINE_LAST_RESORT = "engine_last_resort"


class AbandonmentKind(StrEnum):
    PURE_ABORT = "pure_abort"
    CART_REVIEW = "cart_review"
    AUTHORITATIVE_RECONCILE = "authoritative_reconcile"
    LIFECYCLE_SPECIAL = "lifecycle_special"
    TERMINAL = "terminal"


class PendingRecovery(BaseModel):
    """PII-free instruction minted by a registered failure owner."""

    model_config = _FROZEN

    origin_node: str = Field(min_length=1)
    action: ExceptionAction
    trigger: Literal["node_exception", "stream_cancelled"]
    abandoned_message_id: str | None = None

    @model_validator(mode="after")
    def abandoned_message_matches_trigger(self) -> Self:
        if self.trigger == "stream_cancelled":
            if self.abandoned_message_id is None or not self.abandoned_message_id.strip():
                raise ValueError(
                    "stream-cancelled recovery requires a nonblank abandoned message ID"
                )
        elif self.abandoned_message_id is not None:
            raise ValueError("node-exception recovery forbids an abandoned message ID")
        return self
