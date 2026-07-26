"""Closed failure-lifecycle policy and checkpoint state."""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

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


class AbandonmentKind(StrEnum):
    PURE_ABORT = "pure_abort"
    CART_REVIEW = "cart_review"
    AUTHORITATIVE_RECONCILE = "authoritative_reconcile"
    LIFECYCLE_SPECIAL = "lifecycle_special"
    TERMINAL = "terminal"


class PendingRecovery(BaseModel):
    """PII-free instruction minted by a registered node-error handler."""

    model_config = _FROZEN

    origin_node: str = Field(min_length=1)
    action: ExceptionAction
    trigger: Literal["node_exception", "stream_cancelled"]
