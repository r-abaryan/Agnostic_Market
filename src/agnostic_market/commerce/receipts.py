"""Read-only outcomes from an authoritative idempotency ledger."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, ConfigDict

_FROZEN = ConfigDict(extra="forbid", frozen=True)

IndeterminateReason = Literal["key_conflict", "pending", "unavailable"]


class CommittedReceipt[RecordT](BaseModel):
    model_config = _FROZEN

    kind: Literal["committed"] = "committed"
    record: RecordT


class NotCommittedReceipt(BaseModel):
    model_config = _FROZEN

    kind: Literal["not_committed"] = "not_committed"


class IndeterminateReceipt(BaseModel):
    model_config = _FROZEN

    kind: Literal["indeterminate"] = "indeterminate"
    reason: IndeterminateReason


type ReceiptLookup[RecordT] = CommittedReceipt[RecordT] | NotCommittedReceipt | IndeterminateReceipt


def classify_receipt[RecordT](
    record: RecordT | None,
    matches: Callable[[RecordT], bool],
) -> ReceiptLookup[RecordT]:
    """Classify one ledger row without exposing a conflicting record."""
    if record is None:
        return NotCommittedReceipt()
    if not matches(record):
        return IndeterminateReceipt(reason="key_conflict")
    return CommittedReceipt(record=record)
