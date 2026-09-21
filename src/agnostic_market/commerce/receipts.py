"""Read-only outcomes from an authoritative idempotency ledger."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

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


class CartReceiptCounts(BaseModel):
    """Committed cart-mutation receipts without receipt identifiers or payloads."""

    model_config = _FROZEN

    mutations: int = Field(ge=0)


class OrderReceiptCounts(BaseModel):
    """Cumulative committed order-ledger receipts for one tenant-bound adapter."""

    model_config = _FROZEN

    placements: int = Field(ge=0)
    refunds: int = Field(ge=0)
    returns: int = Field(ge=0)
    cancellations: int = Field(ge=0)


class ProfileReceiptCounts(BaseModel):
    """Cumulative committed profile-change receipts for one tenant-bound adapter."""

    model_config = _FROZEN

    changes: int = Field(ge=0)


class CommerceReceiptCounts(BaseModel):
    """Bounded value-free receipt evidence exposed by the simulator."""

    model_config = _FROZEN

    cart: CartReceiptCounts
    orders: OrderReceiptCounts
    profiles: ProfileReceiptCounts


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
