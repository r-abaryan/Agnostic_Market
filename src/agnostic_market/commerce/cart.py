"""Session-authoritative cart state and confirmed-mutation receipts.

The model selects only bounded candidates. Code resolves their catalog identity, and the
confirmed mutation ledger prevents retry or recovery from applying one approved change twice.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agnostic_market.commerce.receipts import ReceiptLookup, classify_receipt
from agnostic_market.dtos.money import UsdAmount, validate_usd
from agnostic_market.dtos.orchestration import CartOperation
from agnostic_market.dtos.state import CartLine

_FROZEN = ConfigDict(extra="forbid", frozen=True)


class CartMutationError(RuntimeError):
    """A confirmed cart mutation violated its idempotency contract."""


class CartMutationRecord(BaseModel):
    """Authoritative result retained for mutation replay and recovery."""

    model_config = _FROZEN

    operation: CartOperation
    sku: str = Field(min_length=1)
    name: str = Field(min_length=1)
    price_usd: UsdAmount
    quantity: int | None = Field(default=None, strict=True, ge=0)
    pre_confirm_quantity: int = Field(strict=True, ge=0)
    previous_quantity: int = Field(strict=True, ge=0)
    final_quantity: int = Field(strict=True, ge=0)
    outcome: Literal["applied", "unchanged"]

    @model_validator(mode="after")
    def result_is_coherent(self) -> CartMutationRecord:
        if self.operation == "remove":
            if self.quantity is not None or self.final_quantity != 0:
                raise ValueError("remove result has an invalid quantity shape")
        elif self.operation == "add":
            if self.quantity is None or self.quantity < 1:
                raise ValueError("add result requires a positive quantity")
            if self.final_quantity != self.previous_quantity + self.quantity:
                raise ValueError("add result does not match the applied delta")
        elif self.quantity is None or self.final_quantity != self.quantity:
            raise ValueError("set-quantity result does not match its confirmed final quantity")
        changed = self.previous_quantity != self.final_quantity
        if (self.outcome == "applied") != changed:
            raise ValueError("cart mutation outcome does not match the quantity change")
        return self


def _mutation_matches(
    record: CartMutationRecord,
    *,
    operation: CartOperation,
    sku: str,
    name: str,
    price_usd: UsdAmount,
    quantity: int | None,
    pre_confirm_quantity: int,
) -> bool:
    return bool(
        record.operation == operation
        and record.sku == sku
        and record.name == name
        and record.price_usd == price_usd
        and record.quantity == quantity
        and record.pre_confirm_quantity == pre_confirm_quantity
    )


class CartStore:
    """The mutable per-session cart (keyed by sku). Reaped on session close."""

    def __init__(self) -> None:
        self._lines: dict[str, CartLine] = {}
        self._mutations_by_key: dict[str, CartMutationRecord] = {}

    def add_item(self, *, sku: str, name: str, price_usd: UsdAmount, quantity: int) -> CartLine:
        """Add `quantity` of an item — ADDITIVE (a second add increments the existing line).
        `sku`/`name`/`price_usd` are code-resolved from the candidate list, never model-set."""
        existing = self._lines.get(sku)
        new_quantity = (existing.quantity if existing else 0) + quantity
        line = CartLine(sku=sku, name=name, price_usd=price_usd, quantity=new_quantity)
        self._lines[sku] = line
        return line

    def set_quantity(self, sku: str, quantity: int) -> CartLine | None:
        """Set an item's quantity outright. `quantity <= 0` removes the line (returns None)."""
        if quantity <= 0:
            self._lines.pop(sku, None)
            return None
        existing = self._lines.get(sku)
        if existing is None:
            return None
        line = existing.model_copy(update={"quantity": quantity})
        self._lines[sku] = line
        return line

    def remove_item(self, sku: str) -> bool:
        """Drop a line entirely. True if it was present."""
        return self._lines.pop(sku, None) is not None

    def apply_confirmed_mutation(
        self,
        idempotency_key: str,
        *,
        operation: CartOperation,
        sku: str,
        name: str,
        price_usd: UsdAmount,
        quantity: int | None,
        pre_confirm_quantity: int,
    ) -> CartMutationRecord:
        """Apply one confirmed mutation once and retain its factual result."""

        if not idempotency_key.strip():
            raise CartMutationError("cart mutation idempotency key must not be blank")

        checked_price = validate_usd(price_usd)
        existing_record = self._mutations_by_key.get(idempotency_key)
        if existing_record is not None:
            if not _mutation_matches(
                existing_record,
                operation=operation,
                sku=sku,
                name=name,
                price_usd=checked_price,
                quantity=quantity,
                pre_confirm_quantity=pre_confirm_quantity,
            ):
                raise CartMutationError(
                    "cart mutation idempotency key was reused with different parameters"
                )
            return existing_record

        current = self._lines.get(sku)
        previous_quantity = current.quantity if current is not None else 0
        if operation == "add":
            if quantity is None or quantity < 1:
                raise CartMutationError("confirmed add requires a positive quantity")
            final_quantity = previous_quantity + quantity
        elif operation == "remove":
            if quantity is not None:
                raise CartMutationError("confirmed remove forbids a quantity")
            final_quantity = 0
        elif operation == "set_quantity":
            if quantity is None or quantity < 0:
                raise CartMutationError("confirmed set-quantity requires a final quantity")
            final_quantity = quantity
        else:
            raise CartMutationError(f"unsupported confirmed cart operation: {operation!r}")

        record = CartMutationRecord(
            operation=operation,
            sku=sku,
            name=name,
            price_usd=checked_price,
            quantity=quantity,
            pre_confirm_quantity=pre_confirm_quantity,
            previous_quantity=previous_quantity,
            final_quantity=final_quantity,
            outcome="applied" if previous_quantity != final_quantity else "unchanged",
        )
        if final_quantity == 0:
            self._lines.pop(sku, None)
        else:
            self._lines[sku] = CartLine(
                sku=sku,
                name=name,
                price_usd=checked_price,
                quantity=final_quantity,
            )
        self._mutations_by_key[idempotency_key] = record
        return record

    def mutation_receipt(
        self,
        idempotency_key: str,
        *,
        operation: CartOperation,
        sku: str,
        name: str,
        price_usd: UsdAmount,
        quantity: int | None,
        pre_confirm_quantity: int,
    ) -> ReceiptLookup[CartMutationRecord]:
        """Read one confirmed-mutation result without changing the cart."""

        if not idempotency_key.strip():
            raise CartMutationError("cart mutation idempotency key must not be blank")

        checked_price = validate_usd(price_usd)
        return classify_receipt(
            self._mutations_by_key.get(idempotency_key),
            lambda record: _mutation_matches(
                record,
                operation=operation,
                sku=sku,
                name=name,
                price_usd=checked_price,
                quantity=quantity,
                pre_confirm_quantity=pre_confirm_quantity,
            ),
        )

    def view(self) -> tuple[CartLine, ...]:
        """The current lines in a deterministic (insertion) order — for the readback and the
        frontline `view_cart` tool."""
        return tuple(self._lines.values())

    def is_empty(self) -> bool:
        return not self._lines

    def cart_total(self) -> UsdAmount:
        return sum((line.line_total for line in self._lines.values()), start=Decimal("0"))

    def snapshot(self) -> tuple[CartLine, ...]:
        """A frozen copy of the lines for `PendingPlacement` (the confirm/place nodes read
        THIS, never the live cart, so consent is over exactly what was read back)."""
        return tuple(self._lines.values())

    def clear(self) -> None:
        """Clear caller-scoped lines and mutation receipts."""
        self._lines.clear()
        self._mutations_by_key.clear()

    @property
    def line_count(self) -> int:
        """How many DISTINCT lines the cart holds (test/verification surface)."""
        return len(self._lines)
