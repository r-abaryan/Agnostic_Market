"""The session cart — mutable working state (Group B), the counterpart to the OrderStore.

`CartStore` is per-session, server-authoritative cart state: the caller's line items live
HERE, never in the model's memory (the model picks a candidate KEY; code resolves it and
mutates this store). It has the OPPOSITE lifecycle to the OrderStore — mutable, ephemeral,
emptied on placement, cleared on session teardown — so it is a separate store built
alongside the OrderStore in `voice/pipeline.py` (same pattern as `VerificationStore`).

No idempotency key or version on the cart itself: mutations are REVERSIBLE (re-adding an
item is a legitimate action, not a replay to dedup). Idempotency lives only at the
irreversible boundary — `PendingPlacement.idempotency_key` + `OrderStore.place_cart`.
"""

from __future__ import annotations

from agnostic_market.dtos.state import CartLine


class CartStore:
    """The mutable per-session cart (keyed by sku). Reaped on session close."""

    def __init__(self) -> None:
        self._lines: dict[str, CartLine] = {}

    def add_item(self, *, sku: str, name: str, price_usd: float, quantity: int) -> CartLine:
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

    def view(self) -> tuple[CartLine, ...]:
        """The current lines in a deterministic (insertion) order — for the readback and the
        frontline `view_cart` tool."""
        return tuple(self._lines.values())

    def is_empty(self) -> bool:
        return not self._lines

    def cart_total(self) -> float:
        return round(sum(line.line_total for line in self._lines.values()), 2)

    def snapshot(self) -> tuple[CartLine, ...]:
        """A frozen copy of the lines for `PendingPlacement` (the confirm/place nodes read
        THIS, never the live cart, so consent is over exactly what was read back)."""
        return tuple(self._lines.values())

    def clear(self) -> None:
        """Empty the cart (after a successful placement, or on session teardown)."""
        self._lines.clear()

    @property
    def line_count(self) -> int:
        """How many DISTINCT lines the cart holds (test/verification surface)."""
        return len(self._lines)
