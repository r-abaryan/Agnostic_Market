"""Order SoR access — the fixture-backed stub store (Phase 3b; real SoR integration Phase 4).

Moved here from voice/tools.py at 3b: the fixture is COMMERCE data (orders + catalog), not
a voice concern — voice tools read it through this module, keeping the plane arrow
voice -> commerce, never the reverse.

`OrderStore` is the **order-SoR arbiter** (AGENTS §A10 rule 5): `place()` is idempotent by
`idempotency_key` — a seen key returns the SAME placed order and never creates a duplicate,
so ANY replay/retry of the place effect (crash between effect and checkpoint, double
resume) cannot double-order. That dedup lives HERE, in the store, because in production the
merchant's order SoR is the arbiter — the graph must not be the thing that remembers.

`resolve_candidates` is the CODE-side product search for checkout selection: the model
never emits a raw SKU; it picks a `candidate_key` INTO the bounded list this returns, and
code resolves key -> sku -> price (industry-standard narrowed-choice selection). It is a
separate surface from the prose `catalog_search` tool on purpose — same fixture data, one
prose surface to speak from, one structured surface to select from.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agnostic_market.config.loader import ConfigError, load_yaml_layer

_STRICT = ConfigDict(extra="forbid")
_FROZEN = ConfigDict(extra="forbid", frozen=True)

# Stub-store copy for orders this store itself placed (real ETA logic = the Phase-4 SoR).
_PLACED_STATUS = "processing"
_PLACED_ETA = "3-5 business days"


class _OrderEntry(BaseModel):
    model_config = _STRICT

    status: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    eta: str = Field(min_length=1)
    # Captured amount — the refund cumulative-cap join reads this (§A4b); a real SoR always
    # knows what it captured. Required so a refund can never run against an unknown total.
    total_usd: float = Field(ge=0)


class _ProductEntry(BaseModel):
    model_config = _STRICT

    sku: str = Field(min_length=1)
    name: str = Field(min_length=1)
    price_usd: float = Field(ge=0)


class OrdersFixture(BaseModel):
    """Validated stub SoR content for one merchant."""

    model_config = _STRICT

    orders: dict[str, _OrderEntry]
    products: list[_ProductEntry] = Field(min_length=1)


class Candidate(BaseModel):
    """One code-narrowed product option the checkout model may pick (by key, never SKU)."""

    model_config = _FROZEN

    key: str = Field(min_length=1)
    sku: str = Field(min_length=1)
    name: str = Field(min_length=1)
    price_usd: float = Field(ge=0)


class OrderCandidate(BaseModel):
    """One code-narrowed order option the support model may pick (by key, never order_id).

    Same SKU-discipline stance as `Candidate`: the model never emits a raw order id; it
    picks a KEY into this bounded list and code resolves key -> order_id + captured total.
    """

    model_config = _FROZEN

    key: str = Field(min_length=1)
    order_id: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    total_usd: float = Field(ge=0)


class PlacedOrder(BaseModel):
    """An order this store placed (the stub SoR's own record)."""

    model_config = _FROZEN

    order_id: str = Field(min_length=1)
    sku: str = Field(min_length=1)
    name: str = Field(min_length=1)
    quantity: int = Field(ge=1)
    total_usd: float = Field(ge=0)


class RefundRecord(BaseModel):
    """A refund this store issued (the stub SoR's own record)."""

    model_config = _FROZEN

    return_id: str = Field(min_length=1)
    order_id: str = Field(min_length=1)
    amount_usd: float = Field(ge=0)
    destination: str = Field(min_length=1)


class RefundError(ValueError):
    """A refund the store refused: unknown order, or over the order's refundable balance."""


def load_orders_fixture(config_root: Path, merchant_id: str) -> OrdersFixture:
    """Load + validate the merchant's orders fixture. Fails loudly (build time, not mid-call)."""
    path = config_root / "fixtures" / "orders" / f"{merchant_id}.yaml"
    try:
        return OrdersFixture.model_validate(load_yaml_layer(path))
    except ValidationError as exc:
        raise ConfigError(f"orders fixture {path} failed validation:\n{exc}") from exc


def resolve_candidates(fixture: OrdersFixture, query: str) -> list[Candidate]:
    """Code-side product search for checkout selection — bounded, keyed, deterministic.

    `query` may be a short phrase ("rain jacket") or a WHOLE utterance ("I'd like the
    waterproof rain jacket"), so containment is checked both ways. Empty/no-match queries
    return the FULL (tiny, fixture-bounded) catalog so the model can steer the caller to
    real items; keys are 1-based list positions as strings.
    """
    needle = query.strip().lower()
    matched = [
        p
        for p in fixture.products
        if needle and (needle in p.name.lower() or p.name.lower() in needle)
    ]
    products = matched or fixture.products
    return [
        Candidate(key=str(i), sku=p.sku, name=p.name, price_usd=p.price_usd)
        for i, p in enumerate(products, start=1)
    ]


class OrderStore:
    """The stub order SoR: fixture reads + idempotent placement (the dedup ARBITER)."""

    def __init__(self, fixture: OrdersFixture) -> None:
        self.fixture = fixture
        self._placed_by_key: dict[str, PlacedOrder] = {}
        self._next_seq = 9001  # placed-order ids ORD-9001.. (disjoint from fixture ids)
        self._refunds_by_key: dict[str, RefundRecord] = {}
        self._next_refund_seq = 7001  # return ids R-7001..

    def order_summary(self, order_id: str) -> str | None:
        """Human-readable status line for fixture AND just-placed orders; None if unknown."""
        normalized = order_id.strip().upper()
        entry = self.fixture.orders.get(normalized)
        if entry is not None:
            return f"Order {normalized}: {entry.summary} - status {entry.status}, ETA {entry.eta}."
        for placed in self._placed_by_key.values():
            if placed.order_id == normalized:
                return (
                    f"Order {normalized}: {placed.quantity} x {placed.name} "
                    f"(${placed.total_usd:.2f}) - status {_PLACED_STATUS}, ETA {_PLACED_ETA}."
                )
        return None

    def place(
        self, idempotency_key: str, *, sku: str, name: str, quantity: int, total_usd: float
    ) -> PlacedOrder:
        """Place an order, deduplicated by `idempotency_key` (SoR-arbiter rule).

        A repeat call with a seen key returns the ORIGINAL placed order unchanged — the
        replay/retry path can never create a second order or drift the recorded values.
        """
        existing = self._placed_by_key.get(idempotency_key)
        if existing is not None:
            return existing
        placed = PlacedOrder(
            order_id=f"ORD-{self._next_seq}",
            sku=sku,
            name=name,
            quantity=quantity,
            total_usd=total_usd,
        )
        self._next_seq += 1
        self._placed_by_key[idempotency_key] = placed
        return placed

    @property
    def placed_count(self) -> int:
        """How many DISTINCT orders this store has placed (test/verification surface)."""
        return len(self._placed_by_key)

    def refundable_orders(self) -> list[OrderCandidate]:
        """The bounded, keyed list of orders a refund may target (fixture + placed).

        Code-side narrowing for support selection — the model picks a KEY into this, never a
        raw order id (SKU-discipline analogue). Keys are 1-based positions as strings.
        """
        candidates: list[tuple[str, str, float]] = [
            (oid, entry.summary, entry.total_usd) for oid, entry in self.fixture.orders.items()
        ]
        candidates += [
            (p.order_id, f"{p.quantity} x {p.name}", p.total_usd)
            for p in self._placed_by_key.values()
        ]
        return [
            OrderCandidate(key=str(i), order_id=oid, summary=summary, total_usd=total)
            for i, (oid, summary, total) in enumerate(candidates, start=1)
        ]

    def captured_total(self, order_id: str) -> float | None:
        """The captured amount for an order (fixture OR placed); None if unknown.

        The refund cumulative-cap join reads this — a refund against an order whose total
        the store doesn't know is refused, never guessed.
        """
        normalized = order_id.strip().upper()
        entry = self.fixture.orders.get(normalized)
        if entry is not None:
            return entry.total_usd
        for placed in self._placed_by_key.values():
            if placed.order_id == normalized:
                return placed.total_usd
        return None

    def refunded_so_far(self, order_id: str) -> float:
        """Sum of refunds already issued against an order (the cumulative-cap left side)."""
        normalized = order_id.strip().upper()
        return round(
            sum(r.amount_usd for r in self._refunds_by_key.values() if r.order_id == normalized),
            2,
        )

    def issue_refund(
        self,
        idempotency_key: str,
        *,
        order_id: str,
        amount_usd: float,
        destination: str,
    ) -> RefundRecord:
        """Issue a refund, deduplicated by per-INTENT `idempotency_key` (SoR-arbiter rule).

        A repeat call with a seen key returns the ORIGINAL refund unchanged (replay-safe,
        like `place`). The key is per-refund-INTENT, so a SECOND legitimate partial refund
        (different intent, different key) is NOT deduped away. Enforces the cumulative cap
        in one place — the join `sum(prior refunds) + amount <= captured_total` — so two
        partials can never over-refund an order. Refuses (RefundError) an unknown order or
        an over-cap amount; the caller must have already gated destination -> level (§A4b).
        """
        existing = self._refunds_by_key.get(idempotency_key)
        if existing is not None:
            return existing
        captured = self.captured_total(order_id)
        if captured is None:
            raise RefundError(f"unknown order {order_id!r} - cannot refund")
        already = self.refunded_so_far(order_id)
        if round(already + amount_usd, 2) > captured:
            raise RefundError(
                f"refund ${amount_usd:.2f} exceeds refundable balance on {order_id} "
                f"(captured ${captured:.2f}, already refunded ${already:.2f})"
            )
        record = RefundRecord(
            return_id=f"R-{self._next_refund_seq}",
            order_id=order_id.strip().upper(),
            amount_usd=amount_usd,
            destination=destination,
        )
        self._next_refund_seq += 1
        self._refunds_by_key[idempotency_key] = record
        return record

    @property
    def refund_count(self) -> int:
        """How many DISTINCT refunds this store has issued (test/verification surface)."""
        return len(self._refunds_by_key)
