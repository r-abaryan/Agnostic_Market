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

from collections.abc import Iterator, Sequence
from datetime import date, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from agnostic_market.config.loader import ConfigError, load_yaml_layer
from agnostic_market.dtos.state import BatchCancelOutcome, CartLine

_STRICT = ConfigDict(extra="forbid")
_FROZEN = ConfigDict(extra="forbid", frozen=True)

# Stub-store copy for orders this store itself placed (real ETA logic = the Phase-4 SoR).
_PLACED_STATUS = "processing"
_PLACED_ETA = "3-5 business days"


def speak_quantity(quantity: int, name: str) -> str:
    """'one waterproof rain jacket' / 'two waterproof rain jackets' — not '1 x name'
    (TTS reads the 'x' separator literally as 'ex'). Lives in commerce (next to the order
    renderers that call it) so the plane arrow stays agents/voice -> commerce; the cart
    readback, the view_cart tool, and the order summaries all import it from here.

    Pluralization is naive-but-safe: a name that already reads plural (ends in 's', like the
    fixture's "trail running shoes") is NOT double-pluralized ("shoess"). A real catalog
    would carry singular/plural forms; this is the stub-fixture heuristic."""
    plural = name if quantity == 1 or name.endswith("s") else f"{name}s"
    return f"{quantity} {plural}"


def speak_lines(lines: Sequence[CartLine] | Sequence[PlacedLine]) -> str:
    """Speech-native rendering of N line items: 'a waterproof rain jacket', or '2 rain
    jackets and 1 pair of socks' — never '2 x jacket' (TTS 'ex'). One code-authored surface
    for every multi-line readback (placement confirm, the cancel readback, order summaries),
    so a cart and the order it becomes always speak the same way."""
    parts = [speak_quantity(line.quantity, line.name) for line in lines]
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return f"{', '.join(parts[:-1])}, and {parts[-1]}"


# Bare fulfillment status -> a natural spoken phrase. KNOWN statuses only: an unmapped status
# (a new fixture/SoR value) must NOT be given an invented meaning — the renderer speaks the raw
# word instead (fail-closed). This map is the ONLY place a status becomes a human phrase, so a
# code renderer structurally cannot say "on the way" for a `processing` order (the live-call
# embellishment class the deterministic renderer exists to prevent).
_STATUS_PHRASE: dict[str, str] = {
    "shipped": "on its way",
    "processing": "being prepared",
    "delivered": "delivered",
    "cancelled": "cancelled",
}


def _eta_clause(eta: str | None, today: date) -> str:
    """The spoken ETA fragment, from the mixed-format `eta` (a date "2026-07-09" OR a duration
    "3-5 business days" OR None/garbage). Returns "" when there is nothing safe to say.

    A date BEFORE `today` is framed as PAST and TERMINAL — never "arriving", and never a
    promise to "check the latest status" (that dangling promise is live-call F-11.1: the model
    said it would check, then ended the turn without checking; a code renderer that says the
    same and ENDs re-introduces it). The renderer states what the store holds and promises
    nothing it won't do; a fresh check is the caller's to ask for.
    """
    if not eta:
        return ""
    try:
        eta_date = datetime.strptime(eta, "%Y-%m-%d").date()
    except ValueError:
        # A duration ("3-5 business days") is safe to speak as-is with a lead-in; any other
        # non-date free-text is omitted rather than read raw (no ISO garbage to TTS).
        return f", expected in {eta.replace('-', ' to ')}" if _looks_like_duration(eta) else ""
    spoken = eta_date.strftime("%A %d %B")
    if eta_date < today:
        return f" - that was expected by {spoken}"
    if eta_date == today:
        return " - expected to arrive today"
    return f", expected to arrive by {spoken}"


def _looks_like_duration(eta: str) -> str | bool:
    """A duration free-text like "3-5 business days" (has a digit + a time-unit word), vs an
    unparseable string we should drop. Deliberately narrow — the stub's only duration is
    `_PLACED_ETA`; a real catalog would carry a typed ETA."""
    lowered = eta.lower()
    return any(ch.isdigit() for ch in eta) and any(
        unit in lowered for unit in ("day", "week", "hour", "month")
    )


def _is_past_date_eta(eta: str | None, today: date) -> bool:
    if not eta:
        return False
    try:
        return datetime.strptime(eta, "%Y-%m-%d").date() < today
    except ValueError:
        return False


def render_order_status_line(
    *, order_id: str, status: str, items: str, eta: str | None, today: date
) -> str:
    """CODE-authored spoken order-status line (the deterministic read renderer). Pure — every
    input is passed (`today` too, so it is trivially testable and never a stale build-time
    date). Reuses the already-composed `items` string (no re-pluralization).

    Humanizes KNOWN statuses via `_STATUS_PHRASE`; an unknown status speaks the raw word
    (fail-closed, no invented meaning). The ETA is framed relative to `today` (`_eta_clause`).
    """
    order_id = order_id.strip().upper()
    phrase = _STATUS_PHRASE.get(status)
    if phrase is None:
        # Fail-closed: don't guess a phrase for a status we don't model.
        return f"Your order {order_id} - {items} - is {status}."
    # A shipped order whose ETA has PASSED must not say "on its way, that was expected by X" —
    # "on its way" (still coming) contradicts an overdue date. Speak the past-expectation as the
    # WHOLE status instead (terminal, no dangling "let me check" — F-11.1).
    if status == "shipped" and _is_past_date_eta(eta, today):
        spoken = datetime.strptime(eta, "%Y-%m-%d").date().strftime("%A %d %B")
        return f"Your order {order_id} - {items} - was expected by {spoken}."
    # An ETA is only meaningful for an IN-FLIGHT order — a delivered order already arrived and a
    # cancelled one won't; speaking "expected to arrive by X" for either is wrong.
    eta_clause = _eta_clause(eta, today) if status in ("shipped", "processing") else ""
    return f"Your order {order_id} - {items} - is {phrase}{eta_clause}."


def render_cart_line(lines: Sequence[CartLine], total_usd: float) -> str:
    """CODE-authored spoken cart line (the deterministic view_cart renderer). Empty cart is the
    caller's job to detect (the node speaks the empty line); this renders a non-empty cart."""
    return f"You've got {speak_lines(lines)} in your cart, ${total_usd:.2f} in total."


def render_order_list_line(candidates: Sequence[OrderCandidate]) -> str:
    """CODE-authored spoken order list (P7 enumeration — the identity flow's apply node and
    the verified `list_orders` tool both speak from THIS). Id + summary + humanized status
    ONLY — no addresses, no contact, no totals (not needed to answer "what orders do I
    have", and short lines TTS better). Statuses humanize via `_STATUS_PHRASE`, fail-closed
    (an unknown status speaks the raw word, never an invented phrase)."""
    if not candidates:
        return "I don't see any orders on your account."
    parts = [
        f"{c.order_id}, {c.summary}, {_STATUS_PHRASE.get(c.status, c.status)}"
        for c in candidates
    ]
    if len(parts) == 1:
        return f"You've got 1 order: {parts[0]}."
    listed = "; ".join(parts[:-1]) + f"; and {parts[-1]}"
    return f"You've got {len(parts)} orders: {listed}."


def _cancel_outcome_clause(o: BatchCancelOutcome) -> str:
    """One target's truthful spoken clause, rendered from its STRUCTURED outcome (INV-25 —
    never from pending intent). A `cancelled` outcome states the reversed amount; the honest
    declines name the reason without inventing one."""
    if o.outcome == "cancelled":
        assert o.amount_usd is not None  # BatchCancelOutcome enforces the discriminant
        amount = f"${o.amount_usd:.2f}"
        return (
            f"your order for {o.summary} ({o.order_id}) is cancelled and {amount} goes back "
            "to your original payment method"
        )
    if o.outcome == "already_cancelled":
        return f"your order for {o.summary} ({o.order_id}) was already cancelled"
    if o.outcome == "not_cancellable":
        return (
            f"your order for {o.summary} ({o.order_id}) has already shipped, so that one "
            "needs a return instead"
        )
    if o.outcome == "has_refunds":
        return (
            f"your order for {o.summary} ({o.order_id}) already has a refund, so our support "
            "team needs to sort out the rest"
        )
    # store_refused — an effect-time refusal with no typed reason; honest, promises nothing.
    return f"I couldn't cancel your order for {o.summary} ({o.order_id}) - nothing changed on it"


def render_batch_cancel_outcome(outcomes: Sequence[BatchCancelOutcome]) -> str:
    """CODE-authored spoken result of a cancel batch — the ONLY speakable, composed from the
    per-target OUTCOMES (INV-25). A batch of ONE cancelled order reads byte-identical to the
    single-cancel void line (single = batch-of-one, no speech regression). Multiple targets
    join speech-natively; mixed success/decline states each truthfully."""
    if not outcomes:
        # Never reached in the flow (a batch always has >=1 target), but fail honest.
        return "Nothing to cancel - nothing has changed."
    if len(outcomes) == 1:
        only = outcomes[0]
        if only.outcome == "cancelled":
            # The exact single-cancel void phrasing (kept identical on purpose).
            assert only.amount_usd is not None  # BatchCancelOutcome enforces the discriminant
            amount = f"${only.amount_usd:.2f}"
            return (
                f"Done - I've cancelled your order for {only.summary}. The {amount} charge "
                "goes back to your original payment method."
            )
        # A one-target decline: the clause carried by a leading capital, no "Done".
        clause = _cancel_outcome_clause(only)
        return clause[0].upper() + clause[1:] + "."
    clauses = [_cancel_outcome_clause(o) for o in outcomes]
    joined = "; ".join(clauses[:-1]) + f"; and {clauses[-1]}"
    return joined[0].upper() + joined[1:] + "."


class _OrderEntry(BaseModel):
    model_config = _STRICT

    status: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    eta: str = Field(min_length=1)
    # The owning customer (a closed slug into the customers fixture, e.g. "CUST-001") — the
    # P7 order-read/enumeration authorization basis. REQUIRED (no default) on purpose: an
    # order without an owner would be silently unreadable under object binding; the fixture
    # must fail loudly at load. Cross-checked against the customers fixture at session build
    # (identity.assert_orders_have_customers).
    customer_ref: str = Field(min_length=1)
    # Captured amount — the refund cumulative-cap join reads this (§A4b); a real SoR always
    # knows what it captured. Required so a refund can never run against an unknown total.
    total_usd: float = Field(ge=0)
    # Timezone-aware ISO 8601 delivery instant (e.g. "2026-07-01T00:00:00Z") — the return
    # window counts from here (Group C). Required for DELIVERED orders (fail-loud at fixture
    # load); shipped-not-yet-delivered orders have none and are trivially in window.
    delivered_at: str | None = None

    @model_validator(mode="after")
    def _delivered_requires_timestamp(self) -> _OrderEntry:
        if self.status == "delivered" and self.delivered_at is None:
            raise ValueError("a 'delivered' order requires delivered_at (return-window basis)")
        return self


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

    Same SKU-discipline stance as `Candidate`: the model never AUTHORS a raw order id; it
    picks a KEY into this bounded list and code resolves key -> order_id + captured total.
    (It MAY relay a caller-STATED order number for the guest path — code resolves that
    fail-closed, exactly as the `order_status` tool already accepts one.) `status` is the
    EFFECTIVE status (cancelled overlay wins) — the model needs it to pick the right
    remedy (money back on an unshipped order is a cancel, not a refund).
    """

    model_config = _FROZEN

    key: str = Field(min_length=1)
    order_id: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    total_usd: float = Field(ge=0)
    status: str = Field(min_length=1)


class PlacedLine(BaseModel):
    """One line of a placed multi-line order (Group B)."""

    model_config = _FROZEN

    sku: str = Field(min_length=1)
    name: str = Field(min_length=1)
    price_usd: float = Field(ge=0)
    quantity: int = Field(ge=1)


class PlacedOrder(BaseModel):
    """An order this store placed (the stub SoR's own record). Group B: an order is
    MULTI-LINE (the whole cart placed as one order, one id, one total). No single-line
    `sku`/`name`/`quantity` shim — every consumer renders ALL `lines` (a `lines[0]` shim
    would let the cancel readback name only the first line = consent over a mis-described
    order)."""

    model_config = _FROZEN

    order_id: str = Field(min_length=1)
    total_usd: float = Field(ge=0)
    lines: tuple[PlacedLine, ...] = Field(min_length=1)


class RefundRecord(BaseModel):
    """A refund this store issued (the stub SoR's own record). `refund_id` is the refund
    REFERENCE ("R-7001..") — distinct from a return/RMA id (Group C's `ReturnRecord`)."""

    model_config = _FROZEN

    refund_id: str = Field(min_length=1)
    order_id: str = Field(min_length=1)
    amount_usd: float = Field(ge=0)
    destination: str = Field(min_length=1)


class RefundError(ValueError):
    """A refund the store refused: unknown order, or over the order's refundable balance."""


class CancelRecord(BaseModel):
    """A cancellation this store performed (the stub SoR's own record). `total_usd` is the
    captured amount the void reverses — the spoken outcome states the money movement."""

    model_config = _FROZEN

    order_id: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    total_usd: float = Field(ge=0)


class CancelError(ValueError):
    """A cancel the store refused: unknown order, or not in a cancellable (processing) state."""


class ReturnRecord(BaseModel):
    """A return/RMA this store created (Group C). Stub returns stay OPEN until the Phase-4
    real SoR (nothing "receives" a package in the build phase); `refund_due_usd` is the
    refund RECORDED on the return, released when the return is processed — that release is
    the money-movement moment and re-runs the destination->level check there (§A4c).
    `destination` is always "original" in v1 (a destination change at release time is its
    own L2 interaction)."""

    model_config = _FROZEN

    rma_id: str = Field(min_length=1)
    order_id: str = Field(min_length=1)
    refund_due_usd: float = Field(ge=0)
    destination: str = Field(min_length=1)


class ReturnError(ValueError):
    """A return the store refused: unknown/ineligible order, an open return already exists,
    or the recorded refund would exceed the order's refundable balance."""


# Fulfillment states an order can be cancelled from (§A4b / industry: only the pre-shipment
# window; once shipped, direct cancellation ends and becomes a return). Placed orders start
# in _PLACED_STATUS, which is included here so a just-placed order is cancellable.
_CANCELLABLE_STATUSES = frozenset({"processing"})
# Public: the support flow branches its caller-facing declines on these effective statuses.
CANCELLED_STATUS = "cancelled"
# Goods are (or were) on their way: a refund on these is RETURN-FIRST above the merchant's
# returnless threshold — without it the caller keeps both the goods and the money.
FULFILLED_STATUSES = frozenset({"shipped", "delivered"})


class LastOrderPointer:
    """The session's "that order" reference — a bounded conversational pointer (Group C, L4
    from live call #10: "one thousand one" was mis-bound to the salient ORD-9001).

    The CartStore pattern: one mutable per-session object, closed into tools + flows at
    build, dies with the session (never a checkpointed state channel — nothing for a replay
    to resurrect, and no Command/tool-injection machinery). Set ONLY on an explicit
    order_status lookup or a successful order effect (place/cancel/return). It is a bare
    ID for REFERENCE RESOLUTION — never a cache of order state: any status/delivery/refund
    claim still requires an order_status read that turn (the frontline grounding rule).
    """

    def __init__(self) -> None:
        self._order_id: str | None = None

    def set(self, order_id: str) -> None:
        self._order_id = order_id.strip().upper()

    def get(self) -> str | None:
        return self._order_id

    def clear(self) -> None:
        self._order_id = None


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
        self._next_refund_seq = 7001  # refund references R-7001..
        self._cancels_by_key: dict[str, CancelRecord] = {}
        self._cancelled_ids: set[str] = set()  # order_ids voided (status overlay)
        self._returns_by_key: dict[str, ReturnRecord] = {}
        self._next_return_seq = 3001  # return ids RMA-3001.. (disjoint from ORD-/R- spaces)

    def order_status(self, order_id: str) -> str | None:
        """The effective fulfillment status of an order; None if unknown.

        The cancel guardrail reads this LIVE to decide eligibility — a voided order reads
        `cancelled` (the overlay wins), else the fixture/placed status. Kept separate from
        `order_summary` (the spoken line) so the guardrail branches on a bare status string.
        """
        normalized = order_id.strip().upper()
        if normalized in self._cancelled_ids:
            return CANCELLED_STATUS
        entry = self.fixture.orders.get(normalized)
        if entry is not None:
            return entry.status
        for placed in self._placed_by_key.values():
            if placed.order_id == normalized:
                return _PLACED_STATUS
        return None

    def order_summary(self, order_id: str) -> str | None:
        """Human-readable status line for fixture AND just-placed orders; None if unknown.

        A cancelled order reads back `cancelled` (the overlay wins) so a caller asking after
        a cancel hears the truth."""
        normalized = order_id.strip().upper()
        cancelled = normalized in self._cancelled_ids
        entry = self.fixture.orders.get(normalized)
        if entry is not None:
            status = CANCELLED_STATUS if cancelled else entry.status
            return f"Order {normalized}: {entry.summary} - status {status}, ETA {entry.eta}."
        for placed in self._placed_by_key.values():
            if placed.order_id == normalized:
                status = CANCELLED_STATUS if cancelled else _PLACED_STATUS
                return (
                    f"Order {normalized}: {speak_lines(placed.lines)} "
                    f"(${placed.total_usd:.2f}) - status {status}, ETA {_PLACED_ETA}."
                )
        return None

    def order_owner(self, order_id: str) -> str | None:
        """The owning customer_ref of a FIXTURE order; None for unknown AND for session-placed
        orders (those are owned by "this session's caller" — `is_session_placed` is that
        check; a placed order joins a customer's durable account at the Phase-4 real SoR)."""
        entry = self.fixture.orders.get(order_id.strip().upper())
        return entry.customer_ref if entry is not None else None

    def is_session_placed(self, order_id: str) -> bool:
        """True when THIS session's store placed the order (the caller placed it on this
        call — readable by them without any further verification, P7 rung 1)."""
        normalized = order_id.strip().upper()
        return any(p.order_id == normalized for p in self._placed_by_key.values())

    def _iter_owned_orders(self, customer_ref: str) -> Iterator[OrderCandidate]:
        """Yield the account/session order view without materializing full history."""
        index = 1
        for order_id, entry in self.fixture.orders.items():
            if entry.customer_ref != customer_ref:
                continue
            yield OrderCandidate(
                key=str(index),
                order_id=order_id,
                summary=entry.summary,
                total_usd=entry.total_usd,
                status=self.order_status(order_id) or "unknown",
            )
            index += 1
        for record in self._placed_by_key.values():
            yield OrderCandidate(
                key=str(index),
                order_id=record.order_id,
                summary=speak_lines(record.lines),
                total_usd=record.total_usd,
                status=self.order_status(record.order_id) or "unknown",
            )
            index += 1

    def owned_orders(self, customer_ref: str) -> list[OrderCandidate]:
        """The bounded, keyed list of orders the IDENTIFIED caller may hear enumerated (P7
        rung 2): fixture orders owned by `customer_ref` + everything placed THIS session
        (placed-by-this-caller by construction). Effective status (cancelled overlay wins).
        Same OrderCandidate shape as `actionable_orders` — whose MODEL-VISIBLE subset the
        support assemble scopes through `order_read_allowed` (SECURITY §7d, call #15)."""
        return list(self._iter_owned_orders(customer_ref))

    def owned_cancellable_orders(
        self, customer_ref: str, *, limit: int
    ) -> tuple[list[OrderCandidate], bool]:
        """The bounded set of the caller's orders that are CURRENTLY cancellable (F-16.2
        Milestone B) — the resolver's target universe for a "cancel all my orders" scope.
        Returns at most `limit` candidates plus a `has_more` overflow flag (queried with
        `limit = cap + 1` so overflow is detected WITHOUT loading full history or silently
        truncating). Bounded by construction so a real account's history can't blow the batch.
        Only `is_cancellable` orders — a scope names nothing about shipped/delivered history."""
        if limit < 1:
            raise ValueError("limit must be at least 1")
        cancellable: list[OrderCandidate] = []
        for candidate in self._iter_owned_orders(customer_ref):
            if not self.is_cancellable(candidate.order_id):
                continue
            if len(cancellable) == limit:
                return cancellable, True
            cancellable.append(candidate)
        return cancellable, False

    def order_eta(self, order_id: str) -> str | None:
        """The raw ETA for an order (fixture `entry.eta`, a date "2026-07-09"; or `_PLACED_ETA`
        for a placed order, a duration "3-5 business days"); None if unknown. The public accessor
        the deterministic read renderer reads — mirrors the fixture/placed fork in
        `order_summary`, so the render node never touches the private `fixture.orders` dict."""
        normalized = order_id.strip().upper()
        entry = self.fixture.orders.get(normalized)
        if entry is not None:
            return entry.eta
        for placed in self._placed_by_key.values():
            if placed.order_id == normalized:
                return _PLACED_ETA
        return None

    def place_cart(
        self, idempotency_key: str, *, lines: Sequence[CartLine], total_usd: float
    ) -> PlacedOrder:
        """Place a WHOLE cart as ONE multi-line order, deduplicated by `idempotency_key`
        (SoR-arbiter rule — Group B).

        A repeat call with a seen key returns the ORIGINAL placed order unchanged — the
        replay/retry path can never create a second order or drift the recorded values. One
        order id, one total, one idempotency scope for the whole cart (a caller reasons
        about ONE order). `lines`/`total_usd` are code-computed by the flow (never model
        arithmetic).
        """
        existing = self._placed_by_key.get(idempotency_key)
        if existing is not None:
            return existing
        placed = PlacedOrder(
            order_id=f"ORD-{self._next_seq}",
            total_usd=total_usd,
            lines=tuple(
                PlacedLine(sku=ln.sku, name=ln.name, price_usd=ln.price_usd, quantity=ln.quantity)
                for ln in lines
            ),
        )
        self._next_seq += 1
        self._placed_by_key[idempotency_key] = placed
        return placed

    @property
    def placed_count(self) -> int:
        """How many DISTINCT orders this store has placed (test/verification surface)."""
        return len(self._placed_by_key)

    def identical_cart_order(self, lines: Sequence[CartLine]) -> PlacedOrder | None:
        """A LIVE (non-cancelled) order this session already placed with the SAME line set
        (sku→quantity), regardless of order (Group B).

        The placement guardrail reads this to disambiguate a probable repeat: a caller
        saying "complete the purchase" after the cart already placed must hear "this would
        be a SECOND order", not a readback identical to the first (live 2026-07-10: that
        path silently created a duplicate $387 order). A cancelled match doesn't count —
        re-ordering after a cancel is a normal intent.
        """
        want = {ln.sku: ln.quantity for ln in lines}
        for placed in self._placed_by_key.values():
            if placed.order_id in self._cancelled_ids:
                continue
            have = {ln.sku: ln.quantity for ln in placed.lines}
            if have == want:
                return placed
        return None

    def actionable_orders(self) -> list[OrderCandidate]:
        """The bounded, keyed list of orders a support action (refund OR cancel) may target
        (fixture + placed), each with its EFFECTIVE status (cancelled overlay wins).

        Code-side narrowing for support selection — the model picks a KEY into this, never a
        raw order id (SKU-discipline analogue). Keys are 1-based positions as strings over
        the FULL list, so a key is stable while the assemble node filters the MODEL-VISIBLE
        subset to authorized orders (SECURITY §7d, call #15 — the full list feeds only the
        code-side id resolution for the guest path, never the prompt). Cancelled orders stay
        listed (the caller may ask about them); the status lets the model — and the
        guardrails — answer honestly instead of proposing a dead action.
        """
        candidates: list[tuple[str, str, float]] = [
            (oid, entry.summary, entry.total_usd) for oid, entry in self.fixture.orders.items()
        ]
        candidates += [
            (p.order_id, speak_lines(p.lines), p.total_usd)
            for p in self._placed_by_key.values()
        ]
        return [
            OrderCandidate(
                key=str(i),
                order_id=oid,
                summary=summary,
                total_usd=total,
                status=self.order_status(oid) or "unknown",
            )
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

    def delivered_at_epoch(self, order_id: str) -> float | None:
        """The delivery instant as a UTC epoch; None if not (yet) delivered.

        The return-window basis (Group C): shipped-not-yet-delivered and just-placed orders
        return None — their window hasn't started, so they are trivially in window. Parsed
        as an AWARE datetime (the fixture stores tz-aware ISO 8601), compared as instants —
        no naive-date boundary ambiguity on the window's last day.
        """
        entry = self.fixture.orders.get(order_id.strip().upper())
        if entry is None or entry.delivered_at is None:
            return None
        return datetime.fromisoformat(entry.delivered_at).timestamp()

    def return_for_order(self, order_id: str) -> ReturnRecord | None:
        """The open return for an order, if one exists — THE open-return lookup (guards need
        the record itself to speak its RMA id; a separate bool would be a second way)."""
        normalized = order_id.strip().upper()
        for record in self._returns_by_key.values():
            if record.order_id == normalized:
                return record
        return None

    def return_refund_due(self, order_id: str) -> float:
        """Sum of refunds PROMISED on open returns for an order — the promise side of the
        cumulative-cap join (refunds paid + refunds promised may never exceed captured)."""
        normalized = order_id.strip().upper()
        return round(
            sum(
                r.refund_due_usd
                for r in self._returns_by_key.values()
                if r.order_id == normalized
            ),
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
        in one place — the join `refunds paid + refunds PROMISED on open returns + amount
        <= captured_total` (Group C: a return records a refund promise released at Phase 4;
        paying a refund alongside it would double-promise the same dollars) — so partials
        and returns can never over-refund an order. Refuses (RefundError) an unknown order,
        a CANCELLED order (the void already reversed the charge — refunding on top would
        return the money twice), or an over-cap amount; the caller must have already gated
        destination -> level (§A4b).
        """
        existing = self._refunds_by_key.get(idempotency_key)
        if existing is not None:
            return existing
        captured = self.captured_total(order_id)
        if captured is None:
            raise RefundError(f"unknown order {order_id!r} - cannot refund")
        if self.order_status(order_id) == CANCELLED_STATUS:
            raise RefundError(
                f"order {order_id.strip().upper()} is cancelled - the charge is already "
                "reversed, nothing left to refund"
            )
        already = self.refunded_so_far(order_id)
        promised = self.return_refund_due(order_id)
        if round(already + promised + amount_usd, 2) > captured:
            raise RefundError(
                f"refund ${amount_usd:.2f} exceeds refundable balance on {order_id} "
                f"(captured ${captured:.2f}, already refunded ${already:.2f}, "
                f"promised on open returns ${promised:.2f})"
            )
        record = RefundRecord(
            refund_id=f"R-{self._next_refund_seq}",
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

    def create_return(
        self,
        idempotency_key: str,
        *,
        order_id: str,
        refund_due_usd: float,
        destination: str,
    ) -> ReturnRecord:
        """Create a return/RMA, deduplicated by per-INTENT `idempotency_key` (SoR-arbiter
        rule, same shape as `issue_refund`).

        Re-validates at effect time (§A4c): the order must exist and be FULFILLED
        (shipped/delivered) — which covers cancelled AND processing, and is also the
        structural invariant that makes cancel-vs-open-return conflicts impossible (open
        returns exist only on fulfilled orders; `cancel_order` refuses non-processing) —
        must have no open return already, and the recorded refund promise must fit the
        refundable balance (paid + promised <= captured, the same join `issue_refund`
        enforces from the other side). The refund is RECORDED, not paid: release happens
        at the Phase-4 SoR once the return is processed, re-running the destination->level
        check there.
        """
        existing = self._returns_by_key.get(idempotency_key)
        if existing is not None:
            return existing
        normalized = order_id.strip().upper()
        status = self.order_status(normalized)
        if status is None:
            raise ReturnError(f"unknown order {order_id!r} - cannot create a return")
        if status not in FULFILLED_STATUSES:
            raise ReturnError(
                f"order {normalized} is {status} - only a shipped or delivered order "
                "can be returned"
            )
        open_return = self.return_for_order(normalized)
        if open_return is not None:
            raise ReturnError(
                f"a return for {normalized} is already open ({open_return.rma_id})"
            )
        captured = self.captured_total(normalized)
        assert captured is not None  # status resolved above, so the order is known
        already = self.refunded_so_far(normalized)
        # No `return_refund_due` term here: the already-open check above guarantees zero
        # open returns for this order, so the promised side is structurally 0 (unlike
        # issue_refund, which CAN run alongside an open return and must count it).
        if round(already + refund_due_usd, 2) > captured:
            raise ReturnError(
                f"return refund ${refund_due_usd:.2f} exceeds refundable balance on "
                f"{normalized} (captured ${captured:.2f}, already refunded ${already:.2f})"
            )
        record = ReturnRecord(
            rma_id=f"RMA-{self._next_return_seq}",
            order_id=normalized,
            refund_due_usd=refund_due_usd,
            destination=destination,
        )
        self._next_return_seq += 1
        self._returns_by_key[idempotency_key] = record
        return record

    @property
    def return_count(self) -> int:
        """How many DISTINCT returns this store has created (test/verification surface)."""
        return len(self._returns_by_key)

    def is_cancellable(self, order_id: str) -> bool:
        """Whether an order is currently in a cancellable (pre-shipment) state — the cancel
        guardrail reads this LIVE to decide eligibility. Unknown/shipped/delivered/cancelled
        all return False."""
        return self.order_status(order_id) in _CANCELLABLE_STATUSES

    def cancel_order(self, idempotency_key: str, *, order_id: str) -> CancelRecord:
        """Void an order, deduplicated by per-INTENT `idempotency_key` (SoR-arbiter rule).

        A repeat call with a seen key returns the ORIGINAL cancel unchanged (replay-safe,
        like `place`/`issue_refund`). Refuses (CancelError) an unknown order, one not in a
        cancellable state, or one that already has refunds issued against it (a void
        reverses the FULL charge — on top of a prior partial refund that returns money
        twice; the mixed case belongs to a person). The caller (guardrail) has already
        checked all three, but the store RE-VALIDATES (§A4c server-side re-validation) so a
        stale proposal can't void a now-shipped or part-refunded order. Flips the order's
        effective status to `cancelled` (the overlay `order_status`/`order_summary` read
        back).
        """
        existing = self._cancels_by_key.get(idempotency_key)
        if existing is not None:
            return existing
        normalized = order_id.strip().upper()
        status = self.order_status(normalized)
        if status is None:
            raise CancelError(f"unknown order {order_id!r} - cannot cancel")
        if status not in _CANCELLABLE_STATUSES:
            raise CancelError(
                f"order {normalized} is {status!r}, not cancellable "
                f"(only {sorted(_CANCELLABLE_STATUSES)})"
            )
        if self.refunded_so_far(normalized) > 0:
            raise CancelError(
                f"order {normalized} already has refunds issued - a void on top would "
                "return funds twice"
            )
        captured = self.captured_total(normalized)
        assert captured is not None  # status was known, so the order exists with a total
        summary = self.order_item_summary(normalized)
        self._cancelled_ids.add(normalized)
        record = CancelRecord(order_id=normalized, summary=summary, total_usd=captured)
        self._cancels_by_key[idempotency_key] = record
        return record

    def order_item_summary(self, order_id: str) -> str:
        """The short item summary for an order (fixture or placed) — for spoken readbacks."""
        normalized = order_id.strip().upper()
        entry = self.fixture.orders.get(normalized)
        if entry is not None:
            return entry.summary
        for placed in self._placed_by_key.values():
            if placed.order_id == normalized:
                return speak_lines(placed.lines)
        return normalized

    @property
    def cancel_count(self) -> int:
        """How many DISTINCT orders this store has cancelled (test/verification surface)."""
        return len(self._cancels_by_key)
