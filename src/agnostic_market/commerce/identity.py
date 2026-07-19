"""Caller identity + order-read authorization — the P7 session authorization layer.

Two rungs, matching the guest-lookup / logged-in split (P7 decisions, 2026-07-16):

- **Rung 1 (per-order, no OTP)**: knowing an order id + the account contact on file
  (code-matched, never model-judged) grants THAT order for the session — the industry
  guest-lookup pair. A contact match binds nothing account-wide: no enumeration, no level.
- **Rung 2 (account-wide)**: a committed OTP to the on-file contact (the identity flow)
  BINDS the session to a customer and unlocks enumeration (`list_orders`) plus rung 1 for
  every owned order. Level alone is NOT identity — a profile-flow OTP earns L2 without
  binding anyone (see `PendingIdentity.grants_at_mint` for the binding invariant).

`CallerIdentityStore` follows the CartStore/LastOrderPointer pattern: one mutable object
per session, closed into tools/nodes at build, live-read, NEVER checkpointed (a replayed
checkpoint cannot resurrect a grant), cleared by the thread reaper on session close.

PII discipline (SECURITY §6): `CustomerEntry.contact` is the stub MATCHING value only — it
is never spoken, never logged, never telemetered; `masked_contact` is the only speakable
form. `match_contact` takes the caller's spoken claim and returns a verdict — the claim
VALUE is never logged or telemetered by any identity/gate path (events carry closed slugs
only). The frontline answered-turn utterance dataset (which DOES record raw text) is
contact-redacted at the `write_event` chokepoint (`redact_contact`, telemetry.py) — the
gap SECURITY §7d named is closed there. Honest residual: the claim still exists where the
caller put it — the transcript (HumanMessage) and the model's tool-call args — neither of
which is the persisted telemetry dataset.

ACCEPTED, DOCUMENTED GAP (P7 decision 5 — throttle deferred): claim matching and the
order+contact pair check are UNTHROTTLED across sessions. Within one session the identity
flow bounds claim attempts (`contact_reask_max`, then a human) and OTP attempts
(`otp_max_attempts`, then a human) — both merchant-tunable within platform ceilings — but an
attacker can redial for fresh attempts — cross-session probing of contact claims and
order/contact pairs is unbounded in the build phase. Mitigations today: fail-closed tools
(no data on decline), ONE combined not-found response (order existence is never confirmed),
and the softened re-ask (never asserts a contact is not on file). The fix is NOT a per-flow
counter here: the platform rate/abuse layer (security-hardening build) must own claim
attempts, OTP attempts, order-probe rates, and tool floods uniformly, backed by the Phase-4
durable store; Phase-4 per-customer OTP secrets additionally remove the global-stub-code
constraint that makes neutral "dispatch anyway" flows unsafe today.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from agnostic_market.commerce.orders import OrdersFixture, OrderStore
from agnostic_market.commerce.spoken import spoken_digits, spoken_email
from agnostic_market.config.loader import ConfigError, load_yaml_layer

_STRICT = ConfigDict(extra="forbid")
_FROZEN = ConfigDict(extra="forbid", frozen=True)


def _digits(value: str) -> str:
    return "".join(ch for ch in value if ch.isdigit())


def _match_key(contact: str) -> str:
    """The normalized form `match_contact` compares on — emails lowercased/joined, phones
    last-10 digits (country-code tolerance). Uniqueness must be checked on THIS key, not the
    raw value, or two differently-written contacts could still collide at match time."""
    if "@" in contact:
        return "".join(contact.lower().split())
    digits = _digits(contact)
    return digits[-10:] if len(digits) >= 10 else digits


class CustomerEntry(BaseModel):
    """One customer in the stub identity SoR. `contact` = the matching value (never spoken,
    never logged); `masked_contact` = the only speakable form."""

    model_config = _STRICT

    contact: str = Field(min_length=1)
    masked_contact: str = Field(min_length=1)


class CustomersFixture(BaseModel):
    """Validated stub customer directory content for one merchant."""

    model_config = _STRICT

    customers: dict[str, CustomerEntry] = Field(min_length=1)

    @model_validator(mode="after")
    def _contacts_unique(self) -> CustomersFixture:
        """`match_contact` is first-match over this dict — a shared contact would silently
        deny every later-listed owner, so the stub directory cannot REPRESENT one. Fail
        loudly at load (same stance as `assert_orders_have_customers`); Phase 4's real SoR
        owns genuine shared-contact semantics."""
        seen: dict[str, str] = {}
        for ref, entry in self.customers.items():
            key = _match_key(entry.contact)
            if key in seen:
                raise ValueError(
                    f"customers {seen[key]} and {ref} share a contact - the stub directory "
                    "matches first-entry-wins and cannot represent shared contacts"
                )
            seen[key] = ref
        return self


class BoundIdentity(BaseModel):
    """A session's verified customer binding. `customer_ref` is a closed fixture slug
    (CUST-001), not PII; `masked_contact` is the speakable factor reference."""

    model_config = _FROZEN

    customer_ref: str = Field(min_length=1)
    masked_contact: str = Field(min_length=1)


def load_customers_fixture(config_root: Path, merchant_id: str) -> CustomersFixture:
    """Load + validate the merchant's customers fixture. Fails loudly (build time, not
    mid-call)."""
    path = config_root / "fixtures" / "customers" / f"{merchant_id}.yaml"
    try:
        return CustomersFixture.model_validate(load_yaml_layer(path))
    except ValidationError as exc:
        raise ConfigError(f"customers fixture {path} failed validation:\n{exc}") from exc


def assert_orders_have_customers(orders: OrdersFixture, customers: CustomersFixture) -> None:
    """Every fixture order's `customer_ref` must name a customer — checked at session build
    (a typo must fail the build loudly, never silently make an order unlistable)."""
    unknown = {
        oid: entry.customer_ref
        for oid, entry in orders.orders.items()
        if entry.customer_ref not in customers.customers
    }
    if unknown:
        raise ConfigError(
            "orders fixture names customer_refs missing from the customers fixture: "
            + ", ".join(f"{oid} -> {ref}" for oid, ref in sorted(unknown.items()))
        )


class CustomerDirectory:
    """The stub identity SoR lookup: a spoken contact claim -> the customer it names.

    Stub matching semantics (build phase; Phase 4 replaces this with the real SoR lookup):
    the claim is SPOKEN-FORM normalized first (live call #12 F-12.1 — STT delivers emails
    as "k c at example dot com" and phone numbers as digit words; a literal compare failed
    every voice caller). An email claim (spells an '@') matches on lowercased,
    whitespace-stripped equality; a phone claim matches on digits-only equality OR
    last-10-digits equality (country-code tolerance) — never across types. The leniency is
    a ROUTING convenience only: the OTP to the on-file factor is what proves identity, not
    this match. Never logs the claim.

    The `fixture` is REQUIRED — customer data lives in `config/fixtures/customers/*.yaml`, never
    hardcoded (load via `load_customers_fixture`); tests use the same loader, no baked-in default.
    """

    def __init__(self, fixture: CustomersFixture) -> None:
        self.fixture = fixture

    def masked_contact(self, customer_ref: str) -> str | None:
        entry = self.fixture.customers.get(customer_ref)
        return entry.masked_contact if entry else None

    def match_contact(self, claim: str) -> BoundIdentity | None:
        cleaned = claim.strip()
        if not cleaned:
            return None
        email = spoken_email(cleaned)
        if email is not None:
            for ref, entry in self.fixture.customers.items():
                if "@" in entry.contact and _match_key(entry.contact) == email:
                    return BoundIdentity(customer_ref=ref, masked_contact=entry.masked_contact)
            return None
        claimed = spoken_digits(cleaned)
        if not claimed:
            return None
        key = _match_key(claimed)
        for ref, entry in self.fixture.customers.items():
            if "@" in entry.contact:
                continue  # never match a phone claim against an email contact
            if _match_key(entry.contact) == key:
                return BoundIdentity(customer_ref=ref, masked_contact=entry.masked_contact)
        return None


class CallerIdentityStore:
    """The session's authorization state — rung-1 per-order grants + the rung-2 binding.

    Single source of truth for "may this session read that order / list an account": written
    only by the order_status tool (grants) and the identity flow's apply node (bind), read
    live everywhere. `clear()` is the Clock-B reap hook — both rungs drop together.
    """

    def __init__(self) -> None:
        self._bound: BoundIdentity | None = None
        self._granted_orders: set[str] = set()
        # TEST-ONLY rung-2 grants (see `grant_mutation_for_test`). Empty in production — no
        # production writer exists, so `order_mutation_allowed`'s check of it is inert at runtime.
        self._mutation_granted_for_test: set[str] = set()

    def bind(self, identity: BoundIdentity) -> None:
        self._bound = identity

    def current(self) -> BoundIdentity | None:
        return self._bound

    def grant_order(self, order_id: str) -> None:
        self._granted_orders.add(order_id.strip().upper())

    def order_granted(self, order_id: str) -> bool:
        return order_id.strip().upper() in self._granted_orders

    def grant_mutation_for_test(self, order_id: str) -> None:
        """TEST SEAM ONLY — pretend this session is rung-2-authorized to MUTATE that order,
        without a real OTP bind. Exists so the money-logic suites can pin post-authorization
        cancel/refund/return math without re-driving the identity flow (and without binding,
        which can't span the two customers the fixture orders belong to). NEVER call this from
        production code: a real mutation authority comes only from a session-placed order or an
        OTP-bound identity (`order_mutation_allowed`). Enforced by grep in CI review — the only
        callers are `tests/support_helpers.py:authorize_fixture_orders` and scoping-suite pins."""
        self._mutation_granted_for_test.add(order_id.strip().upper())

    def mutation_granted_for_test(self, order_id: str) -> bool:
        return order_id.strip().upper() in self._mutation_granted_for_test

    def clear(self) -> None:
        self._bound = None
        self._granted_orders.clear()
        self._mutation_granted_for_test.clear()


def order_read_allowed(order_id: str, *, store: OrderStore, identity: CallerIdentityStore) -> bool:
    """May this session read that order? The ONE order-read authorization check — the
    order_status tool, the L3 render-divert router, and the render node all call THIS (the
    shared-predicate stance: two independent computations would drift, and a drifted render
    path would leak around the tool's gate).

    True when the order was PLACED this session (the caller placed it moments ago), the
    session already holds a rung-1 grant for it, or the bound identity owns it. PURE read —
    the contact-match GRANT mutation stays in the tool body (explicit, single writer).
    """
    if store.is_session_placed(order_id):
        return True
    if identity.order_granted(order_id):
        return True
    bound = identity.current()
    if bound is not None:
        owner = store.order_owner(order_id)
        return owner is not None and owner == bound.customer_ref
    return False


def order_mutation_allowed(
    order_id: str, *, store: OrderStore, identity: CallerIdentityStore
) -> bool:
    """May this session ACT irreversibly on that order (cancel/refund/return)? The rung-2
    authorization check — deliberately STRICTER than `order_read_allowed`: a rung-1
    contact-match grant (`identity.order_granted`, earned via `try_grant_by_contact`)
    authorizes a READ but NOT a mutation. The account contact is guessable/leaked, so
    contact-matching one order must never let one caller cancel another's; a mutation requires
    that the caller either placed the order THIS session (per-session store — no cross-caller
    path) or is OTP-BOUND to the customer that owns it (SECURITY §7d). An unbound caller who
    fails this is routed into the identity OTP flow, not granted (support/flow.py)."""
    if store.is_session_placed(order_id):
        return True
    # Test seam (inert in production — no prod writer, see `grant_mutation_for_test`): lets the
    # money-logic suites pin post-authorization math without an OTP bind that can't span the
    # fixture's two customers.
    if identity.mutation_granted_for_test(order_id):
        return True
    bound = identity.current()
    if bound is None:
        return False
    owner = store.order_owner(order_id)
    return owner is not None and owner == bound.customer_ref


def try_grant_by_contact(
    order_id: str,
    claim: str,
    *,
    store: OrderStore,
    customers: CustomerDirectory,
    identity: CallerIdentityStore,
) -> Literal["granted", "mismatch"]:
    """The ONE rung-1 grant decision: does this contact CLAIM own that ORDER? — shared by the
    order_status tool and the support-selection gate (they previously each ran this same
    match->owner->compare->grant sequence, a drift risk between two auth surfaces).

    Grants THAT order for the session on a match (the code-matched guest-lookup pair — the
    claim is never model-judged), and returns a CLOSED verdict; the caller owns its own
    telemetry, response wording, and retry counters (they differ by surface). `mismatch`
    covers wrong-pair, unknown claim, AND an unresolved `order_id` (owner is None) uniformly —
    the existence-oracle discipline lives in the callers' single combined not-found line.

    Callers MUST have already handled their own pre-checks (an existing authorization, an
    empty claim): this function assumes a non-empty claim and performs no grant on mismatch.
    """
    matched = customers.match_contact(claim)
    owner = store.order_owner(order_id)
    if matched is None or owner is None or owner != matched.customer_ref:
        return "mismatch"
    identity.grant_order(order_id)
    return "granted"
