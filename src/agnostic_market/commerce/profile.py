"""Profile SoR access — the fixture-backed stub store for account data (Group C; real SoR
integration Phase 4).

`ProfileStore` follows the OrderStore shape: fixture reads + idempotent per-INTENT updates
(the store is the dedup arbiter, AGENTS §A10 rule 5). One instance per session, built at
session build alongside OrderStore.

PII discipline (SECURITY §6): the on-file CONTACT is stored as a MASKED factor reference
(e.g. "number ending 0119") — the voice plane never holds a raw phone number or email; the
real value lives with the Phase-4 SoR. A changed address/contact VALUE is spoken to the
caller (their own interaction) and kept inside the store's records, but is NEVER logged or
telemetered — flow telemetry carries the field slug only.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agnostic_market.commerce.identity import CustomersFixture
from agnostic_market.config.loader import ConfigError, load_yaml_layer
from agnostic_market.dtos.confirmation import ProfileField

_STRICT = ConfigDict(extra="forbid")
_FROZEN = ConfigDict(extra="forbid", frozen=True)


class CustomerProfile(BaseModel):
    """One customer's stub profile content."""

    model_config = _STRICT

    address_on_file: str = Field(min_length=1)
    # MASKED factor reference, never a raw number ("number ending 0119") — this is what the
    # step-up OTP is dispatched against and what readbacks may speak.
    contact_on_file: str = Field(min_length=1)


class ProfileFixture(BaseModel):
    """Validated stub profiles for a merchant, keyed by `customer_ref` (Fix 5 Milestone B). The
    KEY is the owning customer — absence is a FAILED LOOKUP, never a fallback to another customer;
    the `customer_ref` is derived from the live session binding, never a model argument."""

    model_config = _STRICT

    profiles: dict[str, CustomerProfile] = Field(min_length=1)


class ProfileChangeRecord(BaseModel):
    """A profile change this store performed (store-internal audit record). `new_value`
    stays INSIDE the store — telemetry/audit log the field slug only, never the value.
    `customer_ref` scopes the change to its owner (Fix 5 Milestone B)."""

    model_config = _FROZEN

    customer_ref: str = Field(min_length=1)
    field: ProfileField
    new_value: str = Field(min_length=1)


class ProfileError(ValueError):
    """A profile change the store refused."""


def load_profile_fixture(config_root: Path, merchant_id: str) -> ProfileFixture:
    """Load + validate the merchant's profile fixture. Fails loudly (build time, not mid-call)."""
    path = config_root / "fixtures" / "profiles" / f"{merchant_id}.yaml"
    try:
        return ProfileFixture.model_validate(load_yaml_layer(path))
    except ValidationError as exc:
        raise ConfigError(f"profile fixture {path} failed validation:\n{exc}") from exc


def assert_profiles_have_customers(
    profiles: ProfileFixture, customers: CustomersFixture
) -> None:
    """Every profile owner must exist in the customer directory.

    This is a build-time fixture integrity check, not a runtime authorization decision. Only
    closed customer refs appear in the error; profile/contact/address values never do.
    """
    unknown = sorted(set(profiles.profiles) - set(customers.customers))
    if unknown:
        raise ConfigError(
            "profiles fixture names customer_refs missing from the customers fixture: "
            + ", ".join(unknown)
        )


class ProfileStore:
    """The stub profile SoR (Fix 5 Milestone B: customer-scoped): per-customer on-file reads
    (update overlay wins) + idempotent updates. EVERY read/update requires a `customer_ref`,
    which the flow derives from the LIVE session binding (never a model argument). A customer
    with no fixture profile has none — reads/updates fail closed, never falling back to another
    customer's data. Changes are keyed by `(customer_ref, intent_key)` so a replay is
    per-customer scoped and one customer's change can never surface for another.

    The `fixture` is REQUIRED — profile data lives in `config/fixtures/profiles/*.yaml`, never
    hardcoded (load via `load_profile_fixture`); tests use the same loader, no baked-in default."""

    def __init__(self, fixture: ProfileFixture) -> None:
        self.fixture = fixture
        self._changes_by_key: dict[tuple[str, str], ProfileChangeRecord] = {}

    def has_profile(self, customer_ref: str) -> bool:
        """Whether THIS customer has a profile on file. The flow checks this BEFORE a mutation
        and fails closed (neutral handover) when False — never revealing whether other profiles
        exist."""
        return customer_ref in self.fixture.profiles

    def _current(self, customer_ref: str, field: ProfileField) -> str:
        base = self.fixture.profiles.get(customer_ref)
        if base is None:
            raise ProfileError(f"no profile on file for {customer_ref}")
        latest = None
        for (ref, _key), record in self._changes_by_key.items():  # insertion-ordered
            if ref == customer_ref and record.field == field:
                latest = record.new_value  # last write for THIS customer wins
        if latest is not None:
            return latest
        return base.address_on_file if field == "address" else base.contact_on_file

    def address_on_file(self, customer_ref: str) -> str:
        """The bound customer's effective delivery address (latest change wins over fixture)."""
        return self._current(customer_ref, "address")

    def contact_on_file(self, customer_ref: str) -> str:
        """The bound customer's effective MASKED contact factor reference (latest change wins).
        This is the OTP factor for that customer — never another customer's."""
        return self._current(customer_ref, "contact")

    def update_profile(
        self, idempotency_key: str, *, customer_ref: str, field: ProfileField, new_value: str
    ) -> ProfileChangeRecord:
        """Apply a profile change for `customer_ref`, deduplicated by `(customer_ref, intent)`
        (SoR-arbiter rule): a replayed effect returns the ORIGINAL record and never applies
        twice. Fails closed for a customer with no profile on file. The caller must have already
        gated the L2 step-up AND proven a binding to THIS customer (§A4c — the flow's live
        re-read of both level and binding at place time)."""
        if not self.has_profile(customer_ref):
            raise ProfileError(f"no profile on file for {customer_ref}")
        key = (customer_ref, idempotency_key)
        existing = self._changes_by_key.get(key)
        if existing is not None:
            return existing
        cleaned = new_value.strip()
        if not cleaned:
            raise ProfileError(f"empty {field} value - nothing to update")
        record = ProfileChangeRecord(customer_ref=customer_ref, field=field, new_value=cleaned)
        self._changes_by_key[key] = record
        return record

    @property
    def change_count(self) -> int:
        """How many DISTINCT changes this store has applied (test/verification surface)."""
        return len(self._changes_by_key)
