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

from agnostic_market.config.loader import ConfigError, load_yaml_layer
from agnostic_market.dtos.confirmation import ProfileField

_STRICT = ConfigDict(extra="forbid")
_FROZEN = ConfigDict(extra="forbid", frozen=True)


class ProfileFixture(BaseModel):
    """Validated stub profile content for one merchant's demo caller."""

    model_config = _STRICT

    address_on_file: str = Field(min_length=1)
    # MASKED factor reference, never a raw number ("number ending 0119") — this is what the
    # step-up OTP is dispatched against and what readbacks may speak.
    contact_on_file: str = Field(min_length=1)


class ProfileChangeRecord(BaseModel):
    """A profile change this store performed (store-internal audit record). `new_value`
    stays INSIDE the store — telemetry/audit log the field slug only, never the value."""

    model_config = _FROZEN

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


# The injectable-fake seam for builders that never enter the profile flow (eval/tests) —
# same fixture-backed test seam as OtpProvider. EXCEPTION: hardcoded fake on-file values,
# because this IS the test seam; production loads a merchant fixture.
_DEFAULT_FIXTURE = ProfileFixture(
    address_on_file="12 Harbor Lane, Springfield",
    contact_on_file="number ending 0119",
)


class ProfileStore:
    """The stub profile SoR: on-file reads (update overlay wins) + idempotent updates."""

    def __init__(self, fixture: ProfileFixture | None = None) -> None:
        self.fixture = fixture or _DEFAULT_FIXTURE
        self._changes_by_key: dict[str, ProfileChangeRecord] = {}

    def _current(self, field: ProfileField) -> str:
        latest = None
        for record in self._changes_by_key.values():  # insertion-ordered: last write wins
            if record.field == field:
                latest = record.new_value
        if latest is not None:
            return latest
        return (
            self.fixture.address_on_file if field == "address" else self.fixture.contact_on_file
        )

    def address_on_file(self) -> str:
        """The effective delivery address (the latest applied change wins over the fixture)."""
        return self._current("address")

    def contact_on_file(self) -> str:
        """The effective MASKED contact factor reference (latest change wins)."""
        return self._current("contact")

    def update_profile(
        self, idempotency_key: str, *, field: ProfileField, new_value: str
    ) -> ProfileChangeRecord:
        """Apply a profile change, deduplicated by per-INTENT `idempotency_key` (SoR-arbiter
        rule): a replayed effect returns the ORIGINAL record and never applies twice. The
        caller must have already gated the L2 step-up (`profile_change_required_level`) —
        the store re-validates shape, not identity (§A4c identity re-check is the flow's
        live level read at place time)."""
        existing = self._changes_by_key.get(idempotency_key)
        if existing is not None:
            return existing
        cleaned = new_value.strip()
        if not cleaned:
            raise ProfileError(f"empty {field} value - nothing to update")
        record = ProfileChangeRecord(field=field, new_value=cleaned)
        self._changes_by_key[idempotency_key] = record
        return record

    @property
    def change_count(self) -> int:
        """How many DISTINCT changes this store has applied (test/verification surface)."""
        return len(self._changes_by_key)
