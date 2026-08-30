"""Masked payment-instrument references for the fixture-backed commerce build.

Only caller-speakable masked references enter this directory. Raw payment credentials remain
outside the voice plane, and Support receives this narrow lookup instead of the identity/contact
directory. The real payment system of record replaces this fixture in Phase 4.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from agnostic_market.commerce.identity import CustomersFixture
from agnostic_market.config.loader import ConfigError, load_yaml_layer
from agnostic_market.tenancy.context import TenantBound, normalize_tenant_id

_STRICT = ConfigDict(extra="forbid")


def _digit_count(value: str) -> int:
    return sum(character.isdigit() for character in value)


class PaymentInstrumentEntry(BaseModel):
    """One customer's masked alternative refund instrument."""

    model_config = _STRICT

    masked_ref: str = Field(min_length=1)

    @field_validator("masked_ref")
    @classmethod
    def _reference_is_masked(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("payment instrument reference must not be blank")
        if _digit_count(cleaned) > 4:
            raise ValueError("payment instrument reference must be masked")
        return cleaned


class PaymentInstrumentsFixture(BaseModel):
    """Merchant fixture keyed by the owning customer reference.

    The mapping may be empty: not every customer has an alternative instrument on file, and
    absence must remain an explicit fail-closed runtime outcome.
    """

    model_config = _STRICT

    payment_instruments: dict[str, PaymentInstrumentEntry]


def load_payment_instruments_fixture(
    config_root: Path, merchant_id: str
) -> PaymentInstrumentsFixture:
    """Load and validate one merchant's masked payment-instrument fixture."""
    path = config_root / "fixtures" / "payment_instruments" / f"{merchant_id}.yaml"
    try:
        return PaymentInstrumentsFixture.model_validate(load_yaml_layer(path))
    except ValidationError as exc:
        raise ConfigError(f"payment-instruments fixture {path} failed validation:\n{exc}") from exc


def assert_payment_instruments_have_customers(
    payment_instruments: PaymentInstrumentsFixture,
    customers: CustomersFixture,
) -> None:
    """Fail the build when an instrument names a customer absent from the identity fixture."""
    unknown = sorted(set(payment_instruments.payment_instruments) - set(customers.customers))
    if unknown:
        raise ConfigError(
            "payment-instruments fixture names customer_refs missing from the customers "
            "fixture: " + ", ".join(unknown)
        )


@runtime_checkable
class PaymentInstrumentPort(TenantBound, Protocol):
    """Masked payment-instrument lookup required by refund capabilities."""

    def new_instrument_ref(self, customer_ref: str) -> str | None: ...


class PaymentInstrumentDirectory:
    """Customer-scoped lookup for caller-speakable alternative refund destinations."""

    def __init__(self, tenant_id: str, fixture: PaymentInstrumentsFixture) -> None:
        self.tenant_id = normalize_tenant_id(tenant_id, boundary="payment instrument directory")
        self._references = {
            customer_ref: entry.masked_ref
            for customer_ref, entry in fixture.payment_instruments.items()
        }

    def new_instrument_ref(self, customer_ref: str) -> str | None:
        return self._references.get(customer_ref)
