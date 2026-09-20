"""Cross-family integrity for fixture-backed merchant bundles."""

from __future__ import annotations

from agnostic_market.commerce.identity import (
    CustomersFixture,
    assert_orders_have_customers,
)
from agnostic_market.commerce.orders import OrdersFixture
from agnostic_market.commerce.payment_instruments import (
    PaymentInstrumentsFixture,
    assert_payment_instruments_have_customers,
)
from agnostic_market.commerce.profile import ProfileFixture, assert_profiles_have_customers
from agnostic_market.commerce.verification import VerificationFixture
from agnostic_market.config.loader import ConfigError


def assert_fixture_bundle_integrity(
    *,
    orders: OrdersFixture,
    customers: CustomersFixture,
    payment_instruments: PaymentInstrumentsFixture,
    profiles: ProfileFixture,
    verification: VerificationFixture,
) -> None:
    """Reject references that cannot compose one coherent merchant fixture bundle."""

    assert_orders_have_customers(orders, customers)
    assert_profiles_have_customers(profiles, customers)
    assert_payment_instruments_have_customers(payment_instruments, customers)

    expected_factor_refs = {entry.factor_ref for entry in customers.customers.values()}
    factor_refs = set(verification.otp_codes_by_factor_ref)
    if factor_refs == expected_factor_refs:
        return

    details: list[str] = []
    missing = sorted(expected_factor_refs - factor_refs)
    unknown = sorted(factor_refs - expected_factor_refs)
    if missing:
        details.append("missing factors: " + ", ".join(missing))
    if unknown:
        details.append("unknown factors: " + ", ".join(unknown))
    raise ConfigError("verification fixture does not match customers: " + "; ".join(details))
