"""Masked payment-instrument fixture and customer-scoped lookup. Zero network."""

from __future__ import annotations

from pathlib import Path

import pytest

from agnostic_market.commerce.identity import load_customers_fixture
from agnostic_market.commerce.payment_instruments import (
    PaymentInstrumentDirectory,
    PaymentInstrumentEntry,
    PaymentInstrumentsFixture,
    assert_payment_instruments_have_customers,
    load_payment_instruments_fixture,
)
from agnostic_market.config.loader import ConfigError


def test_fixture_loads_customer_scoped_masked_references(config_root: Path) -> None:
    fixture = load_payment_instruments_fixture(config_root, "acme_store")
    directory = PaymentInstrumentDirectory(fixture)

    assert set(fixture.payment_instruments) == {"CUST-001", "CUST-002"}
    assert directory.new_instrument_ref("CUST-001")
    assert directory.new_instrument_ref("CUST-002")
    assert directory.new_instrument_ref("CUST-UNKNOWN") is None


def test_empty_fixture_is_valid_and_lookup_fails_closed() -> None:
    directory = PaymentInstrumentDirectory(PaymentInstrumentsFixture(payment_instruments={}))

    assert directory.new_instrument_ref("CUST-001") is None


def test_fixture_rejects_an_unmasked_payment_instrument() -> None:
    with pytest.raises(ValueError, match="must be masked"):
        PaymentInstrumentEntry(masked_ref="4111 1111 1111 1111")


def test_instrument_owners_are_cross_checked_without_exposing_values(
    config_root: Path,
) -> None:
    customers = load_customers_fixture(config_root, "acme_store")
    masked_ref = "card ending 1234"
    fixture = PaymentInstrumentsFixture(
        payment_instruments={"CUST-UNKNOWN": PaymentInstrumentEntry(masked_ref=masked_ref)}
    )

    with pytest.raises(ConfigError, match="CUST-UNKNOWN") as exc_info:
        assert_payment_instruments_have_customers(fixture, customers)
    assert masked_ref not in str(exc_info.value)


def test_missing_fixture_fails_loudly(config_root: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_payment_instruments_fixture(config_root, "ghost_store")
