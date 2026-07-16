"""commerce/identity.py units (P7): claim matching, the authorization check, the session
store, and the fixture loader/cross-check fail-loud contracts. Zero network."""

from __future__ import annotations

from pathlib import Path

import pytest

from agnostic_market.commerce.identity import (
    BoundIdentity,
    CallerIdentityStore,
    CustomerDirectory,
    CustomerEntry,
    CustomersFixture,
    assert_orders_have_customers,
    load_customers_fixture,
    order_read_allowed,
)
from agnostic_market.commerce.orders import OrderStore, load_orders_fixture
from agnostic_market.config.loader import ConfigError
from agnostic_market.dtos.state import CartLine


def _directory() -> CustomerDirectory:
    return CustomerDirectory()  # the default fake: CUST-001 phone, CUST-002 email


# --- claim matching (stub semantics: routing convenience — the OTP proves identity) --------


def test_email_match_normalizes_case_and_whitespace() -> None:
    d = _directory()
    assert d.match_contact(" Casey@Example.COM ").customer_ref == "CUST-002"
    assert d.match_contact("casey @ example.com").customer_ref == "CUST-002"


def test_phone_match_tolerates_formats_and_country_code() -> None:
    d = _directory()
    assert d.match_contact("+1 555 010 0119").customer_ref == "CUST-001"  # exact
    assert d.match_contact("555-010-0119").customer_ref == "CUST-001"  # last-10, punctuated
    assert d.match_contact("5550100119").customer_ref == "CUST-001"  # bare digits


def test_spoken_email_form_matches(config_root: Path) -> None:
    # Live call #12 F-12.1: STT delivers spoken emails WITHOUT an '@' character — the
    # matcher must normalize " at "/" dot " before deciding email-vs-phone.
    d = _directory()
    assert d.match_contact("casey at example dot com").customer_ref == "CUST-002"
    # A trailing sentence period from STT must not break a correct address. (A leading
    # phrase like "It's ..." never reaches here — the model extracts the contact; spaces
    # are JOINED because spelled-out emails arrive as "k c at ..." = "kc@...".)
    assert d.match_contact("casey@example.com.").customer_ref == "CUST-002"
    assert d.match_contact("k c at example dot com") is None  # normalized but WRONG address


def test_spoken_digit_phone_matches() -> None:
    # A phone number spoken as words ("five five five ... oh one one nine") must match.
    d = _directory()
    assert (
        d.match_contact("five five five zero one zero zero one one nine").customer_ref
        == "CUST-001"
    )
    assert (
        d.match_contact("five five five, oh one oh, oh one one nine").customer_ref
        == "CUST-001"
    )


def test_no_cross_type_and_no_short_match() -> None:
    d = _directory()
    assert d.match_contact("nobody@nowhere.example") is None
    assert d.match_contact("0119") is None  # too short for last-10; no exact
    assert d.match_contact("casey") is None  # neither email nor digits
    assert d.match_contact("   ") is None


def test_match_returns_the_masked_form_only() -> None:
    # The matched identity carries the SPEAKABLE masked contact — never the raw value.
    matched = _directory().match_contact("casey@example.com")
    assert matched.masked_contact == "email ending example dot com"
    assert "@" not in matched.masked_contact or "casey" not in matched.masked_contact


# --- the session authorization store ---------------------------------------------------------


def test_store_grants_are_per_order_and_clear_drops_both_rungs() -> None:
    store = CallerIdentityStore()
    store.grant_order(" ord-1001 ")
    assert store.order_granted("ORD-1001")  # normalized
    assert not store.order_granted("ORD-1003")  # never siblings
    store.bind(BoundIdentity(customer_ref="CUST-001", masked_contact="number ending 0119"))
    assert store.current() is not None
    store.clear()  # the reaper hook: BOTH rungs drop together
    assert store.current() is None
    assert not store.order_granted("ORD-1001")


def test_order_read_allowed_is_the_one_shared_check(config_root: Path) -> None:
    store = OrderStore(load_orders_fixture(config_root, "acme_store"))
    identity = CallerIdentityStore()
    # Unverified: nothing readable.
    assert not order_read_allowed("ORD-1001", store=store, identity=identity)
    # Rung 1: a grant unlocks exactly that order.
    identity.grant_order("ORD-1001")
    assert order_read_allowed("ORD-1001", store=store, identity=identity)
    assert not order_read_allowed("ORD-1003", store=store, identity=identity)
    # Rung 2: a binding unlocks all OWNED orders — and only those.
    identity.clear()
    identity.bind(BoundIdentity(customer_ref="CUST-001", masked_contact="number ending 0119"))
    assert order_read_allowed("ORD-1003", store=store, identity=identity)
    assert not order_read_allowed("ORD-1002", store=store, identity=identity)  # CUST-002's
    # Session-placed: readable with NO identity at all (placed-by-this-caller).
    placed = store.place_cart(
        "k1", lines=[CartLine(sku="SKU-GRN-15", name="socks", price_usd=14.5, quantity=1)],
        total_usd=14.5,
    )
    assert order_read_allowed(placed.order_id, store=store, identity=CallerIdentityStore())


# --- fixture loader + build-time cross-check (fail-loud) ------------------------------------


def test_customers_fixture_loads_from_config(config_root: Path) -> None:
    fixture = load_customers_fixture(config_root, "acme_store")
    assert set(fixture.customers) == {"CUST-001", "CUST-002"}


def test_missing_customers_fixture_fails_loudly(config_root: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_customers_fixture(config_root, "ghost_store")


def test_malformed_customers_fixture_rejects_extra_keys(tmp_path: Path) -> None:
    customers_dir = tmp_path / "fixtures" / "customers"
    customers_dir.mkdir(parents=True)
    (customers_dir / "m1.yaml").write_text(
        'customers:\n  CUST-1: { contact: "x@y.z", masked_contact: "email", extra_pii: "no" }\n',
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="failed validation"):
        load_customers_fixture(tmp_path, "m1")


def test_unknown_customer_ref_fails_loud_at_build(config_root: Path) -> None:
    # A customer_ref naming nobody must fail the BUILD (a typo must never silently make an
    # order unlistable at call time).
    orders = load_orders_fixture(config_root, "acme_store")
    only_one = CustomersFixture(
        customers={"CUST-001": CustomerEntry(contact="+1 555 010 0119", masked_contact="m")}
    )
    with pytest.raises(ConfigError, match="ORD-1002 -> CUST-002"):
        assert_orders_have_customers(orders, only_one)
