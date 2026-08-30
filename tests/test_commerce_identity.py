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
    classify_contact_claims,
    load_customers_fixture,
    order_mutation_allowed,
    order_read_allowed,
    try_grant_orders_by_contact,
)
from agnostic_market.commerce.orders import GuestOrderScope, OrderStore, load_orders_fixture
from agnostic_market.config.loader import ConfigError
from agnostic_market.dtos.state import CartLine


@pytest.mark.parametrize(
    "utterance",
    (
        "+1 555 010 0119 or 555 010 0119",
        "casey@example.com, again casey at example dot com",
    ),
)
def test_contact_claim_classification_deduplicates_by_directory_match_key(
    utterance: str,
) -> None:
    assert classify_contact_claims(utterance).disposition == "single"


def test_contact_claim_classification_rejects_two_distinct_claims() -> None:
    selection = classify_contact_claims("casey@example.com or 555 010 0119")
    assert selection.disposition == "multiple"
    assert selection.claim is None


def _directory(config_root: Path) -> CustomerDirectory:
    # From the acme_store YAML fixture (CUST-001 phone, CUST-002 email) — no hardcoded data.
    return CustomerDirectory("acme_store", load_customers_fixture(config_root, "acme_store"))


# --- claim matching (stub semantics: routing convenience — the OTP proves identity) --------


def test_email_match_normalizes_case_and_whitespace(config_root: Path) -> None:
    d = _directory(config_root)
    assert d.match_contact(" Casey@Example.COM ").customer_ref == "CUST-002"
    assert d.match_contact("casey @ example.com").customer_ref == "CUST-002"


def test_phone_match_tolerates_formats_and_country_code(config_root: Path) -> None:
    d = _directory(config_root)
    assert d.match_contact("+1 555 010 0119").customer_ref == "CUST-001"  # exact
    assert d.match_contact("555-010-0119").customer_ref == "CUST-001"  # last-10, punctuated
    assert d.match_contact("5550100119").customer_ref == "CUST-001"  # bare digits


def test_spoken_email_form_matches(config_root: Path) -> None:
    # Live call #12 F-12.1: STT delivers spoken emails WITHOUT an '@' character — the
    # matcher must normalize " at "/" dot " before deciding email-vs-phone.
    d = _directory(config_root)
    assert d.match_contact("casey at example dot com").customer_ref == "CUST-002"
    # A trailing sentence period from STT must not break a correct address. (A leading
    # phrase like "It's ..." never reaches here — the model extracts the contact; spaces
    # are JOINED because spelled-out emails arrive as "k c at ..." = "kc@...".)
    assert d.match_contact("casey@example.com.").customer_ref == "CUST-002"
    assert d.match_contact("k c at example dot com") is None  # normalized but WRONG address


def test_spoken_digit_phone_matches(config_root: Path) -> None:
    # A phone number spoken as words ("five five five ... oh one one nine") must match.
    d = _directory(config_root)
    assert (
        d.match_contact("five five five zero one zero zero one one nine").customer_ref == "CUST-001"
    )
    assert d.match_contact("five five five, oh one oh, oh one one nine").customer_ref == "CUST-001"


def test_no_cross_type_and_no_short_match(config_root: Path) -> None:
    d = _directory(config_root)
    assert d.match_contact("nobody@nowhere.example") is None
    assert d.match_contact("0119") is None  # too short for last-10; no exact
    assert d.match_contact("casey") is None  # neither email nor digits
    assert d.match_contact("   ") is None


def test_match_returns_the_masked_form_only(config_root: Path) -> None:
    # The matched identity carries the SPEAKABLE masked contact — never the raw value.
    matched = _directory(config_root).match_contact("casey@example.com")
    assert matched.masked_contact == "email ending example dot com"
    assert "@" not in matched.masked_contact or "casey" not in matched.masked_contact


# --- the session authorization store ---------------------------------------------------------


def test_store_grants_are_per_order_and_clear_drops_both_rungs() -> None:
    store = CallerIdentityStore()
    store.grant_orders(" ord-1001 ")
    assert store.order_granted("ORD-1001")  # normalized
    assert not store.order_granted("ORD-1003")  # never siblings
    store.bind(BoundIdentity(customer_ref="CUST-001", masked_contact="number ending 0119"))
    assert store.current() is not None
    store.clear()  # the reaper hook: BOTH rungs drop together
    assert store.current() is None
    assert not store.order_granted("ORD-1001")


def test_store_grants_validate_the_complete_set_before_mutating() -> None:
    store = CallerIdentityStore()
    store.grant_orders("ORD-1001")

    with pytest.raises(ValueError, match="one or more non-blank"):
        store.grant_orders("ORD-1003", " ")
    with pytest.raises(ValueError, match="one or more non-blank"):
        store.grant_orders()

    assert store.order_granted("ORD-1001")
    assert not store.order_granted("ORD-1003")


def test_contact_grant_is_atomic_across_a_same_owner_set(config_root: Path) -> None:
    orders = OrderStore("acme_store", load_orders_fixture(config_root, "acme_store").orders)
    customers = _directory(config_root)
    identity = CallerIdentityStore()

    with pytest.raises(ValueError, match="one or more order ids"):
        try_grant_orders_by_contact(
            "+1 555 010 0119",
            store=orders,
            customers=customers,
            identity=identity,
        )

    assert (
        try_grant_orders_by_contact(
            "+1 555 010 0119",
            "ORD-1001",
            "ORD-1003",
            store=orders,
            customers=customers,
            identity=identity,
        )
        == "granted"
    )
    assert identity.order_granted("ORD-1001")
    assert identity.order_granted("ORD-1003")


def test_contact_grant_rejects_mixed_owners_without_partial_authority(
    config_root: Path,
) -> None:
    orders = OrderStore("acme_store", load_orders_fixture(config_root, "acme_store").orders)
    identity = CallerIdentityStore()
    identity.grant_orders("ORD-1001")

    assert (
        try_grant_orders_by_contact(
            "+1 555 010 0119",
            "ORD-1003",
            "ORD-1002",
            store=orders,
            customers=_directory(config_root),
            identity=identity,
        )
        == "mismatch"
    )
    assert identity.order_granted("ORD-1001")
    assert not identity.order_granted("ORD-1003")
    assert not identity.order_granted("ORD-1002")


def test_bound_principal_refuses_contact_grants(config_root: Path) -> None:
    orders = OrderStore("acme_store", load_orders_fixture(config_root, "acme_store").orders)
    identity = CallerIdentityStore()
    identity.bind(BoundIdentity(customer_ref="CUST-001", masked_contact="number ending 0119"))

    assert (
        try_grant_orders_by_contact(
            "casey@example.com",
            "ORD-1002",
            store=orders,
            customers=_directory(config_root),
            identity=identity,
        )
        == "mismatch"
    )
    assert not identity.order_granted("ORD-1002")


def test_order_read_allowed_is_the_one_shared_check(config_root: Path) -> None:
    store = OrderStore("acme_store", load_orders_fixture(config_root, "acme_store").orders)
    guest_orders = GuestOrderScope(tenant_id="acme_store", session_id="identity-read")
    identity = CallerIdentityStore()
    # Unverified: nothing readable.
    assert not order_read_allowed(
        "ORD-1001", store=store, guest_orders=guest_orders, identity=identity
    )
    # Rung 1: a grant unlocks exactly that order.
    identity.grant_orders("ORD-1001")
    assert order_read_allowed("ORD-1001", store=store, guest_orders=guest_orders, identity=identity)
    assert not order_read_allowed(
        "ORD-1003", store=store, guest_orders=guest_orders, identity=identity
    )
    # Rung 2: a binding unlocks all OWNED orders — and only those.
    identity.clear()
    identity.bind(BoundIdentity(customer_ref="CUST-001", masked_contact="number ending 0119"))
    assert order_read_allowed("ORD-1003", store=store, guest_orders=guest_orders, identity=identity)
    assert not order_read_allowed(
        "ORD-1002", store=store, guest_orders=guest_orders, identity=identity
    )
    # Session-placed: readable with NO identity at all (placed-by-this-caller).
    placed = store.place_cart(
        "k1",
        lines=[CartLine(sku="SKU-GRN-15", name="socks", price_usd=14.5, quantity=1)],
        total_usd=14.5,
    )
    guest_orders.record(placed.order_id)
    assert order_read_allowed(
        placed.order_id,
        store=store,
        guest_orders=guest_orders,
        identity=CallerIdentityStore(),
    )


def test_bound_principal_cannot_be_widened_by_a_residual_guest_grant(
    config_root: Path,
) -> None:
    store = OrderStore("acme_store", load_orders_fixture(config_root, "acme_store").orders)
    guest_orders = GuestOrderScope(tenant_id="acme_store", session_id="identity-bound")
    identity = CallerIdentityStore()
    identity.grant_orders("ORD-1002")
    identity.bind(BoundIdentity(customer_ref="CUST-001", masked_contact="number ending 0119"))

    assert not order_read_allowed(
        "ORD-1002", store=store, guest_orders=guest_orders, identity=identity
    )
    assert not order_mutation_allowed(
        "ORD-1002", store=store, guest_orders=guest_orders, identity=identity
    )


def test_guest_scope_reference_requires_a_committed_tenant_placement(config_root: Path) -> None:
    store = OrderStore("acme_store", load_orders_fixture(config_root, "acme_store").orders)
    guest_orders = GuestOrderScope(tenant_id="acme_store", session_id="corrupted-scope")
    guest_orders.record("ORD-1002")
    identity = CallerIdentityStore()

    assert not store.is_guest_order("ORD-1002", guest_orders)
    assert not order_read_allowed(
        "ORD-1002", store=store, guest_orders=guest_orders, identity=identity
    )
    assert not order_mutation_allowed(
        "ORD-1002", store=store, guest_orders=guest_orders, identity=identity
    )


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


def test_duplicate_customer_contact_fails_fixture_load() -> None:
    # `match_contact` is first-match — a shared contact would silently deny the later-listed
    # owner, so the stub directory rejects it at load. Uniqueness is checked on the MATCH
    # key (last-10 digits), so two spellings of one number still collide.
    with pytest.raises(ValueError, match="share a contact"):
        CustomersFixture(
            customers={
                "CUST-001": CustomerEntry(contact="+1 555 010 0119", masked_contact="m1"),
                "CUST-003": CustomerEntry(contact="555 010 0119", masked_contact="m3"),
            }
        )
