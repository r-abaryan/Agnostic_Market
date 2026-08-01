"""ProfileStore (the customer-scoped profile SoR dedup arbiter) + fixture loading. Zero network.

Fix 5 Milestone B: profiles are keyed by `customer_ref` (the flow supplies it from the LIVE
binding, never a model arg). A customer with no fixture profile fails CLOSED — reads/updates
raise, never falling back to another customer's data. The fixture lives in YAML (no hardcoded
store data); the only populated customer is CUST-001.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agnostic_market.commerce.identity import load_customers_fixture
from agnostic_market.commerce.profile import (
    ProfileError,
    ProfileFixture,
    ProfileStore,
    assert_profiles_have_customers,
    load_profile_fixture,
)
from agnostic_market.config.loader import ConfigError

_OWNER = "CUST-001"  # the only customer with a fixture profile
_STRANGER = "CUST-002"  # a real customer (customers fixture) with NO profile on file


def _store(config_root: Path) -> ProfileStore:
    return ProfileStore(load_profile_fixture(config_root, "acme_store"))


def test_fixture_loads_and_reads_through(config_root: Path) -> None:
    store = _store(config_root)
    assert store.address_on_file(_OWNER) == "12 Harbor Lane, Springfield"
    # The contact is a MASKED factor reference by design — never a raw number.
    assert store.contact_on_file(_OWNER) == "number ending 0119"


def test_unknown_merchant_fixture_fails_loudly(config_root: Path) -> None:
    with pytest.raises(ConfigError):
        load_profile_fixture(config_root, "no_such_merchant")


def test_checked_in_profile_fixtures_are_customer_coherent(config_root: Path) -> None:
    fixture_paths = sorted((config_root / "fixtures" / "profiles").glob("*.yaml"))
    assert fixture_paths

    for fixture_path in fixture_paths:
        merchant_id = fixture_path.stem
        profiles = load_profile_fixture(config_root, merchant_id)
        customers = load_customers_fixture(config_root, merchant_id)
        assert_profiles_have_customers(profiles, customers)


def test_unknown_profile_owner_fails_loudly_without_pii(config_root: Path) -> None:
    loaded = load_profile_fixture(config_root, "acme_store")
    profile = next(iter(loaded.profiles.values()))
    customers = load_customers_fixture(config_root, "acme_store")

    unknown = ProfileFixture(profiles={"CUST-UNKNOWN": profile})
    with pytest.raises(ConfigError, match="CUST-UNKNOWN") as exc_info:
        assert_profiles_have_customers(unknown, customers)
    message = str(exc_info.value)
    assert profile.address_on_file not in message
    assert profile.contact_on_file not in message


def test_update_is_idempotent_by_customer_and_key(config_root: Path) -> None:
    store = _store(config_root)
    first = store.update_profile(
        "k1", customer_ref=_OWNER, field="address", new_value="7 Elm St, Dover"
    )
    replay = store.update_profile(
        "k1", customer_ref=_OWNER, field="address", new_value="7 Elm St, Dover"
    )
    assert replay is first  # the ORIGINAL record, not an equal copy
    assert store.change_count == 1


def test_update_overlay_wins_over_fixture_per_field(config_root: Path) -> None:
    store = _store(config_root)
    before_contact = store.contact_on_file(_OWNER)
    store.update_profile("k1", customer_ref=_OWNER, field="address", new_value="7 Elm St, Dover")
    assert store.address_on_file(_OWNER) == "7 Elm St, Dover"
    assert store.contact_on_file(_OWNER) == before_contact  # other field untouched
    store.update_profile("k2", customer_ref=_OWNER, field="address", new_value="9 Oak Ave, Leeds")
    assert store.address_on_file(_OWNER) == "9 Oak Ave, Leeds"  # latest change wins


def test_blank_value_is_refused(config_root: Path) -> None:
    store = _store(config_root)
    with pytest.raises(ProfileError, match="empty"):
        store.update_profile("k1", customer_ref=_OWNER, field="contact", new_value="   ")
    assert store.change_count == 0


# --- Fix 5 Milestone B: customer isolation + fail-closed ------------------------------------


def test_customer_without_profile_fails_closed_on_read(config_root: Path) -> None:
    store = _store(config_root)
    assert not store.has_profile(_STRANGER)
    with pytest.raises(ProfileError, match="no profile on file"):
        store.address_on_file(_STRANGER)
    with pytest.raises(ProfileError, match="no profile on file"):
        store.contact_on_file(_STRANGER)


def test_customer_without_profile_fails_closed_on_update(config_root: Path) -> None:
    store = _store(config_root)
    with pytest.raises(ProfileError, match="no profile on file"):
        store.update_profile("k1", customer_ref=_STRANGER, field="address", new_value="X")
    assert store.change_count == 0


def test_one_customers_change_never_surfaces_for_another(config_root: Path) -> None:
    # Isolation: even if two customers had profiles, a change to one must never bleed into the
    # other's reads. Here CUST-001 changes; the stranger still has NO profile (not CUST-001's).
    store = _store(config_root)
    store.update_profile("k1", customer_ref=_OWNER, field="address", new_value="7 Elm St, Dover")
    assert store.address_on_file(_OWNER) == "7 Elm St, Dover"
    with pytest.raises(ProfileError, match="no profile on file"):
        store.address_on_file(_STRANGER)  # never falls back to CUST-001's changed address


def test_idempotency_is_scoped_per_customer(config_root: Path) -> None:
    # Use the real fixture record under two test-local owners so the permanent demo fixture still
    # contains no invented CUST-002 data. The SAME intent key must create one isolated record per
    # customer and replay independently inside each scope.
    loaded = load_profile_fixture(config_root, "acme_store")
    profile = loaded.profiles[_OWNER]
    store = ProfileStore(ProfileFixture(profiles={_OWNER: profile, _STRANGER: profile}))

    first = store.update_profile(
        "shared-key", customer_ref=_OWNER, field="address", new_value="7 Elm St"
    )
    second = store.update_profile(
        "shared-key", customer_ref=_STRANGER, field="address", new_value="9 Oak Ave"
    )

    assert first.customer_ref == _OWNER
    assert second.customer_ref == _STRANGER
    assert first is not second
    assert store.change_count == 2
    assert store.address_on_file(_OWNER) == "7 Elm St"
    assert store.address_on_file(_STRANGER) == "9 Oak Ave"
    assert (
        store.update_profile(
            "shared-key", customer_ref=_OWNER, field="address", new_value="7 Elm St"
        )
        is first
    )
    assert (
        store.update_profile(
            "shared-key", customer_ref=_STRANGER, field="address", new_value="9 Oak Ave"
        )
        is second
    )
