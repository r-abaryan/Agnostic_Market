"""ProfileStore (the profile SoR dedup arbiter) + fixture loading. Zero network."""

from __future__ import annotations

from pathlib import Path

import pytest

from agnostic_market.commerce.profile import (
    ProfileError,
    ProfileStore,
    load_profile_fixture,
)
from agnostic_market.config.loader import ConfigError


def test_fixture_loads_and_reads_through(config_root: Path) -> None:
    store = ProfileStore(load_profile_fixture(config_root, "acme_store"))
    assert store.address_on_file() == "12 Harbor Lane, Springfield"
    # The contact is a MASKED factor reference by design — never a raw number.
    assert store.contact_on_file() == "number ending 0119"


def test_unknown_merchant_fixture_fails_loudly(config_root: Path) -> None:
    with pytest.raises(ConfigError):
        load_profile_fixture(config_root, "no_such_merchant")


def test_update_is_idempotent_by_key() -> None:
    store = ProfileStore()
    first = store.update_profile("k1", field="address", new_value="7 Elm St, Dover")
    replay = store.update_profile("k1", field="address", new_value="7 Elm St, Dover")
    assert replay is first  # the ORIGINAL record, not an equal copy
    assert store.change_count == 1


def test_update_overlay_wins_over_fixture_per_field() -> None:
    store = ProfileStore()
    before_contact = store.contact_on_file()
    store.update_profile("k1", field="address", new_value="7 Elm St, Dover")
    assert store.address_on_file() == "7 Elm St, Dover"
    assert store.contact_on_file() == before_contact  # other field untouched
    store.update_profile("k2", field="address", new_value="9 Oak Ave, Leeds")
    assert store.address_on_file() == "9 Oak Ave, Leeds"  # latest change wins


def test_blank_value_is_refused() -> None:
    store = ProfileStore()
    with pytest.raises(ProfileError, match="empty"):
        store.update_profile("k1", field="contact", new_value="   ")
    assert store.change_count == 0
