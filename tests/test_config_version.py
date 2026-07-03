"""config_version hashing: deterministic for the same bundle, changes when content changes."""

from __future__ import annotations

from agnostic_market.config.loader import config_version


def test_same_bundle_same_hash() -> None:
    bundle = {"a": 1, "b": {"c": 2}}
    assert config_version(bundle) == config_version(dict(bundle))


def test_key_order_does_not_matter() -> None:
    assert config_version({"a": 1, "b": 2}) == config_version({"b": 2, "a": 1})


def test_changed_value_changes_hash() -> None:
    assert config_version({"a": 1}) != config_version({"a": 2})


def test_registry_pins_a_version_per_merchant(registry) -> None:
    acme = registry.get("acme_store")
    demo = registry.get("demo_shop")
    assert acme.config_version and demo.config_version
    assert acme.config_version != demo.config_version
