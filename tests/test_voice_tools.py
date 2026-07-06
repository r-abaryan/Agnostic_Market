"""Orders-fixture preload + the read-only tools it feeds (zero network)."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.tools import BaseTool

from agnostic_market.config.loader import ConfigError
from agnostic_market.voice.tools import build_voice_tools, load_orders_fixture


def _tools_by_name(config_root: Path) -> dict[str, BaseTool]:
    fixture = load_orders_fixture(config_root, "acme_store")
    return {t.name: t for t in build_voice_tools(fixture)}


def test_fixture_loads_and_validates(config_root: Path) -> None:
    fixture = load_orders_fixture(config_root, "acme_store")
    assert "ORD-1001" in fixture.orders
    assert fixture.products


def test_order_status_answers_known_order(config_root: Path) -> None:
    result = _tools_by_name(config_root)["order_status"].invoke({"order_id": "ORD-1001"})
    assert "shipped" in result


def test_order_status_is_case_insensitive_on_id(config_root: Path) -> None:
    result = _tools_by_name(config_root)["order_status"].invoke({"order_id": " ord-1001 "})
    assert "shipped" in result


def test_unknown_order_is_a_spoken_not_found_not_an_exception(config_root: Path) -> None:
    result = _tools_by_name(config_root)["order_status"].invoke({"order_id": "ORD-9999"})
    assert "No order found" in result


def test_catalog_search_matches_and_misses(config_root: Path) -> None:
    tools = _tools_by_name(config_root)
    assert "SKU-RED-42" in tools["catalog_search"].invoke({"query": "running"})
    miss = tools["catalog_search"].invoke({"query": "zzz-nothing"})
    assert "No catalog items" in miss
    # A miss steers the model to REAL items instead of invented categories.
    assert "rain jacket" in miss


def test_missing_fixture_fails_loudly_at_build(config_root: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_orders_fixture(config_root, "ghost_store")


def test_malformed_fixture_fails_loudly_at_build(tmp_path: Path) -> None:
    orders_dir = tmp_path / "fixtures" / "orders"
    orders_dir.mkdir(parents=True)
    (orders_dir / "m1.yaml").write_text(
        'orders:\n  ORD-1: { status: "shipped" }\nproducts: []\n', encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="failed validation"):
        load_orders_fixture(tmp_path, "m1")
