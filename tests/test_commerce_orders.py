"""OrderStore (the SoR dedup arbiter) + resolve_candidates. Zero network."""

from __future__ import annotations

from pathlib import Path

from agnostic_market.commerce.orders import OrderStore, load_orders_fixture, resolve_candidates


def _store(config_root: Path) -> OrderStore:
    return OrderStore(load_orders_fixture(config_root, "acme_store"))


# --- the idempotency arbiter (A10 rule 5: replay/retry can never double-order) ---------


def test_place_is_idempotent_by_key(config_root: Path) -> None:
    store = _store(config_root)
    first = store.place("key-1", sku="SKU-BLU-07", name="rain jacket", quantity=2, total_usd=258.0)
    replay = store.place("key-1", sku="SKU-BLU-07", name="rain jacket", quantity=2, total_usd=258.0)
    assert replay is first  # the ORIGINAL order, not an equal copy
    assert store.placed_count == 1


def test_distinct_keys_place_distinct_orders(config_root: Path) -> None:
    store = _store(config_root)
    a = store.place("key-a", sku="SKU-GRN-15", name="socks", quantity=1, total_usd=14.5)
    b = store.place("key-b", sku="SKU-GRN-15", name="socks", quantity=1, total_usd=14.5)
    assert a.order_id != b.order_id
    assert store.placed_count == 2


def test_placed_order_is_queryable_by_status_read_through(config_root: Path) -> None:
    store = _store(config_root)
    placed = store.place("key-1", sku="SKU-BLU-07", name="rain jacket", quantity=2, total_usd=258.0)
    summary = store.order_summary(placed.order_id)
    assert summary is not None
    assert "rain jacket" in summary
    # Fixture orders still resolve too.
    assert "shipped" in (store.order_summary("ORD-1001") or "")


# --- candidate resolution (the model picks a KEY, never a SKU) --------------------------


def test_resolve_candidates_narrows_on_match(config_root: Path) -> None:
    fixture = load_orders_fixture(config_root, "acme_store")
    candidates = resolve_candidates(fixture, "rain jacket")
    assert [c.sku for c in candidates] == ["SKU-BLU-07"]
    assert candidates[0].key == "1"
    assert candidates[0].price_usd == 129.00  # price comes from the fixture, never the model


def test_resolve_candidates_returns_full_catalog_on_miss(config_root: Path) -> None:
    fixture = load_orders_fixture(config_root, "acme_store")
    candidates = resolve_candidates(fixture, "zzz-nothing")
    assert len(candidates) == len(fixture.products)
    assert [c.key for c in candidates] == [str(i) for i in range(1, len(candidates) + 1)]
