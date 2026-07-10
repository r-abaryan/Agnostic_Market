"""OrderStore (the SoR dedup arbiter) + resolve_candidates. Zero network."""

from __future__ import annotations

from pathlib import Path

import pytest

from agnostic_market.commerce.orders import (
    CancelError,
    OrderStore,
    load_orders_fixture,
    resolve_candidates,
)


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


# --- cancel-order (Group A): eligibility + idempotent void ------------------------------


def test_processing_order_is_cancellable_shipped_is_not(config_root: Path) -> None:
    store = _store(config_root)
    assert store.is_cancellable("ORD-1002") is True  # processing
    assert store.is_cancellable("ORD-1001") is False  # shipped
    assert store.is_cancellable("ORD-1003") is False  # delivered
    assert store.is_cancellable("NOPE") is False  # unknown


def test_cancel_voids_processing_and_reads_back_cancelled(config_root: Path) -> None:
    store = _store(config_root)
    rec = store.cancel_order("ck-1", order_id="ORD-1002")
    assert rec.order_id == "ORD-1002"
    assert store.order_status("ORD-1002") == "cancelled"
    assert "cancelled" in (store.order_summary("ORD-1002") or "")
    assert store.is_cancellable("ORD-1002") is False  # can't cancel a cancelled order


def test_cancel_is_idempotent_by_key(config_root: Path) -> None:
    store = _store(config_root)
    a = store.cancel_order("ck-1", order_id="ORD-1002")
    b = store.cancel_order("ck-1", order_id="ORD-1002")
    assert a is b  # the ORIGINAL cancel, not a re-void
    assert store.cancel_count == 1


def test_cancel_refuses_shipped_and_unknown(config_root: Path) -> None:
    store = _store(config_root)
    with pytest.raises(CancelError):
        store.cancel_order("ck-1", order_id="ORD-1001")  # shipped
    with pytest.raises(CancelError):
        store.cancel_order("ck-2", order_id="NOPE-404")  # unknown
    assert store.cancel_count == 0


def test_placed_order_is_cancellable(config_root: Path) -> None:
    store = _store(config_root)
    placed = store.place("k1", sku="SKU-BLU-07", name="rain jacket", quantity=1, total_usd=129.0)
    assert store.is_cancellable(placed.order_id) is True  # placed orders start processing
    store.cancel_order("ck-1", order_id=placed.order_id)
    assert store.order_status(placed.order_id) == "cancelled"


def test_cancel_record_carries_the_reversed_amount(config_root: Path) -> None:
    # The spoken outcome states the money movement, so the record must know the captured
    # total it reversed.
    store = _store(config_root)
    rec = store.cancel_order("ck-1", order_id="ORD-1002")
    assert rec.total_usd == 129.00


# --- the refund<->cancel cross-effect invariant (money may only come back ONCE) ---------


def test_cancel_refuses_an_order_with_refunds_issued(config_root: Path) -> None:
    # A void reverses the FULL charge; on top of a prior partial refund that returns money
    # twice — the mixed case belongs to a person, never an automatic void.
    store = _store(config_root)
    store.issue_refund("r1", order_id="ORD-1002", amount_usd=50.0, destination="original")
    with pytest.raises(CancelError):
        store.cancel_order("ck-1", order_id="ORD-1002")
    assert store.cancel_count == 0
    assert store.order_status("ORD-1002") == "processing"  # untouched


# --- actionable_orders: effective status drives remedy selection ------------------------


def test_actionable_orders_carry_effective_status(config_root: Path) -> None:
    store = _store(config_root)
    placed = store.place("k1", sku="SKU-BLU-07", name="rain jacket", quantity=2, total_usd=258.0)
    by_id = {o.order_id: o for o in store.actionable_orders()}
    assert by_id["ORD-1001"].status == "shipped"
    assert by_id["ORD-1002"].status == "processing"
    assert by_id["ORD-1003"].status == "delivered"
    assert by_id[placed.order_id].status == "processing"
    store.cancel_order("ck-1", order_id=placed.order_id)
    by_id = {o.order_id: o for o in store.actionable_orders()}
    assert by_id[placed.order_id].status == "cancelled"  # the overlay wins
