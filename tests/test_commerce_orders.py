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
from agnostic_market.dtos.state import CartLine


def _store(config_root: Path) -> OrderStore:
    return OrderStore(load_orders_fixture(config_root, "acme_store"))


def _line(sku: str, name: str, price: float, qty: int) -> CartLine:
    return CartLine(sku=sku, name=name, price_usd=price, quantity=qty)


def _place1(store: OrderStore, key: str, sku: str, name: str, qty: int, total: float):
    """Place a single-line order via the multi-line place_cart (Group B): tests that only
    need 'an order exists' express it as one line."""
    return store.place_cart(key, lines=[_line(sku, name, round(total / qty, 2), qty)],
                            total_usd=total)


# --- the idempotency arbiter (A10 rule 5: replay/retry can never double-order) ---------


def test_place_is_idempotent_by_key(config_root: Path) -> None:
    store = _store(config_root)
    first = _place1(store, "key-1", "SKU-BLU-07", "rain jacket", 2, 258.0)
    replay = _place1(store, "key-1", "SKU-BLU-07", "rain jacket", 2, 258.0)
    assert replay is first  # the ORIGINAL order, not an equal copy
    assert store.placed_count == 1


def test_distinct_keys_place_distinct_orders(config_root: Path) -> None:
    store = _store(config_root)
    a = _place1(store, "key-a", "SKU-GRN-15", "socks", 1, 14.5)
    b = _place1(store, "key-b", "SKU-GRN-15", "socks", 1, 14.5)
    assert a.order_id != b.order_id
    assert store.placed_count == 2


def test_place_cart_places_one_multi_line_order(config_root: Path) -> None:
    # Group B: the whole cart becomes ONE order (one id, one total, multi-line summary).
    store = _store(config_root)
    order = store.place_cart(
        "k1",
        lines=[_line("SKU-BLU-07", "rain jacket", 129.0, 2),
               _line("SKU-GRN-15", "pair of socks", 14.5, 1)],
        total_usd=272.5,
    )
    assert len(order.lines) == 2
    assert store.placed_count == 1
    summary = store.order_summary(order.order_id) or ""
    assert "2 rain jackets and 1 pair of socks" in summary  # both lines, speech-native
    assert "$272.50" in summary


def test_placed_order_is_queryable_by_status_read_through(config_root: Path) -> None:
    store = _store(config_root)
    placed = _place1(store, "key-1", "SKU-BLU-07", "rain jacket", 2, 258.0)
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
    placed = _place1(store, "k1", "SKU-BLU-07", "rain jacket", 1, 129.0)
    assert store.is_cancellable(placed.order_id) is True  # placed orders start processing
    store.cancel_order("ck-1", order_id=placed.order_id)
    assert store.order_status(placed.order_id) == "cancelled"


def test_identical_cart_order_lookup_ignores_cancelled(config_root: Path) -> None:
    # The placement guardrail's duplicate probe: the same LINE SET this session flips the
    # readback to the "SECOND order" form; a cancelled match must NOT count (re-order normal).
    store = _store(config_root)
    lines = [_line("SKU-BLU-07", "rain jacket", 129.0, 3)]
    placed = store.place_cart("k1", lines=lines, total_usd=387.0)
    assert store.identical_cart_order(lines) is placed
    assert store.identical_cart_order([_line("SKU-BLU-07", "rain jacket", 129.0, 2)]) is None
    assert store.identical_cart_order([_line("SKU-RED-42", "shoes", 89.99, 3)]) is None
    store.cancel_order("ck-1", order_id=placed.order_id)
    assert store.identical_cart_order(lines) is None


def test_identical_cart_order_is_order_independent(config_root: Path) -> None:
    # Same two lines in either add-order are the SAME cart (dedup is by sku->qty, not sequence).
    store = _store(config_root)
    a = _line("SKU-BLU-07", "rain jacket", 129.0, 2)
    b = _line("SKU-GRN-15", "socks", 14.5, 1)
    placed = store.place_cart("k1", lines=[a, b], total_usd=272.5)
    assert store.identical_cart_order([b, a]) is placed


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
    placed = _place1(store, "k1", "SKU-BLU-07", "rain jacket", 2, 258.0)
    by_id = {o.order_id: o for o in store.actionable_orders()}
    assert by_id["ORD-1001"].status == "shipped"
    assert by_id["ORD-1002"].status == "processing"
    assert by_id["ORD-1003"].status == "delivered"
    assert by_id[placed.order_id].status == "processing"
    store.cancel_order("ck-1", order_id=placed.order_id)
    by_id = {o.order_id: o for o in store.actionable_orders()}
    assert by_id[placed.order_id].status == "cancelled"  # the overlay wins
