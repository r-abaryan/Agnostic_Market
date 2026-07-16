"""OrderStore (the SoR dedup arbiter) + resolve_candidates. Zero network."""

from __future__ import annotations

from pathlib import Path

import pytest

from agnostic_market.commerce.orders import (
    CancelError,
    OrderStore,
    RefundError,
    ReturnError,
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


# --- returns: create_return re-validation matrix + the promise side of the cap (Group C) --


def test_create_return_is_idempotent_by_key(config_root: Path) -> None:
    store = _store(config_root)
    first = store.create_return(
        "rk-1", order_id="ORD-1001", refund_due_usd=179.98, destination="original"
    )
    replay = store.create_return(
        "rk-1", order_id="ORD-1001", refund_due_usd=179.98, destination="original"
    )
    assert replay is first  # the ORIGINAL return, not an equal copy
    assert first.rma_id == "RMA-3001"
    assert store.return_count == 1


def test_create_return_refuses_unshipped_cancelled_and_unknown(config_root: Path) -> None:
    store = _store(config_root)
    with pytest.raises(ReturnError, match="processing"):
        store.create_return("rk-1", order_id="ORD-1002", refund_due_usd=10.0,
                            destination="original")
    store.cancel_order("ck-1", order_id="ORD-1002")
    with pytest.raises(ReturnError, match="cancelled"):
        store.create_return("rk-2", order_id="ORD-1002", refund_due_usd=10.0,
                            destination="original")
    with pytest.raises(ReturnError, match="unknown"):
        store.create_return("rk-3", order_id="ORD-9999", refund_due_usd=10.0,
                            destination="original")
    assert store.return_count == 0


def test_second_return_on_same_order_is_refused_naming_the_open_rma(config_root: Path) -> None:
    store = _store(config_root)
    first = store.create_return(
        "rk-1", order_id="ORD-1001", refund_due_usd=100.0, destination="original"
    )
    with pytest.raises(ReturnError, match=first.rma_id):
        store.create_return("rk-2", order_id="ORD-1001", refund_due_usd=50.0,
                            destination="original")
    assert store.return_count == 1


def test_return_promise_cannot_exceed_refundable_balance(config_root: Path) -> None:
    # $80 already refunded on the $179.98 order: a return may promise at most the remainder.
    store = _store(config_root)
    store.issue_refund("i1", order_id="ORD-1001", amount_usd=80.0, destination="original")
    with pytest.raises(ReturnError, match="exceeds refundable balance"):
        store.create_return("rk-1", order_id="ORD-1001", refund_due_usd=179.98,
                            destination="original")
    ok = store.create_return("rk-2", order_id="ORD-1001", refund_due_usd=99.98,
                             destination="original")
    assert ok.refund_due_usd == 99.98


def test_open_return_promise_blocks_a_refund_on_the_same_dollars(config_root: Path) -> None:
    # The store is the arbiter (§A4c): a refund paid alongside an open return's recorded
    # promise would double-return the same money (refund now + release at Phase 4).
    store = _store(config_root)
    store.create_return("rk-1", order_id="ORD-1001", refund_due_usd=179.98,
                        destination="original")
    with pytest.raises(RefundError, match="promised on open returns"):
        store.issue_refund("i1", order_id="ORD-1001", amount_usd=10.0, destination="original")
    assert store.refund_count == 0


def test_delivered_at_epoch_is_aware_and_none_before_delivery(config_root: Path) -> None:
    from datetime import UTC, datetime

    store = _store(config_root)
    epoch = store.delivered_at_epoch("ORD-1003")
    assert epoch == datetime(2026, 7, 1, tzinfo=UTC).timestamp()  # aware instant, no drift
    assert store.delivered_at_epoch("ORD-1001") is None  # shipped: window not started
    assert store.delivered_at_epoch("ORD-9999") is None  # unknown order


def test_delivered_fixture_entry_requires_delivered_at() -> None:
    # Fail-loud at fixture load: a delivered order without a delivery instant would make
    # the return window silently unenforceable.
    from agnostic_market.commerce.orders import _OrderEntry

    with pytest.raises(ValueError, match="delivered_at"):
        _OrderEntry(
            status="delivered", summary="x", eta="2026-07-01", total_usd=10.0,
            customer_ref="CUST-001",
        )


# --- deterministic read renderers (L3): the code-authored spoken lines that skip the second
#     model pass. Pure functions, so most tests need no store. -----------------------------

from datetime import date  # noqa: E402

from agnostic_market.commerce.orders import (  # noqa: E402
    render_cart_line,
    render_order_status_line,
)

_TODAY = date(2026, 7, 15)


def test_render_past_eta_frames_as_past_and_is_terminal() -> None:
    # The live-call #9 P6 scenario: a shipped order whose ETA has passed. Must frame as PAST
    # ("was expected by"), never "arriving" — AND must NOT promise a follow-up check (F-11.1:
    # the model said "let me check the latest status" then ended the turn without checking).
    line = render_order_status_line(
        order_id="ORD-1001", status="shipped",
        items="2 pairs of trail running shoes", eta="2026-07-09", today=_TODAY,
    )
    assert "was expected by" in line
    assert "arriv" not in line.lower()  # no "arriving"/"arrive" for a past date
    # F-11.1: terminal, no dangling promise to check.
    assert "check" not in line.lower()
    assert "let me" not in line.lower()


def test_render_future_eta_frames_as_upcoming() -> None:
    line = render_order_status_line(
        order_id="ORD-1001", status="shipped",
        items="2 pairs of trail running shoes", eta="2026-07-20", today=_TODAY,
    )
    assert "expected to arrive by" in line
    assert "was expected" not in line


def test_render_today_eta_frames_as_today() -> None:
    # Midnight boundary: eta == today is NOT past.
    line = render_order_status_line(
        order_id="ORD-1001", status="shipped", items="shoes", eta="2026-07-15", today=_TODAY,
    )
    assert "today" in line.lower()
    assert "was expected" not in line


def test_render_duration_eta_has_no_date_logic() -> None:
    line = render_order_status_line(
        order_id="ORD-9001", status="processing", items="a rain jacket",
        eta="3-5 business days", today=_TODAY,
    )
    assert "3 to 5 business days" in line
    assert "expected by" not in line  # no absolute-date framing for a duration


def test_render_malformed_eta_is_omitted_not_spoken_raw() -> None:
    line = render_order_status_line(
        order_id="ORD-1001", status="shipped", items="shoes", eta="soon-ish", today=_TODAY,
    )
    assert "soon-ish" not in line  # never speak an unparseable raw ETA
    assert "on its way" in line  # the status still renders


def test_render_unknown_status_fails_closed() -> None:
    # A status not in the phrase map must NOT get an invented phrase — speak the raw word.
    line = render_order_status_line(
        order_id="ORD-1001", status="returned", items="shoes", eta=None, today=_TODAY,
    )
    assert "returned" in line
    assert "on its way" not in line  # no wrong-status humanization


def test_render_delivered_and_cancelled_omit_eta() -> None:
    # An ETA is meaningless for a delivered (already arrived) or cancelled (won't arrive) order.
    delivered = render_order_status_line(
        order_id="ORD-1003", status="delivered", items="socks", eta="2026-07-01", today=_TODAY,
    )
    assert "delivered" in delivered and "expected" not in delivered
    cancelled = render_order_status_line(
        order_id="ORD-9001", status="cancelled", items="jacket", eta="2026-07-20", today=_TODAY,
    )
    assert "cancelled" in cancelled and "arrive" not in cancelled


def test_render_cart_line() -> None:
    line = render_cart_line([_line("SKU-1", "rain jacket", 129.0, 2)], 258.0)
    assert "2 rain jackets" in line and "$258.00" in line


def test_order_eta_accessor_forks_fixture_placed_and_unknown(config_root: Path) -> None:
    store = _store(config_root)
    assert store.order_eta("ORD-1001") == "2026-07-09"  # fixture entry.eta
    _place1(store, "k1", "SKU-RED-42", "trail running shoes", 1, 89.99)
    assert store.order_eta("ORD-9001") == "3-5 business days"  # placed -> _PLACED_ETA
    assert store.order_eta("ORD-NOPE") is None  # unknown


# --- P7: order ownership + the enumeration renderer -----------------------------------------

from agnostic_market.commerce.orders import render_order_list_line  # noqa: E402


def test_order_owner_and_is_session_placed_fork(config_root: Path) -> None:
    store = _store(config_root)
    assert store.order_owner("ord-1001") == "CUST-001"  # fixture, normalized
    assert store.order_owner("ORD-9999") is None  # unknown
    _place1(store, "k1", "SKU-GRN-15", "merino hiking socks", 2, 14.50)
    assert store.order_owner("ORD-9001") is None  # placed: session-owned, no fixture ref
    assert store.is_session_placed("ord-9001")
    assert not store.is_session_placed("ORD-1001")


def test_owned_orders_filters_by_ref_and_includes_placed(config_root: Path) -> None:
    store = _store(config_root)
    _place1(store, "k1", "SKU-GRN-15", "merino hiking socks", 2, 14.50)
    owned = store.owned_orders("CUST-001")
    ids = [c.order_id for c in owned]
    assert "ORD-1001" in ids and "ORD-1003" in ids  # theirs
    assert "ORD-1002" not in ids  # CUST-002's, never listed
    assert "ORD-9001" in ids  # placed THIS session = the caller's


def test_owned_orders_carries_the_effective_status(config_root: Path) -> None:
    # The cancelled overlay wins — an identified caller must hear the truth about a voided
    # order, not the stale fixture status.
    store = _store(config_root)
    _place1(store, "k1", "SKU-GRN-15", "merino hiking socks", 2, 14.50)
    store.cancel_order("c1", order_id="ORD-9001")
    placed = next(c for c in store.owned_orders("CUST-001") if c.order_id == "ORD-9001")
    assert placed.status == "cancelled"


def test_order_entry_requires_customer_ref() -> None:
    # REQUIRED, no default: an order without an owner would be silently unreadable under
    # object binding — the fixture must fail loudly at load.
    from agnostic_market.commerce.orders import _OrderEntry

    with pytest.raises(ValueError, match="customer_ref"):
        _OrderEntry(status="shipped", summary="x", eta="2026-07-09", total_usd=10.0)


def test_render_order_list_line_speaks_ids_items_status_only() -> None:
    from agnostic_market.commerce.orders import OrderCandidate

    line = render_order_list_line(
        [
            OrderCandidate(key="1", order_id="ORD-1001", summary="2 pairs of shoes",
                           total_usd=179.98, status="shipped"),
            OrderCandidate(key="2", order_id="ORD-1003", summary="3 pairs of socks",
                           total_usd=43.50, status="delivered"),
        ]
    )
    assert "You've got 2 orders" in line
    assert "ORD-1001" in line and "on its way" in line  # humanized via _STATUS_PHRASE
    assert "ORD-1003" in line and "delivered" in line
    assert "$" not in line  # no totals, no addresses, no contact — ids + items + status only


def test_render_order_list_line_empty_and_unknown_status() -> None:
    from agnostic_market.commerce.orders import OrderCandidate

    assert render_order_list_line([]) == "I don't see any orders on your account."
    line = render_order_list_line(
        [OrderCandidate(key="1", order_id="ORD-1", summary="a thing",
                        total_usd=1.0, status="returned")]
    )
    assert "You've got 1 order" in line
    assert "returned" in line  # fail-closed: the raw status word, no invented phrase
