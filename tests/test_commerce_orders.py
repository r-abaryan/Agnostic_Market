"""OrderStore (the SoR dedup arbiter) + resolve_candidates. Zero network."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from agnostic_market.commerce.orders import (
    CancelError,
    Candidate,
    OrdersFixture,
    OrderStore,
    RecentOrderContext,
    RefundError,
    ReturnError,
    load_orders_fixture,
    lookup_catalog,
    match_named_items,
    number_candidates,
    resolve_candidates,
)
from agnostic_market.dtos.state import CartLine

_ORIGINAL_INSTRUMENT = "original payment method"


def _store(config_root: Path) -> OrderStore:
    return OrderStore(load_orders_fixture(config_root, "acme_store"))


def _line(sku: str, name: str, price: float, qty: int) -> CartLine:
    return CartLine(sku=sku, name=name, price_usd=price, quantity=qty)


def _place1(store: OrderStore, key: str, sku: str, name: str, qty: int, total: float):
    """Place a single-line order via the multi-line place_cart (Group B): tests that only
    need 'an order exists' express it as one line."""
    return store.place_cart(
        key, lines=[_line(sku, name, round(total / qty, 2), qty)], total_usd=total
    )


def test_recent_order_context_marks_a_bounded_set_incomplete() -> None:
    context = RecentOrderContext(max_refs=2)
    context.record(["ORD-1001", "ORD-1002", "ORD-1003"], operation="list")
    snapshot = context.snapshot()
    assert snapshot.order_refs == ("ORD-1002", "ORD-1003")
    assert snapshot.focused_order_ref == "ORD-1003"
    assert snapshot.complete is False


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
        lines=[
            _line("SKU-BLU-07", "rain jacket", 129.0, 2),
            _line("SKU-GRN-15", "pair of socks", 14.5, 1),
        ],
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


def _candidate(key: str, name: str) -> Candidate:
    return Candidate(key=key, sku=f"SKU-{key}", name=name, price_usd=1.0)


def test_match_named_items_returns_only_actual_matches() -> None:
    items = [_candidate("1", "trail running shoes"), _candidate("2", "rain jacket")]

    assert match_named_items(items, "I want the trail running shoes") == [items[0]]


@pytest.mark.parametrize("query", ("", "   ", "zzz-nothing"))
def test_match_named_items_returns_empty_without_an_actual_match(query: str) -> None:
    items = [_candidate("1", "trail running shoes"), _candidate("2", "rain jacket")]

    assert match_named_items(items, query) == []


def test_match_named_items_preserves_ambiguity_for_the_caller_to_narrow() -> None:
    items = [_candidate("1", "trail running shoes"), _candidate("2", "trail hiking shoes")]

    assert match_named_items(items, "trail") == items


@pytest.mark.parametrize(
    ("query", "matched_names"),
    (
        ("running", ("trail running shoes",)),
        ("  RUNNING  ", ("trail running shoes",)),
        ("I want the trail running shoes", ()),
        ("", ()),
        ("   ", ()),
        ("zzz-nothing", ()),
    ),
)
def test_catalog_lookup_preserves_the_legacy_one_way_match_contract(
    config_root: Path,
    query: str,
    matched_names: tuple[str, ...],
) -> None:
    fixture = load_orders_fixture(config_root, "acme_store")

    result = lookup_catalog(fixture, query)

    assert tuple(product.name for product in result.matches) == matched_names
    assert result.available == tuple(fixture.products)


def test_number_candidates_is_shared_by_catalog_and_live_cart_items() -> None:
    lines = (
        _line("SKU-1", "trail shoes", 79.0, 2),
        _line("SKU-2", "rain jacket", 129.0, 1),
    )

    assert number_candidates(lines) == [
        Candidate(key="1", sku="SKU-1", name="trail shoes", price_usd=79.0),
        Candidate(key="2", sku="SKU-2", name="rain jacket", price_usd=129.0),
    ]


def test_orders_fixture_requires_unique_skus_but_allows_duplicate_names() -> None:
    products = [
        {"sku": "SKU-1", "name": "trail runner", "price_usd": 80.0},
        {"sku": "SKU-2", "name": "trail runner", "price_usd": 95.0},
    ]

    fixture = OrdersFixture.model_validate({"orders": {}, "products": products})
    assert [product.sku for product in fixture.products] == ["SKU-1", "SKU-2"]

    products[1]["sku"] = "SKU-1"
    with pytest.raises(ValidationError, match="product SKUs must be unique"):
        OrdersFixture.model_validate({"orders": {}, "products": products})


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
    store.issue_refund(
        "r1",
        order_id="ORD-1002",
        amount_usd=50.0,
        destination="original",
        instrument_ref=_ORIGINAL_INSTRUMENT,
    )
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
        store.create_return(
            "rk-1", order_id="ORD-1002", refund_due_usd=10.0, destination="original"
        )
    store.cancel_order("ck-1", order_id="ORD-1002")
    with pytest.raises(ReturnError, match="cancelled"):
        store.create_return(
            "rk-2", order_id="ORD-1002", refund_due_usd=10.0, destination="original"
        )
    with pytest.raises(ReturnError, match="unknown"):
        store.create_return(
            "rk-3", order_id="ORD-9999", refund_due_usd=10.0, destination="original"
        )
    assert store.return_count == 0


def test_second_return_on_same_order_is_refused_naming_the_open_rma(config_root: Path) -> None:
    store = _store(config_root)
    first = store.create_return(
        "rk-1", order_id="ORD-1001", refund_due_usd=100.0, destination="original"
    )
    with pytest.raises(ReturnError, match=first.rma_id):
        store.create_return(
            "rk-2", order_id="ORD-1001", refund_due_usd=50.0, destination="original"
        )
    assert store.return_count == 1


def test_return_promise_cannot_exceed_refundable_balance(config_root: Path) -> None:
    # $80 already refunded on the $179.98 order: a return may promise at most the remainder.
    store = _store(config_root)
    store.issue_refund(
        "i1",
        order_id="ORD-1001",
        amount_usd=80.0,
        destination="original",
        instrument_ref=_ORIGINAL_INSTRUMENT,
    )
    with pytest.raises(ReturnError, match="exceeds refundable balance"):
        store.create_return(
            "rk-1", order_id="ORD-1001", refund_due_usd=179.98, destination="original"
        )
    ok = store.create_return(
        "rk-2", order_id="ORD-1001", refund_due_usd=99.98, destination="original"
    )
    assert ok.refund_due_usd == 99.98


def test_open_return_promise_blocks_a_refund_on_the_same_dollars(config_root: Path) -> None:
    # The store is the arbiter (§A4c): a refund paid alongside an open return's recorded
    # promise would double-return the same money (refund now + release at Phase 4).
    store = _store(config_root)
    store.create_return("rk-1", order_id="ORD-1001", refund_due_usd=179.98, destination="original")
    with pytest.raises(RefundError, match="promised on open returns"):
        store.issue_refund(
            "i1",
            order_id="ORD-1001",
            amount_usd=10.0,
            destination="original",
            instrument_ref=_ORIGINAL_INSTRUMENT,
        )
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
            status="delivered",
            summary="x",
            eta="2026-07-01",
            total_usd=10.0,
            customer_ref="CUST-001",
        )


# --- deterministic read renderers (L3): the code-authored spoken lines that skip the second
#     model pass. Pure functions, so most tests need no store. -----------------------------

from datetime import date  # noqa: E402

from agnostic_market.commerce.orders import (  # noqa: E402
    render_batch_cancel_outcome,
    render_cart_line,
    render_order_status_line,
)
from agnostic_market.dtos.state import BatchCancelOutcome  # noqa: E402

_TODAY = date(2026, 7, 15)


def test_render_batch_cancel_single_matches_legacy_single_cancel_line() -> None:
    # A batch-of-one cancelled order reads byte-identical to the retired single-cancel void
    # line (single = batch-of-one; no speech regression).
    out = [
        BatchCancelOutcome(
            order_id="ORD-1002", summary="the rain jacket", outcome="cancelled", amount_usd=129.0
        )
    ]
    assert render_batch_cancel_outcome(out) == (
        "Done - I've cancelled your order for the rain jacket. The $129.00 charge goes back "
        "to your original payment method."
    )


def test_render_batch_cancel_states_each_target_from_its_outcome() -> None:
    # INV-25: each clause comes from the STRUCTURED per-target outcome (amount only for a
    # cancelled one; the declines name the honest reason, never a fabricated completion).
    out = [
        BatchCancelOutcome(
            order_id="ORD-1002", summary="the rain jacket", outcome="cancelled", amount_usd=129.0
        ),
        BatchCancelOutcome(order_id="ORD-1001", summary="the shoes", outcome="not_cancellable"),
        BatchCancelOutcome(order_id="ORD-9", summary="the boots", outcome="store_refused"),
        BatchCancelOutcome(order_id="ORD-8", summary="the hat", outcome="not_completed"),
    ]
    line = render_batch_cancel_outcome(out)
    assert "ORD-1002" in line and "$129.00" in line
    assert "ORD-1001" in line and "already shipped" in line.lower()
    assert "ORD-9" in line and "couldn't cancel" in line.lower()
    assert "ORD-8" in line and "request" in line.lower() and "did not complete" in line.lower()


def test_batch_cancel_outcome_requires_amount_exactly_for_cancelled() -> None:
    with pytest.raises(ValueError, match="requires amount_usd"):
        BatchCancelOutcome(order_id="ORD-1", summary="one item", outcome="cancelled")
    with pytest.raises(ValueError, match="only a cancelled outcome"):
        BatchCancelOutcome(
            order_id="ORD-1",
            summary="one item",
            outcome="store_refused",
            amount_usd=10.0,
        )


def test_render_past_eta_frames_as_past_and_is_terminal() -> None:
    # The live-call #9 P6 scenario: a shipped order whose ETA has passed. Must frame as PAST
    # ("was expected by"), never "arriving" — AND must NOT promise a follow-up check (F-11.1:
    # the model said "let me check the latest status" then ended the turn without checking).
    line = render_order_status_line(
        order_id="ORD-1001",
        status="shipped",
        items="2 pairs of trail running shoes",
        eta="2026-07-09",
        today=_TODAY,
    )
    assert "was expected by" in line
    assert "arriv" not in line.lower()  # no "arriving"/"arrive" for a past date
    # F-11.1: terminal, no dangling promise to check.
    assert "check" not in line.lower()
    assert "let me" not in line.lower()


def test_render_future_eta_frames_as_upcoming() -> None:
    line = render_order_status_line(
        order_id="ORD-1001",
        status="shipped",
        items="2 pairs of trail running shoes",
        eta="2026-07-20",
        today=_TODAY,
    )
    assert "expected to arrive by" in line
    assert "was expected" not in line


def test_render_today_eta_frames_as_today() -> None:
    # Midnight boundary: eta == today is NOT past.
    line = render_order_status_line(
        order_id="ORD-1001",
        status="shipped",
        items="shoes",
        eta="2026-07-15",
        today=_TODAY,
    )
    assert "today" in line.lower()
    assert "was expected" not in line


def test_render_duration_eta_has_no_date_logic() -> None:
    line = render_order_status_line(
        order_id="ORD-9001",
        status="processing",
        items="a rain jacket",
        eta="3-5 business days",
        today=_TODAY,
    )
    assert "3 to 5 business days" in line
    assert "expected by" not in line  # no absolute-date framing for a duration


def test_render_malformed_eta_is_omitted_not_spoken_raw() -> None:
    line = render_order_status_line(
        order_id="ORD-1001",
        status="shipped",
        items="shoes",
        eta="soon-ish",
        today=_TODAY,
    )
    assert "soon-ish" not in line  # never speak an unparseable raw ETA
    assert "on its way" in line  # the status still renders


def test_render_unknown_status_fails_closed() -> None:
    # A status not in the phrase map must NOT get an invented phrase — speak the raw word.
    line = render_order_status_line(
        order_id="ORD-1001",
        status="returned",
        items="shoes",
        eta=None,
        today=_TODAY,
    )
    assert "returned" in line
    assert "on its way" not in line  # no wrong-status humanization


def test_render_delivered_and_cancelled_omit_eta() -> None:
    # An ETA is meaningless for a delivered (already arrived) or cancelled (won't arrive) order.
    delivered = render_order_status_line(
        order_id="ORD-1003",
        status="delivered",
        items="socks",
        eta="2026-07-01",
        today=_TODAY,
    )
    assert "delivered" in delivered and "expected" not in delivered
    cancelled = render_order_status_line(
        order_id="ORD-9001",
        status="cancelled",
        items="jacket",
        eta="2026-07-20",
        today=_TODAY,
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


def test_owned_cancellable_orders_filters_to_cancellable_and_is_bounded(config_root: Path) -> None:
    # F-16.2 Milestone B: the "cancel all" target universe — only CANCELLABLE orders, bounded.
    # CUST-001's fixture orders are shipped (ORD-1001) + delivered (ORD-1003), neither
    # cancellable; a placed session order IS. So exactly one cancellable, no overflow at limit 5.
    store = _store(config_root)
    _place1(store, "k1", "SKU-GRN-15", "socks", 1, 14.50)  # ORD-9001, processing
    items, has_more = store.owned_cancellable_orders("CUST-001", limit=5)
    assert [c.order_id for c in items] == ["ORD-9001"]  # the shipped/delivered ones excluded
    assert has_more is False


def test_owned_cancellable_orders_flags_overflow_without_truncating_silently(
    config_root: Path,
) -> None:
    # limit = cap; a third cancellable order trips has_more (the resolver asks to narrow).
    store = _store(config_root)
    _place1(store, "k1", "SKU-GRN-15", "socks", 1, 14.50)  # ORD-9001
    _place1(store, "k2", "SKU-BLU-07", "jacket", 1, 60.0)  # ORD-9002
    items, has_more = store.owned_cancellable_orders("CUST-001", limit=1)
    assert len(items) == 1 and has_more is True


def test_owned_cancellable_orders_does_not_materialize_owned_orders(
    config_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(config_root)
    _place1(store, "k1", "SKU-GRN-15", "socks", 1, 14.50)

    def forbidden(_customer_ref: str) -> list:
        raise AssertionError("bounded query must not build the full owned_orders list")

    monkeypatch.setattr(store, "owned_orders", forbidden)
    items, has_more = store.owned_cancellable_orders("CUST-001", limit=1)
    assert [item.order_id for item in items] == ["ORD-9001"]
    assert has_more is False


def test_owned_cancellable_orders_rejects_non_positive_limit(config_root: Path) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        _store(config_root).owned_cancellable_orders("CUST-001", limit=0)


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
            OrderCandidate(
                key="1",
                order_id="ORD-1001",
                summary="2 pairs of shoes",
                total_usd=179.98,
                status="shipped",
            ),
            OrderCandidate(
                key="2",
                order_id="ORD-1003",
                summary="3 pairs of socks",
                total_usd=43.50,
                status="delivered",
            ),
        ]
    )
    assert "You've got 2 orders" in line
    assert "ORD-1001" in line and "on its way" in line  # humanized via _STATUS_PHRASE
    assert "ORD-1003" in line and "delivered" in line
    assert "$" not in line  # no totals, no addresses, no contact — ids + items + status only


def test_render_order_list_line_session_scope_discloses_this_call() -> None:
    # Fix 3: the guest/session scope must NOT sound like a complete account history.
    from agnostic_market.commerce.orders import OrderCandidate

    one = [
        OrderCandidate(
            key="1",
            order_id="ORD-9001",
            summary="2 rain jackets",
            total_usd=258.0,
            status="processing",
        )
    ]
    assert "on this call" in render_order_list_line(one, scope="session")
    assert "You've got" not in render_order_list_line(one, scope="session")
    assert "You've got 1 order" in render_order_list_line(one)  # account default unchanged


def test_session_placed_orders_lists_only_placed_keyed_effective_status(config_root: Path) -> None:
    # Fix 3: the GUEST enumeration view — placed records only (never a fixture/account order),
    # keyed 1..N, effective (cancelled-overlay) status.
    store = _store(config_root)
    _place1(store, "k1", "SKU-BLU-07", "jacket", 2, 129.0)  # ORD-9001
    _place1(store, "k2", "SKU-RED-42", "shoes", 1, 89.99)  # ORD-9002
    placed = store.session_placed_orders()
    assert [c.order_id for c in placed] == ["ORD-9001", "ORD-9002"]
    assert [c.key for c in placed] == ["1", "2"]
    assert not any(c.order_id.startswith("ORD-100") for c in placed)  # NO fixture orders
    store.cancel_order("c1", order_id="ORD-9001")
    assert next(c for c in store.session_placed_orders() if c.order_id == "ORD-9001").status == (
        "cancelled"
    )


def test_placed_candidate_mapper_keeps_listings_consistent(config_root: Path) -> None:
    # The extracted _placed_candidate mapper is the ONE placed->OrderCandidate mapping: the placed
    # tail of owned_orders, the placed rows of actionable_orders, and session_placed_orders must
    # agree on the same (order_id, summary, status) for a given placed record (no drift).
    store = _store(config_root)
    _place1(store, "k1", "SKU-BLU-07", "jacket", 2, 129.0)  # ORD-9001
    sess = {c.order_id: (c.summary, c.status) for c in store.session_placed_orders()}
    owned = {c.order_id: (c.summary, c.status) for c in store.owned_orders("CUST-001")}
    action = {c.order_id: (c.summary, c.status) for c in store.actionable_orders()}
    assert sess["ORD-9001"] == owned["ORD-9001"] == action["ORD-9001"]


def test_render_order_list_line_empty_and_unknown_status() -> None:
    from agnostic_market.commerce.orders import OrderCandidate

    assert render_order_list_line([]) == "I don't see any orders on your account."
    line = render_order_list_line(
        [
            OrderCandidate(
                key="1", order_id="ORD-1", summary="a thing", total_usd=1.0, status="returned"
            )
        ]
    )
    assert "You've got 1 order" in line
    assert "returned" in line  # fail-closed: the raw status word, no invented phrase


# --- Fix 5: session-placed teardown drops caller-ephemeral, keeps durable business state ----


def test_clear_session_placed_drops_view_but_keeps_placement_ledger(config_root: Path) -> None:
    store = _store(config_root)
    placed = _place1(store, "k1", "SKU-BLU-07", "jacket", 1, 129.0)  # ORD-9001
    assert store.is_session_placed("ORD-9001")
    store.clear_session_placed()
    assert not store.is_session_placed("ORD-9001")
    assert store.session_placed_orders() == []
    assert "ORD-9001" not in {candidate.order_id for candidate in store.actionable_orders()}
    assert store.identical_cart_order(list(placed.lines)) is None
    # The committed placement/idempotency ledger remains: a principal rotation must not erase
    # the order or let a stale replay create it again, even though the new caller cannot see it.
    assert store.placed_count == 1
    assert store.order_status("ORD-9001") == "processing"
    assert store.order_summary("ORD-9001") is not None
    replay = _place1(store, "k1", "SKU-BLU-07", "jacket", 1, 129.0)
    assert replay is placed
    assert store.placed_count == 1
    assert not store.is_session_placed("ORD-9001")  # replay never restores prior authority
    # Fixture/account orders are durable-SoR state — untouched.
    assert store.order_status("ORD-1001") == "shipped"
    assert [c.order_id for c in store.owned_orders("CUST-001")] == ["ORD-1001", "ORD-1003"]


def test_clear_session_placed_never_undoes_committed_effects(config_root: Path) -> None:
    # A guest placed orders this call and had one cancelled + a (partial) refund on another. On a
    # principal switch the guest VIEW drops, but the committed cancel/refund records + status
    # overlay are durable business outcomes and must survive (never undo a completed action).
    store = _store(config_root)
    voided = _place1(store, "k1", "SKU-BLU-07", "jacket", 1, 129.0)  # ORD-9001 -> cancelled
    refunded = _place1(store, "k2", "SKU-RED-42", "shoes", 1, 89.99)  # ORD-9002 -> refunded
    store.cancel_order("c1", order_id=voided.order_id)
    store.issue_refund(
        "r1",
        order_id=refunded.order_id,
        amount_usd=20.0,
        destination="original",
        instrument_ref=_ORIGINAL_INSTRUMENT,
    )
    before_cancels, before_refunds = store.cancel_count, store.refund_count
    store.clear_session_placed()
    assert store.cancel_count == before_cancels  # committed cancel record retained
    assert store.refund_count == before_refunds  # committed refund record retained
    assert store.placed_count == 2  # committed placement records retained
    assert store.order_status(voided.order_id) == "cancelled"  # status overlay retained
    assert store.order_status(refunded.order_id) == "processing"
    assert store.refunded_so_far(refunded.order_id) == 20.0
