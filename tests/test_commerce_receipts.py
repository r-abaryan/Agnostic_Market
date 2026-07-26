"""Authoritative, read-only idempotency receipt contracts."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from agnostic_market.commerce.orders import (
    CancelError,
    OrderStore,
    PlacementError,
    RefundError,
    ReturnError,
    load_orders_fixture,
)
from agnostic_market.commerce.profile import ProfileError, ProfileStore, load_profile_fixture
from agnostic_market.commerce.receipts import (
    CommittedReceipt,
    IndeterminateReceipt,
    NotCommittedReceipt,
)
from agnostic_market.dtos.state import CartLine

_OWNER = "CUST-001"
_OTHER_CUSTOMER = "CUST-002"
_ORIGINAL_INSTRUMENT = "original payment method"


def _orders(config_root: Path) -> OrderStore:
    return OrderStore(load_orders_fixture(config_root, "acme_store"))


def _profiles(config_root: Path) -> ProfileStore:
    return ProfileStore(load_profile_fixture(config_root, "acme_store"))


def _line(*, quantity: int = 1) -> CartLine:
    return CartLine(
        sku="SKU-BLU-07",
        name="waterproof rain jacket",
        price_usd=129.0,
        quantity=quantity,
    )


def test_receipt_result_shapes_are_closed_and_record_safe() -> None:
    assert IndeterminateReceipt(reason="pending").kind == "indeterminate"
    assert IndeterminateReceipt(reason="unavailable").kind == "indeterminate"
    with pytest.raises(ValidationError):
        IndeterminateReceipt.model_validate(
            {"kind": "indeterminate", "reason": "free-form backend failure"}
        )
    with pytest.raises(ValidationError):
        NotCommittedReceipt.model_validate({"kind": "not_committed", "record": "must not leak"})
    with pytest.raises(ValidationError):
        CommittedReceipt[str].model_validate({"kind": "committed"})


def test_placement_receipt_is_exact_and_does_not_restore_principal_visibility(
    config_root: Path,
) -> None:
    store = _orders(config_root)
    lines = [_line()]
    assert store.placement_receipt("place-key", lines=lines, total_usd=129.0).kind == (
        "not_committed"
    )
    placed = store.place_cart("place-key", lines=lines, total_usd=129.0)
    assert placed.order_id == "ORD-9001"
    store.clear_session_placed()

    receipt = store.placement_receipt("place-key", lines=lines, total_usd=129.0)
    assert isinstance(receipt, CommittedReceipt)
    assert receipt.record is placed
    assert not store.is_session_placed(placed.order_id)

    conflict = store.placement_receipt(
        "place-key",
        lines=[_line(quantity=2)],
        total_usd=258.0,
    )
    assert conflict == IndeterminateReceipt(reason="key_conflict")
    with pytest.raises(PlacementError, match="different parameters"):
        store.place_cart("place-key", lines=[_line(quantity=2)], total_usd=258.0)
    assert store.placed_count == 1
    assert not store.is_session_placed(placed.order_id)


def test_refund_receipt_proves_the_masked_instrument_and_rejects_key_conflict(
    config_root: Path,
) -> None:
    store = _orders(config_root)
    assert (
        store.refund_receipt(
            "refund-key",
            order_id="ORD-1002",
            amount_usd=40.0,
            destination="original",
            instrument_ref=_ORIGINAL_INSTRUMENT,
        ).kind
        == "not_committed"
    )
    record = store.issue_refund(
        "refund-key",
        order_id="ORD-1002",
        amount_usd=40.0,
        destination="original",
        instrument_ref=_ORIGINAL_INSTRUMENT,
    )
    assert record.refund_id == "R-7001"

    receipt = store.refund_receipt(
        "refund-key",
        order_id="ord-1002",
        amount_usd=40.0,
        destination="original",
        instrument_ref=f" {_ORIGINAL_INSTRUMENT} ",
    )
    assert isinstance(receipt, CommittedReceipt)
    assert receipt.record is record
    assert receipt.record.instrument_ref == _ORIGINAL_INSTRUMENT

    conflict = store.refund_receipt(
        "refund-key",
        order_id="ORD-1002",
        amount_usd=40.0,
        destination="original",
        instrument_ref="card ending 9999",
    )
    assert conflict == IndeterminateReceipt(reason="key_conflict")
    with pytest.raises(RefundError, match="different parameters"):
        store.issue_refund(
            "refund-key",
            order_id="ORD-1002",
            amount_usd=40.0,
            destination="original",
            instrument_ref="card ending 9999",
        )
    assert store.refund_count == 1
    assert store.refunded_so_far("ORD-1002") == 40.0


def test_cancel_receipt_is_per_target_and_cross_operation_keys_are_independent(
    config_root: Path,
) -> None:
    store = _orders(config_root)
    line = _line()
    assert store.cancel_receipt("shared-key", order_id="ORD-1002").kind == "not_committed"
    placed = store.place_cart("shared-key", lines=[line], total_usd=129.0)
    cancelled = store.cancel_order("shared-key", order_id="ORD-1002")

    placement = store.placement_receipt("shared-key", lines=[line], total_usd=129.0)
    cancellation = store.cancel_receipt("shared-key", order_id="ord-1002")
    assert isinstance(placement, CommittedReceipt)
    assert placement.record is placed
    assert isinstance(cancellation, CommittedReceipt)
    assert cancellation.record is cancelled

    conflict = store.cancel_receipt("shared-key", order_id=placed.order_id)
    assert conflict == IndeterminateReceipt(reason="key_conflict")
    with pytest.raises(CancelError, match="different parameters"):
        store.cancel_order("shared-key", order_id=placed.order_id)
    assert store.cancel_count == 1
    assert store.order_status(placed.order_id) == "processing"


def test_return_receipt_distinguishes_absent_exact_and_conflicting_intents(
    config_root: Path,
) -> None:
    store = _orders(config_root)
    assert (
        store.return_receipt(
            "return-key",
            order_id="ORD-1001",
            refund_due_usd=100.0,
            destination="original",
        ).kind
        == "not_committed"
    )
    record = store.create_return(
        "return-key",
        order_id="ORD-1001",
        refund_due_usd=100.0,
        destination="original",
    )
    assert record.rma_id == "RMA-3001"
    assert record.summary == "2 pairs of trail running shoes"
    receipt = store.return_receipt(
        "return-key",
        order_id="ord-1001",
        refund_due_usd=100.0,
        destination="original",
    )
    assert isinstance(receipt, CommittedReceipt)
    assert receipt.record is record

    conflict = store.return_receipt(
        "return-key",
        order_id="ORD-1001",
        refund_due_usd=99.0,
        destination="original",
    )
    assert conflict == IndeterminateReceipt(reason="key_conflict")
    with pytest.raises(ReturnError, match="different parameters"):
        store.create_return(
            "return-key",
            order_id="ORD-1001",
            refund_due_usd=99.0,
            destination="original",
        )
    assert store.return_count == 1


def test_profile_receipt_is_customer_scoped_neutral_and_non_mutating(
    config_root: Path,
) -> None:
    store = _profiles(config_root)
    record = store.update_profile(
        "profile-key",
        customer_ref=_OWNER,
        field="address",
        new_value="7 Elm Street",
    )

    receipt = store.profile_change_receipt(
        "profile-key",
        customer_ref=_OWNER,
        field="address",
        new_value=" 7 Elm Street ",
    )
    assert isinstance(receipt, CommittedReceipt)
    assert receipt.record is record
    assert store.address_on_file(_OWNER) == "7 Elm Street"

    other_customer = store.profile_change_receipt(
        "profile-key",
        customer_ref=_OTHER_CUSTOMER,
        field="address",
        new_value="7 Elm Street",
    )
    assert other_customer == NotCommittedReceipt()
    assert not hasattr(other_customer, "record")

    conflict = store.profile_change_receipt(
        "profile-key",
        customer_ref=_OWNER,
        field="contact",
        new_value="number ending 9999",
    )
    assert conflict == IndeterminateReceipt(reason="key_conflict")
    assert not hasattr(conflict, "record")
    with pytest.raises(ProfileError, match="different parameters"):
        store.update_profile(
            "profile-key",
            customer_ref=_OWNER,
            field="contact",
            new_value="number ending 9999",
        )
    assert store.change_count == 1
    assert store.address_on_file(_OWNER) == "7 Elm Street"
