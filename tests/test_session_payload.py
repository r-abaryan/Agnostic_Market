"""Validated snapshots for encrypted restorable session state."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agnostic_market.commerce.cart import CartSessionState, CartStore
from agnostic_market.commerce.orders import GuestOrderScope, RecentOrderContext
from agnostic_market.dtos.state import CartLine
from agnostic_market.durability.session_payload import (
    CartMutationSessionOperationResult,
    DurableSessionPayload,
    SessionOperationReceiptPayload,
)


def test_payload_round_trip_preserves_cart_receipts_and_bounded_context() -> None:
    cart = CartStore()
    record = cart.apply_confirmed_mutation(
        "mutation-1",
        operation="add",
        sku="SKU-1",
        name="Trail Jacket",
        price_usd="79.00",
        quantity=1,
        pre_confirm_quantity=0,
    )
    recent = RecentOrderContext(max_refs=2)
    recent.record(
        ("ord-1", "ord-2", "ord-3"),
        operation="cancel",
        focused_order_ref="ord-3",
        outcomes=(("ord-3", "cancelled"),),
    )
    payload = DurableSessionPayload(
        cart=cart.export_state(),
        recent_orders=recent.snapshot(),
        guest_order_refs=("ORD-3",),
    )

    restored_payload = DurableSessionPayload.from_bytes(payload.to_bytes())
    restored_cart = CartStore.from_state(restored_payload.cart)
    restored_recent = RecentOrderContext.from_snapshot(
        restored_payload.recent_orders,
        max_refs=2,
    )
    restored_guest = GuestOrderScope.from_refs(
        tenant_id="acme_store",
        session_id="session-1",
        order_refs=restored_payload.guest_order_refs,
    )

    assert restored_payload == payload
    assert restored_cart.view() == (
        CartLine(sku="SKU-1", name="Trail Jacket", price_usd="79.00", quantity=1),
    )
    assert (
        restored_cart.mutation_receipt(
            "mutation-1",
            operation="add",
            sku="SKU-1",
            name="Trail Jacket",
            price_usd="79.00",
            quantity=1,
            pre_confirm_quantity=0,
        ).record
        == record
    )
    assert restored_recent.snapshot() == recent.snapshot()
    assert restored_guest.order_refs == ("ORD-3",)

    operation_receipt = SessionOperationReceiptPayload(
        operation_id="mutation-1",
        request_fingerprint="a" * 64,
        result=CartMutationSessionOperationResult(record=record),
    )
    assert (
        SessionOperationReceiptPayload.from_bytes(operation_receipt.to_bytes()) == operation_receipt
    )


def test_payload_rejects_duplicate_or_noncanonical_authority_references() -> None:
    with pytest.raises(ValidationError, match="duplicate guest orders"):
        DurableSessionPayload(guest_order_refs=("ORD-1", "ORD-1"))
    with pytest.raises(ValidationError, match="canonical"):
        DurableSessionPayload(guest_order_refs=("ord-1",))
    with pytest.raises(ValidationError, match="duplicate SKUs"):
        CartSessionState(
            lines=(
                CartLine(sku="SKU-1", name="A", price_usd="1.00", quantity=1),
                CartLine(sku="SKU-1", name="B", price_usd="2.00", quantity=1),
            )
        )


def test_restore_rejects_recent_context_that_exceeds_the_current_policy_bound() -> None:
    recent = RecentOrderContext(max_refs=2)
    recent.record(("ORD-1", "ORD-2"), operation="read")

    with pytest.raises(ValueError, match="configured bound"):
        RecentOrderContext.from_snapshot(recent.snapshot(), max_refs=1)
