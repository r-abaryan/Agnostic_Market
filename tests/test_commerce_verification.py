"""Unit tests for the step-up seams: VerificationStore + fake OtpProvider/RiskProvider,
and the refund ledger on OrderStore. Zero network, no graph."""

from __future__ import annotations

from pathlib import Path

import pytest

from agnostic_market.commerce.orders import OrderStore, RefundError, load_orders_fixture
from agnostic_market.commerce.verification import OtpProvider, RiskProvider, VerificationStore
from agnostic_market.dtos.confirmation import refund_required_level
from agnostic_market.dtos.state import CartLine


def _store(config_root: Path) -> OrderStore:
    return OrderStore(load_orders_fixture(config_root, "acme_store"))


# --- the destination -> level FLOOR (§A4b) ---------------------------------------------


@pytest.mark.parametrize(
    ("amount", "destination", "expected"),
    [
        (5.0, "new_instrument", 2),  # small amount to a new card STILL needs L2
        (999.0, "new_instrument", 2),
        (5.0, "new_address", 2),
        (5.0, "original", 1),
        (999.0, "original", 1),  # large to original is L1 (amount gate is separate config)
    ],
)
def test_refund_required_level_is_destination_first(
    amount: float, destination: str, expected: int
) -> None:
    assert refund_required_level(amount, destination) == expected  # type: ignore[arg-type]


# --- VerificationStore: level authority --------------------------------------------------


def test_level_starts_at_l1_and_rises_only_on_correct_committed_otp() -> None:
    otp = OtpProvider()
    store = VerificationStore(otp)
    assert store.current_level() == 1
    assert store.verify_otp("000000") is False
    assert store.current_level() == 1  # a wrong code NEVER raises the level
    assert store.verify_otp("482913") is True
    assert store.current_level() == 2
    # the grant is recorded for dispute defense (§A4a) — method only, no PII/code value
    assert store.grants == [{"method": "otp", "raised_to": 2}]


def test_clear_resets_the_grant() -> None:
    store = VerificationStore(OtpProvider())
    store.verify_otp("482913")
    assert store.current_level() == 2
    store.clear()
    assert store.current_level() == 1
    assert store.grants == []


# --- OtpProvider: idempotent dispatch (S3) ----------------------------------------------


def test_otp_dispatch_is_idempotent_per_attempt() -> None:
    otp = OtpProvider()
    otp.dispatch("attempt-1")
    otp.dispatch("attempt-1")  # a replayed dispatch node
    otp.dispatch("attempt-1")
    assert otp.dispatch_count == 1  # one code sent for the attempt
    otp.dispatch("attempt-2")  # a legitimate re-collect uses a new attempt key
    assert otp.dispatch_count == 2


# --- RiskProvider ------------------------------------------------------------------------


def test_risk_provider_reports_flag() -> None:
    assert RiskProvider(flagged=False).check_sim_swap() is False
    assert RiskProvider(flagged=True).check_sim_swap() is True


# --- the refund ledger: idempotency + cumulative cap + per-intent key --------------------


def test_refund_is_idempotent_per_intent_key(config_root: Path) -> None:
    store = _store(config_root)
    r1 = store.issue_refund("i1", order_id="ORD-1002", amount_usd=50.0, destination="original")
    r2 = store.issue_refund("i1", order_id="ORD-1002", amount_usd=50.0, destination="original")
    assert r1.return_id == r2.return_id  # a replay returns the SAME refund
    assert store.refund_count == 1


def test_second_legitimate_partial_refund_is_not_deduped(config_root: Path) -> None:
    # An order_id-derived key would silently dedupe this; the per-INTENT key must not.
    store = _store(config_root)
    store.issue_refund("intent-1", order_id="ORD-1002", amount_usd=50.0, destination="original")
    store.issue_refund("intent-2", order_id="ORD-1002", amount_usd=40.0, destination="original")
    assert store.refund_count == 2
    assert store.refunded_so_far("ORD-1002") == 90.0


def test_cumulative_cap_refuses_over_refund(config_root: Path) -> None:
    store = _store(config_root)  # ORD-1002 captured $129.00
    store.issue_refund("i1", order_id="ORD-1002", amount_usd=100.0, destination="original")
    with pytest.raises(RefundError):
        # 100 + 40 = 140 > 129 -> refused (the join two partials could otherwise slip)
        store.issue_refund("i2", order_id="ORD-1002", amount_usd=40.0, destination="original")
    assert store.refunded_so_far("ORD-1002") == 100.0  # the refused one did not land


def test_refund_against_unknown_order_is_refused(config_root: Path) -> None:
    store = _store(config_root)
    with pytest.raises(RefundError):
        store.issue_refund("i1", order_id="NOPE-404", amount_usd=1.0, destination="original")


def test_refund_against_a_cancelled_order_is_refused(config_root: Path) -> None:
    # The void already reversed the charge — a refund on top returns the money twice
    # (found live 2026-07-10: cancel ORD-9001, then an under-threshold refund would land).
    store = _store(config_root)
    store.cancel_order("ck-1", order_id="ORD-1002")
    with pytest.raises(RefundError):
        store.issue_refund("i1", order_id="ORD-1002", amount_usd=50.0, destination="original")
    assert store.refund_count == 0


def test_refund_can_target_a_just_placed_order(config_root: Path) -> None:
    store = _store(config_root)
    placed = store.place_cart(
        "k1", lines=[CartLine(sku="SKU-BLU-07", name="rain jacket", price_usd=129.0, quantity=1)],
        total_usd=129.0)
    rec = store.issue_refund(
        "i1", order_id=placed.order_id, amount_usd=129.0, destination="original"
    )
    assert rec.order_id == placed.order_id
    assert store.refunded_so_far(placed.order_id) == 129.0
