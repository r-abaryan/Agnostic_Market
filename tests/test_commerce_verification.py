"""Unit tests for the step-up seams: VerificationStore + fake OtpProvider/RiskProvider,
and the refund ledger on OrderStore. Zero network, no graph."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from agnostic_market.commerce.orders import OrderStore, RefundError, load_orders_fixture
from agnostic_market.commerce.verification import (
    OtpProvider,
    RiskProvider,
    VerificationStore,
    load_verification_fixture,
)
from agnostic_market.config.loader import ConfigError
from agnostic_market.dtos.confirmation import (
    ISSUE_REFUND_POLICY,
    refund_required_level,
    validate_confirmation_rendering,
)
from agnostic_market.dtos.state import CartLine

_TEST_OTP = "482913"
_ORIGINAL_INSTRUMENT = "original payment method"


def _store(config_root: Path) -> OrderStore:
    return OrderStore("acme_store", load_orders_fixture(config_root, "acme_store").orders)


def test_verification_fixture_loads_the_temporary_merchant_code(config_root: Path) -> None:
    fixture = load_verification_fixture(config_root, "acme_store")
    assert fixture.otp_code == _TEST_OTP


def test_verification_fixture_rejects_a_non_six_digit_code(tmp_path: Path) -> None:
    fixture_dir = tmp_path / "fixtures" / "verification"
    fixture_dir.mkdir(parents=True)
    (fixture_dir / "broken.yaml").write_text('otp_code: "12345"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="failed validation"):
        load_verification_fixture(tmp_path, "broken")


def test_concurrent_otp_replay_dispatches_one_attempt() -> None:
    class _SlowMissSet(set[str]):
        def __init__(self) -> None:
            super().__init__()
            self.add_calls = 0

        def __contains__(self, value: object) -> bool:
            present = super().__contains__(value)
            if not present:
                time.sleep(0.05)
            return present

        def add(self, value: str) -> None:
            self.add_calls += 1
            super().add(value)

    provider = OtpProvider("acme_store", valid_code=_TEST_OTP)
    dispatched = _SlowMissSet()
    provider._dispatched = dispatched

    with ThreadPoolExecutor(max_workers=2) as executor:
        tuple(executor.map(provider.dispatch, ("same-attempt", "same-attempt")))

    assert dispatched.add_calls == 1
    assert provider.dispatch_count == 1


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


def test_confirmation_rendering_rejects_a_missing_declared_value() -> None:
    rendered = {
        "total_amount": "$129.00",
        "new_payment_instrument_ref": "card ending 4471",
    }
    with pytest.raises(ValueError, match=r"cannot render declared fields: \['order_id'\]"):
        validate_confirmation_rendering(
            ISSUE_REFUND_POLICY,
            rendered,
            "a $129.00 refund to card ending 4471",
        )


def test_confirmation_rendering_rejects_a_declared_value_not_spoken() -> None:
    rendered = {
        "order_id": "ORD-1002",
        "total_amount": "$129.00",
        "new_payment_instrument_ref": "card ending 4471",
    }
    with pytest.raises(ValueError, match=r"does not speak declared fields: \['order_id'\]"):
        validate_confirmation_rendering(
            ISSUE_REFUND_POLICY,
            rendered,
            "a $129.00 refund to card ending 4471",
        )


# --- VerificationStore: level authority --------------------------------------------------


def test_level_starts_at_l1_and_rises_only_on_correct_committed_otp() -> None:
    otp = OtpProvider("acme_store", valid_code=_TEST_OTP)
    store = VerificationStore(otp)
    assert store.current_level() == 1
    assert store.verify_otp("000000") is False
    assert store.current_level() == 1  # a wrong code NEVER raises the level
    assert store.verify_otp(_TEST_OTP) is True
    assert store.current_level() == 2
    # the grant is recorded for dispute defense (§A4a) — method only, no PII/code value
    assert len(store.grants) == 1
    assert store.grants[0].method == "otp"
    assert store.grants[0].raised_to == 2
    assert store.grants[0].proof_id


def test_spoken_digit_code_verifies() -> None:
    # Live call #12 F-12.2: the CORRECT code arrived as words ("four eight two nine one
    # three"), failed the literal compare, and exhausted a legitimate caller to a human.
    # verify_otp digit-normalizes the committed spoken answer; the compare stays EXACT.
    store = VerificationStore(OtpProvider("acme_store", valid_code=_TEST_OTP))
    assert store.verify_otp("It should be four eight two nine one three.") is True
    assert store.current_level() == 2


def test_spoken_digit_code_stays_exact_no_overmatch() -> None:
    store = VerificationStore(OtpProvider("acme_store", valid_code=_TEST_OTP))
    assert store.verify_otp("one two three four five six") is False  # wrong code, spoken
    assert store.verify_otp("four eight two nine one") is False  # too short
    assert store.verify_otp("oh four eight two nine one three") is False  # extra digit
    assert store.current_level() == 1  # nothing above raised anything


def test_clear_resets_the_grant() -> None:
    store = VerificationStore(OtpProvider("acme_store", valid_code=_TEST_OTP))
    store.verify_otp(_TEST_OTP)
    assert store.current_level() == 2
    store.clear()
    assert store.current_level() == 1
    assert store.grants == []


# --- OtpProvider: idempotent dispatch (S3) ----------------------------------------------


def test_otp_dispatch_is_idempotent_per_attempt() -> None:
    otp = OtpProvider("acme_store", valid_code=_TEST_OTP)
    otp.dispatch("attempt-1")
    otp.dispatch("attempt-1")  # a replayed dispatch node
    otp.dispatch("attempt-1")
    assert otp.dispatch_count == 1  # one code sent for the attempt
    otp.dispatch("attempt-2")  # a legitimate re-collect uses a new attempt key
    assert otp.dispatch_count == 2


# --- RiskProvider ------------------------------------------------------------------------


def test_risk_provider_reports_flag() -> None:
    assert RiskProvider("acme_store", flagged=False).check_sim_swap() is False
    assert RiskProvider("acme_store", flagged=True).check_sim_swap() is True


# --- the refund ledger: idempotency + cumulative cap + per-intent key --------------------


def test_refund_is_idempotent_per_intent_key(config_root: Path) -> None:
    store = _store(config_root)
    r1 = store.issue_refund(
        "i1",
        order_id="ORD-1002",
        amount_usd=50.0,
        destination="original",
        instrument_ref=_ORIGINAL_INSTRUMENT,
    )
    r2 = store.issue_refund(
        "i1",
        order_id="ORD-1002",
        amount_usd=50.0,
        destination="original",
        instrument_ref=_ORIGINAL_INSTRUMENT,
    )
    assert r1.refund_id == r2.refund_id  # a replay returns the SAME refund
    assert store.refund_count == 1


def test_second_legitimate_partial_refund_is_not_deduped(config_root: Path) -> None:
    # An order_id-derived key would silently dedupe this; the per-INTENT key must not.
    store = _store(config_root)
    store.issue_refund(
        "intent-1",
        order_id="ORD-1002",
        amount_usd=50.0,
        destination="original",
        instrument_ref=_ORIGINAL_INSTRUMENT,
    )
    store.issue_refund(
        "intent-2",
        order_id="ORD-1002",
        amount_usd=40.0,
        destination="original",
        instrument_ref=_ORIGINAL_INSTRUMENT,
    )
    assert store.refund_count == 2
    assert store.refunded_so_far("ORD-1002") == 90.0


def test_cumulative_cap_refuses_over_refund(config_root: Path) -> None:
    store = _store(config_root)  # ORD-1002 captured $129.00
    store.issue_refund(
        "i1",
        order_id="ORD-1002",
        amount_usd=100.0,
        destination="original",
        instrument_ref=_ORIGINAL_INSTRUMENT,
    )
    with pytest.raises(RefundError):
        # 100 + 40 = 140 > 129 -> refused (the join two partials could otherwise slip)
        store.issue_refund(
            "i2",
            order_id="ORD-1002",
            amount_usd=40.0,
            destination="original",
            instrument_ref=_ORIGINAL_INSTRUMENT,
        )
    assert store.refunded_so_far("ORD-1002") == 100.0  # the refused one did not land


def test_refund_against_unknown_order_is_refused(config_root: Path) -> None:
    store = _store(config_root)
    with pytest.raises(RefundError):
        store.issue_refund(
            "i1",
            order_id="NOPE-404",
            amount_usd=1.0,
            destination="original",
            instrument_ref=_ORIGINAL_INSTRUMENT,
        )


def test_refund_against_a_cancelled_order_is_refused(config_root: Path) -> None:
    # The void already reversed the charge — a refund on top returns the money twice
    # (found live 2026-07-10: cancel ORD-9001, then an under-threshold refund would land).
    store = _store(config_root)
    store.cancel_order("ck-1", order_id="ORD-1002")
    with pytest.raises(RefundError):
        store.issue_refund(
            "i1",
            order_id="ORD-1002",
            amount_usd=50.0,
            destination="original",
            instrument_ref=_ORIGINAL_INSTRUMENT,
        )
    assert store.refund_count == 0


def test_refund_can_target_a_just_placed_order(config_root: Path) -> None:
    store = _store(config_root)
    placed = store.place_cart(
        "k1",
        lines=[CartLine(sku="SKU-BLU-07", name="rain jacket", price_usd=129.0, quantity=1)],
        total_usd=129.0,
    )
    rec = store.issue_refund(
        "i1",
        order_id=placed.order_id,
        amount_usd=129.0,
        destination="original",
        instrument_ref=_ORIGINAL_INSTRUMENT,
    )
    assert rec.order_id == placed.order_id
    assert store.refunded_so_far(placed.order_id) == 129.0
