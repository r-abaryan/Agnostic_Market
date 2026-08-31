"""Unit tests for the step-up seams: VerificationStore + fake OtpProvider/RiskProvider,
and the refund ledger on OrderStore. Zero network, no graph."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from agnostic_market.commerce.orders import OrderStore, RefundError, load_orders_fixture
from agnostic_market.commerce.verification import (
    OrderRiskSubject,
    OtpChallenge,
    OtpChallengeConflictError,
    OtpChallengeRequest,
    OtpProvider,
    OtpVerification,
    OtpVerificationStatus,
    RiskDecision,
    RiskProvider,
    VerificationContractError,
    VerificationStore,
    VerificationSubject,
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
_OTHER_OTP = "739204"
_FACTOR_REFS = {"CUST-001": "FACTOR-001", "CUST-002": "FACTOR-002"}
_CODES = {"FACTOR-001": _TEST_OTP, "FACTOR-002": _OTHER_OTP}
_ORIGINAL_INSTRUMENT = "original payment method"


def _store(config_root: Path) -> OrderStore:
    return OrderStore("acme_store", load_orders_fixture(config_root, "acme_store").orders)


def _provider(*, clock=None, proof_ttl_seconds: float = 300) -> OtpProvider:
    kwargs = {"clock": clock} if clock is not None else {}
    return OtpProvider(
        "acme_store",
        codes_by_factor_ref=_CODES,
        challenge_ttl_seconds=300,
        proof_ttl_seconds=proof_ttl_seconds,
        **kwargs,
    )


def _subject(
    *,
    customer_ref: str = "CUST-001",
    session_id: str = "session-1",
    purpose: str = "identity",
) -> VerificationSubject:
    return VerificationSubject(
        tenant_id="acme_store",
        session_id=session_id,
        customer_ref=customer_ref,
        factor_ref=_FACTOR_REFS[customer_ref],
        purpose=purpose,
    )


def _request(
    *,
    subject: VerificationSubject | None = None,
    key: str = "dispatch-1",
    max_attempts: int = 3,
) -> OtpChallengeRequest:
    return OtpChallengeRequest(
        subject=subject or _subject(),
        dispatch_idempotency_key=key,
        max_attempts=max_attempts,
    )


async def _store_and_challenge(*, max_attempts: int = 3):
    store = VerificationStore(_provider(), session_id="session-1")
    subject = store.subject(
        customer_ref="CUST-001",
        factor_ref="FACTOR-001",
        purpose="identity",
    )
    challenge = await store.dispatch_otp(
        subject=subject,
        dispatch_idempotency_key="dispatch-1",
        max_attempts=max_attempts,
    )
    return store, challenge


def test_verification_store_rejects_unproven_level_hydration() -> None:
    with pytest.raises(TypeError, match="initial_level"):
        VerificationStore(
            _provider(),
            session_id="session-1",
            initial_level=2,
        )


def test_verification_fixture_loads_factor_specific_codes(config_root: Path) -> None:
    fixture = load_verification_fixture(config_root, "acme_store")
    assert fixture.otp_codes_by_factor_ref == _CODES
    assert fixture.challenge_ttl_seconds == 300
    assert fixture.proof_ttl_seconds == 300


def test_verification_fixture_rejects_a_non_six_digit_code(tmp_path: Path) -> None:
    fixture_dir = tmp_path / "fixtures" / "verification"
    fixture_dir.mkdir(parents=True)
    (fixture_dir / "broken.yaml").write_text(
        'otp_codes_by_factor_ref: {CUST-001: "12345"}\nchallenge_ttl_seconds: 300\n'
        "proof_ttl_seconds: 300\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="failed validation"):
        load_verification_fixture(tmp_path, "broken")


def test_verification_fixture_rejects_an_infinite_challenge_ttl(tmp_path: Path) -> None:
    fixture_dir = tmp_path / "fixtures" / "verification"
    fixture_dir.mkdir(parents=True)
    (fixture_dir / "broken.yaml").write_text(
        'otp_codes_by_factor_ref: {CUST-001: "482913"}\nchallenge_ttl_seconds: .inf\n'
        "proof_ttl_seconds: 300\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="failed validation"):
        load_verification_fixture(tmp_path, "broken")


def test_challenge_contract_rejects_an_infinite_expiry() -> None:
    with pytest.raises(ValueError):
        OtpChallenge(challenge_id="challenge-1", expires_at_epoch=float("inf"))


def test_verified_outcome_rejects_an_infinite_timestamp() -> None:
    with pytest.raises(ValueError):
        OtpVerification(
            challenge_id="challenge-1",
            status=OtpVerificationStatus.VERIFIED,
            subject=_subject(),
            verified_at_epoch=float("inf"),
            proof_expires_at_epoch=1_000,
        )


def test_verified_outcome_rejects_a_nonpositive_proof_lifetime() -> None:
    with pytest.raises(ValueError, match="expiry must follow"):
        OtpVerification(
            challenge_id="challenge-1",
            status=OtpVerificationStatus.VERIFIED,
            subject=_subject(),
            verified_at_epoch=100,
            proof_expires_at_epoch=100,
        )


def test_fixture_provider_rejects_an_infinite_challenge_ttl() -> None:
    with pytest.raises(ValueError):
        OtpProvider(
            "acme_store",
            codes_by_factor_ref=_CODES,
            challenge_ttl_seconds=float("inf"),
            proof_ttl_seconds=300,
        )


def test_fixture_provider_rejects_an_infinite_proof_ttl() -> None:
    with pytest.raises(ValueError):
        OtpProvider(
            "acme_store",
            codes_by_factor_ref=_CODES,
            challenge_ttl_seconds=300,
            proof_ttl_seconds=float("inf"),
        )


async def test_concurrent_otp_replay_dispatches_one_attempt() -> None:
    provider = _provider()
    request = _request()

    challenges = await asyncio.gather(provider.dispatch(request), provider.dispatch(request))

    assert challenges[0] == challenges[1]
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


async def test_level_starts_at_l1_and_rises_only_on_correct_committed_otp() -> None:
    store, challenge = await _store_and_challenge()
    identity_subject = store.subject(
        customer_ref="CUST-001",
        factor_ref="FACTOR-001",
        purpose="identity",
    )
    assert store.current_level() == 1
    assert (
        await store.verify_otp(challenge.challenge_id, "000000") is OtpVerificationStatus.MISMATCHED
    )
    assert store.current_level() == 1  # a wrong code NEVER raises the level
    assert (
        await store.verify_otp(challenge.challenge_id, _TEST_OTP) is OtpVerificationStatus.VERIFIED
    )
    assert store.current_level() == 2
    assert store.challenge_satisfies(challenge.challenge_id, identity_subject, 2)
    assert not store.challenge_satisfies(
        challenge.challenge_id,
        identity_subject.model_copy(update={"purpose": "profile"}),
        2,
    )
    # the grant is recorded for dispute defense (§A4a) — method only, no PII/code value
    assert len(store.grants) == 1
    assert store.grants[0].method == "otp"
    assert store.grants[0].raised_to == 2
    assert store.grants[0].proof_id


async def test_fresh_same_factor_proof_authorizes_another_protected_purpose() -> None:
    store, challenge = await _store_and_challenge()
    assert (
        await store.verify_otp(challenge.challenge_id, _TEST_OTP) is OtpVerificationStatus.VERIFIED
    )

    profile_subject = store.subject(
        customer_ref="CUST-001",
        factor_ref="FACTOR-001",
        purpose="profile",
    )

    assert store.authorization_satisfies(profile_subject, 2)
    assert not store.challenge_satisfies(challenge.challenge_id, profile_subject, 2)


@pytest.mark.parametrize(
    "updates",
    [
        {"tenant_id": "other_store"},
        {"session_id": "other-session"},
        {"customer_ref": "CUST-002"},
        {"factor_ref": "FACTOR-002"},
    ],
)
async def test_reusable_proof_cannot_cross_an_authority_dimension(
    updates: dict[str, str],
) -> None:
    store, challenge = await _store_and_challenge()
    assert (
        await store.verify_otp(challenge.challenge_id, _TEST_OTP) is OtpVerificationStatus.VERIFIED
    )
    subject = store.subject(
        customer_ref="CUST-001",
        factor_ref="FACTOR-001",
        purpose="profile",
    )

    assert not store.authorization_satisfies(subject.model_copy(update=updates), 2)


async def test_expired_proof_authorizes_nothing() -> None:
    now = [100.0]
    provider = _provider(clock=lambda: now[0], proof_ttl_seconds=30)
    store = VerificationStore(provider, session_id="session-1", clock=lambda: now[0])
    subject = store.subject(
        customer_ref="CUST-001",
        factor_ref="FACTOR-001",
        purpose="identity",
    )
    challenge = await store.dispatch_otp(
        subject=subject,
        dispatch_idempotency_key="expiring-proof",
        max_attempts=3,
    )
    assert (
        await store.verify_otp(challenge.challenge_id, _TEST_OTP) is OtpVerificationStatus.VERIFIED
    )
    proof = store.proof_for_challenge(challenge.challenge_id)
    assert proof is not None

    now[0] = proof.expires_at_epoch

    assert store.current_level() == 1
    assert not store.authorization_satisfies(subject, 2)
    assert not store.challenge_satisfies(challenge.challenge_id, subject, 2)
    assert store.proof_for_challenge(challenge.challenge_id) is None
    assert store.grants == []
    assert (
        await store.verify_otp(challenge.challenge_id, _TEST_OTP) is OtpVerificationStatus.EXPIRED
    )


async def test_spoken_digit_code_verifies() -> None:
    # Live call #12 F-12.2: the CORRECT code arrived as words ("four eight two nine one
    # three"), failed the literal compare, and exhausted a legitimate caller to a human.
    # verify_otp digit-normalizes the committed spoken answer; the compare stays EXACT.
    store, challenge = await _store_and_challenge()
    assert (
        await store.verify_otp(
            challenge.challenge_id,
            "It should be four eight two nine one three.",
        )
        is OtpVerificationStatus.VERIFIED
    )
    assert store.current_level() == 2


async def test_spoken_digit_code_stays_exact_no_overmatch() -> None:
    store, challenge = await _store_and_challenge(max_attempts=4)
    assert (
        await store.verify_otp(challenge.challenge_id, "one two three four five six")
        is OtpVerificationStatus.MISMATCHED
    )
    assert (
        await store.verify_otp(challenge.challenge_id, "four eight two nine one")
        is OtpVerificationStatus.MISMATCHED
    )
    assert (
        await store.verify_otp(challenge.challenge_id, "oh four eight two nine one three")
        is OtpVerificationStatus.MISMATCHED
    )
    assert store.current_level() == 1  # nothing above raised anything


async def test_clear_resets_the_grant() -> None:
    store, challenge = await _store_and_challenge()
    await store.verify_otp(challenge.challenge_id, _TEST_OTP)
    assert store.current_level() == 2
    await store.clear()
    assert store.current_level() == 1
    assert store.grants == []


async def test_clear_discards_only_that_sessions_provider_challenges() -> None:
    provider = _provider()
    first = VerificationStore(provider, session_id="session-1")
    second = VerificationStore(provider, session_id="session-2")
    first_subject = first.subject(
        customer_ref="CUST-001",
        factor_ref="FACTOR-001",
        purpose="identity",
    )
    second_subject = second.subject(
        customer_ref="CUST-002",
        factor_ref="FACTOR-002",
        purpose="identity",
    )
    first_challenge = await first.dispatch_otp(
        subject=first_subject,
        dispatch_idempotency_key="session-1-attempt",
        max_attempts=3,
    )
    second_challenge = await second.dispatch_otp(
        subject=second_subject,
        dispatch_idempotency_key="session-2-attempt",
        max_attempts=3,
    )

    await first.clear()

    assert provider.active_challenge_count == 1
    assert (
        await provider.verify(first_challenge.challenge_id, _TEST_OTP)
    ).status is OtpVerificationStatus.UNKNOWN
    assert (
        await second.verify_otp(second_challenge.challenge_id, _OTHER_OTP)
        is OtpVerificationStatus.VERIFIED
    )


async def test_provider_rejects_cross_session_challenge_cleanup_atomically() -> None:
    provider = _provider()
    first = await provider.dispatch(_request(key="session-1-attempt"))
    second = await provider.dispatch(
        _request(
            subject=_subject(customer_ref="CUST-002", session_id="session-2"),
            key="session-2-attempt",
        )
    )

    with pytest.raises(VerificationContractError, match="session boundary"):
        await provider.retain_only("session-1", (first.challenge_id, second.challenge_id))

    assert provider.active_challenge_count == 2


async def test_session_cleanup_discards_a_provider_commit_missing_from_local_state() -> None:
    provider = _provider()
    store = VerificationStore(provider, session_id="session-1")
    await provider.dispatch(_request(key="orphaned-provider-commit"))

    await store.clear()

    assert provider.active_challenge_count == 0


async def test_provider_rejects_retaining_an_unknown_challenge() -> None:
    provider = _provider()
    challenge = await provider.dispatch(_request())

    with pytest.raises(VerificationContractError, match="unknown challenge"):
        await provider.retain_only("session-1", (challenge.challenge_id, "missing"))

    assert provider.active_challenge_count == 1


# --- OtpProvider: idempotent dispatch (S3) ----------------------------------------------


async def test_otp_dispatch_is_idempotent_per_attempt() -> None:
    otp = _provider()
    first = _request(key="attempt-1")
    await otp.dispatch(first)
    await otp.dispatch(first)
    await otp.dispatch(first)
    assert otp.dispatch_count == 1  # one code sent for the attempt
    await otp.dispatch(_request(key="attempt-2"))
    assert otp.dispatch_count == 2


async def test_dispatch_key_reuse_with_a_different_subject_is_rejected() -> None:
    otp = _provider()
    await otp.dispatch(_request(key="shared-key"))

    with pytest.raises(OtpChallengeConflictError):
        await otp.dispatch(
            _request(
                key="shared-key",
                subject=_subject(customer_ref="CUST-002"),
            )
        )


async def test_code_for_another_factor_cannot_verify_the_challenge() -> None:
    store, challenge = await _store_and_challenge(max_attempts=2)

    assert (
        await store.verify_otp(challenge.challenge_id, _OTHER_OTP)
        is OtpVerificationStatus.MISMATCHED
    )
    assert store.current_level() == 1


async def test_provider_attempt_budget_is_authoritative() -> None:
    store, challenge = await _store_and_challenge(max_attempts=2)

    assert (
        await store.verify_otp(challenge.challenge_id, "000000") is OtpVerificationStatus.MISMATCHED
    )
    assert (
        await store.verify_otp(challenge.challenge_id, "000000") is OtpVerificationStatus.EXHAUSTED
    )
    assert (
        await store.verify_otp(challenge.challenge_id, _TEST_OTP) is OtpVerificationStatus.EXHAUSTED
    )


async def test_verified_outcome_is_stable_after_challenge_expiry() -> None:
    now = [100.0]
    provider = _provider(clock=lambda: now[0])
    challenge = await provider.dispatch(_request())

    first = await provider.verify(challenge.challenge_id, _TEST_OTP)
    now[0] = challenge.expires_at_epoch + 1
    replay = await provider.verify(challenge.challenge_id, _TEST_OTP)

    assert first.status is OtpVerificationStatus.VERIFIED
    assert replay == first


async def test_verified_challenge_rejects_a_different_code_on_replay() -> None:
    provider = _provider()
    challenge = await provider.dispatch(_request())

    assert (
        await provider.verify(challenge.challenge_id, _TEST_OTP)
    ).status is OtpVerificationStatus.VERIFIED
    assert (
        await provider.verify(challenge.challenge_id, "000000")
    ).status is OtpVerificationStatus.CONFLICT


async def test_unverified_challenge_expires() -> None:
    now = [100.0]
    provider = _provider(clock=lambda: now[0])
    challenge = await provider.dispatch(_request())
    now[0] = challenge.expires_at_epoch

    assert (
        await provider.verify(challenge.challenge_id, _TEST_OTP)
    ).status is OtpVerificationStatus.EXPIRED


async def test_store_rejects_an_unknown_challenge_before_provider_verification() -> None:
    store = VerificationStore(_provider(), session_id="session-1")

    with pytest.raises(VerificationContractError, match="undispatched"):
        await store.verify_otp("unknown-challenge", _TEST_OTP)


def test_failed_provider_outcome_cannot_carry_proof_material() -> None:
    with pytest.raises(ValueError, match="proof material"):
        OtpVerification(
            challenge_id="challenge-1",
            status=OtpVerificationStatus.MISMATCHED,
            subject=_subject(),
        )


# --- RiskProvider ------------------------------------------------------------------------


async def test_risk_provider_reports_flag() -> None:
    subject = OrderRiskSubject(
        tenant_id="acme_store",
        session_id="session-1",
        order_refs=("ORD-1001",),
    )
    assert await RiskProvider("acme_store", flagged=False).assess(subject) is RiskDecision.CLEAR
    assert await RiskProvider("acme_store", flagged=True).assess(subject) is RiskDecision.BLOCKED


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
