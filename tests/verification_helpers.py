from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from agnostic_market.commerce.identity import CustomersFixture
from agnostic_market.commerce.verification import (
    OtpProvider,
    OtpVerificationStatus,
    VerificationFixture,
    VerificationStore,
)
from agnostic_market.config.loader import load_yaml_layer
from agnostic_market.dtos.orchestration import VerificationProof, VerificationPurpose

_TEST_ROOT = Path(__file__).resolve().parent
_CUSTOMERS = CustomersFixture.model_validate(
    load_yaml_layer(_TEST_ROOT / "synthetic_customers.yaml")
)
_VERIFICATION = VerificationFixture.model_validate(
    load_yaml_layer(_TEST_ROOT / "synthetic_verification.yaml")
)
TEST_FACTOR_REFS = {
    customer_ref: entry.factor_ref for customer_ref, entry in _CUSTOMERS.customers.items()
}
TEST_OTP_CODES = {
    customer_ref: _VERIFICATION.otp_codes_by_factor_ref[factor_ref]
    for customer_ref, factor_ref in TEST_FACTOR_REFS.items()
}
TEST_OTP_TTL_SECONDS = _VERIFICATION.challenge_ttl_seconds
TEST_PROOF_TTL_SECONDS = _VERIFICATION.proof_ttl_seconds


def make_otp_provider(
    tenant_id: str = "acme_store",
    *,
    clock: Callable[[], float] | None = None,
) -> OtpProvider:
    kwargs = {"clock": clock} if clock is not None else {}
    return OtpProvider(
        tenant_id,
        codes_by_factor_ref={
            TEST_FACTOR_REFS[customer_ref]: code for customer_ref, code in TEST_OTP_CODES.items()
        },
        challenge_ttl_seconds=TEST_OTP_TTL_SECONDS,
        proof_ttl_seconds=TEST_PROOF_TTL_SECONDS,
        **kwargs,
    )


async def grant_verification(
    store: VerificationStore,
    *,
    customer_ref: str = "CUST-001",
    purpose: VerificationPurpose = "identity",
    dispatch_idempotency_key: str = "synthetic-verification-grant",
) -> VerificationProof:
    subject = store.subject(
        customer_ref=customer_ref,
        factor_ref=TEST_FACTOR_REFS[customer_ref],
        purpose=purpose,
    )
    challenge = await store.dispatch_otp(
        subject=subject,
        dispatch_idempotency_key=dispatch_idempotency_key,
        max_attempts=3,
    )
    status = await store.verify_otp(challenge.challenge_id, TEST_OTP_CODES[customer_ref])
    if status is not OtpVerificationStatus.VERIFIED:
        raise AssertionError(f"synthetic verification failed: {status.value}")
    proof = store.proof_for_challenge(challenge.challenge_id)
    if proof is None:
        raise AssertionError("synthetic verification did not mint a proof")
    return proof
