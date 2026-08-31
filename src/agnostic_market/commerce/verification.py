"""Subject-bound verification contracts and build-phase provider implementations.

The store binds every challenge and proof to tenant, session, customer, factor, and
purpose. Provider-owned expiry and attempt limits remain authoritative across graph replay.
"""

from __future__ import annotations

import asyncio
import re
import threading
import time
import uuid
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Literal, Protocol, runtime_checkable

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    model_validator,
)

from agnostic_market.commerce.spoken import spoken_digits
from agnostic_market.config.loader import ConfigError, load_yaml_layer
from agnostic_market.dtos.orchestration import (
    NonEmptyText,
    NonNegativeFiniteEpoch,
    VerificationProof,
    VerificationPurpose,
)
from agnostic_market.tenancy.context import TenantBound, normalize_tenant_id

# Verification levels (AGENTS §A4a): 0 none · 1 possession-lite · 2 strong (possession
# factor + risk signals). A session starts at L1 in 3c (a connected, greeted caller);
# L2 is earned only by a committed OTP match behind a clear risk check.
# This ladder is the PLATFORM's own vocabulary — it is NOT a NIST AAL mapping: a
# contact-delivered OTP is a single possession factor, not AAL2 (two distinct factors).
# Never present L2 as "NIST AAL2" in UX or compliance material (SECURITY §7d).
# L2 is a level, not an identity. Binding also requires the exact identity challenge proof.
_INITIAL_LEVEL = 1
_STRICT = ConfigDict(extra="forbid", frozen=True)
type PositiveFiniteSeconds = Annotated[
    float,
    Field(gt=0, allow_inf_nan=False),
]
_POSITIVE_FINITE_SECONDS_ADAPTER = TypeAdapter(PositiveFiniteSeconds)


class VerificationFixture(BaseModel):
    """Validated build-phase verification settings for one merchant."""

    model_config = _STRICT

    otp_codes_by_factor_ref: dict[NonEmptyText, str] = Field(min_length=1)
    challenge_ttl_seconds: PositiveFiniteSeconds
    proof_ttl_seconds: PositiveFiniteSeconds

    @model_validator(mode="after")
    def codes_are_six_digits(self) -> VerificationFixture:
        if any(
            re.fullmatch(r"\d{6}", code) is None for code in self.otp_codes_by_factor_ref.values()
        ):
            raise ValueError("fixture OTP codes must contain exactly 6 digits")
        return self


class VerificationContractError(RuntimeError):
    """A verification provider violated its typed authority contract."""


class OtpChallengeConflictError(VerificationContractError):
    """A dispatch key was reused for a different challenge request."""


class OtpVerificationStatus(StrEnum):
    VERIFIED = "verified"
    MISMATCHED = "mismatched"
    EXPIRED = "expired"
    EXHAUSTED = "exhausted"
    UNKNOWN = "unknown"
    CONFLICT = "conflict"


class VerificationSubject(BaseModel):
    """The complete authority scope for one verification challenge."""

    model_config = _STRICT

    tenant_id: NonEmptyText
    session_id: NonEmptyText
    customer_ref: NonEmptyText
    factor_ref: NonEmptyText
    purpose: VerificationPurpose


class OrderRiskSubject(BaseModel):
    """The exact order action assessed outside an OTP challenge."""

    model_config = _STRICT

    tenant_id: NonEmptyText
    session_id: NonEmptyText
    order_refs: tuple[NonEmptyText, ...] = Field(min_length=1)
    purpose: Literal["cancel"] = "cancel"


type RiskSubject = VerificationSubject | OrderRiskSubject


class RiskDecision(StrEnum):
    CLEAR = "clear"
    BLOCKED = "blocked"


class OtpChallengeRequest(BaseModel):
    """One idempotent request for an expiring, attempt-bounded challenge."""

    model_config = _STRICT

    subject: VerificationSubject
    dispatch_idempotency_key: NonEmptyText
    max_attempts: int = Field(ge=1)


class OtpChallenge(BaseModel):
    model_config = _STRICT

    challenge_id: NonEmptyText
    expires_at_epoch: NonNegativeFiniteEpoch


class OtpVerification(BaseModel):
    model_config = _STRICT

    challenge_id: NonEmptyText
    status: OtpVerificationStatus
    subject: VerificationSubject | None = None
    verified_at_epoch: NonNegativeFiniteEpoch | None = None
    proof_expires_at_epoch: NonNegativeFiniteEpoch | None = None

    @model_validator(mode="after")
    def verified_outcome_is_complete(self) -> OtpVerification:
        if self.status is OtpVerificationStatus.VERIFIED:
            if (
                self.subject is None
                or self.verified_at_epoch is None
                or self.proof_expires_at_epoch is None
            ):
                raise ValueError("verified OTP outcome requires its subject and proof lifetime")
            if self.proof_expires_at_epoch <= self.verified_at_epoch:
                raise ValueError("OTP proof expiry must follow verification")
        elif (
            self.subject is not None
            or self.verified_at_epoch is not None
            or self.proof_expires_at_epoch is not None
        ):
            raise ValueError("only a verified OTP outcome may carry proof material")
        return self


@runtime_checkable
class OtpPort(TenantBound, Protocol):
    """Idempotent challenge dispatch and authoritative verification outcomes."""

    async def dispatch(self, request: OtpChallengeRequest) -> OtpChallenge: ...

    async def verify(self, challenge_id: str, code: str) -> OtpVerification: ...

    async def retain_only(
        self,
        session_id: str,
        challenge_ids: Collection[str],
    ) -> None: ...


@runtime_checkable
class RiskPort(TenantBound, Protocol):
    """Subject-bound risk decisions for verification and protected actions."""

    async def assess(self, subject: RiskSubject) -> RiskDecision: ...


def load_verification_fixture(config_root: Path, merchant_id: str) -> VerificationFixture:
    """Load the temporary fake-verification fixture, failing loudly at session build."""
    path = config_root / "fixtures" / "verification" / f"{merchant_id}.yaml"
    try:
        return VerificationFixture.model_validate(load_yaml_layer(path))
    except ValidationError as exc:
        raise ConfigError(f"verification fixture {path} failed validation:\n{exc}") from exc


class VerificationStore:
    """Authoritative per-session verification proofs and OTP service seam."""

    def __init__(
        self,
        otp: OtpPort,
        *,
        session_id: str,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._otp = otp
        self.session_id = session_id.strip()
        if not self.session_id:
            raise ValueError("verification store requires a session id")
        self._subjects_by_challenge: dict[str, VerificationSubject] = {}
        self._proofs_by_challenge: dict[str, VerificationProof] = {}
        self._clock = clock
        self._lock = threading.RLock()
        self._operation_lock = asyncio.Lock()

    @property
    def tenant_id(self) -> str:
        return self._otp.tenant_id

    def uses_otp_provider(self, otp: OtpPort) -> bool:
        return self._otp is otp

    @property
    def grants(self) -> list[VerificationProof]:
        """Return a snapshot of the session's fresh challenge-bound proofs."""
        with self._lock:
            return [
                proof for proof in self._proofs_by_challenge.values() if self._proof_is_fresh(proof)
            ]

    async def dispatch_otp(
        self,
        *,
        subject: VerificationSubject,
        dispatch_idempotency_key: str,
        max_attempts: int,
    ) -> OtpChallenge:
        if subject.tenant_id != self.tenant_id or subject.session_id != self.session_id:
            raise ValueError("OTP subject does not match the verification store")
        request = OtpChallengeRequest(
            subject=subject,
            dispatch_idempotency_key=dispatch_idempotency_key,
            max_attempts=max_attempts,
        )
        async with self._operation_lock:
            challenge = await self._otp.dispatch(request)
            with self._lock:
                existing = self._subjects_by_challenge.get(challenge.challenge_id)
                if existing is not None and existing != subject:
                    raise VerificationContractError("OTP challenge changed authority scope")
                self._subjects_by_challenge[challenge.challenge_id] = subject
        return challenge

    def current_level(self) -> int:
        with self._lock:
            return max(
                (
                    proof.raised_to
                    for proof in self._proofs_by_challenge.values()
                    if self._proof_is_fresh(proof)
                ),
                default=_INITIAL_LEVEL,
            )

    def _proof_is_fresh(self, proof: VerificationProof) -> bool:
        return self._clock() < proof.expires_at_epoch

    def _proof_satisfies(
        self,
        proof: VerificationProof,
        subject: VerificationSubject,
        required_level: int,
        *,
        exact_purpose: bool,
    ) -> bool:
        return bool(
            self._proof_is_fresh(proof)
            and proof.raised_to >= required_level
            and proof.tenant_id == subject.tenant_id
            and proof.session_id == subject.session_id
            and proof.customer_ref == subject.customer_ref
            and proof.factor_ref == subject.factor_ref
            and (not exact_purpose or proof.purpose == subject.purpose)
        )

    def subject(
        self,
        *,
        customer_ref: str,
        factor_ref: str,
        purpose: VerificationPurpose,
    ) -> VerificationSubject:
        return VerificationSubject(
            tenant_id=self.tenant_id,
            session_id=self.session_id,
            customer_ref=customer_ref,
            factor_ref=factor_ref,
            purpose=purpose,
        )

    async def verify_otp(self, challenge_id: str, code: str) -> OtpVerificationStatus:
        """Verify a COMMITTED OTP code; on match, raise the level to L2 and record the grant.

        Consent/commit discipline (only committed transcript reaches here) is the caller's
        responsibility (VOICE_PIPELINE §0) — this store trusts that `code` is committed.

        The spoken answer is DIGIT-NORMALIZED here (live call #12 F-12.2: the CORRECT code
        arrived as "four eight two nine one three", failed the literal compare, and
        exhausted a legitimate caller to a human). Normalizing at THIS seam keeps the
        provider contract digits-in (a real OTP service verifies a digit string) and fixes
        every step-up family at once; the compare stays EXACT equality, so word noise
        around the value fails closed (a re-collect, never a match).
        """
        spoken = spoken_digits(code)
        async with self._operation_lock:
            with self._lock:
                expected_subject = self._subjects_by_challenge.get(challenge_id)
            if expected_subject is None:
                raise VerificationContractError("OTP verification used an undispatched challenge")
            outcome = await self._otp.verify(challenge_id, spoken or code.strip())
            with self._lock:
                if outcome.challenge_id != challenge_id:
                    raise VerificationContractError("OTP provider returned the wrong challenge")
                if self._subjects_by_challenge.get(challenge_id) != expected_subject:
                    raise VerificationContractError(
                        "OTP challenge authority changed during verification"
                    )
                if outcome.status is not OtpVerificationStatus.VERIFIED:
                    return outcome.status
                if (
                    outcome.subject != expected_subject
                    or outcome.verified_at_epoch is None
                    or outcome.proof_expires_at_epoch is None
                ):
                    raise VerificationContractError("OTP proof changed authority scope")
                if challenge_id not in self._proofs_by_challenge:
                    proof = VerificationProof(
                        tenant_id=expected_subject.tenant_id,
                        session_id=expected_subject.session_id,
                        customer_ref=expected_subject.customer_ref,
                        factor_ref=expected_subject.factor_ref,
                        purpose=expected_subject.purpose,
                        challenge_id=challenge_id,
                        verified_at_epoch=outcome.verified_at_epoch,
                        expires_at_epoch=outcome.proof_expires_at_epoch,
                    )
                    self._proofs_by_challenge[challenge_id] = proof
                proof = self._proofs_by_challenge[challenge_id]
                return (
                    OtpVerificationStatus.VERIFIED
                    if self._proof_is_fresh(proof)
                    else OtpVerificationStatus.EXPIRED
                )

    def proof_for_challenge(self, challenge_id: str) -> VerificationProof | None:
        """Return a fresh proof minted by the named challenge."""
        with self._lock:
            proof = self._proofs_by_challenge.get(challenge_id)
            return proof if proof is not None and self._proof_is_fresh(proof) else None

    def authorization_satisfies(
        self,
        subject: VerificationSubject,
        required_level: int,
    ) -> bool:
        """Check a fresh proof for the same session, customer, and factor.

        Purpose remains part of the proof's audit identity and each protected action must
        run its own risk assessment. It is intentionally not a second OTP boundary.
        """
        with self._lock:
            return any(
                self._proof_satisfies(
                    proof,
                    subject,
                    required_level,
                    exact_purpose=False,
                )
                for proof in self._proofs_by_challenge.values()
            )

    def challenge_satisfies(
        self,
        challenge_id: str,
        subject: VerificationSubject,
        required_level: int,
    ) -> bool:
        """Report whether this exact subject earned the required level."""
        with self._lock:
            proof = self._proofs_by_challenge.get(challenge_id)
            return bool(
                self._subjects_by_challenge.get(challenge_id) == subject
                and proof is not None
                and self._proof_satisfies(
                    proof,
                    subject,
                    required_level,
                    exact_purpose=True,
                )
            )

    async def retain_only(self, proof: VerificationProof) -> None:
        """Replace prior verification history with one proof already earned this session."""
        async with self._operation_lock:
            with self._lock:
                if self._proofs_by_challenge.get(
                    proof.challenge_id
                ) != proof or not self._proof_is_fresh(proof):
                    raise ValueError("verification proof was not earned by this session")
                subject = self._subjects_by_challenge[proof.challenge_id]
            await self._otp.retain_only(self.session_id, (proof.challenge_id,))
            with self._lock:
                self._subjects_by_challenge = {proof.challenge_id: subject}
                self._proofs_by_challenge = {proof.challenge_id: proof}

    async def clear(self) -> None:
        """Reset the granted level (Clock-B teardown; also drops any recorded grants)."""
        async with self._operation_lock:
            await self._otp.retain_only(self.session_id, ())
            with self._lock:
                self._subjects_by_challenge.clear()
                self._proofs_by_challenge.clear()


@dataclass
class _FixtureChallengeState:
    request: OtpChallengeRequest
    challenge: OtpChallenge
    remaining_attempts: int
    verified_at_epoch: float | None = None
    proof_expires_at_epoch: float | None = None


@dataclass
class OtpProvider:
    """Thread-safe build fixture with factor codes and provider-owned challenge state."""

    tenant_id: str
    codes_by_factor_ref: Mapping[str, str]
    challenge_ttl_seconds: float
    proof_ttl_seconds: float
    clock: Callable[[], float] = field(default=time.time, repr=False)
    _challenge_by_dispatch: dict[str, str] = field(default_factory=dict)
    _challenges: dict[str, _FixtureChallengeState] = field(default_factory=dict)
    _dispatch_count: int = field(default=0, init=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    def __post_init__(self) -> None:
        self.tenant_id = normalize_tenant_id(self.tenant_id, boundary="OTP provider")
        self.codes_by_factor_ref = MappingProxyType(dict(self.codes_by_factor_ref))
        if not self.codes_by_factor_ref:
            raise ValueError("fake OTP provider requires at least one factor")
        if any(
            not factor_ref.strip() or re.fullmatch(r"\d{6}", code) is None
            for factor_ref, code in self.codes_by_factor_ref.items()
        ):
            raise ValueError("fake OTP factors require non-empty refs and 6-digit codes")
        self.challenge_ttl_seconds = _POSITIVE_FINITE_SECONDS_ADAPTER.validate_python(
            self.challenge_ttl_seconds
        )
        self.proof_ttl_seconds = _POSITIVE_FINITE_SECONDS_ADAPTER.validate_python(
            self.proof_ttl_seconds
        )

    async def dispatch(self, request: OtpChallengeRequest) -> OtpChallenge:
        if request.subject.tenant_id != self.tenant_id:
            raise ValueError("OTP challenge tenant does not match the provider")
        if request.subject.factor_ref not in self.codes_by_factor_ref:
            raise ValueError("OTP challenge factor is unavailable")
        with self._lock:
            existing_id = self._challenge_by_dispatch.get(request.dispatch_idempotency_key)
            if existing_id is not None:
                existing = self._challenges[existing_id]
                if existing.request != request:
                    raise OtpChallengeConflictError(
                        "OTP dispatch key was reused for a different request"
                    )
                return existing.challenge
            challenge_id = uuid.uuid4().hex
            challenge = OtpChallenge(
                challenge_id=challenge_id,
                expires_at_epoch=self.clock() + self.challenge_ttl_seconds,
            )
            self._challenge_by_dispatch[request.dispatch_idempotency_key] = challenge_id
            self._challenges[challenge_id] = _FixtureChallengeState(
                request=request,
                challenge=challenge,
                remaining_attempts=request.max_attempts,
            )
            self._dispatch_count += 1
            return challenge

    async def verify(self, challenge_id: str, code: str) -> OtpVerification:
        with self._lock:
            state = self._challenges.get(challenge_id)
            if state is None:
                return OtpVerification(
                    challenge_id=challenge_id,
                    status=OtpVerificationStatus.UNKNOWN,
                )
            now = self.clock()
            expected_code = self.codes_by_factor_ref[state.request.subject.factor_ref]
            if state.verified_at_epoch is not None:
                if code.strip() != expected_code:
                    return OtpVerification(
                        challenge_id=challenge_id,
                        status=OtpVerificationStatus.CONFLICT,
                    )
                if state.proof_expires_at_epoch is None:
                    raise VerificationContractError(
                        "verified fixture challenge has no proof expiry"
                    )
                return OtpVerification(
                    challenge_id=challenge_id,
                    status=OtpVerificationStatus.VERIFIED,
                    subject=state.request.subject,
                    verified_at_epoch=state.verified_at_epoch,
                    proof_expires_at_epoch=state.proof_expires_at_epoch,
                )
            if now >= state.challenge.expires_at_epoch:
                return OtpVerification(
                    challenge_id=challenge_id,
                    status=OtpVerificationStatus.EXPIRED,
                )
            if state.remaining_attempts <= 0:
                return OtpVerification(
                    challenge_id=challenge_id,
                    status=OtpVerificationStatus.EXHAUSTED,
                )
            state.remaining_attempts -= 1
            if code.strip() != expected_code:
                status = (
                    OtpVerificationStatus.EXHAUSTED
                    if state.remaining_attempts == 0
                    else OtpVerificationStatus.MISMATCHED
                )
                return OtpVerification(challenge_id=challenge_id, status=status)
            state.verified_at_epoch = now
            state.proof_expires_at_epoch = now + self.proof_ttl_seconds
            return OtpVerification(
                challenge_id=challenge_id,
                status=OtpVerificationStatus.VERIFIED,
                subject=state.request.subject,
                verified_at_epoch=now,
                proof_expires_at_epoch=state.proof_expires_at_epoch,
            )

    async def retain_only(
        self,
        session_id: str,
        challenge_ids: Collection[str],
    ) -> None:
        normalized_session_id = session_id.strip()
        if not normalized_session_id:
            raise ValueError("OTP cleanup requires a session id")
        retained_ids = frozenset(challenge_ids)
        with self._lock:
            retained_states = tuple(
                state
                for challenge_id in retained_ids
                if (state := self._challenges.get(challenge_id)) is not None
            )
            if len(retained_states) != len(retained_ids):
                raise VerificationContractError("OTP cleanup retained an unknown challenge")
            if any(
                state.request.subject.session_id != normalized_session_id
                for state in retained_states
            ):
                raise VerificationContractError("OTP cleanup crossed a session boundary")
            discarded_ids = tuple(
                challenge_id
                for challenge_id, state in self._challenges.items()
                if state.request.subject.session_id == normalized_session_id
                and challenge_id not in retained_ids
            )
            for challenge_id in discarded_ids:
                state = self._challenges.pop(challenge_id, None)
                assert state is not None
                dispatch_key = state.request.dispatch_idempotency_key
                if self._challenge_by_dispatch.get(dispatch_key) == challenge_id:
                    self._challenge_by_dispatch.pop(dispatch_key, None)

    @property
    def dispatch_count(self) -> int:
        """Distinct step-up attempts a code was sent for (test/verification surface)."""
        with self._lock:
            return self._dispatch_count

    @property
    def active_challenge_count(self) -> int:
        with self._lock:
            return len(self._challenges)


@dataclass
class RiskProvider:
    """Fake SIM-swap / port-out risk check on the number-on-file (injectable later).

    §A4a: the number is only a routing hint; a flagged number means an OTP to it cannot be
    trusted — the flow must escalate to a human, not proceed. `flagged` is a build-phase
    switch so tests can drive both branches.
    """

    tenant_id: str
    flagged: bool = False

    def __post_init__(self) -> None:
        self.tenant_id = normalize_tenant_id(self.tenant_id, boundary="risk provider")

    async def assess(self, subject: RiskSubject) -> RiskDecision:
        if subject.tenant_id != self.tenant_id:
            raise ValueError("risk subject tenant does not match the provider")
        return RiskDecision.BLOCKED if self.flagged else RiskDecision.CLEAR
