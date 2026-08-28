"""Step-up verification seams — the authoritative level store + the OTP/risk providers.

The whole point of Phase 3c's T3 loop: an L1 caller asking for a refund to a NEW instrument
must be raised to L2 (§A4b) before the money moves — and that level must be AUTHORITATIVE,
never trusted from checkpoint carry (§A388).

`VerificationStore` is that authority. It is built ONE INSTANCE PER SESSION (like
`OrderStore` in the pipeline) and dies with the session — there is no `session_id` key and
no cross-session state in the build phase (the session-keyed, durable form lands with the
Redis/real-verification-layer swap in Phase 4). The support flow's guardrail READS
`current_level()` LIVE inside the node; `verify_apply` WRITES it via `verify_otp`. Because
the level lives only here (not in a checkpointed graph channel), a replayed/forked
checkpoint cannot re-grant a level — the store-as-truth path removes that bug class.

`OtpProvider` / `RiskProvider` are the two external seams step-up needs, as fakes with an
injectable-real shape (same stance as the fixture `OrderStore`). They are SECURITY-critical:
- the risk provider gates whether an OTP to the number-on-file can be trusted AT ALL
  (SIM-swap / port-out — §A4a: ANI is never the authenticator; a flagged number means no
  OTP, escalate to a human or an app-push/passkey factor);
- OTP dispatch is IDEMPOTENT per step-up attempt — a replayed dispatch node (LangGraph
  re-runs an interrupted node from the top) must not re-send the code.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agnostic_market.commerce.spoken import spoken_digits
from agnostic_market.config.loader import ConfigError, load_yaml_layer
from agnostic_market.dtos.orchestration import VerificationProof

logger = logging.getLogger("agnostic_market.commerce.verification")

# Verification levels (AGENTS §A4a): 0 none · 1 possession-lite · 2 strong (possession
# factor + risk signals). A session starts at L1 in 3c (a connected, greeted caller);
# L2 is earned only by a committed OTP match behind a clear risk check.
# This ladder is the PLATFORM's own vocabulary — it is NOT a NIST AAL mapping: a
# contact-delivered OTP is a single possession factor, not AAL2 (two distinct factors).
# Never present L2 as "NIST AAL2" in UX or compliance material (SECURITY §7d).
# NOTE (P7): L2 is a LEVEL, not an identity — binding a session to a customer additionally
# requires the identity flow's own OTP chain (see PendingIdentity.grants_at_mint).
_INITIAL_LEVEL = 1
_STRICT = ConfigDict(extra="forbid", frozen=True)


class VerificationFixture(BaseModel):
    """Validated build-phase verification settings for one merchant."""

    model_config = _STRICT

    otp_code: str = Field(pattern=r"^\d{6}$")


def load_verification_fixture(config_root: Path, merchant_id: str) -> VerificationFixture:
    """Load the temporary fake-verification fixture, failing loudly at session build."""
    path = config_root / "fixtures" / "verification" / f"{merchant_id}.yaml"
    try:
        return VerificationFixture.model_validate(load_yaml_layer(path))
    except ValidationError as exc:
        raise ConfigError(f"verification fixture {path} failed validation:\n{exc}") from exc


class VerificationStore:
    """Authoritative, per-session verification level + the OTP verify seam.

    Single source of truth for `verification_level`: written only by `verify_otp` (never by
    a graph node writing a literal), read live by the support guardrail. `clear()` is the
    Clock-B reap hook — belt-and-suspenders, since a per-session instance already dies with
    the session.
    """

    def __init__(self, otp: OtpProvider, *, initial_level: int = _INITIAL_LEVEL) -> None:
        self._otp = otp
        self._level = initial_level
        # Records the method + signals that granted each raise — dispute-defense audit
        # (§A4a "log the verification method + signals per action"). No PII, no code value.
        self.grants: list[VerificationProof] = []

    def current_level(self) -> int:
        return self._level

    def verify_otp(self, code: str) -> bool:
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
        if self._otp.verify(spoken or code):
            self._level = 2
            self.grants.append(VerificationProof())
            return True
        return False

    def fresh_proof_since(self, grant_count: int) -> VerificationProof | None:
        """Return this verification chain's newest proof, never an older session grant."""
        fresh = self.grants[grant_count:]
        return fresh[-1] if fresh else None

    def retain_only(self, proof: VerificationProof) -> None:
        """Replace prior verification history with one proof already earned this session."""
        if proof not in self.grants:
            raise ValueError("verification proof was not earned by this session")
        self._level = proof.raised_to
        self.grants[:] = [proof]

    def clear(self) -> None:
        """Reset the granted level (Clock-B teardown; also drops any recorded grants)."""
        self._level = _INITIAL_LEVEL
        self.grants.clear()


@dataclass
class OtpProvider:
    """Fake OTP dispatch/verify — a stand-in for a real OTP service (injectable later).

    `dispatch` is idempotent per `attempt_key`: a step-up attempt sends exactly one code
    however many times a replayed dispatch node runs. A genuine re-collect (wrong code)
    uses a NEW attempt_key, which legitimately sends a fresh code.
    """

    valid_code: str
    _dispatched: set[str] = field(default_factory=set)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    def __post_init__(self) -> None:
        if re.fullmatch(r"\d{6}", self.valid_code) is None:
            raise ValueError("fake OTP code must contain exactly 6 digits")

    def dispatch(self, attempt_key: str) -> None:
        with self._lock:
            if attempt_key in self._dispatched:
                logger.debug("otp dispatch replay for %s - not re-sending", attempt_key)
                return
            self._dispatched.add(attempt_key)

    def verify(self, code: str) -> bool:
        return code.strip() == self.valid_code

    @property
    def dispatch_count(self) -> int:
        """Distinct step-up attempts a code was sent for (test/verification surface)."""
        with self._lock:
            return len(self._dispatched)


@dataclass
class RiskProvider:
    """Fake SIM-swap / port-out risk check on the number-on-file (injectable later).

    §A4a: the number is only a routing hint; a flagged number means an OTP to it cannot be
    trusted — the flow must escalate to a human, not proceed. `flagged` is a build-phase
    switch so tests can drive both branches.
    """

    flagged: bool = False

    def check_sim_swap(self) -> bool:
        """True when the number-on-file shows SIM-swap/port-out risk (do NOT trust an OTP)."""
        return self.flagged
