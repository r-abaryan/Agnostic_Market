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
from dataclasses import dataclass, field

logger = logging.getLogger("agnostic_market.commerce.verification")

# Verification levels (AGENTS §A4a): 0 none · 1 possession-lite · 2 strong (possession
# factor + risk signals). A session starts at L1 in 3c (a connected, greeted caller);
# L2 is earned only by a committed OTP match behind a clear risk check.
_INITIAL_LEVEL = 1


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
        self.grants: list[dict[str, object]] = []

    def current_level(self) -> int:
        return self._level

    def verify_otp(self, code: str) -> bool:
        """Verify a COMMITTED OTP code; on match, raise the level to L2 and record the grant.

        Consent/commit discipline (only committed transcript reaches here) is the caller's
        responsibility (VOICE_PIPELINE §0) — this store trusts that `code` is committed.
        """
        if self._otp.verify(code):
            self._level = 2
            self.grants.append({"method": "otp", "raised_to": 2})
            return True
        return False

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

    valid_code: str = "482913"
    _dispatched: set[str] = field(default_factory=set)

    def dispatch(self, attempt_key: str) -> None:
        if attempt_key in self._dispatched:
            logger.debug("otp dispatch replay for %s - not re-sending", attempt_key)
            return
        self._dispatched.add(attempt_key)

    def verify(self, code: str) -> bool:
        return code.strip() == self.valid_code

    @property
    def dispatch_count(self) -> int:
        """Distinct step-up attempts a code was sent for (test/verification surface)."""
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
