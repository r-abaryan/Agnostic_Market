"""Shared caller-facing copy — code-authored, one source so the whole call sounds consistent
(a robotic "Anything else?" on every turn is the thing this replaces).

FACTUAL only: warm conversational connectors, never policy statements (policy is owned by the
model prompt, agents/spoken_policy.py) and never product opinions ("nice choice" read as
scripted no matter the wording — dropped after live feedback; a real preference nod needs
catalog data + a lighter touch than a canned line, a future feature). Used INSIDE graph nodes
that author their own spoken line (the one-author rule).
"""

from __future__ import annotations

import itertools

from agnostic_market.dtos.orchestration import PrincipalCompletionKind

# A small rotation, not one fixed line — variety across a call without sounding scripted;
# deterministic order (a cycle, not random) so tests are stable and the wording is reviewable.
_CLOSES = (
    "Anything else I can help with?",
    "What else can I get you?",
    "Is there anything else on your mind?",
    "Anything else you'd like to do?",
)
_close_cycle = itertools.cycle(_CLOSES)


def warm_close() -> str:
    """One warm, factual turn-closing line, rotating across `_CLOSES`. Cosmetic only — the
    rotation is a module-level cycle (variety, never correctness); cross-session sharing is
    harmless."""
    return next(_close_cycle)


def all_closes() -> tuple[str, ...]:
    """The full close set (test surface: a rendered/ack line ends with ONE of these)."""
    return _CLOSES


# The ONE closing line for a GUEST enumeration (session-scoped list): it both discloses that the
# list is partial (this call only) and offers the path to more (verify -> account history). Used
# INSTEAD OF warm_close() on that path so the caller hears exactly one closing invitation.
GUEST_LIST_CLOSE = "To pull up any other orders on an account, I can verify you - just let me know."

# Shared factual question for flows that need the account contact. Authorization-specific
# instructions and decline copy stay with their owning flows.
ACCOUNT_CONTACT_QUESTION = "What email address or phone number is on the account?"
ORDER_NUMBER_QUESTION = "What is the order number, for example ORD-1234?"


def guest_list_close() -> str:
    return GUEST_LIST_CLOSE


# The answer to "am I verified?": current state, not the identity flow's transition lines
# ("You're now verified on the new account."), which announce a change that just happened.
IDENTITY_STATUS_VERIFIED = "You're verified on this call."
IDENTITY_STATUS_UNVERIFIED = "You're not verified on this call."


def identity_status_line(*, verified: bool) -> str:
    return IDENTITY_STATUS_VERIFIED if verified else IDENTITY_STATUS_UNVERIFIED


_PRINCIPAL_COMPLETION_LINES: dict[PrincipalCompletionKind, str] = {
    "switch_account": "You're now verified on the new account.",
    "verify_identity": "You're now verified.",
}


def principal_completion_line(completion_kind: PrincipalCompletionKind) -> str | None:
    """Render a completed identity outcome; continuations are intentionally silent here."""
    if completion_kind == "continue_request":
        return None
    return _PRINCIPAL_COMPLETION_LINES[completion_kind]
