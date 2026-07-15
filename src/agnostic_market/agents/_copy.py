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
