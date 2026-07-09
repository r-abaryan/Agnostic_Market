"""Deterministic consent + escape classification — shared by every gated flow.

Committed-transcript-only (VOICE_PIPELINE §0): these decide whether a caller's words
authorize an irreversible action (checkout place, refund) or break out of a flow. They are
CODE, not a prompt — a model must never be the thing that decides "was that a yes." Checked
in a fixed order (human -> no -> yes -> unclear) so negatives beat the positives they
contain ("no, don't do it" contains "do it").

Lifted out of checkout/flow.py at 3c so the support flow reuses the exact same classifier
rather than a second, drifting copy (one source of truth).
"""

from __future__ import annotations

import re
from typing import Literal

_HUMAN_RE = re.compile(
    r"\b(?:human|person|agent|representative|operator|somebody real)\b", re.IGNORECASE
)
_NO_RE = re.compile(
    r"\b(?:no|nope|nah|don'?t|do not|cancel|stop|never ?mind|forget it|wrong)\b", re.IGNORECASE
)
_YES_RE = re.compile(
    r"\b(?:yes|yeah|yep|yup|sure|correct|right|confirm|confirmed|go ahead|place it|do it|"
    r"sounds good|please do|ok(?:ay)?)\b",
    re.IGNORECASE,
)
_ABORT_RE = re.compile(
    r"\b(?:stop|never ?mind|forget it|cancel (?:that|it|this)|no thanks|don'?t bother)\b",
    re.IGNORECASE,
)

Consent = Literal["human", "no", "yes", "unclear"]


def wants_human(text: str) -> bool:
    """§A9 no-trap escape: the caller asked for a person."""
    return bool(_HUMAN_RE.search(text))


def is_abort(text: str) -> bool:
    """Explicit abort of the in-flight gated action (entry-router escape)."""
    return bool(_ABORT_RE.search(text))


def classify_consent(text: str) -> Consent:
    """'human' | 'no' | 'yes' | 'unclear' — deterministic, order matters."""
    if wants_human(text):
        return "human"
    if _NO_RE.search(text):
        return "no"
    if _YES_RE.search(text):
        return "yes"
    return "unclear"
