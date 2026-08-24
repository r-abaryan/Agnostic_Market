"""Deterministic closed-consent classification.

Committed-transcript-only (VOICE_PIPELINE §0): these decide whether a caller's words
authorize an irreversible action (checkout place, refund). They are CODE, not a prompt: a
model must never decide whether a reply grants consent. Negation is checked before the bounded
affirmative grammar so ambiguous language fails closed.

Lifted out of checkout/flow.py at 3c so the support flow reuses the exact same classifier
rather than a second, drifting copy (one source of truth).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from agnostic_market.dtos.state import HandoffSource

_NO_RE = re.compile(
    r"\b(?:no|nope|nah|don'?t|do not|cancel|stop|never ?mind|forget it|wrong)\b", re.IGNORECASE
)
_AFFIRMATIVE_SIGNALS = (
    ("sounds", "good"),
    ("please", "do"),
    ("go", "ahead"),
    ("do", "it"),
    ("that's", "right"),
    ("that", "is", "right"),
    ("that's", "correct"),
    ("that", "is", "correct"),
    ("confirm", "it"),
    ("yes", "sir"),
    ("sure",),
    ("alright",),
    ("yes",),
    ("yeah",),
    ("yep",),
    ("yup",),
    ("confirm",),
    ("confirmed",),
    ("ok",),
    ("okay",),
)
# These may reinforce an affirmative signal, but never authorize on their own.
_AFFIRMATIVE_FILLERS = (
    ("thank", "you"),
    ("and",),
    ("please",),
    ("then",),
    ("absolutely",),
    ("definitely",),
    ("thanks",),
)
# The cancel-action phrase, neutralized before classifying consent AT THE CANCEL READBACK:
# "cancel" sits in _NO_RE (correct when confirming a purchase/refund — "cancel that" =
# don't do it) but is AFFIRMATIVE when the question being answered IS "shall I cancel?".
_CANCEL_PHRASE_RE = re.compile(
    r"\bcancel(?:ling|led)?\b(?:\s+(?:that|it|this|the|my)\b)?(?:\s+(?:order|purchase)\b)?",
    re.IGNORECASE,
)

Consent = Literal["no", "yes", "unclear"]
ConfirmationVerdict = Literal["human", "no", "yes", "unclear"]


@dataclass(frozen=True, slots=True)
class ConfirmationDecision:
    verdict: ConfirmationVerdict
    handoff_source: HandoffSource | None = None


def _normalize_consent_reply(text: str) -> str:
    normalized = text.casefold().replace("\N{RIGHT SINGLE QUOTATION MARK}", "'")
    return " ".join(re.sub(r"[,.!?;:]+", " ", normalized).split())


def _is_bounded_affirmative(text: str) -> bool:
    words = tuple(text.split())
    pending = [(0, False)]
    visited: set[tuple[int, bool]] = set()
    while pending:
        index, has_signal = pending.pop()
        if (index, has_signal) in visited:
            continue
        visited.add((index, has_signal))
        if index == len(words):
            if has_signal:
                return True
            continue
        pending.extend(
            (index + len(phrase), True)
            for phrase in _AFFIRMATIVE_SIGNALS
            if words[index : index + len(phrase)] == phrase
        )
        pending.extend(
            (index + len(phrase), has_signal)
            for phrase in _AFFIRMATIVE_FILLERS
            if words[index : index + len(phrase)] == phrase
        )
    return False


def classify_consent(text: str) -> Consent:
    """Classify one closed consent reply without inferring other intents."""
    normalized = _normalize_consent_reply(text)
    if _NO_RE.search(normalized):
        return "no"
    if _is_bounded_affirmative(normalized):
        return "yes"
    return "unclear"


def classify_cancel_consent(text: str) -> Consent:
    """`classify_consent` for the CANCEL readback, where the action word inverts polarity.

    The question being answered is "shall I cancel your order?" — so "yeah, cancel it" is a
    YES, while plain `classify_consent` would read the `cancel` as a no. Neutralize the
    cancel-action phrase, then classify what remains: "yeah cancel it" -> "yeah" -> yes;
    "don't cancel" -> "don't" -> no; bare "cancel it" -> "" -> unclear (the confirm node's
    existing one re-confirm asks "yes or no?"). Negations survive the strip, so they still
    win (§4a discipline unchanged).
    """
    return classify_consent(_CANCEL_PHRASE_RE.sub(" ", text))


def classify_confirmation(
    answer: Mapping[str, object],
    *,
    cancel_action: bool = False,
) -> ConfirmationDecision:
    """Combine code-owned consent with an engine-authored semantic handoff marker."""

    source_value = answer.get("handoff_source")
    if source_value is not None:
        try:
            source = HandoffSource(source_value)
        except (TypeError, ValueError) as exc:
            raise ValueError("confirmation handoff source is invalid") from exc
        return ConfirmationDecision("human", source)
    text = str(answer.get("text", ""))
    verdict = classify_cancel_consent(text) if cancel_action else classify_consent(text)
    return ConfirmationDecision(verdict)
