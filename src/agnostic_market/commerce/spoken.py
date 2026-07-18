"""Spoken-form input normalization — the LISTEN direction (live call #12, F-12.1/F-12.2).

`speak_*`/`render_*` in orders.py author what the agent SAYS; this module normalizes what
the caller SAID after STT. STT frequently delivers structured values as words — an email as
"k c at example dot com" (no '@' character), an OTP as "four eight two nine one three" —
and a literal compare fails a legitimate caller (live: the CORRECT code, spoken, exhausted
the OTP retries to a human). Normalization is a VOICE-plane reality, so it lives here in
code, once, shared by the contact matcher and the OTP verify — the model-facing prompts
keep their "pass the value EXACTLY as spoken" rule (the model must never reformat; code
normalizes deterministically).
"""

from __future__ import annotations

import re

# Digit words as STT emits them. "oh" is the common spoken zero in phone numbers/codes
# ("five five five oh one oh"); as a standalone token in a NUMBER context it is
# unambiguous — claims/codes are extracted values, not free prose.
_DIGIT_WORDS = {
    "zero": "0",
    "oh": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
}


def spoken_digits(text: str) -> str:
    """Every digit in `text`, with spoken digit-words mapped ("four eight two" -> "482").

    Tokens that are neither digit-words nor digit-bearing are dropped ("It should be
    four eight two nine one three." -> "482913"). Filler number-words around the value
    ("two texts said...") would over-capture — acceptable: the compare stays EXACT
    equality at the caller, so over-capture fails closed (a re-collect, never a match).
    """
    out: list[str] = []
    for token in re.split(r"[^a-z0-9]+", text.lower()):
        if token in _DIGIT_WORDS:
            out.append(_DIGIT_WORDS[token])
        else:
            out.append("".join(ch for ch in token if ch.isdigit()))
    return "".join(out)


def spoken_email(text: str) -> str | None:
    """The email `text` spells, or None if it doesn't spell one.

    Handles the spoken form STT produces: " at " -> "@", " dot " -> "." ("k c at example
    dot com" -> "kc@example.com"), then lowercases, strips whitespace, and drops a trailing
    sentence period. A typed/converted email passes through unchanged.
    """
    lowered = text.strip().lower()
    lowered = re.sub(r"\s+at\s+", "@", lowered)
    lowered = re.sub(r"\s+dot\s+", ".", lowered)
    if "@" not in lowered:
        return None
    return "".join(lowered.split()).rstrip(".")


# --- contact-span redaction for PERSISTED utterances (SECURITY §7d transcript-telemetry
# --- gap): callers speak their email/phone to verify an order, and the answered-turn
# --- classifier dataset records the raw utterance — these spans must not persist. ---------

_TYPED_EMAIL = re.compile(r"\S+@\S+")
# The spoken form: up to a few local-part tokens, " at ", a domain token, then one or more
# " dot <label>" tails ("casey at example dot com", "k c at example dot com").
_SPOKEN_EMAIL = re.compile(
    r"(?:[\w.'-]+\s+){0,3}[\w.'-]+\s+at\s+[\w-]+(?:\s+dot\s+[\w-]+)+", re.IGNORECASE
)
# A contiguous digit-bearing token run this long is phone-shaped. Deliberately above an
# order-id mention (ORD-1001 = 4 digits, and two ids read with a connector between them)
# and below any dialable number (7 = the shortest local form).
_MIN_PHONE_DIGITS = 7


def redact_contact(text: str) -> str:
    """`text` with contact-shaped spans replaced ([email]/[phone]) — for values being
    PERSISTED (telemetry/logs), never for the listen path (the matcher needs the real one).

    Best-effort and deliberately biased toward OVER-capture (a redacted-but-harmless span
    costs one dataset utterance; an unredacted contact is PII at rest): typed emails, the
    spoken " at ... dot ..." form, and any contiguous token run — digits or spoken digit
    words — accumulating >= _MIN_PHONE_DIGITS digits. Rejoins tokens single-spaced (a log
    value, not a transcript). Order-id mentions survive (4 digits, under the phone line).
    """
    redacted = _TYPED_EMAIL.sub("[email]", text)
    redacted = _SPOKEN_EMAIL.sub("[email]", redacted)
    out: list[str] = []
    run: list[str] = []
    run_digits = 0

    def _flush() -> None:
        nonlocal run, run_digits
        if run_digits >= _MIN_PHONE_DIGITS:
            out.append("[phone]")
        else:
            out.extend(run)
        run, run_digits = [], 0

    for token in redacted.split():
        bare = re.sub(r"[^a-z0-9]", "", token.lower())
        digits = 1 if bare in _DIGIT_WORDS else sum(ch.isdigit() for ch in bare)
        if digits:
            run.append(token)
            run_digits += digits
        else:
            _flush()
            out.append(token)
    _flush()
    return " ".join(out)
