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
