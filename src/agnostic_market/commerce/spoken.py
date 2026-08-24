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
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

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
_ORDER_FUSED_LABEL = re.compile(r"ord(\d+)$")
_ORDER_LABEL_TOKENS = frozenset({"ord", "order"})
_ORDER_VALUE_FILLERS = frozenset({"ord", "number", "id", "is"})
_ORDER_SET_CONNECTORS = frozenset({"and", "also", "plus"})
_ORDER_SET_BLOCKERS = frozenset({"dont", "never", "no", "not", "rather", "instead"})
# Seven digits is the shortest dialable local number accepted by the voice boundary.
_MIN_PHONE_DIGITS = 7


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


def caller_stated_phone(utterance: str, proposed: str) -> bool:
    """Whether `proposed` is one complete phone-shaped run in committed caller speech."""
    proposed_digits = spoken_digits(proposed)
    if len(proposed_digits) < _MIN_PHONE_DIGITS:
        return False
    runs: list[str] = []
    current: list[str] = []
    for token in utterance.split():
        digits = spoken_digits(token)
        if digits:
            current.append(digits)
        elif current:
            runs.append("".join(current))
            current = []
    if current:
        runs.append("".join(current))
    return proposed_digits in runs


@dataclass(frozen=True)
class _OrderReferenceSpan:
    reference: str
    token_start: int
    token_end: int
    char_start: int
    char_end: int
    collapsed_or_label: bool = False


def _order_reference_at(
    tokens: tuple[re.Match[str], ...], index: int
) -> _OrderReferenceSpan | None:
    value = tokens[index].group()
    fused = _ORDER_FUSED_LABEL.fullmatch(value)
    collapsed_or_label = False
    if fused is not None:
        return _OrderReferenceSpan(
            reference=f"ORD-{fused.group(1)}",
            token_start=index,
            token_end=index + 1,
            char_start=tokens[index].start(),
            char_end=tokens[index].end(),
        )
    if value in _ORDER_LABEL_TOKENS:
        cursor = index + 1
    elif index + 2 < len(tokens) and tuple(
        token.group() for token in tokens[index : index + 3]
    ) == ("o", "r", "d"):
        cursor = index + 3
    elif index + 1 < len(tokens) and (value, tokens[index + 1].group()) == ("or", "d"):
        cursor = index + 2
        collapsed_or_label = True
    else:
        return None

    while cursor < len(tokens) and tokens[cursor].group() in _ORDER_VALUE_FILLERS:
        cursor += 1
    if cursor < len(tokens) and tokens[cursor].group() == "dash":
        cursor += 1
    if cursor >= len(tokens):
        return None

    payload = tokens[cursor].group()
    if payload.isdigit():
        digits = payload
        cursor += 1
    elif payload in _DIGIT_WORDS:
        parts: list[str] = []
        while cursor < len(tokens) and tokens[cursor].group() in _DIGIT_WORDS:
            parts.append(_DIGIT_WORDS[tokens[cursor].group()])
            cursor += 1
        digits = "".join(parts)
    else:
        nested_fused = _ORDER_FUSED_LABEL.fullmatch(payload)
        if nested_fused is None:
            return None
        digits = nested_fused.group(1)
        cursor += 1

    return _OrderReferenceSpan(
        reference=f"ORD-{digits}",
        token_start=index,
        token_end=cursor,
        char_start=tokens[index].start(),
        char_end=tokens[cursor - 1].end(),
        collapsed_or_label=collapsed_or_label,
    )


def _is_additive_order_join(text: str) -> bool:
    words = re.findall(r"[a-z0-9]+", text)
    if words:
        return all(word in _ORDER_SET_CONNECTORS for word in words)
    return any(mark in text for mark in (",", ";", "&", "+"))


def caller_stated_order_ids(utterance: str) -> tuple[str, ...]:
    """Extract one unambiguous, strong-labelled order-reference set from one utterance."""
    normalized = utterance.casefold().replace("don't", "dont")
    tokens = tuple(re.finditer(r"[a-z0-9]+", normalized))
    spans: list[_OrderReferenceSpan] = []
    index = 0
    while index < len(tokens):
        span = _order_reference_at(tokens, index)
        if span is not None:
            if span.collapsed_or_label and spans:
                return ()
            spans.append(span)
            index = span.token_end
            continue
        index += 1

    if not spans:
        return ()

    consumed = {
        token_index for span in spans for token_index in range(span.token_start, span.token_end)
    }
    if any(
        token.group() in _ORDER_SET_BLOCKERS
        for token_index, token in enumerate(tokens)
        if token_index not in consumed
    ):
        return ()
    if any(
        not _is_additive_order_join(normalized[left.char_end : right.char_start])
        for left, right in pairwise(spans)
    ):
        return ()

    tail = normalized[spans[-1].char_end :]
    tail_words = re.findall(r"[a-z0-9]+", tail)
    if tail_words and (tail_words[0] in _ORDER_SET_CONNECTORS or tail_words[0] == "or"):
        return ()

    return tuple(dict.fromkeys(span.reference for span in spans))


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
_SPOKEN_EMAIL_PREFIX_FILLERS = frozenset(
    {"address", "again", "contact", "email", "is", "its", "my", "the"}
)


@dataclass(frozen=True, slots=True)
class ContactCandidate:
    """One ephemeral contact-shaped span; syntax only, never directory authority."""

    kind: Literal["email", "phone"]
    claim: str


def _spoken_email_candidate(span: str) -> str | None:
    separators = tuple(re.finditer(r"\s+at\s+", span, re.IGNORECASE))
    if not separators:
        return None
    separator = separators[-1]
    local, domain = span[: separator.start()], span[separator.end() :]
    local_tokens = local.split()
    while len(local_tokens) > 1 and local_tokens[0].casefold().strip(".,:;'") in (
        _SPOKEN_EMAIL_PREFIX_FILLERS
    ):
        local_tokens.pop(0)
    if not local_tokens:
        return None
    # The redaction grammar intentionally captures context before the address. Authorization
    # takes only the final lexical token, plus a contiguous run of single-character tokens
    # for STT forms such as "k c at example dot com".
    bounded_local = [local_tokens[-1]]
    for token in reversed(local_tokens[:-1]):
        bare = re.sub(r"[^a-z0-9]", "", token.casefold())
        if len(bare) != 1:
            break
        bounded_local.append(token)
    bounded_local.reverse()
    return spoken_email(f"{' '.join(bounded_local)} at {domain}")


def scan_contact_candidates(text: str) -> tuple[ContactCandidate, ...]:
    """Return bounded contact spans without consulting a customer directory.

    The scanner shares the redaction grammars below but is precision-biased for
    authorization: an order-labelled numeric run is excluded instead of being guessed as a
    phone number. Directory-equivalent deduplication remains the Identity layer's job.
    """

    candidates: list[ContactCandidate] = []
    for match in _TYPED_EMAIL.finditer(text):
        claim = spoken_email(match.group(0).strip(".,;:!?()[]{}"))
        if claim is not None:
            candidates.append(ContactCandidate(kind="email", claim=claim))
    for match in _SPOKEN_EMAIL.finditer(text):
        claim = _spoken_email_candidate(match.group(0))
        if claim is not None:
            candidates.append(ContactCandidate(kind="email", claim=claim))

    # Remove email spans before numeric scanning so digits inside an address cannot become
    # a second phone candidate. Preserve whitespace to keep surrounding numeric runs apart.
    numeric_text = _TYPED_EMAIL.sub(" ", text)
    numeric_text = _SPOKEN_EMAIL.sub(" ", numeric_text)
    run: list[str] = []
    run_digits = 0
    order_tainted = False
    preceding_order_label = False

    def flush() -> None:
        nonlocal run, run_digits, order_tainted
        if run_digits >= _MIN_PHONE_DIGITS and not order_tainted:
            candidates.append(ContactCandidate(kind="phone", claim="".join(run)))
        run = []
        run_digits = 0
        order_tainted = False

    for token in numeric_text.split():
        bare = re.sub(r"[^a-z0-9]", "", token.casefold())
        value = spoken_digits(token)
        is_numeric = bool(value) and (bare in _DIGIT_WORDS or any(ch.isdigit() for ch in bare))
        if is_numeric:
            if not run:
                order_tainted = preceding_order_label or bare.startswith("ord")
            else:
                order_tainted = order_tainted or bare.startswith("ord")
            run.append(value)
            run_digits += len(value)
            preceding_order_label = False
            continue
        flush()
        preceding_order_label = bare in _ORDER_LABEL_TOKENS
    flush()
    return tuple(candidates)


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
