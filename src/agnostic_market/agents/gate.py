"""The deterministic escalation gate — a SLIM, high-precision fast-path (not the router).

Design stance (settled 2026-07-07). The LLM is the PRIMARY escalation decider — it reads
intent, including paraphrases regex fundamentally can't catch. This gate is a narrow
deterministic floor for the highest-certainty IRREVERSIBLE actions only (cancel / refund /
place-order), where catching them sub-ms and pre-generation is genuinely worth a pattern.
It is NOT the intent router and NOT the safety guarantee.

What actually makes the frontline safe is STRUCTURAL: it holds no state-changing tools, so
an escalation MISS (gate or model) still cannot mutate anything — worst case is "answered
without acting" (recoverable). So the gate is optimized for PRECISION (never false-trip a
read — a gate false-trip is a bad UX bug it owns alone), and recall is a bonus, not a
mandate. We deliberately do NOT grow this toward address/payment/cart paraphrases — a
regex chasing paraphrases it can never fully catch is a known dead end. Those belong to
the model.

Authority model: this set is CODE (a platform floor, like resolver.py's `_ALWAYS_LOCKED`);
Phase-4 onboarding may let a merchant ADD triggers (additive-only), same shape as
policy-within-bounds.
"""

from __future__ import annotations

import re

from agnostic_market.dtos.state import HandoffReasonCode

# Irreversible-action triggers ONLY. High precision: each is a request to DO the
# irreversible thing, guarded against the read/question phrasings that would collide.
_RULES: tuple[tuple[re.Pattern[str], HandoffReasonCode, str], ...] = (
    # cancel an order (NOT "why was my order cancelled" — a read about a past cancel)
    (
        re.compile(
            r"\bcancel\b(?:(?!\b(?:why|was|been|already)\b).)*\b(?:order|purchase|it|this)\b",
            re.IGNORECASE,
        ),
        "cancel_order",
        "support",
    ),
    # refund / return REQUEST (NOT "what's your return policy" — a policy question).
    # 'refund'/'money back' as a request; bare 'return' only with a request verb or object.
    (
        re.compile(
            r"\b(?:want|need|get|give|like|request|issue|process)\b[^.?!]*\b(?:refund|money back)\b"
            r"|\b(?:refund|money back|reimburse)\b\s+(?:me|my|this|it|the order)\b"
            r"|\b(?:want|need|start|request|initiate|process|make)\b[^.?!]*\breturn\b"
            r"|\breturn\b\s+(?:this|it|these|them|my (?:order|item|purchase))\b",
            re.IGNORECASE,
        ),
        "refund",
        "support",
    ),
    # place order / checkout — self-contained purchase commands (no object needed)
    (
        re.compile(
            r"\b(?:checkout|check out|place (?:the |my )?order|buy it now|complete (?:my |the )?"
            r"(?:order|purchase|checkout))\b",
            re.IGNORECASE,
        ),
        "cart_write",
        "checkout",
    ),
)


def gate_check(text: str) -> tuple[HandoffReasonCode, str] | None:
    """Return (reason_code, destination) for a high-certainty irreversible request, else None.

    First matching rule wins. Everything else — address/payment/cart changes, paraphrases,
    ambiguous intents — deliberately returns None and is left to the model (the primary
    escalation decider). A None here is NOT "safe to answer"; it means "the gate has no
    high-certainty opinion — ask the model."
    """
    stripped = text.strip()
    if not stripped:
        return None
    for pattern, reason_code, destination in _RULES:
        if pattern.search(stripped):
            return reason_code, destination
    return None
