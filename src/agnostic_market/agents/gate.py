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
from typing import Literal

from agnostic_market.dtos.state import HandoffDestination, HandoffReasonCode

# Irreversible-action triggers ONLY. High precision: each is a request to DO the
# irreversible thing, guarded against the read/question phrasings that would collide.
_RULES: tuple[tuple[re.Pattern[str], HandoffReasonCode, HandoffDestination], ...] = (
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
    # 'refund'/'money back' as a request; bare 'return' only with a request verb or object;
    # 'send (it/them/…) back' — the return phrasing with no return/refund token at all
    # (live-eval miss: the model answered it with policy once policy facts existed).
    (
        re.compile(
            r"\b(?:want|need|get|give|like|request|issue|process)\b[^.?!]*\b(?:refund|money back)\b"
            r"|\b(?:refund|money back|reimburse)\b\s+(?:me|my|this|it|the order)\b"
            r"|\b(?:want|need|start|request|initiate|process|make)\b[^.?!]*\breturn\b"
            r"|\breturn\b\s+(?:this|it|these|them|my (?:order|item|purchase))\b"
            r"|\bsend\s+(?:(?:it|this|these|them|everything)\s+)?back\b",
            re.IGNORECASE,
        ),
        "refund",
        "support",
    ),
    # place order / checkout — self-contained purchase commands (no object needed). Routes to
    # the "checkout" DESTINATION, which since Group B is served by the CART flow (the cart
    # owns both mutation and the whole-cart placement tail; "checkout" is a legacy name kept
    # so the handover enum stays stable). ADD-to-cart phrasings are NOT here — reversible,
    # the model's job (request_handover), consistent with the irreversible-only stance.
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


def gate_check(text: str) -> tuple[HandoffReasonCode, HandoffDestination] | None:
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


# --- state-VERIFICATION questions (forced-read detector; live call #16 F-16.1) ------------
#
# Same precision-over-recall stance as the gate above, and DELIBERATELY narrow: this catches
# only utterances ASKING WHETHER a known / referenced order (or order-set) is in a specific
# STATE now (cancelled / shipped / delivered). It is NOT a "status family" that can grow into
# a semantic router — a hit routes to a code read (frontline/graph.py forced_status_node), whose
# only job is to answer that class of question from the store instead of letting the model
# answer it from conversation memory (F-16.1: "so both are cancelled?" -> a memory-lie).
#
# It must NOT fire on:
#   - imperatives ("cancel my order")               -> the gate handles those (runs first).
#   - cause reads ("why was my order cancelled")    -> _STATUS_EXCLUDE (why).
#   - refund status ("did my refund go through")    -> _STATUS_EXCLUDE (refund/money).
#   - generic status asks ("status of ORD-1001")    -> no state participle; normal read-render.
#   - location asks ("where's my order")            -> no participle; the model reads.
_STATUS_EXCLUDE = re.compile(r"\b(?:why|refund|refunded|money|reimburse)\b", re.IGNORECASE)
_STATE_PARTICIPLE = r"(?:cancell?ed|shipped|delivered|arrived)"
_PLURAL_TOKEN = re.compile(
    r"\b(?:both|all|they|them|those|these|orders|purchases)\b", re.IGNORECASE
)

_STATUS_RULES: tuple[re.Pattern[str], ...] = (
    # explicit id: "is ORD-1002 cancelled?", "has order ORD-1001 shipped?"
    re.compile(
        rf"\b(?:is|was|has|have)\b[^.?!]*\bORD-\d+\b[^.?!]*\b{_STATE_PARTICIPLE}\b",
        re.I,
    ),
    # plural/collective verification: "so both of them are now canceled?", "are they all
    # cancelled?", "have both shipped?"
    re.compile(rf"\b(?:both|all|they|them|those|these)\b[^.?!]*\b{_STATE_PARTICIPLE}\b", re.I),
    # singular state check: "is it cancelled?", "has my order shipped?", "was that delivered?"
    re.compile(
        rf"\b(?:is|was|has|have|are)\b\s+(?:it|this|that|they|both|my\s+(?:order|purchase|"
        rf"package)|the\s+order|the\s+orders)\b[^.?!]*\b{_STATE_PARTICIPLE}\b",
        re.I,
    ),
    # "did it ship?" / "did my order arrive?" / "did that get cancelled?"
    re.compile(
        rf"\bdid\b\s+(?:it|that|this|they|both|my\s+(?:order|purchase|package)|the\s+order)\b"
        rf"[^.?!]*\b(?:ship|arrive|{_STATE_PARTICIPLE}|get\s+{_STATE_PARTICIPLE})\b",
        re.I,
    ),
)
# "double check" / "can you check again" — a verification imperative; the mode (list vs one)
# comes from whether the utterance carries a plural/collective token.
_DOUBLE_CHECK = re.compile(
    r"\b(?:double[-\s]?check|check\b[^.?!]{0,40}\bagain|confirm)\b",
    re.IGNORECASE,
)
_ORDER_REFERENCE = re.compile(
    r"\b(?:order|purchase|package|both|all|they|them|those|these|ORD-\d+)\b",
    re.IGNORECASE,
)


def status_check(text: str) -> Literal["one", "list"] | None:
    """Detect a STATE-verification question and its scope, else None.

    "one"  -> the caller asks whether ONE order is in a state (resolve via a stated id / the
              session pointer).
    "list" -> the caller asks about several / "both" / "all" / "my orders".
    None   -> NOT a state-verification question; the normal pipeline handles it (gate/model).
              None is not "safe" — it means "no high-certainty read opinion here."

    Deliberately narrow (precision over recall): a false positive steals a turn the model
    would have handled (a bug this feature owns), a false negative is today's behavior.
    """
    stripped = text.strip()
    if not stripped or _STATUS_EXCLUDE.search(stripped):
        return None
    if any(rule.search(stripped) for rule in _STATUS_RULES):
        return "list" if _PLURAL_TOKEN.search(stripped) else "one"
    if _DOUBLE_CHECK.search(stripped) and _ORDER_REFERENCE.search(stripped):
        return "list" if _PLURAL_TOKEN.search(stripped) else "one"
    return None
