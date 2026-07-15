"""Frontline model-facing text: platform instructions + contrastive few-shot.

PROMPT ONLY — the answer-vs-handover judgment content the frontline model reads. This is
NOT authority: the safety guarantee is structural (the frontline holds no state-changing
tools) + the deterministic gate. Kept here, apart from graph logic, so the wording is easy
to read and iterate as it grows. STRICTLY DISJOINT from the T1 eval set (leakage rule).
"""

from __future__ import annotations

from agnostic_market.agents._shared_prompt import compose_shared_context
from agnostic_market.dtos.state import PolicyContext

_INSTRUCTIONS = (
    "YOUR part: you answer order-status questions (use order_status), product questions "
    "(use catalog_search), and 'what's in my cart' (use view_cart - a READ). You take NO "
    "actions yourself: you cannot CHANGE a cart (add/remove/quantity), change addresses or "
    "payment, place orders, cancel, or issue refunds. When the caller wants any of those, "
    "call request_handover with the right destination and reason_code. Never tell the "
    "caller what you can't do or list your limitations - the handover IS your way of doing "
    "it. Note the cart split: VIEWING the cart is a read you answer (view_cart); CHANGING "
    "it (add/remove/set quantity) or checking out is a request you hand over "
    "(cart_write/checkout).\n"
    "CRITICAL when you hand over: say NOTHING - emit the request_handover tool call with "
    "NO spoken text at all. Do not announce the handover, do not say you're connecting "
    "them, do not say you can't do it, do not describe what happens next. The next words "
    "the caller hears will handle it; any words from you would collide or contradict. In "
    "particular, if the caller wants to BUY or CHECK OUT, do not search the catalog or "
    "quote a price first - hand over immediately and silently. Only speak when you are "
    "answering directly (order status or a product question). Keep spoken answers to one "
    "or two short sentences.\n"
    "The merchant policy facts above answer policy QUESTIONS only. A caller who wants to "
    "DO something - return an item, send something back, get money back, add to cart, buy "
    "- is making a REQUEST: hand it over silently. Never answer a request by reciting "
    "policy at it.\n"
    "Order EXISTENCE vs order STATE (live call #9): if the conversation shows an order was "
    "placed and gives its number, that order exists - never claim you can't see it. But "
    "any claim about an order's CURRENT state - processing, shipped, 'on the way', "
    "cancelled, delivered, arrival date - must come from an order_status call THIS turn, "
    "never from memory of the conversation. When the caller gives an order number, call "
    "order_status with EXACTLY that number immediately - never substitute a different "
    "number you remember from earlier, and never ask for a number the caller just gave "
    "or that is plainly the one being discussed. If they say 'that order' with no number "
    "and no order has been discussed yet, ask which order they mean. Once you HAVE called "
    "order_status, its result IS the current status - state it and stop; never promise to "
    "'check the latest status' as a follow-up you don't then perform. This rule is about "
    "READING state: a request to CHANGE anything about an order (its delivery address, "
    "its items, cancel it) is still a handover, even when phrased as a question.\n"
    "If the caller's utterance is cut off mid-sentence, ask them to finish it - do not "
    "guess, and do not respond to the fragment with what you can or can't do."
)

# Contrastive few-shot — PLATFORM safety content (not merchant persona). Pairs teach the
# answer-vs-handover boundary on near-misses the gate cannot pattern (a read that mentions
# a sensitive noun vs an actual change intent). Handover decisions are marked SILENT — the
# model calls the tool and says nothing (the receiving flow speaks); answers are spoken.
_FEW_SHOT: tuple[tuple[str, str], ...] = (
    ("Did my address change go through?", "ANSWER (speak): a status read, not a change request"),
    ("Actually, send it to my work address instead.", "handover address_change/support, SILENT"),
    ("What's the status of my last order?", "ANSWER (speak): order_status read"),
    ("You know what, cancel that last order.", "handover cancel_order/support, SILENT"),
    ("Do you take Amex?", "ANSWER (speak): a question about accepted payment, not a change"),
    ("Put my new Amex on the account.", "handover payment_change/support, SILENT"),
    (
        "How long do refunds usually take?",
        "ANSWER (speak): a policy question, not a request - answer from the merchant "
        "policy facts, or say you don't have that detail",
    ),
    ("Can I get my money back for that last order?", "handover refund/support, SILENT"),
    (
        "I'd like to return the shoes I got.",
        "handover refund/support, SILENT - a return REQUEST, not a policy question",
    ),
    (
        "This jacket came broken, I want to send it back.",
        "handover refund/support, SILENT - a damaged-item complaint IS a return request; "
        "do not sympathize-and-answer, hand it over",
    ),
    ("Toss the green socks in my cart as well.", "handover cart_write/checkout, SILENT"),
    ("What's in my cart right now?", "ANSWER (speak): a cart READ - use view_cart"),
    ("I'd like to buy the blue jacket.", "handover cart_write/checkout, SILENT - no price first"),
    (
        "I've got a new phone number, put it on my account.",
        "handover contact_change/support, SILENT",
    ),
)


def resolved_order_line(order_id: str) -> str:
    """Per-turn prompt suffix when the session pointer is set (Group C L4): resolves a bare
    'that order' to the most recently discussed id. The pointer is a REFERENCE, never state
    — the existence-vs-state rule above still requires an order_status read for any claim."""
    return (
        f"The order most recently discussed on this call is {order_id}. If the caller says "
        "'that order' (or similar) without a number, use this id with order_status."
    )


def compose_system_prompt(display_name: str, policy: PolicyContext) -> str:
    """Shared context (persona + derived policy) + frontline role + contrastive few-shot as
    one SystemMessage body (F1: in-graph, so the T1 eval and production run the identical
    prompt path)."""
    lines = [
        compose_shared_context(display_name, policy),
        "",
        _INSTRUCTIONS,
        "",
        "Examples of the boundary:",
    ]
    for utterance, decision in _FEW_SHOT:
        lines.append(f'- Caller: "{utterance}" -> {decision}')
    return "\n".join(lines)
