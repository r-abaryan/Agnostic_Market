"""Checkout model-facing text: the assemble-node instructions + candidate rendering.

PROMPT ONLY — what the checkout (reasoning-tier) model reads while turning a request into a
`propose_order(candidate_key, quantity)`. The model picks a KEY into the code-narrowed
candidate list; it never authors a SKU, a price, or a total (those are code). Consent
classification, the readback, and the guardrail are CODE in flow.py, not here.
"""

from __future__ import annotations

from agnostic_market.agents._shared_prompt import compose_shared_context
from agnostic_market.commerce.orders import Candidate
from agnostic_market.dtos.state import PolicyContext

_CHECKOUT_INSTRUCTIONS = (
    "YOUR part: completing a purchase. Determine WHICH item the caller "
    "wants and HOW MANY, using only the numbered options below. When you are certain, call "
    "propose_order with the option number and quantity - do not announce totals yourself; "
    "the order will be read back for confirmation. If anything is unclear, ask ONE short "
    "clarifying question. Every turn you do exactly ONE thing: a tool call WITH NO spoken "
    "text alongside it (the readback that follows is the voice - words like 'I'll get "
    "that set up' collide with it), OR that one spoken question. NEVER narrate instead "
    "of acting: alone, narration sounds like something happened when nothing did. "
    "If the conversation already shows this order PLACED (an "
    "order number was announced for it), there is nothing left to complete - do NOT "
    "propose it again; the caller asking to 'complete' or 'finish' an already-placed "
    "purchase is asking about the order they have, so call leave_checkout. If the caller "
    "no longer wants to buy or asks about something unrelated, call leave_checkout "
    "instead of forcing the purchase - and when you leave, say NOTHING: emit only the "
    "tool call, no spoken text. Another part of the "
    "system answers the caller the instant you leave; any words from you (an apology, a "
    "'let me get you to the right place') would collide with it.\n"
    "Current options:\n{candidates}"
)


def render_candidates(candidates: list[Candidate]) -> str:
    """The numbered option list the model chooses from (keyed 1..N; TTS never reads this —
    it's model-facing, the spoken readback is authored separately in flow.py)."""
    return "\n".join(f"[{c.key}] {c.name} - ${c.price_usd:.2f} each" for c in candidates)


def compose_checkout_prompt(
    display_name: str, candidates: list[Candidate], policy: PolicyContext
) -> str:
    """The assemble node's SystemMessage body: shared context (persona + derived policy) +
    checkout role + the current candidate list."""
    shared = compose_shared_context(display_name, policy)
    return f"{shared}\n{_CHECKOUT_INSTRUCTIONS.format(candidates=render_candidates(candidates))}"
