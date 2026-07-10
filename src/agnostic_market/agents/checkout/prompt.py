"""Checkout model-facing text: the assemble-node instructions + candidate rendering.

PROMPT ONLY — what the checkout (reasoning-tier) model reads while turning a request into a
`propose_order(candidate_key, quantity)`. The model picks a KEY into the code-narrowed
candidate list; it never authors a SKU, a price, or a total (those are code). Consent
classification, the readback, and the guardrail are CODE in flow.py, not here.
"""

from __future__ import annotations

from agnostic_market.commerce.orders import Candidate

_CHECKOUT_INSTRUCTIONS = (
    "You are completing a purchase for {display_name}. Determine WHICH item the caller "
    "wants and HOW MANY, using only the numbered options below. When you are certain, call "
    "propose_order with the option number and quantity - do not announce totals yourself; "
    "the order will be read back for confirmation. If anything is unclear, ask ONE short "
    "clarifying question. Every turn you must do exactly one of those two things - call a "
    "tool or ask the question. NEVER narrate instead of acting: no 'I'll get that set up', "
    "no 'working on it' - to the caller a sentence like that sounds like something "
    "happened when nothing did. If the caller no longer wants to buy or asks about "
    "something unrelated, call leave_checkout instead of forcing the purchase - and when "
    "you leave, say NOTHING: emit only the tool call, no spoken text. Another part of the "
    "system answers the caller the instant you leave; any words from you (an apology, a "
    "'let me get you to the right place') would collide with it.\n"
    "Current options:\n{candidates}"
)


def render_candidates(candidates: list[Candidate]) -> str:
    """The numbered option list the model chooses from (keyed 1..N; TTS never reads this —
    it's model-facing, the spoken readback is authored separately in flow.py)."""
    return "\n".join(f"[{c.key}] {c.name} - ${c.price_usd:.2f} each" for c in candidates)


def compose_checkout_prompt(display_name: str, candidates: list[Candidate]) -> str:
    """The assemble node's SystemMessage body: instructions + the current candidate list."""
    return _CHECKOUT_INSTRUCTIONS.format(
        display_name=display_name, candidates=render_candidates(candidates)
    )
