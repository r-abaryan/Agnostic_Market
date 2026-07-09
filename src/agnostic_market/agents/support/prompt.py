"""Support model-facing text: the assemble-node instructions + order rendering.

PROMPT ONLY — what the support (reasoning-tier) model reads while turning a refund request
into a `propose_refund(order_key, amount_usd, destination)`. The model picks a KEY into the
code-narrowed order list; it never authors an order id, and it never decides verification
level, reads a card number, or authorizes the refund. Step-up (SIM-swap check, OTP), the
readback, consent classification, and the guardrail are CODE in flow.py, not here.
"""

from __future__ import annotations

from agnostic_market.commerce.orders import OrderCandidate

_SUPPORT_INSTRUCTIONS = (
    "You are helping a caller with a REFUND for {display_name}. Work out WHICH past order "
    "they mean (use only the numbered orders below), HOW MUCH to refund, and WHERE it "
    "should go: 'original' (back to how they paid) or 'new_instrument' (a different card) "
    "or 'new_address'. When you are certain, call propose_refund with the order number, the "
    "amount in dollars, and the destination. NEVER ask for or repeat a card number - a "
    "refund to a new card uses a card already on file, and the caller may have to verify "
    "their identity first; that is handled for you. Do not announce that the refund is done "
    "- it will be read back for confirmation. If anything is unclear, ask ONE short "
    "question. If the caller no longer wants a refund or asks about something unrelated, "
    "call leave_support instead of forcing it.\n"
    "Past orders:\n{orders}"
)


def render_orders(orders: list[OrderCandidate]) -> str:
    """The numbered order list the model chooses from (keyed 1..N; model-facing only —
    the spoken readback is authored separately in flow.py)."""
    return "\n".join(f"[{o.key}] {o.order_id} - {o.summary} (${o.total_usd:.2f})" for o in orders)


def compose_support_prompt(display_name: str, orders: list[OrderCandidate]) -> str:
    """The assemble node's SystemMessage body: instructions + the current order list."""
    return _SUPPORT_INSTRUCTIONS.format(display_name=display_name, orders=render_orders(orders))
