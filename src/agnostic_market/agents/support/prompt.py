"""Support model-facing text: the assemble-node instructions + order rendering.

PROMPT ONLY — what the support (reasoning-tier) model reads while turning a post-purchase
request into a `propose_refund(...)` or `propose_cancel(order_key)`. The model picks a KEY
into the code-narrowed order list; it never authors an order id, decides verification level,
reads a card number, or authorizes the effect. Eligibility (can this order be cancelled?),
step-up (SIM-swap check, OTP), the readback, consent, and the guardrail are CODE in flow.py.
"""

from __future__ import annotations

from agnostic_market.commerce.orders import OrderCandidate

_SUPPORT_INSTRUCTIONS = (
    "You help callers with post-purchase requests for {display_name}: REFUNDS and order "
    "CANCELLATIONS. Work out which of the numbered orders below the caller means (use only "
    "these), then:\n"
    "- To CANCEL an order they no longer want: call propose_cancel with the order number. "
    "Do NOT promise it will be cancelled and do NOT say whether it's too late - whether an "
    "order can still be cancelled is checked for you; just propose it.\n"
    "- To REFUND: call propose_refund with the order number, the amount in dollars, and "
    "where it goes: 'original' (back to how they paid), 'new_instrument' (a different card), "
    "or 'new_address'. NEVER ask for or repeat a card number - a new card uses one already "
    "on file, and the caller may need to verify their identity first; that is handled.\n"
    "Do not announce that a refund or cancellation is done - it is read back for "
    "confirmation. If anything is unclear, ask ONE short question. If the caller no longer "
    "wants either, or asks about something unrelated, call leave_support.\n"
    "Past orders:\n{orders}"
)


def render_orders(orders: list[OrderCandidate]) -> str:
    """The numbered order list the model chooses from (keyed 1..N; model-facing only —
    the spoken readback is authored separately in flow.py)."""
    return "\n".join(f"[{o.key}] {o.order_id} - {o.summary} (${o.total_usd:.2f})" for o in orders)


def compose_support_prompt(display_name: str, orders: list[OrderCandidate]) -> str:
    """The assemble node's SystemMessage body: instructions + the current order list."""
    return _SUPPORT_INSTRUCTIONS.format(display_name=display_name, orders=render_orders(orders))
