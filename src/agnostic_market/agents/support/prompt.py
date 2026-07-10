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
    "these). Each order shows its status - use it to pick the RIGHT remedy:\n"
    "- 'processing' (not yet shipped): if the caller wants their money back or doesn't "
    "want the order, call propose_cancel with the order number - cancelling returns the "
    "full charge to how they paid. A refund is NOT the remedy for an unshipped order.\n"
    "- 'shipped' or 'delivered': call propose_refund with the order number, the amount in "
    "dollars, and where it goes: 'original' (back to how they paid), 'new_instrument' (a "
    "different card), or 'new_address'. Assume 'original' - only use another destination "
    "if the caller themselves asks for it, and never read the options out loud. NEVER ask "
    "for or repeat a card number - a new card uses one already on file, and the caller may "
    "need to verify their identity first; that is handled.\n"
    "- 'cancelled': the charge is already reversed - just tell them that; there is nothing "
    "to refund or cancel.\n"
    "Do NOT promise an outcome and do NOT say whether it's too late - eligibility is "
    "checked for you; just propose. Do not announce that a refund or cancellation is done "
    "- it is read back for confirmation. If anything is unclear, ask ONE short question. "
    "Every turn you must either call a tool or ask that question - never narrate instead "
    "of acting ('I'll take care of that' sounds like something happened when nothing "
    "did). If the caller no longer wants either, or asks about something unrelated, call "
    "leave_support - and when you leave, say NOTHING: emit only the tool call, no spoken "
    "text. Another part of the system answers the caller the instant you leave; any words "
    "from you would collide with it.\n"
    "Past orders:\n{orders}"
)


def render_orders(orders: list[OrderCandidate]) -> str:
    """The numbered order list the model chooses from (keyed 1..N; model-facing only —
    the spoken readback is authored separately in flow.py). Status included: remedy
    selection (cancel vs refund vs nothing) is status-driven."""
    return "\n".join(
        f"[{o.key}] {o.order_id} - {o.summary} (${o.total_usd:.2f}, {o.status})"
        for o in orders
    )


def compose_support_prompt(display_name: str, orders: list[OrderCandidate]) -> str:
    """The assemble node's SystemMessage body: instructions + the current order list."""
    return _SUPPORT_INSTRUCTIONS.format(display_name=display_name, orders=render_orders(orders))
