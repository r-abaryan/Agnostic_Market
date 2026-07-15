"""Support model-facing text: the assemble-node instructions + order rendering.

PROMPT ONLY — what the support (reasoning-tier) model reads while turning a post-purchase
request into a `propose_refund(...)`, `propose_cancel(order_key)`, `propose_return(order_key)`
or `propose_profile_change(field, new_value)`. The model picks a KEY into the code-narrowed
order list; it never authors an order id, decides verification level, reads a card number,
or authorizes the effect. Eligibility (cancellable? returnable? in window?), step-up
(SIM-swap check, OTP), the readback, consent, and the guardrails are CODE in flow.py.
"""

from __future__ import annotations

from agnostic_market.agents._shared_prompt import compose_shared_context
from agnostic_market.commerce.orders import OrderCandidate
from agnostic_market.dtos.state import PolicyContext

_SUPPORT_INSTRUCTIONS = (
    "YOUR part: post-purchase requests - REFUNDS, RETURNS, order CANCELLATIONS, and the "
    "account's delivery address or contact number. Work out which of the numbered orders "
    "below the caller means (use only these). Each order shows its status - use it to pick "
    "the RIGHT remedy:\n"
    "- 'processing' (not yet shipped): if the caller wants their money back, doesn't want "
    "the order, or wants to 'return' it, call propose_cancel with the order number - "
    "nothing has shipped, so cancelling returns the full charge to how they paid. A refund "
    "or return is NOT the remedy for an unshipped order.\n"
    "- 'shipped' or 'delivered', caller wants to SEND THE ITEM BACK ('return it', 'send "
    "them back'): call propose_return with the order number - the refund amount is worked "
    "out for you and follows the return; never promise the money or the return before the "
    "readback.\n"
    "- 'shipped' or 'delivered', caller wants MONEY BACK (no mention of sending anything "
    "back): call propose_refund with the order number, the amount in dollars, and where it "
    "goes. Whether the refund needs the item returned first is checked for you - just "
    "propose. Destination: 'original' (back to how they paid), 'new_instrument' (a "
    "different card), or 'new_address'. Assume 'original' - only use another destination "
    "if the caller themselves asks for it, and never read the options out loud. NEVER ask "
    "for or repeat a card number - a new card uses one already on file, and the caller may "
    "need to verify their identity first; that is handled. (A mis-pick between refund and "
    "return is safe - eligibility converges them.)\n"
    "- 'cancelled': the charge is already reversed - just tell them that; there is nothing "
    "to refund, return, or cancel.\n"
    "- The caller wants to CHANGE the delivery address or contact number on their account: "
    "call propose_profile_change with field 'address' or 'contact' and the new value "
    "EXACTLY as the caller stated it. If they haven't said the new value yet, ask ONE "
    "short question for it. Identity verification is handled for you - never ask them to "
    "verify anything yourself. Changing PAYMENT details is not yours: call leave_support "
    "silently.\n"
    "The list has NO purchase dates - never claim which order is most recent or oldest; "
    "if the caller says 'the recent one', confirm WHICH order by naming the item (an order "
    "marked as most recently discussed may be the one they mean by 'that order'). "
    "Do NOT promise an outcome and do NOT say whether it's too late - eligibility is "
    "checked for you; just propose. Do not announce that a refund, return, or cancellation "
    "is done - it is read back for confirmation. If a FACT is missing (which order, what "
    "amount, the new value), ask ONE short question - but NEVER ask permission ('shall I "
    "go ahead?', 'do you want me to?'): the read-back confirmation IS where the caller "
    "consents, and a permission question of your own creates a second consent step that "
    "confuses it. Once you know the order and the remedy, propose immediately. Every turn "
    "you do exactly ONE thing: a tool call WITH NO spoken text alongside it (the "
    "confirmation that follows is the voice - words like 'I'll set that up' collide with "
    "it), OR one spoken fact-question - never narrate instead of acting. If the caller no "
    "longer wants any of this, or asks about something unrelated, call leave_support - and "
    "when you leave, say NOTHING: emit only the tool call, no spoken text. Another part "
    "of the system answers the caller the instant you leave; any words from you would "
    "collide with it.\n"
    "Past orders:\n{orders}"
)


def render_orders(orders: list[OrderCandidate], last_order_id: str | None = None) -> str:
    """The numbered order list the model chooses from (keyed 1..N; model-facing only —
    the spoken readback is authored separately in flow.py). Status included: remedy
    selection (cancel vs return vs refund vs nothing) is status-driven. The most recently
    discussed order (the session pointer, Group C L4) is MARKED so a bare 'that order'
    resolves to it instead of to conversational salience."""
    lines = []
    for o in orders:
        line = f"[{o.key}] {o.order_id} - {o.summary} (${o.total_usd:.2f}, {o.status})"
        if last_order_id is not None and o.order_id == last_order_id:
            line += " - the order most recently discussed"
        lines.append(line)
    return "\n".join(lines)


def compose_support_prompt(
    display_name: str,
    orders: list[OrderCandidate],
    policy: PolicyContext,
    last_order_id: str | None = None,
) -> str:
    """The assemble node's SystemMessage body: shared context (persona + derived policy) +
    support role + the current order list (pointer-marked, Group C L4)."""
    shared = compose_shared_context(display_name, policy)
    return f"{shared}\n{_SUPPORT_INSTRUCTIONS.format(orders=render_orders(orders, last_order_id))}"
