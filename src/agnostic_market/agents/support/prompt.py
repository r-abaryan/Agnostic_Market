"""Support model-facing text: the assemble-node instructions + order rendering.

PROMPT ONLY — what the support (reasoning-tier) model reads while turning a post-purchase
request into a `propose_refund(...)`, `propose_cancel(order_keys|scope)`,
`propose_return(order_key)` or `propose_profile_change(field, new_value)`. The model picks
a KEY into the code-narrowed
order list — scoped to the caller's AUTHORIZED orders (live call #15: an unscoped list was
recited to an unverified caller by a clarify question; the model can't speak what it never
saw) — or, for an order NOT listed, relays the caller's STATED order number (the guest
path; `order_status` precedent — the model still never AUTHORS an id, and code resolves the
stated one fail-closed). It never decides verification level, reads a card number, or
authorizes the effect. Eligibility (cancellable? returnable? in window?), step-up (SIM-swap
check, OTP), the readback, consent, and the guardrails are CODE in flow.py.
"""

from __future__ import annotations

from agnostic_market.agents._shared_prompt import compose_shared_context
from agnostic_market.commerce.orders import OrderCandidate
from agnostic_market.dtos.orchestration import (
    CancelOrders,
    ChangeProfile,
    RefundOrder,
    ReturnOrder,
)
from agnostic_market.dtos.state import PolicyContext


def render_orders(orders: list[OrderCandidate], last_order_id: str | None = None) -> str:
    """The numbered order list the model chooses from (model-facing only — the spoken
    readback is authored separately in flow.py). The caller receives ONLY authorized
    candidates (the assemble node scopes them; call #15), so an empty list means nothing
    is verified yet — the placeholder keeps the model on the ask-for-the-order-number
    path without implying any count. Status included: remedy selection (cancel vs return
    vs refund vs nothing) is status-driven. The most recently discussed order (from bounded
    recent-order context, Group C L4) is MARKED so a bare 'that order' resolves to it instead of to
    conversational salience."""
    if not orders:
        return "(none verified for this caller yet - ask for the order number to act on one)"
    lines = []
    for o in orders:
        line = f"[{o.key}] {o.order_id} - {o.summary} (${o.total_usd:.2f}, {o.status})"
        if last_order_id is not None and o.order_id == last_order_id:
            line += " - the order most recently discussed"
        lines.append(line)
    return "\n".join(lines)


def compose_support_capability_prompt(
    display_name: str,
    orders: list[OrderCandidate],
    policy: PolicyContext,
    request: CancelOrders | RefundOrder | ReturnOrder | ChangeProfile,
    proposal_tool_name: str,
    last_order_id: str | None = None,
) -> str:
    """Narrow slot-gathering prompt for one already-selected Support capability."""

    shared = compose_shared_context(display_name, policy)
    retained = request.model_dump_json(exclude_none=True)
    instructions = (
        f"{shared}\n"
        f"You are gathering missing fields for exactly one {request.kind.value} request. "
        f"The retained typed request is {retained}. Every field already present is fixed: never "
        "replace its target, scope, amount, destination, or profile field. Use the caller's newest "
        "message only to fill a missing field. When enough information is present, call exactly "
        f"{proposal_tool_name}. If a required fact is still missing, call "
        "request_support_clarification. "
        "If the caller changes topic or abandons this request, call leave_support. Emit exactly "
        "one tool call and no prose. Never claim an effect completed or ask for verification."
    )
    if isinstance(request, ChangeProfile):
        # A profile change names no order, so the candidate block's "ask for the order
        # number" placeholder would be the wrong question for this capability entirely.
        return instructions
    return f"{instructions}\nAuthorized order candidates:\n{render_orders(orders, last_order_id)}"
