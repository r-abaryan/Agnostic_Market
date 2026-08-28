"""Cart model-facing instructions and candidate rendering.

The cart model fills one missing typed request field, asks for clarification, or leaves the
capability. It selects only code-issued keys and never authors a SKU, price, total, or effect.

Every rule below that reads like scar tissue IS scar tissue — each was paid for with a live
failure in checkout/support and must survive the fold into the cart flow (act-or-ask,
tool-calls-carry-no-text, no-narration, already-placed⇒leave, silent-leave).
"""

from __future__ import annotations

from agnostic_market.agents._shared_prompt import compose_shared_context
from agnostic_market.commerce.catalog import Candidate
from agnostic_market.dtos.orchestration import (
    CartItemChoices,
    ModifyCart,
    ResolvedCartItemRef,
)
from agnostic_market.dtos.state import PolicyContext

_CART_CAPABILITY_INSTRUCTIONS = (
    "YOUR part: fill exactly the one missing field of the active cart request. The operation and "
    "every supplied field are FIXED; never replace them. Every response contains tool calls and "
    "NO spoken text. Call {expected_tool} when the caller supplied the missing field. If the "
    "caller "
    "did not supply it clearly, call request_cart_clarification. If they changed subject or no "
    "longer want this cart request, call leave_cart. Emit exactly one tool call.\n"
    "Active operation: {operation}. Fixed item: {item_state}. Fixed quantity: {quantity_state}.\n"
    "Current code-bounded options:\n{candidates}"
)


def render_candidates(candidates: list[Candidate]) -> str:
    """The numbered option list the model chooses from (keyed 1..N; TTS never reads this —
    it's model-facing, the spoken lines are authored separately in flow.py)."""
    return "\n".join(f"[{c.key}] {c.name} - ${c.price_usd:.2f} each" for c in candidates)


def compose_cart_capability_prompt(
    display_name: str,
    candidates: list[Candidate],
    policy: PolicyContext,
    request: ModifyCart,
    expected_tool: str,
) -> str:
    """Tool-only prompt for one missing typed Cart slot; code owns options and rendering."""
    if isinstance(request.item, CartItemChoices):
        item_state = "code-bounded ambiguous choices"
    elif isinstance(request.item, ResolvedCartItemRef):
        item_state = candidates[0].name if len(candidates) == 1 else "code-resolved item"
    else:
        item_state = "missing"
    quantity_state = "missing" if request.quantity is None else str(request.quantity)
    body = _CART_CAPABILITY_INSTRUCTIONS.format(
        expected_tool=expected_tool,
        operation=request.operation,
        item_state=item_state,
        quantity_state=quantity_state,
        candidates=render_candidates(candidates) or "(none)",
    )
    return f"{compose_shared_context(display_name, policy)}\n{body}"
