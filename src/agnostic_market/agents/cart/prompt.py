"""Cart model-facing instructions and candidate rendering.

The cart model fills the missing typed request fields, asks for clarification, or leaves the
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
    "YOUR part: fill every missing field of the active cart request that the caller supplied in "
    "this turn, in one call. The operation and every field already marked FIXED below must never "
    "be replaced or repeated. Every response contains tool calls and NO spoken text. Call "
    "{expected_tool} with each missing field the caller gave, omitting any field they did not. "
    "If they supplied none of the missing fields clearly, call request_cart_clarification. If "
    "they changed subject or no longer want this cart request, call leave_cart. Emit exactly "
    "one tool call.\n"
    "An option marked JUST OFFERED is one the assistant named in its own previous turn. "
    "Choose it when the caller refers back to it rather than naming a product, for example "
    "by agreeing, or by saying those, that one, or it. When the caller names a different "
    "product, choose the option they named instead: a prior offer never overrides what they "
    "just asked for.\n"
    "When more than one option is marked JUST OFFERED and the caller's reply does not "
    "single one out, call request_cart_clarification rather than picking one.\n"
    "A JUST REFERENCED option was discussed, but was not necessarily offered to add. Use it "
    "only when the caller explicitly asks to add that product. Bare agreement such as yes "
    "or go ahead cannot select JUST REFERENCED; without a JUST OFFERED option, ask for "
    "clarification. If several JUST REFERENCED options fit and the caller does not single "
    "one out, ask which item rather than picking. An explicitly named different item wins.\n"
    "Active operation: {operation}. Fixed item: {item_state}. Fixed quantity: {quantity_state}.\n"
    "Current code-bounded options:\n{candidates}"
)


def render_candidates(
    candidates: list[Candidate],
    offered_skus: tuple[str, ...] = (),
    referenced_skus: tuple[str, ...] = (),
) -> str:
    """The numbered option list the model chooses from (keyed 1..N; TTS never reads this -
    it's model-facing, the spoken lines are authored separately in flow.py).

    A just-offered option is MARKED, never substituted: the caller may be referring back to
    it or naming something else entirely, and only their words settle which.
    """
    offered = set(offered_skus)
    referenced = set(referenced_skus)
    return "\n".join(
        f"[{c.key}] {c.name} - ${c.price_usd:.2f} each"
        + (" (JUST OFFERED)" if c.sku in offered else "")
        + (" (JUST REFERENCED)" if c.sku in referenced else "")
        for c in candidates
    )


def compose_cart_capability_prompt(
    display_name: str,
    candidates: list[Candidate],
    policy: PolicyContext,
    request: ModifyCart,
    expected_tool: str,
    offered_skus: tuple[str, ...] = (),
    referenced_skus: tuple[str, ...] = (),
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
        candidates=render_candidates(candidates, offered_skus, referenced_skus) or "(none)",
    )
    return f"{compose_shared_context(display_name, policy)}\n{body}"
