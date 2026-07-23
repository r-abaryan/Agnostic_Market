"""Cart model-facing text: the assemble-node instructions + candidate rendering (Group B).

PROMPT ONLY — what the cart (reasoning-tier) model reads while turning a request into a cart
tool call (`add_to_cart`/`remove_from_cart`/`set_quantity`/`review_cart`/`buy_now`/
`go_to_checkout`/`request_cart_clarification`/`leave_cart`). The model picks a KEY into the
code-narrowed candidate list;
it never authors a SKU, a price, or a total (those are code). Consent classification, the
readback, the guardrail, and the cart mutations are CODE in flow.py, not here.

Every rule below that reads like scar tissue IS scar tissue — each was paid for with a live
failure in checkout/support and must survive the fold into the cart flow (act-or-ask,
tool-calls-carry-no-text, no-narration, already-placed⇒leave, silent-leave).
"""

from __future__ import annotations

from agnostic_market.agents._shared_prompt import compose_shared_context
from agnostic_market.commerce.cart import CartStore
from agnostic_market.commerce.orders import Candidate, speak_lines
from agnostic_market.dtos.state import PolicyContext

_CART_INSTRUCTIONS = (
    "YOUR part: managing the caller's cart and placing the order. Use ONLY the numbered "
    "options below; the caller picks by item, you pick the option NUMBER. Each option is a "
    "single product with a name and a price - there are NO colors, sizes, styles, or other "
    "variants to choose. Never ask the caller which color/size/style they want (there is "
    "only one of each item); if they ask, tell them plainly the item comes as listed. If "
    "what they describe clearly matches an option, add it; only ask when you genuinely "
    "can't tell WHICH option they mean.\n"
    "- add an item: call add_to_cart with the option number and how many. Several items at "
    "once ('one of each', 'the shoes and the jacket') = several add_to_cart calls in the "
    "SAME turn, one per item - never just the first.\n"
    "- remove an item: remove_from_cart with the option number; change an amount: "
    "set_quantity (0 removes it). These batch with adds in the same turn too.\n"
    "- caller asks what's in the cart: call review_cart (do NOT recite it yourself).\n"
    "- caller wants to buy ONE item right now: buy_now with the option number and quantity.\n"
    "- caller is done shopping / says checkout / place the order: go_to_checkout (this places "
    "the WHOLE cart; the total is read back for confirmation - never announce it yourself).\n"
    "Every response contains tool calls and NO spoken text. Cart changes (add/remove/"
    "set_quantity) may be several calls in one response, one per item. Every other response "
    "contains exactly ONE call. When a required detail is missing, call "
    "request_cart_clarification: detail='action' when the requested cart operation is unclear, "
    "detail='item' when the product is unclear, or detail='quantity' when the amount is unclear. "
    "The system asks the question. review_cart, buy_now, go_to_checkout, "
    "request_cart_clarification, and leave_cart are always ONE call on their own. NEVER narrate "
    "instead of acting: narration can sound like something happened when nothing did.\n"
    "If the conversation already shows an order PLACED (an order number was announced) and "
    "the caller says 'complete'/'finish' the purchase, there is nothing to complete - they "
    "mean the order they already have: call leave_cart (do NOT re-add or re-place it). If "
    "a caller mentions a product but gives no clear cart action ('how about the shoes?'), ask "
    "what they want to do. If they explicitly say they are only browsing, comparing products, "
    "or asking what is available rather than changing the cart, call leave_cart. Also call "
    "leave_cart when the caller no longer wants to shop or asks about something unrelated "
    "(an order status, a refund, a policy question). When you leave, say NOTHING: emit only "
    "the tool call, no spoken text. Another part of the system answers the instant you leave; "
    "any words from you would collide with it.\n"
    "{cart_state}"
    "Current options:\n{candidates}"
)


def render_candidates(candidates: list[Candidate]) -> str:
    """The numbered option list the model chooses from (keyed 1..N; TTS never reads this —
    it's model-facing, the spoken lines are authored separately in flow.py)."""
    return "\n".join(f"[{c.key}] {c.name} - ${c.price_usd:.2f} each" for c in candidates)


def _render_cart_state(cart: CartStore) -> str:
    """The current cart, so the model knows what's already in it (a remove/set_quantity
    targets an existing line; 'checkout' with an empty cart is a clarify, not a place)."""
    if cart.is_empty():
        return "The cart is currently EMPTY.\n"
    return f"Currently in the cart: {speak_lines(cart.view())} (${cart.cart_total():.2f}).\n"


def compose_cart_prompt(
    display_name: str, candidates: list[Candidate], cart: CartStore, policy: PolicyContext
) -> str:
    """The assemble node's SystemMessage body: shared context (persona + derived policy) +
    cart role + the live cart state + the current candidate list."""
    shared = compose_shared_context(display_name, policy)
    body = _CART_INSTRUCTIONS.format(
        cart_state=_render_cart_state(cart), candidates=render_candidates(candidates)
    )
    return f"{shared}\n{body}"
