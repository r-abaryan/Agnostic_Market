"""Read-only voice tools over the order + cart stores (commerce owns the data).

The stores are built once at session build (no per-turn disk I/O, no blocking reads inside
async tool nodes) and replaced by the real SoR/catalog integration in Phase 4. `order_status`
reads THROUGH the store so an order placed this session is immediately queryable. `view_cart`
reads the SAME session cart the cart flow mutates (Group B). These tools are all READ-ONLY —
the frontline's structural safety invariant (no state-changing tools) is preserved; the
rung-1 grant `order_status` records is authorization bookkeeping, not commerce state.

P7 OBJECT BINDING (order_status): low-entropy order ids (ORD-1001) must not be readable by
anyone who can count — the guest-lookup pair (order id + the account contact, code-matched)
gates the read. The gate is IN THE TOOL BODY, fail-closed: an unbound ask without contact
returns one neutral question; a mismatch OR unknown order returns one combined not-found
line. A bound principal cannot consume or acquire a foreign guest grant and receives one
access-neutral line instead. The tool never confirms an order id exists or who owns it. A
contact match grants exactly THAT order for an unbound session; account-wide enumeration
(`list_orders`) requires the OTP-bound identity flow (rung 2).
"""

from __future__ import annotations

from langchain_core.tools import BaseTool, tool

from agnostic_market.agents._copy import ACCOUNT_CONTACT_QUESTION
from agnostic_market.agents.telemetry import write_event
from agnostic_market.commerce.cart import CartStore
from agnostic_market.commerce.identity import (
    CallerIdentityStore,
    CustomerDirectory,
    order_read_allowed,
    try_grant_orders_by_contact,
)
from agnostic_market.commerce.orders import (
    BOUND_ORDER_READ_UNAVAILABLE_LINE,
    ORDER_CONTACT_NOT_FOUND_LINE,
    OrderStore,
    RecentOrderContext,
    lookup_catalog,
    render_order_list_line,
    speak_lines,
)

# The model-facing decline strings (fail-closed: NO order data, no existence confirmation).
_ASK_CONTACT = (
    "Not verified for that order. Do not say whether the order exists. Ask the caller ONE "
    f'short question exactly: "{ACCOUNT_CONTACT_QUESTION}" Then call order_status again with '
    "the order id AND account_contact."
)
_COMBINED_NOT_FOUND = (
    "No order matches those details. Do not say which detail failed. Tell the caller exactly: "
    f'"{ORDER_CONTACT_NOT_FOUND_LINE}"'
)
_BOUND_ORDER_READ_DENIED = (
    "The caller is already verified on this call, but this order read is not authorized. "
    "Do not say whether the order exists or which account owns it. Tell the caller exactly: "
    f'"{BOUND_ORDER_READ_UNAVAILABLE_LINE}"'
)
# The "do not ask yourself" sentence is load-bearing (live call #12 F-12.3: the frontline
# asked for the email itself, duplicating the identity flow's question and wasting a turn).
_UNVERIFIED_LIST = (
    "The caller is not verified on this call. Do not list, guess, or count any orders. "
    "Do NOT ask the caller for their email or phone yourself - the verification step asks "
    "for that. Call request_handover with destination 'support' and reason_code "
    "'list_orders', silently - no spoken text at all."
)


def build_voice_tools(
    store: OrderStore,
    cart: CartStore,
    recent_orders: RecentOrderContext,
    identity: CallerIdentityStore,
    customers: CustomerDirectory,
) -> list[BaseTool]:
    """The read-only tool set, closing over the session's stores. `cart`, `recent_orders`, and
    `identity` are the SAME instances the flows mutate (pass one instance to both this and
    the graph, or the frontline reads different session state than the flows write —
    split-brain). All params REQUIRED on purpose: a call site that silently built its own
    identity store would be exactly that split-brain."""

    @tool
    def order_status(order_id: str, account_contact: str = "") -> str:
        """Look up the current status of an order by its order id (e.g. 'ORD-1001').
        account_contact: the email or phone number the caller says is on the account -
        include it when the previous result asked for verification."""
        if order_read_allowed(order_id, store=store, identity=identity):
            summary = store.order_summary(order_id)
            if summary is None:
                # Unreachable for a granted/owned/placed id, but never leak on a logic gap.
                return _COMBINED_NOT_FOUND
            recent_orders.record([order_id], operation="read")
            return summary
        if identity.current() is not None:
            write_event(
                {
                    "event": "order_read_denied",
                    "order_id_known": store.order_owner(order_id) is not None,
                }
            )
            return _BOUND_ORDER_READ_DENIED
        claim = account_contact.strip()
        if not claim:
            write_event({"event": "order_read_denied", "order_id_known": False})
            return _ASK_CONTACT
        if (
            try_grant_orders_by_contact(
                claim, order_id, store=store, customers=customers, identity=identity
            )
            == "mismatch"
        ):
            # ONE combined response for wrong-pair AND unknown-order: which detail failed
            # is never revealed (existence-oracle discipline). `order_id_known` is
            # telemetry-internal (abuse monitoring), never spoken — a separate owner read
            # (not the grant decision) so the shared verdict stays wording-agnostic.
            write_event(
                {
                    "event": "order_read_denied",
                    "order_id_known": store.order_owner(order_id) is not None,
                }
            )
            return _COMBINED_NOT_FOUND
        write_event(
            {
                "event": "order_read_granted",
                "order_id": order_id.strip().upper(),
                "method": "contact_match",
            }
        )
        summary = store.order_summary(order_id)
        assert summary is not None  # the owner lookup just found it
        # A FOUND, AUTHORIZED order becomes "the order most recently discussed" (Group C
        # L4) — never set on a declined read (a probe must not hijack "that order").
        recent_orders.record([order_id], operation="read")
        return summary

    @tool
    def list_orders() -> str:
        """List the orders on the caller's account. Requires the caller to be verified -
        call this FIRST for any 'what orders do I have' ask; the result says what to do."""
        bound = identity.current()
        if bound is None:
            return _UNVERIFIED_LIST
        orders = store.owned_orders(bound.customer_ref)
        if orders:
            recent_orders.record([order.order_id for order in orders], operation="list")
        else:
            recent_orders.clear()
        return render_order_list_line(orders)

    @tool
    def catalog_search(query: str) -> str:
        """Search the product catalog for items matching a text query."""
        result = lookup_catalog(store.fixture, query)
        matches = [f"{p.name} (sku {p.sku}, ${p.price_usd:.2f})" for p in result.matches]
        if not matches:
            # Tiny fixture catalog: list what DOES exist so the model steers the caller
            # to real items instead of inventing categories (observed live 2026-07-06).
            available = "; ".join(p.name for p in result.available)
            return f"No catalog items match {query!r}. The catalog carries: {available}."
        return "Matching items: " + "; ".join(matches)

    @tool
    def view_cart() -> str:
        """Show what's currently in the caller's cart (a READ — does not change anything)."""
        if cart.is_empty():
            return "The cart is empty."
        return f"The cart has {speak_lines(cart.view())} - ${cart.cart_total():.2f} total."

    return [order_status, list_orders, catalog_search, view_cart]
