"""Read-only voice tools over the order + cart stores (commerce owns the data).

The stores are built once at session build (no per-turn disk I/O, no blocking reads inside
async tool nodes) and replaced by the real SoR/catalog integration in Phase 4. An unknown
order id is a normal conversational outcome (spoken "not found"), not an exception.
`order_status` reads THROUGH the store so an order placed this session is immediately
queryable. `view_cart` reads the SAME session cart the cart flow mutates (Group B) — the
caller can ask "what's in my cart" and get a real answer at the frontline (a READ), while
cart MUTATIONS hand over to the cart flow. These tools are all READ-ONLY — the frontline's
structural safety invariant (no state-changing tools) is preserved.
"""

from __future__ import annotations

from langchain_core.tools import BaseTool, tool

from agnostic_market.commerce.cart import CartStore
from agnostic_market.commerce.orders import OrderStore, speak_lines


def build_voice_tools(store: OrderStore, cart: CartStore) -> list[BaseTool]:
    """The read-only tool set, closing over the session's order + cart stores. `cart` is the
    SAME instance the cart flow mutates (pass one instance to both this and the graph, or the
    frontline answers 'what's in my cart' from a different cart — split-brain)."""

    @tool
    def order_status(order_id: str) -> str:
        """Look up the current status of an order by its order id (e.g. 'ORD-1001')."""
        summary = store.order_summary(order_id)
        if summary is None:
            return (
                f"No order found with id {order_id!r}. Ask the caller to double-check the order id."
            )
        return summary

    @tool
    def catalog_search(query: str) -> str:
        """Search the product catalog for items matching a text query."""
        needle = query.strip().lower()
        matches = [
            f"{p.name} (sku {p.sku}, ${p.price_usd:.2f})"
            for p in store.fixture.products
            if needle and needle in p.name.lower()
        ]
        if not matches:
            # Tiny fixture catalog: list what DOES exist so the model steers the caller
            # to real items instead of inventing categories (observed live 2026-07-06).
            available = "; ".join(p.name for p in store.fixture.products)
            return f"No catalog items match {query!r}. The catalog carries: {available}."
        return "Matching items: " + "; ".join(matches)

    @tool
    def view_cart() -> str:
        """Show what's currently in the caller's cart (a READ — does not change anything)."""
        if cart.is_empty():
            return "The cart is empty."
        return f"The cart has {speak_lines(cart.view())} - ${cart.cart_total():.2f} total."

    return [order_status, catalog_search, view_cart]
