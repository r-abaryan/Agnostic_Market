"""Read-only voice tools over the order store (commerce/orders.py owns the data).

The store is built once at session build from the merchant's fixture (no per-turn disk
I/O, no blocking reads inside async tool nodes) and replaced by the real SoR/catalog
integration in Phase 4. An unknown order id is a normal conversational outcome (spoken
"not found"), not an exception. `order_status` reads THROUGH the store so an order placed
by the checkout flow this session is immediately queryable.
"""

from __future__ import annotations

from langchain_core.tools import BaseTool, tool

from agnostic_market.commerce.orders import OrderStore


def build_voice_tools(store: OrderStore) -> list[BaseTool]:
    """The read-only tool set, closing over the session's order store."""

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

    return [order_status, catalog_search]
