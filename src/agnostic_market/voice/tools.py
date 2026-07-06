"""Phase-2 read-only tools + the orders fixture they answer from.

The fixture (config/fixtures/orders/<merchant_id>.yaml) is a stub order-SoR/catalog for
the minimal voice loop — PRELOADED once at session build (no per-turn disk I/O, no
blocking reads inside async tool nodes) and replaced by the real SoR/catalog integration
in Phase 4. An unknown order id is a normal conversational outcome (spoken "not found"),
not an exception; a malformed fixture fails loudly at build, never mid-call.

Fixture shapes are LOCAL to this module on purpose: the real commerce DTOs land in
Phase 3/4 (pre-inventing them in dtos/ would create a second source of truth).
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agnostic_market.config.loader import ConfigError, load_yaml_layer

_STRICT = ConfigDict(extra="forbid")


class _OrderEntry(BaseModel):
    model_config = _STRICT

    status: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    eta: str = Field(min_length=1)


class _ProductEntry(BaseModel):
    model_config = _STRICT

    sku: str = Field(min_length=1)
    name: str = Field(min_length=1)
    price_usd: float = Field(ge=0)


class OrdersFixture(BaseModel):
    """Validated stub SoR content for one merchant."""

    model_config = _STRICT

    orders: dict[str, _OrderEntry]
    products: list[_ProductEntry] = Field(min_length=1)


def load_orders_fixture(config_root: Path, merchant_id: str) -> OrdersFixture:
    """Load + validate the merchant's orders fixture. Fails loudly (build time, not mid-call)."""
    path = config_root / "fixtures" / "orders" / f"{merchant_id}.yaml"
    try:
        return OrdersFixture.model_validate(load_yaml_layer(path))
    except ValidationError as exc:
        raise ConfigError(f"orders fixture {path} failed validation:\n{exc}") from exc


def build_voice_tools(fixture: OrdersFixture) -> list[BaseTool]:
    """The Phase-2 read-only tool set, closing over the preloaded fixture data."""

    @tool
    def order_status(order_id: str) -> str:
        """Look up the current status of an order by its order id (e.g. 'ORD-1001')."""
        entry = fixture.orders.get(order_id.strip().upper())
        if entry is None:
            return (
                f"No order found with id {order_id!r}. Ask the caller to double-check the order id."
            )
        return f"Order {order_id}: {entry.summary} - status {entry.status}, ETA {entry.eta}."

    @tool
    def catalog_search(query: str) -> str:
        """Search the product catalog for items matching a text query."""
        needle = query.strip().lower()
        matches = [
            f"{p.name} (sku {p.sku}, ${p.price_usd:.2f})"
            for p in fixture.products
            if needle and needle in p.name.lower()
        ]
        if not matches:
            # Tiny fixture catalog: list what DOES exist so the model steers the caller
            # to real items instead of inventing categories (observed live 2026-07-06).
            available = "; ".join(p.name for p in fixture.products)
            return f"No catalog items match {query!r}. The catalog carries: {available}."
        return "Matching items: " + "; ".join(matches)

    return [order_status, catalog_search]
