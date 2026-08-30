"""Commerce plane — order SoR access (Phase 3b: fixture-backed stub; real SoR Phase 4)."""

from agnostic_market.commerce.cart import CartStore
from agnostic_market.commerce.catalog import (
    Candidate,
    CatalogFixture,
    CatalogPort,
    CatalogProduct,
    CatalogProductSet,
    FixtureCatalog,
)
from agnostic_market.commerce.identity import (
    BoundIdentity,
    CallerIdentityStore,
    CustomerDirectory,
    CustomerDirectoryPort,
    CustomersFixture,
    assert_orders_have_customers,
    load_customers_fixture,
    order_mutation_allowed,
    order_read_allowed,
)
from agnostic_market.commerce.orders import (
    OrderPort,
    OrdersFixture,
    OrderStore,
    PlacedOrder,
    load_orders_fixture,
    speak_lines,
    speak_quantity,
)
from agnostic_market.commerce.payment_instruments import (
    PaymentInstrumentDirectory,
    PaymentInstrumentEntry,
    PaymentInstrumentPort,
    PaymentInstrumentsFixture,
    assert_payment_instruments_have_customers,
    load_payment_instruments_fixture,
)

__all__ = [
    "BoundIdentity",
    "CallerIdentityStore",
    "Candidate",
    "CartStore",
    "CatalogFixture",
    "CatalogPort",
    "CatalogProduct",
    "CatalogProductSet",
    "CustomerDirectory",
    "CustomerDirectoryPort",
    "CustomersFixture",
    "FixtureCatalog",
    "OrderPort",
    "OrderStore",
    "OrdersFixture",
    "PaymentInstrumentDirectory",
    "PaymentInstrumentEntry",
    "PaymentInstrumentPort",
    "PaymentInstrumentsFixture",
    "PlacedOrder",
    "assert_orders_have_customers",
    "assert_payment_instruments_have_customers",
    "load_customers_fixture",
    "load_orders_fixture",
    "load_payment_instruments_fixture",
    "order_mutation_allowed",
    "order_read_allowed",
    "speak_lines",
    "speak_quantity",
]
