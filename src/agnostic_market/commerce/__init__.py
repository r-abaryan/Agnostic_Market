"""Commerce plane — order SoR access (Phase 3b: fixture-backed stub; real SoR Phase 4)."""

from agnostic_market.commerce.cart import CartStore
from agnostic_market.commerce.identity import (
    BoundIdentity,
    CallerIdentityStore,
    CustomerDirectory,
    CustomersFixture,
    assert_orders_have_customers,
    load_customers_fixture,
    order_mutation_allowed,
    order_read_allowed,
)
from agnostic_market.commerce.orders import (
    Candidate,
    OrdersFixture,
    OrderStore,
    PlacedOrder,
    load_orders_fixture,
    resolve_candidates,
    speak_lines,
    speak_quantity,
)
from agnostic_market.commerce.payment_instruments import (
    PaymentInstrumentDirectory,
    PaymentInstrumentEntry,
    PaymentInstrumentsFixture,
    assert_payment_instruments_have_customers,
    load_payment_instruments_fixture,
)

__all__ = [
    "BoundIdentity",
    "CallerIdentityStore",
    "Candidate",
    "CartStore",
    "CustomerDirectory",
    "CustomersFixture",
    "OrderStore",
    "OrdersFixture",
    "PaymentInstrumentDirectory",
    "PaymentInstrumentEntry",
    "PaymentInstrumentsFixture",
    "PlacedOrder",
    "assert_orders_have_customers",
    "assert_payment_instruments_have_customers",
    "load_customers_fixture",
    "load_orders_fixture",
    "load_payment_instruments_fixture",
    "order_mutation_allowed",
    "order_read_allowed",
    "resolve_candidates",
    "speak_lines",
    "speak_quantity",
]
