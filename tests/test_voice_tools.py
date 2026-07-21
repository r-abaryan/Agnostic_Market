"""Orders-fixture preload + the read-only tools it feeds (zero network).

P7 object binding: `order_status` is gated by the guest-lookup pair (order id + account
contact, code-matched) — the gate lives in the tool body, fail-closed, with ONE combined
not-found response so an order id's existence is never confirmed. `list_orders` (rung 2)
fails closed until the identity flow binds the session.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.tools import BaseTool

from agnostic_market.commerce.cart import CartStore
from agnostic_market.commerce.identity import (
    BoundIdentity,
    CallerIdentityStore,
    CustomerDirectory,
    load_customers_fixture,
)
from agnostic_market.commerce.orders import OrderStore, RecentOrderContext, load_orders_fixture
from agnostic_market.config.loader import ConfigError
from agnostic_market.dtos.state import CartLine
from agnostic_market.voice.tools import build_voice_tools

# The fixture pair (config/fixtures/customers/acme_store.yaml): CUST-001 owns ORD-1001 +
# ORD-1003 (phone on file); CUST-002 owns ORD-1002 (email on file).
_CUST1_PHONE = "+1 555 010 0119"
_CUST2_EMAIL = "casey@example.com"


class Harness:
    def __init__(self, config_root: Path, cart: CartStore | None = None) -> None:
        self.store = OrderStore(load_orders_fixture(config_root, "acme_store"))
        self.recent_orders = RecentOrderContext(max_refs=10)
        self.identity = CallerIdentityStore()
        customers = CustomerDirectory(load_customers_fixture(config_root, "acme_store"))
        self.tools: dict[str, BaseTool] = {
            t.name: t
            for t in build_voice_tools(
                self.store, cart or CartStore(), self.recent_orders, self.identity, customers
            )
        }

    def status(self, order_id: str, contact: str = "") -> str:
        args: dict = {"order_id": order_id}
        if contact:
            args["account_contact"] = contact
        return self.tools["order_status"].invoke(args)


def test_fixture_loads_and_validates(config_root: Path) -> None:
    fixture = load_orders_fixture(config_root, "acme_store")
    assert "ORD-1001" in fixture.orders
    assert fixture.orders["ORD-1001"].customer_ref == "CUST-001"
    assert fixture.products


# --- the object-binding gate (P7 rung 1) ---------------------------------------------


def test_order_status_fails_closed_without_a_verifier(config_root: Path) -> None:
    # No contact, unbound session -> NO order data, NO existence confirmation; the result
    # instructs the ask-for-contact exchange. The pointer is untouched (a probe must not
    # hijack "that order").
    h = Harness(config_root)
    result = h.status("ORD-1001")
    assert "shipped" not in result and "shoes" not in result
    assert "email or phone" in result
    assert h.recent_orders.snapshot().focused_order_ref is None


def test_order_plus_matching_contact_answers_and_grants(config_root: Path) -> None:
    h = Harness(config_root)
    result = h.status(" ord-1001 ", _CUST1_PHONE)
    assert "shipped" in result  # answered (id normalized, case-insensitive)
    assert h.recent_orders.snapshot().focused_order_ref == "ORD-1001"
    # The grant is remembered: a repeat ask needs NO contact.
    assert "shipped" in h.status("ORD-1001")


def test_wrong_pair_and_unknown_order_are_indistinguishable(config_root: Path) -> None:
    # THE existence-oracle pin: a wrong contact on a REAL order and any contact on a
    # NONEXISTENT order produce byte-identical responses — probing cannot confirm an id.
    h = Harness(config_root)
    wrong_pair = h.status("ORD-1001", _CUST2_EMAIL)  # real order, not their contact
    unknown = h.status("ORD-9999", _CUST2_EMAIL)  # no such order
    assert wrong_pair == unknown
    assert "shipped" not in wrong_pair and "ORD-1001 -" not in wrong_pair
    assert h.recent_orders.snapshot().focused_order_ref is None


def test_contact_match_grants_only_that_order(config_root: Path) -> None:
    # A rung-1 grant is per-order: CUST-001's other order still declines without its own
    # pair (auto-extending would be server-side enumeration without the OTP), the session
    # stays UNBOUND, and list_orders still fails closed.
    h = Harness(config_root)
    assert "shipped" in h.status("ORD-1001", _CUST1_PHONE)
    sibling = h.status("ORD-1003")  # same owner, no pair given
    assert "delivered" not in sibling and "socks" not in sibling
    assert h.identity.current() is None
    listed = h.tools["list_orders"].invoke({})
    assert "ORD-" not in listed  # rung 2 untouched by a rung-1 grant


def test_bound_identity_reads_owned_orders_only(config_root: Path) -> None:
    # Authorization, not just authentication: a BOUND identity reads its OWN orders with no
    # contact, but someone else's order still declines (the authz/authn split).
    h = Harness(config_root)
    h.identity.bind(BoundIdentity(customer_ref="CUST-001", masked_contact="number ending 0119"))
    assert "shipped" in h.status("ORD-1001")
    assert "delivered" in h.status("ORD-1003")
    other = h.status("ORD-1002")  # CUST-002's order
    assert "rain jacket" not in other and "processing" not in other


def test_sequential_probe_discloses_nothing(config_root: Path) -> None:
    # ORD-1001/1002/1003 probed without a verifier -> three IDENTICAL no-data responses
    # (the low-entropy-id sequential-probing challenge that reversed decision 1).
    h = Harness(config_root)
    responses = {h.status(oid) for oid in ("ORD-1001", "ORD-1002", "ORD-1003")}
    assert len(responses) == 1
    assert "email or phone" in next(iter(responses))


def test_session_placed_order_is_readable_immediately(config_root: Path) -> None:
    # The caller placed it on THIS call — readable with no verification (rung 1 by
    # construction; the store is per-session, so no cross-caller path exists).
    h = Harness(config_root)
    placed = h.store.place_cart(
        "k1",
        lines=[CartLine(sku="SKU-GRN-15", name="merino hiking socks", price_usd=14.50, quantity=2)],
        total_usd=29.00,
    )
    assert "socks" in h.status(placed.order_id)


# --- list_orders (P7 rung 2) -----------------------------------------------------------


def test_list_orders_fails_closed_when_unverified(config_root: Path) -> None:
    h = Harness(config_root)
    result = h.tools["list_orders"].invoke({})
    assert "ORD-" not in result  # NO order data, not even a count
    assert "request_handover" in result and "list_orders" in result  # instructs the handover


def test_list_orders_scopes_to_the_bound_customer(config_root: Path) -> None:
    h = Harness(config_root)
    h.identity.bind(
        BoundIdentity(customer_ref="CUST-002", masked_contact="email ending example dot com")
    )
    result = h.tools["list_orders"].invoke({})
    assert "ORD-1002" in result
    assert "ORD-1001" not in result and "ORD-1003" not in result  # never someone else's


def test_summaries_never_carry_customer_linkage(config_root: Path) -> None:
    # Even an AUTHORIZED read speaks id + items + status only — no customer_ref, no contact
    # (sequential probing must never disclose cross-customer linkage).
    h = Harness(config_root)
    result = h.status("ORD-1001", _CUST1_PHONE)
    assert "CUST-" not in result and "0119" not in result and "@" not in result


# --- the untouched read tools ------------------------------------------------------------


def test_catalog_search_matches_and_misses(config_root: Path) -> None:
    h = Harness(config_root)
    assert "SKU-RED-42" in h.tools["catalog_search"].invoke({"query": "running"})
    miss = h.tools["catalog_search"].invoke({"query": "zzz-nothing"})
    assert "No catalog items" in miss
    # A miss steers the model to REAL items instead of invented categories.
    assert "rain jacket" in miss


def test_view_cart_reads_the_session_cart(config_root: Path) -> None:
    cart = CartStore()
    h = Harness(config_root, cart)
    # empty cart -> a real answer, not an escalation
    assert "empty" in h.tools["view_cart"].invoke({}).lower()
    # after a mutation, the SAME cart instance is read back (split-brain guard)
    cart.add_item(sku="SKU-BLU-07", name="rain jacket", price_usd=129.0, quantity=2)
    out = h.tools["view_cart"].invoke({})
    assert "2 rain jackets" in out and "$258.00" in out


def test_missing_fixture_fails_loudly_at_build(config_root: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_orders_fixture(config_root, "ghost_store")


def test_malformed_fixture_fails_loudly_at_build(tmp_path: Path) -> None:
    orders_dir = tmp_path / "fixtures" / "orders"
    orders_dir.mkdir(parents=True)
    (orders_dir / "m1.yaml").write_text(
        'orders:\n  ORD-1: { status: "shipped" }\nproducts: []\n', encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="failed validation"):
        load_orders_fixture(tmp_path, "m1")
