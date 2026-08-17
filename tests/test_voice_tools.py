"""Orders-fixture preload + the read-only tools it feeds (zero network).

P7 object binding: `order_status` is gated by the guest-lookup pair (order id + account
contact, code-matched) — the gate lives in the tool body, fail-closed, with ONE combined
not-found response so an order id's existence is never confirmed. `list_orders` (rung 2)
fails closed until the identity flow binds the session.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.tools import BaseTool

from agnostic_market.agents._copy import ACCOUNT_CONTACT_QUESTION
from agnostic_market.commerce.cart import CartStore
from agnostic_market.commerce.identity import (
    BoundIdentity,
    CallerIdentityStore,
    CustomerDirectory,
    load_customers_fixture,
)
from agnostic_market.commerce.orders import (
    BOUND_ORDER_READ_UNAVAILABLE_LINE,
    ORDER_CONTACT_NOT_FOUND_LINE,
    OrderStore,
    RecentOrderContext,
    load_orders_fixture,
)
from agnostic_market.config.loader import ConfigError
from agnostic_market.dtos.state import CartLine
from agnostic_market.voice.tools import (
    _ASK_CONTACT,
    _BOUND_ORDER_READ_DENIED,
    _COMBINED_NOT_FOUND,
    build_voice_tools,
)

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


def _telemetry(tmp_path: Path) -> list[dict]:
    path = tmp_path / "telemetry.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_fixture_loads_and_validates(config_root: Path) -> None:
    fixture = load_orders_fixture(config_root, "acme_store")
    assert "ORD-1001" in fixture.orders
    assert fixture.orders["ORD-1001"].customer_ref == "CUST-001"
    assert fixture.products


# --- the object-binding gate (P7 rung 1) ---------------------------------------------


def test_order_status_fails_closed_without_a_verifier(config_root: Path, tmp_path: Path) -> None:
    # No contact, unbound session -> NO order data, NO existence confirmation; the result
    # instructs the ask-for-contact exchange. The pointer is untouched (a probe must not
    # hijack "that order").
    h = Harness(config_root)
    result = h.status("ORD-1001")
    assert "shipped" not in result and "shoes" not in result
    assert result == _ASK_CONTACT
    assert h.recent_orders.snapshot().focused_order_ref is None
    assert _telemetry(tmp_path) == [{"event": "order_read_denied", "order_id_known": False}]


def test_order_plus_matching_contact_answers_and_grants(config_root: Path, tmp_path: Path) -> None:
    h = Harness(config_root)
    result = h.status(" ord-1001 ", _CUST1_PHONE)
    assert "shipped" in result  # answered (id normalized, case-insensitive)
    assert h.recent_orders.snapshot().focused_order_ref == "ORD-1001"
    # The grant is remembered: a repeat ask needs NO contact.
    assert "shipped" in h.status("ORD-1001")
    assert _telemetry(tmp_path) == [
        {
            "event": "order_read_granted",
            "order_id": "ORD-1001",
            "method": "contact_match",
        }
    ]


def test_wrong_pair_and_unknown_order_are_indistinguishable(
    config_root: Path, tmp_path: Path
) -> None:
    # THE existence-oracle pin: a wrong contact on a REAL order and any contact on a
    # NONEXISTENT order produce byte-identical responses — probing cannot confirm an id.
    h = Harness(config_root)
    wrong_pair = h.status("ORD-1001", _CUST2_EMAIL)  # real order, not their contact
    unknown = h.status("ORD-9999", _CUST2_EMAIL)  # no such order
    assert wrong_pair == unknown == _COMBINED_NOT_FOUND
    assert "shipped" not in wrong_pair and "ORD-1001 -" not in wrong_pair
    assert h.recent_orders.snapshot().focused_order_ref is None
    assert _telemetry(tmp_path) == [
        {"event": "order_read_denied", "order_id_known": True},
        {"event": "order_read_denied", "order_id_known": False},
    ]


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
    assert other == _BOUND_ORDER_READ_DENIED


@pytest.mark.parametrize("order_id", ["ORD-1002", "ORD-9999", " "])
@pytest.mark.parametrize("contact", ["", _CUST2_EMAIL])
def test_bound_unreadable_order_uses_one_access_neutral_result(
    config_root: Path,
    tmp_path: Path,
    order_id: str,
    contact: str,
) -> None:
    h = Harness(config_root)
    h.identity.bind(BoundIdentity(customer_ref="CUST-001", masked_contact="number ending 0119"))

    assert h.status(order_id, contact) == _BOUND_ORDER_READ_DENIED
    assert h.recent_orders.snapshot().focused_order_ref is None
    assert _telemetry(tmp_path) == [
        {
            "event": "order_read_denied",
            "order_id_known": order_id.strip().upper() == "ORD-1002",
        }
    ]


def test_bound_principal_ignores_residual_grant_and_matching_foreign_contact(
    config_root: Path,
) -> None:
    h = Harness(config_root)
    h.identity.grant_orders("ORD-1002")
    h.identity.bind(BoundIdentity(customer_ref="CUST-001", masked_contact="number ending 0119"))

    assert h.status("ORD-1002", _CUST2_EMAIL) == _BOUND_ORDER_READ_DENIED
    assert not h.recent_orders.snapshot().order_refs


def test_order_read_copy_contracts_are_embedded_once() -> None:
    assert ACCOUNT_CONTACT_QUESTION == "What email address or phone number is on the account?"
    assert ORDER_CONTACT_NOT_FOUND_LINE == (
        "I couldn't find an order matching those details - could you double-check the order "
        "number and the email or phone on the account?"
    )
    assert BOUND_ORDER_READ_UNAVAILABLE_LINE == (
        "I couldn't retrieve an order with that number on this call. Please double-check the "
        "order number, or ask to switch accounts if you meant a different account."
    )
    assert (
        "Not verified for that order. Do not say whether the order exists. Ask the caller ONE "
        f'short question exactly: "{ACCOUNT_CONTACT_QUESTION}" Then call order_status again with '
        "the order id AND account_contact."
    ) == _ASK_CONTACT
    assert (
        "No order matches those details. Do not say which detail failed. Tell the caller exactly: "
        f'"{ORDER_CONTACT_NOT_FOUND_LINE}"'
    ) == _COMBINED_NOT_FOUND
    assert (
        "The caller is already verified on this call, but this order read is not authorized. "
        "Do not say whether the order exists or which account owns it. Tell the caller exactly: "
        f'"{BOUND_ORDER_READ_UNAVAILABLE_LINE}"'
    ) == _BOUND_ORDER_READ_DENIED


def test_sequential_probe_discloses_nothing(config_root: Path) -> None:
    # ORD-1001/1002/1003 probed without a verifier -> three IDENTICAL no-data responses
    # (the low-entropy-id sequential-probing challenge that reversed decision 1).
    h = Harness(config_root)
    responses = {h.status(oid) for oid in ("ORD-1001", "ORD-1002", "ORD-1003")}
    assert responses == {_ASK_CONTACT}


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
    assert h.tools["catalog_search"].invoke({"query": "running"}) == (
        "Matching items: trail running shoes (sku SKU-RED-42, $89.99)"
    )
    miss = h.tools["catalog_search"].invoke({"query": "zzz-nothing"})
    assert miss == (
        "No catalog items match 'zzz-nothing'. The catalog carries: "
        "trail running shoes; waterproof rain jacket; merino hiking socks."
    )
    # A miss steers the model to REAL items instead of invented categories.
    assert "rain jacket" in miss
    assert h.tools["catalog_search"].invoke({"query": "   "}) == (
        "No catalog items match '   '. The catalog carries: "
        "trail running shoes; waterproof rain jacket; merino hiking socks."
    )


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
