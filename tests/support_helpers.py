"""Shared support-flow engine harness — ONE builder for the support/returns/profile suites
(extracted from test_support_flow rather than copied per file)."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

from llm_fakes import FakeChatModel
from turn_helpers import TEST_CANCELLATION_QUIESCENCE_TIMEOUT_SECONDS

from agnostic_market.agents.engine import ReasoningEngine, build_checkpointer
from agnostic_market.agents.frontline import build_frontline_graph
from agnostic_market.agents.tooling import wrap_readonly_tool
from agnostic_market.commerce.cart import CartStore
from agnostic_market.commerce.identity import (
    CallerIdentityStore,
    CustomerDirectory,
    load_customers_fixture,
)
from agnostic_market.commerce.orders import OrderStore, RecentOrderContext, load_orders_fixture
from agnostic_market.commerce.payment_instruments import (
    PaymentInstrumentDirectory,
    PaymentInstrumentsFixture,
    load_payment_instruments_fixture,
)
from agnostic_market.commerce.profile import ProfileStore, load_profile_fixture
from agnostic_market.commerce.verification import OtpProvider, RiskProvider, VerificationStore
from agnostic_market.dtos.state import PolicyContext
from agnostic_market.voice.context import CallerContext


class SupportHarness(NamedTuple):
    """Everything a support-family test asserts against, by name."""

    engine: ReasoningEngine
    store: OrderStore
    verification: VerificationStore
    otp: OtpProvider
    profile: ProfileStore
    recent_orders: RecentOrderContext
    identity: CallerIdentityStore
    customers: CustomerDirectory
    payment_instruments: PaymentInstrumentDirectory
    caller_context: CallerContext


_FIXTURE_ORDERS = ("ORD-1001", "ORD-1002", "ORD-1003")
TEST_OTP = "482913"


def authorize_fixture_orders(harness: SupportHarness) -> SupportHarness:
    """Pre-authorize the fixture orders as if the caller had fully verified, so suites pinning
    post-authorization MONEY logic aren't re-testing the auth gate (the gate's OWN tests build
    unauthorized harnesses and never call this). Grants BOTH rungs:
      - rung-1 read grant (`grant_order`) so each order appears in the model's scoped candidate
        list (`order_read_allowed`);
      - the TEST-ONLY rung-2 mutation grant (`grant_mutation_for_test`) so cancel/refund/return
        proceed (Fix 2: rung-1 alone no longer authorizes a mutation; a real bind can't span the
        two customers the fixture orders belong to — this is the sanctioned test seam)."""
    for order_id in _FIXTURE_ORDERS:
        harness.identity.grant_order(order_id)
        harness.identity.grant_mutation_for_test(order_id)
    return harness


def build_support_engine(
    config_root: Path,
    *,
    policy: PolicyContext,
    reasoning: FakeChatModel | None = None,
    frontline: FakeChatModel | None = None,
    risk_flagged: bool = False,
    thread_id: str = "support-1",
    payment_instruments_fixture: PaymentInstrumentsFixture | None = None,
) -> SupportHarness:
    """The production graph shape behind a ReasoningEngine, with fakes + per-test stores.

    The recent-order and identity-store instances are SHARED between the tools and the graph
    (the split-brain rule); the neutral reasoning default clarifies — suites pass their
    own force_tool/scripted fakes.
    """
    from agnostic_market.voice.tools import build_voice_tools

    store = OrderStore(load_orders_fixture(config_root, "acme_store"))
    recent_orders = RecentOrderContext(max_refs=policy.cancel_batch_max)
    cart = CartStore()
    identity = CallerIdentityStore()
    customers = CustomerDirectory(load_customers_fixture(config_root, "acme_store"))
    instrument_fixture = (
        payment_instruments_fixture
        if payment_instruments_fixture is not None
        else load_payment_instruments_fixture(config_root, "acme_store")
    )
    payment_instruments = PaymentInstrumentDirectory(instrument_fixture)
    tools = [
        wrap_readonly_tool(t, "acme_store")
        for t in build_voice_tools(store, cart, recent_orders, identity, customers)
    ]
    otp = OtpProvider(valid_code=TEST_OTP)
    verification = VerificationStore(otp)
    profile = ProfileStore(load_profile_fixture(config_root, "acme_store"))
    caller_context = CallerContext(
        verification_store=verification,
        cart_store=cart,
        recent_orders=recent_orders,
        identity_store=identity,
        order_store=store,
    )
    graph = build_frontline_graph(
        frontline or FakeChatModel(emit_tool_calls=False),
        tools,
        display_name="Acme Store",
        tenant_id="acme_store",
        reasoning_model=reasoning or FakeChatModel(emit_tool_calls=False),
        store=store,
        policy=policy,
        verification_store=verification,
        otp=otp,
        risk=RiskProvider(flagged=risk_flagged),
        profile_store=profile,
        cart_store=cart,
        recent_orders=recent_orders,
        identity_store=identity,
        customers=customers,
        payment_instruments=payment_instruments,
        lifecycle=caller_context,
        checkpointer=build_checkpointer(),
    )
    engine = ReasoningEngine(
        graph,
        thread_id=thread_id,
        cancellation_quiescence_timeout_seconds=(TEST_CANCELLATION_QUIESCENCE_TIMEOUT_SECONDS),
        lifecycle=caller_context,
    )
    caller_context.attach_engine(engine)
    return SupportHarness(
        engine,
        store,
        verification,
        otp,
        profile,
        recent_orders,
        identity,
        customers,
        payment_instruments,
        caller_context,
    )
