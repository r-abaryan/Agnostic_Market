"""Shared support-flow engine harness — ONE builder for the support/returns/profile suites
(extracted from test_support_flow rather than copied per file)."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

from llm_fakes import (
    TEST_CALLER_AUDIBLE_MODEL_TEXT_MAX_CHARS,
    TEST_STRUCTURED_OUTPUT_METHOD,
    FakeChatModel,
)
from routing_helpers import make_routing_session
from turn_helpers import TEST_CANCELLATION_QUIESCENCE_TIMEOUT_SECONDS
from verification_helpers import make_otp_provider

from agnostic_market.agents.engine import ReasoningEngine
from agnostic_market.agents.frontline import build_frontline_graph
from agnostic_market.agents.telemetry import InMemoryTelemetrySink, TenantTelemetry
from agnostic_market.checkpoints import build_checkpointer
from agnostic_market.commerce.cart import CartStore
from agnostic_market.commerce.catalog import FixtureCatalog
from agnostic_market.commerce.identity import (
    BoundIdentity,
    CallerIdentityStore,
    CustomerDirectory,
    load_customers_fixture,
)
from agnostic_market.commerce.orders import (
    GuestOrderScope,
    OrdersFixture,
    OrderStore,
    RecentOrderContext,
    load_orders_fixture,
)
from agnostic_market.commerce.payment_instruments import (
    PaymentInstrumentDirectory,
    PaymentInstrumentsFixture,
    load_payment_instruments_fixture,
)
from agnostic_market.commerce.profile import ProfileStore, load_profile_fixture
from agnostic_market.commerce.verification import (
    OtpProvider,
    RiskPort,
    RiskProvider,
    VerificationStore,
)
from agnostic_market.dtos.orchestration import RouteResolution
from agnostic_market.dtos.state import PolicyContext
from agnostic_market.durability.session_state import SessionStateCoordinator
from agnostic_market.session import CallerContext


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
    guest_orders: GuestOrderScope
    caller_context: CallerContext
    telemetry: InMemoryTelemetrySink


def authorize_customer(harness: SupportHarness, customer_ref: str) -> SupportHarness:
    """Bind one real fixture principal for tests below the identity-flow boundary."""
    masked_contact = harness.customers.masked_contact(customer_ref)
    if masked_contact is None:
        raise ValueError(f"test customer does not exist: {customer_ref}")
    harness.identity.bind(BoundIdentity(customer_ref=customer_ref, masked_contact=masked_contact))
    return harness


def build_support_engine(
    config_root: Path,
    *,
    policy: PolicyContext,
    reasoning: FakeChatModel | None = None,
    frontline: FakeChatModel | None = None,
    risk_flagged: bool = False,
    risk: RiskPort | None = None,
    thread_id: str = "support-1",
    payment_instruments_fixture: PaymentInstrumentsFixture | None = None,
    orders_fixture: OrdersFixture | None = None,
    routing_resolution: RouteResolution | None = None,
) -> SupportHarness:
    """The production graph shape behind a ReasoningEngine, with fakes + per-test stores.

    The recent-order and identity-store instances are SHARED between the tools and the graph
    (the split-brain rule); the neutral reasoning default clarifies — suites pass their
    own force_tool/scripted fakes.
    """
    if risk is not None and risk_flagged:
        raise ValueError("provide either a risk port or risk_flagged, not both")
    fixture = orders_fixture or load_orders_fixture(config_root, "acme_store")
    catalog = FixtureCatalog("acme_store", fixture)
    store = OrderStore("acme_store", fixture.orders)
    recent_orders = RecentOrderContext(max_refs=policy.cancel_batch_max)
    cart = CartStore()
    identity = CallerIdentityStore()
    guest_orders = GuestOrderScope(tenant_id="acme_store", session_id=thread_id)
    customers = CustomerDirectory("acme_store", load_customers_fixture(config_root, "acme_store"))
    instrument_fixture = (
        payment_instruments_fixture
        if payment_instruments_fixture is not None
        else load_payment_instruments_fixture(config_root, "acme_store")
    )
    payment_instruments = PaymentInstrumentDirectory("acme_store", instrument_fixture)
    otp = make_otp_provider()
    verification = VerificationStore(otp, session_id=thread_id)
    profile = ProfileStore("acme_store", load_profile_fixture(config_root, "acme_store"))
    telemetry_sink = InMemoryTelemetrySink()
    telemetry = TenantTelemetry("acme_store", telemetry_sink, telemetry_sink).bind_session(
        thread_id
    )
    caller_context = CallerContext(
        verification_store=verification,
        session_state=SessionStateCoordinator(cart, recent_orders, guest_orders),
        identity_store=identity,
        telemetry=telemetry.operational,
    )
    assembly = build_frontline_graph(
        frontline or FakeChatModel(emit_tool_calls=False),
        display_name="Acme Store",
        tenant_id="acme_store",
        reasoning_model=reasoning or FakeChatModel(emit_tool_calls=False),
        store=store,
        catalog=catalog,
        guest_orders=guest_orders,
        policy=policy,
        verification_store=verification,
        risk=risk or RiskProvider("acme_store", flagged=risk_flagged),
        profile_store=profile,
        cart_store=cart,
        recent_orders=recent_orders,
        identity_store=identity,
        customers=customers,
        payment_instruments=payment_instruments,
        lifecycle=caller_context,
        structured_output_method=TEST_STRUCTURED_OUTPUT_METHOD,
        caller_audible_model_text_max_chars=TEST_CALLER_AUDIBLE_MODEL_TEXT_MAX_CHARS,
        response_model_node_timeout_seconds=2.0,
        reasoning_model_node_timeout_seconds=6.0,
        session_telemetry=telemetry,
        checkpointer=build_checkpointer(),
    )
    engine = ReasoningEngine(
        assembly.graph,
        tenant_id="acme_store",
        deployment_id="test-deployment",
        thread_id=thread_id,
        checkpoint_io_timeout_seconds=2.0,
        cancellation_quiescence_timeout_seconds=(TEST_CANCELLATION_QUIESCENCE_TIMEOUT_SECONDS),
        routing=make_routing_session(
            assembly.capability_registry,
            identity_store=identity,
            cart_store=cart,
            recent_orders=recent_orders,
            resolution=routing_resolution,
            continue_active=routing_resolution is not None,
            telemetry=telemetry.routing_evidence,
        ),
        telemetry=telemetry.operational,
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
        guest_orders,
        caller_context,
        telemetry_sink,
    )
