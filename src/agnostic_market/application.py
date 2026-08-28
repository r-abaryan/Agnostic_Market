"""Shared application-session composition for voice and evaluation."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver

from agnostic_market.agents.capabilities import CapabilityRegistry
from agnostic_market.agents.engine import ReasoningEngine
from agnostic_market.agents.frontline import FrontlineGraphAssembly, build_frontline_graph
from agnostic_market.agents.routing import RoutingRecognizer, RoutingSession
from agnostic_market.checkpoints import SchemaValidatedCheckpointSaver, build_checkpointer
from agnostic_market.commerce.cart import CartStore
from agnostic_market.commerce.catalog import CatalogPort, FixtureCatalog
from agnostic_market.commerce.identity import (
    CallerIdentityStore,
    CustomerDirectory,
    assert_orders_have_customers,
    load_customers_fixture,
)
from agnostic_market.commerce.orders import (
    GuestOrderScope,
    OrderStore,
    RecentOrderContext,
    load_orders_fixture,
)
from agnostic_market.commerce.payment_instruments import (
    PaymentInstrumentDirectory,
    assert_payment_instruments_have_customers,
    load_payment_instruments_fixture,
)
from agnostic_market.commerce.profile import (
    ProfileStore,
    assert_profiles_have_customers,
    load_profile_fixture,
)
from agnostic_market.commerce.verification import (
    OtpProvider,
    RiskProvider,
    VerificationStore,
    load_verification_fixture,
)
from agnostic_market.dtos.config import MerchantConfig
from agnostic_market.dtos.llm import StructuredOutputMethod
from agnostic_market.session import CallerContext
from agnostic_market.tenancy.context import TenantContext


@dataclass(frozen=True, slots=True)
class ApplicationModels:
    response: BaseChatModel
    reasoning: BaseChatModel
    response_structured_output_method: StructuredOutputMethod


@dataclass(frozen=True, slots=True)
class ApplicationSettings:
    display_name: str
    caller_audible_model_text_max_chars: int
    checkpoint_io_timeout_seconds: float
    response_model_node_timeout_seconds: float
    reasoning_model_node_timeout_seconds: float
    cancellation_quiescence_timeout_seconds: float

    @classmethod
    def from_merchant_config(cls, config: MerchantConfig) -> ApplicationSettings:
        return cls(
            display_name=config.display_name,
            caller_audible_model_text_max_chars=(
                config.runtime.caller_audible_model_text_max_chars
            ),
            checkpoint_io_timeout_seconds=config.runtime.checkpoint_io_timeout_seconds,
            response_model_node_timeout_seconds=(
                config.runtime.response_model_node_timeout_seconds
            ),
            reasoning_model_node_timeout_seconds=(
                config.runtime.reasoning_model_node_timeout_seconds
            ),
            cancellation_quiescence_timeout_seconds=(
                config.runtime.cancellation_quiescence_timeout_seconds
            ),
        )


@dataclass(frozen=True, slots=True)
class TenantServices:
    tenant_id: str
    catalog: CatalogPort
    order_store: OrderStore
    customers: CustomerDirectory
    payment_instruments: PaymentInstrumentDirectory
    profile_store: ProfileStore
    otp: OtpProvider
    risk: RiskProvider
    checkpointer: SchemaValidatedCheckpointSaver


@dataclass(frozen=True, slots=True)
class ApplicationSessionState:
    session_id: str
    thread_id: str
    cart_store: CartStore
    verification_store: VerificationStore
    recent_orders: RecentOrderContext
    identity_store: CallerIdentityStore
    guest_orders: GuestOrderScope
    caller_context: CallerContext


type SessionStateFactory = Callable[[TenantContext, TenantServices], ApplicationSessionState]
type RoutingFactory = Callable[[CapabilityRegistry], RoutingRecognizer]


@dataclass(frozen=True, slots=True)
class ApplicationSession:
    assembly: FrontlineGraphAssembly
    engine: ReasoningEngine
    tenant: TenantContext
    services: TenantServices
    state: ApplicationSessionState


def _validate_session_state(
    tenant: TenantContext,
    state: ApplicationSessionState,
) -> None:
    if not state.session_id.strip() or not state.thread_id.strip():
        raise ValueError("application session requires non-empty session and thread ids")
    if state.guest_orders.tenant_id != tenant.tenant_id:
        raise ValueError("session guest-order scope does not match the application tenant")
    if state.guest_orders.session_id != state.session_id:
        raise ValueError("guest-order scope does not match the application session")

    lifecycle_stores = (
        ("cart", state.caller_context.cart_store, state.cart_store),
        ("verification", state.caller_context.verification_store, state.verification_store),
        ("recent-order", state.caller_context.recent_orders, state.recent_orders),
        ("identity", state.caller_context.identity_store, state.identity_store),
        ("guest-order", state.caller_context.guest_orders, state.guest_orders),
    )
    mismatched = [
        name for name, lifecycle_store, store in lifecycle_stores if lifecycle_store is not store
    ]
    if mismatched:
        raise ValueError(
            "caller lifecycle does not own the application session stores: " + ", ".join(mismatched)
        )


def build_fixture_tenant_services(
    config_root: Path,
    tenant_id: str,
    *,
    checkpointer: BaseCheckpointSaver | None = None,
) -> TenantServices:
    """Load validated development adapters behind the production composition boundary."""
    orders_fixture = load_orders_fixture(config_root, tenant_id)
    customers_fixture = load_customers_fixture(config_root, tenant_id)
    profile_fixture = load_profile_fixture(config_root, tenant_id)
    payment_fixture = load_payment_instruments_fixture(config_root, tenant_id)
    verification_fixture = load_verification_fixture(config_root, tenant_id)
    assert_orders_have_customers(orders_fixture, customers_fixture)
    assert_profiles_have_customers(profile_fixture, customers_fixture)
    assert_payment_instruments_have_customers(payment_fixture, customers_fixture)
    checkpoint_boundary = (
        checkpointer
        if isinstance(checkpointer, SchemaValidatedCheckpointSaver)
        else build_checkpointer(checkpointer)
    )
    return TenantServices(
        tenant_id=tenant_id,
        catalog=FixtureCatalog(tenant_id, orders_fixture),
        order_store=OrderStore(tenant_id, orders_fixture.orders),
        customers=CustomerDirectory(customers_fixture),
        payment_instruments=PaymentInstrumentDirectory(payment_fixture),
        profile_store=ProfileStore(profile_fixture),
        otp=OtpProvider(valid_code=verification_fixture.otp_code),
        risk=RiskProvider(),
        checkpointer=checkpoint_boundary,
    )


def build_in_memory_session_state(
    tenant: TenantContext,
    services: TenantServices,
    *,
    session_id: str | None = None,
    thread_id: str | None = None,
) -> ApplicationSessionState:
    """Build isolated caller state while retaining injected tenant services."""
    resolved_session_id = session_id or uuid.uuid4().hex
    resolved_thread_id = thread_id or uuid.uuid4().hex
    cart_store = CartStore()
    verification_store = VerificationStore(services.otp)
    recent_orders = RecentOrderContext(max_refs=tenant.policy.cancel_batch_max)
    identity_store = CallerIdentityStore()
    guest_orders = GuestOrderScope(
        tenant_id=tenant.tenant_id,
        session_id=resolved_session_id,
    )
    caller_context = CallerContext(
        verification_store=verification_store,
        cart_store=cart_store,
        recent_orders=recent_orders,
        identity_store=identity_store,
        guest_orders=guest_orders,
    )
    return ApplicationSessionState(
        session_id=resolved_session_id,
        thread_id=resolved_thread_id,
        cart_store=cart_store,
        verification_store=verification_store,
        recent_orders=recent_orders,
        identity_store=identity_store,
        guest_orders=guest_orders,
        caller_context=caller_context,
    )


def build_application_session(
    tenant: TenantContext,
    settings: ApplicationSettings,
    models: ApplicationModels,
    services: TenantServices,
    *,
    deployment_id: str,
    routing_factory: RoutingFactory,
    session_state_factory: SessionStateFactory = build_in_memory_session_state,
) -> ApplicationSession:
    """Construct the one graph, router, engine, and caller lifecycle."""
    if services.tenant_id != tenant.tenant_id:
        raise ValueError("tenant services do not match the application tenant")
    if services.catalog.tenant_id != tenant.tenant_id:
        raise ValueError("catalog service does not match the application tenant")
    if services.order_store.tenant_id != tenant.tenant_id:
        raise ValueError("order service does not match the application tenant")
    state = session_state_factory(tenant, services)
    _validate_session_state(tenant, state)
    assembly = build_frontline_graph(
        models.response,
        display_name=settings.display_name,
        tenant_id=tenant.tenant_id,
        reasoning_model=models.reasoning,
        store=services.order_store,
        catalog=services.catalog,
        guest_orders=state.guest_orders,
        cart_store=state.cart_store,
        policy=tenant.policy,
        verification_store=state.verification_store,
        otp=services.otp,
        risk=services.risk,
        profile_store=services.profile_store,
        recent_orders=state.recent_orders,
        identity_store=state.identity_store,
        customers=services.customers,
        payment_instruments=services.payment_instruments,
        lifecycle=state.caller_context,
        structured_output_method=models.response_structured_output_method,
        caller_audible_model_text_max_chars=settings.caller_audible_model_text_max_chars,
        response_model_node_timeout_seconds=settings.response_model_node_timeout_seconds,
        reasoning_model_node_timeout_seconds=settings.reasoning_model_node_timeout_seconds,
        checkpointer=services.checkpointer,
    )
    routing = RoutingSession(
        routing_factory(assembly.capability_registry),
        identity_store=state.identity_store,
        cart_store=state.cart_store,
        recent_orders=state.recent_orders,
        registry=assembly.capability_registry,
    )
    engine = ReasoningEngine(
        assembly.graph,
        tenant_id=tenant.tenant_id,
        deployment_id=deployment_id,
        thread_id=state.thread_id,
        checkpoint_io_timeout_seconds=settings.checkpoint_io_timeout_seconds,
        cancellation_quiescence_timeout_seconds=(settings.cancellation_quiescence_timeout_seconds),
        routing=routing,
        lifecycle=state.caller_context,
    )
    state.caller_context.attach_engine(engine)
    return ApplicationSession(
        assembly=assembly,
        engine=engine,
        tenant=tenant,
        services=services,
        state=state,
    )
