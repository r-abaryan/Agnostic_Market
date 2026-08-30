"""Shared application-session composition for voice and evaluation."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, fields
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver

from agnostic_market.agents.capabilities import CapabilityRegistry
from agnostic_market.agents.engine import ReasoningEngine
from agnostic_market.agents.frontline import FrontlineGraphAssembly, build_frontline_graph
from agnostic_market.agents.routing import RoutingRecognizer, RoutingSession
from agnostic_market.agents.telemetry import (
    SessionTelemetry,
    TenantTelemetry,
)
from agnostic_market.checkpoints import SchemaValidatedCheckpointSaver, build_checkpointer
from agnostic_market.commerce.cart import CartStore
from agnostic_market.commerce.catalog import CatalogPort, FixtureCatalog
from agnostic_market.commerce.identity import (
    CallerIdentityStore,
    CustomerDirectory,
    CustomerDirectoryPort,
    assert_orders_have_customers,
    load_customers_fixture,
)
from agnostic_market.commerce.orders import (
    GuestOrderScope,
    OrderPort,
    OrderStore,
    RecentOrderContext,
    load_orders_fixture,
)
from agnostic_market.commerce.payment_instruments import (
    PaymentInstrumentDirectory,
    PaymentInstrumentPort,
    assert_payment_instruments_have_customers,
    load_payment_instruments_fixture,
)
from agnostic_market.commerce.profile import (
    ProfilePort,
    ProfileStore,
    assert_profiles_have_customers,
    load_profile_fixture,
)
from agnostic_market.commerce.verification import (
    OtpPort,
    OtpProvider,
    RiskPort,
    RiskProvider,
    VerificationStore,
    load_verification_fixture,
)
from agnostic_market.dtos.config import MerchantConfig
from agnostic_market.dtos.llm import StructuredOutputMethod
from agnostic_market.session import CallerContext
from agnostic_market.tenancy.context import TenantBound, TenantContext, normalize_tenant_id


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
    order_store: OrderPort
    customers: CustomerDirectoryPort
    payment_instruments: PaymentInstrumentPort
    profile_store: ProfilePort
    otp: OtpPort
    risk: RiskPort
    checkpointer: SchemaValidatedCheckpointSaver
    telemetry: TenantTelemetry


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
    telemetry: SessionTelemetry


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
    services: TenantServices,
    state: ApplicationSessionState,
) -> None:
    if not state.session_id.strip() or not state.thread_id.strip():
        raise ValueError("application session requires non-empty session and thread ids")
    if state.guest_orders.tenant_id != tenant.tenant_id:
        raise ValueError("session guest-order scope does not match the application tenant")
    if state.guest_orders.session_id != state.session_id:
        raise ValueError("guest-order scope does not match the application session")
    if (
        state.telemetry.tenant_id != tenant.tenant_id
        or state.telemetry.session_id != state.session_id
    ):
        raise ValueError("telemetry scope does not match the application session")
    if (
        state.telemetry.operational.sink is not services.telemetry.operational_sink
        or state.telemetry.routing_evidence.sink is not services.telemetry.routing_evidence_sink
    ):
        raise ValueError("session telemetry does not use the tenant telemetry service")
    if state.caller_context.telemetry is not state.telemetry.operational:
        raise ValueError("caller lifecycle does not own the application session telemetry")
    if not state.verification_store.uses_otp_provider(services.otp):
        raise ValueError("session verification does not use the tenant OTP service")

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
    telemetry: TenantTelemetry,
    checkpointer: BaseCheckpointSaver | None = None,
) -> TenantServices:
    """Load validated development adapters behind the production composition boundary."""
    tenant_id = normalize_tenant_id(tenant_id, boundary="fixture tenant services")
    if telemetry.tenant_id != tenant_id:
        raise ValueError("telemetry service does not match the fixture tenant")
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
        customers=CustomerDirectory(tenant_id, customers_fixture),
        payment_instruments=PaymentInstrumentDirectory(tenant_id, payment_fixture),
        profile_store=ProfileStore(tenant_id, profile_fixture),
        otp=OtpProvider(tenant_id, valid_code=verification_fixture.otp_code),
        risk=RiskProvider(tenant_id),
        checkpointer=checkpoint_boundary,
        telemetry=telemetry,
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
    telemetry = services.telemetry.bind_session(resolved_session_id)
    caller_context = CallerContext(
        verification_store=verification_store,
        cart_store=cart_store,
        recent_orders=recent_orders,
        identity_store=identity_store,
        guest_orders=guest_orders,
        telemetry=telemetry.operational,
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
        telemetry=telemetry,
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
    mismatched_services: list[str] = []
    for service_field in fields(services):
        if service_field.name in {"tenant_id", "checkpointer"}:
            continue
        service = getattr(services, service_field.name)
        if not isinstance(service, TenantBound):
            raise TypeError(f"{service_field.name} does not expose a tenant identity")
        if service.tenant_id != tenant.tenant_id:
            mismatched_services.append(service_field.name)
    if mismatched_services:
        raise ValueError(
            "tenant services do not match the application tenant: " + ", ".join(mismatched_services)
        )
    state = session_state_factory(tenant, services)
    _validate_session_state(tenant, services, state)
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
        session_telemetry=state.telemetry,
        checkpointer=services.checkpointer,
    )
    routing = RoutingSession(
        routing_factory(assembly.capability_registry),
        identity_store=state.identity_store,
        cart_store=state.cart_store,
        recent_orders=state.recent_orders,
        registry=assembly.capability_registry,
        telemetry=state.telemetry.routing_evidence,
    )
    engine = ReasoningEngine(
        assembly.graph,
        tenant_id=tenant.tenant_id,
        deployment_id=deployment_id,
        thread_id=state.thread_id,
        checkpoint_io_timeout_seconds=settings.checkpoint_io_timeout_seconds,
        cancellation_quiescence_timeout_seconds=(settings.cancellation_quiescence_timeout_seconds),
        routing=routing,
        telemetry=state.telemetry.operational,
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
