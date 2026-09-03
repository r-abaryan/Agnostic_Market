"""Shared application-session composition for voice and evaluation."""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, fields
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, get_args, get_type_hints

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
from agnostic_market.tenancy.context import TenantBound, TenantContext


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


class ApplicationResponsibility(StrEnum):
    DURABLE_TENANT_BUSINESS_STATE = "durable_tenant_business_state"
    DURABLE_PLATFORM_SESSION_STATE = "durable_platform_session_state"
    PROCESS_LOCAL_RUNTIME_COORDINATION = "process_local_runtime_coordination"


@dataclass(frozen=True, slots=True)
class TenantServices:
    tenant_id: Annotated[str, ApplicationResponsibility.DURABLE_PLATFORM_SESSION_STATE]
    catalog: Annotated[CatalogPort, ApplicationResponsibility.DURABLE_TENANT_BUSINESS_STATE]
    order_store: Annotated[OrderPort, ApplicationResponsibility.DURABLE_TENANT_BUSINESS_STATE]
    customers: Annotated[
        CustomerDirectoryPort, ApplicationResponsibility.DURABLE_TENANT_BUSINESS_STATE
    ]
    payment_instruments: Annotated[
        PaymentInstrumentPort, ApplicationResponsibility.DURABLE_TENANT_BUSINESS_STATE
    ]
    profile_store: Annotated[ProfilePort, ApplicationResponsibility.DURABLE_TENANT_BUSINESS_STATE]
    otp: Annotated[OtpPort, ApplicationResponsibility.DURABLE_PLATFORM_SESSION_STATE]
    risk: Annotated[RiskPort, ApplicationResponsibility.DURABLE_PLATFORM_SESSION_STATE]
    checkpointer: Annotated[
        SchemaValidatedCheckpointSaver,
        ApplicationResponsibility.DURABLE_PLATFORM_SESSION_STATE,
    ]
    telemetry: Annotated[TenantTelemetry, ApplicationResponsibility.DURABLE_PLATFORM_SESSION_STATE]


@dataclass(frozen=True, slots=True)
class ApplicationSessionState:
    session_id: Annotated[str, ApplicationResponsibility.DURABLE_PLATFORM_SESSION_STATE]
    thread_id: Annotated[str, ApplicationResponsibility.DURABLE_PLATFORM_SESSION_STATE]
    cart_store: Annotated[CartStore, ApplicationResponsibility.DURABLE_PLATFORM_SESSION_STATE]
    verification_store: Annotated[
        VerificationStore, ApplicationResponsibility.DURABLE_PLATFORM_SESSION_STATE
    ]
    recent_orders: Annotated[
        RecentOrderContext, ApplicationResponsibility.DURABLE_PLATFORM_SESSION_STATE
    ]
    identity_store: Annotated[
        CallerIdentityStore, ApplicationResponsibility.DURABLE_PLATFORM_SESSION_STATE
    ]
    guest_orders: Annotated[
        GuestOrderScope, ApplicationResponsibility.DURABLE_PLATFORM_SESSION_STATE
    ]
    caller_context: Annotated[
        CallerContext, ApplicationResponsibility.PROCESS_LOCAL_RUNTIME_COORDINATION
    ]
    telemetry: Annotated[
        SessionTelemetry, ApplicationResponsibility.PROCESS_LOCAL_RUNTIME_COORDINATION
    ]


def _derive_application_responsibilities() -> dict[tuple[str, str], ApplicationResponsibility]:
    responsibilities: dict[tuple[str, str], ApplicationResponsibility] = {}
    for owner in (TenantServices, ApplicationSessionState):
        annotations = get_type_hints(owner, include_extras=True)
        for declared_field in fields(owner):
            declared_responsibilities = tuple(
                value
                for value in get_args(annotations[declared_field.name])[1:]
                if isinstance(value, ApplicationResponsibility)
            )
            if len(declared_responsibilities) != 1:
                raise TypeError(
                    f"{owner.__name__}.{declared_field.name} requires one application "
                    "responsibility"
                )
            responsibilities[(owner.__name__, declared_field.name)] = declared_responsibilities[0]
    return responsibilities


APPLICATION_RESPONSIBILITIES = MappingProxyType(_derive_application_responsibilities())


type SessionStateFactory = Callable[
    [TenantContext, TenantServices], Awaitable[ApplicationSessionState]
]
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
    if state.verification_store.session_id != state.session_id:
        raise ValueError("session verification does not match the application session")
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
    tenant: TenantContext,
    *,
    telemetry: TenantTelemetry,
    checkpointer: BaseCheckpointSaver | None = None,
) -> TenantServices:
    """Load validated development adapters behind the production composition boundary."""
    tenant_id = tenant.tenant_id
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
    expected_factor_refs = {entry.factor_ref for entry in customers_fixture.customers.values()}
    factor_refs = set(verification_fixture.otp_codes_by_factor_ref)
    if factor_refs != expected_factor_refs:
        missing = sorted(expected_factor_refs - factor_refs)
        unknown = sorted(factor_refs - expected_factor_refs)
        details = []
        if missing:
            details.append("missing factors: " + ", ".join(missing))
        if unknown:
            details.append("unknown factors: " + ", ".join(unknown))
        raise ValueError("verification fixture does not match customers: " + "; ".join(details))
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
        otp=OtpProvider(
            tenant_id,
            codes_by_factor_ref=verification_fixture.otp_codes_by_factor_ref,
            challenge_ttl_seconds=verification_fixture.challenge_ttl_seconds,
            proof_ttl_seconds=verification_fixture.proof_ttl_seconds,
        ),
        risk=RiskProvider(tenant_id),
        checkpointer=checkpoint_boundary,
        telemetry=telemetry,
    )


async def build_in_memory_session_state(
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
    verification_store = VerificationStore(services.otp, session_id=resolved_session_id)
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


async def build_application_session(
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
    state = await session_state_factory(tenant, services)
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
