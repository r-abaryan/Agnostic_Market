"""Shared application-session composition and dependency-scope contracts."""

from __future__ import annotations

import asyncio
import inspect
import threading
from dataclasses import fields, replace
from pathlib import Path
from shutil import copy2, copytree

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import CheckpointTuple
from langgraph.checkpoint.memory import InMemorySaver
from llm_fakes import (
    TEST_CALLER_AUDIBLE_MODEL_TEXT_MAX_CHARS,
    TEST_STRUCTURED_OUTPUT_METHOD,
    FakeChatModel,
    NativeAsyncBlockingFakeChatModel,
)
from policy_helpers import make_policy
from routing_helpers import ArchitectureRoutingRecognizer
from telemetry_helpers import make_tenant_telemetry
from turn_helpers import (
    TEST_CANCELLATION_QUIESCENCE_TIMEOUT_SECONDS,
    committed_turn_events,
    engine_events,
)

from agnostic_market.agents.recovery import AUTOMATION_TERMINAL_LINE, TURN_FALLBACK_LINE
from agnostic_market.agents.telemetry import InMemoryTelemetrySink, SessionTelemetry
from agnostic_market.application import (
    APPLICATION_RESPONSIBILITIES,
    ApplicationModels,
    ApplicationResponsibility,
    ApplicationSessionState,
    ApplicationSettings,
    TenantServices,
    build_application_session,
    build_fixture_tenant_services,
    build_in_memory_session_state,
)
from agnostic_market.checkpoints import CheckpointScopeError
from agnostic_market.commerce.cart import CartStore
from agnostic_market.commerce.catalog import CatalogPort, FixtureCatalog
from agnostic_market.commerce.identity import (
    CustomerDirectory,
    CustomerDirectoryPort,
    load_customers_fixture,
    order_mutation_allowed,
    order_read_allowed,
)
from agnostic_market.commerce.orders import (
    GuestOrderScope,
    OrderPort,
    OrderStore,
    load_orders_fixture,
)
from agnostic_market.commerce.payment_instruments import (
    PaymentInstrumentDirectory,
    PaymentInstrumentPort,
    load_payment_instruments_fixture,
)
from agnostic_market.commerce.profile import ProfilePort, ProfileStore, load_profile_fixture
from agnostic_market.commerce.verification import (
    OtpPort,
    OtpProvider,
    RiskPort,
    RiskProvider,
    VerificationStore,
    load_verification_fixture,
)
from agnostic_market.config.registry import ConfigRegistry
from agnostic_market.dtos.events import (
    CommittedTurn,
    InterruptEvent,
    SpokenMessageEvent,
    TokenEvent,
)
from agnostic_market.dtos.recovery import PendingRecovery
from agnostic_market.dtos.state import CartLine, ReasoningState
from agnostic_market.tenancy.context import TenantContext, build_tenant_context

_TEST_DEPLOYMENT_ID = "test-application-artifact"


def test_application_composition_boundary_is_native_async() -> None:
    assert inspect.iscoroutinefunction(build_in_memory_session_state)
    assert inspect.iscoroutinefunction(build_application_session)


class _CheckpointReadOutageSaver(InMemorySaver):
    def __init__(self) -> None:
        super().__init__()
        self.read_attempts = 0
        self.delete_attempts = 0
        self.read_available = False

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        self.read_attempts += 1
        if not self.read_available:
            raise OSError("injected checkpoint read outage")
        return await super().aget_tuple(config)

    async def adelete_thread(self, thread_id: str) -> None:
        self.delete_attempts += 1
        self.read_available = True
        await super().adelete_thread(thread_id)


class _PostCommitBlockingOrderStore(OrderStore):
    def __init__(self, tenant_id: str, orders) -> None:
        super().__init__(tenant_id, orders)
        self.entered = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()

    def place_cart(self, idempotency_key: str, *, lines, total_usd):
        try:
            placed = super().place_cart(
                idempotency_key,
                lines=lines,
                total_usd=total_usd,
            )
            self.entered.set()
            if not self.release.wait(TEST_CANCELLATION_QUIESCENCE_TIMEOUT_SECONDS):
                raise TimeoutError("test did not release committed placement")
            return placed
        finally:
            self.finished.set()


def _tenant(tenant_id: str = "acme_store") -> TenantContext:
    return TenantContext(
        tenant_id=tenant_id,
        config_version="test-config",
        policy=make_policy(),
    )


def _settings() -> ApplicationSettings:
    return ApplicationSettings(
        display_name="Acme Store",
        caller_audible_model_text_max_chars=TEST_CALLER_AUDIBLE_MODEL_TEXT_MAX_CHARS,
        checkpoint_io_timeout_seconds=2.0,
        response_model_node_timeout_seconds=2.0,
        reasoning_model_node_timeout_seconds=6.0,
        cancellation_quiescence_timeout_seconds=(TEST_CANCELLATION_QUIESCENCE_TIMEOUT_SECONDS),
    )


def _models() -> ApplicationModels:
    return ApplicationModels(
        response=FakeChatModel(),
        reasoning=FakeChatModel(),
        response_structured_output_method=TEST_STRUCTURED_OUTPUT_METHOD,
    )


def _routing_factory(_registry):
    return ArchitectureRoutingRecognizer()


def test_application_composition_fields_have_one_explicit_owner() -> None:
    declared_fields = {
        (owner.__name__, field.name)
        for owner in (TenantServices, ApplicationSessionState)
        for field in fields(owner)
    }

    assert APPLICATION_RESPONSIBILITIES.keys() == declared_fields
    assert set(APPLICATION_RESPONSIBILITIES.values()) == set(ApplicationResponsibility)
    assert (
        APPLICATION_RESPONSIBILITIES[("TenantServices", "order_store")]
        is ApplicationResponsibility.DURABLE_TENANT_BUSINESS_STATE
    )
    assert (
        APPLICATION_RESPONSIBILITIES[("TenantServices", "checkpointer")]
        is ApplicationResponsibility.DURABLE_PLATFORM_SESSION_STATE
    )
    assert (
        APPLICATION_RESPONSIBILITIES[("ApplicationSessionState", "cart_store")]
        is ApplicationResponsibility.DURABLE_PLATFORM_SESSION_STATE
    )
    assert (
        APPLICATION_RESPONSIBILITIES[("ApplicationSessionState", "caller_context")]
        is ApplicationResponsibility.PROCESS_LOCAL_RUNTIME_COORDINATION
    )


def test_fixture_tenant_services_implement_the_replacement_ports(config_root: Path) -> None:
    tenant = _tenant()
    services = build_fixture_tenant_services(
        config_root,
        tenant,
        telemetry=make_tenant_telemetry("acme_store"),
    )
    contracts = (
        (services.catalog, CatalogPort),
        (services.order_store, OrderPort),
        (services.customers, CustomerDirectoryPort),
        (services.payment_instruments, PaymentInstrumentPort),
        (services.profile_store, ProfilePort),
        (services.otp, OtpPort),
        (services.risk, RiskPort),
    )

    assert all(isinstance(service, contract) for service, contract in contracts)
    assert {service.tenant_id for service, _contract in contracts} == {"acme_store"}


def _fixture_order_store(services: TenantServices) -> OrderStore:
    assert isinstance(services.order_store, OrderStore)
    return services.order_store


def test_fixture_tenant_services_use_normalized_context_identity(config_root: Path) -> None:
    tenant = _tenant("  acme_store  ")
    services = build_fixture_tenant_services(
        config_root,
        tenant,
        telemetry=make_tenant_telemetry("acme_store"),
    )

    assert services.tenant_id == "acme_store"
    assert {
        services.catalog.tenant_id,
        services.order_store.tenant_id,
        services.customers.tenant_id,
        services.payment_instruments.tenant_id,
        services.profile_store.tenant_id,
        services.otp.tenant_id,
        services.risk.tenant_id,
        services.telemetry.tenant_id,
    } == {"acme_store"}


@pytest.mark.asyncio
async def test_application_uses_explicit_deployment_artifact_for_checkpoint_namespace(
    config_root: Path,
) -> None:
    tenant = _tenant()
    services = build_fixture_tenant_services(
        config_root,
        tenant,
        telemetry=make_tenant_telemetry(tenant.tenant_id),
    )

    async def fixed_state(context, dependencies):
        return await build_in_memory_session_state(
            context,
            dependencies,
            session_id="same-session",
            thread_id="same-thread",
        )

    first = await build_application_session(
        tenant,
        _settings(),
        _models(),
        services,
        deployment_id="artifact-a",
        routing_factory=_routing_factory,
        session_state_factory=fixed_state,
    )
    second = await build_application_session(
        tenant,
        _settings(),
        _models(),
        services,
        deployment_id="artifact-b",
        routing_factory=_routing_factory,
        session_state_factory=fixed_state,
    )

    assert first.engine._checkpoint_binding.deployment_id == "artifact-a"
    assert second.engine._checkpoint_binding.deployment_id == "artifact-b"
    assert (
        first.engine._checkpoint_binding.storage_thread_id
        != second.engine._checkpoint_binding.storage_thread_id
    )
    await first.state.caller_context.aclose_session()
    await second.state.caller_context.aclose_session()


@pytest.mark.asyncio
async def test_shared_tenant_services_keep_guest_visibility_session_isolated(
    config_root: Path,
) -> None:
    tenant = _tenant()
    services = build_fixture_tenant_services(
        config_root,
        tenant,
        telemetry=make_tenant_telemetry(tenant.tenant_id),
    )

    async def first_state(context, dependencies):
        return await build_in_memory_session_state(
            context,
            dependencies,
            session_id="session-a",
            thread_id="thread-a",
        )

    async def second_state(context, dependencies):
        return await build_in_memory_session_state(
            context,
            dependencies,
            session_id="session-b",
            thread_id="thread-b",
        )

    first = await build_application_session(
        tenant,
        _settings(),
        _models(),
        services,
        deployment_id=_TEST_DEPLOYMENT_ID,
        routing_factory=_routing_factory,
        session_state_factory=first_state,
    )
    second = await build_application_session(
        tenant,
        _settings(),
        _models(),
        services,
        deployment_id=_TEST_DEPLOYMENT_ID,
        routing_factory=_routing_factory,
        session_state_factory=second_state,
    )
    placed = services.order_store.place_cart(
        "shared-store-placement",
        lines=(
            CartLine(
                sku="SKU-GRN-15",
                name="merino hiking socks",
                price_usd=14.50,
                quantity=1,
            ),
        ),
        total_usd=14.50,
    )
    first.state.guest_orders.record(placed.order_id)

    assert first.services is second.services
    assert first.engine is not second.engine
    assert first.state.guest_orders.contains(placed.order_id)
    assert not second.state.guest_orders.contains(placed.order_id)
    assert services.order_store.guest_orders(second.state.guest_orders) == []
    await first.state.caller_context.aclose_session()
    await second.state.caller_context.aclose_session()


@pytest.mark.asyncio
async def test_two_tenants_isolate_identical_logical_ids_on_one_checkpoint_backend(
    config_root: Path,
    tmp_path: Path,
) -> None:
    isolated_config_root = tmp_path / "config"
    copytree(config_root, isolated_config_root)
    for family in ("orders", "customers", "payment_instruments", "profiles", "verification"):
        copy2(
            isolated_config_root / "fixtures" / family / "acme_store.yaml",
            isolated_config_root / "fixtures" / family / "demo_shop.yaml",
        )
    registry = ConfigRegistry(isolated_config_root).load()
    tenants = tuple(
        build_tenant_context(registry, tenant_id) for tenant_id in ("acme_store", "demo_shop")
    )
    orders = load_orders_fixture(isolated_config_root, "acme_store")
    backend = InMemorySaver()
    services = tuple(
        build_fixture_tenant_services(
            isolated_config_root,
            tenant,
            telemetry=make_tenant_telemetry(tenant.tenant_id),
            checkpointer=backend,
        )
        for tenant in tenants
    )

    async def same_logical_state(context, dependencies):
        return await build_in_memory_session_state(
            context,
            dependencies,
            session_id="shared-logical-session",
            thread_id="shared-logical-thread",
        )

    response_models = (
        FakeChatModel(raise_transport=True),
        FakeChatModel(raise_transport=True),
    )
    reasoning_models = (
        FakeChatModel(raise_transport=True),
        FakeChatModel(raise_transport=True),
    )
    models = tuple(
        ApplicationModels(
            response=response,
            reasoning=reasoning,
            response_structured_output_method=TEST_STRUCTURED_OUTPUT_METHOD,
        )
        for response, reasoning in zip(response_models, reasoning_models, strict=True)
    )
    applications = []
    for tenant, dependencies, application_models in zip(
        tenants,
        services,
        models,
        strict=True,
    ):
        applications.append(
            await build_application_session(
                tenant,
                _settings(),
                application_models,
                dependencies,
                deployment_id=_TEST_DEPLOYMENT_ID,
                routing_factory=_routing_factory,
                session_state_factory=same_logical_state,
            )
        )
    product = orders.products[0]
    line = CartLine(
        sku=product.sku,
        name=product.name,
        price_usd=product.price_usd,
        quantity=1,
    )

    try:
        first_order = services[0].order_store.place_cart(
            "shared-idempotency-key",
            lines=(line,),
            total_usd=product.price_usd,
        )
        second_order = services[1].order_store.place_cart(
            "shared-idempotency-key",
            lines=(line,),
            total_usd=product.price_usd,
        )
        applications[0].state.guest_orders.record(first_order.order_id)
        applications[1].state.guest_orders.record(second_order.order_id)
        applications[0].state.cart_store.add_item(
            sku=product.sku,
            name=product.name,
            price_usd=product.price_usd,
            quantity=1,
        )

        first_view = await engine_events(applications[0].engine, "what is in my cart")
        second_view = await engine_events(applications[1].engine, "what is in my cart")

        assert first_order.order_id == second_order.order_id
        assert services[0].order_store is not services[1].order_store
        assert _fixture_order_store(services[0]).placed_count == 1
        assert _fixture_order_store(services[1]).placed_count == 1
        assert applications[0].state.session_id == applications[1].state.session_id
        assert applications[0].state.thread_id == applications[1].state.thread_id
        assert (
            applications[0].engine._checkpoint_binding.storage_thread_id
            != applications[1].engine._checkpoint_binding.storage_thread_id
        )
        assert any(
            isinstance(event, SpokenMessageEvent) and product.name in event.text
            for event in first_view
        )
        assert [event.text for event in second_view if isinstance(event, SpokenMessageEvent)] == [
            "Your cart's empty at the moment."
        ]
        assert all(model.invoke_count == 0 for model in (*response_models, *reasoning_models))
        assert services[0].order_store.is_guest_order(
            first_order.order_id,
            applications[0].state.guest_orders,
        )
        assert services[1].order_store.is_guest_order(
            second_order.order_id,
            applications[1].state.guest_orders,
        )
        with pytest.raises(ValueError, match="order-store tenant"):
            services[0].order_store.is_guest_order(
                second_order.order_id,
                applications[1].state.guest_orders,
            )
    finally:
        await applications[0].state.caller_context.aclose_session()
        await applications[1].state.caller_context.aclose_session()


@pytest.mark.asyncio
async def test_confirmed_placement_grants_authority_only_to_the_placing_session(
    config_root: Path,
) -> None:
    tenant = _tenant()
    services = build_fixture_tenant_services(
        config_root,
        tenant,
        telemetry=make_tenant_telemetry(tenant.tenant_id),
    )

    def session_state(session_id: str, thread_id: str):
        async def build_state(context, dependencies):
            return await build_in_memory_session_state(
                context,
                dependencies,
                session_id=session_id,
                thread_id=thread_id,
            )

        return build_state

    first = await build_application_session(
        tenant,
        _settings(),
        _models(),
        services,
        deployment_id=_TEST_DEPLOYMENT_ID,
        routing_factory=_routing_factory,
        session_state_factory=session_state("placing-session", "placing-thread"),
    )
    second = await build_application_session(
        tenant,
        _settings(),
        _models(),
        services,
        deployment_id=_TEST_DEPLOYMENT_ID,
        routing_factory=_routing_factory,
        session_state_factory=session_state("other-session", "other-thread"),
    )
    first.state.cart_store.add_item(
        sku="SKU-GRN-15",
        name="merino hiking socks",
        price_usd=14.50,
        quantity=1,
    )

    readback = await engine_events(first.engine, "place my order")
    pending_state = ReasoningState.model_validate(
        first.engine._graph.get_state(first.engine._config).values
    )
    pending = pending_state.pending_placement

    assert len([event for event in readback if isinstance(event, InterruptEvent)]) == 1
    assert pending is not None
    assert _fixture_order_store(services).placed_count == 0

    outcome = await engine_events(first.engine, "yes")
    receipt = services.order_store.placement_receipt(
        pending.idempotency_key,
        lines=pending.lines,
        total_usd=pending.total_usd,
    )

    assert receipt.kind == "committed"
    order_id = receipt.record.order_id
    assert any(
        isinstance(event, SpokenMessageEvent) and order_id in event.text for event in outcome
    )
    assert first.state.guest_orders.contains(order_id)
    assert services.order_store.guest_orders(first.state.guest_orders)[0].order_id == order_id
    assert order_read_allowed(
        order_id,
        store=services.order_store,
        guest_orders=first.state.guest_orders,
        identity=first.state.identity_store,
    )
    assert order_mutation_allowed(
        order_id,
        store=services.order_store,
        guest_orders=first.state.guest_orders,
        identity=first.state.identity_store,
    )

    assert services.order_store.guest_orders(second.state.guest_orders) == []
    assert not order_read_allowed(
        order_id,
        store=services.order_store,
        guest_orders=second.state.guest_orders,
        identity=second.state.identity_store,
    )
    assert not order_mutation_allowed(
        order_id,
        store=services.order_store,
        guest_orders=second.state.guest_orders,
        identity=second.state.identity_store,
    )

    await first.state.caller_context.aclose_session()

    assert first.state.guest_orders.order_refs == ()
    assert not order_read_allowed(
        order_id,
        store=services.order_store,
        guest_orders=first.state.guest_orders,
        identity=first.state.identity_store,
    )
    assert _fixture_order_store(services).placed_count == 1
    assert _fixture_order_store(services).order_summary(order_id) is not None
    assert (
        services.order_store.placement_receipt(
            pending.idempotency_key,
            lines=pending.lines,
            total_usd=pending.total_usd,
        ).kind
        == "committed"
    )
    await second.state.caller_context.aclose_session()


@pytest.mark.asyncio
async def test_application_completes_stt_shaped_cart_clarification_and_consent(
    config_root: Path,
) -> None:
    tenant = _tenant()
    services = build_fixture_tenant_services(
        config_root,
        tenant,
        telemetry=make_tenant_telemetry(tenant.tenant_id),
    )
    response = FakeChatModel(raise_transport=True)
    reasoning = FakeChatModel(
        scripted_calls=[
            [("provide_cart_item", {"candidate_key": "1"})],
            [("provide_cart_quantity", {"quantity": 2})],
        ]
    )
    application = await build_application_session(
        tenant,
        _settings(),
        ApplicationModels(
            response=response,
            reasoning=reasoning,
            response_structured_output_method=TEST_STRUCTURED_OUTPUT_METHOD,
        ),
        services,
        deployment_id=_TEST_DEPLOYMENT_ID,
        routing_factory=_routing_factory,
    )

    try:
        item_turn = await engine_events(
            application.engine,
            "could ya add one of those to my cart",
        )
        item_state = ReasoningState.model_validate(
            application.engine._graph.get_state(application.engine._config).values
        )

        assert any(
            isinstance(event, SpokenMessageEvent) and "how many" in event.text.casefold()
            for event in item_turn
        )
        assert item_state.active_invocation is not None
        assert item_state.active_invocation.request.kind == "modify_cart"
        assert item_state.pending_cart_mutation is None
        assert application.state.cart_store.is_empty()

        quantity_turn = await engine_events(application.engine, "make it two")
        quantity_state = ReasoningState.model_validate(
            application.engine._graph.get_state(application.engine._config).values
        )
        pending = quantity_state.pending_cart_mutation

        assert len([event for event in quantity_turn if isinstance(event, InterruptEvent)]) == 1
        assert pending is not None and pending.quantity == 2
        assert application.state.cart_store.is_empty()

        outcome = await engine_events(application.engine, "yeah go ahead")

        assert application.state.cart_store.view()[0].quantity == 2
        assert _fixture_order_store(services).placed_count == 0
        assert response.invoke_count == 0
        assert reasoning.invoke_count == 2
        assert any(
            isinstance(event, SpokenMessageEvent) and pending.name in event.text
            for event in outcome
        )
        operational = services.telemetry.operational_sink
        assert isinstance(operational, InMemoryTelemetrySink)
        assert [
            record.event for record in operational.records if record.event == "cart_item_added"
        ] == ["cart_item_added"]
    finally:
        await application.state.caller_context.aclose_session()


@pytest.mark.asyncio
async def test_application_redelivery_commits_one_placement_and_one_receipt(
    config_root: Path,
) -> None:
    tenant = _tenant()
    services = build_fixture_tenant_services(
        config_root,
        tenant,
        telemetry=make_tenant_telemetry(tenant.tenant_id),
    )
    response = FakeChatModel(raise_transport=True)
    reasoning = FakeChatModel(raise_transport=True)
    recognizer = ArchitectureRoutingRecognizer()

    async def fixed_state(context, dependencies):
        return await build_in_memory_session_state(
            context,
            dependencies,
            session_id="redelivered-placement-session",
            thread_id="redelivered-placement-thread",
        )

    application = await build_application_session(
        tenant,
        _settings(),
        ApplicationModels(
            response=response,
            reasoning=reasoning,
            response_structured_output_method=TEST_STRUCTURED_OUTPUT_METHOD,
        ),
        services,
        deployment_id=_TEST_DEPLOYMENT_ID,
        routing_factory=lambda _registry: recognizer,
        session_state_factory=fixed_state,
    )
    product = load_orders_fixture(config_root, tenant.tenant_id).products[0]
    application.state.cart_store.add_item(
        sku=product.sku,
        name=product.name,
        price_usd=product.price_usd,
        quantity=1,
    )
    dispatch = CommittedTurn(text="place my order", message_id="place-dispatch")
    consent = CommittedTurn(text="sure go ahead", message_id="place-consent")

    try:
        readback = await committed_turn_events(application.engine, dispatch)
        paused = ReasoningState.model_validate(
            application.engine._graph.get_state(application.engine._config).values
        )
        pending = paused.pending_placement

        assert len([event for event in readback if isinstance(event, InterruptEvent)]) == 1
        assert pending is not None
        assert _fixture_order_store(services).placed_count == 0

        assert await committed_turn_events(application.engine, dispatch) == []
        duplicate_pause = ReasoningState.model_validate(
            application.engine._graph.get_state(application.engine._config).values
        )
        assert duplicate_pause.pending_placement == pending

        outcome = await committed_turn_events(application.engine, consent)
        assert await committed_turn_events(application.engine, consent) == []

        receipt = services.order_store.placement_receipt(
            pending.idempotency_key,
            lines=pending.lines,
            total_usd=pending.total_usd,
        )
        assert receipt.kind == "committed"
        assert _fixture_order_store(services).placed_count == 1
        assert application.state.guest_orders.order_refs == (receipt.record.order_id,)
        assert any(
            isinstance(event, SpokenMessageEvent) and receipt.record.order_id in event.text
            for event in outcome
        )
        assert response.invoke_count == 0 and reasoning.invoke_count == 0
        assert [context.utterance for context in recognizer.contexts] == ["place my order"]
        operational = services.telemetry.operational_sink
        assert isinstance(operational, InMemoryTelemetrySink)
        assert [
            record.attributes["order_id"]
            for record in operational.records
            if record.event == "checkout_confirmed"
        ] == [receipt.record.order_id]
    finally:
        await application.state.caller_context.aclose_session()


@pytest.mark.asyncio
async def test_application_reconciles_receipt_after_external_effect_cancellation(
    config_root: Path,
) -> None:
    tenant = _tenant()
    services = build_fixture_tenant_services(
        config_root,
        tenant,
        telemetry=make_tenant_telemetry(tenant.tenant_id),
    )
    fixture = load_orders_fixture(config_root, tenant.tenant_id)
    order_store = _PostCommitBlockingOrderStore(tenant.tenant_id, fixture.orders)
    services = replace(services, order_store=order_store)
    response = FakeChatModel(raise_transport=True)
    reasoning = FakeChatModel(raise_transport=True)
    recognizer = ArchitectureRoutingRecognizer()
    application = await build_application_session(
        tenant,
        _settings(),
        ApplicationModels(
            response=response,
            reasoning=reasoning,
            response_structured_output_method=TEST_STRUCTURED_OUTPUT_METHOD,
        ),
        services,
        deployment_id=_TEST_DEPLOYMENT_ID,
        routing_factory=lambda _registry: recognizer,
    )
    product = fixture.products[0]
    application.state.cart_store.add_item(
        sku=product.sku,
        name=product.name,
        price_usd=product.price_usd,
        quantity=1,
    )

    try:
        await committed_turn_events(
            application.engine,
            CommittedTurn(text="place my order", message_id="cancelled-place-dispatch"),
        )
        pending = ReasoningState.model_validate(
            application.engine._graph.get_state(application.engine._config).values
        ).pending_placement
        assert pending is not None

        confirming = asyncio.create_task(
            committed_turn_events(
                application.engine,
                CommittedTurn(text="yes", message_id="cancelled-place-consent"),
            )
        )
        assert await asyncio.to_thread(
            order_store.entered.wait,
            TEST_CANCELLATION_QUIESCENCE_TIMEOUT_SECONDS,
        )
        confirming.cancel("transport disconnected")
        order_store.release.set()
        with pytest.raises(asyncio.CancelledError, match="transport disconnected"):
            await confirming
        assert await asyncio.to_thread(
            order_store.finished.wait,
            TEST_CANCELLATION_QUIESCENCE_TIMEOUT_SECONDS,
        )

        interrupted = ReasoningState.model_validate(
            application.engine._graph.get_state(application.engine._config).values
        )
        assert isinstance(interrupted.pending_recovery, PendingRecovery)
        assert order_store.placed_count == 1

        recovered = await committed_turn_events(
            application.engine,
            CommittedTurn(text="continue", message_id="cancelled-place-recovery"),
        )
        completed = ReasoningState.model_validate(
            application.engine._graph.get_state(application.engine._config).values
        )
        receipt = order_store.placement_receipt(
            pending.idempotency_key,
            lines=pending.lines,
            total_usd=pending.total_usd,
        )

        assert receipt.kind == "committed"
        assert order_store.placed_count == 1
        assert completed.pending_recovery is None
        assert completed.pending_placement is None
        assert any(
            isinstance(event, SpokenMessageEvent) and receipt.record.order_id in event.text
            for event in recovered
        )
        assert response.invoke_count == 0 and reasoning.invoke_count == 0
        assert [context.utterance for context in recognizer.contexts] == ["place my order"]
        operational = services.telemetry.operational_sink
        assert isinstance(operational, InMemoryTelemetrySink)
        assert [
            record.attributes["order_id"]
            for record in operational.records
            if record.event == "checkout_confirmed"
        ] == [receipt.record.order_id]
    finally:
        order_store.release.set()
        await application.state.caller_context.aclose_session()


@pytest.mark.asyncio
async def test_application_engine_reconstruction_resumes_the_same_session_dependencies(
    config_root: Path,
) -> None:
    tenant = _tenant()
    services = build_fixture_tenant_services(
        config_root,
        tenant,
        telemetry=make_tenant_telemetry(tenant.tenant_id),
    )
    response = FakeChatModel(raise_transport=True)
    reasoning = FakeChatModel(raise_transport=True)

    async def fixed_identity_state(context, dependencies):
        return await build_in_memory_session_state(
            context,
            dependencies,
            session_id="reconstructed-session",
            thread_id="reconstructed-thread",
        )

    first = await build_application_session(
        tenant,
        _settings(),
        ApplicationModels(
            response=response,
            reasoning=reasoning,
            response_structured_output_method=TEST_STRUCTURED_OUTPUT_METHOD,
        ),
        services,
        deployment_id=_TEST_DEPLOYMENT_ID,
        routing_factory=_routing_factory,
        session_state_factory=fixed_identity_state,
    )
    product = load_orders_fixture(config_root, tenant.tenant_id).products[0]
    first.state.cart_store.add_item(
        sku=product.sku,
        name=product.name,
        price_usd=product.price_usd,
        quantity=1,
    )
    reconstructed = None

    try:
        await committed_turn_events(
            first.engine,
            CommittedTurn(text="place my order", message_id="reconstructed-dispatch"),
        )
        paused = ReasoningState.model_validate(
            first.engine._graph.get_state(first.engine._config).values
        )
        pending = paused.pending_placement
        assert pending is not None

        reconstructed = await build_application_session(
            tenant,
            _settings(),
            ApplicationModels(
                response=response,
                reasoning=reasoning,
                response_structured_output_method=TEST_STRUCTURED_OUTPUT_METHOD,
            ),
            services,
            deployment_id=_TEST_DEPLOYMENT_ID,
            routing_factory=_routing_factory,
            session_state_factory=fixed_identity_state,
        )

        assert reconstructed.state is not first.state
        assert reconstructed.state.caller_context is not first.state.caller_context
        assert reconstructed.state.cart_store is not first.state.cart_store
        assert reconstructed.state.session_id == first.state.session_id
        assert reconstructed.state.thread_id == first.state.thread_id
        assert reconstructed.engine is not first.engine
        assert await reconstructed.engine.apending_interrupt()

        outcome = await committed_turn_events(
            reconstructed.engine,
            CommittedTurn(text="yes", message_id="reconstructed-consent"),
        )
        receipt = services.order_store.placement_receipt(
            pending.idempotency_key,
            lines=pending.lines,
            total_usd=pending.total_usd,
        )

        assert receipt.kind == "committed"
        assert _fixture_order_store(services).placed_count == 1
        assert reconstructed.state.guest_orders.order_refs == (receipt.record.order_id,)
        assert any(
            isinstance(event, SpokenMessageEvent) and receipt.record.order_id in event.text
            for event in outcome
        )
        assert response.invoke_count == 0 and reasoning.invoke_count == 0
    finally:
        if reconstructed is None:
            await first.state.caller_context.aclose_session()
        else:
            # The first context represents the lost worker. Only its replacement closes the
            # recovered checkpoint namespace.
            await reconstructed.state.caller_context.aclose_session()


@pytest.mark.asyncio
async def test_application_principal_rotation_retires_old_thread_and_completes_typed_read(
    config_root: Path,
) -> None:
    tenant = _tenant()
    customers = load_customers_fixture(config_root, tenant.tenant_id)
    orders = load_orders_fixture(config_root, tenant.tenant_id)
    verification = load_verification_fixture(config_root, tenant.tenant_id)
    customer_ref, customer = next(iter(customers.customers.items()))
    owned_order_ids = tuple(
        order_id for order_id, order in orders.orders.items() if order.customer_ref == customer_ref
    )
    foreign_order_ids = tuple(
        order_id for order_id, order in orders.orders.items() if order.customer_ref != customer_ref
    )
    assert owned_order_ids and foreign_order_ids
    services = build_fixture_tenant_services(
        config_root,
        tenant,
        telemetry=make_tenant_telemetry(tenant.tenant_id),
    )
    response = FakeChatModel(raise_transport=True)
    reasoning = FakeChatModel(
        force_tool="propose_identity",
        canned_args={"propose_identity": {"contact_claim": customer.contact}},
        tool_call_limit=1,
    )
    application = await build_application_session(
        tenant,
        _settings(),
        ApplicationModels(
            response=response,
            reasoning=reasoning,
            response_structured_output_method=TEST_STRUCTURED_OUTPUT_METHOD,
        ),
        services,
        deployment_id=_TEST_DEPLOYMENT_ID,
        routing_factory=_routing_factory,
    )
    old_thread_id = application.engine.thread_id
    old_config = application.engine._config

    try:
        verification_prompt = await engine_events(application.engine, "show me my orders")
        assert [
            event.prompt for event in verification_prompt if isinstance(event, InterruptEvent)
        ] == ["For security, please read me the 6-digit code we just sent you."]
        assert application.state.identity_store.current() is None

        completed = await engine_events(
            application.engine,
            verification.otp_codes_by_factor_ref[customer.factor_ref],
        )
        bound = application.state.identity_store.current()
        state = ReasoningState.model_validate(
            application.engine._graph.get_state(application.engine._config).values
        )
        spoken = " ".join(
            event.text for event in completed if isinstance(event, SpokenMessageEvent)
        )

        assert bound is not None and bound.customer_ref == customer_ref
        assert application.engine.thread_id != old_thread_id
        with pytest.raises(CheckpointScopeError, match="namespace"):
            application.engine._graph.get_state(old_config)
        assert state.active_invocation is None
        assert state.pending_identity is None
        assert all(order_id in spoken for order_id in owned_order_ids)
        assert all(order_id not in spoken for order_id in foreign_order_ids)
        assert response.invoke_count == 0
        assert reasoning.invoke_count == 1
    finally:
        await application.state.caller_context.aclose_session()


@pytest.mark.asyncio
async def test_application_request_person_route_terminalizes_and_closes_once(
    config_root: Path,
) -> None:
    tenant = _tenant()
    services = build_fixture_tenant_services(
        config_root,
        tenant,
        telemetry=make_tenant_telemetry(tenant.tenant_id),
    )
    response = FakeChatModel(raise_transport=True)
    reasoning = FakeChatModel(raise_transport=True)
    recognizer = ArchitectureRoutingRecognizer()
    application = await build_application_session(
        tenant,
        _settings(),
        ApplicationModels(
            response=response,
            reasoning=reasoning,
            response_structured_output_method=TEST_STRUCTURED_OUTPUT_METHOD,
        ),
        services,
        deployment_id=_TEST_DEPLOYMENT_ID,
        routing_factory=lambda _registry: recognizer,
    )

    try:
        requested = await engine_events(application.engine, "i want a person")
        repeated = await engine_events(application.engine, "what is in my cart")
        state = ReasoningState.model_validate(
            application.engine._graph.get_state(application.engine._config).values
        )

        assert [
            (event.node, event.text) for event in requested if isinstance(event, SpokenMessageEvent)
        ] == [("automation_terminal_response", AUTOMATION_TERMINAL_LINE)]
        assert [event.text for event in repeated if isinstance(event, SpokenMessageEvent)] == [
            AUTOMATION_TERMINAL_LINE
        ]
        assert state.automation_terminal
        assert state.active_invocation is None
        assert response.invoke_count == 0 and reasoning.invoke_count == 0
        assert [context.utterance for context in recognizer.contexts] == ["i want a person"]
        assert _fixture_order_store(services).placed_count == 0
    finally:
        await application.state.caller_context.aclose_session()

    operational = services.telemetry.operational_sink
    assert isinstance(operational, InMemoryTelemetrySink)
    assert [record.event for record in operational.records].count("caller_context_closed") == 1
    assert application.state.cart_store.is_empty()
    assert application.state.guest_orders.order_refs == ()


@pytest.mark.asyncio
async def test_natural_catalog_request_reaches_grounded_owner_and_speech(
    config_root: Path,
) -> None:
    tenant = _tenant()
    services = build_fixture_tenant_services(
        config_root,
        tenant,
        telemetry=make_tenant_telemetry(tenant.tenant_id),
    )
    response = FakeChatModel(
        emit_tool_calls=False,
        text_response="Yes, we carry the waterproof rain jacket for $129.00.",
        record_prompts=True,
    )
    reasoning = FakeChatModel(raise_transport=True)
    application = await build_application_session(
        tenant,
        _settings(),
        ApplicationModels(
            response=response,
            reasoning=reasoning,
            response_structured_output_method=TEST_STRUCTURED_OUTPUT_METHOD,
        ),
        services,
        deployment_id=_TEST_DEPLOYMENT_ID,
        routing_factory=_routing_factory,
    )

    events = await engine_events(
        application.engine,
        "Do you carry the waterproof rain jacket?",
    )

    assert [event.text for event in events if isinstance(event, TokenEvent)] == [
        "Yes, we carry the waterproof rain jacket for $129.00."
    ]
    assert response.invoke_count == 1
    assert reasoning.invoke_count == 0
    prompt = response._seen_prompts[-1]
    assert "waterproof rain jacket; SKU SKU-BLU-07; price $129.00" in prompt
    assert "trail running shoes" not in prompt
    assert application.state.cart_store.is_empty()
    assert _fixture_order_store(services).placed_count == 0
    operational_sink = services.telemetry.operational_sink
    routing_sink = services.telemetry.routing_evidence_sink
    assert isinstance(operational_sink, InMemoryTelemetrySink)
    assert isinstance(routing_sink, InMemoryTelemetrySink)
    assert not any(record.event == "capability_answered" for record in operational_sink.records)
    assert [
        record.attributes["capability"]
        for record in routing_sink.records
        if record.event == "capability_answered"
    ] == ["search_catalog"]
    await application.state.caller_context.aclose_session()


@pytest.mark.asyncio
async def test_application_response_model_timeout_is_bounded_and_effect_free(
    config_root: Path,
) -> None:
    tenant = _tenant()
    services = build_fixture_tenant_services(
        config_root,
        tenant,
        telemetry=make_tenant_telemetry(tenant.tenant_id),
    )
    response = NativeAsyncBlockingFakeChatModel()
    reasoning = FakeChatModel(raise_transport=True)
    application = await build_application_session(
        tenant,
        replace(_settings(), response_model_node_timeout_seconds=0.05),
        ApplicationModels(
            response=response,
            reasoning=reasoning,
            response_structured_output_method=TEST_STRUCTURED_OUTPUT_METHOD,
        ),
        services,
        deployment_id=_TEST_DEPLOYMENT_ID,
        routing_factory=_routing_factory,
    )

    try:
        events = await engine_events(application.engine, "tell me about the rain jacket")
        state = ReasoningState.model_validate(
            application.engine._graph.get_state(application.engine._config).values
        )

        assert response.started.is_set() and response.cancelled.is_set()
        assert reasoning.invoke_count == 0
        assert [event.text for event in events if isinstance(event, SpokenMessageEvent)] == [
            TURN_FALLBACK_LINE
        ]
        assert application.state.cart_store.is_empty()
        assert _fixture_order_store(services).placed_count == 0
        assert state.active_invocation is None
        assert not state.automation_terminal
    finally:
        await application.state.caller_context.aclose_session()


@pytest.mark.asyncio
async def test_application_reasoning_model_timeout_is_bounded_and_effect_free(
    config_root: Path,
) -> None:
    tenant = _tenant()
    services = build_fixture_tenant_services(
        config_root,
        tenant,
        telemetry=make_tenant_telemetry(tenant.tenant_id),
    )
    response = FakeChatModel(raise_transport=True)
    reasoning = NativeAsyncBlockingFakeChatModel()
    application = await build_application_session(
        tenant,
        replace(_settings(), reasoning_model_node_timeout_seconds=0.05),
        ApplicationModels(
            response=response,
            reasoning=reasoning,
            response_structured_output_method=TEST_STRUCTURED_OUTPUT_METHOD,
        ),
        services,
        deployment_id=_TEST_DEPLOYMENT_ID,
        routing_factory=_routing_factory,
    )

    try:
        events = await engine_events(application.engine, "add something to my cart")
        state = ReasoningState.model_validate(
            application.engine._graph.get_state(application.engine._config).values
        )

        assert reasoning.started.is_set() and reasoning.cancelled.is_set()
        assert response.invoke_count == 0
        assert [event.text for event in events if isinstance(event, SpokenMessageEvent)] == [
            "Your cart is empty. Please review your cart before trying again."
        ]
        assert application.state.cart_store.is_empty()
        assert _fixture_order_store(services).placed_count == 0
        assert state.active_invocation is None
        assert not state.automation_terminal
    finally:
        await application.state.caller_context.aclose_session()


@pytest.mark.asyncio
async def test_application_checkpoint_read_outage_prevents_admission_and_effects(
    config_root: Path,
) -> None:
    tenant = _tenant()
    backend = _CheckpointReadOutageSaver()
    services = build_fixture_tenant_services(
        config_root,
        tenant,
        telemetry=make_tenant_telemetry(tenant.tenant_id),
        checkpointer=backend,
    )
    response = FakeChatModel(raise_transport=True)
    reasoning = FakeChatModel(raise_transport=True)
    recognizer = ArchitectureRoutingRecognizer()
    application = await build_application_session(
        tenant,
        _settings(),
        ApplicationModels(
            response=response,
            reasoning=reasoning,
            response_structured_output_method=TEST_STRUCTURED_OUTPUT_METHOD,
        ),
        services,
        deployment_id=_TEST_DEPLOYMENT_ID,
        routing_factory=lambda _registry: recognizer,
    )

    try:
        events = await engine_events(application.engine, "place my order")

        assert [event.text for event in events if isinstance(event, SpokenMessageEvent)] == [
            AUTOMATION_TERMINAL_LINE
        ]
        assert backend.read_attempts >= 1
        assert backend.delete_attempts == 0
        assert recognizer.contexts == []
        assert response.invoke_count == 0 and reasoning.invoke_count == 0
        assert application.state.cart_store.is_empty()
        assert _fixture_order_store(services).placed_count == 0
        assert application.engine._terminal_latched
    finally:
        await application.state.caller_context.aclose_session()


async def test_application_rejects_services_from_another_tenant(config_root: Path) -> None:
    tenant = _tenant()
    services = build_fixture_tenant_services(
        config_root,
        tenant,
        telemetry=make_tenant_telemetry(tenant.tenant_id),
    )

    with pytest.raises(ValueError, match="services do not match"):
        await build_application_session(
            tenant,
            _settings(),
            _models(),
            replace(services, tenant_id="other_store"),
            deployment_id=_TEST_DEPLOYMENT_ID,
            routing_factory=_routing_factory,
        )


async def test_application_rejects_every_cross_tenant_service(config_root: Path) -> None:
    tenant = _tenant()
    services = build_fixture_tenant_services(
        config_root,
        tenant,
        telemetry=make_tenant_telemetry(tenant.tenant_id),
    )
    orders = load_orders_fixture(config_root, tenant.tenant_id)
    customers = load_customers_fixture(config_root, tenant.tenant_id)
    instruments = load_payment_instruments_fixture(config_root, tenant.tenant_id)
    profiles = load_profile_fixture(config_root, tenant.tenant_id)
    verification = load_verification_fixture(config_root, tenant.tenant_id)
    mismatches = {
        "catalog": FixtureCatalog("other_store", orders),
        "order_store": OrderStore("other_store", orders.orders),
        "customers": CustomerDirectory("other_store", customers),
        "payment_instruments": PaymentInstrumentDirectory("other_store", instruments),
        "profile_store": ProfileStore("other_store", profiles),
        "otp": OtpProvider(
            "other_store",
            codes_by_factor_ref=verification.otp_codes_by_factor_ref,
            challenge_ttl_seconds=verification.challenge_ttl_seconds,
            proof_ttl_seconds=verification.proof_ttl_seconds,
        ),
        "risk": RiskProvider("other_store"),
        "telemetry": make_tenant_telemetry("other_store"),
    }

    async def session_state_must_not_build(_tenant, _services):
        raise AssertionError("cross-tenant services reached session-state construction")

    for field_name, service in mismatches.items():
        with pytest.raises(ValueError, match=field_name):
            await build_application_session(
                tenant,
                _settings(),
                _models(),
                replace(services, **{field_name: service}),
                deployment_id=_TEST_DEPLOYMENT_ID,
                routing_factory=_routing_factory,
                session_state_factory=session_state_must_not_build,
            )


async def test_application_rejects_a_service_without_tenant_identity(
    config_root: Path,
) -> None:
    tenant = _tenant()
    services = build_fixture_tenant_services(
        config_root,
        tenant,
        telemetry=make_tenant_telemetry(tenant.tenant_id),
    )

    with pytest.raises(TypeError, match="customers does not expose a tenant identity"):
        await build_application_session(
            tenant,
            _settings(),
            _models(),
            replace(services, customers=object()),  # type: ignore[arg-type]
            deployment_id=_TEST_DEPLOYMENT_ID,
            routing_factory=_routing_factory,
        )


async def test_application_rejects_a_cross_tenant_guest_scope(config_root: Path) -> None:
    tenant = _tenant()
    services = build_fixture_tenant_services(
        config_root,
        tenant,
        telemetry=make_tenant_telemetry(tenant.tenant_id),
    )

    async def mismatched_state(_context, dependencies):
        return await build_in_memory_session_state(
            _tenant("other_store"),
            dependencies,
            session_id="wrong-tenant-session",
            thread_id="wrong-tenant-thread",
        )

    with pytest.raises(ValueError, match="guest-order scope does not match"):
        await build_application_session(
            tenant,
            _settings(),
            _models(),
            services,
            deployment_id=_TEST_DEPLOYMENT_ID,
            routing_factory=_routing_factory,
            session_state_factory=mismatched_state,
        )


async def test_application_rejects_a_guest_scope_from_another_session(
    config_root: Path,
) -> None:
    tenant = _tenant()
    services = build_fixture_tenant_services(
        config_root,
        tenant,
        telemetry=make_tenant_telemetry(tenant.tenant_id),
    )

    async def mismatched_state(context, dependencies):
        state = await build_in_memory_session_state(
            context,
            dependencies,
            session_id="session-a",
            thread_id="thread-a",
        )
        return replace(
            state,
            guest_orders=GuestOrderScope(
                tenant_id=context.tenant_id,
                session_id="session-b",
            ),
        )

    with pytest.raises(ValueError, match="scope does not match the application session"):
        await build_application_session(
            tenant,
            _settings(),
            _models(),
            services,
            deployment_id=_TEST_DEPLOYMENT_ID,
            routing_factory=_routing_factory,
            session_state_factory=mismatched_state,
        )


async def test_application_rejects_split_brain_lifecycle_stores(config_root: Path) -> None:
    tenant = _tenant()
    services = build_fixture_tenant_services(
        config_root,
        tenant,
        telemetry=make_tenant_telemetry(tenant.tenant_id),
    )

    async def mismatched_state(context, dependencies):
        state = await build_in_memory_session_state(context, dependencies)
        return replace(state, cart_store=CartStore())

    with pytest.raises(ValueError, match=r"lifecycle.*cart"):
        await build_application_session(
            tenant,
            _settings(),
            _models(),
            services,
            deployment_id=_TEST_DEPLOYMENT_ID,
            routing_factory=_routing_factory,
            session_state_factory=mismatched_state,
        )


async def test_application_rejects_verification_store_using_another_otp_service(
    config_root: Path,
) -> None:
    tenant = _tenant()
    services = build_fixture_tenant_services(
        config_root,
        tenant,
        telemetry=make_tenant_telemetry(tenant.tenant_id),
    )

    async def mismatched_state(context, dependencies):
        state = await build_in_memory_session_state(context, dependencies)
        verification = VerificationStore(
            OtpProvider(
                context.tenant_id,
                codes_by_factor_ref={"CUST-001": "111111"},
                challenge_ttl_seconds=300,
                proof_ttl_seconds=300,
            ),
            session_id=state.session_id,
        )
        return replace(
            state,
            verification_store=verification,
            caller_context=replace(
                state.caller_context,
                verification_store=verification,
            ),
        )

    with pytest.raises(ValueError, match="tenant OTP service"):
        await build_application_session(
            tenant,
            _settings(),
            _models(),
            services,
            deployment_id=_TEST_DEPLOYMENT_ID,
            routing_factory=_routing_factory,
            session_state_factory=mismatched_state,
        )


async def test_application_rejects_verification_store_for_another_session(
    config_root: Path,
) -> None:
    tenant = _tenant()
    services = build_fixture_tenant_services(
        config_root,
        tenant,
        telemetry=make_tenant_telemetry(tenant.tenant_id),
    )

    async def mismatched_state(context, dependencies):
        state = await build_in_memory_session_state(context, dependencies)
        verification = VerificationStore(
            dependencies.otp,
            session_id="foreign-session",
        )
        return replace(
            state,
            verification_store=verification,
            caller_context=replace(
                state.caller_context,
                verification_store=verification,
            ),
        )

    with pytest.raises(ValueError, match=r"verification.*session"):
        await build_application_session(
            tenant,
            _settings(),
            _models(),
            services,
            deployment_id=_TEST_DEPLOYMENT_ID,
            routing_factory=_routing_factory,
            session_state_factory=mismatched_state,
        )


async def test_application_rejects_split_brain_lifecycle_telemetry(config_root: Path) -> None:
    tenant = _tenant()
    services = build_fixture_tenant_services(
        config_root,
        tenant,
        telemetry=make_tenant_telemetry(tenant.tenant_id),
    )

    async def mismatched_state(context, dependencies):
        state = await build_in_memory_session_state(context, dependencies)
        other = make_tenant_telemetry(context.tenant_id).bind_session(state.session_id)
        return replace(
            state,
            caller_context=replace(state.caller_context, telemetry=other.operational),
        )

    with pytest.raises(ValueError, match=r"lifecycle.*telemetry"):
        await build_application_session(
            tenant,
            _settings(),
            _models(),
            services,
            deployment_id=_TEST_DEPLOYMENT_ID,
            routing_factory=_routing_factory,
            session_state_factory=mismatched_state,
        )


async def test_application_rejects_session_telemetry_from_another_sink(
    config_root: Path,
) -> None:
    tenant = _tenant()
    services = build_fixture_tenant_services(
        config_root,
        tenant,
        telemetry=make_tenant_telemetry(tenant.tenant_id),
    )

    async def mismatched_state(context, dependencies):
        state = await build_in_memory_session_state(context, dependencies)
        other = make_tenant_telemetry(context.tenant_id).bind_session(state.session_id)
        return replace(state, telemetry=other)

    with pytest.raises(ValueError, match="tenant telemetry service"):
        await build_application_session(
            tenant,
            _settings(),
            _models(),
            services,
            deployment_id=_TEST_DEPLOYMENT_ID,
            routing_factory=_routing_factory,
            session_state_factory=mismatched_state,
        )


async def test_application_rejects_swapped_telemetry_purposes(config_root: Path) -> None:
    tenant = _tenant()
    services = build_fixture_tenant_services(
        config_root,
        tenant,
        telemetry=make_tenant_telemetry(tenant.tenant_id),
    )

    async def mismatched_state(context, dependencies):
        state = await build_in_memory_session_state(context, dependencies)
        return replace(
            state,
            telemetry=SessionTelemetry(
                operational=state.telemetry.routing_evidence,
                routing_evidence=state.telemetry.operational,
            ),
        )

    with pytest.raises(ValueError, match="telemetry purpose"):
        await build_application_session(
            tenant,
            _settings(),
            _models(),
            services,
            deployment_id=_TEST_DEPLOYMENT_ID,
            routing_factory=_routing_factory,
            session_state_factory=mismatched_state,
        )
