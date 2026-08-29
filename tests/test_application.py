"""Shared application-session composition and dependency-scope contracts."""

from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path

import pytest
from llm_fakes import (
    TEST_CALLER_AUDIBLE_MODEL_TEXT_MAX_CHARS,
    TEST_STRUCTURED_OUTPUT_METHOD,
    FakeChatModel,
)
from policy_helpers import make_policy
from routing_helpers import ArchitectureRoutingRecognizer
from telemetry_helpers import make_tenant_telemetry
from turn_helpers import (
    TEST_CANCELLATION_QUIESCENCE_TIMEOUT_SECONDS,
    engine_events,
)

from agnostic_market.agents.telemetry import InMemoryTelemetrySink, SessionTelemetry
from agnostic_market.application import (
    ApplicationModels,
    ApplicationSettings,
    TenantServices,
    build_application_session,
    build_fixture_tenant_services,
    build_in_memory_session_state,
)
from agnostic_market.commerce.cart import CartStore
from agnostic_market.commerce.catalog import FixtureCatalog
from agnostic_market.commerce.identity import order_mutation_allowed, order_read_allowed
from agnostic_market.commerce.orders import GuestOrderScope, OrderStore, load_orders_fixture
from agnostic_market.dtos.events import InterruptEvent, SpokenMessageEvent, TokenEvent
from agnostic_market.dtos.state import CartLine, ReasoningState
from agnostic_market.tenancy.context import TenantContext

_TEST_DEPLOYMENT_ID = "test-application-artifact"


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


def test_tenant_services_fields_have_explicit_lifetime_and_mutability() -> None:
    contracts = {
        "tenant_id": ("tenant", "immutable"),
        "catalog": ("tenant", "immutable"),
        "order_store": ("tenant", "mutable"),
        "customers": ("tenant", "immutable"),
        "payment_instruments": ("tenant", "immutable"),
        "profile_store": ("tenant", "mutable"),
        "otp": ("tenant", "mutable"),
        "risk": ("tenant", "immutable"),
        "checkpointer": ("tenant", "mutable"),
        "telemetry": ("tenant", "mutable"),
    }

    assert {field.name for field in fields(TenantServices)} == contracts.keys()
    assert set(contracts.values()) <= {
        ("tenant", "immutable"),
        ("tenant", "mutable"),
        ("session", "immutable"),
        ("session", "mutable"),
    }


def test_application_uses_explicit_deployment_artifact_for_checkpoint_namespace(
    config_root: Path,
) -> None:
    tenant = _tenant()
    services = build_fixture_tenant_services(
        config_root,
        tenant.tenant_id,
        telemetry=make_tenant_telemetry(tenant.tenant_id),
    )

    def fixed_state(context, dependencies):
        return build_in_memory_session_state(
            context,
            dependencies,
            session_id="same-session",
            thread_id="same-thread",
        )

    first = build_application_session(
        tenant,
        _settings(),
        _models(),
        services,
        deployment_id="artifact-a",
        routing_factory=_routing_factory,
        session_state_factory=fixed_state,
    )
    second = build_application_session(
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


def test_shared_tenant_services_keep_guest_visibility_session_isolated(
    config_root: Path,
) -> None:
    tenant = _tenant()
    services = build_fixture_tenant_services(
        config_root,
        tenant.tenant_id,
        telemetry=make_tenant_telemetry(tenant.tenant_id),
    )

    def first_state(context, dependencies):
        return build_in_memory_session_state(
            context,
            dependencies,
            session_id="session-a",
            thread_id="thread-a",
        )

    def second_state(context, dependencies):
        return build_in_memory_session_state(
            context,
            dependencies,
            session_id="session-b",
            thread_id="thread-b",
        )

    first = build_application_session(
        tenant,
        _settings(),
        _models(),
        services,
        deployment_id=_TEST_DEPLOYMENT_ID,
        routing_factory=_routing_factory,
        session_state_factory=first_state,
    )
    second = build_application_session(
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


@pytest.mark.asyncio
async def test_confirmed_placement_grants_authority_only_to_the_placing_session(
    config_root: Path,
) -> None:
    tenant = _tenant()
    services = build_fixture_tenant_services(
        config_root,
        tenant.tenant_id,
        telemetry=make_tenant_telemetry(tenant.tenant_id),
    )

    def session_state(session_id: str, thread_id: str):
        def build_state(context, dependencies):
            return build_in_memory_session_state(
                context,
                dependencies,
                session_id=session_id,
                thread_id=thread_id,
            )

        return build_state

    first = build_application_session(
        tenant,
        _settings(),
        _models(),
        services,
        deployment_id=_TEST_DEPLOYMENT_ID,
        routing_factory=_routing_factory,
        session_state_factory=session_state("placing-session", "placing-thread"),
    )
    second = build_application_session(
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
    assert services.order_store.placed_count == 0

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
    assert services.order_store.placed_count == 1
    assert services.order_store.order_summary(order_id) is not None
    assert (
        services.order_store.placement_receipt(
            pending.idempotency_key,
            lines=pending.lines,
            total_usd=pending.total_usd,
        ).kind
        == "committed"
    )


@pytest.mark.asyncio
async def test_natural_catalog_request_reaches_grounded_owner_and_speech(
    config_root: Path,
) -> None:
    tenant = _tenant()
    services = build_fixture_tenant_services(
        config_root,
        tenant.tenant_id,
        telemetry=make_tenant_telemetry(tenant.tenant_id),
    )
    response = FakeChatModel(
        emit_tool_calls=False,
        text_response="Yes, we carry the waterproof rain jacket for $129.00.",
        record_prompts=True,
    )
    reasoning = FakeChatModel(raise_transport=True)
    application = build_application_session(
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
    assert services.order_store.placed_count == 0
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


def test_application_rejects_services_from_another_tenant(config_root: Path) -> None:
    tenant = _tenant()
    services = build_fixture_tenant_services(
        config_root,
        tenant.tenant_id,
        telemetry=make_tenant_telemetry(tenant.tenant_id),
    )

    with pytest.raises(ValueError, match="services do not match"):
        build_application_session(
            tenant,
            _settings(),
            _models(),
            replace(services, tenant_id="other_store"),
            deployment_id=_TEST_DEPLOYMENT_ID,
            routing_factory=_routing_factory,
        )


def test_application_rejects_a_catalog_from_another_tenant(config_root: Path) -> None:
    tenant = _tenant()
    services = build_fixture_tenant_services(
        config_root,
        tenant.tenant_id,
        telemetry=make_tenant_telemetry(tenant.tenant_id),
    )
    fixture = load_orders_fixture(config_root, tenant.tenant_id)

    with pytest.raises(ValueError, match="catalog service does not match"):
        build_application_session(
            tenant,
            _settings(),
            _models(),
            replace(services, catalog=FixtureCatalog("other_store", fixture)),
            deployment_id=_TEST_DEPLOYMENT_ID,
            routing_factory=_routing_factory,
        )


def test_application_rejects_an_order_store_from_another_tenant(config_root: Path) -> None:
    tenant = _tenant()
    services = build_fixture_tenant_services(
        config_root,
        tenant.tenant_id,
        telemetry=make_tenant_telemetry(tenant.tenant_id),
    )
    fixture = load_orders_fixture(config_root, tenant.tenant_id)

    with pytest.raises(ValueError, match="order service does not match"):
        build_application_session(
            tenant,
            _settings(),
            _models(),
            replace(services, order_store=OrderStore("other_store", fixture.orders)),
            deployment_id=_TEST_DEPLOYMENT_ID,
            routing_factory=_routing_factory,
        )


def test_application_rejects_a_cross_tenant_guest_scope(config_root: Path) -> None:
    tenant = _tenant()
    services = build_fixture_tenant_services(
        config_root,
        tenant.tenant_id,
        telemetry=make_tenant_telemetry(tenant.tenant_id),
    )

    def mismatched_state(_context, dependencies):
        return build_in_memory_session_state(
            _tenant("other_store"),
            dependencies,
            session_id="wrong-tenant-session",
            thread_id="wrong-tenant-thread",
        )

    with pytest.raises(ValueError, match="guest-order scope does not match"):
        build_application_session(
            tenant,
            _settings(),
            _models(),
            services,
            deployment_id=_TEST_DEPLOYMENT_ID,
            routing_factory=_routing_factory,
            session_state_factory=mismatched_state,
        )


def test_application_rejects_a_guest_scope_from_another_session(config_root: Path) -> None:
    tenant = _tenant()
    services = build_fixture_tenant_services(
        config_root,
        tenant.tenant_id,
        telemetry=make_tenant_telemetry(tenant.tenant_id),
    )

    def mismatched_state(context, dependencies):
        state = build_in_memory_session_state(
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
        build_application_session(
            tenant,
            _settings(),
            _models(),
            services,
            deployment_id=_TEST_DEPLOYMENT_ID,
            routing_factory=_routing_factory,
            session_state_factory=mismatched_state,
        )


def test_application_rejects_split_brain_lifecycle_stores(config_root: Path) -> None:
    tenant = _tenant()
    services = build_fixture_tenant_services(
        config_root,
        tenant.tenant_id,
        telemetry=make_tenant_telemetry(tenant.tenant_id),
    )

    def mismatched_state(context, dependencies):
        state = build_in_memory_session_state(context, dependencies)
        return replace(state, cart_store=CartStore())

    with pytest.raises(ValueError, match=r"lifecycle.*cart"):
        build_application_session(
            tenant,
            _settings(),
            _models(),
            services,
            deployment_id=_TEST_DEPLOYMENT_ID,
            routing_factory=_routing_factory,
            session_state_factory=mismatched_state,
        )


def test_application_rejects_split_brain_lifecycle_telemetry(config_root: Path) -> None:
    tenant = _tenant()
    services = build_fixture_tenant_services(
        config_root,
        tenant.tenant_id,
        telemetry=make_tenant_telemetry(tenant.tenant_id),
    )

    def mismatched_state(context, dependencies):
        state = build_in_memory_session_state(context, dependencies)
        other = make_tenant_telemetry(context.tenant_id).bind_session(state.session_id)
        return replace(
            state,
            caller_context=replace(state.caller_context, telemetry=other.operational),
        )

    with pytest.raises(ValueError, match=r"lifecycle.*telemetry"):
        build_application_session(
            tenant,
            _settings(),
            _models(),
            services,
            deployment_id=_TEST_DEPLOYMENT_ID,
            routing_factory=_routing_factory,
            session_state_factory=mismatched_state,
        )


def test_application_rejects_session_telemetry_from_another_sink(config_root: Path) -> None:
    tenant = _tenant()
    services = build_fixture_tenant_services(
        config_root,
        tenant.tenant_id,
        telemetry=make_tenant_telemetry(tenant.tenant_id),
    )

    def mismatched_state(context, dependencies):
        state = build_in_memory_session_state(context, dependencies)
        other = make_tenant_telemetry(context.tenant_id).bind_session(state.session_id)
        return replace(state, telemetry=other)

    with pytest.raises(ValueError, match="tenant telemetry service"):
        build_application_session(
            tenant,
            _settings(),
            _models(),
            services,
            deployment_id=_TEST_DEPLOYMENT_ID,
            routing_factory=_routing_factory,
            session_state_factory=mismatched_state,
        )


def test_application_rejects_swapped_telemetry_purposes(config_root: Path) -> None:
    tenant = _tenant()
    services = build_fixture_tenant_services(
        config_root,
        tenant.tenant_id,
        telemetry=make_tenant_telemetry(tenant.tenant_id),
    )

    def mismatched_state(context, dependencies):
        state = build_in_memory_session_state(context, dependencies)
        return replace(
            state,
            telemetry=SessionTelemetry(
                operational=state.telemetry.routing_evidence,
                routing_evidence=state.telemetry.operational,
            ),
        )

    with pytest.raises(ValueError, match="telemetry purpose"):
        build_application_session(
            tenant,
            _settings(),
            _models(),
            services,
            deployment_id=_TEST_DEPLOYMENT_ID,
            routing_factory=_routing_factory,
            session_state_factory=mismatched_state,
        )
