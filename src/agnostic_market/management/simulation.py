"""Isolated text simulation over one immutable merchant publication."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from agnostic_market.agents.engine import GraphTurnLatencyMeasurement
from agnostic_market.agents.telemetry import InMemoryTelemetrySink, TelemetryRecord, TenantTelemetry
from agnostic_market.application import (
    ApplicationModels,
    ApplicationSession,
    ApplicationSessionState,
    ApplicationSettings,
    RoutingFactory,
    TenantServices,
    build_application_session,
    build_fixture_tenant_services_from_fixtures,
    build_in_memory_session_state,
)
from agnostic_market.commerce.receipts import CommerceReceiptCounts
from agnostic_market.config.loader import config_version
from agnostic_market.dtos.config import MerchantConfig
from agnostic_market.dtos.events import CommittedTurn, TurnEvent, TurnFacts
from agnostic_market.dtos.money import UsdAmount
from agnostic_market.dtos.session import AuthorityIdentifier
from agnostic_market.management.contracts import (
    PublishedMerchantVersion,
    published_merchant_runtime_version,
)
from agnostic_market.management.service import (
    MerchantManagementNotFoundError,
    MerchantManagementService,
)
from agnostic_market.tenancy.context import TenantContext

_CONTRACT = ConfigDict(extra="forbid", frozen=True, strict=True)
_AUTHORITY = TypeAdapter(AuthorityIdentifier)


class MerchantSimulationError(RuntimeError):
    """Base error for the isolated development simulation boundary."""


class MerchantSimulationNotFoundError(MerchantSimulationError):
    """The selected publication or simulation does not exist."""


class MerchantSimulationConflictError(MerchantSimulationError):
    """The requested simulation identity is already active."""


class MerchantSimulationReplayConflictError(MerchantSimulationError):
    """A turn request identity was reused with different input."""


class SimulationSessionStatus(BaseModel):
    model_config = _CONTRACT

    schema_version: Literal[1] = 1
    tenant_id: AuthorityIdentifier
    simulation_id: AuthorityIdentifier
    publication_version_id: AuthorityIdentifier
    config_version: str = Field(min_length=1)
    turn_count: int = Field(ge=0)


class SimulationTurnLatency(BaseModel):
    model_config = _CONTRACT

    total_seconds: float = Field(ge=0)
    time_to_first_model_seconds: float | None = Field(default=None, ge=0)
    tool_count: int = Field(ge=0)
    tool_to_next_model_seconds: float | None = Field(default=None, ge=0)

    @classmethod
    def from_measurement(cls, measurement: GraphTurnLatencyMeasurement) -> SimulationTurnLatency:
        return cls(
            total_seconds=measurement.total_seconds,
            time_to_first_model_seconds=measurement.time_to_first_model_seconds,
            tool_count=measurement.tool_count,
            tool_to_next_model_seconds=measurement.tool_to_next_model_seconds,
        )


class SimulationCartLine(BaseModel):
    """One value-complete cart line exposed by the management projection."""

    model_config = _CONTRACT

    sku: str = Field(min_length=1)
    name: str = Field(min_length=1)
    price_usd: UsdAmount
    quantity: int = Field(ge=1)
    line_total: UsdAmount


class SimulationStateProjection(BaseModel):
    """Bounded, non-authority state suitable for a development workbench."""

    model_config = _CONTRACT

    schema_version: Literal[1] = 1
    tenant_id: AuthorityIdentifier
    simulation_id: AuthorityIdentifier
    publication_version_id: AuthorityIdentifier
    config_version: str = Field(min_length=1)
    turn_count: int = Field(ge=0)
    session_revision: int = Field(ge=0)
    cart_lines: tuple[SimulationCartLine, ...]
    cart_total_usd: UsdAmount
    recent_order_count: int = Field(ge=0)
    recent_order_context_complete: bool
    guest_order_count: int = Field(ge=0)
    has_discardable_state: bool
    committed_receipts: CommerceReceiptCounts


class SimulationTurnResult(BaseModel):
    model_config = _CONTRACT

    schema_version: Literal[1] = 1
    tenant_id: AuthorityIdentifier
    simulation_id: AuthorityIdentifier
    publication_version_id: AuthorityIdentifier
    config_version: str = Field(min_length=1)
    request_id: AuthorityIdentifier
    replayed: bool
    turn_number: int = Field(ge=1)
    session_revision: int = Field(ge=0)
    events: tuple[TurnEvent, ...]
    operational_records: tuple[TelemetryRecord, ...]
    routing_records: tuple[TelemetryRecord, ...]
    latency: tuple[SimulationTurnLatency, ...]
    state: SimulationStateProjection


type SimulatorModelsFactory = Callable[[MerchantConfig], ApplicationModels]
type SimulatorRoutingFactory = Callable[[MerchantConfig], RoutingFactory]


def build_published_tenant_context(version: PublishedMerchantVersion) -> TenantContext:
    """Pin a development simulation to one immutable publication."""

    return TenantContext(
        tenant_id=version.tenant_id,
        config_version=published_merchant_runtime_version(version),
        policy=version.config.policies.to_policy_context(),
    )


def build_published_fixture_tenant_services(
    version: PublishedMerchantVersion,
    tenant: TenantContext,
    *,
    telemetry: TenantTelemetry,
) -> TenantServices:
    """Compose simulation adapters from the publication's exact fixtures."""

    if (
        tenant.tenant_id != version.tenant_id
        or tenant.config_version != published_merchant_runtime_version(version)
        or tenant.policy != version.config.policies.to_policy_context()
    ):
        raise ValueError("tenant context does not match the published merchant version")
    fixtures = version.fixtures
    return build_fixture_tenant_services_from_fixtures(
        tenant,
        catalog_fixture=fixtures.catalog,
        orders_fixture=fixtures.orders,
        customers_fixture=fixtures.customers,
        profile_fixture=fixtures.profiles,
        payment_fixture=fixtures.payment_instruments,
        verification_fixture=fixtures.verification,
        telemetry=telemetry,
        checkpointer=None,
    )


@dataclass(slots=True)
class _OpenSimulation:
    version: PublishedMerchantVersion
    application: ApplicationSession
    operational: InMemoryTelemetrySink
    routing: InMemoryTelemetrySink
    latency: list[GraphTurnLatencyMeasurement]
    turn_count: int = 0
    replayed_turns: dict[str, tuple[str, SimulationTurnResult]] = field(default_factory=dict)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class PublishedMerchantSimulator:
    """Own process-local engine sessions without activating production adapters."""

    def __init__(
        self,
        management: MerchantManagementService,
        *,
        models_factory: SimulatorModelsFactory,
        routing_factory: SimulatorRoutingFactory,
        deployment_id: str = "development-text-simulator",
        max_turns: int = 100,
    ) -> None:
        if max_turns <= 0:
            raise ValueError("simulation turn limit must be positive")
        self._management = management
        self._models_factory = models_factory
        self._routing_factory = routing_factory
        self._deployment_id = _AUTHORITY.validate_python(deployment_id, strict=True)
        self._max_turns = max_turns
        self._sessions: dict[tuple[str, str], _OpenSimulation] = {}
        self._sessions_lock = asyncio.Lock()
        self._closed = False

    @staticmethod
    def _key(tenant_id: str, simulation_id: str) -> tuple[str, str]:
        return (
            _AUTHORITY.validate_python(tenant_id, strict=True),
            _AUTHORITY.validate_python(simulation_id, strict=True),
        )

    async def _load_version(
        self,
        tenant_id: str,
        version_id: str | None,
    ) -> PublishedMerchantVersion:
        if version_id is not None:
            version_id = _AUTHORITY.validate_python(version_id, strict=True)
        try:
            if version_id is None:
                return await asyncio.to_thread(self._management.get_active_version, tenant_id)
            return await asyncio.to_thread(self._management.get_version, tenant_id, version_id)
        except MerchantManagementNotFoundError as exc:
            raise MerchantSimulationNotFoundError(
                "selected merchant publication does not exist"
            ) from exc

    async def _build(
        self,
        version: PublishedMerchantVersion,
        simulation_id: str,
    ) -> _OpenSimulation:
        tenant = build_published_tenant_context(version)
        operational = InMemoryTelemetrySink()
        routing = InMemoryTelemetrySink()
        telemetry = TenantTelemetry(tenant.tenant_id, operational, routing)
        services = build_published_fixture_tenant_services(
            version,
            tenant,
            telemetry=telemetry,
        )
        latency: list[GraphTurnLatencyMeasurement] = []

        async def isolated_state(
            tenant_context: TenantContext,
            tenant_services: TenantServices,
        ) -> ApplicationSessionState:
            return await build_in_memory_session_state(
                tenant_context,
                tenant_services,
                session_id=simulation_id,
                thread_id=f"simulation:{simulation_id}",
            )

        application = await build_application_session(
            tenant,
            ApplicationSettings.from_merchant_config(version.config),
            self._models_factory(version.config),
            services,
            deployment_id=self._deployment_id,
            routing_factory=self._routing_factory(version.config),
            session_state_factory=isolated_state,
            graph_turn_latency_observer=latency.append,
        )
        return _OpenSimulation(
            version=version,
            application=application,
            operational=operational,
            routing=routing,
            latency=latency,
        )

    @staticmethod
    def _status(simulation_id: str, opened: _OpenSimulation) -> SimulationSessionStatus:
        return SimulationSessionStatus(
            tenant_id=opened.version.tenant_id,
            simulation_id=simulation_id,
            publication_version_id=opened.version.version_id,
            config_version=opened.application.tenant.config_version,
            turn_count=opened.turn_count,
        )

    @staticmethod
    def _state(simulation_id: str, opened: _OpenSimulation) -> SimulationStateProjection:
        caller = opened.application.state.caller_context
        recent = caller.recent_orders.snapshot()
        return SimulationStateProjection(
            tenant_id=opened.version.tenant_id,
            simulation_id=simulation_id,
            publication_version_id=opened.version.version_id,
            config_version=opened.application.tenant.config_version,
            turn_count=opened.turn_count,
            session_revision=caller.session_revision,
            cart_lines=tuple(
                SimulationCartLine(
                    sku=line.sku,
                    name=line.name,
                    price_usd=line.price_usd,
                    quantity=line.quantity,
                    line_total=line.line_total,
                )
                for line in caller.cart_store.view()
            ),
            cart_total_usd=caller.cart_store.cart_total(),
            recent_order_count=len(recent.order_refs),
            recent_order_context_complete=recent.complete,
            guest_order_count=len(caller.guest_orders.order_refs),
            has_discardable_state=caller.has_discardable_state(),
            committed_receipts=CommerceReceiptCounts(
                cart=caller.cart_store.receipt_counts(),
                orders=opened.application.services.order_store.receipt_counts(),
                profiles=opened.application.services.profile_store.receipt_counts(),
            ),
        )

    async def start(
        self,
        tenant_id: str,
        simulation_id: str,
        *,
        version_id: str | None = None,
    ) -> SimulationSessionStatus:
        key = self._key(tenant_id, simulation_id)
        tenant_id, simulation_id = key
        async with self._sessions_lock:
            if self._closed:
                raise MerchantSimulationConflictError("simulator is closed")
            if key in self._sessions:
                raise MerchantSimulationConflictError("simulation is already active")
        version = await self._load_version(tenant_id, version_id)
        opened = await self._build(version, simulation_id)
        async with self._sessions_lock:
            if self._closed or key in self._sessions:
                await opened.application.state.caller_context.aclose_session()
                reason = "simulator is closed" if self._closed else "simulation is already active"
                raise MerchantSimulationConflictError(reason)
            self._sessions[key] = opened
        return self._status(simulation_id, opened)

    async def status(self, tenant_id: str, simulation_id: str) -> SimulationSessionStatus:
        tenant_id, simulation_id = self._key(tenant_id, simulation_id)
        opened = await self._require_open(tenant_id, simulation_id)
        async with opened.lock:
            await self._require_current(tenant_id, simulation_id, opened)
            return self._status(simulation_id, opened)

    async def inspect_state(
        self,
        tenant_id: str,
        simulation_id: str,
    ) -> SimulationStateProjection:
        tenant_id, simulation_id = self._key(tenant_id, simulation_id)
        opened = await self._require_open(tenant_id, simulation_id)
        async with opened.lock:
            await self._require_current(tenant_id, simulation_id, opened)
            return self._state(simulation_id, opened)

    async def _require_open(self, tenant_id: str, simulation_id: str) -> _OpenSimulation:
        async with self._sessions_lock:
            opened = self._sessions.get((tenant_id, simulation_id))
        if opened is None:
            raise MerchantSimulationNotFoundError("simulation does not exist")
        return opened

    async def _require_current(
        self,
        tenant_id: str,
        simulation_id: str,
        opened: _OpenSimulation,
    ) -> None:
        async with self._sessions_lock:
            if self._sessions.get((tenant_id, simulation_id)) is not opened:
                raise MerchantSimulationNotFoundError("simulation does not exist")

    async def turn(
        self,
        tenant_id: str,
        simulation_id: str,
        *,
        request_id: str,
        text: str,
        readback_interrupted: bool = False,
    ) -> SimulationTurnResult:
        tenant_id, simulation_id = self._key(tenant_id, simulation_id)
        request_id = _AUTHORITY.validate_python(request_id, strict=True)
        opened = await self._require_open(tenant_id, simulation_id)
        async with opened.lock:
            await self._require_current(tenant_id, simulation_id, opened)
            turn = CommittedTurn(text=text, message_id=request_id)
            facts = TurnFacts(readback_interrupted=readback_interrupted)
            fingerprint = config_version(
                {
                    "turn": turn.model_dump(mode="json"),
                    "facts": facts.model_dump(mode="json"),
                }
            )
            replay = opened.replayed_turns.get(request_id)
            if replay is not None:
                expected_fingerprint, result = replay
                if expected_fingerprint != fingerprint:
                    raise MerchantSimulationReplayConflictError(
                        "simulation request id was reused with different input"
                    )
                return result.model_copy(update={"replayed": True})
            if opened.turn_count >= self._max_turns:
                raise MerchantSimulationConflictError("simulation turn limit reached")

            operational_mark = opened.operational.mark()
            routing_mark = opened.routing.mark()
            latency_mark = len(opened.latency)
            events = tuple(
                [
                    event
                    async for event in opened.application.engine.stream_turn(
                        turn,
                        facts,
                    )
                ]
            )
            opened.turn_count += 1
            result = SimulationTurnResult(
                tenant_id=tenant_id,
                simulation_id=simulation_id,
                publication_version_id=opened.version.version_id,
                config_version=opened.application.tenant.config_version,
                request_id=request_id,
                replayed=False,
                turn_number=opened.turn_count,
                session_revision=opened.application.state.caller_context.session_revision,
                events=events,
                operational_records=opened.operational.read_since(operational_mark).records,
                routing_records=opened.routing.read_since(routing_mark).records,
                latency=tuple(
                    SimulationTurnLatency.from_measurement(measurement)
                    for measurement in opened.latency[latency_mark:]
                ),
                state=self._state(simulation_id, opened),
            )
            opened.replayed_turns[request_id] = (fingerprint, result)
            return result

    async def reset(self, tenant_id: str, simulation_id: str) -> SimulationSessionStatus:
        key = self._key(tenant_id, simulation_id)
        tenant_id, simulation_id = key
        opened = await self._require_open(tenant_id, simulation_id)
        async with opened.lock:
            await self._require_current(tenant_id, simulation_id, opened)
            async with self._sessions_lock:
                self._sessions.pop(key)
            await opened.application.state.caller_context.aclose_session()
            replacement = await self._build(opened.version, simulation_id)
            async with self._sessions_lock:
                if self._closed or key in self._sessions:
                    await replacement.application.state.caller_context.aclose_session()
                    reason = (
                        "simulator is closed"
                        if self._closed
                        else "simulation identity was reused during reset"
                    )
                    raise MerchantSimulationConflictError(reason)
                self._sessions[key] = replacement
            return self._status(simulation_id, replacement)

    async def close(self, tenant_id: str, simulation_id: str) -> None:
        key = self._key(tenant_id, simulation_id)
        tenant_id, simulation_id = key
        opened = await self._require_open(tenant_id, simulation_id)
        async with opened.lock:
            await self._require_current(tenant_id, simulation_id, opened)
            async with self._sessions_lock:
                self._sessions.pop(key)
            await opened.application.state.caller_context.aclose_session()

    async def aclose(self) -> None:
        async with self._sessions_lock:
            self._closed = True
            sessions = tuple(self._sessions.values())
            self._sessions.clear()
        failures: list[Exception] = []
        for opened in sessions:
            async with opened.lock:
                try:
                    await opened.application.state.caller_context.aclose_session()
                except Exception as exc:
                    failures.append(exc)
        if failures:
            raise ExceptionGroup("one or more simulations failed to close", failures)
