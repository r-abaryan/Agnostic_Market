"""Concrete deployment-shaped probes for durable reasoning-graph latency certification.

These probes measure the reasoning graph's whole-turn span. They do not cover LiveKit
voice processing or TTS, so their evidence cannot authorize network startup; that
requires a separately produced voice-processing artifact.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

from agnostic_market.agents.engine import GraphTurnLatencyMeasurement
from agnostic_market.agents.routing_activation import build_qualified_semantic_router_factory
from agnostic_market.agents.telemetry import DisabledTelemetrySink, TenantTelemetry
from agnostic_market.application import (
    TenantServices,
    build_fixture_tenant_services,
    prepare_application_routing,
)
from agnostic_market.config.registry import ConfigRegistry
from agnostic_market.dtos.events import CommittedTurn, InterruptEvent, SpokenMessageEvent, TurnFacts
from agnostic_market.dtos.platform import PlatformRuntimeConfig
from agnostic_market.dtos.session import AdmittedSessionAuthority, TransportAuthority
from agnostic_market.durability.latency import (
    DurableLatencyMethodology,
    LatencyCartLine,
    LatencyJourney,
    LatencyJourneyContract,
    LatencyJourneyCorpus,
    LatencyMeasurementSurface,
    bind_latency_journey_contracts,
)
from agnostic_market.durability.platform_runtime import (
    DurablePlatformResources,
    DurableSessionResources,
)
from agnostic_market.durability.timing import InMemoryDurabilityTimingObserver
from agnostic_market.llm.gateway import LLMGateway, load_provider_credentials
from agnostic_market.llm.providers import (
    ConformanceRegistry,
    load_conformance_targets,
    require_llm_certification,
)
from agnostic_market.secrets.base import SecretResolver
from agnostic_market.tenancy.context import build_tenant_context
from agnostic_market.voice.pipeline import VoiceLoop, build_voice_loop


class DeploymentLatencyProbeError(RuntimeError):
    """A concrete probe failed to execute its frozen behavior contract."""


@dataclass(frozen=True, slots=True)
class _ProbeDependencies:
    config_root: Path
    platform_config: PlatformRuntimeConfig
    application_dsn: str
    deployment_id: str
    merchant_id: str
    secrets: SecretResolver
    contracts: dict[str, LatencyJourneyContract]


@dataclass(slots=True)
class _OpenProbeExecution:
    loop: VoiceLoop
    platform_resources: DurablePlatformResources
    services: TenantServices
    measurements: list[GraphTurnLatencyMeasurement]
    startup_elapsed_seconds: float

    async def aclose(self) -> None:
        await self.loop.aclose(release_job_resources=self.platform_resources.aclose)


@dataclass(slots=True)
class _TurnProbeExecution:
    execution: _OpenProbeExecution
    contract: LatencyJourneyContract
    sample_id: str

    async def run(self) -> float:
        before_orders = _placed_order_count(self.execution.services)
        events = [
            event
            async for event in self.execution.loop.engine.stream_turn(
                CommittedTurn(
                    text=self.contract.utterance,
                    message_id=f"{self.sample_id}-turn",
                ),
                TurnFacts(),
            )
        ]
        if len(self.execution.measurements) != 1:
            raise DeploymentLatencyProbeError(
                "journey did not produce exactly one graph-span measurement"
            )
        _require_expected_event(self.contract, events)
        _require_expected_cart(self.contract, self.execution.loop)
        if _placed_order_count(self.execution.services) != before_orders:
            raise DeploymentLatencyProbeError("latency journey committed an order")
        return self.execution.measurements[0].total_seconds

    async def aclose(self) -> None:
        await self.execution.aclose()


class DeploymentLatencyProbeFactory:
    """Build fresh production-shaped resources for every frozen latency sample."""

    def __init__(
        self,
        *,
        config_root: Path,
        platform_config: PlatformRuntimeConfig,
        application_dsn: str,
        deployment_id: str,
        methodology: DurableLatencyMethodology,
        corpus: LatencyJourneyCorpus,
        secrets: SecretResolver,
    ) -> None:
        if not deployment_id.strip():
            raise ValueError("deployment latency probe requires a deployment id")
        if methodology.measurement_surface is not LatencyMeasurementSurface.REASONING_GRAPH:
            raise ValueError("the concrete latency probe measures only the reasoning graph")
        self._dependencies = _ProbeDependencies(
            config_root=config_root,
            platform_config=platform_config,
            application_dsn=application_dsn,
            deployment_id=deployment_id,
            merchant_id=corpus.merchant_id,
            secrets=secrets,
            contracts=bind_latency_journey_contracts(methodology, corpus),
        )
        self._run_id = uuid.uuid4().hex

    async def startup_probe(
        self,
        sample_id: str,
        observer: InMemoryDurabilityTimingObserver,
    ) -> _OpenProbeExecution:
        return await self._open_execution(sample_id, observer)

    async def turn_probe(
        self,
        journey: LatencyJourney,
        sample_id: str,
        observer: InMemoryDurabilityTimingObserver,
    ) -> _TurnProbeExecution:
        try:
            contract = self._dependencies.contracts[journey.journey_id]
        except KeyError as exc:
            raise DeploymentLatencyProbeError("latency journey is not frozen") from exc
        execution = await self._open_execution(sample_id, observer)
        try:
            await _prepare_journey(contract, execution)
        except BaseException as failure:
            await _close_after_failure(execution, failure)
        return _TurnProbeExecution(execution, contract, sample_id)

    async def _open_execution(
        self,
        sample_id: str,
        observer: InMemoryDurabilityTimingObserver,
    ) -> _OpenProbeExecution:
        dependencies = self._dependencies
        started = time.perf_counter()
        resources: DurablePlatformResources | None = None
        durable_session: DurableSessionResources | None = None
        loop: VoiceLoop | None = None
        try:
            registry = ConfigRegistry(dependencies.config_root).load()
            resolved = registry.get(dependencies.merchant_id)
            tenant = build_tenant_context(registry, dependencies.merchant_id)
            credentials = load_provider_credentials(
                dependencies.config_root / "base" / "providers.yaml"
            )
            targets = load_conformance_targets(
                dependencies.config_root / "conformance" / "targets.yaml"
            )
            conformance = ConformanceRegistry(
                dependencies.config_root / "conformance" / "reports.json",
                max_report_age_days=targets.max_report_age_days,
            )
            require_llm_certification(resolved.config, conformance)
            gateway = LLMGateway(credentials, dependencies.secrets)
            routing = prepare_application_routing(
                build_qualified_semantic_router_factory(
                    dependencies.config_root,
                    selection=resolved.config.llm.routing,
                    credentials=credentials,
                    secrets=dependencies.secrets,
                    structured_output_method=gateway.structured_output_method(
                        resolved.config.llm.routing
                    ),
                    timeout_seconds=resolved.config.runtime.semantic_router_timeout_seconds,
                    input_max_chars=resolved.config.runtime.semantic_router_input_max_chars,
                    max_report_age_days=targets.max_report_age_days,
                )
            )
            resources = await DurablePlatformResources.open(
                dependencies.platform_config,
                dependencies.secrets,
                durability_timing=observer,
                application_dsn=dependencies.application_dsn,
            )
            authority = _sample_authority(self._run_id, sample_id)
            durable_session = await resources.acquire_fresh_session(
                tenant_id=tenant.tenant_id,
                admitted_authority=authority,
                deployment_id=dependencies.deployment_id,
                config_version=tenant.config_version,
            )
            telemetry = TenantTelemetry(
                tenant.tenant_id,
                DisabledTelemetrySink(),
                DisabledTelemetrySink(),
            )
            services = build_fixture_tenant_services(
                dependencies.config_root,
                tenant,
                telemetry=telemetry,
                checkpointer=durable_session.checkpointer,
            )
            measurements: list[GraphTurnLatencyMeasurement] = []
            loop = await build_voice_loop(
                tenant,
                resolved,
                credentials,
                dependencies.secrets,
                deployment_id=dependencies.deployment_id,
                tenant_services=services,
                routing_recognizer_factory=routing,
                durable_session=durable_session,
                graph_turn_latency_observer=measurements.append,
            )
            return _OpenProbeExecution(
                loop=loop,
                platform_resources=resources,
                services=services,
                measurements=measurements,
                startup_elapsed_seconds=time.perf_counter() - started,
            )
        except BaseException as failure:
            await _cleanup_partial_execution(
                resources=resources,
                durable_session=durable_session,
                loop=loop,
                failure=failure,
            )


def _sample_authority(run_id: str, sample_id: str) -> AdmittedSessionAuthority:
    prefix = f"latency-{run_id}-{sample_id}"
    return AdmittedSessionAuthority(
        logical_session_id=prefix,
        transport=TransportAuthority(
            provider="livekit",
            room_id=f"{prefix}-room",
            assignment_id=f"{prefix}-assignment",
            worker_id=f"{prefix}-worker",
        ),
    )


async def _prepare_journey(
    contract: LatencyJourneyContract,
    execution: _OpenProbeExecution,
) -> None:
    before_orders = _placed_order_count(execution.services)
    for index, utterance in enumerate(contract.setup_turns, start=1):
        async for _event in execution.loop.engine.stream_turn(
            CommittedTurn(
                text=utterance,
                message_id=f"setup-{contract.journey_id}-{index}",
            ),
            TurnFacts(),
        ):
            pass
    _require_cart_lines(contract.initial_cart, execution.loop, phase="setup")
    if _placed_order_count(execution.services) != before_orders:
        raise DeploymentLatencyProbeError("latency journey setup committed an order")
    execution.measurements.clear()


def _require_expected_cart(
    contract: LatencyJourneyContract,
    loop: VoiceLoop,
) -> None:
    _require_cart_lines(contract.expected_cart, loop, phase="measured turn")


def _require_cart_lines(
    expected: tuple[LatencyCartLine, ...],
    loop: VoiceLoop,
    *,
    phase: str,
) -> None:
    actual = tuple(
        LatencyCartLine(sku=line.sku, quantity=line.quantity)
        for line in loop.application.state.cart_store.snapshot()
    )
    if actual != expected:
        raise DeploymentLatencyProbeError(f"journey cart state is invalid after {phase}")


def _placed_order_count(services: TenantServices) -> int:
    count = getattr(services.order_store, "placed_count", None)
    if not isinstance(count, int):
        raise DeploymentLatencyProbeError("latency probe requires the fixture order adapter")
    return count


def _require_expected_event(contract: LatencyJourneyContract, events: list[object]) -> None:
    if contract.expected_event_kind == "interrupt":
        expected = [event for event in events if isinstance(event, InterruptEvent)]
        unexpected = [event for event in events if not isinstance(event, InterruptEvent)]
        text = expected[0].prompt if len(expected) == 1 else ""
    else:
        expected = [event for event in events if isinstance(event, SpokenMessageEvent)]
        unexpected = [event for event in events if not isinstance(event, SpokenMessageEvent)]
        text = expected[0].text if len(expected) == 1 else ""
        if len(expected) == 1 and expected[0].node != contract.expected_event_node:
            raise DeploymentLatencyProbeError(
                "journey produced spoken message from "
                f"{expected[0].node!r}; expected {contract.expected_event_node!r}"
            )
    if len(expected) != 1 or unexpected:
        raise DeploymentLatencyProbeError("journey produced the wrong caller-event shape")
    normalized = text.casefold()
    if any(
        fragment.casefold() not in normalized for fragment in contract.expected_event_text_contains
    ):
        raise DeploymentLatencyProbeError("journey output did not satisfy its frozen contract")


async def _close_after_failure(
    execution: _OpenProbeExecution,
    failure: BaseException,
) -> NoReturn:
    try:
        await execution.aclose()
    except BaseException as cleanup_failure:
        raise failure from cleanup_failure
    raise failure


async def _cleanup_partial_execution(
    *,
    resources: DurablePlatformResources | None,
    durable_session: DurableSessionResources | None,
    loop: VoiceLoop | None,
    failure: BaseException,
) -> NoReturn:
    try:
        if loop is not None:
            await loop.aclose(release_job_resources=resources.aclose if resources else None)
        else:
            if durable_session is not None:
                await durable_session.closer.begin_close()
                await durable_session.closer.finalize_close()
            if resources is not None:
                await resources.aclose()
    except BaseException as cleanup_failure:
        raise failure from cleanup_failure
    raise failure
