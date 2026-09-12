"""Fail-closed durable-path latency evidence contracts."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from agnostic_market.durability.latency import (
    DurableLatencyActivationError,
    DurableLatencyCertificationRun,
    DurableLatencyMethodology,
    DurableLatencyReport,
    LatencyAbortStage,
    LatencyCartLine,
    LatencyCertificationAbort,
    LatencyCertificationOutcome,
    LatencyEnvironment,
    LatencyJourney,
    LatencyJourneyContract,
    LatencyJourneyCorpus,
    LatencyMeasurementSurface,
    LatencyObservation,
    LatencyObservationOutcome,
    LatencyPhase,
    LatencyTier,
    ThermalState,
    bind_latency_journey_contracts,
    build_latency_report,
    latency_journey_contract_fingerprint,
    latency_journey_corpus_fingerprint,
    load_latency_certification_run,
    load_latency_journey_corpus,
    load_latency_methodology,
    methodology_fingerprint,
    require_deployment_latency_evidence,
    require_deployment_latency_report,
    run_latency_certification,
    write_latency_certification_run,
)
from agnostic_market.durability.timing import (
    DurabilityOperation,
    DurabilityTimingOutcome,
    DurabilityTimingSample,
    observe_duration,
)


def _corpus() -> LatencyJourneyCorpus:
    return LatencyJourneyCorpus(
        schema_version="1",
        merchant_id="acme_store",
        journeys=(
            LatencyJourneyContract(
                journey_id="simple-cart-read",
                utterance="what is in my cart?",
                initial_cart=(),
                expected_cart=(),
                expected_event_kind="spoken_message",
                expected_event_node="cart_view_render",
                expected_event_text_contains=("empty",),
            ),
            LatencyJourneyContract(
                journey_id="commerce-cart-confirmation",
                utterance="add one trail running shoes to my cart",
                initial_cart=(),
                expected_cart=(),
                expected_event_kind="interrupt",
                expected_event_text_contains=("trail running shoes",),
            ),
            LatencyJourneyContract(
                journey_id="checkout-placement-readback",
                setup_turns=("add one waterproof rain jacket to my cart", "yes"),
                utterance="place my order",
                initial_cart=(LatencyCartLine(sku="SKU-BLU-07", quantity=1),),
                expected_cart=(LatencyCartLine(sku="SKU-BLU-07", quantity=1),),
                expected_event_kind="interrupt",
                expected_event_text_contains=("$129.00",),
            ),
        ),
    )


def _methodology() -> DurableLatencyMethodology:
    corpus = _corpus()
    return DurableLatencyMethodology(
        schema_version="3",
        environment=LatencyEnvironment.CONTROLLED,
        backend_location="deployment-region-a",
        journey_corpus_fingerprint=latency_journey_corpus_fingerprint(corpus),
        runtime_contract_fingerprint="a" * 64,
        measurement_surface=LatencyMeasurementSurface.REASONING_GRAPH,
        startup_treatment="fresh_job_resources",
        concurrency=2,
        warmup_runs=2,
        startup_samples=20,
        startup_p95_limit_seconds=1.5,
        required_startup_operations=(DurabilityOperation.CHECKPOINT_READ,),
        samples_per_journey=20,
        sample_timeout_seconds=10.0,
        reported_statistics=("p50", "p95", "maximum"),
        journeys=(
            LatencyJourney(
                journey_id="simple-cart-read",
                required_operations=(DurabilityOperation.CHECKPOINT_READ,),
                journey_contract_fingerprint=latency_journey_contract_fingerprint(
                    corpus.journeys[0], merchant_id=corpus.merchant_id
                ),
                tier=LatencyTier.SIMPLE,
            ),
            LatencyJourney(
                journey_id="commerce-cart-confirmation",
                required_operations=(DurabilityOperation.CHECKPOINT_READ,),
                journey_contract_fingerprint=latency_journey_contract_fingerprint(
                    corpus.journeys[1], merchant_id=corpus.merchant_id
                ),
                tier=LatencyTier.NON_CHECKOUT_COMMERCE,
            ),
            LatencyJourney(
                journey_id="checkout-placement-readback",
                required_operations=(DurabilityOperation.CHECKPOINT_READ,),
                journey_contract_fingerprint=latency_journey_contract_fingerprint(
                    corpus.journeys[2], merchant_id=corpus.merchant_id
                ),
                tier=LatencyTier.CHECKOUT,
            ),
        ),
    )


def _completed_report(run: DurableLatencyCertificationRun) -> DurableLatencyReport:
    assert run.outcome is LatencyCertificationOutcome.COMPLETED
    assert run.abort is None
    assert run.report is not None
    return run.report


def _activation_requirements() -> dict[str, object]:
    return {
        "expected_journey_corpus": _corpus(),
        "expected_runtime_contract_fingerprint": "a" * 64,
        "required_measurement_surface": LatencyMeasurementSurface.REASONING_GRAPH,
    }


def test_journey_corpus_is_exactly_bound_to_the_preregistered_methodology() -> None:
    corpus = _corpus()
    contracts = corpus.journeys
    methodology = _methodology()

    assert bind_latency_journey_contracts(methodology, corpus) == {
        contract.journey_id: contract for contract in contracts
    }

    changed = contracts[0].model_copy(update={"utterance": "show the cart"})
    with pytest.raises(DurableLatencyActivationError, match="fingerprint mismatch"):
        bind_latency_journey_contracts(
            methodology,
            corpus.model_copy(update={"journeys": (changed, *contracts[1:])}),
        )


def test_repository_journey_corpus_has_the_published_contract_fingerprints() -> None:
    corpus = load_latency_journey_corpus(
        Path(__file__).parents[1] / "config" / "eval" / "durable_latency_journeys.yaml"
    )

    assert {
        contract.journey_id: latency_journey_contract_fingerprint(
            contract,
            merchant_id=corpus.merchant_id,
        )
        for contract in corpus.journeys
    } == {
        "simple-cart-read": "a9304596004c7bd0515380f0cbab5ef5caf62bbf8b6ef607328304a1556bbe67",
        "commerce-cart-confirmation": (
            "d2da9a766025b7e196688531453fc0babe455cbe39d0227eb26b0b786af97083"
        ),
        "checkout-placement-readback": (
            "0c57f13de9e35722c199b81e204a8aee2f518a2acec3c9e181f3cf1e5bb458d3"
        ),
    }


def test_certification_run_writer_is_atomic_and_does_not_overwrite(tmp_path: Path) -> None:
    methodology = _methodology()
    report = build_latency_report(
        methodology,
        _observations(methodology),
        run_at=datetime(2026, 9, 9, tzinfo=UTC),
        deployment_id="deployment-a",
    )
    run = DurableLatencyCertificationRun(
        run_at=report.run_at,
        deployment_id=report.deployment_id,
        methodology_fingerprint=report.methodology_fingerprint,
        methodology=methodology,
        outcome=LatencyCertificationOutcome.COMPLETED,
        report=report,
    )
    path = tmp_path / "latency-run.json"

    write_latency_certification_run(path, run)

    assert load_latency_certification_run(path) == run
    with pytest.raises(FileExistsError):
        write_latency_certification_run(path, run)


@pytest.mark.parametrize("phase", (LatencyPhase.STARTUP, LatencyPhase.TURN))
def test_missing_required_operation_fails_certification(phase: LatencyPhase) -> None:
    methodology = _methodology()
    if phase is LatencyPhase.STARTUP:
        methodology = methodology.model_copy(
            update={
                "required_startup_operations": (DurabilityOperation.POOL_OPEN,),
            }
        )
    else:
        journey = methodology.journeys[0].model_copy(
            update={
                "required_operations": (DurabilityOperation.CHECKPOINT_WRITE,),
            }
        )
        methodology = methodology.model_copy(
            update={
                "journeys": (journey, *methodology.journeys[1:]),
            }
        )
    report = build_latency_report(
        methodology,
        _observations(methodology),
        run_at=datetime(2026, 9, 9, tzinfo=UTC),
        deployment_id="deployment-a",
    )
    assert not report.gate.passed
    assert any("missing required" in failure for failure in report.gate.failures)


def test_required_coverage_is_mandatory_unique_and_fingerprinted() -> None:
    methodology = _methodology()
    for operations in ((), (DurabilityOperation.POOL_OPEN,) * 2):
        with pytest.raises(ValidationError):
            DurableLatencyMethodology.model_validate(
                {
                    **methodology.model_dump(),
                    "required_startup_operations": operations,
                }
            )
        with pytest.raises(ValidationError):
            LatencyJourney.model_validate(
                {
                    **methodology.journeys[0].model_dump(),
                    "required_operations": operations,
                }
            )
    changed = methodology.model_copy(
        update={
            "required_startup_operations": (DurabilityOperation.POOL_OPEN,),
        }
    )
    assert methodology_fingerprint(changed) != methodology_fingerprint(methodology)
    changed = methodology.model_copy(
        update={
            "journeys": (
                methodology.journeys[0].model_copy(
                    update={
                        "required_operations": (DurabilityOperation.CHECKPOINT_WRITE,),
                    }
                ),
                *methodology.journeys[1:],
            )
        }
    )
    assert methodology_fingerprint(changed) != methodology_fingerprint(methodology)
    with pytest.raises(ValidationError):
        DurableLatencyMethodology.model_validate(
            {**methodology.model_dump(), "schema_version": "2"}
        )
    missing_treatment = methodology.model_dump()
    missing_treatment.pop("startup_treatment")
    with pytest.raises(ValidationError):
        DurableLatencyMethodology.model_validate(missing_treatment)


def _observations(
    methodology: DurableLatencyMethodology,
) -> tuple[LatencyObservation, ...]:
    component = DurabilityTimingSample(
        operation=DurabilityOperation.CHECKPOINT_READ,
        elapsed_seconds=0.01,
        outcome=DurabilityTimingOutcome.SUCCESS,
    )
    startup = tuple(
        LatencyObservation.from_timing(
            sample_id=f"startup-{index}",
            phase=LatencyPhase.STARTUP,
            thermal_state=ThermalState.COLD,
            elapsed_seconds=0.5,
            component_samples=(component,),
        )
        for index in range(methodology.startup_samples)
    )
    turns = tuple(
        LatencyObservation.from_timing(
            sample_id=f"{journey.journey_id}-{index}",
            phase=LatencyPhase.TURN,
            thermal_state=ThermalState.WARM,
            journey_id=journey.journey_id,
            tier=journey.tier,
            elapsed_seconds={
                LatencyTier.SIMPLE: 0.8,
                LatencyTier.NON_CHECKOUT_COMMERCE: 1.5,
                LatencyTier.CHECKOUT: 2.5,
            }[journey.tier],
            component_samples=(component,),
        )
        for journey in methodology.journeys
        for index in range(methodology.samples_per_journey)
    )
    return startup + turns


def test_latency_report_requires_exact_frozen_coverage_and_passes_within_limits() -> None:
    methodology = _methodology()

    report = build_latency_report(
        methodology,
        _observations(methodology),
        run_at=datetime(2026, 9, 9, tzinfo=UTC),
        deployment_id="deployment-a",
    )

    assert report.gate.passed is True
    assert report.gate.failures == ()
    assert report.methodology_fingerprint == methodology_fingerprint(methodology)
    assert report.startup.statistics.count == methodology.startup_samples
    assert [result.journey_id for result in report.journeys] == [
        journey.journey_id for journey in methodology.journeys
    ]
    assert report.journeys[0].statistics.p95_seconds == 0.8
    assert report.journeys[0].p95_limit_seconds == 1.0


def test_latency_report_fails_closed_for_missing_or_duplicate_samples() -> None:
    methodology = _methodology()
    observations = _observations(methodology)

    with pytest.raises(ValueError, match="exactly 20 startup samples"):
        build_latency_report(
            methodology,
            observations[1:],
            run_at=datetime(2026, 9, 9, tzinfo=UTC),
            deployment_id="deployment-a",
        )

    with pytest.raises(ValueError, match="sample ids must be unique"):
        build_latency_report(
            methodology,
            (*observations, observations[0]),
            run_at=datetime(2026, 9, 9, tzinfo=UTC),
            deployment_id="deployment-a",
        )


def test_latency_report_retains_failures_instead_of_moving_the_gate() -> None:
    methodology = _methodology()
    observations = list(_observations(methodology))
    slow_indices = [
        index
        for index, sample in enumerate(observations)
        if sample.journey_id == "simple-cart-read"
    ][-2:]
    for index in slow_indices:
        observations[index] = observations[index].model_copy(update={"elapsed_seconds": 1.01})

    report = build_latency_report(
        methodology,
        observations,
        run_at=datetime(2026, 9, 9, tzinfo=UTC),
        deployment_id="deployment-a",
    )

    assert report.gate.passed is False
    assert report.gate.failures == ("journey simple-cart-read p95 1.010000s exceeds 1.000000s",)
    assert report.journeys[0].statistics.p95_seconds == 1.01


def test_latency_report_retains_failed_component_work_as_a_gate_failure() -> None:
    methodology = _methodology()
    observations = list(_observations(methodology))
    failed_component = DurabilityTimingSample(
        operation=DurabilityOperation.REGISTRY_RESTORE,
        elapsed_seconds=0.02,
        outcome=DurabilityTimingOutcome.ERROR,
    )
    observations[0] = LatencyObservation.from_timing(
        sample_id=observations[0].sample_id,
        phase=LatencyPhase.STARTUP,
        thermal_state=ThermalState.COLD,
        elapsed_seconds=0.5,
        component_samples=(failed_component,),
    )

    report = build_latency_report(
        methodology,
        observations,
        run_at=datetime(2026, 9, 9, tzinfo=UTC),
        deployment_id="deployment-a",
    )

    assert report.gate.passed is False
    assert report.gate.failures == (
        "sample startup-0 contains non-success durable component registry.restore",
        "sample startup-0 missing required operations: checkpoint.read",
    )


def test_latency_methodology_rejects_movable_or_ambiguous_contracts() -> None:
    methodology = _methodology()

    with pytest.raises(ValidationError):
        DurableLatencyMethodology.model_validate(
            {
                **methodology.model_dump(),
                "reported_statistics": ("p50", "maximum"),
            }
        )
    with pytest.raises(ValidationError):
        DurableLatencyMethodology.model_validate(
            {
                **methodology.model_dump(),
                "journeys": (
                    *methodology.journeys,
                    LatencyJourney(
                        journey_id="simple-cart-read",
                        required_operations=(DurabilityOperation.CHECKPOINT_READ,),
                        journey_contract_fingerprint="4" * 64,
                        tier=LatencyTier.SIMPLE,
                    ),
                ),
            }
        )

    changed = methodology.model_copy(update={"backend_location": "deployment-region-b"})
    assert methodology_fingerprint(changed) != methodology_fingerprint(methodology)

    changed = methodology.model_copy(update={"sample_timeout_seconds": 11.0})
    assert methodology_fingerprint(changed) != methodology_fingerprint(methodology)

    changed_journey = methodology.journeys[0].model_copy(
        update={"journey_contract_fingerprint": "f" * 64}
    )
    changed = methodology.model_copy(
        update={"journeys": (changed_journey, *methodology.journeys[1:])}
    )
    assert methodology_fingerprint(changed) != methodology_fingerprint(methodology)


def test_only_an_exact_passing_deployment_report_can_authorize_activation() -> None:
    controlled = _methodology()
    controlled_report = build_latency_report(
        controlled,
        _observations(controlled),
        run_at=datetime(2026, 9, 9, tzinfo=UTC),
        deployment_id="deployment-a",
    )

    with pytest.raises(DurableLatencyActivationError, match="deployment-shaped"):
        require_deployment_latency_report(
            controlled_report,
            expected_methodology_fingerprint=methodology_fingerprint(controlled),
            expected_deployment_id="deployment-a",
        )

    deployment = controlled.model_copy(update={"environment": LatencyEnvironment.DEPLOYMENT})
    deployment_report = build_latency_report(
        deployment,
        _observations(deployment),
        run_at=datetime(2026, 9, 9, tzinfo=UTC),
        deployment_id="deployment-a",
    )
    require_deployment_latency_report(
        deployment_report,
        expected_methodology_fingerprint=methodology_fingerprint(deployment),
        expected_deployment_id="deployment-a",
    )

    with pytest.raises(DurableLatencyActivationError, match="runtime contract mismatch"):
        require_deployment_latency_report(
            deployment_report,
            expected_methodology_fingerprint="0" * 64,
            expected_deployment_id="deployment-b",
        )


def test_deployment_evidence_loader_binds_the_run_to_the_preregistered_methodology(
    tmp_path,
) -> None:
    methodology = _methodology().model_copy(update={"environment": LatencyEnvironment.DEPLOYMENT})
    report = build_latency_report(
        methodology,
        _observations(methodology),
        run_at=datetime(2026, 9, 9, tzinfo=UTC),
        deployment_id="deployment-a",
    )
    run = DurableLatencyCertificationRun(
        run_at=report.run_at,
        deployment_id=report.deployment_id,
        methodology_fingerprint=report.methodology_fingerprint,
        methodology=methodology,
        outcome=LatencyCertificationOutcome.COMPLETED,
        report=report,
    )
    methodology_path = tmp_path / "methodology.yaml"
    report_path = tmp_path / "report.json"
    methodology_path.write_text(methodology.model_dump_json(), encoding="utf-8")
    report_path.write_text(run.model_dump_json(), encoding="utf-8")

    assert load_latency_methodology(methodology_path) == methodology
    assert load_latency_certification_run(report_path) == run
    assert (
        require_deployment_latency_evidence(
            methodology_path,
            report_path,
            expected_deployment_id="deployment-a",
            **_activation_requirements(),
        )
        == report
    )

    changed_contract = _corpus().journeys[0].model_copy(update={"utterance": "show my cart"})
    changed_corpus = _corpus().model_copy(
        update={"journeys": (changed_contract, *_corpus().journeys[1:])}
    )
    with pytest.raises(DurableLatencyActivationError, match="corpus fingerprint mismatch"):
        require_deployment_latency_evidence(
            methodology_path,
            report_path,
            expected_deployment_id="deployment-a",
            expected_journey_corpus=changed_corpus,
            expected_runtime_contract_fingerprint="a" * 64,
            required_measurement_surface=LatencyMeasurementSurface.REASONING_GRAPH,
        )

    with pytest.raises(DurableLatencyActivationError, match="deployment runtime"):
        require_deployment_latency_evidence(
            methodology_path,
            report_path,
            expected_deployment_id="deployment-a",
            expected_journey_corpus=_corpus(),
            expected_runtime_contract_fingerprint="b" * 64,
            required_measurement_surface=LatencyMeasurementSurface.REASONING_GRAPH,
        )

    with pytest.raises(DurableLatencyActivationError, match="measurement surface"):
        require_deployment_latency_evidence(
            methodology_path,
            report_path,
            expected_deployment_id="deployment-a",
            expected_journey_corpus=_corpus(),
            expected_runtime_contract_fingerprint="a" * 64,
            required_measurement_surface=LatencyMeasurementSurface.VOICE_PROCESSING,
        )

    changed = methodology.model_copy(update={"backend_location": "deployment-region-b"})
    methodology_path.write_text(changed.model_dump_json(), encoding="utf-8")
    with pytest.raises(DurableLatencyActivationError, match="deployment methodology"):
        require_deployment_latency_evidence(
            methodology_path,
            report_path,
            expected_deployment_id="deployment-a",
            **_activation_requirements(),
        )


def test_deployment_evidence_loader_rejects_missing_or_aborted_artifacts(tmp_path) -> None:
    methodology = _methodology().model_copy(update={"environment": LatencyEnvironment.DEPLOYMENT})
    methodology_path = tmp_path / "methodology.yaml"
    methodology_path.write_text(methodology.model_dump_json(), encoding="utf-8")
    report_path = tmp_path / "report.json"

    with pytest.raises(DurableLatencyActivationError, match="report is missing or invalid"):
        require_deployment_latency_evidence(
            methodology_path,
            report_path,
            expected_deployment_id="deployment-a",
            **_activation_requirements(),
        )

    aborted = DurableLatencyCertificationRun(
        run_at=datetime(2026, 9, 9, tzinfo=UTC),
        deployment_id="deployment-a",
        methodology_fingerprint=methodology_fingerprint(methodology),
        methodology=methodology,
        outcome=LatencyCertificationOutcome.ABORTED,
        abort=LatencyCertificationAbort(
            stage=LatencyAbortStage.WARMUP_PREPARE,
            sample_id="warmup-turn-simple-cart-read-0001",
            journey_id="simple-cart-read",
            error_type="TimeoutError",
        ),
    )
    report_path.write_text(aborted.model_dump_json(), encoding="utf-8")
    with pytest.raises(DurableLatencyActivationError, match="deployment methodology"):
        require_deployment_latency_evidence(
            methodology_path,
            report_path,
            expected_deployment_id="deployment-a",
            **_activation_requirements(),
        )


def test_activation_rederives_the_gate_from_embedded_observations() -> None:
    methodology = _methodology().model_copy(update={"environment": LatencyEnvironment.DEPLOYMENT})
    observations = list(_observations(methodology))
    simple_indices = [
        index
        for index, observation in enumerate(observations)
        if observation.journey_id == "simple-cart-read"
    ][-2:]
    for index in simple_indices:
        observations[index] = observations[index].model_copy(update={"elapsed_seconds": 1.01})
    failed_report = build_latency_report(
        methodology,
        observations,
        run_at=datetime(2026, 9, 9, tzinfo=UTC),
        deployment_id="deployment-a",
    )
    tampered_report = failed_report.model_copy(
        update={"gate": failed_report.gate.model_copy(update={"passed": True, "failures": ()})}
    )

    with pytest.raises(DurableLatencyActivationError, match="derived results"):
        require_deployment_latency_report(
            tampered_report,
            expected_methodology_fingerprint=methodology_fingerprint(methodology),
            expected_deployment_id="deployment-a",
        )


async def test_certification_runner_owns_warmup_sample_count_and_concurrency() -> None:
    methodology = _methodology()
    methodology = methodology.model_copy(
        update={
            "journeys": (
                methodology.journeys[0].model_copy(update={"journey_id": "startup"}),
                *methodology.journeys[1:],
            )
        }
    )
    active = 0
    maximum_active = 0
    startup_calls = 0
    turn_calls: list[str] = []
    turn_runs = 0
    closed_executions = 0

    async def observe_one(observer) -> None:
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        try:
            with observe_duration(observer, DurabilityOperation.CHECKPOINT_READ):
                await asyncio.sleep(0.001)
        finally:
            active -= 1

    class StartedExecution:
        startup_elapsed_seconds = 0.5

        async def aclose(self) -> None:
            nonlocal closed_executions
            closed_executions += 1

    class PreparedTurnExecution(StartedExecution):
        async def run(self) -> float:
            nonlocal turn_runs
            turn_runs += 1
            await observe_one(self.observer)
            return 0.5

        def __init__(self, observer) -> None:
            self.observer = observer

    async def startup_probe(_sample_id: str, observer) -> StartedExecution:
        nonlocal startup_calls
        startup_calls += 1
        await observe_one(observer)
        return StartedExecution()

    async def turn_probe(
        journey: LatencyJourney,
        _sample_id: str,
        observer,
    ) -> PreparedTurnExecution:
        turn_calls.append(journey.journey_id)
        with observe_duration(observer, DurabilityOperation.POOL_OPEN):
            await asyncio.sleep(0)
        return PreparedTurnExecution(observer)

    report = _completed_report(
        await run_latency_certification(
            methodology,
            deployment_id="deployment-a",
            startup_probe=startup_probe,
            turn_probe=turn_probe,
            run_at=datetime(2026, 9, 9, tzinfo=UTC),
        )
    )

    assert report.gate.passed is True
    assert startup_calls == methodology.startup_samples
    assert len(turn_calls) == (
        len(methodology.journeys) * (methodology.warmup_runs + methodology.samples_per_journey)
    )
    assert turn_runs == len(turn_calls)
    assert closed_executions == startup_calls + len(turn_calls)
    assert maximum_active == methodology.concurrency
    assert report.startup.statistics.p50_seconds == 0.5
    assert all(result.statistics.p50_seconds == 0.5 for result in report.journeys)
    assert {component.operation for component in report.journeys[0].components} == {
        DurabilityOperation.CHECKPOINT_READ
    }


@pytest.mark.parametrize("measure_turn,fail_cleanup", ((False, False), (True, False), (True, True)))
async def test_turn_evidence_excludes_cleanup_but_retains_cleanup_failure(
    measure_turn: bool, fail_cleanup: bool
) -> None:
    class StartedExecution:
        startup_elapsed_seconds = 0.5

        async def aclose(self) -> None:
            pass

    class TurnExecution:
        def __init__(self, observer, sample_id: str) -> None:
            self.observer = observer
            self.sample_id = sample_id

        async def run(self) -> float:
            if measure_turn:
                with observe_duration(self.observer, DurabilityOperation.CHECKPOINT_READ):
                    await asyncio.sleep(0)
            return 0.5

        async def aclose(self) -> None:
            with observe_duration(self.observer, DurabilityOperation.REGISTRY_FINALIZE_CLOSE):
                if fail_cleanup and self.sample_id.startswith("sample-"):
                    raise RuntimeError("synthetic cleanup failure")

    async def startup_probe(_sample_id, observer):
        with observe_duration(observer, DurabilityOperation.POOL_OPEN):
            await asyncio.sleep(0)
        return StartedExecution()

    async def turn_probe(_journey, sample_id, observer):
        return TurnExecution(observer, sample_id)

    report = _completed_report(
        await run_latency_certification(
            _methodology().model_copy(
                update={
                    "required_startup_operations": (DurabilityOperation.POOL_OPEN,),
                }
            ),
            deployment_id="deployment-a",
            startup_probe=startup_probe,
            turn_probe=turn_probe,
            run_at=datetime(2026, 9, 9, tzinfo=UTC),
        )
    )
    turns = [sample for sample in report.observations if sample.phase is LatencyPhase.TURN]
    assert len(turns) == 60
    assert report.gate.passed is (measure_turn and not fail_cleanup)
    expected_operations = {DurabilityOperation.CHECKPOINT_READ} if measure_turn else set()
    for sample in turns:
        assert {component.operation for component in sample.components} == expected_operations
        assert sample.error_type == (
            "RuntimeError"
            if fail_cleanup
            else None
            if measure_turn
            else "MissingComponentEvidenceError"
        )


async def test_certification_runner_retains_probe_failure_and_completes_coverage() -> None:
    methodology = _methodology()

    class StartedExecution:
        startup_elapsed_seconds = 0.5

        async def aclose(self) -> None:
            return None

    class PreparedTurnExecution(StartedExecution):
        def __init__(self, sample_id: str, observer) -> None:
            self.sample_id = sample_id
            self.observer = observer

        async def run(self) -> float:
            with observe_duration(self.observer, DurabilityOperation.CHECKPOINT_READ):
                if self.sample_id == "sample-turn-simple-cart-read-0001":
                    raise RuntimeError("synthetic failure detail must not enter evidence")
            return 0.5

    async def startup_probe(_sample_id: str, observer) -> StartedExecution:
        with observe_duration(observer, DurabilityOperation.POOL_OPEN):
            await asyncio.sleep(0)
        return StartedExecution()

    async def turn_probe(
        _journey: LatencyJourney,
        sample_id: str,
        observer,
    ) -> PreparedTurnExecution:
        return PreparedTurnExecution(sample_id, observer)

    report = _completed_report(
        await run_latency_certification(
            methodology,
            deployment_id="deployment-a",
            startup_probe=startup_probe,
            turn_probe=turn_probe,
            run_at=datetime(2026, 9, 9, tzinfo=UTC),
        )
    )

    failed = next(
        observation
        for observation in report.observations
        if observation.sample_id == "sample-turn-simple-cart-read-0001"
    )
    assert failed.outcome is LatencyObservationOutcome.ERROR
    assert failed.error_type == "RuntimeError"
    assert report.gate.passed is False
    assert "synthetic failure detail" not in report.model_dump_json()


async def test_certification_runner_bounds_each_sample_and_retains_timeouts() -> None:
    methodology = _methodology().model_copy(update={"sample_timeout_seconds": 0.005})

    class StartedExecution:
        startup_elapsed_seconds = 0.5

        async def aclose(self) -> None:
            return None

    class PreparedTurnExecution(StartedExecution):
        def __init__(self, sample_id: str, observer) -> None:
            self.sample_id = sample_id
            self.observer = observer

        async def run(self) -> float:
            with observe_duration(self.observer, DurabilityOperation.CHECKPOINT_READ):
                if self.sample_id.startswith("sample-turn-"):
                    await asyncio.Event().wait()
            return 0.5

    async def startup_probe(_sample_id: str, observer) -> StartedExecution:
        with observe_duration(observer, DurabilityOperation.POOL_OPEN):
            await asyncio.sleep(0)
        return StartedExecution()

    async def turn_probe(
        _journey: LatencyJourney,
        sample_id: str,
        observer,
    ) -> PreparedTurnExecution:
        return PreparedTurnExecution(sample_id, observer)

    async with asyncio.timeout(1.0):
        report = _completed_report(
            await run_latency_certification(
                methodology,
                deployment_id="deployment-a",
                startup_probe=startup_probe,
                turn_probe=turn_probe,
                run_at=datetime(2026, 9, 9, tzinfo=UTC),
            )
        )

    turn_observations = tuple(
        observation for observation in report.observations if observation.phase is LatencyPhase.TURN
    )
    assert len(turn_observations) == (len(methodology.journeys) * methodology.samples_per_journey)
    assert all(
        observation.outcome is LatencyObservationOutcome.ERROR
        and observation.error_type == "TimeoutError"
        for observation in turn_observations
    )
    assert report.gate.passed is False


@pytest.mark.parametrize(
    ("failure_stage", "expected_stage", "expected_cleanup_error"),
    (
        ("prepare", LatencyAbortStage.WARMUP_PREPARE, None),
        ("run", LatencyAbortStage.WARMUP_RUN, "RuntimeError"),
        ("cleanup", LatencyAbortStage.WARMUP_CLEANUP, None),
    ),
)
async def test_certification_runner_records_redacted_warmup_abort_evidence(
    failure_stage: str,
    expected_stage: LatencyAbortStage,
    expected_cleanup_error: str | None,
) -> None:
    methodology = _methodology()

    class WarmupExecution:
        async def run(self) -> float:
            if failure_stage == "run":
                raise TimeoutError("sensitive warmup run detail")
            return 0.5

        async def aclose(self) -> None:
            if failure_stage in {"run", "cleanup"}:
                raise RuntimeError("sensitive warmup cleanup detail")

    async def startup_probe(_sample_id, _observer):
        raise AssertionError("sampling must not start after an aborted warmup")

    async def turn_probe(_journey, _sample_id, _observer):
        if failure_stage == "prepare":
            raise TimeoutError("sensitive warmup prepare detail")
        return WarmupExecution()

    run = await run_latency_certification(
        methodology,
        deployment_id="deployment-a",
        startup_probe=startup_probe,
        turn_probe=turn_probe,
        run_at=datetime(2026, 9, 9, tzinfo=UTC),
    )

    assert run.outcome is LatencyCertificationOutcome.ABORTED
    assert run.report is None
    assert run.abort is not None
    assert run.abort.stage is expected_stage
    assert run.abort.sample_id == "warmup-turn-simple-cart-read-0001"
    assert run.abort.journey_id == "simple-cart-read"
    expected_error = "RuntimeError" if failure_stage == "cleanup" else "TimeoutError"
    assert run.abort.error_type == expected_error
    assert run.abort.cleanup_error_type == expected_cleanup_error
    serialized = run.model_dump_json()
    assert "sensitive warmup" not in serialized


async def test_certification_runner_does_not_replace_cancellation_with_cleanup_failure() -> None:
    methodology = _methodology()

    class StartedExecution:
        startup_elapsed_seconds = 0.5

        async def aclose(self) -> None:
            return None

    class CancelledTurnExecution:
        async def run(self) -> float:
            raise asyncio.CancelledError("certification cancelled")

        async def aclose(self) -> None:
            raise RuntimeError("cleanup failed")

    async def startup_probe(_sample_id: str, _observer) -> StartedExecution:
        return StartedExecution()

    async def turn_probe(
        _journey: LatencyJourney,
        _sample_id: str,
        _observer,
    ) -> CancelledTurnExecution:
        return CancelledTurnExecution()

    with pytest.raises(asyncio.CancelledError, match="certification cancelled") as cancelled:
        await run_latency_certification(
            methodology,
            deployment_id="deployment-a",
            startup_probe=startup_probe,
            turn_probe=turn_probe,
            run_at=datetime(2026, 9, 9, tzinfo=UTC),
        )

    assert isinstance(cancelled.value.__cause__, RuntimeError)


def test_failed_samples_remain_evidence_without_inventing_latency_values() -> None:
    methodology = _methodology()
    observations = tuple(
        observation.model_copy(
            update={
                "elapsed_seconds": None,
                "outcome": LatencyObservationOutcome.ERROR,
                "error_type": "SyntheticProbeError",
                "components": (),
            }
        )
        for observation in _observations(methodology)
    )

    report = build_latency_report(
        methodology,
        observations,
        run_at=datetime(2026, 9, 9, tzinfo=UTC),
        deployment_id="deployment-a",
    )

    assert report.gate.passed is False
    assert report.startup.statistics.count == 0
    assert report.startup.statistics.p95_seconds is None
    assert all(result.statistics.count == 0 for result in report.journeys)


def test_abort_evidence_records_a_private_exception_class_name() -> None:
    """A failure an envelope exists to record must not be rejected by its own field type."""
    abort = LatencyCertificationAbort(
        stage=LatencyAbortStage.WARMUP_RUN,
        sample_id="sample-warmup-0001",
        journey_id="simple-cart-read",
        error_type="_InternalProviderError",
        cleanup_error_type="_InternalCleanupError",
    )

    assert abort.error_type == "_InternalProviderError"
    assert abort.cleanup_error_type == "_InternalCleanupError"
