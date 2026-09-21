"""Published-version text simulation contracts."""

from __future__ import annotations

from pathlib import Path

import pytest
from llm_fakes import TEST_STRUCTURED_OUTPUT_METHOD, FakeChatModel
from routing_helpers import ArchitectureRoutingRecognizer

from agnostic_market.application import ApplicationModels
from agnostic_market.dtos.events import SpokenMessageEvent, TokenEvent
from agnostic_market.management.contracts import (
    MerchantDraftPublicationRequest,
    MerchantDraftSeedRequest,
)
from agnostic_market.management.repository import SqliteMerchantConfigurationRepository
from agnostic_market.management.service import MerchantManagementService
from agnostic_market.management.simulation import (
    MerchantSimulationConflictError,
    MerchantSimulationNotFoundError,
    MerchantSimulationReplayConflictError,
    PublishedMerchantSimulator,
)


def _service(tmp_path: Path, config_root: Path) -> MerchantManagementService:
    version_ids = iter(("version-1", "version-2", "version-3"))
    repository = SqliteMerchantConfigurationRepository(
        tmp_path / "simulation.sqlite3",
        active_config_root=config_root,
        version_id_factory=version_ids.__next__,
    )
    return MerchantManagementService(config_root, repository)


def _publish(
    service: MerchantManagementService,
    *,
    request_id: str,
    expected_active_version_id: str | None = None,
) -> str:
    draft = service.get_draft("acme_store", "simulation-draft")
    preview = service.preview_draft(
        "acme_store",
        draft.draft_id,
        expected_revision=draft.revision,
    )
    receipt = service.publish_draft(
        "acme_store",
        MerchantDraftPublicationRequest(
            tenant_id="acme_store",
            draft_id=draft.draft_id,
            draft_revision=draft.revision,
            expected_preview_fingerprint=preview.preview_fingerprint,
            expected_active_version_id=expected_active_version_id,
            actor_id="operator-1",
            request_id=request_id,
        ),
    )
    return receipt.version_id


def _simulator(
    service: MerchantManagementService,
    *,
    max_turns: int = 100,
    reasoning: FakeChatModel | None = None,
) -> PublishedMerchantSimulator:
    def models_factory(_config) -> ApplicationModels:
        return ApplicationModels(
            response=FakeChatModel(
                structured_args={
                    "AnswerResponse": (
                        {
                            "decision": "answer",
                            "answer": "Returns are accepted within 30 days.",
                        },
                    )
                },
            ),
            reasoning=reasoning or FakeChatModel(emit_tool_calls=False),
            response_structured_output_method=TEST_STRUCTURED_OUTPUT_METHOD,
        )

    def routing_factory(_config):
        def build(_registry):
            return ArchitectureRoutingRecognizer()

        return build

    return PublishedMerchantSimulator(
        service,
        models_factory=models_factory,
        routing_factory=routing_factory,
        max_turns=max_turns,
    )


def _seed_and_publish(service: MerchantManagementService) -> str:
    service.seed_draft(
        "acme_store",
        MerchantDraftSeedRequest(
            tenant_id="acme_store",
            draft_id="simulation-draft",
            actor_id="operator-1",
            request_id="seed-1",
        ),
    )
    return _publish(service, request_id="publish-1")


@pytest.mark.asyncio
async def test_simulator_runs_the_real_engine_and_replays_exact_turns(
    tmp_path: Path,
    config_root: Path,
) -> None:
    service = _service(tmp_path, config_root)
    version_id = _seed_and_publish(service)
    simulator = _simulator(service)

    status = await simulator.start("acme_store", "simulation-1")
    result = await simulator.turn(
        "acme_store",
        "simulation-1",
        request_id="turn-1",
        text="What is your return policy?",
    )
    replayed = await simulator.turn(
        "acme_store",
        "simulation-1",
        request_id="turn-1",
        text="What is your return policy?",
    )

    assert status.publication_version_id == version_id
    assert status.turn_count == 0
    assert "".join(
        event.text for event in result.events if isinstance(event, TokenEvent | SpokenMessageEvent)
    ) == ("Returns are accepted within 30 days.")
    assert result.turn_number == 1
    assert result.replayed is False
    assert replayed == result.model_copy(update={"replayed": True})
    assert tuple(record.event.value for record in result.routing_records) == (
        "semantic_route",
        "capability_answered",
    )
    assert len(result.latency) == 1
    assert result.state.turn_count == 1
    assert result.state.session_revision == result.session_revision
    assert result.state.cart_lines == ()
    assert result.state.cart_total_usd == 0
    assert result.state.committed_receipts.model_dump() == {
        "cart": {"mutations": 0},
        "orders": {
            "placements": 0,
            "refunds": 0,
            "returns": 0,
            "cancellations": 0,
        },
        "profiles": {"changes": 0},
    }

    with pytest.raises(MerchantSimulationReplayConflictError, match="different input"):
        await simulator.turn(
            "acme_store",
            "simulation-1",
            request_id="turn-1",
            text="Show me the catalog instead.",
        )

    await simulator.close("acme_store", "simulation-1")
    with pytest.raises(MerchantSimulationNotFoundError, match="does not exist"):
        await simulator.status("acme_store", "simulation-1")


@pytest.mark.asyncio
async def test_simulator_projects_bounded_commerce_state_after_a_real_cart_flow(
    tmp_path: Path,
    config_root: Path,
) -> None:
    service = _service(tmp_path, config_root)
    _seed_and_publish(service)
    simulator = _simulator(
        service,
        reasoning=FakeChatModel(
            scripted_calls=[
                [("provide_cart_item", {"candidate_key": "1"})],
                [("provide_cart_quantity", {"quantity": 2})],
            ]
        ),
    )
    await simulator.start("acme_store", "simulation-1")

    await simulator.turn(
        "acme_store",
        "simulation-1",
        request_id="turn-1",
        text="could ya add one of those to my cart",
    )
    await simulator.turn(
        "acme_store",
        "simulation-1",
        request_id="turn-2",
        text="make it two",
    )
    result = await simulator.turn(
        "acme_store",
        "simulation-1",
        request_id="turn-3",
        text="yeah go ahead",
    )
    current = await simulator.inspect_state("acme_store", "simulation-1")

    assert current == result.state
    assert current.turn_count == 3
    assert current.session_revision >= 1
    assert len(current.cart_lines) == 1
    assert current.cart_lines[0].quantity == 2
    assert current.cart_total_usd == current.cart_lines[0].line_total
    assert current.recent_order_count == 0
    assert current.guest_order_count == 0
    assert current.has_discardable_state is True
    assert current.committed_receipts.cart.mutations == 1
    assert current.committed_receipts.orders.placements == 0
    assert current.committed_receipts.profiles.changes == 0
    serialized = current.model_dump_json()
    serialized_state = current.model_dump(mode="json")
    assert serialized_state["cart_lines"][0]["line_total"] == str(current.cart_lines[0].line_total)
    assert type(current).model_validate_json(serialized) == current
    assert "customer_ref" not in serialized
    assert "order_refs" not in serialized

    await simulator.aclose()


@pytest.mark.asyncio
async def test_reset_keeps_the_selected_publication_and_clears_session_state(
    tmp_path: Path,
    config_root: Path,
) -> None:
    service = _service(tmp_path, config_root)
    first_version_id = _seed_and_publish(service)
    simulator = _simulator(service)
    first = await simulator.start("acme_store", "simulation-1")
    await simulator.turn(
        "acme_store",
        "simulation-1",
        request_id="turn-1",
        text="What is your return policy?",
    )

    draft = service.get_draft("acme_store", "simulation-draft")
    changed_override = dict(draft.merchant_override)
    changed_override["display_name"] = "Later Published Store"
    service.save_draft(
        "acme_store",
        draft.model_copy(
            update={
                "revision": draft.revision + 1,
                "request_id": "draft-update-2",
                "merchant_override": changed_override,
            }
        ),
        expected_revision=draft.revision,
    )
    second_version_id = _publish(
        service,
        request_id="publish-2",
        expected_active_version_id=first_version_id,
    )

    reset = await simulator.reset("acme_store", "simulation-1")
    second = await simulator.start("acme_store", "simulation-2")

    assert first.publication_version_id == first_version_id
    assert reset.publication_version_id == first_version_id
    assert reset.config_version == first.config_version
    assert reset.turn_count == 0
    assert second.publication_version_id == second_version_id
    assert second.config_version != first.config_version

    await simulator.aclose()


@pytest.mark.asyncio
async def test_simulator_rejects_cross_tenant_publication_selection(
    tmp_path: Path,
    config_root: Path,
) -> None:
    service = _service(tmp_path, config_root)
    version_id = _seed_and_publish(service)
    simulator = _simulator(service)

    with pytest.raises(MerchantSimulationNotFoundError, match="publication"):
        await simulator.start(
            "demo_shop",
            "simulation-1",
            version_id=version_id,
        )


@pytest.mark.asyncio
async def test_simulator_bounds_unique_turns_without_blocking_exact_replay(
    tmp_path: Path,
    config_root: Path,
) -> None:
    service = _service(tmp_path, config_root)
    _seed_and_publish(service)
    simulator = _simulator(service, max_turns=1)
    await simulator.start("acme_store", "simulation-1")
    first = await simulator.turn(
        "acme_store",
        "simulation-1",
        request_id="turn-1",
        text="What is your return policy?",
    )

    replay = await simulator.turn(
        "acme_store",
        "simulation-1",
        request_id="turn-1",
        text="What is your return policy?",
    )
    assert replay == first.model_copy(update={"replayed": True})
    with pytest.raises(MerchantSimulationConflictError, match="turn limit"):
        await simulator.turn(
            "acme_store",
            "simulation-1",
            request_id="turn-2",
            text="What is a shoe midsole?",
        )

    await simulator.aclose()
