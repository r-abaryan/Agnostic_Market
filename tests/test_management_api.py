"""HTTP adapter contracts for local merchant administration."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from shutil import copytree
from unittest.mock import Mock

from fastapi.testclient import TestClient
from llm_fakes import TEST_STRUCTURED_OUTPUT_METHOD, FakeChatModel
from routing_helpers import ArchitectureRoutingRecognizer

from agnostic_market.application import ApplicationModels
from agnostic_market.commerce.catalog import load_catalog_fixture
from agnostic_market.commerce.identity import load_customers_fixture
from agnostic_market.commerce.orders import load_orders_fixture
from agnostic_market.commerce.payment_instruments import load_payment_instruments_fixture
from agnostic_market.commerce.profile import load_profile_fixture
from agnostic_market.commerce.verification import load_verification_fixture
from agnostic_market.config.loader import load_yaml_layer
from agnostic_market.management.api import create_management_app
from agnostic_market.management.contracts import MerchantDraft, MerchantFixtureBundle
from agnostic_market.management.datasets import load_merchant_scenario_dataset
from agnostic_market.management.repository import SqliteMerchantConfigurationRepository
from agnostic_market.management.service import MerchantManagementService
from agnostic_market.management.simulation import PublishedMerchantSimulator
from scripts import management_api

_NOW = datetime(2026, 9, 20, 12, tzinfo=UTC)


def _fixtures(config_root: Path) -> MerchantFixtureBundle:
    return MerchantFixtureBundle(
        catalog=load_catalog_fixture(config_root, "acme_store"),
        orders=load_orders_fixture(config_root, "acme_store"),
        customers=load_customers_fixture(config_root, "acme_store"),
        payment_instruments=load_payment_instruments_fixture(config_root, "acme_store"),
        profiles=load_profile_fixture(config_root, "acme_store"),
        verification=load_verification_fixture(config_root, "acme_store"),
    )


def _client(tmp_path: Path, config_root: Path) -> TestClient:
    repository = SqliteMerchantConfigurationRepository(
        tmp_path / "management-api.sqlite3",
        active_config_root=config_root,
        clock=lambda: _NOW,
        version_id_factory=lambda: "version-1",
    )
    service = MerchantManagementService(config_root, repository, clock=lambda: _NOW)
    return TestClient(create_management_app(service, development_actor_id="local-operator"))


def _simulation_client(tmp_path: Path, config_root: Path) -> TestClient:
    repository = SqliteMerchantConfigurationRepository(
        tmp_path / "management-simulation-api.sqlite3",
        active_config_root=config_root,
        clock=lambda: _NOW,
        version_id_factory=lambda: "version-1",
    )
    service = MerchantManagementService(config_root, repository, clock=lambda: _NOW)

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
                }
            ),
            reasoning=FakeChatModel(emit_tool_calls=False),
            response_structured_output_method=TEST_STRUCTURED_OUTPUT_METHOD,
        )

    def routing_factory(_config):
        def build(_registry):
            return ArchitectureRoutingRecognizer()

        return build

    simulator = PublishedMerchantSimulator(
        service,
        models_factory=models_factory,
        routing_factory=routing_factory,
    )
    return TestClient(
        create_management_app(
            service,
            development_actor_id="local-operator",
            simulator=simulator,
        )
    )


def _draft_payload(config_root: Path, *, request_id: str = "draft-request-1") -> dict[str, object]:
    return {
        "schema_version": 1,
        "revision": 1,
        "request_id": request_id,
        "merchant_override": load_yaml_layer(config_root / "merchants" / "acme_store.yaml"),
        "fixtures": _fixtures(config_root).model_dump(mode="json"),
    }


def _save_draft(client: TestClient, config_root: Path) -> dict[str, object]:
    response = client.put(
        "/v1/merchants/acme_store/drafts/draft-1",
        params={"expected_revision": 0},
        json=_draft_payload(config_root),
    )
    assert response.status_code == 200
    return response.json()


def test_api_injects_the_development_actor_and_path_authority(
    tmp_path: Path,
    config_root: Path,
) -> None:
    client = _client(tmp_path, config_root)

    draft = _save_draft(client, config_root)
    assert draft["created_at"] == _NOW.isoformat().replace("+00:00", "Z")
    assert draft["updated_at"] == _NOW.isoformat().replace("+00:00", "Z")

    assert draft["tenant_id"] == "acme_store"
    assert draft["draft_id"] == "draft-1"
    assert draft["actor_id"] == "local-operator"
    assert "actor_id" not in _draft_payload(config_root)
    assert "tenant_id" not in _draft_payload(config_root)


def test_api_lists_configured_merchants_and_seeds_the_first_draft(
    tmp_path: Path,
    config_root: Path,
) -> None:
    client = _client(tmp_path, config_root)

    merchants = client.get("/v1/merchants")
    seeded = client.post(
        "/v1/merchants/acme_store/drafts/working/seed",
        json={"schema_version": 1, "request_id": "seed-request-1"},
    )
    replay = client.post(
        "/v1/merchants/acme_store/drafts/working/seed",
        json={"schema_version": 1, "request_id": "seed-request-1"},
    )

    assert merchants.status_code == 200
    summaries = {item["tenant_id"]: item for item in merchants.json()["merchants"]}
    assert summaries["acme_store"] == {
        "schema_version": 1,
        "tenant_id": "acme_store",
        "configured": True,
        "managed": False,
    }
    assert seeded.status_code == 200
    draft = MerchantDraft.model_validate_json(seeded.content)
    assert draft.tenant_id == "acme_store"
    assert draft.draft_id == "working"
    assert draft.revision == 1
    assert draft.actor_id == "local-operator"
    assert draft.fixtures.catalog.products
    assert replay.status_code == 200
    assert replay.json() == seeded.json()


def test_draft_seed_resolves_merchant_identity_instead_of_assuming_its_filename(
    tmp_path: Path,
    config_root: Path,
) -> None:
    isolated_root = tmp_path / "config"
    copytree(config_root, isolated_root)
    (isolated_root / "merchants" / "acme_store.yaml").rename(
        isolated_root / "merchants" / "storefront.yaml"
    )
    client = _client(tmp_path, isolated_root)

    response = client.post(
        "/v1/merchants/acme_store/drafts/working/seed",
        json={"schema_version": 1, "request_id": "seed-request-1"},
    )

    assert response.status_code == 200
    assert response.json()["tenant_id"] == "acme_store"


def test_api_rejects_unknown_fields_without_echoing_request_values(
    tmp_path: Path,
    config_root: Path,
) -> None:
    client = _client(tmp_path, config_root)
    payload = _draft_payload(config_root)
    payload["actor_id"] = "attacker-controlled-actor"

    response = client.put(
        "/v1/merchants/acme_store/drafts/draft-1",
        params={"expected_revision": 0},
        json=payload,
    )

    assert response.status_code == 422
    assert response.json() == {
        "schema_version": 1,
        "code": "invalid_request",
    }
    assert "attacker-controlled-actor" not in response.text


def test_api_rejects_coerced_body_types(tmp_path: Path, config_root: Path) -> None:
    client = _client(tmp_path, config_root)
    payload = _draft_payload(config_root)
    payload["revision"] = "1"

    response = client.put(
        "/v1/merchants/acme_store/drafts/draft-1",
        params={"expected_revision": 0},
        json=payload,
    )

    assert response.status_code == 422
    assert response.json() == {
        "schema_version": 1,
        "code": "invalid_request",
    }


def test_api_maps_not_found_and_revision_conflicts_to_bounded_errors(
    tmp_path: Path,
    config_root: Path,
) -> None:
    client = _client(tmp_path, config_root)

    missing = client.get("/v1/merchants/acme_store/drafts/missing")
    assert missing.status_code == 404
    assert missing.json()["code"] == "not_found"

    _save_draft(client, config_root)
    conflict = client.put(
        "/v1/merchants/acme_store/drafts/draft-1",
        params={"expected_revision": 0},
        json=_draft_payload(config_root, request_id="draft-request-2"),
    )
    assert conflict.status_code == 409
    assert conflict.json()["code"] == "state_conflict"


def test_api_preview_publish_version_diff_and_audit_flow(
    tmp_path: Path,
    config_root: Path,
) -> None:
    client = _client(tmp_path, config_root)
    _save_draft(client, config_root)

    validation = client.post(
        "/v1/merchants/acme_store/drafts/draft-1/validation",
        json={"schema_version": 1, "draft_revision": 1, "request_id": "validate-1"},
    )
    assert validation.status_code == 200
    assert validation.json()["findings"] == []

    preview = client.post(
        "/v1/merchants/acme_store/drafts/draft-1/preview",
        json={"schema_version": 1, "draft_revision": 1},
    )
    assert preview.status_code == 200

    publish = client.post(
        "/v1/merchants/acme_store/drafts/draft-1/publication",
        json={
            "schema_version": 1,
            "draft_revision": 1,
            "expected_preview_fingerprint": preview.json()["preview_fingerprint"],
            "expected_active_version_id": None,
            "request_id": "publish-1",
        },
    )
    assert publish.status_code == 200
    assert publish.json()["version_id"] == "version-1"

    versions = client.get("/v1/merchants/acme_store/versions")
    assert versions.status_code == 200
    assert [item["version_id"] for item in versions.json()["versions"]] == ["version-1"]

    diff = client.get(
        "/v1/merchants/acme_store/version-diff",
        params={"base_version_id": "version-1", "target_version_id": "version-1"},
    )
    assert diff.status_code == 200
    assert diff.json()["changed"] is False

    audit = client.get("/v1/merchants/acme_store/audit")
    assert audit.status_code == 200
    assert [entry["event"] for entry in audit.json()["records"]] == [
        "draft_created",
        "draft_validated",
        "version_published",
    ]


def test_api_runs_and_resets_an_isolated_published_version_simulation(
    tmp_path: Path,
    config_root: Path,
) -> None:
    with _simulation_client(tmp_path, config_root) as client:
        seeded = client.post(
            "/v1/merchants/acme_store/drafts/working/seed",
            json={"schema_version": 1, "request_id": "seed-1"},
        )
        assert seeded.status_code == 200
        preview = client.post(
            "/v1/merchants/acme_store/drafts/working/preview",
            json={"schema_version": 1, "draft_revision": 1},
        )
        publication = client.post(
            "/v1/merchants/acme_store/drafts/working/publication",
            json={
                "schema_version": 1,
                "draft_revision": 1,
                "expected_preview_fingerprint": preview.json()["preview_fingerprint"],
                "expected_active_version_id": None,
                "request_id": "publish-1",
            },
        )
        assert publication.status_code == 200

        started = client.post(
            "/v1/merchants/acme_store/simulations/demo",
            json={"schema_version": 1, "version_id": None},
        )
        result = client.post(
            "/v1/merchants/acme_store/simulations/demo/turns",
            json={
                "schema_version": 1,
                "request_id": "turn-1",
                "text": "What is your return policy?",
                "readback_interrupted": False,
            },
        )
        state = client.get("/v1/merchants/acme_store/simulations/demo/state")
        replay = client.post(
            "/v1/merchants/acme_store/simulations/demo/turns",
            json={
                "schema_version": 1,
                "request_id": "turn-1",
                "text": "What is your return policy?",
                "readback_interrupted": False,
            },
        )
        conflict = client.post(
            "/v1/merchants/acme_store/simulations/demo/turns",
            json={
                "schema_version": 1,
                "request_id": "turn-1",
                "text": "Show me the catalog.",
                "readback_interrupted": False,
            },
        )
        reset = client.post("/v1/merchants/acme_store/simulations/demo/reset")
        closed = client.delete("/v1/merchants/acme_store/simulations/demo")
        missing = client.get("/v1/merchants/acme_store/simulations/demo")

    assert started.status_code == 200
    assert started.json()["publication_version_id"] == publication.json()["version_id"]
    assert result.status_code == 200
    assert result.json()["turn_number"] == 1
    assert result.json()["replayed"] is False
    assert result.json()["state"]["turn_count"] == 1
    assert state.status_code == 200
    assert state.json() == result.json()["state"]
    assert [record["event"] for record in result.json()["routing_records"]] == [
        "semantic_route",
        "capability_answered",
    ]
    assert replay.status_code == 200
    assert replay.json()["replayed"] is True
    assert conflict.status_code == 409
    assert conflict.json()["code"] == "replay_conflict"
    assert reset.status_code == 200
    assert reset.json()["publication_version_id"] == publication.json()["version_id"]
    assert reset.json()["turn_count"] == 0
    assert closed.status_code == 204
    assert missing.status_code == 404


def test_api_reports_unconfigured_simulation_without_exposing_details(
    tmp_path: Path,
    config_root: Path,
) -> None:
    client = _client(tmp_path, config_root)

    response = client.post(
        "/v1/merchants/acme_store/simulations/demo",
        json={"schema_version": 1, "version_id": None},
    )

    assert response.status_code == 503
    assert response.json() == {"schema_version": 1, "code": "service_unavailable"}


def test_api_catalog_pagination_is_bounded_and_value_carrying_only_by_design(
    tmp_path: Path,
    config_root: Path,
) -> None:
    client = _client(tmp_path, config_root)
    _save_draft(client, config_root)

    page = client.get(
        "/v1/merchants/acme_store/drafts/draft-1/catalog",
        params={"offset": 0, "limit": 1},
    )

    assert page.status_code == 200
    body = page.json()
    assert body["offset"] == 0
    assert body["limit"] == 1
    assert body["total"] >= 1
    assert len(body["products"]) == 1


def test_api_catalog_import_preserves_other_fixture_families(
    tmp_path: Path,
    config_root: Path,
) -> None:
    client = _client(tmp_path, config_root)
    original = MerchantDraft.model_validate_json(json.dumps(_save_draft(client, config_root)))
    catalog_fixture = _fixtures(config_root).catalog
    catalog = catalog_fixture.model_copy(
        update={
            "products": (
                catalog_fixture.products[0].model_copy(update={"name": "Updated catalog name"}),
                *catalog_fixture.products[1:],
            )
        }
    ).model_dump(mode="json")

    imported = client.put(
        "/v1/merchants/acme_store/drafts/draft-1/catalog",
        json={
            "schema_version": 1,
            "expected_draft_revision": 1,
            "request_id": "catalog-import-1",
            "catalog": catalog,
        },
    )

    assert imported.status_code == 200
    body = MerchantDraft.model_validate_json(imported.content)
    assert body.revision == 2
    assert body.actor_id == "local-operator"
    assert body.fixtures.catalog.products[0].name == "Updated catalog name"
    for family in ("orders", "customers", "payment_instruments", "profiles", "verification"):
        assert getattr(body.fixtures, family) == getattr(original.fixtures, family)


def test_api_imports_a_versioned_scenario_dataset_atomically(
    tmp_path: Path,
    config_root: Path,
) -> None:
    client = _client(tmp_path, config_root)
    draft = _save_draft(client, config_root)
    assert isinstance(draft["revision"], int)
    draft_revision = draft["revision"]
    dataset = load_merchant_scenario_dataset(config_root / "datasets" / "fashion-service-v1.yaml")

    response = client.put(
        "/v1/merchants/acme_store/drafts/draft-1/dataset",
        json={
            "schema_version": 1,
            "expected_draft_revision": draft_revision,
            "request_id": "dataset-import-1",
            "dataset": dataset.model_dump(mode="json"),
        },
    )

    assert response.status_code == 200
    imported = response.json()
    assert imported["revision"] == draft_revision + 1
    assert imported["dataset_manifest"] == dataset.manifest.model_dump(mode="json")
    assert imported["fixtures"] == dataset.fixtures.model_dump(mode="json")

    without_provenance = client.put(
        "/v1/merchants/acme_store/drafts/draft-1",
        params={"expected_revision": imported["revision"]},
        json={
            "schema_version": 1,
            "revision": imported["revision"] + 1,
            "request_id": "draft-without-provenance",
            "merchant_override": imported["merchant_override"],
            "fixtures": imported["fixtures"],
        },
    )
    assert without_provenance.status_code == 409

    preserved = client.put(
        "/v1/merchants/acme_store/drafts/draft-1",
        params={"expected_revision": imported["revision"]},
        json={
            "schema_version": 1,
            "revision": imported["revision"] + 1,
            "request_id": "draft-preserving-provenance",
            "merchant_override": imported["merchant_override"],
            "fixtures": imported["fixtures"],
            "dataset_manifest": imported["dataset_manifest"],
        },
    )
    assert preserved.status_code == 200
    assert preserved.json()["dataset_manifest"] == imported["dataset_manifest"]


def test_api_openapi_does_not_accept_actor_or_tenant_in_write_bodies(
    tmp_path: Path,
    config_root: Path,
) -> None:
    client = _client(tmp_path, config_root)

    schema = client.get("/openapi.json").json()
    draft_write = schema["components"]["schemas"]["MerchantDraftWrite"]

    assert "actor_id" not in draft_write["properties"]
    assert "tenant_id" not in draft_write["properties"]
    assert "created_at" not in draft_write["properties"]
    assert "updated_at" not in draft_write["properties"]
    assert draft_write["additionalProperties"] is False


def test_management_api_runner_is_fixed_to_loopback(
    tmp_path: Path,
    config_root: Path,
    monkeypatch,
) -> None:
    run = Mock()
    monkeypatch.setattr(management_api.uvicorn, "run", run)

    management_api.serve(
        config_root=config_root,
        database=tmp_path / "state" / "management.sqlite3",
        actor_id="local-operator",
        port=8123,
    )

    app = run.call_args.args[0]
    assert app.title == "Agnostic Market administration API"
    assert run.call_args.kwargs == {"host": "127.0.0.1", "port": 8123}
    assert (tmp_path / "state").is_dir()


def test_management_ui_is_same_origin_accessible_and_hardened(
    tmp_path: Path,
    config_root: Path,
) -> None:
    client = _client(tmp_path, config_root)

    page = client.get("/admin")
    script = client.get("/admin/assets/app.js")
    styles = client.get("/admin/assets/styles.css")
    favicon = client.get("/admin/assets/favicon.svg")

    assert page.status_code == 200
    assert page.headers["content-security-policy"] == (
        "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; "
        "img-src 'self' data:; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; "
        "form-action 'self'"
    )
    assert page.headers["cache-control"] == "no-store"
    assert 'id="merchant-identity"' in page.text
    assert 'id="simulation-transcript"' in page.text
    assert 'id="send-simulation-turn"' in page.text
    assert 'id="simulation-receipts"' in page.text
    assert 'aria-live="polite"' in page.text
    assert '<main id="main-content"' in page.text
    assert "https://" not in page.text
    assert script.status_code == 200
    assert styles.status_code == 200
    assert favicon.status_code == 200
    assert 'from "./client.js"' in script.text


def test_internal_response_validation_failures_are_bounded(
    tmp_path: Path,
    config_root: Path,
    monkeypatch,
) -> None:
    repository = SqliteMerchantConfigurationRepository(
        tmp_path / "management-response-errors.sqlite3",
        active_config_root=config_root,
    )
    service = MerchantManagementService(config_root, repository)
    sentinel = "fixture-value-must-not-be-reflected"
    client = TestClient(
        create_management_app(service, development_actor_id="local-operator"),
        raise_server_exceptions=False,
    )

    monkeypatch.setattr(service, "get_active_version", lambda _tenant_id: {"value": sentinel})
    response_failure = client.get("/v1/merchants/acme_store/versions/active")

    monkeypatch.setattr(service, "list_versions", lambda _tenant_id: ({"value": sentinel},))
    contract_failure = client.get("/v1/merchants/acme_store/versions")

    for response in (response_failure, contract_failure):
        assert response.status_code == 500
        assert response.json() == {"schema_version": 1, "code": "service_unavailable"}
        assert sentinel not in response.text
