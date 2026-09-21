"""Versioned synthetic dataset and cross-tenant publication contracts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError
from telemetry_helpers import make_tenant_telemetry

from agnostic_market.config.loader import load_yaml_layer
from agnostic_market.management.contracts import (
    MerchantDatasetManifest,
    MerchantDraft,
    MerchantDraftPublicationRequest,
    MerchantScenarioDataset,
    PublishedMerchantVersion,
    merchant_dataset_entity_counts,
    merchant_dataset_manifest_fingerprint,
    merchant_fixture_bundle_fingerprint,
)
from agnostic_market.management.datasets import load_merchant_scenario_dataset
from agnostic_market.management.repository import SqliteMerchantConfigurationRepository
from agnostic_market.management.service import MerchantManagementService
from agnostic_market.management.simulation import (
    build_published_fixture_tenant_services,
    build_published_tenant_context,
)

_NOW = datetime(2026, 9, 20, 12, tzinfo=UTC)


def _datasets(config_root: Path) -> tuple[MerchantScenarioDataset, MerchantScenarioDataset]:
    return (
        load_merchant_scenario_dataset(config_root / "datasets" / "fashion-service-v1.yaml"),
        load_merchant_scenario_dataset(config_root / "datasets" / "grocery-service-v1.yaml"),
    )


def test_scenario_datasets_are_content_bound_and_intentionally_overlap(
    config_root: Path,
) -> None:
    fashion, grocery = _datasets(config_root)

    assert fashion.manifest.tenant_id == "acme_store"
    assert grocery.manifest.tenant_id == "demo_shop"
    assert fashion.manifest.fixture_fingerprint == merchant_fixture_bundle_fingerprint(
        fashion.fixtures
    )
    assert grocery.manifest.fixture_fingerprint == merchant_fixture_bundle_fingerprint(
        grocery.fixtures
    )
    assert fashion.manifest.entity_counts == merchant_dataset_entity_counts(fashion.fixtures)
    assert grocery.manifest.entity_counts == merchant_dataset_entity_counts(grocery.fixtures)
    assert fashion.fixtures.catalog.products[0].sku == grocery.fixtures.catalog.products[0].sku
    assert fashion.fixtures.catalog.products[0].name != grocery.fixtures.catalog.products[0].name
    assert "ORD-SHARED-01" in fashion.fixtures.orders.orders
    assert "ORD-SHARED-01" in grocery.fixtures.orders.orders
    assert fashion.fixtures.orders.orders["ORD-FASHION-05"].status == "cancelled"
    assert grocery.fixtures.orders.orders["ORD-GROCERY-05"].status == "cancelled"
    assert "CUST-FASHION-03" not in fashion.fixtures.profiles.profiles
    assert "CUST-GROCERY-03" not in grocery.fixtures.payment_instruments.payment_instruments
    assert "missing-dependent-data" in fashion.manifest.scenario_tags
    assert "missing-dependent-data" in grocery.manifest.scenario_tags


def test_scenario_dataset_rejects_manifest_count_and_content_drift(config_root: Path) -> None:
    fashion, _grocery = _datasets(config_root)
    payload = fashion.model_dump()
    payload["manifest"]["entity_counts"]["orders"] += 1

    with pytest.raises(ValidationError, match="entity counts"):
        MerchantScenarioDataset.model_validate(payload)

    payload = fashion.model_dump()
    payload["fixtures"]["catalog"]["products"][0]["name"] = "Unbound replacement"
    with pytest.raises(ValidationError, match="fingerprint"):
        MerchantScenarioDataset.model_validate(payload)


def test_two_dataset_publications_remain_tenant_isolated(
    tmp_path: Path,
    config_root: Path,
) -> None:
    fashion, grocery = _datasets(config_root)
    version_ids = iter(("fashion-version-1", "grocery-version-1"))
    repository = SqliteMerchantConfigurationRepository(
        tmp_path / "datasets.sqlite3",
        active_config_root=config_root,
        clock=lambda: _NOW,
        version_id_factory=version_ids.__next__,
    )
    service = MerchantManagementService(config_root, repository, clock=lambda: _NOW)

    published = {}
    for dataset in (fashion, grocery):
        tenant_id = dataset.manifest.tenant_id
        draft = service.save_draft(
            tenant_id,
            MerchantDraft(
                tenant_id=tenant_id,
                draft_id="scenario-draft",
                revision=1,
                actor_id="dataset-steward",
                request_id=f"seed-{tenant_id}",
                created_at=_NOW,
                updated_at=_NOW,
                merchant_override=load_yaml_layer(config_root / "merchants" / f"{tenant_id}.yaml"),
                fixtures=dataset.fixtures,
                dataset_manifest=dataset.manifest,
            ),
            expected_revision=0,
        )
        preview = service.preview_draft(
            tenant_id,
            draft.draft_id,
            expected_revision=draft.revision,
        )
        receipt = service.publish_draft(
            tenant_id,
            MerchantDraftPublicationRequest(
                tenant_id=tenant_id,
                draft_id=draft.draft_id,
                draft_revision=draft.revision,
                expected_preview_fingerprint=preview.preview_fingerprint,
                actor_id="dataset-steward",
                request_id=f"publish-{tenant_id}",
            ),
        )
        published[tenant_id] = service.get_version(tenant_id, receipt.version_id)

    fashion_version = published["acme_store"]
    grocery_version = published["demo_shop"]
    assert fashion_version.dataset_manifest == fashion.manifest
    assert grocery_version.dataset_manifest == grocery.manifest
    assert fashion_version.dataset_fingerprint == merchant_dataset_manifest_fingerprint(
        fashion.manifest
    )
    assert grocery_version.dataset_fingerprint == merchant_dataset_manifest_fingerprint(
        grocery.manifest
    )
    tampered = json.loads(fashion_version.model_dump_json())
    tampered["dataset_manifest"]["revision"] = 2
    with pytest.raises(ValueError, match="dataset fingerprint"):
        PublishedMerchantVersion.model_validate_json(json.dumps(tampered))

    fashion_tenant = build_published_tenant_context(fashion_version)
    grocery_tenant = build_published_tenant_context(grocery_version)
    fashion_services = build_published_fixture_tenant_services(
        fashion_version,
        fashion_tenant,
        telemetry=make_tenant_telemetry(fashion_tenant.tenant_id),
    )
    grocery_services = build_published_fixture_tenant_services(
        grocery_version,
        grocery_tenant,
        telemetry=make_tenant_telemetry(grocery_tenant.tenant_id),
    )

    fashion_product = fashion_services.catalog.resolve_products(("SKU-SHARED-01",))[0]
    grocery_product = grocery_services.catalog.resolve_products(("SKU-SHARED-01",))[0]
    assert fashion_product is not None and grocery_product is not None
    assert fashion_product.name == "weatherproof commuter jacket"
    assert grocery_product.name == "family vegetable box"
    assert fashion_services.order_store.order_item_summary("ORD-SHARED-01") == (
        "1 weatherproof commuter jacket"
    )
    assert grocery_services.order_store.order_item_summary("ORD-SHARED-01") == (
        "1 family vegetable box"
    )


def test_version_diff_reports_dataset_provenance_without_exposing_values(
    tmp_path: Path,
    config_root: Path,
) -> None:
    fashion, _grocery = _datasets(config_root)
    version_ids = iter(("dataset-version-1", "dataset-version-2"))
    service = MerchantManagementService(
        config_root,
        SqliteMerchantConfigurationRepository(
            tmp_path / "dataset-diff.sqlite3",
            active_config_root=config_root,
            clock=lambda: _NOW,
            version_id_factory=version_ids.__next__,
        ),
        clock=lambda: _NOW,
    )
    draft = service.save_draft(
        "acme_store",
        MerchantDraft(
            tenant_id="acme_store",
            draft_id="scenario-draft",
            revision=1,
            actor_id="dataset-steward",
            request_id="seed-dataset",
            created_at=_NOW,
            updated_at=_NOW,
            merchant_override=load_yaml_layer(config_root / "merchants" / "acme_store.yaml"),
            fixtures=fashion.fixtures,
            dataset_manifest=fashion.manifest,
        ),
        expected_revision=0,
    )
    first_preview = service.preview_draft(
        "acme_store", draft.draft_id, expected_revision=draft.revision
    )
    first = service.publish_draft(
        "acme_store",
        MerchantDraftPublicationRequest(
            tenant_id="acme_store",
            draft_id=draft.draft_id,
            draft_revision=draft.revision,
            expected_preview_fingerprint=first_preview.preview_fingerprint,
            actor_id="dataset-steward",
            request_id="publish-dataset-1",
        ),
    )
    updated_manifest = MerchantDatasetManifest.model_validate(
        fashion.manifest.model_dump()
        | {
            "revision": 2,
            "source_kind": "management_derived",
            "source_ref": "approved-refresh",
        }
    )
    updated = service.save_draft(
        "acme_store",
        draft.model_copy(
            update={
                "revision": 2,
                "request_id": "refresh-dataset",
                "dataset_manifest": updated_manifest,
            }
        ),
        expected_revision=1,
    )
    second_preview = service.preview_draft(
        "acme_store", updated.draft_id, expected_revision=updated.revision
    )
    second = service.publish_draft(
        "acme_store",
        MerchantDraftPublicationRequest(
            tenant_id="acme_store",
            draft_id=updated.draft_id,
            draft_revision=updated.revision,
            expected_preview_fingerprint=second_preview.preview_fingerprint,
            expected_active_version_id=first.version_id,
            actor_id="dataset-steward",
            request_id="publish-dataset-2",
        ),
    )

    difference = service.compare_versions("acme_store", first.version_id, second.version_id)

    assert tuple(change.path for change in difference.dataset_changes) == (
        ("manifest", "revision"),
        ("manifest", "source_kind"),
        ("manifest", "source_ref"),
    )
    assert difference.changed is True
    serialized = difference.model_dump_json()
    assert "approved-refresh" not in serialized
    assert fashion.manifest.dataset_id not in serialized
