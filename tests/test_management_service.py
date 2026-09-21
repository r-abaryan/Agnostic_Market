"""Application-service contracts for merchant configuration management."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from shutil import copytree
from typing import Any

import pytest
from pydantic import ValidationError
from telemetry_helpers import make_tenant_telemetry

from agnostic_market.commerce.catalog import load_catalog_fixture
from agnostic_market.commerce.identity import load_customers_fixture
from agnostic_market.commerce.orders import load_orders_fixture
from agnostic_market.commerce.payment_instruments import load_payment_instruments_fixture
from agnostic_market.commerce.profile import load_profile_fixture
from agnostic_market.commerce.verification import load_verification_fixture
from agnostic_market.config.loader import load_yaml_layer
from agnostic_market.management.contracts import (
    MerchantCatalogImportRequest,
    MerchantDatasetImportRequest,
    MerchantDatasetManifest,
    MerchantDraft,
    MerchantDraftPublicationRequest,
    MerchantDraftValidationRequest,
    MerchantFixtureBundle,
    MerchantRetirementRequest,
    MerchantRollbackRequest,
    MerchantScenarioDataset,
    merchant_dataset_entity_counts,
    merchant_fixture_bundle_fingerprint,
)
from agnostic_market.management.repository import SqliteMerchantConfigurationRepository
from agnostic_market.management.service import (
    MerchantManagementConflictError,
    MerchantManagementNotFoundError,
    MerchantManagementReplayConflictError,
    MerchantManagementScopeError,
    MerchantManagementService,
    MerchantManagementValidationError,
)
from agnostic_market.management.simulation import (
    build_published_fixture_tenant_services,
    build_published_tenant_context,
)

_NOW = datetime(2026, 9, 19, 14, tzinfo=UTC)


def _bundle(config_root: Path) -> MerchantFixtureBundle:
    return MerchantFixtureBundle(
        catalog=load_catalog_fixture(config_root, "acme_store"),
        orders=load_orders_fixture(config_root, "acme_store"),
        customers=load_customers_fixture(config_root, "acme_store"),
        payment_instruments=load_payment_instruments_fixture(config_root, "acme_store"),
        profiles=load_profile_fixture(config_root, "acme_store"),
        verification=load_verification_fixture(config_root, "acme_store"),
    )


def _override(config_root: Path) -> dict[str, Any]:
    return load_yaml_layer(config_root / "merchants" / "acme_store.yaml")


def _draft(
    config_root: Path,
    *,
    revision: int = 1,
    request_id: str = "draft-request-1",
    override: dict[str, Any] | None = None,
    fixtures: MerchantFixtureBundle | None = None,
) -> MerchantDraft:
    return MerchantDraft(
        tenant_id="acme_store",
        draft_id="draft-1",
        revision=revision,
        actor_id="operator-1",
        request_id=request_id,
        created_at=_NOW,
        updated_at=_NOW + timedelta(seconds=revision - 1),
        merchant_override=_override(config_root) if override is None else override,
        fixtures=_bundle(config_root) if fixtures is None else fixtures,
    )


def _service(
    tmp_path: Path,
    config_root: Path,
) -> MerchantManagementService:
    version_ids = iter(("version-1", "version-2", "version-3"))
    repository = SqliteMerchantConfigurationRepository(
        tmp_path / "management.sqlite3",
        active_config_root=config_root,
        clock=lambda: _NOW,
        version_id_factory=version_ids.__next__,
    )
    return MerchantManagementService(config_root, repository, clock=lambda: _NOW)


def _publication(
    service: MerchantManagementService | None = None,
    *,
    draft_revision: int = 1,
    request_id: str = "publish-request-1",
    expected_active_version_id: str | None = None,
    expected_preview_fingerprint: str | None = None,
) -> MerchantDraftPublicationRequest:
    if expected_preview_fingerprint is None:
        expected_preview_fingerprint = (
            "0" * 64
            if service is None
            else service.preview_draft(
                "acme_store",
                "draft-1",
                expected_revision=draft_revision,
            ).preview_fingerprint
        )
    return MerchantDraftPublicationRequest(
        tenant_id="acme_store",
        draft_id="draft-1",
        draft_revision=draft_revision,
        expected_preview_fingerprint=expected_preview_fingerprint,
        expected_active_version_id=expected_active_version_id,
        actor_id="operator-1",
        request_id=request_id,
    )


def _validation(
    *,
    draft_revision: int = 1,
    request_id: str = "validate-request-1",
    tenant_id: str = "acme_store",
) -> MerchantDraftValidationRequest:
    return MerchantDraftValidationRequest(
        tenant_id=tenant_id,
        draft_id="draft-1",
        draft_revision=draft_revision,
        actor_id="operator-1",
        request_id=request_id,
    )


def test_service_validates_and_previews_through_the_production_resolver(
    tmp_path: Path, config_root: Path
) -> None:
    service = _service(tmp_path, config_root)
    draft = service.save_draft("acme_store", _draft(config_root), expected_revision=0)

    validation = service.validate_draft("acme_store", _validation(draft_revision=draft.revision))
    preview = service.preview_draft("acme_store", draft.draft_id, expected_revision=draft.revision)

    assert validation.valid is True
    assert validation.findings == ()
    assert preview.config.merchant_id == draft.tenant_id
    assert preview.config.secrets_ref == draft.merchant_override["secrets_ref"]
    assert preview.fixtures == draft.fixtures


def test_service_maps_invalid_configuration_to_bounded_findings(
    tmp_path: Path, config_root: Path
) -> None:
    service = _service(tmp_path, config_root)
    invalid_override = _override(config_root)
    invalid_override["_platform"] = {"guardrails": {"tenant_isolation": False}}
    draft = service.save_draft(
        "acme_store",
        _draft(config_root, override=invalid_override),
        expected_revision=0,
    )

    validation = service.validate_draft("acme_store", _validation(draft_revision=draft.revision))

    assert tuple(finding.code for finding in validation.findings) == ("configuration_invalid",)
    assert validation.findings[0].path == ("merchant_override",)
    with pytest.raises(MerchantManagementValidationError) as failure:
        service.preview_draft("acme_store", draft.draft_id, expected_revision=draft.revision)
    assert failure.value.result == validation
    assert "_platform" not in str(failure.value)
    with pytest.raises(MerchantManagementValidationError):
        service.publish_draft("acme_store", _publication())
    with pytest.raises(MerchantManagementNotFoundError, match="no active"):
        service.get_active_version("acme_store")


def test_service_contains_prevalidation_policy_type_failures(
    tmp_path: Path, config_root: Path
) -> None:
    service = _service(tmp_path, config_root)
    invalid_override = _override(config_root)
    invalid_override["policies"]["refunds"]["require_human_above_usd"] = "not-money"
    draft = service.save_draft(
        "acme_store",
        _draft(config_root, override=invalid_override),
        expected_revision=0,
    )

    validation = service.validate_draft("acme_store", _validation(draft_revision=draft.revision))

    assert tuple(finding.code for finding in validation.findings) == ("configuration_invalid",)


def test_service_validates_secret_reference_syntax_without_resolving_values(
    tmp_path: Path, config_root: Path
) -> None:
    service = _service(tmp_path, config_root)
    invalid_override = _override(config_root)
    invalid_override["secrets_ref"] = "not-a-provider-reference"
    draft = service.save_draft(
        "acme_store",
        _draft(config_root, override=invalid_override),
        expected_revision=0,
    )

    validation = service.validate_draft("acme_store", _validation(draft_revision=draft.revision))

    assert tuple((finding.code, finding.path) for finding in validation.findings) == (
        ("secret_reference_invalid", ("secrets_ref",)),
    )
    assert "not-a-provider-reference" not in validation.model_dump_json()


def test_service_rejects_cross_tenant_and_stale_draft_operations(
    tmp_path: Path, config_root: Path
) -> None:
    service = _service(tmp_path, config_root)
    draft = _draft(config_root)

    with pytest.raises(MerchantManagementScopeError, match="tenant scope"):
        service.save_draft("other_store", draft, expected_revision=0)

    service.save_draft("acme_store", draft, expected_revision=0)
    with pytest.raises(MerchantManagementScopeError, match="tenant scope"):
        service.publish_draft("other_store", _publication())
    with pytest.raises(MerchantManagementNotFoundError, match="does not exist"):
        service.get_draft("other_store", draft.draft_id)
    with pytest.raises(MerchantManagementConflictError, match="expected revision"):
        service.validate_draft("acme_store", _validation(draft_revision=2))


def test_draft_changes_do_not_affect_active_sessions_until_publication(
    tmp_path: Path, config_root: Path
) -> None:
    service = _service(tmp_path, config_root)
    service.save_draft("acme_store", _draft(config_root), expected_revision=0)
    first_receipt = service.publish_draft("acme_store", _publication(service))
    first_active = service.get_active_version("acme_store")

    changed_override = _override(config_root)
    changed_override["display_name"] = "Updated Development Store"
    service.save_draft(
        "acme_store",
        _draft(
            config_root,
            revision=2,
            request_id="draft-request-2",
            override=changed_override,
        ),
        expected_revision=1,
    )

    assert service.get_active_version("acme_store") == first_active
    second_receipt = service.publish_draft(
        "acme_store",
        _publication(
            service,
            draft_revision=2,
            request_id="publish-request-2",
            expected_active_version_id=first_receipt.version_id,
        ),
    )
    second_active = service.get_active_version("acme_store")
    assert second_active.version_id == second_receipt.version_id
    assert second_active.config.display_name == "Updated Development Store"
    assert tuple(version.version_id for version in service.list_versions("acme_store")) == (
        first_receipt.version_id,
        second_receipt.version_id,
    )
    assert service.list_tenant_ids() == ("acme_store",)


def test_validation_command_is_scoped_audited_and_replay_stable(
    tmp_path: Path, config_root: Path
) -> None:
    validation_times = iter(tuple(_NOW + timedelta(seconds=offset) for offset in range(4)))
    repository = SqliteMerchantConfigurationRepository(
        tmp_path / "validation-replay.sqlite3",
        active_config_root=config_root,
        clock=lambda: _NOW,
    )
    service = MerchantManagementService(
        config_root,
        repository,
        clock=validation_times.__next__,
    )
    service.save_draft("acme_store", _draft(config_root), expected_revision=0)
    request = _validation()

    result = service.validate_draft("acme_store", request)
    replay = service.validate_draft("acme_store", request)

    assert replay == result
    assert tuple(record.event for record in service.list_audit_history("acme_store")) == (
        "draft_created",
        "draft_validated",
    )
    with pytest.raises(MerchantManagementScopeError, match="tenant scope"):
        service.validate_draft("other_store", request)
    with pytest.raises(MerchantManagementReplayConflictError, match="different parameters"):
        service.validate_draft(
            "acme_store",
            request.model_copy(update={"actor_id": "operator-2"}),
        )


def test_complete_dataset_import_is_scoped_replay_safe_and_draft_only(
    tmp_path: Path, config_root: Path
) -> None:
    service = _service(tmp_path, config_root)
    draft = service.save_draft("acme_store", _draft(config_root), expected_revision=0)
    published = service.publish_draft("acme_store", _publication(service))
    first_version = service.get_active_version("acme_store")
    first_tenant = build_published_tenant_context(first_version)
    first_services = build_published_fixture_tenant_services(
        first_version,
        first_tenant,
        telemetry=make_tenant_telemetry(first_tenant.tenant_id),
    )
    first_product = draft.fixtures.catalog.products[0]
    changed_catalog = draft.fixtures.catalog.model_copy(
        update={
            "products": (
                first_product.model_copy(update={"name": "Imported Development Product"}),
                *draft.fixtures.catalog.products[1:],
            )
        }
    )
    changed_fixtures = draft.fixtures.model_copy(update={"catalog": changed_catalog})
    request = MerchantDatasetImportRequest(
        tenant_id="acme_store",
        draft_id=draft.draft_id,
        expected_draft_revision=draft.revision,
        actor_id="operator-2",
        request_id="dataset-import-1",
        dataset=MerchantScenarioDataset(
            manifest=MerchantDatasetManifest(
                dataset_id="service-import-v1",
                tenant_id="acme_store",
                revision=1,
                generated_at=_NOW,
                source_kind="hand_authored_synthetic",
                source_ref="service-import-source-v1",
                entity_counts=merchant_dataset_entity_counts(changed_fixtures),
                fixture_fingerprint=merchant_fixture_bundle_fingerprint(changed_fixtures),
                covered_capabilities=("search_catalog",),
                scenario_tags=("catalog",),
            ),
            fixtures=changed_fixtures,
        ),
    )

    with pytest.raises(MerchantManagementScopeError, match="tenant scope"):
        service.import_dataset("other_store", request)
    imported = service.import_dataset("acme_store", request)
    replayed = service.import_dataset("acme_store", request)

    assert imported.revision == 2
    assert replayed == imported
    assert service.get_active_version("acme_store").version_id == published.version_id
    assert (
        service.get_active_version("acme_store").fixtures.catalog.products[0].name
        != "Imported Development Product"
    )
    assert tuple(record.event for record in service.list_audit_history("acme_store")) == (
        "draft_created",
        "version_published",
        "draft_updated",
    )
    with pytest.raises(MerchantManagementReplayConflictError, match="different parameters"):
        service.import_dataset(
            "acme_store",
            request.model_copy(update={"actor_id": "operator-3"}),
        )

    replacement = service.publish_draft(
        "acme_store",
        _publication(
            service,
            draft_revision=imported.revision,
            request_id="publish-request-2",
            expected_active_version_id=published.version_id,
        ),
    )
    assert service.get_active_version("acme_store").version_id == replacement.version_id
    assert (
        service.get_active_version("acme_store").fixtures.catalog.products[0].name
        == "Imported Development Product"
    )
    second_version = service.get_active_version("acme_store")
    second_tenant = build_published_tenant_context(second_version)
    second_services = build_published_fixture_tenant_services(
        second_version,
        second_tenant,
        telemetry=make_tenant_telemetry(second_tenant.tenant_id),
    )
    assert first_tenant.config_version != second_tenant.config_version
    assert first_services.catalog.browse().products[0].name != "Imported Development Product"
    assert second_services.catalog.browse().products[0].name == "Imported Development Product"
    with pytest.raises(ValueError, match="does not match"):
        build_published_fixture_tenant_services(
            second_version,
            first_tenant,
            telemetry=make_tenant_telemetry(first_tenant.tenant_id),
        )
    wrong_policy_tenant = replace(
        second_tenant,
        policy=second_tenant.policy.model_copy(
            update={
                "refund_require_human_above_usd": (
                    second_tenant.policy.refund_require_human_above_usd + 1
                )
            }
        ),
    )
    with pytest.raises(ValueError, match="does not match"):
        build_published_fixture_tenant_services(
            second_version,
            wrong_policy_tenant,
            telemetry=make_tenant_telemetry(wrong_policy_tenant.tenant_id),
        )


def test_catalog_import_exact_retry_replays_the_committed_draft(
    tmp_path: Path, config_root: Path
) -> None:
    service = _service(tmp_path, config_root)
    draft = service.save_draft("acme_store", _draft(config_root), expected_revision=0)
    first_product = draft.fixtures.catalog.products[0]
    changed_catalog = draft.fixtures.catalog.model_copy(
        update={
            "products": (
                first_product.model_copy(update={"name": "Imported Development Product"}),
                *draft.fixtures.catalog.products[1:],
            )
        }
    )
    request = MerchantCatalogImportRequest(
        tenant_id="acme_store",
        draft_id=draft.draft_id,
        expected_draft_revision=draft.revision,
        actor_id="operator-2",
        request_id="catalog-import-1",
        catalog=changed_catalog,
    )

    imported = service.import_catalog("acme_store", request)
    replayed = service.import_catalog("acme_store", request)

    assert replayed == imported
    with pytest.raises(MerchantManagementReplayConflictError, match="different parameters"):
        service.import_catalog(
            "acme_store",
            request.model_copy(update={"actor_id": "operator-3"}),
        )


def test_service_publish_retry_ignores_the_new_preview_observation_time(
    tmp_path: Path, config_root: Path
) -> None:
    preview_times = iter(tuple(_NOW + timedelta(seconds=offset) for offset in range(4)))
    repository = SqliteMerchantConfigurationRepository(
        tmp_path / "retry.sqlite3",
        active_config_root=config_root,
        clock=lambda: _NOW,
        version_id_factory=lambda: "version-1",
    )
    service = MerchantManagementService(
        config_root,
        repository,
        clock=preview_times.__next__,
    )
    service.save_draft("acme_store", _draft(config_root), expected_revision=0)
    request = _publication(service)

    committed = service.publish_draft("acme_store", request)
    replayed = service.publish_draft("acme_store", request)

    assert replayed == committed.model_copy(update={"replayed": True})

    with pytest.raises(MerchantManagementReplayConflictError, match="different parameters"):
        service.publish_draft("acme_store", request.model_copy(update={"actor_id": "operator-2"}))
    with pytest.raises(MerchantManagementConflictError, match="operation expectation"):
        service.publish_draft(
            "acme_store", request.model_copy(update={"request_id": "publish-request-2"})
        )


def test_service_publish_retry_replays_after_the_source_draft_advances(
    tmp_path: Path, config_root: Path
) -> None:
    service = _service(tmp_path, config_root)
    service.save_draft("acme_store", _draft(config_root), expected_revision=0)
    request = _publication(service)
    committed = service.publish_draft("acme_store", request)
    changed_override = _override(config_root)
    changed_override["display_name"] = "Later Draft"
    service.save_draft(
        "acme_store",
        _draft(
            config_root,
            revision=2,
            request_id="draft-request-2",
            override=changed_override,
        ),
        expected_revision=1,
    )

    replayed = service.publish_draft("acme_store", request)

    assert replayed == committed.model_copy(update={"replayed": True})


def test_service_rejects_publication_when_the_approved_resolution_changes(
    tmp_path: Path, config_root: Path
) -> None:
    isolated_config = tmp_path / "config"
    copytree(config_root, isolated_config)
    service = _service(tmp_path / "state", isolated_config)
    service.save_draft("acme_store", _draft(isolated_config), expected_revision=0)
    approved = service.preview_draft("acme_store", "draft-1", expected_revision=1)
    request = _publication(
        expected_preview_fingerprint=approved.preview_fingerprint,
    )
    base_path = isolated_config / "base" / "base.yaml"
    base_path.write_text(
        base_path.read_text(encoding="utf-8").replace(
            "I'm an automated AI assistant.",
            "This call uses an automated assistant.",
        ),
        encoding="utf-8",
    )

    with pytest.raises(MerchantManagementConflictError, match="approved preview"):
        service.publish_draft("acme_store", request)

    with pytest.raises(MerchantManagementNotFoundError, match="no active"):
        service.get_active_version("acme_store")


def test_service_compares_versions_without_returning_changed_values(
    tmp_path: Path, config_root: Path
) -> None:
    service = _service(tmp_path, config_root)
    service.save_draft("acme_store", _draft(config_root), expected_revision=0)
    first = service.publish_draft("acme_store", _publication(service))
    changed_override = _override(config_root)
    changed_override["display_name"] = "Confidential Development Name"
    service.save_draft(
        "acme_store",
        _draft(
            config_root,
            revision=2,
            request_id="draft-request-2",
            override=changed_override,
        ),
        expected_revision=1,
    )
    second = service.publish_draft(
        "acme_store",
        _publication(
            service,
            draft_revision=2,
            request_id="publish-request-2",
            expected_active_version_id=first.version_id,
        ),
    )

    difference = service.compare_versions("acme_store", first.version_id, second.version_id)

    assert difference.changed is True
    assert tuple((change.path, change.kind) for change in difference.config_changes) == (
        (("display_name",), "changed"),
    )
    assert difference.fixture_changes == ()
    assert "Confidential Development Name" not in difference.model_dump_json()
    assert (
        service.compare_versions("acme_store", second.version_id, second.version_id).changed
        is False
    )


def test_service_fixture_diff_never_discloses_dynamic_fixture_keys(
    tmp_path: Path, config_root: Path
) -> None:
    service = _service(tmp_path, config_root)
    first_draft = service.save_draft("acme_store", _draft(config_root), expected_revision=0)
    first = service.publish_draft("acme_store", _publication(service))
    fixture_payload = first_draft.fixtures.model_dump()
    renamed_order = fixture_payload["orders"]["orders"].pop("ORD-1001")
    fixture_payload["orders"]["orders"]["ORD-private@example.com"] = renamed_order
    changed_fixtures = MerchantFixtureBundle.model_validate(fixture_payload)
    service.save_draft(
        "acme_store",
        _draft(
            config_root,
            revision=2,
            request_id="draft-request-2",
            fixtures=changed_fixtures,
        ),
        expected_revision=1,
    )
    second = service.publish_draft(
        "acme_store",
        _publication(
            service,
            draft_revision=2,
            request_id="publish-request-2",
            expected_active_version_id=first.version_id,
        ),
    )

    difference = service.compare_versions("acme_store", first.version_id, second.version_id)

    assert tuple((change.path, change.kind) for change in difference.fixture_changes) == (
        (("orders",), "changed"),
    )
    assert "ORD-private@example.com" not in difference.model_dump_json()


def test_service_version_comparison_is_tenant_scoped(tmp_path: Path, config_root: Path) -> None:
    service = _service(tmp_path, config_root)
    service.save_draft("acme_store", _draft(config_root), expected_revision=0)
    version = service.publish_draft("acme_store", _publication(service))

    with pytest.raises(MerchantManagementNotFoundError, match="does not exist"):
        service.compare_versions("other_store", version.version_id, version.version_id)


def test_service_retirement_is_tenant_scoped_and_preserves_history(
    tmp_path: Path, config_root: Path
) -> None:
    service = _service(tmp_path, config_root)
    service.save_draft("acme_store", _draft(config_root), expected_revision=0)
    published = service.publish_draft("acme_store", _publication(service))
    request = MerchantRetirementRequest(
        tenant_id="acme_store",
        expected_active_version_id=published.version_id,
        actor_id="operator-1",
        request_id="retire-request-1",
    )

    with pytest.raises(MerchantManagementScopeError, match="tenant scope"):
        service.retire("other_store", request)
    retired = service.retire("acme_store", request)

    assert retired.retired_version_id == published.version_id
    with pytest.raises(MerchantManagementNotFoundError, match="no active"):
        service.get_active_version("acme_store")
    assert service.get_version("acme_store", published.version_id).version_number == 1


def test_service_audit_history_is_ordered_tenant_scoped_and_value_free(
    tmp_path: Path, config_root: Path
) -> None:
    service = _service(tmp_path, config_root)
    draft = _draft(config_root)
    service.save_draft("acme_store", draft, expected_revision=0)
    service.save_draft("acme_store", draft, expected_revision=0)
    service.validate_draft("acme_store", _validation())
    published = service.publish_draft("acme_store", _publication(service))
    service.retire(
        "acme_store",
        MerchantRetirementRequest(
            tenant_id="acme_store",
            expected_active_version_id=published.version_id,
            actor_id="operator-1",
            request_id="retire-request-1",
        ),
    )

    history = service.list_audit_history("acme_store")

    assert tuple(record.event for record in history) == (
        "draft_created",
        "draft_validated",
        "version_published",
        "version_retired",
    )
    assert all(record.tenant_id == "acme_store" for record in history)
    assert "secrets://" not in "".join(record.model_dump_json() for record in history)
    assert service.list_audit_history("other_store") == ()


def test_service_rollback_is_tenant_scoped_and_publishes_a_new_active_version(
    tmp_path: Path, config_root: Path
) -> None:
    service = _service(tmp_path, config_root)
    service.save_draft("acme_store", _draft(config_root), expected_revision=0)
    first = service.publish_draft("acme_store", _publication(service))
    changed_override = _override(config_root)
    changed_override["display_name"] = "Updated Development Store"
    service.save_draft(
        "acme_store",
        _draft(
            config_root,
            revision=2,
            request_id="draft-request-2",
            override=changed_override,
        ),
        expected_revision=1,
    )
    second = service.publish_draft(
        "acme_store",
        _publication(
            service,
            draft_revision=2,
            request_id="publish-request-2",
            expected_active_version_id=first.version_id,
        ),
    )
    rollback = MerchantRollbackRequest(
        tenant_id="acme_store",
        expected_active_version_id=second.version_id,
        source_version_id=first.version_id,
        actor_id="operator-2",
        request_id="rollback-request-1",
    )

    with pytest.raises(MerchantManagementScopeError, match="tenant scope"):
        service.rollback("other_store", rollback)
    receipt = service.rollback("acme_store", rollback)
    active = service.get_active_version("acme_store")

    assert receipt.kind == "rollback"
    assert active.version_id == receipt.version_id
    assert active.config.display_name != "Updated Development Store"


def test_draft_override_must_be_json_serializable(config_root: Path) -> None:
    payload = _draft(config_root).model_dump()
    payload["merchant_override"]["unsupported"] = object()

    with pytest.raises(ValidationError):
        MerchantDraft.model_validate(payload)
