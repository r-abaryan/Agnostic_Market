"""Merchant-management DTOs bind existing schemas without duplicating them."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from agnostic_market.commerce.catalog import load_catalog_fixture
from agnostic_market.commerce.identity import load_customers_fixture
from agnostic_market.commerce.orders import load_orders_fixture
from agnostic_market.commerce.payment_instruments import load_payment_instruments_fixture
from agnostic_market.commerce.profile import load_profile_fixture
from agnostic_market.commerce.verification import load_verification_fixture
from agnostic_market.config.registry import ConfigRegistry
from agnostic_market.management.contracts import (
    MerchantAuditRecord,
    MerchantFixtureBundle,
    MerchantPublicationRequest,
    MerchantValidationFinding,
    MerchantValidationResult,
    MerchantVersionChange,
    MerchantVersionDiff,
    PublicationReceipt,
    PublishedMerchantVersion,
    ResolvedMerchantPreview,
    management_contract_schema_fingerprint,
    merchant_fixture_bundle_fingerprint,
    merchant_preview_fingerprint,
)

_NOW = datetime(2026, 9, 19, 12, tzinfo=UTC)


def _bundle(config_root: Path) -> MerchantFixtureBundle:
    return MerchantFixtureBundle(
        catalog=load_catalog_fixture(config_root, "acme_store"),
        orders=load_orders_fixture(config_root, "acme_store"),
        customers=load_customers_fixture(config_root, "acme_store"),
        payment_instruments=load_payment_instruments_fixture(config_root, "acme_store"),
        profiles=load_profile_fixture(config_root, "acme_store"),
        verification=load_verification_fixture(config_root, "acme_store"),
    )


def _preview(config_root: Path) -> ResolvedMerchantPreview:
    resolved = ConfigRegistry(config_root).load().get("acme_store")
    fixtures = _bundle(config_root)
    source_draft_fingerprint = "a" * 64
    fixture_fingerprint = merchant_fixture_bundle_fingerprint(fixtures)
    schema_fingerprint = management_contract_schema_fingerprint()
    return ResolvedMerchantPreview(
        tenant_id="acme_store",
        draft_id="draft-1",
        draft_revision=1,
        resolved_at=_NOW,
        config=resolved.config,
        fixtures=fixtures,
        source_draft_fingerprint=source_draft_fingerprint,
        config_fingerprint=resolved.config_version,
        fixture_fingerprint=fixture_fingerprint,
        schema_fingerprint=schema_fingerprint,
        preview_fingerprint=merchant_preview_fingerprint(
            tenant_id="acme_store",
            draft_id="draft-1",
            draft_revision=1,
            source_draft_fingerprint=source_draft_fingerprint,
            config_fingerprint=resolved.config_version,
            fixture_fingerprint=fixture_fingerprint,
            schema_fingerprint=schema_fingerprint,
        ),
    )


def test_resolved_preview_reuses_the_runtime_config_and_fixture_schemas(config_root: Path) -> None:
    preview = _preview(config_root)

    assert preview.config.merchant_id == "acme_store"
    assert preview.fixtures.catalog.products
    assert preview.fixtures.orders.orders
    assert preview.schema_fingerprint == management_contract_schema_fingerprint()
    assert preview.fixture_fingerprint == merchant_fixture_bundle_fingerprint(preview.fixtures)


def test_management_contracts_reject_unknown_outer_fields(config_root: Path) -> None:
    payload = _preview(config_root).model_dump()
    payload["ui_only_state"] = "must not cross the management boundary"

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ResolvedMerchantPreview.model_validate(payload)


def test_resolved_preview_rejects_payload_fingerprint_drift(config_root: Path) -> None:
    payload = _preview(config_root).model_dump()
    payload["config_fingerprint"] = "0" * 64

    with pytest.raises(ValidationError, match="config fingerprint does not match"):
        ResolvedMerchantPreview.model_validate(payload)


def test_publication_request_cannot_cross_preview_tenants(config_root: Path) -> None:
    with pytest.raises(ValidationError, match="does not match the request tenant"):
        MerchantPublicationRequest(
            tenant_id="another_store",
            actor_id="operator-1",
            request_id="publish-request-1",
            preview=_preview(config_root),
        )


def test_fixture_bundle_rejects_cross_family_reference_drift(config_root: Path) -> None:
    payload = _bundle(config_root).model_dump()
    payload["orders"]["orders"]["ORD-1001"]["customer_ref"] = "CUST-UNKNOWN"

    with pytest.raises(ValidationError, match="inconsistent cross-family references"):
        MerchantFixtureBundle.model_validate(payload)


def test_published_version_is_bound_to_exact_preview_content(config_root: Path) -> None:
    preview = _preview(config_root)
    version = PublishedMerchantVersion(
        tenant_id=preview.tenant_id,
        version_id="version-1",
        version_number=1,
        source_draft_id=preview.draft_id,
        source_draft_revision=preview.draft_revision,
        published_at=_NOW,
        published_by="operator-1",
        config=preview.config,
        fixtures=preview.fixtures,
        config_fingerprint=preview.config_fingerprint,
        fixture_fingerprint=preview.fixture_fingerprint,
        schema_fingerprint=preview.schema_fingerprint,
    )

    assert version.config_fingerprint == preview.config_fingerprint
    with pytest.raises(ValidationError, match="does not match the version tenant"):
        PublishedMerchantVersion.model_validate(
            version.model_dump() | {"tenant_id": "another_store"}
        )


def test_validation_result_derives_validity_from_bounded_findings() -> None:
    result = MerchantValidationResult(
        tenant_id="acme_store",
        draft_id="draft-1",
        draft_revision=1,
        validated_at=_NOW,
        schema_fingerprint=management_contract_schema_fingerprint(),
        findings=(
            MerchantValidationFinding(
                code="fixture_reference_missing",
                path=("orders", "ORD-1", "customer_ref"),
                severity="error",
            ),
        ),
    )

    assert result.valid is False
    with pytest.raises(ValidationError, match="Input should be"):
        MerchantValidationFinding.model_validate(
            {"code": "raw_exception", "path": ["orders"], "severity": "error"}
        )


@pytest.mark.parametrize(
    "payload",
    (
        {"kind": "publish", "source_version_id": "version-old"},
        {"kind": "rollback", "source_version_id": None},
    ),
)
def test_publication_receipt_distinguishes_publish_from_rollback(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError, match="source version"):
        PublicationReceipt.model_validate(
            {
                "tenant_id": "acme_store",
                "request_id": "request-1",
                "version_id": "version-2",
                "version_number": 2,
                "committed_at": _NOW,
                "replayed": False,
                **payload,
            }
        )


def test_audit_record_requires_the_identity_owned_by_its_event() -> None:
    with pytest.raises(ValidationError, match="draft audit events require a draft id"):
        MerchantAuditRecord(
            event_id="event-1",
            event="draft_validated",
            tenant_id="acme_store",
            actor_id="operator-1",
            request_id="request-1",
            occurred_at=_NOW,
            payload_fingerprint="0" * 64,
        )


def test_version_diff_requires_deterministic_unique_paths() -> None:
    with pytest.raises(ValidationError, match="ordered and unique"):
        MerchantVersionDiff(
            tenant_id="acme_store",
            base_version_id="version-1",
            target_version_id="version-2",
            config_changes=(
                MerchantVersionChange(path=("z",), kind="changed"),
                MerchantVersionChange(path=("a",), kind="added"),
            ),
            fixture_changes=(),
            dataset_changes=(),
        )
