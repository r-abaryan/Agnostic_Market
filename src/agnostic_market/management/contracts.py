"""Strict contracts for the merchant configuration management boundary."""

from __future__ import annotations

from datetime import datetime
from functools import cache
from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)

from agnostic_market.commerce.catalog import CatalogFixture
from agnostic_market.commerce.fixture_integrity import assert_fixture_bundle_integrity
from agnostic_market.commerce.identity import CustomersFixture
from agnostic_market.commerce.orders import OrdersFixture
from agnostic_market.commerce.payment_instruments import PaymentInstrumentsFixture
from agnostic_market.commerce.profile import ProfileFixture
from agnostic_market.commerce.verification import VerificationFixture
from agnostic_market.config.loader import ConfigError, config_version
from agnostic_market.dtos.config import MerchantConfig
from agnostic_market.dtos.orchestration import CapabilityId
from agnostic_market.dtos.session import AuthorityIdentifier

_CONTRACT = ConfigDict(extra="forbid", frozen=True, strict=True)
_SHA256 = r"^[0-9a-f]{64}$"

ManagementFindingCode = Literal[
    "configuration_invalid",
    "fixture_invalid",
    "fixture_reference_missing",
    "merchant_identity_mismatch",
    "secret_reference_invalid",
]
ManagementAuditEvent = Literal[
    "draft_created",
    "draft_updated",
    "draft_validated",
    "version_published",
    "version_retired",
    "version_rolled_back",
]
PublicationKind = Literal["publish", "rollback"]
MerchantVersionChangeKind = Literal["added", "removed", "changed"]


def _require_aware(value: datetime, *, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")


class MerchantFixtureBundle(BaseModel):
    """One typed snapshot of every development fixture family for a merchant."""

    model_config = _CONTRACT

    catalog: CatalogFixture
    orders: OrdersFixture
    customers: CustomersFixture
    payment_instruments: PaymentInstrumentsFixture
    profiles: ProfileFixture
    verification: VerificationFixture

    @model_validator(mode="after")
    def validate_cross_family_integrity(self) -> Self:
        try:
            assert_fixture_bundle_integrity(
                orders=self.orders,
                customers=self.customers,
                payment_instruments=self.payment_instruments,
                profiles=self.profiles,
                verification=self.verification,
            )
        except ConfigError as exc:
            raise ValueError("fixture bundle has inconsistent cross-family references") from exc
        return self


def merchant_fixture_bundle_fingerprint(bundle: MerchantFixtureBundle) -> str:
    return config_version(bundle.model_dump(mode="json"))


class MerchantDatasetEntityCounts(BaseModel):
    """Value-free inventory used to audit a synthetic scenario dataset."""

    model_config = _CONTRACT

    catalog_products: int = Field(ge=0)
    orders: int = Field(ge=0)
    customers: int = Field(ge=0)
    payment_instruments: int = Field(ge=0)
    profiles: int = Field(ge=0)
    verification_factors: int = Field(ge=0)


def merchant_dataset_entity_counts(bundle: MerchantFixtureBundle) -> MerchantDatasetEntityCounts:
    return MerchantDatasetEntityCounts(
        catalog_products=len(bundle.catalog.products),
        orders=len(bundle.orders.orders),
        customers=len(bundle.customers.customers),
        payment_instruments=len(bundle.payment_instruments.payment_instruments),
        profiles=len(bundle.profiles.profiles),
        verification_factors=len(bundle.verification.otp_codes_by_factor_ref),
    )


class MerchantDatasetManifest(BaseModel):
    """Versioned provenance and intended coverage for one synthetic fixture bundle."""

    model_config = _CONTRACT

    schema_version: Literal[1] = 1
    dataset_id: AuthorityIdentifier
    tenant_id: AuthorityIdentifier
    revision: int = Field(ge=1)
    generated_at: datetime
    source_kind: Literal[
        "hand_authored_synthetic",
        "generated_synthetic",
        "management_derived",
    ]
    source_ref: AuthorityIdentifier
    entity_counts: MerchantDatasetEntityCounts
    fixture_fingerprint: str = Field(pattern=_SHA256)
    covered_capabilities: tuple[AuthorityIdentifier, ...] = Field(max_length=32)
    scenario_tags: tuple[AuthorityIdentifier, ...] = Field(min_length=1, max_length=32)

    @field_validator("generated_at", mode="before")
    @classmethod
    def parse_json_timestamp(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return value

    @field_validator("covered_capabilities", "scenario_tags", mode="before")
    @classmethod
    def yaml_sequences_are_canonical_tuples(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value

    @field_validator("covered_capabilities")
    @classmethod
    def capabilities_are_current(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        supported = {capability.value for capability in CapabilityId}
        unknown = sorted(set(value) - supported)
        if unknown:
            raise ValueError("dataset manifest contains unknown capability identifiers")
        return value

    @model_validator(mode="after")
    def metadata_is_canonical(self) -> Self:
        _require_aware(self.generated_at, field_name="generated_at")
        for label, values in (
            ("covered capabilities", self.covered_capabilities),
            ("scenario tags", self.scenario_tags),
        ):
            if values != tuple(sorted(values)) or len(values) != len(set(values)):
                raise ValueError(f"dataset manifest {label} must be ordered and unique")
        return self


def merchant_dataset_manifest_fingerprint(manifest: MerchantDatasetManifest) -> str:
    return config_version(manifest.model_dump(mode="json"))


class MerchantScenarioDataset(BaseModel):
    """One importable, tenant-bound synthetic scenario dataset."""

    model_config = _CONTRACT

    schema_version: Literal[1] = 1
    manifest: MerchantDatasetManifest
    fixtures: MerchantFixtureBundle

    @model_validator(mode="after")
    def manifest_matches_fixtures(self) -> Self:
        if self.manifest.fixture_fingerprint != merchant_fixture_bundle_fingerprint(self.fixtures):
            raise ValueError("dataset manifest fingerprint does not match its fixtures")
        if self.manifest.entity_counts != merchant_dataset_entity_counts(self.fixtures):
            raise ValueError("dataset manifest entity counts do not match its fixtures")
        return self


@cache
def management_contract_schema_fingerprint() -> str:
    """Identify the exact schemas accepted at the management boundary."""

    schemas = {
        "merchant_config": MerchantConfig.model_json_schema(),
        "merchant_fixtures": MerchantFixtureBundle.model_json_schema(),
        "merchant_dataset": MerchantScenarioDataset.model_json_schema(),
    }
    return config_version(schemas)


class MerchantDraft(BaseModel):
    """One optimistic-concurrency revision of an unpublished merchant bundle."""

    model_config = _CONTRACT

    schema_version: Literal[1] = 1
    tenant_id: AuthorityIdentifier
    draft_id: AuthorityIdentifier
    revision: int = Field(ge=1)
    actor_id: AuthorityIdentifier
    request_id: AuthorityIdentifier
    created_at: datetime
    updated_at: datetime
    merchant_override: dict[str, JsonValue]
    fixtures: MerchantFixtureBundle
    dataset_manifest: MerchantDatasetManifest | None = None

    @model_validator(mode="after")
    def validate_draft_timestamps(self) -> Self:
        _require_aware(self.created_at, field_name="created_at")
        _require_aware(self.updated_at, field_name="updated_at")
        if self.updated_at < self.created_at:
            raise ValueError("draft updated_at cannot precede created_at")
        if self.dataset_manifest is not None:
            if self.dataset_manifest.tenant_id != self.tenant_id:
                raise ValueError("draft dataset manifest does not match its tenant")
            MerchantScenarioDataset(manifest=self.dataset_manifest, fixtures=self.fixtures)
        return self


def merchant_draft_fingerprint(draft: MerchantDraft) -> str:
    """Bind a preview to the exact stored draft that produced it."""

    return config_version(draft.model_dump(mode="json"))


class MerchantValidationFinding(BaseModel):
    """One bounded validation result, without exposing exception text."""

    model_config = _CONTRACT

    code: ManagementFindingCode
    path: tuple[str, ...] = Field(min_length=1)
    severity: Literal["error", "warning"]


class MerchantValidationResult(BaseModel):
    model_config = _CONTRACT

    schema_version: Literal[1] = 1
    tenant_id: AuthorityIdentifier
    draft_id: AuthorityIdentifier
    draft_revision: int = Field(ge=1)
    validated_at: datetime
    schema_fingerprint: str = Field(pattern=_SHA256)
    findings: tuple[MerchantValidationFinding, ...]

    @model_validator(mode="after")
    def validate_timestamp(self) -> Self:
        _require_aware(self.validated_at, field_name="validated_at")
        return self

    @property
    def valid(self) -> bool:
        return not any(finding.severity == "error" for finding in self.findings)


class MerchantDraftValidationRequest(BaseModel):
    """Validate one stored draft revision under an auditable command identity."""

    model_config = _CONTRACT

    schema_version: Literal[1] = 1
    tenant_id: AuthorityIdentifier
    draft_id: AuthorityIdentifier
    draft_revision: int = Field(ge=1)
    actor_id: AuthorityIdentifier
    request_id: AuthorityIdentifier


class MerchantDraftSeedRequest(BaseModel):
    """Create the first draft from validated active development configuration."""

    model_config = _CONTRACT

    schema_version: Literal[1] = 1
    tenant_id: AuthorityIdentifier
    draft_id: AuthorityIdentifier
    actor_id: AuthorityIdentifier
    request_id: AuthorityIdentifier


class MerchantDatasetImportRequest(BaseModel):
    """Replace a draft's complete validated development dataset atomically."""

    model_config = _CONTRACT

    schema_version: Literal[1] = 1
    tenant_id: AuthorityIdentifier
    draft_id: AuthorityIdentifier
    expected_draft_revision: int = Field(ge=1)
    actor_id: AuthorityIdentifier
    request_id: AuthorityIdentifier
    dataset: MerchantScenarioDataset

    @model_validator(mode="after")
    def dataset_matches_tenant(self) -> Self:
        if self.dataset.manifest.tenant_id != self.tenant_id:
            raise ValueError("imported dataset does not match the request tenant")
        return self


class MerchantCatalogImportRequest(BaseModel):
    """Replace only a draft catalog while preserving its other fixture families."""

    model_config = _CONTRACT

    schema_version: Literal[1] = 1
    tenant_id: AuthorityIdentifier
    draft_id: AuthorityIdentifier
    expected_draft_revision: int = Field(ge=1)
    actor_id: AuthorityIdentifier
    request_id: AuthorityIdentifier
    catalog: CatalogFixture


def merchant_preview_fingerprint(
    *,
    tenant_id: str,
    draft_id: str,
    draft_revision: int,
    source_draft_fingerprint: str,
    config_fingerprint: str,
    fixture_fingerprint: str,
    schema_fingerprint: str,
    dataset_fingerprint: str | None = None,
) -> str:
    """Identify the stable, value-free content of one resolved preview."""

    return config_version(
        {
            "preview_schema_version": 2,
            "tenant_id": tenant_id,
            "draft_id": draft_id,
            "draft_revision": draft_revision,
            "source_draft_fingerprint": source_draft_fingerprint,
            "config_fingerprint": config_fingerprint,
            "fixture_fingerprint": fixture_fingerprint,
            "schema_fingerprint": schema_fingerprint,
            "dataset_fingerprint": dataset_fingerprint,
        }
    )


class ResolvedMerchantPreview(BaseModel):
    """Validated effective state shown before publication."""

    model_config = _CONTRACT

    schema_version: Literal[2] = 2
    tenant_id: AuthorityIdentifier
    draft_id: AuthorityIdentifier
    draft_revision: int = Field(ge=1)
    resolved_at: datetime
    config: MerchantConfig
    fixtures: MerchantFixtureBundle
    dataset_manifest: MerchantDatasetManifest | None = None
    source_draft_fingerprint: str = Field(pattern=_SHA256)
    config_fingerprint: str = Field(pattern=_SHA256)
    fixture_fingerprint: str = Field(pattern=_SHA256)
    schema_fingerprint: str = Field(pattern=_SHA256)
    dataset_fingerprint: str | None = Field(default=None, pattern=_SHA256)
    preview_fingerprint: str = Field(pattern=_SHA256)

    @model_validator(mode="after")
    def validate_preview_binding(self) -> Self:
        _require_aware(self.resolved_at, field_name="resolved_at")
        if self.config.merchant_id != self.tenant_id:
            raise ValueError("resolved config does not match the preview tenant")
        if self.config_fingerprint != config_version(self.config.model_dump(mode="json")):
            raise ValueError("resolved config fingerprint does not match its payload")
        if self.fixture_fingerprint != merchant_fixture_bundle_fingerprint(self.fixtures):
            raise ValueError("fixture fingerprint does not match its payload")
        if (self.dataset_manifest is None) != (self.dataset_fingerprint is None):
            raise ValueError("preview dataset manifest and fingerprint must be present together")
        if self.dataset_manifest is not None:
            if self.dataset_manifest.tenant_id != self.tenant_id:
                raise ValueError("preview dataset manifest does not match its tenant")
            MerchantScenarioDataset(manifest=self.dataset_manifest, fixtures=self.fixtures)
            if self.dataset_fingerprint != merchant_dataset_manifest_fingerprint(
                self.dataset_manifest
            ):
                raise ValueError("preview dataset fingerprint does not match its manifest")
        if self.schema_fingerprint != management_contract_schema_fingerprint():
            raise ValueError("management schema fingerprint is not current")
        if self.preview_fingerprint != merchant_preview_fingerprint(
            tenant_id=self.tenant_id,
            draft_id=self.draft_id,
            draft_revision=self.draft_revision,
            source_draft_fingerprint=self.source_draft_fingerprint,
            config_fingerprint=self.config_fingerprint,
            fixture_fingerprint=self.fixture_fingerprint,
            schema_fingerprint=self.schema_fingerprint,
            dataset_fingerprint=self.dataset_fingerprint,
        ):
            raise ValueError("preview fingerprint does not match its bound content")
        return self


class MerchantPublicationRequest(BaseModel):
    """Publish one exact validated preview against an expected active version."""

    model_config = _CONTRACT

    schema_version: Literal[2] = 2
    tenant_id: AuthorityIdentifier
    expected_active_version_id: AuthorityIdentifier | None = None
    actor_id: AuthorityIdentifier
    request_id: AuthorityIdentifier
    preview: ResolvedMerchantPreview

    @model_validator(mode="after")
    def validate_preview_tenant(self) -> Self:
        if self.preview.tenant_id != self.tenant_id:
            raise ValueError("publication preview does not match the request tenant")
        return self


class MerchantDraftPublicationRequest(BaseModel):
    """Request that the service resolve and publish one stored draft revision."""

    model_config = _CONTRACT

    schema_version: Literal[2] = 2
    tenant_id: AuthorityIdentifier
    draft_id: AuthorityIdentifier
    draft_revision: int = Field(ge=1)
    expected_preview_fingerprint: str = Field(pattern=_SHA256)
    expected_active_version_id: AuthorityIdentifier | None = None
    actor_id: AuthorityIdentifier
    request_id: AuthorityIdentifier


def merchant_publication_intent_fingerprint(
    *,
    tenant_id: str,
    draft_id: str,
    draft_revision: int,
    expected_preview_fingerprint: str,
    expected_active_version_id: str | None,
    actor_id: str,
    request_id: str,
) -> str:
    """Identify the stable client-approved publication command."""

    return config_version(
        {
            "kind": "publish",
            "request_schema_version": 2,
            "tenant_id": tenant_id,
            "draft_id": draft_id,
            "draft_revision": draft_revision,
            "expected_preview_fingerprint": expected_preview_fingerprint,
            "expected_active_version_id": expected_active_version_id,
            "actor_id": actor_id,
            "request_id": request_id,
        }
    )


class PublishedMerchantVersion(BaseModel):
    """Content-bound snapshot that repositories must store as an immutable version."""

    model_config = _CONTRACT

    schema_version: Literal[1] = 1
    tenant_id: AuthorityIdentifier
    version_id: AuthorityIdentifier
    version_number: int = Field(ge=1)
    previous_version_id: AuthorityIdentifier | None = None
    source_draft_id: AuthorityIdentifier
    source_draft_revision: int = Field(ge=1)
    published_at: datetime
    published_by: AuthorityIdentifier
    config: MerchantConfig
    fixtures: MerchantFixtureBundle
    dataset_manifest: MerchantDatasetManifest | None = None
    config_fingerprint: str = Field(pattern=_SHA256)
    fixture_fingerprint: str = Field(pattern=_SHA256)
    schema_fingerprint: str = Field(pattern=_SHA256)
    dataset_fingerprint: str | None = Field(default=None, pattern=_SHA256)

    @model_validator(mode="after")
    def validate_published_binding(self) -> Self:
        _require_aware(self.published_at, field_name="published_at")
        if self.config.merchant_id != self.tenant_id:
            raise ValueError("published config does not match the version tenant")
        if self.config_fingerprint != config_version(self.config.model_dump(mode="json")):
            raise ValueError("published config fingerprint does not match its payload")
        if self.fixture_fingerprint != merchant_fixture_bundle_fingerprint(self.fixtures):
            raise ValueError("published fixture fingerprint does not match its payload")
        if (self.dataset_manifest is None) != (self.dataset_fingerprint is None):
            raise ValueError("published dataset manifest and fingerprint must be present together")
        if self.dataset_manifest is not None:
            if self.dataset_manifest.tenant_id != self.tenant_id:
                raise ValueError("published dataset manifest does not match its tenant")
            MerchantScenarioDataset(manifest=self.dataset_manifest, fixtures=self.fixtures)
            if self.dataset_fingerprint != merchant_dataset_manifest_fingerprint(
                self.dataset_manifest
            ):
                raise ValueError("published dataset fingerprint does not match its manifest")
        if self.schema_fingerprint != management_contract_schema_fingerprint():
            raise ValueError("published management schema fingerprint is not current")
        return self


def published_merchant_runtime_version(version: PublishedMerchantVersion) -> str:
    """Derive the session pin for one complete immutable publication."""

    return config_version(
        {
            "tenant_id": version.tenant_id,
            "version_id": version.version_id,
            "config_fingerprint": version.config_fingerprint,
            "fixture_fingerprint": version.fixture_fingerprint,
            "schema_fingerprint": version.schema_fingerprint,
            "dataset_fingerprint": version.dataset_fingerprint,
        }
    )


class MerchantVersionChange(BaseModel):
    """One value-free path change between two immutable versions."""

    model_config = _CONTRACT

    path: tuple[str, ...] = Field(min_length=1)
    kind: MerchantVersionChangeKind


class MerchantVersionDiff(BaseModel):
    """Deterministic comparison that never returns configuration or fixture values."""

    model_config = _CONTRACT

    schema_version: Literal[1] = 1
    tenant_id: AuthorityIdentifier
    base_version_id: AuthorityIdentifier
    target_version_id: AuthorityIdentifier
    config_changes: tuple[MerchantVersionChange, ...]
    fixture_changes: tuple[MerchantVersionChange, ...]
    dataset_changes: tuple[MerchantVersionChange, ...]

    @model_validator(mode="after")
    def validate_ordered_unique_paths(self) -> Self:
        for label, changes in (
            ("config", self.config_changes),
            ("fixture", self.fixture_changes),
            ("dataset", self.dataset_changes),
        ):
            paths = tuple(change.path for change in changes)
            if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
                raise ValueError(f"{label} change paths must be ordered and unique")
        return self

    @property
    def changed(self) -> bool:
        return bool(self.config_changes or self.fixture_changes or self.dataset_changes)


class PublicationReceipt(BaseModel):
    model_config = _CONTRACT

    schema_version: Literal[1] = 1
    kind: PublicationKind
    tenant_id: AuthorityIdentifier
    request_id: AuthorityIdentifier
    version_id: AuthorityIdentifier
    version_number: int = Field(ge=1)
    source_version_id: AuthorityIdentifier | None = None
    committed_at: datetime
    replayed: bool

    @model_validator(mode="after")
    def validate_publication(self) -> Self:
        _require_aware(self.committed_at, field_name="committed_at")
        if (self.kind == "rollback") != (self.source_version_id is not None):
            raise ValueError("only rollback publication receipts require a source version")
        return self


class MerchantRetirementRequest(BaseModel):
    model_config = _CONTRACT

    schema_version: Literal[1] = 1
    tenant_id: AuthorityIdentifier
    expected_active_version_id: AuthorityIdentifier
    actor_id: AuthorityIdentifier
    request_id: AuthorityIdentifier


class MerchantRetirementReceipt(BaseModel):
    model_config = _CONTRACT

    schema_version: Literal[1] = 1
    tenant_id: AuthorityIdentifier
    retired_version_id: AuthorityIdentifier
    request_id: AuthorityIdentifier
    retired_at: datetime
    replayed: bool

    @model_validator(mode="after")
    def validate_timestamp(self) -> Self:
        _require_aware(self.retired_at, field_name="retired_at")
        return self


class MerchantRollbackRequest(BaseModel):
    """Request publication of a new version derived from a prior snapshot."""

    model_config = _CONTRACT

    schema_version: Literal[1] = 1
    tenant_id: AuthorityIdentifier
    expected_active_version_id: AuthorityIdentifier
    source_version_id: AuthorityIdentifier
    actor_id: AuthorityIdentifier
    request_id: AuthorityIdentifier


class MerchantAuditRecord(BaseModel):
    model_config = _CONTRACT

    schema_version: Literal[1] = 1
    event_id: AuthorityIdentifier
    event: ManagementAuditEvent
    tenant_id: AuthorityIdentifier
    actor_id: AuthorityIdentifier
    request_id: AuthorityIdentifier
    occurred_at: datetime
    draft_id: AuthorityIdentifier | None = None
    version_id: AuthorityIdentifier | None = None
    payload_fingerprint: str = Field(pattern=_SHA256)

    @model_validator(mode="after")
    def validate_audit_binding(self) -> Self:
        _require_aware(self.occurred_at, field_name="occurred_at")
        if self.event.startswith("draft_") and self.draft_id is None:
            raise ValueError("draft audit events require a draft id")
        if self.event.startswith("version_") and self.version_id is None:
            raise ValueError("version audit events require a version id")
        return self
