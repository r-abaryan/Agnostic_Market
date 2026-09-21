"""Tenant-scoped application service for merchant configuration management."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from pydantic import JsonValue, ValidationError

from agnostic_market.commerce.catalog import load_catalog_fixture
from agnostic_market.commerce.identity import load_customers_fixture
from agnostic_market.commerce.orders import load_orders_fixture
from agnostic_market.commerce.payment_instruments import load_payment_instruments_fixture
from agnostic_market.commerce.profile import load_profile_fixture
from agnostic_market.commerce.verification import load_verification_fixture
from agnostic_market.config.loader import ConfigError
from agnostic_market.config.registry import (
    ConfigRegistry,
    ResolvedConfig,
    UnknownMerchantError,
    resolve_merchant_override,
)
from agnostic_market.config.resolver import (
    ConfigResolutionError,
    PolicyBoundsViolationError,
    SafetyLockViolationError,
)
from agnostic_market.management.contracts import (
    MerchantAuditRecord,
    MerchantCatalogImportRequest,
    MerchantDatasetImportRequest,
    MerchantDatasetManifest,
    MerchantDraft,
    MerchantDraftPublicationRequest,
    MerchantDraftSeedRequest,
    MerchantDraftValidationRequest,
    MerchantFixtureBundle,
    MerchantPublicationRequest,
    MerchantRetirementReceipt,
    MerchantRetirementRequest,
    MerchantRollbackRequest,
    MerchantScenarioDataset,
    MerchantValidationFinding,
    MerchantValidationResult,
    MerchantVersionChange,
    MerchantVersionDiff,
    PublicationReceipt,
    PublishedMerchantVersion,
    ResolvedMerchantPreview,
    management_contract_schema_fingerprint,
    merchant_dataset_entity_counts,
    merchant_dataset_manifest_fingerprint,
    merchant_draft_fingerprint,
    merchant_fixture_bundle_fingerprint,
    merchant_preview_fingerprint,
    merchant_publication_intent_fingerprint,
)
from agnostic_market.management.repository import (
    ManagementRepositoryConflictError,
    ManagementRepositoryDataError,
    ManagementRepositoryError,
    ManagementRepositoryNotFoundError,
    ManagementRepositoryReplayConflictError,
    MerchantConfigurationRepository,
)
from agnostic_market.secrets.reference import SecretReference

type Clock = Callable[[], datetime]


class MerchantManagementError(RuntimeError):
    """Base error for the merchant management application boundary."""


class MerchantManagementScopeError(MerchantManagementError):
    """An operation attempted to cross its explicit tenant scope."""


class MerchantManagementNotFoundError(MerchantManagementError):
    """A tenant-scoped draft or version does not exist."""


class MerchantManagementConflictError(MerchantManagementError):
    """A requested stored revision no longer matches current state."""


class MerchantManagementReplayConflictError(MerchantManagementError):
    """A management request id was reused with different parameters."""


class MerchantManagementValidationError(MerchantManagementError):
    """A draft failed bounded management validation."""

    def __init__(self, result: MerchantValidationResult) -> None:
        super().__init__("merchant draft did not pass validation")
        self.result = result


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _repository_call[ResultT](operation: Callable[[], ResultT]) -> ResultT:
    try:
        return operation()
    except ManagementRepositoryReplayConflictError as exc:
        raise MerchantManagementReplayConflictError(
            "management request id was reused with different parameters"
        ) from exc
    except ManagementRepositoryConflictError as exc:
        raise MerchantManagementConflictError(
            "management state no longer matches the operation expectation"
        ) from exc
    except ManagementRepositoryNotFoundError as exc:
        raise MerchantManagementNotFoundError("management source record does not exist") from exc
    except ManagementRepositoryDataError as exc:
        raise MerchantManagementError("management repository data is invalid") from exc
    except ManagementRepositoryError as exc:
        raise MerchantManagementError("management repository operation failed") from exc


def _configuration_finding() -> MerchantValidationFinding:
    return MerchantValidationFinding(
        code="configuration_invalid",
        path=("merchant_override",),
        severity="error",
    )


def _secret_reference_findings(config: ResolvedConfig) -> tuple[MerchantValidationFinding, ...]:
    references = (
        (("secrets_ref",), config.config.secrets_ref),
        (("integration", "order_sor", "ref"), config.config.integration.order_sor.ref),
    )
    findings: list[MerchantValidationFinding] = []
    for path, reference in references:
        try:
            SecretReference.from_uri(reference)
        except (ValueError, ValidationError):
            findings.append(
                MerchantValidationFinding(
                    code="secret_reference_invalid",
                    path=path,
                    severity="error",
                )
            )
    return tuple(findings)


_MISSING = object()


def _value_free_changes(
    base: object,
    target: object,
    *,
    path: tuple[str, ...] = (),
) -> tuple[MerchantVersionChange, ...]:
    if isinstance(base, dict) and isinstance(target, dict):
        changes: list[MerchantVersionChange] = []
        for key in sorted(base.keys() | target.keys()):
            base_value = base.get(key, _MISSING)
            target_value = target.get(key, _MISSING)
            child_path = (*path, key)
            if base_value is _MISSING:
                changes.append(MerchantVersionChange(path=child_path, kind="added"))
            elif target_value is _MISSING:
                changes.append(MerchantVersionChange(path=child_path, kind="removed"))
            else:
                changes.extend(_value_free_changes(base_value, target_value, path=child_path))
        return tuple(changes)
    if base == target:
        return ()
    return (MerchantVersionChange(path=path, kind="changed"),)


def _fixture_family_changes(
    base: MerchantFixtureBundle,
    target: MerchantFixtureBundle,
) -> tuple[MerchantVersionChange, ...]:
    return tuple(
        MerchantVersionChange(path=(family,), kind="changed")
        for family in sorted(MerchantFixtureBundle.model_fields)
        if getattr(base, family) != getattr(target, family)
    )


def _dataset_manifest_changes(
    base: MerchantDatasetManifest | None,
    target: MerchantDatasetManifest | None,
) -> tuple[MerchantVersionChange, ...]:
    if base is None and target is None:
        return ()
    if base is None:
        return (MerchantVersionChange(path=("manifest",), kind="added"),)
    if target is None:
        return (MerchantVersionChange(path=("manifest",), kind="removed"),)
    return _value_free_changes(
        base.model_dump(mode="json"),
        target.model_dump(mode="json"),
        path=("manifest",),
    )


class MerchantManagementService:
    """Resolve, validate, preview, and publish through one tenant-scoped boundary."""

    def __init__(
        self,
        config_root: Path,
        repository: MerchantConfigurationRepository,
        *,
        clock: Clock = _utc_now,
    ) -> None:
        self._config_root = config_root
        self._repository = repository
        self._clock = clock

    def list_tenant_ids(self) -> tuple[str, ...]:
        return _repository_call(self._repository.list_tenant_ids)

    def list_configured_tenant_ids(self) -> tuple[str, ...]:
        """Return validated active development merchants visible to the workbench."""

        try:
            configured = ConfigRegistry(self._config_root).load().merchant_ids
        except ConfigError as exc:
            raise MerchantManagementError("active merchant configuration is invalid") from exc
        return tuple(sorted(configured))

    def list_audit_history(self, tenant_id: str) -> tuple[MerchantAuditRecord, ...]:
        records = _repository_call(lambda: self._repository.list_audit_records(tenant_id))
        for record in records:
            self._require_scope(tenant_id, record.tenant_id)
        return records

    @staticmethod
    def _require_scope(scope_tenant_id: str, object_tenant_id: str) -> None:
        if scope_tenant_id != object_tenant_id:
            raise MerchantManagementScopeError("management operation crossed its tenant scope")

    def save_draft(
        self,
        scope_tenant_id: str,
        draft: MerchantDraft,
        *,
        expected_revision: int,
    ) -> MerchantDraft:
        self._require_scope(scope_tenant_id, draft.tenant_id)
        return _repository_call(
            lambda: self._repository.save_draft(draft, expected_revision=expected_revision)
        )

    def write_draft(
        self,
        scope_tenant_id: str,
        *,
        draft_id: str,
        revision: int,
        actor_id: str,
        request_id: str,
        merchant_override: dict[str, JsonValue],
        fixtures: MerchantFixtureBundle,
        dataset_manifest: MerchantDatasetManifest | None,
        expected_revision: int,
    ) -> MerchantDraft:
        """Store editable draft content with service-owned timestamps."""

        existing = _repository_call(lambda: self._repository.get_draft(scope_tenant_id, draft_id))
        now = self._clock()
        created_at = now if existing is None else existing.created_at
        updated_at = (
            existing.updated_at if existing is not None and existing.revision == revision else now
        )
        return self.save_draft(
            scope_tenant_id,
            MerchantDraft(
                tenant_id=scope_tenant_id,
                draft_id=draft_id,
                revision=revision,
                actor_id=actor_id,
                request_id=request_id,
                created_at=created_at,
                updated_at=updated_at,
                merchant_override=merchant_override,
                fixtures=fixtures,
                dataset_manifest=dataset_manifest,
            ),
            expected_revision=expected_revision,
        )

    def seed_draft(
        self,
        scope_tenant_id: str,
        request: MerchantDraftSeedRequest,
    ) -> MerchantDraft:
        """Copy validated development configuration into the first managed revision."""

        self._require_scope(scope_tenant_id, request.tenant_id)
        existing = _repository_call(
            lambda: self._repository.get_draft(request.tenant_id, request.draft_id)
        )
        if existing is not None:
            if (
                existing.revision == 1
                and existing.actor_id == request.actor_id
                and existing.request_id == request.request_id
            ):
                return existing
            raise MerchantManagementConflictError("merchant draft already exists")
        try:
            registry = ConfigRegistry(self._config_root).load()
            merchant_override = registry.source_override(request.tenant_id)
            fixtures = MerchantFixtureBundle(
                catalog=load_catalog_fixture(self._config_root, request.tenant_id),
                orders=load_orders_fixture(self._config_root, request.tenant_id),
                customers=load_customers_fixture(self._config_root, request.tenant_id),
                payment_instruments=load_payment_instruments_fixture(
                    self._config_root, request.tenant_id
                ),
                profiles=load_profile_fixture(self._config_root, request.tenant_id),
                verification=load_verification_fixture(self._config_root, request.tenant_id),
            )
        except UnknownMerchantError as exc:
            raise MerchantManagementNotFoundError("configured merchant does not exist") from exc
        except (ConfigError, ValidationError) as exc:
            raise MerchantManagementError(
                "active development configuration cannot seed a draft"
            ) from exc
        now = self._clock()
        return self.save_draft(
            scope_tenant_id,
            MerchantDraft(
                tenant_id=request.tenant_id,
                draft_id=request.draft_id,
                revision=1,
                actor_id=request.actor_id,
                request_id=request.request_id,
                created_at=now,
                updated_at=now,
                merchant_override=merchant_override,
                fixtures=fixtures,
            ),
            expected_revision=0,
        )

    def import_dataset(
        self,
        scope_tenant_id: str,
        request: MerchantDatasetImportRequest,
    ) -> MerchantDraft:
        self._require_scope(scope_tenant_id, request.tenant_id)
        draft = self.get_draft(request.tenant_id, request.draft_id)
        if draft.revision == request.expected_draft_revision + 1:
            if draft.request_id == request.request_id:
                if (
                    draft.actor_id == request.actor_id
                    and draft.fixtures == request.dataset.fixtures
                    and draft.dataset_manifest == request.dataset.manifest
                ):
                    return draft
                raise MerchantManagementReplayConflictError(
                    "management request id was reused with different parameters"
                )
            raise MerchantManagementConflictError(
                "merchant draft no longer matches the expected revision"
            )
        if draft.revision != request.expected_draft_revision:
            raise MerchantManagementConflictError(
                "merchant draft no longer matches the expected revision"
            )
        if (
            draft.fixtures == request.dataset.fixtures
            and draft.dataset_manifest == request.dataset.manifest
        ):
            raise MerchantManagementConflictError("dataset import does not change the draft")
        updated = MerchantDraft(
            tenant_id=draft.tenant_id,
            draft_id=draft.draft_id,
            revision=draft.revision + 1,
            actor_id=request.actor_id,
            request_id=request.request_id,
            created_at=draft.created_at,
            updated_at=self._clock(),
            merchant_override=draft.merchant_override,
            fixtures=request.dataset.fixtures,
            dataset_manifest=request.dataset.manifest,
        )
        return _repository_call(
            lambda: self._repository.save_draft(
                updated,
                expected_revision=request.expected_draft_revision,
            )
        )

    def import_catalog(
        self,
        scope_tenant_id: str,
        request: MerchantCatalogImportRequest,
    ) -> MerchantDraft:
        """Replace one catalog through the same atomic draft-revision boundary."""

        self._require_scope(scope_tenant_id, request.tenant_id)
        draft = self.get_draft(request.tenant_id, request.draft_id)
        if (
            draft.revision == request.expected_draft_revision + 1
            and draft.request_id == request.request_id
        ):
            if (
                draft.actor_id == request.actor_id
                and draft.fixtures.catalog == request.catalog
                and draft.dataset_manifest is not None
                and draft.dataset_manifest.source_kind == "management_derived"
                and draft.dataset_manifest.source_ref == request.request_id
            ):
                return draft
            raise MerchantManagementReplayConflictError(
                "management request id was reused with different parameters"
            )
        fixtures = draft.fixtures.model_copy(update={"catalog": request.catalog})
        dataset_manifest = None
        if draft.dataset_manifest is not None:
            dataset_manifest = MerchantDatasetManifest(
                dataset_id=draft.dataset_manifest.dataset_id,
                tenant_id=draft.tenant_id,
                revision=draft.dataset_manifest.revision + 1,
                generated_at=self._clock(),
                source_kind="management_derived",
                source_ref=request.request_id,
                entity_counts=merchant_dataset_entity_counts(fixtures),
                fixture_fingerprint=merchant_fixture_bundle_fingerprint(fixtures),
                covered_capabilities=draft.dataset_manifest.covered_capabilities,
                scenario_tags=draft.dataset_manifest.scenario_tags,
            )
        if dataset_manifest is None:
            dataset_manifest = MerchantDatasetManifest(
                dataset_id=f"{request.tenant_id}-catalog-import",
                tenant_id=request.tenant_id,
                revision=1,
                generated_at=self._clock(),
                source_kind="management_derived",
                source_ref=request.request_id,
                entity_counts=merchant_dataset_entity_counts(fixtures),
                fixture_fingerprint=merchant_fixture_bundle_fingerprint(fixtures),
                covered_capabilities=("search_catalog",),
                scenario_tags=("catalog",),
            )
        return self.import_dataset(
            scope_tenant_id,
            MerchantDatasetImportRequest(
                tenant_id=request.tenant_id,
                draft_id=request.draft_id,
                expected_draft_revision=request.expected_draft_revision,
                actor_id=request.actor_id,
                request_id=request.request_id,
                dataset=MerchantScenarioDataset(
                    manifest=dataset_manifest,
                    fixtures=fixtures,
                ),
            ),
        )

    def get_draft(self, tenant_id: str, draft_id: str) -> MerchantDraft:
        draft = _repository_call(lambda: self._repository.get_draft(tenant_id, draft_id))
        if draft is None:
            raise MerchantManagementNotFoundError("merchant draft does not exist")
        self._require_scope(tenant_id, draft.tenant_id)
        return draft

    def _current_draft(
        self,
        tenant_id: str,
        draft_id: str,
        expected_revision: int,
    ) -> MerchantDraft:
        draft = self.get_draft(tenant_id, draft_id)
        if draft.revision != expected_revision:
            raise MerchantManagementConflictError(
                "merchant draft no longer matches the expected revision"
            )
        return draft

    def _inspect_draft(
        self,
        draft: MerchantDraft,
    ) -> tuple[MerchantValidationResult, ResolvedConfig | None]:
        findings: list[MerchantValidationFinding] = []
        resolved: ResolvedConfig | None = None
        try:
            resolved = resolve_merchant_override(
                self._config_root,
                draft.merchant_override,
                source=f"merchant draft {draft.draft_id}",
            )
        except (
            ConfigError,
            ConfigResolutionError,
            PolicyBoundsViolationError,
            SafetyLockViolationError,
        ):
            findings.append(_configuration_finding())
        if resolved is not None:
            if resolved.config.merchant_id != draft.tenant_id:
                findings.append(
                    MerchantValidationFinding(
                        code="merchant_identity_mismatch",
                        path=("merchant_override", "merchant_id"),
                        severity="error",
                    )
                )
            findings.extend(_secret_reference_findings(resolved))
        result = MerchantValidationResult(
            tenant_id=draft.tenant_id,
            draft_id=draft.draft_id,
            draft_revision=draft.revision,
            validated_at=self._clock(),
            schema_fingerprint=management_contract_schema_fingerprint(),
            findings=tuple(findings),
        )
        return result, resolved

    def validate_draft(
        self,
        scope_tenant_id: str,
        request: MerchantDraftValidationRequest,
    ) -> MerchantValidationResult:
        self._require_scope(scope_tenant_id, request.tenant_id)
        draft = self._current_draft(
            request.tenant_id,
            request.draft_id,
            request.draft_revision,
        )
        result, _ = self._inspect_draft(draft)
        return _repository_call(lambda: self._repository.record_validation(request, result))

    def preview_draft(
        self,
        tenant_id: str,
        draft_id: str,
        *,
        expected_revision: int,
    ) -> ResolvedMerchantPreview:
        draft = self._current_draft(tenant_id, draft_id, expected_revision)
        validation, resolved = self._inspect_draft(draft)
        if not validation.valid or resolved is None:
            raise MerchantManagementValidationError(validation)
        source_draft_fingerprint = merchant_draft_fingerprint(draft)
        fixture_fingerprint = merchant_fixture_bundle_fingerprint(draft.fixtures)
        dataset_fingerprint = (
            merchant_dataset_manifest_fingerprint(draft.dataset_manifest)
            if draft.dataset_manifest is not None
            else None
        )
        preview_fingerprint = merchant_preview_fingerprint(
            tenant_id=draft.tenant_id,
            draft_id=draft.draft_id,
            draft_revision=draft.revision,
            source_draft_fingerprint=source_draft_fingerprint,
            config_fingerprint=resolved.config_version,
            fixture_fingerprint=fixture_fingerprint,
            schema_fingerprint=validation.schema_fingerprint,
            dataset_fingerprint=dataset_fingerprint,
        )
        return ResolvedMerchantPreview(
            tenant_id=draft.tenant_id,
            draft_id=draft.draft_id,
            draft_revision=draft.revision,
            resolved_at=validation.validated_at,
            config=resolved.config,
            fixtures=draft.fixtures,
            dataset_manifest=draft.dataset_manifest,
            source_draft_fingerprint=source_draft_fingerprint,
            config_fingerprint=resolved.config_version,
            fixture_fingerprint=fixture_fingerprint,
            schema_fingerprint=validation.schema_fingerprint,
            dataset_fingerprint=dataset_fingerprint,
            preview_fingerprint=preview_fingerprint,
        )

    def publish_draft(
        self,
        scope_tenant_id: str,
        request: MerchantDraftPublicationRequest,
    ) -> PublicationReceipt:
        self._require_scope(scope_tenant_id, request.tenant_id)
        request_fingerprint = merchant_publication_intent_fingerprint(
            tenant_id=request.tenant_id,
            draft_id=request.draft_id,
            draft_revision=request.draft_revision,
            expected_preview_fingerprint=request.expected_preview_fingerprint,
            expected_active_version_id=request.expected_active_version_id,
            actor_id=request.actor_id,
            request_id=request.request_id,
        )
        replay = _repository_call(
            lambda: self._repository.replay_publication(
                tenant_id=request.tenant_id,
                request_id=request.request_id,
                request_fingerprint=request_fingerprint,
            )
        )
        if replay is not None:
            return replay
        preview = self.preview_draft(
            request.tenant_id,
            request.draft_id,
            expected_revision=request.draft_revision,
        )
        if preview.preview_fingerprint != request.expected_preview_fingerprint:
            raise MerchantManagementConflictError(
                "resolved merchant state no longer matches the approved preview"
            )
        return _repository_call(
            lambda: self._repository.publish(
                MerchantPublicationRequest(
                    tenant_id=request.tenant_id,
                    expected_active_version_id=request.expected_active_version_id,
                    actor_id=request.actor_id,
                    request_id=request.request_id,
                    preview=preview,
                )
            )
        )

    def get_active_version(self, tenant_id: str) -> PublishedMerchantVersion:
        version = _repository_call(lambda: self._repository.get_active_version(tenant_id))
        if version is None:
            raise MerchantManagementNotFoundError("merchant has no active published version")
        self._require_scope(tenant_id, version.tenant_id)
        return version

    def get_version(self, tenant_id: str, version_id: str) -> PublishedMerchantVersion:
        version = _repository_call(lambda: self._repository.get_version(tenant_id, version_id))
        if version is None:
            raise MerchantManagementNotFoundError("merchant version does not exist")
        self._require_scope(tenant_id, version.tenant_id)
        return version

    def list_versions(self, tenant_id: str) -> tuple[PublishedMerchantVersion, ...]:
        versions = _repository_call(lambda: self._repository.list_versions(tenant_id))
        for version in versions:
            self._require_scope(tenant_id, version.tenant_id)
        return versions

    def compare_versions(
        self,
        tenant_id: str,
        base_version_id: str,
        target_version_id: str,
    ) -> MerchantVersionDiff:
        base = self.get_version(tenant_id, base_version_id)
        target = self.get_version(tenant_id, target_version_id)
        return MerchantVersionDiff(
            tenant_id=tenant_id,
            base_version_id=base.version_id,
            target_version_id=target.version_id,
            config_changes=_value_free_changes(
                base.config.model_dump(mode="json"),
                target.config.model_dump(mode="json"),
            ),
            fixture_changes=_fixture_family_changes(base.fixtures, target.fixtures),
            dataset_changes=_dataset_manifest_changes(
                base.dataset_manifest,
                target.dataset_manifest,
            ),
        )

    def retire(
        self,
        scope_tenant_id: str,
        request: MerchantRetirementRequest,
    ) -> MerchantRetirementReceipt:
        self._require_scope(scope_tenant_id, request.tenant_id)
        return _repository_call(lambda: self._repository.retire(request))

    def rollback(
        self,
        scope_tenant_id: str,
        request: MerchantRollbackRequest,
    ) -> PublicationReceipt:
        self._require_scope(scope_tenant_id, request.tenant_id)
        return _repository_call(lambda: self._repository.rollback(request))
