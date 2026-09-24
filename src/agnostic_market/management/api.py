"""Thin loopback administration API over the merchant management service."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path as FileSystemPath
from typing import Annotated, Literal

from fastapi import Body, FastAPI, Path, Query, Request
from fastapi.exceptions import RequestValidationError, ResponseValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, ValidationError

from agnostic_market.commerce.catalog import CatalogFixture, CatalogProduct
from agnostic_market.dtos.config import VoiceConfig
from agnostic_market.dtos.session import AuthorityIdentifier
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
    MerchantRetirementReceipt,
    MerchantRetirementRequest,
    MerchantRollbackRequest,
    MerchantScenarioDataset,
    MerchantValidationResult,
    MerchantVersionChange,
    PublicationReceipt,
    PublishedMerchantVersion,
    ResolvedMerchantPreview,
)
from agnostic_market.management.service import (
    MerchantManagementConflictError,
    MerchantManagementError,
    MerchantManagementNotFoundError,
    MerchantManagementReplayConflictError,
    MerchantManagementScopeError,
    MerchantManagementService,
    MerchantManagementValidationError,
)
from agnostic_market.management.simulation import (
    MerchantSimulationConflictError,
    MerchantSimulationError,
    MerchantSimulationNotFoundError,
    MerchantSimulationReplayConflictError,
    PublishedMerchantSimulator,
    SimulationSessionStatus,
    SimulationStateProjection,
    SimulationTurnResult,
)
from agnostic_market.management.voice_preview import (
    CAPTURE_SAMPLE_RATE,
    MAX_CAPTURE_SECONDS,
    MAX_SPEECH_CHARS,
    VoicePreview,
)

_TRANSPORT = ConfigDict(extra="forbid", frozen=True)
_AUTHORITY = TypeAdapter(AuthorityIdentifier)
_UI_ROOT = FileSystemPath(__file__).with_name("ui")
_UI_SECURITY_HEADERS = {
    "Cache-Control": "no-store",
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; "
        "img-src 'self' data:; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; "
        "form-action 'self'"
    ),
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
}
_TenantPath = Annotated[str, Path(min_length=1, pattern=r"^\S+$")]
_DraftPath = Annotated[str, Path(min_length=1, pattern=r"^\S+$")]
_VersionPath = Annotated[str, Path(min_length=1, pattern=r"^\S+$")]
_SimulationPath = Annotated[str, Path(min_length=1, pattern=r"^\S+$")]

ManagementApiErrorCode = Literal[
    "invalid_request",
    "not_found",
    "scope_violation",
    "state_conflict",
    "replay_conflict",
    "draft_invalid",
    "service_unavailable",
]


class ManagementApiError(BaseModel):
    """Bounded failure payload that never reflects request or exception text."""

    model_config = _TRANSPORT

    schema_version: Literal[1] = 1
    code: ManagementApiErrorCode
    validation: MerchantValidationResult | None = None


class MerchantSummary(BaseModel):
    model_config = _TRANSPORT

    schema_version: Literal[1] = 1
    tenant_id: AuthorityIdentifier
    configured: bool
    managed: bool


class MerchantListResponse(BaseModel):
    model_config = _TRANSPORT

    schema_version: Literal[1] = 1
    merchants: tuple[MerchantSummary, ...]


class MerchantVersionListResponse(BaseModel):
    model_config = _TRANSPORT

    schema_version: Literal[1] = 1
    versions: tuple[PublishedMerchantVersion, ...]


class MerchantAuditListResponse(BaseModel):
    model_config = _TRANSPORT

    schema_version: Literal[1] = 1
    records: tuple[MerchantAuditRecord, ...]


class MerchantCatalogPage(BaseModel):
    model_config = _TRANSPORT

    schema_version: Literal[1] = 1
    offset: int = Field(ge=0)
    limit: int = Field(ge=1, le=100)
    total: int = Field(ge=0)
    products: tuple[CatalogProduct, ...]


class MerchantVersionDiffResponse(BaseModel):
    model_config = _TRANSPORT

    schema_version: Literal[1] = 1
    tenant_id: AuthorityIdentifier
    base_version_id: AuthorityIdentifier
    target_version_id: AuthorityIdentifier
    config_changes: tuple[MerchantVersionChange, ...]
    fixture_changes: tuple[MerchantVersionChange, ...]
    dataset_changes: tuple[MerchantVersionChange, ...]
    changed: bool


class MerchantDraftWrite(BaseModel):
    """Editable draft fields; URL scope and operator identity are server-owned."""

    model_config = _TRANSPORT

    schema_version: Literal[1] = 1
    revision: int = Field(ge=1, strict=True)
    request_id: AuthorityIdentifier
    merchant_override: dict[str, JsonValue]
    fixtures: MerchantFixtureBundle
    dataset_manifest: MerchantDatasetManifest | None = None


class MerchantDatasetImport(BaseModel):
    model_config = _TRANSPORT

    schema_version: Literal[1] = 1
    expected_draft_revision: int = Field(ge=1, strict=True)
    request_id: AuthorityIdentifier
    dataset: MerchantScenarioDataset


class MerchantCatalogImport(BaseModel):
    model_config = _TRANSPORT

    schema_version: Literal[1] = 1
    expected_draft_revision: int = Field(ge=1, strict=True)
    request_id: AuthorityIdentifier
    catalog: CatalogFixture


class MerchantDraftValidation(BaseModel):
    model_config = _TRANSPORT

    schema_version: Literal[1] = 1
    draft_revision: int = Field(ge=1, strict=True)
    request_id: AuthorityIdentifier


class MerchantDraftSeed(BaseModel):
    model_config = _TRANSPORT

    schema_version: Literal[1] = 1
    request_id: AuthorityIdentifier


class MerchantDraftPreview(BaseModel):
    model_config = _TRANSPORT

    schema_version: Literal[1] = 1
    draft_revision: int = Field(ge=1, strict=True)


class MerchantDraftPublication(BaseModel):
    model_config = _TRANSPORT

    schema_version: Literal[1] = 1
    draft_revision: int = Field(ge=1, strict=True)
    expected_preview_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_active_version_id: AuthorityIdentifier | None = None
    request_id: AuthorityIdentifier


class MerchantRetirement(BaseModel):
    model_config = _TRANSPORT

    schema_version: Literal[1] = 1
    expected_active_version_id: AuthorityIdentifier
    request_id: AuthorityIdentifier


class MerchantRollback(BaseModel):
    model_config = _TRANSPORT

    schema_version: Literal[1] = 1
    expected_active_version_id: AuthorityIdentifier
    source_version_id: AuthorityIdentifier
    request_id: AuthorityIdentifier


class MerchantSimulationStart(BaseModel):
    model_config = _TRANSPORT

    schema_version: Literal[1] = 1
    version_id: AuthorityIdentifier | None = None


class MerchantSimulationTurn(BaseModel):
    model_config = _TRANSPORT

    schema_version: Literal[1] = 1
    request_id: AuthorityIdentifier
    text: str = Field(min_length=1, max_length=20_000)
    readback_interrupted: bool = False


class _UncachedStaticFiles(StaticFiles):
    """Serve workbench assets without heuristic caching.

    StaticFiles sends only last-modified and etag. With no cache-control a browser is free to
    reuse a module without revalidating, which serves a stale script against fresh markup and
    looks like the page silently ignoring its own controls. These assets are loopback-only
    development state and are edited constantly, so correctness beats the saved request.
    """

    def is_not_modified(self, response_headers, request_headers) -> bool:
        return False

    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        response.headers["cache-control"] = "no-store"
        return response


class MerchantVoicePreviewIdentity(BaseModel):
    """Which configured engines a preview exercises. Identity, never credentials."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tts_provider: str
    tts_model: str
    voice_id: str
    stt_provider: str
    stt_model: str
    capture_sample_rate: int
    max_capture_seconds: int


class MerchantVoiceSpeechRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = Field(min_length=1, max_length=MAX_SPEECH_CHARS)


class MerchantVoiceTranscript(BaseModel):
    """An empty transcript means silence, which is a result rather than a failure."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str


logger = logging.getLogger("agnostic_market.management.api")


def _error_response(
    code: ManagementApiErrorCode,
    status_code: int,
    *,
    validation: MerchantValidationResult | None = None,
) -> JSONResponse:
    payload = ManagementApiError(code=code, validation=validation)
    return JSONResponse(
        status_code=status_code,
        content=payload.model_dump(mode="json", exclude_none=True),
    )


def _catalog_page(
    products: tuple[CatalogProduct, ...],
    *,
    offset: int,
    limit: int,
) -> MerchantCatalogPage:
    return MerchantCatalogPage(
        offset=offset,
        limit=limit,
        total=len(products),
        products=products[offset : offset + limit],
    )


def create_management_app(
    service: MerchantManagementService,
    *,
    development_actor_id: str,
    simulator: PublishedMerchantSimulator | None = None,
    voice_preview: Callable[[VoiceConfig], VoicePreview] | None = None,
) -> FastAPI:
    """Build the local adapter with one deployment-owned development actor."""

    actor_id = _AUTHORITY.validate_python(development_actor_id, strict=True)

    @asynccontextmanager
    async def lifespan(_application: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            if simulator is not None:
                await simulator.aclose()

    app = FastAPI(
        title="Agnostic Market administration API",
        version="1",
        description="Loopback development adapter for versioned merchant configuration.",
        lifespan=lifespan,
    )

    def require_simulator() -> PublishedMerchantSimulator:
        if simulator is None:
            raise MerchantSimulationError("text simulation is not configured")
        return simulator

    def require_voice_preview(tenant_id: str, version_id: str) -> VoicePreview:
        if voice_preview is None:
            raise MerchantSimulationError("voice preview is not configured")
        # Bound to the published version the simulator pins, not to the current config tree.
        published = service.get_version(tenant_id, version_id)
        return voice_preview(published.config.voice)

    @app.get("/admin", include_in_schema=False)
    def management_ui() -> FileResponse:
        return FileResponse(
            _UI_ROOT / "index.html",
            media_type="text/html",
            headers=_UI_SECURITY_HEADERS,
        )

    @app.exception_handler(RequestValidationError)
    async def invalid_http_request(
        _request: Request,
        _exc: RequestValidationError,
    ) -> JSONResponse:
        return _error_response("invalid_request", 422)

    @app.exception_handler(ValidationError)
    async def invalid_contract(
        _request: Request,
        _exc: ValidationError,
    ) -> JSONResponse:
        return _error_response("service_unavailable", 500)

    @app.exception_handler(ResponseValidationError)
    async def invalid_response_contract(
        _request: Request,
        _exc: ResponseValidationError,
    ) -> JSONResponse:
        return _error_response("service_unavailable", 500)

    @app.exception_handler(MerchantManagementValidationError)
    async def invalid_draft(
        _request: Request,
        exc: MerchantManagementValidationError,
    ) -> JSONResponse:
        return _error_response("draft_invalid", 422, validation=exc.result)

    @app.exception_handler(MerchantManagementNotFoundError)
    async def not_found(
        _request: Request,
        _exc: MerchantManagementNotFoundError,
    ) -> JSONResponse:
        return _error_response("not_found", 404)

    @app.exception_handler(MerchantManagementReplayConflictError)
    async def replay_conflict(
        _request: Request,
        _exc: MerchantManagementReplayConflictError,
    ) -> JSONResponse:
        return _error_response("replay_conflict", 409)

    @app.exception_handler(MerchantManagementConflictError)
    async def state_conflict(
        _request: Request,
        _exc: MerchantManagementConflictError,
    ) -> JSONResponse:
        return _error_response("state_conflict", 409)

    @app.exception_handler(MerchantManagementScopeError)
    async def scope_violation(
        _request: Request,
        _exc: MerchantManagementScopeError,
    ) -> JSONResponse:
        return _error_response("scope_violation", 403)

    @app.exception_handler(MerchantManagementError)
    async def service_failure(
        _request: Request,
        exc: MerchantManagementError,
    ) -> JSONResponse:
        # The response is deliberately opaque; the cause must still reach the operator.
        logger.exception("management request failed", exc_info=exc)
        return _error_response("service_unavailable", 503)

    @app.exception_handler(MerchantSimulationNotFoundError)
    async def simulation_not_found(
        _request: Request,
        _exc: MerchantSimulationNotFoundError,
    ) -> JSONResponse:
        return _error_response("not_found", 404)

    @app.exception_handler(MerchantSimulationReplayConflictError)
    async def simulation_replay_conflict(
        _request: Request,
        _exc: MerchantSimulationReplayConflictError,
    ) -> JSONResponse:
        return _error_response("replay_conflict", 409)

    @app.exception_handler(MerchantSimulationConflictError)
    async def simulation_state_conflict(
        _request: Request,
        _exc: MerchantSimulationConflictError,
    ) -> JSONResponse:
        return _error_response("state_conflict", 409)

    @app.exception_handler(MerchantSimulationError)
    async def simulation_failure(
        _request: Request,
        exc: MerchantSimulationError,
    ) -> JSONResponse:
        logger.exception("management simulation failed", exc_info=exc)
        return _error_response("service_unavailable", 503)

    @app.get("/v1/merchants", response_model=MerchantListResponse)
    def list_merchants() -> MerchantListResponse:
        configured = frozenset(service.list_configured_tenant_ids())
        managed = frozenset(service.list_tenant_ids())
        return MerchantListResponse(
            merchants=tuple(
                MerchantSummary(
                    tenant_id=tenant_id,
                    configured=tenant_id in configured,
                    managed=tenant_id in managed,
                )
                for tenant_id in sorted(configured | managed)
            )
        )

    @app.put(
        "/v1/merchants/{tenant_id}/drafts/{draft_id}",
        response_model=MerchantDraft,
    )
    def save_draft(
        tenant_id: _TenantPath,
        draft_id: _DraftPath,
        payload: Annotated[MerchantDraftWrite, Body()],
        expected_revision: Annotated[int, Query(ge=0)],
    ) -> MerchantDraft:
        return service.write_draft(
            tenant_id,
            draft_id=draft_id,
            revision=payload.revision,
            actor_id=actor_id,
            request_id=payload.request_id,
            merchant_override=payload.merchant_override,
            fixtures=payload.fixtures,
            dataset_manifest=payload.dataset_manifest,
            expected_revision=expected_revision,
        )

    @app.get(
        "/v1/merchants/{tenant_id}/drafts/{draft_id}",
        response_model=MerchantDraft,
    )
    def get_draft(tenant_id: _TenantPath, draft_id: _DraftPath) -> MerchantDraft:
        return service.get_draft(tenant_id, draft_id)

    @app.post(
        "/v1/merchants/{tenant_id}/drafts/{draft_id}/seed",
        response_model=MerchantDraft,
    )
    def seed_draft(
        tenant_id: _TenantPath,
        draft_id: _DraftPath,
        payload: Annotated[MerchantDraftSeed, Body()],
    ) -> MerchantDraft:
        return service.seed_draft(
            tenant_id,
            MerchantDraftSeedRequest(
                tenant_id=tenant_id,
                draft_id=draft_id,
                actor_id=actor_id,
                request_id=payload.request_id,
            ),
        )

    @app.put(
        "/v1/merchants/{tenant_id}/drafts/{draft_id}/dataset",
        response_model=MerchantDraft,
    )
    def import_dataset(
        tenant_id: _TenantPath,
        draft_id: _DraftPath,
        payload: Annotated[MerchantDatasetImport, Body()],
    ) -> MerchantDraft:
        return service.import_dataset(
            tenant_id,
            MerchantDatasetImportRequest(
                tenant_id=tenant_id,
                draft_id=draft_id,
                actor_id=actor_id,
                expected_draft_revision=payload.expected_draft_revision,
                request_id=payload.request_id,
                dataset=payload.dataset,
            ),
        )

    @app.put(
        "/v1/merchants/{tenant_id}/drafts/{draft_id}/catalog",
        response_model=MerchantDraft,
    )
    def import_catalog(
        tenant_id: _TenantPath,
        draft_id: _DraftPath,
        payload: Annotated[MerchantCatalogImport, Body()],
    ) -> MerchantDraft:
        return service.import_catalog(
            tenant_id,
            MerchantCatalogImportRequest(
                tenant_id=tenant_id,
                draft_id=draft_id,
                actor_id=actor_id,
                **payload.model_dump(exclude={"schema_version"}),
            ),
        )

    @app.post(
        "/v1/merchants/{tenant_id}/drafts/{draft_id}/validation",
        response_model=MerchantValidationResult,
    )
    def validate_draft(
        tenant_id: _TenantPath,
        draft_id: _DraftPath,
        payload: Annotated[MerchantDraftValidation, Body()],
    ) -> MerchantValidationResult:
        return service.validate_draft(
            tenant_id,
            MerchantDraftValidationRequest(
                tenant_id=tenant_id,
                draft_id=draft_id,
                actor_id=actor_id,
                **payload.model_dump(exclude={"schema_version"}),
            ),
        )

    @app.post(
        "/v1/merchants/{tenant_id}/drafts/{draft_id}/preview",
        response_model=ResolvedMerchantPreview,
    )
    def preview_draft(
        tenant_id: _TenantPath,
        draft_id: _DraftPath,
        payload: Annotated[MerchantDraftPreview, Body()],
    ) -> ResolvedMerchantPreview:
        return service.preview_draft(
            tenant_id,
            draft_id,
            expected_revision=payload.draft_revision,
        )

    @app.post(
        "/v1/merchants/{tenant_id}/drafts/{draft_id}/publication",
        response_model=PublicationReceipt,
    )
    def publish_draft(
        tenant_id: _TenantPath,
        draft_id: _DraftPath,
        payload: Annotated[MerchantDraftPublication, Body()],
    ) -> PublicationReceipt:
        return service.publish_draft(
            tenant_id,
            MerchantDraftPublicationRequest(
                tenant_id=tenant_id,
                draft_id=draft_id,
                actor_id=actor_id,
                **payload.model_dump(exclude={"schema_version"}),
            ),
        )

    @app.get(
        "/v1/merchants/{tenant_id}/versions/active",
        response_model=PublishedMerchantVersion,
    )
    def get_active_version(tenant_id: _TenantPath) -> PublishedMerchantVersion:
        return service.get_active_version(tenant_id)

    @app.get(
        "/v1/merchants/{tenant_id}/versions",
        response_model=MerchantVersionListResponse,
    )
    def list_versions(tenant_id: _TenantPath) -> MerchantVersionListResponse:
        return MerchantVersionListResponse(versions=service.list_versions(tenant_id))

    @app.get(
        "/v1/merchants/{tenant_id}/versions/{version_id}",
        response_model=PublishedMerchantVersion,
    )
    def get_version(
        tenant_id: _TenantPath,
        version_id: _VersionPath,
    ) -> PublishedMerchantVersion:
        return service.get_version(tenant_id, version_id)

    @app.get(
        "/v1/merchants/{tenant_id}/versions/{version_id}/voice",
        response_model=MerchantVoicePreviewIdentity,
    )
    def voice_identity(
        tenant_id: _TenantPath,
        version_id: _VersionPath,
    ) -> MerchantVoicePreviewIdentity:
        """Name the engines a preview will use, so the page never guesses at them."""
        preview = require_voice_preview(tenant_id, version_id)
        tts_provider, tts_model, voice_id = preview.tts_identity
        stt_provider, stt_model = preview.stt_identity
        return MerchantVoicePreviewIdentity(
            tts_provider=tts_provider,
            tts_model=tts_model,
            voice_id=voice_id,
            stt_provider=stt_provider,
            stt_model=stt_model,
            capture_sample_rate=CAPTURE_SAMPLE_RATE,
            max_capture_seconds=MAX_CAPTURE_SECONDS,
        )

    @app.post("/v1/merchants/{tenant_id}/versions/{version_id}/voice/speech")
    async def synthesize_speech(
        tenant_id: _TenantPath,
        version_id: _VersionPath,
        payload: Annotated[MerchantVoiceSpeechRequest, Body()],
    ) -> Response:
        speech = await require_voice_preview(tenant_id, version_id).synthesize(payload.text)
        # Audio is returned as bytes rather than base64 so the page can stream it straight
        # into an audio element without holding a second copy in memory.
        return Response(content=speech.audio_wav, media_type="audio/wav")

    @app.post(
        "/v1/merchants/{tenant_id}/versions/{version_id}/voice/transcript",
        response_model=MerchantVoiceTranscript,
    )
    async def transcribe_capture(
        tenant_id: _TenantPath,
        version_id: _VersionPath,
        request: Request,
    ) -> MerchantVoiceTranscript:
        # Raw 16-bit mono PCM at CAPTURE_SAMPLE_RATE. A container would need decoding here,
        # and the browser can resample without one.
        captured = await request.body()
        text = await require_voice_preview(tenant_id, version_id).transcribe(captured)
        return MerchantVoiceTranscript(text=text)

    @app.post(
        "/v1/merchants/{tenant_id}/simulations/{simulation_id}",
        response_model=SimulationSessionStatus,
    )
    async def start_simulation(
        tenant_id: _TenantPath,
        simulation_id: _SimulationPath,
        payload: Annotated[MerchantSimulationStart, Body()],
    ) -> SimulationSessionStatus:
        return await require_simulator().start(
            tenant_id,
            simulation_id,
            version_id=payload.version_id,
        )

    @app.get(
        "/v1/merchants/{tenant_id}/simulations/{simulation_id}",
        response_model=SimulationSessionStatus,
    )
    async def get_simulation(
        tenant_id: _TenantPath,
        simulation_id: _SimulationPath,
    ) -> SimulationSessionStatus:
        return await require_simulator().status(tenant_id, simulation_id)

    @app.post(
        "/v1/merchants/{tenant_id}/simulations/{simulation_id}/turns",
        response_model=SimulationTurnResult,
    )
    async def simulate_turn(
        tenant_id: _TenantPath,
        simulation_id: _SimulationPath,
        payload: Annotated[MerchantSimulationTurn, Body()],
    ) -> SimulationTurnResult:
        return await require_simulator().turn(
            tenant_id,
            simulation_id,
            request_id=payload.request_id,
            text=payload.text,
            readback_interrupted=payload.readback_interrupted,
        )

    @app.get(
        "/v1/merchants/{tenant_id}/simulations/{simulation_id}/state",
        response_model=SimulationStateProjection,
    )
    async def inspect_simulation_state(
        tenant_id: _TenantPath,
        simulation_id: _SimulationPath,
    ) -> SimulationStateProjection:
        return await require_simulator().inspect_state(tenant_id, simulation_id)

    @app.post(
        "/v1/merchants/{tenant_id}/simulations/{simulation_id}/reset",
        response_model=SimulationSessionStatus,
    )
    async def reset_simulation(
        tenant_id: _TenantPath,
        simulation_id: _SimulationPath,
    ) -> SimulationSessionStatus:
        return await require_simulator().reset(tenant_id, simulation_id)

    @app.delete(
        "/v1/merchants/{tenant_id}/simulations/{simulation_id}",
        status_code=204,
    )
    async def close_simulation(
        tenant_id: _TenantPath,
        simulation_id: _SimulationPath,
    ) -> Response:
        await require_simulator().close(tenant_id, simulation_id)
        return Response(status_code=204)

    @app.get(
        "/v1/merchants/{tenant_id}/version-diff",
        response_model=MerchantVersionDiffResponse,
    )
    def compare_versions(
        tenant_id: _TenantPath,
        base_version_id: Annotated[str, Query(min_length=1, pattern=r"^\S+$")],
        target_version_id: Annotated[str, Query(min_length=1, pattern=r"^\S+$")],
    ) -> MerchantVersionDiffResponse:
        difference = service.compare_versions(tenant_id, base_version_id, target_version_id)
        return MerchantVersionDiffResponse(
            **difference.model_dump(),
            changed=difference.changed,
        )

    @app.get(
        "/v1/merchants/{tenant_id}/drafts/{draft_id}/catalog",
        response_model=MerchantCatalogPage,
    )
    def get_draft_catalog(
        tenant_id: _TenantPath,
        draft_id: _DraftPath,
        offset: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=100)] = 25,
    ) -> MerchantCatalogPage:
        products = service.get_draft(tenant_id, draft_id).fixtures.catalog.products
        return _catalog_page(products, offset=offset, limit=limit)

    @app.get(
        "/v1/merchants/{tenant_id}/versions/{version_id}/catalog",
        response_model=MerchantCatalogPage,
    )
    def get_version_catalog(
        tenant_id: _TenantPath,
        version_id: _VersionPath,
        offset: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=100)] = 25,
    ) -> MerchantCatalogPage:
        products = service.get_version(tenant_id, version_id).fixtures.catalog.products
        return _catalog_page(products, offset=offset, limit=limit)

    @app.post(
        "/v1/merchants/{tenant_id}/active/retirement",
        response_model=MerchantRetirementReceipt,
    )
    def retire(
        tenant_id: _TenantPath,
        payload: Annotated[MerchantRetirement, Body()],
    ) -> MerchantRetirementReceipt:
        return service.retire(
            tenant_id,
            MerchantRetirementRequest(
                tenant_id=tenant_id,
                actor_id=actor_id,
                **payload.model_dump(exclude={"schema_version"}),
            ),
        )

    @app.post(
        "/v1/merchants/{tenant_id}/rollbacks",
        response_model=PublicationReceipt,
    )
    def rollback(
        tenant_id: _TenantPath,
        payload: Annotated[MerchantRollback, Body()],
    ) -> PublicationReceipt:
        return service.rollback(
            tenant_id,
            MerchantRollbackRequest(
                tenant_id=tenant_id,
                actor_id=actor_id,
                **payload.model_dump(exclude={"schema_version"}),
            ),
        )

    @app.get(
        "/v1/merchants/{tenant_id}/audit",
        response_model=MerchantAuditListResponse,
    )
    def list_audit(tenant_id: _TenantPath) -> MerchantAuditListResponse:
        return MerchantAuditListResponse(records=service.list_audit_history(tenant_id))

    app.mount(
        "/admin/assets",
        _UncachedStaticFiles(directory=_UI_ROOT / "assets"),
        name="management-ui-assets",
    )
    return app
