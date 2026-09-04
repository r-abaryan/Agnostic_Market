"""Resolve one voice job before constructing tenant runtime dependencies."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Literal

from livekit import rtc
from livekit.agents import JobContext
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from agnostic_market.config.registry import ConfigRegistry, ResolvedConfig
from agnostic_market.dtos.session import AdmittedSessionAuthority, TransportAuthority
from agnostic_market.tenancy.context import TenantContext, build_tenant_context
from agnostic_market.tenancy.resolver import TenantResolutionError, TenantResolver

_SIP_RULE_ID_ATTRIBUTE = "sip.ruleID"
_SIP_TRUNK_NUMBER_ATTRIBUTE = "sip.trunkPhoneNumber"


class VoiceJobMetadata(BaseModel):
    """Tenant authority supplied by an approved server-side dispatch."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[1]
    merchant_id: str = Field(min_length=1)
    participant_kind: Literal["sip", "standard"]
    participant_identity: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _validate_authority_fields(self) -> VoiceJobMetadata:
        if self.merchant_id != self.merchant_id.strip():
            raise ValueError("merchant_id must not contain surrounding whitespace")
        if (
            self.participant_identity is not None
            and self.participant_identity != self.participant_identity.strip()
        ):
            raise ValueError("participant_identity must not contain surrounding whitespace")
        if self.participant_kind == "standard" and self.participant_identity is None:
            raise ValueError("standard dispatches require participant_identity")
        return self


@dataclass(frozen=True, slots=True)
class _ResolvedVoiceContext:
    tenant: TenantContext
    resolved: ResolvedConfig


@dataclass(frozen=True, slots=True)
class ConsoleVoiceTenantAdmission(_ResolvedVoiceContext):
    admission_kind: Literal["console"] = field(default="console", init=False)


@dataclass(frozen=True, slots=True)
class NetworkVoiceTenantAdmission(_ResolvedVoiceContext):
    participant_identity: str
    session_authority: AdmittedSessionAuthority
    admission_kind: Literal["network"] = field(default="network", init=False)


type VoiceTenantAdmission = ConsoleVoiceTenantAdmission | NetworkVoiceTenantAdmission


@dataclass(frozen=True, slots=True)
class ConsoleVoiceAdmissionPreflight(_ResolvedVoiceContext):
    participant_kind: Literal["console"] = field(default="console", init=False)


@dataclass(frozen=True, slots=True)
class NetworkVoiceAdmissionPreflight(_ResolvedVoiceContext):
    participant_kind: Literal["sip", "standard"]
    participant_identity: str | None
    session_authority: AdmittedSessionAuthority


type VoiceAdmissionPreflight = ConsoleVoiceAdmissionPreflight | NetworkVoiceAdmissionPreflight


def _parse_job_metadata(raw_metadata: str) -> VoiceJobMetadata | None:
    if not raw_metadata.strip():
        return None
    try:
        return VoiceJobMetadata.model_validate_json(raw_metadata)
    except ValidationError as exc:
        raise TenantResolutionError("voice job metadata is invalid") from exc


def _session_authority(job_context: JobContext) -> AdmittedSessionAuthority:
    try:
        return AdmittedSessionAuthority(
            logical_session_id=job_context.job.dispatch_id,
            transport=TransportAuthority(
                provider="livekit",
                room_id=job_context.job.room.sid,
                assignment_id=job_context.job.id,
                worker_id=job_context.worker_id,
            ),
        )
    except (AttributeError, ValidationError) as exc:
        raise TenantResolutionError("voice job transport authority is invalid") from exc


class VoiceJobAdmission:
    """Resolve tenant authority before connect, then bind the admitted participant."""

    def __init__(
        self,
        registry: ConfigRegistry,
        *,
        development_merchant_id: str | None,
    ) -> None:
        self._registry = registry
        self._resolver = TenantResolver(registry)
        self._development_merchant_id = development_merchant_id

    def preflight(self, job_context: JobContext) -> VoiceAdmissionPreflight:
        if job_context.is_fake_job():
            merchant_id = (self._development_merchant_id or "").strip()
            if not merchant_id:
                raise TenantResolutionError(
                    "console voice admission requires VOICE_AGENT_MERCHANT_ID"
                )
            admitted_merchant_id = self._resolver.resolve_by_id(merchant_id)
            return ConsoleVoiceAdmissionPreflight(
                tenant=build_tenant_context(self._registry, admitted_merchant_id),
                resolved=self._registry.get(admitted_merchant_id),
            )
        metadata = _parse_job_metadata(job_context.job.metadata)
        if metadata is None:
            raise TenantResolutionError(
                "production voice admission requires approved dispatch metadata"
            )
        admitted_merchant_id = self._resolver.resolve_by_id(metadata.merchant_id)

        return NetworkVoiceAdmissionPreflight(
            tenant=build_tenant_context(self._registry, admitted_merchant_id),
            resolved=self._registry.get(admitted_merchant_id),
            participant_kind=metadata.participant_kind,
            participant_identity=metadata.participant_identity,
            session_authority=_session_authority(job_context),
        )

    async def complete(
        self,
        job_context: JobContext,
        preflight: VoiceAdmissionPreflight,
        *,
        timeout_seconds: float,
    ) -> VoiceTenantAdmission:
        if timeout_seconds <= 0:
            raise ValueError("voice admission timeout must be positive")

        try:
            async with asyncio.timeout(timeout_seconds):
                await job_context.connect()
                if isinstance(preflight, ConsoleVoiceAdmissionPreflight):
                    return ConsoleVoiceTenantAdmission(
                        tenant=preflight.tenant,
                        resolved=preflight.resolved,
                    )
                participant_identity = await self._bind_network_participant(
                    job_context,
                    preflight,
                )
        except TimeoutError as exc:
            raise TenantResolutionError(
                "voice admission timed out while connecting or waiting for the participant"
            ) from exc

        return NetworkVoiceTenantAdmission(
            tenant=preflight.tenant,
            resolved=preflight.resolved,
            participant_identity=participant_identity,
            session_authority=preflight.session_authority,
        )

    async def _bind_network_participant(
        self,
        job_context: JobContext,
        preflight: NetworkVoiceAdmissionPreflight,
    ) -> str:
        participant_kind = (
            rtc.ParticipantKind.PARTICIPANT_KIND_SIP
            if preflight.participant_kind == "sip"
            else rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD
        )
        participant = await job_context.wait_for_participant(
            identity=preflight.participant_identity,
            kind=participant_kind,
        )

        if preflight.participant_kind == "sip":
            if not participant.attributes.get(_SIP_RULE_ID_ATTRIBUTE, "").strip():
                raise TenantResolutionError(
                    "production voice admission requires an inbound SIP dispatch rule"
                )
            inbound_number = participant.attributes.get(_SIP_TRUNK_NUMBER_ATTRIBUTE, "")
            if not inbound_number:
                raise TenantResolutionError(
                    "production voice admission has no inbound trunk number"
                )
            admitted_by_did = self._resolver.resolve_by_number(inbound_number)
            if admitted_by_did != preflight.tenant.tenant_id:
                raise TenantResolutionError(
                    "voice admission authorities resolve to different tenants"
                )

        return participant.identity
