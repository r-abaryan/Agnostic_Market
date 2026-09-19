"""Fail-closed activation for a semantically qualified routing recognizer."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from agnostic_market.agents.capabilities import CapabilityRegistry
from agnostic_market.agents.routing import (
    CONTEXT_PROJECTOR_VERSION,
    ROUTE_SCHEMA_FINGERPRINT,
    ROUTER_PROMPT_FINGERPRINT,
    RoutingRecognizer,
    SemanticRouter,
    registry_fingerprint,
)
from agnostic_market.config.loader import load_yaml_layer
from agnostic_market.dtos.config import ProviderModel, ReasoningEffort
from agnostic_market.dtos.llm import ProviderCredentialsConfig, StructuredOutputMethod
from agnostic_market.llm.gateway import LLMGateway
from agnostic_market.secrets.base import SecretResolver

type SemanticRoutingQualificationSchemaVersion = Literal["10"]
SEMANTIC_ROUTING_QUALIFICATION_SCHEMA_VERSION: SemanticRoutingQualificationSchemaVersion = "10"
SEMANTIC_ROUTING_RELEASE_EVIDENCE_SCHEMA_VERSION = 1
RoutingRecognizerFactory = Callable[[CapabilityRegistry], RoutingRecognizer]

_STRICT = ConfigDict(extra="ignore", frozen=True)
_CONTRACT_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True)
_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class RoutingActivationError(RuntimeError):
    """The configured recognizer has no valid semantic qualification."""


class _QualificationGate(BaseModel):
    model_config = _STRICT

    mode: Literal["diagnostic", "shadow", "cutover"]
    passed: bool | None
    failures: tuple[str, ...]


class _QualificationProjection(BaseModel):
    model_config = _STRICT

    exact: bool


class _QualifiedRecognizer(BaseModel):
    model_config = _STRICT

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    reasoning_effort: ReasoningEffort | None
    structured_output_method: StructuredOutputMethod
    route_schema_fingerprint: str = Field(min_length=1)
    prompt_fingerprint: str = Field(min_length=1)
    registry_fingerprint: str = Field(min_length=1)
    input_max_chars: int = Field(gt=0)
    timeout_seconds: float = Field(gt=0)
    projector_version: str = Field(min_length=1)


class SemanticRoutingRuntimeContract(BaseModel):
    """Exact non-secret semantic recognizer identity used by a running worker."""

    model_config = _CONTRACT_STRICT

    schema_version: Literal[1] = 1
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    reasoning_effort: ReasoningEffort | None
    structured_output_method: StructuredOutputMethod
    route_schema_fingerprint: str = Field(min_length=1)
    prompt_fingerprint: str = Field(min_length=1)
    registry_fingerprint: str = Field(min_length=1)
    projector_version: str = Field(min_length=1)
    input_max_chars: int = Field(gt=0)
    timeout_seconds: float = Field(gt=0)
    corpus_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    qualification_evidence_fingerprint: str = Field(pattern=_SHA256_PATTERN)


def semantic_routing_runtime_contract(
    registry: CapabilityRegistry,
    *,
    selection: ProviderModel,
    structured_output_method: StructuredOutputMethod,
    timeout_seconds: float,
    input_max_chars: int,
    corpus_fingerprint: str,
    qualification_evidence_fingerprint: str,
) -> SemanticRoutingRuntimeContract:
    """Derive the recognizer contract from the same inputs used at activation."""
    return SemanticRoutingRuntimeContract(
        provider=selection.provider,
        model=selection.model,
        reasoning_effort=selection.reasoning_effort,
        structured_output_method=structured_output_method,
        route_schema_fingerprint=ROUTE_SCHEMA_FINGERPRINT,
        prompt_fingerprint=ROUTER_PROMPT_FINGERPRINT,
        registry_fingerprint=registry_fingerprint(registry),
        projector_version=CONTEXT_PROJECTOR_VERSION,
        input_max_chars=input_max_chars,
        timeout_seconds=timeout_seconds,
        corpus_fingerprint=corpus_fingerprint,
        qualification_evidence_fingerprint=qualification_evidence_fingerprint,
    )


def semantic_routing_runtime_contract_fingerprint(
    contract: SemanticRoutingRuntimeContract,
) -> str:
    canonical = json.dumps(
        contract.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def semantic_routing_corpus_fingerprint(config_root: Path) -> str:
    """Load the repository-owned frozen corpus identity used by activation."""
    routing_contract = load_yaml_layer(
        config_root / "eval" / "frontline_semantic_route_structural.yaml"
    )
    fingerprint = routing_contract.get("frozen_corpus_fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint.strip():
        raise RoutingActivationError("semantic routing corpus contract has no frozen fingerprint")
    return fingerprint


class _QualificationModels(BaseModel):
    model_config = _STRICT

    candidate: _QualifiedRecognizer


class SemanticRoutingQualification(BaseModel):
    model_config = _STRICT

    schema_version: SemanticRoutingQualificationSchemaVersion
    qualification_run_id: str = Field(min_length=1)
    corpus_role: Literal["regression", "readiness_holdout"]
    run_at: datetime
    corpus_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    gate: _QualificationGate
    projection: _QualificationProjection
    models: _QualificationModels


class SemanticRoutingReleaseEvidence(BaseModel):
    """One packaged routing decision over regression and source-disjoint evidence."""

    model_config = _CONTRACT_STRICT

    schema_version: Literal[1]
    packaged_at: datetime
    holdout_id: str = Field(min_length=1)
    holdout_steward_id: str = Field(min_length=1)
    holdout_frozen_at: datetime
    holdout_lineage_reference: str = Field(min_length=1)
    holdout_generation_recipe_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    holdout_source_disjoint_from_corpus_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    regression_report_sha256: str = Field(pattern=_SHA256_PATTERN)
    readiness_report_sha256: str = Field(pattern=_SHA256_PATTERN)
    regression: SemanticRoutingQualification
    readiness_holdout: SemanticRoutingQualification

    @model_validator(mode="after")
    def validate_release_evidence(self) -> Self:
        if self.packaged_at.tzinfo is None or self.holdout_frozen_at.tzinfo is None:
            raise ValueError("routing release timestamps must be timezone-aware")
        if self.holdout_frozen_at > self.packaged_at:
            raise ValueError("routing holdout cannot freeze after release packaging")
        text = (
            self.holdout_id,
            self.holdout_steward_id,
            self.holdout_lineage_reference,
            self.regression.qualification_run_id,
        )
        if any(not value.strip() or value != value.strip() for value in text):
            raise ValueError("routing release identifiers must be normalized and non-empty")
        if self.regression.corpus_role != "regression":
            raise ValueError("routing release regression evidence has the wrong corpus role")
        if self.readiness_holdout.corpus_role != "readiness_holdout":
            raise ValueError("routing release holdout evidence has the wrong corpus role")
        if self.regression.qualification_run_id != self.readiness_holdout.qualification_run_id:
            raise ValueError("routing release reports must come from one qualification run")
        if self.regression.corpus_fingerprint == self.readiness_holdout.corpus_fingerprint:
            raise ValueError("routing release corpora must be distinct")
        if (
            self.holdout_source_disjoint_from_corpus_fingerprint
            != self.regression.corpus_fingerprint
        ):
            raise ValueError("routing holdout disjointness targets the wrong regression corpus")
        if self.regression.run_at.tzinfo is None or self.readiness_holdout.run_at.tzinfo is None:
            raise ValueError("routing qualification timestamps must be timezone-aware")
        if self.holdout_frozen_at > self.readiness_holdout.run_at:
            raise ValueError("routing holdout must freeze before qualification")
        if max(self.regression.run_at, self.readiness_holdout.run_at) > self.packaged_at:
            raise ValueError("routing reports cannot postdate release packaging")
        for label, qualification in (
            ("regression", self.regression),
            ("readiness holdout", self.readiness_holdout),
        ):
            if (
                qualification.gate.mode != "cutover"
                or qualification.gate.passed is not True
                or qualification.gate.failures
                or not qualification.projection.exact
            ):
                raise ValueError(f"routing {label} evidence is not a passing cutover")
        if self.regression.models.candidate != self.readiness_holdout.models.candidate:
            raise ValueError("routing release reports use different recognizer contracts")
        return self


def semantic_routing_release_evidence_fingerprint(
    evidence: SemanticRoutingReleaseEvidence,
) -> str:
    canonical = json.dumps(
        evidence.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def load_semantic_routing_release_evidence(path: Path) -> SemanticRoutingReleaseEvidence:
    try:
        return SemanticRoutingReleaseEvidence.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        raise RoutingActivationError(
            f"semantic routing release evidence is missing or invalid: {path}"
        ) from exc


@dataclass(frozen=True, slots=True)
class ConfiguredSemanticRouterFactory:
    """Construct the selected recognizer without making an activation claim."""

    selection: ProviderModel
    credentials: ProviderCredentialsConfig
    secrets: SecretResolver
    structured_output_method: StructuredOutputMethod
    timeout_seconds: float
    input_max_chars: int

    def __call__(self, registry: CapabilityRegistry) -> RoutingRecognizer:
        gateway = LLMGateway(self.credentials, self.secrets)
        return SemanticRouter(
            gateway.chat_model(self.selection),
            selection=self.selection,
            structured_output_method=self.structured_output_method,
            timeout_seconds=self.timeout_seconds,
            input_max_chars=self.input_max_chars,
            registry=registry,
        )


@dataclass(frozen=True, slots=True)
class QualifiedSemanticRouterFactory:
    """Build the existing provider recognizer only after its exact contract qualifies."""

    qualification_path: Path
    selection: ProviderModel
    credentials: ProviderCredentialsConfig
    secrets: SecretResolver
    structured_output_method: StructuredOutputMethod
    timeout_seconds: float
    input_max_chars: int
    max_report_age_days: int
    expected_corpus_fingerprint: str
    expected_qualification_evidence_fingerprint: str

    def __post_init__(self) -> None:
        if self.max_report_age_days <= 0:
            raise ValueError("semantic qualification maximum age must be positive")
        if not self.expected_corpus_fingerprint.strip():
            raise ValueError("semantic qualification corpus fingerprint must be non-empty")
        if not self.expected_qualification_evidence_fingerprint.strip():
            raise ValueError("semantic qualification evidence fingerprint must be non-empty")

    def __call__(self, registry: CapabilityRegistry) -> RoutingRecognizer:
        release_evidence = load_semantic_routing_release_evidence(self.qualification_path)
        evidence_fingerprint = semantic_routing_release_evidence_fingerprint(release_evidence)
        if evidence_fingerprint != self.expected_qualification_evidence_fingerprint:
            raise RoutingActivationError(
                "semantic routing activation refused: release evidence changed after preparation"
            )
        qualification = release_evidence.regression
        candidate = qualification.models.candidate
        now = datetime.now(tz=UTC)
        failures: list[str] = []
        for role, report in (
            ("regression", release_evidence.regression),
            ("readiness_holdout", release_evidence.readiness_holdout),
        ):
            run_at = report.run_at
            if run_at.tzinfo is None:
                failures.append(f"{role} qualification timestamp has no timezone")
            elif run_at > now or now - run_at > timedelta(days=self.max_report_age_days):
                failures.append(f"{role} qualification is not current")
        runtime_contract = semantic_routing_runtime_contract(
            registry,
            selection=self.selection,
            structured_output_method=self.structured_output_method,
            timeout_seconds=self.timeout_seconds,
            input_max_chars=self.input_max_chars,
            corpus_fingerprint=self.expected_corpus_fingerprint,
            qualification_evidence_fingerprint=evidence_fingerprint,
        )
        expected = {
            field: getattr(runtime_contract, field) for field in _QualifiedRecognizer.model_fields
        }
        actual = candidate.model_dump()
        mismatches = [
            field for field, expected_value in expected.items() if actual[field] != expected_value
        ]
        if qualification.corpus_fingerprint != self.expected_corpus_fingerprint:
            mismatches.append("corpus_fingerprint")
        if mismatches:
            failures.append("runtime contract mismatches: " + ", ".join(mismatches))
        if failures:
            raise RoutingActivationError(
                "semantic routing activation refused: " + " | ".join(failures)
            )
        return ConfiguredSemanticRouterFactory(
            selection=self.selection,
            credentials=self.credentials,
            secrets=self.secrets,
            structured_output_method=self.structured_output_method,
            timeout_seconds=self.timeout_seconds,
            input_max_chars=self.input_max_chars,
        )(registry)


def build_qualified_semantic_router_factory(
    config_root: Path,
    *,
    selection: ProviderModel,
    credentials: ProviderCredentialsConfig,
    secrets: SecretResolver,
    structured_output_method: StructuredOutputMethod,
    timeout_seconds: float,
    input_max_chars: int,
    max_report_age_days: int,
) -> QualifiedSemanticRouterFactory:
    """Bind the recognizer to the repository's frozen routing-evaluation corpus."""
    expected_corpus_fingerprint = semantic_routing_corpus_fingerprint(config_root)
    qualification_path = config_root / "qualification" / "semantic_routing_release.json"
    release_evidence = load_semantic_routing_release_evidence(qualification_path)
    return QualifiedSemanticRouterFactory(
        qualification_path=qualification_path,
        selection=selection,
        credentials=credentials,
        secrets=secrets,
        structured_output_method=structured_output_method,
        timeout_seconds=timeout_seconds,
        input_max_chars=input_max_chars,
        max_report_age_days=max_report_age_days,
        expected_corpus_fingerprint=expected_corpus_fingerprint,
        expected_qualification_evidence_fingerprint=(
            semantic_routing_release_evidence_fingerprint(release_evidence)
        ),
    )
