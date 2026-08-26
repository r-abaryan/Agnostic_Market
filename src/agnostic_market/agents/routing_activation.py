"""Fail-closed activation for a semantically qualified routing recognizer."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agnostic_market.agents.capabilities import CapabilityRegistry
from agnostic_market.agents.routing import (
    CONTEXT_PROJECTOR_VERSION,
    ROUTE_SCHEMA_FINGERPRINT,
    ROUTER_PROMPT_FINGERPRINT,
    RoutingRecognizer,
    SemanticRouter,
    registry_fingerprint,
)
from agnostic_market.dtos.config import ProviderModel, ReasoningEffort
from agnostic_market.dtos.llm import ProviderCredentialsConfig, StructuredOutputMethod
from agnostic_market.llm.gateway import LLMGateway
from agnostic_market.secrets.base import SecretResolver

type SEMANTIC_ROUTING_QUALIFICATION_SCHEMA_VERSION = Literal["7"]
RoutingRecognizerFactory = Callable[[CapabilityRegistry], RoutingRecognizer]

_STRICT = ConfigDict(extra="ignore", frozen=True)


class RoutingActivationError(RuntimeError):
    """The configured recognizer has no valid semantic qualification."""


class _QualificationGate(BaseModel):
    model_config = _STRICT

    mode: Literal["cutover"]
    passed: Literal[True]
    failures: tuple[str, ...]


class _QualificationProjection(BaseModel):
    model_config = _STRICT

    exact: Literal[True]


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


class _QualificationModels(BaseModel):
    model_config = _STRICT

    candidate: _QualifiedRecognizer


class SemanticRoutingQualification(BaseModel):
    model_config = _STRICT

    schema_version: SEMANTIC_ROUTING_QUALIFICATION_SCHEMA_VERSION
    run_at: datetime
    corpus_fingerprint: str = Field(min_length=1)
    gate: _QualificationGate
    projection: _QualificationProjection
    models: _QualificationModels


def _load_qualification(path: Path) -> SemanticRoutingQualification:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return SemanticRoutingQualification.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise RoutingActivationError(
            f"semantic routing qualification is missing or invalid: {path}"
        ) from exc


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

    def __post_init__(self) -> None:
        if self.max_report_age_days <= 0:
            raise ValueError("semantic qualification maximum age must be positive")
        if not self.expected_corpus_fingerprint.strip():
            raise ValueError("semantic qualification corpus fingerprint must be non-empty")

    def __call__(self, registry: CapabilityRegistry) -> RoutingRecognizer:
        qualification = _load_qualification(self.qualification_path)
        candidate = qualification.models.candidate
        now = datetime.now(tz=UTC)
        run_at = qualification.run_at
        if run_at.tzinfo is None:
            raise RoutingActivationError("semantic routing qualification timestamp has no timezone")
        if run_at > now or now - run_at > timedelta(days=self.max_report_age_days):
            raise RoutingActivationError("semantic routing qualification is not current")
        expected = {
            "provider": self.selection.provider,
            "model": self.selection.model,
            "reasoning_effort": self.selection.reasoning_effort,
            "structured_output_method": self.structured_output_method,
            "route_schema_fingerprint": ROUTE_SCHEMA_FINGERPRINT,
            "prompt_fingerprint": ROUTER_PROMPT_FINGERPRINT,
            "registry_fingerprint": registry_fingerprint(registry),
            "input_max_chars": self.input_max_chars,
            "timeout_seconds": self.timeout_seconds,
            "projector_version": CONTEXT_PROJECTOR_VERSION,
        }
        actual = candidate.model_dump()
        mismatches = [
            field for field, expected_value in expected.items() if actual[field] != expected_value
        ]
        if qualification.corpus_fingerprint != self.expected_corpus_fingerprint:
            mismatches.append("corpus_fingerprint")
        if qualification.gate.failures or mismatches:
            details = ", ".join(mismatches) or "gate failures"
            raise RoutingActivationError(
                f"semantic routing qualification does not match the runtime contract: {details}"
            )
        gateway = LLMGateway(self.credentials, self.secrets)
        return SemanticRouter(
            gateway.chat_model(self.selection),
            selection=self.selection,
            structured_output_method=self.structured_output_method,
            timeout_seconds=self.timeout_seconds,
            input_max_chars=self.input_max_chars,
            registry=registry,
        )
