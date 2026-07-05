"""LLM-plane DTOs — provider credentials (refs only) + conformance verdicts (AGENTS §A11).

Selections reuse `ProviderModel` from dtos/config.py — there is deliberately no second
selection shape. Credentials hold SecretResolver REFS (e.g. `env://NAME`), never values.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool

from agnostic_market.dtos.config import ProviderModel

# Same strict base as dtos/config.py: forbid unknown keys, validate on assignment.
_STRICT = ConfigDict(extra="forbid", validate_assignment=True)

# Cap for ConformanceCheck.detail — the suite composes/clips against this same constant.
DETAIL_MAX_LENGTH = 200


class ProviderEntry(BaseModel):
    """One provider's credential reference — a resolver ref, never a secret value."""

    model_config = _STRICT

    api_key_ref: str = Field(min_length=1)


class ProviderCredentialsConfig(BaseModel):
    """Platform-owned provider -> credential-ref map (config/base/providers.yaml).

    Doubles as the provider whitelist: the gateway serves only providers listed here.
    """

    model_config = _STRICT

    providers: dict[str, ProviderEntry]


class ConformanceCheck(BaseModel):
    """One suite check result.

    `detail` is a short machine-composed reason, length-capped. No free-text model output —
    short capped identifiers from a response (e.g. a tool name) are the only exception
    (reports.json is a shared local artifact; never leak provider content into it).
    """

    model_config = _STRICT

    name: str = Field(min_length=1)
    passed: StrictBool
    detail: str = Field(max_length=DETAIL_MAX_LENGTH)


class ConformanceReport(BaseModel):
    """Suite verdict for one provider:model (persisted in config/conformance/reports.json)."""

    model_config = _STRICT

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    suite_version: str = Field(min_length=1)
    run_at: datetime
    checks: list[ConformanceCheck]
    verdict: Literal["commerce-ready", "chat-only"]


class ConformanceTargetsConfig(BaseModel):
    """Live-run targets + expiry/retry policy (config/conformance/targets.yaml)."""

    model_config = _STRICT

    max_report_age_days: int = Field(ge=1)
    max_retries: int = Field(ge=0)
    # Never empty: a live run that certifies nothing must fail at load, not print PASS.
    targets: list[ProviderModel] = Field(min_length=1)
