"""Merchant configuration DTOs — the effective, validated shape of a resolved config.

Transcribed from the BUILD_PLAN.md merchant-config example (the override layer) plus the
3-layer resolution rules. These model the **effective MerchantConfig** after
base -> template -> override resolution; the resolver (config/resolver.py) produces the
merged dict and validates it into `MerchantConfig`.

Strictness (Phase-0 plan, YAML-footgun defense):
- `extra="forbid"` — unknown keys are rejected (typos, and any override smuggling an
  undeclared key, fail loudly rather than being silently dropped).
- bool fields use `StrictBool` — a bool must already be a real bool, never coerced from a
  string. This is the model-side complement to the loader stripping YAML's implicit-bool
  resolver (config/loader.py), so `no`/`off`/`yes` cannot become booleans by accident.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool

# Reusable strict base: forbid unknown keys, validate on assignment.
_STRICT = ConfigDict(extra="forbid", validate_assignment=True)


class ProviderModel(BaseModel):
    """A provider+model selection (e.g. routing vs reasoning LLM)."""

    model_config = _STRICT

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)


class LLMConfig(BaseModel):
    """Per-merchant LLM selection; conformance-gated downstream (AGENTS §A11)."""

    model_config = _STRICT

    routing: ProviderModel
    reasoning: ProviderModel


class STTConfig(BaseModel):
    model_config = _STRICT

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)


class TTSConfig(BaseModel):
    model_config = _STRICT

    provider: str = Field(min_length=1)
    voice_id: str = Field(min_length=1)


class VoiceConfig(BaseModel):
    model_config = _STRICT

    stt: STTConfig
    tts: TTSConfig


class TelephonyConfig(BaseModel):
    model_config = _STRICT

    provider: str = Field(min_length=1)
    inbound_number: str = Field(min_length=1)


class RefundPolicy(BaseModel):
    """Refund thresholds; values are merchant-set but bounded (>= 0)."""

    model_config = _STRICT

    auto_approve_under_usd: float = Field(ge=0)
    require_human_above_usd: float = Field(ge=0)


class PolicyConfig(BaseModel):
    """Merchant policy values — set within platform-enforced bounds (never disable a cap)."""

    model_config = _STRICT

    max_order_value_usd: float = Field(ge=0)
    refunds: RefundPolicy
    allow_ai_merchant_handoff: StrictBool


class PromptsConfig(BaseModel):
    """Content-addressed persona reference (registry-resolved, not a mutable path)."""

    model_config = _STRICT

    persona_ref: str = Field(min_length=1)


class OrderSoRConfig(BaseModel):
    model_config = _STRICT

    type: str = Field(min_length=1)
    ref: str = Field(min_length=1)
    # Does the merchant's order system-of-record honor our idempotency key? (BUILD_PLAN CR-4)
    idempotency: Literal["supported", "unsupported"]


class CatalogConfig(BaseModel):
    model_config = _STRICT

    source: str = Field(min_length=1)
    freshness_sla_min: int = Field(ge=0)


class IntegrationConfig(BaseModel):
    model_config = _STRICT

    order_sor: OrderSoRConfig
    catalog: CatalogConfig


class IsolationConfig(BaseModel):
    model_config = _STRICT

    tier: Literal["shared", "dedicated"]


class MerchantConfig(BaseModel):
    """The effective, validated merchant config (after 3-layer resolution)."""

    model_config = _STRICT

    schema_version: str = Field(min_length=1)
    merchant_id: str = Field(min_length=1)
    extends_template: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    locale: str = Field(min_length=1)

    llm: LLMConfig
    voice: VoiceConfig
    telephony: TelephonyConfig
    policies: PolicyConfig
    prompts: PromptsConfig
    integration: IntegrationConfig
    isolation: IsolationConfig

    vector_namespace: str = Field(min_length=1)
    # Reference only — resolves via SecretResolver at use time; never an inline secret.
    secrets_ref: str = Field(min_length=1)
