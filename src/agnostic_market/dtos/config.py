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
    # Pinned immutable snapshot, never a floating alias (VOICE_PIPELINE §1a) — a fast-moving
    # TTS vendor can shift behavior under a floating ID.
    model: str = Field(min_length=1)
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
    """Refund thresholds; values are merchant-set but bounded. `auto_approve_under_usd` <=
    `require_human_above_usd` (ordering) and `require_human_above_usd` <= the platform
    ceiling are enforced in the resolver (it sees `_platform.limits`); the DTO only checks
    the local `>= 0` floor."""

    model_config = _STRICT

    auto_approve_under_usd: float = Field(ge=0)
    require_human_above_usd: float = Field(ge=0)
    # Return-first is the industry default for SHIPPED/DELIVERED orders: the refund is
    # issued once the return exists, else the caller keeps both goods and money. A refund
    # at or under this value may skip the return ("returnless refund" — a deliberate
    # policy for items where return shipping costs more than the goods). Default 0 =
    # return-first for every shipped refund; bounded by the platform ceiling.
    returnless_under_usd: float = Field(ge=0, default=0.0)


# Default caller-silence window before a paused confirmation (checkout/refund/cancel) expires
# — a synchronous voice call, so short. Merchant may tune it within the platform max
# (_platform.limits.pending_confirmation_ttl_max_seconds, resolver-enforced).
_DEFAULT_PENDING_TTL_SECONDS = 120.0


class PolicyConfig(BaseModel):
    """Merchant policy values — set within platform-enforced bounds (never disable a cap)."""

    model_config = _STRICT

    max_order_value_usd: float = Field(ge=0)
    refunds: RefundPolicy
    allow_ai_merchant_handoff: StrictBool
    # Optional: merchant-tunable within the platform max. Defaults so existing merchant YAMLs
    # (which don't set it) keep the platform default; the resolver clamps the ceiling.
    pending_confirmation_ttl_seconds: float = Field(
        default=_DEFAULT_PENDING_TTL_SECONDS, gt=0
    )
    # Optional merchant free-text policy facts that have NO enforcing field — refund
    # TIMELINE ("5-7 business days"), return WINDOW ("30 days in original condition"). The
    # ENFORCED policy sentences (returnless threshold, human-review line) are DERIVED from
    # the typed values in agents/spoken_policy.py, never retyped here (drift guard). Keep
    # this to facts nothing else in the config represents; absent => only the derived
    # sentences are spoken. NEVER restate an enforced number here — it would drift.
    spoken_facts_extra: str | None = None


class ComplianceConfig(BaseModel):
    """Regulatory disclosure config (COMPLIANCE §2 / BUILD_PLAN §2.5).

    `call_start_disclosure` wording is merchant-editable (brand/voice/locale); WHETHER it
    plays is enforced in code (voice/pipeline.py plays it first, uninterruptible) — a
    merchant can rephrase the disclosure, never remove it. `{display_name}` is formatted
    at session build.
    """

    model_config = _STRICT

    call_start_disclosure: str = Field(min_length=1)


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
    compliance: ComplianceConfig
    integration: IntegrationConfig
    isolation: IsolationConfig

    vector_namespace: str = Field(min_length=1)
    # Reference only — resolves via SecretResolver at use time; never an inline secret.
    secrets_ref: str = Field(min_length=1)
