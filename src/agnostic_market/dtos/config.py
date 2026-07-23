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

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator

if TYPE_CHECKING:
    from agnostic_market.dtos.state import PolicyContext

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
    # STT recognition aids (Deepgram nova-3). `numerals` requests digit formatting and
    # `keyterms` biases recognition toward merchant tokens such as an order-id label; neither
    # guarantees a particular transcript. Defaults request the provider's pre-tuning behavior;
    # these fields still enter the resolved config hash. Recognition quality is never authority.
    numerals: StrictBool = False
    keyterms: tuple[str, ...] = ()

    @field_validator("keyterms")
    @classmethod
    def _keyterms_are_non_blank_and_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(term.strip() for term in value)
        if any(not term for term in normalized):
            raise ValueError("keyterms must not contain blank values")
        if len({term.casefold() for term in normalized}) != len(normalized):
            raise ValueError("keyterms must be unique")
        return normalized


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


class ReturnsPolicy(BaseModel):
    """Return-eligibility policy; merchant-set within platform bounds (Group C).

    `window_days` counts from DELIVERY (a shipped-not-yet-delivered order is always in
    window). It is ENFORCED by the returns guardrail, so its spoken sentence is DERIVED
    from this value (agents/spoken_policy.py) — never restated in `spoken_facts_extra`.
    Bounded by `_platform.limits.return_window_max_days` in the resolver."""

    model_config = _STRICT

    window_days: int = Field(ge=1, default=30)


# Default caller-silence window before a paused confirmation (checkout/refund/cancel) expires
# — a synchronous voice call, so short. Merchant may tune it within the platform max
# (_platform.limits.pending_confirmation_ttl_max_seconds, resolver-enforced).
_DEFAULT_PENDING_TTL_SECONDS = 120.0


class ClarificationReaskPolicy(BaseModel):
    """Per-flow conversational-liveness budgets.

    Values count additional questions after the initial clarification. The fields are required:
    effective values come from layered config, never a second set of source-code defaults.
    """

    model_config = _STRICT

    identity: int = Field(ge=0)
    support: int = Field(ge=0)
    cart: int = Field(ge=0)


class SecurityPolicy(BaseModel):
    """Attempt-BUDGET knobs — merchant-set within platform CEILINGS (_platform.limits, the
    resolver clamps loudly). Every knob here is a budget where a LARGER value WEAKENS security
    (more guesses / more probe room / a later human offer / a longer tool loop), so the
    ceiling caps the weakening — the mirror of the money knobs, where the ceiling caps a
    refund the merchant would auto-approve.

    NOT here (deliberately): the verification-LEVEL floors (refund/profile/identity_required_
    level, dtos/confirmation.py) are fraud/ATO floors that live in code with no merchant knob
    — a YAML copy would be the drifting second source of truth the removed
    `ToolConfirmationPolicy.min_verification_level` already was. Attempt budgets are tunable;
    the level a factor must reach is not."""

    model_config = _STRICT

    # OTP re-collect budget (shared by refund / profile / identity step-up). More tries =
    # more guessing room against a 6-digit code.
    otp_max_attempts: int = Field(ge=1, default=2)
    # Identity-flow contact re-asks before the silent human handover. More = more match
    # probes at the contact directory.
    contact_reask_max: int = Field(ge=0, default=1)
    # Support-gate failed matches on an order before the corrective offers a person. Later =
    # more order/contact-pair probe room in one session.
    auth_denials_before_human_offer: int = Field(ge=1, default=2)
    # Frontline read-only tool round-trips per turn before the loop is broken (runaway /
    # cost bound, not an auth control — kept here so all conversation budgets live together).
    max_tool_hops: int = Field(ge=1, default=5)


class PolicyConfig(BaseModel):
    """Merchant policy values — set within platform-enforced bounds (never disable a cap)."""

    model_config = _STRICT

    max_order_value_usd: float = Field(ge=0)
    refunds: RefundPolicy
    # Defaults so existing merchant YAMLs (which don't set it) get the platform default.
    returns: ReturnsPolicy = Field(default_factory=ReturnsPolicy)
    allow_ai_merchant_handoff: StrictBool
    # Optional: merchant-tunable within the platform max. Defaults so existing merchant YAMLs
    # (which don't set it) keep the platform default; the resolver clamps the ceiling.
    pending_confirmation_ttl_seconds: float = Field(default=_DEFAULT_PENDING_TTL_SECONDS, gt=0)
    # Attempt-budget security knobs (default_factory so existing merchant YAMLs, which don't
    # set it, keep the platform defaults; the resolver clamps each against its ceiling).
    security: SecurityPolicy = Field(default_factory=SecurityPolicy)
    # Conversational liveness is separate from SecurityPolicy: Cart questions are not auth
    # evidence, even though all three flows share one tracker implementation.
    clarification_reask_max: ClarificationReaskPolicy
    # Max orders in one cancel batch (F-16.2). The required value comes from base.yaml and its
    # path is safety-locked, so template/override layers cannot tune it. Keeping this REQUIRED
    # prevents a second, silent source-code default from drifting away from the base layer.
    cancel_batch_max: int = Field(ge=1)
    # Optional merchant free-text policy facts that have NO enforcing field — refund
    # TIMELINE ("5-7 business days"), condition clauses ("original condition"). The
    # ENFORCED policy sentences (returnless threshold, return window, human-review line)
    # are DERIVED from the typed values in agents/spoken_policy.py, never retyped here
    # (drift guard). Keep this to facts nothing else in the config represents; absent =>
    # only the derived sentences are spoken. NEVER restate an enforced number here.
    spoken_facts_extra: str | None = None

    def to_policy_context(self) -> PolicyContext:
        """The ONE config->runtime mapping (Group C consolidation): every PolicyContext a
        production path builds comes through here, so a new policy field is added in exactly
        one place — the previous per-site field lists (pipeline + tenancy) drifted one field
        at a time."""
        # Local import: state.py pulls in langchain/langgraph; keeping it deferred lets the
        # config layer (resolver/registry) stay importable without the LLM stack. Not a cycle.
        from agnostic_market.dtos.state import PolicyContext

        return PolicyContext(
            max_order_value_usd=self.max_order_value_usd,
            allow_ai_merchant_handoff=self.allow_ai_merchant_handoff,
            refund_auto_approve_under_usd=self.refunds.auto_approve_under_usd,
            refund_require_human_above_usd=self.refunds.require_human_above_usd,
            refund_returnless_under_usd=self.refunds.returnless_under_usd,
            return_window_days=self.returns.window_days,
            pending_ttl_seconds=self.pending_confirmation_ttl_seconds,
            spoken_policy_extra=self.spoken_facts_extra,
            otp_max_attempts=self.security.otp_max_attempts,
            contact_reask_max=self.security.contact_reask_max,
            auth_denials_before_human_offer=self.security.auth_denials_before_human_offer,
            max_tool_hops=self.security.max_tool_hops,
            identity_clarification_reask_max=self.clarification_reask_max.identity,
            support_clarification_reask_max=self.clarification_reask_max.support,
            cart_clarification_reask_max=self.clarification_reask_max.cart,
            cancel_batch_max=self.cancel_batch_max,
        )


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
