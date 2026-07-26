"""Session/reasoning state DTOs.

Phase 0 defined the per-session tenant-binding fields (PolicyContext). Phase 3a adds the
minimal reasoning-graph state (AGENTS.md §A1): `messages` + a `handover` signal. The full
state surface (cart, pending_action, interrupt, carry, route_history, verification_level)
lands in its consuming phase — 3b (checkout/interrupt), 3c (verification), 3d (planner) —
kept out here so nothing imports a half-defined shape prematurely.
"""

from __future__ import annotations

from typing import Annotated, Literal

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field, model_validator

from agnostic_market.dtos.confirmation import ProfileField, RefundDestination
from agnostic_market.dtos.orchestration import (
    CancellableOrderScope,
    IntentRequest,
)
from agnostic_market.dtos.recovery import PendingRecovery

_FROZEN = ConfigDict(extra="forbid", frozen=True)

# Handover destinations (tiers). Only "support"/"human" have a spoken sink in 3a; the rest
# are declared now so the tool schema is stable, and get real flows in 3b/3c/3d.
HandoffDestination = Literal["support", "checkout", "planner", "human"]

# Why a handover fired — a CLOSED enum, never free prose: it is logged (HandoffRequested),
# so prose here would be a PII echo channel (a model-written reason could carry an address).
# Free-text detail, if ever needed, lives in the session payload / telemetry, not this field.
HandoffReasonCode = Literal[
    "address_change",
    "contact_change",  # phone/email on file — the OTP factor itself (Group C profile flow)
    "payment_change",
    "cancel_order",
    "refund",
    "cart_write",
    "list_orders",  # account-wide order enumeration — needs OTP-bound identity (P7)
    "multi_step",
    "verification_required",
    "switch_account",
    "other",
]


class PolicyContext(BaseModel):
    """Loaded merchant policy/limits + flags, bound at session start, read-only per session.

    The resolved policy values the GRAPH reads at decision time (closed into flow nodes at
    build, §A5) — the runtime face of `PolicyConfig`. Refund thresholds + the pending TTL are
    carried here so the support/checkout guardrails can enforce them (they are merchant-tuned
    within platform bounds — the resolver already clamped them).
    """

    model_config = _FROZEN

    max_order_value_usd: float = Field(ge=0)
    allow_ai_merchant_handoff: bool
    refund_auto_approve_under_usd: float = Field(ge=0)
    refund_require_human_above_usd: float = Field(ge=0)
    # Return-first floor for SHIPPED/DELIVERED refunds: above this, the refund waits for
    # the return (industry standard); at/below it may pay out returnless.
    refund_returnless_under_usd: float = Field(ge=0)
    # Return-eligibility window in days, counted from DELIVERY (Group C; shipped-not-yet-
    # delivered is always in window). REQUIRED (no default) on purpose: PolicyContext is
    # constructed at several sites in lockstep — a site that forgets a new field must fail
    # loudly at build, never run with a silent default.
    return_window_days: int = Field(ge=1)
    pending_ttl_seconds: float = Field(gt=0)
    # Attempt-budget security knobs (config `policies.security.*`, resolver-clamped). REQUIRED,
    # no default — the same lockstep discipline as return_window_days: a construction site that
    # forgets one fails loudly. Consumed by the step-up collect loop (otp_max_attempts), the
    # identity contact re-ask (contact_reask_max), the support gate corrective
    # (auth_denials_before_human_offer), and the frontline tool-hop loop breaker (max_tool_hops).
    otp_max_attempts: int = Field(ge=1)
    contact_reask_max: int = Field(ge=0)
    auth_denials_before_human_offer: int = Field(ge=1)
    max_tool_hops: int = Field(ge=1)
    # Consecutive code-authored questions allowed after the initial clarification.
    identity_clarification_reask_max: int = Field(ge=0)
    support_clarification_reask_max: int = Field(ge=0)
    cart_clarification_reask_max: int = Field(ge=0)
    # Max orders in one cancel batch — a SAFETY bound (a big batch must fit the LangGraph
    # recursion/step budget), NOT a merchant knob: it is `_safety_locked` in config, so no
    # tenant can touch it. Over-cap asks the caller to narrow, never silently takes the first
    # N. REQUIRED (lockstep, like the security knobs above).
    cancel_batch_max: int = Field(ge=1)
    # Merchant free-text policy extras (config `policies.spoken_facts_extra`) — facts with
    # NO enforcing field (refund timeline, return conditions). The ENFORCED sentences are
    # DERIVED from the typed values above (agents/spoken_policy.py); this is only the
    # free-text tail. None = no extras (the derived sentences are still spoken).
    spoken_policy_extra: str | None = None


class CartLine(BaseModel):
    """One line item in the session cart (Group B). CODE-resolved from the candidate list —
    `sku`/`name`/`price_usd` are never model-authored; the model picks a candidate KEY and
    code resolves it. Lives here (not commerce) because it is CHECKPOINTED inside
    `PendingPlacement`, and every checkpointed type lives in `dtos` (the base layer).
    `line_total` is DERIVED (price times quantity), never stored — no model arithmetic.
    """

    model_config = _FROZEN

    sku: str = Field(min_length=1)
    name: str = Field(min_length=1)
    price_usd: float = Field(ge=0)
    quantity: int = Field(ge=1)

    @property
    def line_total(self) -> float:
        return round(self.price_usd * self.quantity, 2)


class PendingPlacement(BaseModel):
    """A whole-cart order awaiting HITL confirmation (AGENTS §A10) — the FROZEN snapshot of
    the cart at confirm-entry, minted BEFORE the interrupt (A10a rule 2). Replaces the
    single-line `PendingAction` (Group B): the readback + place effect now cover N lines.

    The snapshot is what the confirm/place nodes read on replay — NEVER the live cart — so
    consent is always over exactly what was read back, even if the cart is mutated later.
    `lines`/`total_usd` are CODE-resolved (never model arithmetic); `idempotency_key` makes
    the place effect dedupable by the order SoR on any replay. (The re-confirm loop needs no
    counter: it is bounded STRUCTURALLY by the confirm node's two fixed interrupt call sites
    — A10a rule 3.)
    """

    model_config = _FROZEN

    lines: tuple[CartLine, ...] = Field(min_length=1)
    total_usd: float = Field(ge=0)
    idempotency_key: str = Field(min_length=1)
    created_at: float  # unix seconds; Clock-A expiry is checked against this on resume
    # Set by the guardrail when a LIVE identical order already exists this session: the
    # readback must disambiguate ("this would be a SECOND order"), never read identically
    # to the first. Carries the existing order's id.
    duplicate_of: str | None = None


class PendingRefund(BaseModel):
    """A refund awaiting step-up verification and/or HITL confirmation (AGENTS §A4b/§A10a).

    Same A10a discipline as PendingPlacement: `idempotency_key` is per-refund-INTENT (never
    derived from `order_id`, so a second legitimate PARTIAL refund is not silently deduped
    as a replay); `amount_usd`/`instrument_ref` are CODE-resolved, never model arithmetic.
    `destination` drives the required verification level in code (new instrument => L2
    regardless of amount, §A4b — the fraud floor). `attempt_key` keys the idempotent OTP
    dispatch (a replayed dispatch node must not re-send); `otp_tries` bounds the re-collect
    loop deterministically (A10a rule 3). The refund `instrument_ref` is a STORED-instrument
    reference, never a raw PAN entered by voice (PCI boundary, SECURITY §6).
    """

    model_config = _FROZEN

    order_id: str = Field(min_length=1)
    summary: str = Field(min_length=1)  # the human order description, spoken in the readback so a
    # wrong owned-order selection is caught (INV-32: ownership alone can't tell WHICH owned order).
    amount_usd: float = Field(ge=0)
    destination: RefundDestination
    instrument_ref: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)
    attempt_key: str = Field(min_length=1)
    otp_tries: int = Field(ge=0, default=0)
    created_at: float


# --- the cancel lifecycle (F-16.2 batch-aware cancel) --------------------------------------
# ONE lifecycle for one-or-many order cancellations (a single cancel is a one-target batch —
# no scalar/batch fork for a future planner to bridge). Two phases, a discriminated union:
#   CancellableOrderScope — PRE-auth: a SEMANTIC scope ("all/both my cancellable orders") the
#                       caller stated with no ids, retained across an identity detour (the
#                       unbound "cancel all" case, Milestone B). Carries NO ids/keys/contact.
#   PendingCancelBatch — POST-resolve: the FROZEN, authorized, preflighted target set the
#                       readback + serialized void operate on.

# The per-target cancel result — a CLOSED enum (checkpoint/planner-facing state follows the
# closed-slug rule; free prose here would be a PII echo channel like HandoffReasonCode).
CancelOutcomeCode = Literal[
    "cancelled",  # the store voided it (an amount is present, from the CancelRecord)
    "already_cancelled",  # preflight: the order was already cancelled
    "not_cancellable",  # preflight: shipped/delivered — no self-serve void (a return instead)
    "has_refunds",  # preflight: a refund already exists — a void would double-return; a person
    "store_refused",  # effect-time refusal (stale since preflight); CancelError carries no
    # typed reason, so a rare effect-time refusal collapses to this generic honest code
    "not_completed",  # recovery: this target was not committed or was never attempted after
    # an earlier target failed; makes no claim about the order's global/current state
]


class BatchCancelOutcome(BaseModel):
    """One target's cancel result — the STRUCTURED per-order fact the final line is rendered
    from (INV-25: never inferred from pending intent). Lives here (not commerce) because it is
    CHECKPOINTED inside PendingCancelBatch and every checkpointed type is a dtos base-layer
    type; commerce/orders.py imports it (as it already imports CartLine)."""

    model_config = _FROZEN

    order_id: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    outcome: CancelOutcomeCode
    # Present ONLY for a "cancelled" outcome (the CancelRecord's reversed total) — a refusal
    # returns no record, so no amount. Optional by construction, not laziness.
    amount_usd: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def amount_matches_outcome(self) -> BatchCancelOutcome:
        if self.outcome == "cancelled" and self.amount_usd is None:
            raise ValueError("a cancelled outcome requires amount_usd from the CancelRecord")
        if self.outcome != "cancelled" and self.amount_usd is not None:
            raise ValueError("only a cancelled outcome may carry amount_usd")
        return self


class CancelTarget(BaseModel):
    """One order in a cancel batch. `summary` is CODE-resolved from the order the model picked
    by KEY (never a model-authored id); `idempotency_key` is per-TARGET-INTENT so a replayed
    void returns the SAME cancel per order (the OrderStore is the dedup arbiter) and a batch
    of N produces exactly N effects, never N+1."""

    model_config = _FROZEN

    order_id: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)


class PendingCancelBatch(BaseModel):
    """POST-resolve cancel batch awaiting HITL confirmation then serialized void (AGENTS
    §A10a). The FROZEN authorized+preflighted target set: `targets` is the ELIGIBLE subset the
    readback names and the void walks (one effect per node completion, checkpointed BETWEEN
    writes); `ineligible` are the preflight declines stated at the readback; `outcomes`
    accumulates completed per-target results (replay-safe progress — a kill mid-batch resumes
    with the still-pending targets, the done ones already recorded). `created_at` drives the
    Clock-A TTL. A single-order cancel is a one-target batch (`targets` length 1)."""

    model_config = _FROZEN

    selector: Literal["resolved_batch"] = "resolved_batch"
    targets: tuple[CancelTarget, ...] = Field(min_length=1)
    ineligible: tuple[BatchCancelOutcome, ...] = ()
    outcomes: tuple[BatchCancelOutcome, ...] = ()
    created_at: float


# The pending_cancel channel is a discriminated union over the two phases (single = batch).
PendingCancel = Annotated[
    CancellableOrderScope | PendingCancelBatch, Field(discriminator="selector")
]


class PendingReturn(BaseModel):
    """A return/RMA awaiting HITL confirmation (Group C; PendingCancel's single-interrupt
    shape plus the honest money outcome).

    `refund_due_usd` is CODE-computed (captured minus refunds minus prior return promises —
    never model arithmetic) and is RECORDED on the return, not paid: the refund releases at
    the Phase-4 SoR once the return is processed, and that release is the money-movement
    moment (§A4c re-runs the destination->level check there). Deliberately NO destination/
    instrument fields: v1 returns always refund to the ORIGINAL payment method (industry
    standard) — the refund guardrail's return steer fires BEFORE the level gate, so carrying
    a caller-requested NEW instrument here would record a payout promise that skipped the
    §A4b L2 step-up. No attempt/tries fields — single interrupt, no step-up (L1 action).
    """

    model_config = _FROZEN

    order_id: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    refund_due_usd: float = Field(ge=0)
    idempotency_key: str = Field(min_length=1)
    created_at: float


class PendingProfileChange(BaseModel):
    """An address/contact change awaiting step-up verification + HITL confirmation (Group C;
    PendingRefund's step-up shape minus money).

    Always L2 (`profile_change_required_level`, the §A4a platform floor): address/contact
    are the account-takeover levers. `factor_ref` RECORDS the MASKED on-file contact at
    proposal time — for a contact change that is the OLD factor (changing the factor
    requires the factor, the ladder constraint). It is a Phase-4 breadcrumb: the stub OTP
    dispatches by `attempt_key` (idempotency), so nothing reads `factor_ref` yet; the
    real SoR delivery in Phase 4 sends to this recorded factor. `new_value` is the
    caller-stated value:
    SPOKEN in the readback/outcome, NEVER logged/telemetered (PII discipline) — and note it
    sits inside the CHECKPOINT, which is acceptable for the in-memory per-session saver
    (reaped on close) but must be revisited at the Phase-4 Redis swap (encrypt or exclude).
    `attempt_key` keys the idempotent OTP dispatch; `otp_tries` bounds the re-collect loop
    (A10a rule 3) — same discipline as PendingRefund.
    """

    model_config = _FROZEN

    # The OTP-BOUND customer this change belongs to (Fix 5 Milestone B) — captured from the live
    # binding at proposal time, re-validated against the live binding at effect time (§A4c), and
    # the scope key for the customer-owned profile read/update. Never a model argument.
    customer_ref: str = Field(min_length=1)
    field: ProfileField
    new_value: str = Field(min_length=1)
    factor_ref: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)
    attempt_key: str = Field(min_length=1)
    otp_tries: int = Field(ge=0, default=0)
    created_at: float


class PendingIdentity(BaseModel):
    """An identity verification awaiting step-up (P7 — the third `_stepup.py` family).

    Minted by the identity flow's assemble AFTER the caller's contact claim code-matched a
    customer. Carries the MASKED contact only (the factor_ref analogue). The raw claim is
    still present in checkpointed caller/model message history; this DTO adds no second raw-value
    field. `grants_at_mint` snapshots
    `len(verification_store.grants)` at mint: THE binding invariant. A stale cross-family L2
    (e.g. an earlier profile-flow OTP) satisfies the factory's level-only confirm check even
    when THIS chain's OTP failed — the flow's collect router and apply node require a NEW
    grant since mint (this chain's OTP actually succeeded) before binding a customer.

    Deliberately NO `idempotency_key` (the bind effect is a session-local set — naturally
    idempotent, no SoR effect to dedup) and NO `created_at` (identity has no confirm
    interrupt to TTL-check; its only interrupt is the factory's OTP collect, which no family
    TTL-checks — a stale-but-correct OTP is still the correct OTP, and abandonment is the
    Clock-B reaper's job).
    """

    model_config = _FROZEN

    customer_ref: str = Field(min_length=1)
    masked_contact: str = Field(min_length=1)
    attempt_key: str = Field(min_length=1)
    otp_tries: int = Field(ge=0, default=0)
    grants_at_mint: int = Field(ge=0)


# Flows a session can be "inside" across turns (entry router bypasses gate/frontline).
# "left_*" are TURN-SCOPED markers (reset at entry): the model left the flow this turn, so
# a same-turn re-entry is blocked — breaks the assemble->gate->handover->assemble cycle
# structurally instead of relying on the recursion limit. Each gated flow carries its own
# sticky value + left-twin so the entry router knows which flow's escape/stickiness rules
# apply. Group B: the "cart" flow owns both cart MUTATION and the whole-cart placement tail
# (the single-line "checkout" flow it replaces is gone — direct-buy normalizes through it).
ActiveFlow = Literal["cart", "left_cart", "support", "left_support", "identity", "left_identity"]


class IdentityClarification(BaseModel):
    """Turn-scoped instruction for Identity's code-authored clarification renderer."""

    model_config = _FROZEN

    flow: Literal["identity"] = "identity"
    detail: Literal["contact"] = "contact"


SupportQuestionDetail = Literal[
    "action",
    "order",
    "amount",
    "refund_destination",
    "profile_field",
    "profile_value",
]
SupportAuthorizationDetail = Literal["order_match", "order_match_human_help"]


class SupportClarification(BaseModel):
    """Turn-scoped instruction for Support's code-authored clarification renderer."""

    model_config = _FROZEN

    flow: Literal["support"] = "support"
    detail: SupportQuestionDetail | SupportAuthorizationDetail


CartClarificationDetail = Literal["action", "item", "quantity"]


class CartClarification(BaseModel):
    """Turn-scoped instruction for Cart's code-authored clarification renderer."""

    model_config = _FROZEN

    flow: Literal["cart"] = "cart"
    detail: CartClarificationDetail


# Selector only: no caller value, target, authority, or multi-turn progress belongs here.
# Each owning flow will map its closed detail value to platform-authored copy before speech.
PendingClarification = Annotated[
    IdentityClarification | SupportClarification | CartClarification,
    Field(discriminator="flow"),
]


ClarificationOwner = Literal["identity", "support", "cart"]


class ClarificationProgress(BaseModel):
    """Consecutive clarification questions for one sticky flow engagement."""

    model_config = _FROZEN

    flow: ClarificationOwner
    reasks: int = Field(ge=0)


class HandoffRequest(BaseModel):
    """A frontline decision to hand a turn to a higher tier (AGENTS.md handover boundary).

    Set by either the deterministic gate (`source="gate"`, pre-generation) or the model's
    `request_handover` tool call (`source="model"`). Carries no free text — `reason_code`
    is a closed enum so the logged handover cannot leak caller PII.
    """

    model_config = _FROZEN

    destination: HandoffDestination
    reason_code: HandoffReasonCode
    source: Literal["gate", "model"]


class ReasoningState(BaseModel):
    """The reasoning graph's state (Phase 3a/3b slice of AGENTS.md §A1).

    `messages` uses the append reducer. `handover`, when set, routes the turn to the
    handover sink. `pending_placement` + `active_flow` carry an in-flight placement across
    the HITL interrupt and across turns (the thread is checkpointed from 3b on).
    Every non-`messages` field MUST default: the engine feeds turns as
    `{"messages": [<new user message>]}` deltas.
    """

    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)
    handover: HandoffRequest | None = None
    # Session-terminal automation state. Set only by the shared human handover and retained
    # until the caller lifecycle deletes the checkpoint; it is never an identity/account block.
    automation_terminal: bool = False
    pending_placement: PendingPlacement | None = None
    pending_refund: PendingRefund | None = None
    pending_cancel: PendingCancel | None = None
    pending_return: PendingReturn | None = None
    pending_profile_change: PendingProfileChange | None = None
    pending_identity: PendingIdentity | None = None
    # Sole cross-identity continuation: caller-stated intent only, never resolved authority.
    pending_request: IntentRequest | None = None
    # PII-free failure disposition. Ordinary node handlers consume it in-graph; the engine
    # may seed the same channel after an externally abandoned stream in Milestone 6D.
    pending_recovery: PendingRecovery | None = None
    # Bounded re-ask counter for the identity flow's contact claim (P7 decision 4: ONE
    # softened re-ask on a no-match — STT mishears emails constantly — then a silent human
    # handover). Spans turns WITHIN the sticky flow; cleared on every flow exit (apply,
    # abort, escape, leave, cross-switch, terminal handover).
    identity_claim_misses: int = 0
    active_flow: ActiveFlow | None = None
    # Turn-scoped, CODE-authored spoken line the cart flow's assemble hands to the speakable
    # `cart_ack` node (mutation acks, the review_cart listing, the empty-cart response).
    # Separate from the closed clarification selector below because these lines carry dynamic
    # cart contents/totals. Reset at entry_node; cleared by cart_ack (clear-before-speak).
    pending_ack: str | None = None
    # Turn-scoped instruction for a flow-owned, code-authored clarification line. Each
    # transactional flow writes its own selector atomically as it yields to its renderer.
    pending_clarification: PendingClarification | None = None
    # Engagement-scoped liveness tracker. Entry preserves it while the same sticky flow
    # continues; real progress and every lifecycle exit clear it.
    clarification_progress: ClarificationProgress | None = None
