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
from pydantic import BaseModel, ConfigDict, Field

_FROZEN = ConfigDict(extra="forbid", frozen=True)

# Handover destinations (tiers). Only "support"/"human" have a spoken sink in 3a; the rest
# are declared now so the tool schema is stable, and get real flows in 3b/3c/3d.
HandoffDestination = Literal["support", "checkout", "planner", "human"]

# Why a handover fired — a CLOSED enum, never free prose: it is logged (HandoffRequested),
# so prose here would be a PII echo channel (a model-written reason could carry an address).
# Free-text detail, if ever needed, lives in the session payload / telemetry, not this field.
HandoffReasonCode = Literal[
    "address_change",
    "payment_change",
    "cancel_order",
    "refund",
    "cart_write",
    "multi_step",
    "verification_required",
    "other",
]


class PolicyContext(BaseModel):
    """Loaded merchant policy/limits + flags, bound at session start, read-only per session.

    Phase 0: a thin immutable carrier of the resolved policy values the tenant context needs.
    Extend (limits/flags surface) as the guardrail node consumes it in Phase 3.
    """

    model_config = _FROZEN

    max_order_value_usd: float = Field(ge=0)
    allow_ai_merchant_handoff: bool


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
    """The frontline graph's state (Phase 3a slice of AGENTS.md §A1).

    `messages` uses the append reducer. `handover`, when set, routes the turn to the
    handover sink. Every non-`messages` field MUST default: livekit's LLMAdapter invokes
    the graph with `{"messages": [...]}` only.
    """

    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)
    handover: HandoffRequest | None = None
