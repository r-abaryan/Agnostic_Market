"""Session/reasoning state DTOs.

Phase 0 defines ONLY the per-session tenant-binding fields that config + tenancy need now.
The full `ReasoningState` (AGENTS.md §A1) and `HandoffCarry` (AGENTS.md handoff contract)
are built in Phase 3 with the reasoning graph — stubbed here with pointers so nothing
imports a half-defined shape prematurely.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

_FROZEN = ConfigDict(extra="forbid", frozen=True)


class PolicyContext(BaseModel):
    """Loaded merchant policy/limits + flags, bound at session start, read-only per session.

    Phase 0: a thin immutable carrier of the resolved policy values the tenant context needs.
    Extend (limits/flags surface) as the guardrail node consumes it in Phase 3.
    """

    model_config = _FROZEN

    max_order_value_usd: float = Field(ge=0)
    allow_ai_merchant_handoff: bool


# --- Deferred to Phase 3 (AGENTS.md §A1 / handoff contract) ---
# ReasoningState — full graph state (messages, cart, pending_action, interrupt, carry, ...).
# HandoffCarry   — the session notepad (intent, recent_context, verification_level, ...).
# Built with the reasoning graph, not now; kept out of Phase 0 to avoid a half-defined shape.
