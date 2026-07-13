"""Critical-value confirmation as a TYPED policy surface (VOICE_PIPELINE §7a).

Prose ("read back the important values") invites inconsistent implementation; this schema
is what the code enforces: each sensitive tool DECLARES which critical fields it binds and
the confirmation strength required before it may execute. The confirm node refuses to fire
its interrupt unless the declared fields are present in the readback — so readback can't
be silently forgotten when a new sensitive tool is added.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

_FROZEN = ConfigDict(extra="forbid", frozen=True)

# The platform's critical-value vocabulary (VOICE_PIPELINE §7): values where one STT error
# becomes a wrong charge/order. A starting set by design — extending it is an explicit,
# reviewable change here, never an ad-hoc string elsewhere.
CRITICAL_CONFIRMATION_FIELDS: frozenset[str] = frozenset(
    {
        "quantity",
        "line_items",  # a whole-cart line summary ("2 rain jackets and 1 pair of socks") —
        # a single `quantity` can't honestly describe N lines (Group B).
        "total_amount",
        "order_id",
        "new_address",
        "new_payment_instrument_ref",
        "new_contact",
    }
)

ConfirmationStrength = Literal["readback", "explicit_yes", "double_confirm"]


class ToolConfirmationPolicy(BaseModel):
    """One sensitive tool's declared confirmation contract (enforced in code, not prompt).

    Deliberately does NOT declare a verification level: the level authority for refunds is
    `refund_required_level` (destination-aware platform code, enforced live in the flow) —
    a static per-tool minimum here would be a second, unenforced source of truth that
    drifts (it did: the removed `min_verification_level=2` contradicted the L1
    refund-to-original path the moment Group A landed).
    """

    model_config = _FROZEN

    tool: str = Field(min_length=1)
    confirm_fields: frozenset[str] = Field(min_length=1)
    strength: ConfirmationStrength

    @field_validator("confirm_fields")
    @classmethod
    def _fields_are_critical(cls, value: frozenset[str]) -> frozenset[str]:
        unknown = value - CRITICAL_CONFIRMATION_FIELDS
        if unknown:
            raise ValueError(
                f"confirm_fields {sorted(unknown)} not in CRITICAL_CONFIRMATION_FIELDS - "
                "extend the vocabulary explicitly if a new critical field is needed"
            )
        return value


# --- Refund destination as a first-class policy dimension (AGENTS §A4b, DESIGN_REVIEW C4) ---
# The refund *destination* — not just the amount — decides the required verification level.
# "original" = back to the instrument that paid; "new_instrument"/"new_address" = anywhere
# else. A new destination is the classic "refund $40 to a new card auto-approves" fraud hole.
RefundDestination = Literal["original", "new_instrument", "new_address"]


def refund_required_level(amount_usd: float, destination: RefundDestination) -> int:
    """The minimum verification level a refund needs — PLATFORM code, not a merchant knob.

    §A4b (a fraud FLOOR, un-removable): refund to a NEW instrument/address requires L2
    **regardless of amount** — closes the amount-only gate. Refund to the ORIGINAL
    instrument is L1. Amount thresholds (auto-approve / human-review) live separately in
    the merchant's config RefundPolicy; this function answers only "what identity strength
    does the destination demand," which no merchant may lower.
    """
    if destination in ("new_instrument", "new_address"):
        return 2
    return 1


# issue_refund's declared confirmation contract (VOICE_PIPELINE §7a): the readback MUST
# speak these critical fields at explicit_yes strength. `return_id` is deliberately NOT
# declared — the store mints it AFTER the effect, so it cannot exist at readback time (it
# is spoken in the outcome line instead); declaring it made the contract check theater.
# The verification level is gated separately by `refund_required_level` (live, in-flow).
ISSUE_REFUND_POLICY = ToolConfirmationPolicy(
    tool="issue_refund",
    confirm_fields=frozenset({"total_amount", "new_payment_instrument_ref"}),
    strength="explicit_yes",
)
