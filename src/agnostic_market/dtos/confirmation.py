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
        "total_amount",
        "order_id",
        "return_id",
        "new_address",
        "new_payment_instrument_ref",
        "new_contact",
    }
)

ConfirmationStrength = Literal["readback", "explicit_yes", "double_confirm"]


class ToolConfirmationPolicy(BaseModel):
    """One sensitive tool's declared confirmation contract (enforced in code, not prompt)."""

    model_config = _FROZEN

    tool: str = Field(min_length=1)
    confirm_fields: frozenset[str] = Field(min_length=1)
    strength: ConfirmationStrength
    min_verification_level: int = Field(ge=0, le=2)

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
