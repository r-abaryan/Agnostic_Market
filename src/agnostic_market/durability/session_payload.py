"""Validated encrypted payload for restorable session-owned state."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agnostic_market.commerce.cart import CartMutationRecord, CartSessionState
from agnostic_market.commerce.orders import RecentOrderSnapshot
from agnostic_market.dtos.session import AuthorityIdentifier

SESSION_PAYLOAD_SCHEMA_VERSION = 1
SESSION_OPERATION_RESULT_SCHEMA_VERSION = 1
_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True)


class PrincipalRetirementMarker(BaseModel):
    model_config = _STRICT

    transition_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def transition_id_is_nonblank(self) -> PrincipalRetirementMarker:
        if not self.transition_id.strip():
            raise ValueError("principal retirement transition id must not be blank")
        return self


class DurableSessionPayload(BaseModel):
    model_config = _STRICT

    schema_version: Literal[1] = SESSION_PAYLOAD_SCHEMA_VERSION
    cart: CartSessionState = Field(default_factory=CartSessionState)
    recent_orders: RecentOrderSnapshot = Field(
        default_factory=lambda: RecentOrderSnapshot(
            focused_order_ref=None,
            order_refs=(),
            operation=None,
            outcomes=(),
            complete=True,
        )
    )
    guest_order_refs: tuple[str, ...] = ()
    principal_retirement: PrincipalRetirementMarker | None = None

    @model_validator(mode="after")
    def guest_order_refs_are_canonical(self) -> Self:
        if len(self.guest_order_refs) != len(set(self.guest_order_refs)):
            raise ValueError("durable session payload contains duplicate guest orders")
        if any(not ref or ref != ref.strip().upper() for ref in self.guest_order_refs):
            raise ValueError("guest order references must be canonical")
        return self

    def to_bytes(self) -> bytes:
        return self.model_dump_json().encode("utf-8")

    @classmethod
    def from_bytes(cls, payload: bytes) -> DurableSessionPayload:
        return cls.model_validate_json(payload)


class EmptySessionOperationResult(BaseModel):
    model_config = _STRICT

    kind: Literal["none"] = "none"


class CartMutationSessionOperationResult(BaseModel):
    model_config = _STRICT

    kind: Literal["cart_mutation"] = "cart_mutation"
    record: CartMutationRecord


SessionOperationResult = Annotated[
    EmptySessionOperationResult | CartMutationSessionOperationResult,
    Field(discriminator="kind"),
]


class SessionOperationReceiptPayload(BaseModel):
    model_config = _STRICT

    schema_version: Literal[1] = SESSION_OPERATION_RESULT_SCHEMA_VERSION
    operation_id: AuthorityIdentifier
    request_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    result: SessionOperationResult

    def to_bytes(self) -> bytes:
        return self.model_dump_json().encode("utf-8")

    @classmethod
    def from_bytes(cls, payload: bytes) -> SessionOperationReceiptPayload:
        return cls.model_validate_json(payload)
