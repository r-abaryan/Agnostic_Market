"""Typed contracts shared by semantic routing, continuation, and capability execution.

Milestone 1 deliberately has no planner DTOs. Dependent work is an unsupported workflow until
Milestone 6 adds the ``plan`` route arm, ``ExecutionPlan``, and ``PlanStep``.
"""

from __future__ import annotations

import uuid
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StringConstraints,
    ValidationError,
    model_validator,
)

from agnostic_market.dtos.confirmation import ProfileField, RefundDestination

_FROZEN = ConfigDict(extra="forbid", frozen=True)
NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
CancelScope = Literal["all_cancellable", "both_cancellable"]


class CapabilityId(StrEnum):
    ANSWER_QUESTION = "answer_question"
    SEARCH_CATALOG = "search_catalog"
    VERIFY_ORDER_STATUS = "verify_order_status"
    LIST_ORDERS = "list_orders"
    VIEW_CART = "view_cart"
    MODIFY_CART = "modify_cart"
    PLACE_ORDER = "place_order"
    CANCEL_ORDERS = "cancel_orders"
    REFUND_ORDER = "refund_order"
    RETURN_ORDER = "return_order"
    CHANGE_PROFILE = "change_profile"
    VERIFY_IDENTITY = "verify_identity"
    SWITCH_ACCOUNT = "switch_account"
    REQUEST_PERSON = "request_person"


class ExplicitOrderTarget(BaseModel):
    model_config = _FROZEN

    selector: Literal["explicit"] = "explicit"
    order_ref: NonEmptyText


class FocusedOrderTarget(BaseModel):
    model_config = _FROZEN

    selector: Literal["focused"] = "focused"


OrderTarget = Annotated[
    ExplicitOrderTarget | FocusedOrderTarget,
    Field(discriminator="selector"),
]


class ExplicitOrderSet(BaseModel):
    model_config = _FROZEN

    selector: Literal["explicit_set"] = "explicit_set"
    order_refs: tuple[NonEmptyText, ...] = Field(min_length=1)


class FocusedOrderSet(BaseModel):
    model_config = _FROZEN

    selector: Literal["focused"] = "focused"


class RecentOrderSet(BaseModel):
    model_config = _FROZEN

    selector: Literal["recent"] = "recent"
    cardinality: Literal["one", "all"]


OrderStatusSelector = Annotated[
    ExplicitOrderSet | FocusedOrderSet | RecentOrderSet,
    Field(discriminator="selector"),
]


class CancellableOrderScope(BaseModel):
    model_config = _FROZEN

    selector: Literal["cancellable_scope"] = "cancellable_scope"
    scope: CancelScope


CancelSelector = Annotated[
    ExplicitOrderSet | FocusedOrderSet | CancellableOrderScope,
    Field(discriminator="selector"),
]


class AnswerQuestion(BaseModel):
    model_config = _FROZEN

    kind: Literal[CapabilityId.ANSWER_QUESTION] = CapabilityId.ANSWER_QUESTION
    topic: Literal["policy", "general"]


class SearchCatalog(BaseModel):
    model_config = _FROZEN

    kind: Literal[CapabilityId.SEARCH_CATALOG] = CapabilityId.SEARCH_CATALOG
    query: NonEmptyText


class VerifyOrderStatus(BaseModel):
    model_config = _FROZEN

    kind: Literal[CapabilityId.VERIFY_ORDER_STATUS] = CapabilityId.VERIFY_ORDER_STATUS
    target: OrderStatusSelector


class ListOrders(BaseModel):
    model_config = _FROZEN

    kind: Literal[CapabilityId.LIST_ORDERS] = CapabilityId.LIST_ORDERS
    scope: Literal["session", "account"]


class ViewCart(BaseModel):
    model_config = _FROZEN

    kind: Literal[CapabilityId.VIEW_CART] = CapabilityId.VIEW_CART


class ModifyCart(BaseModel):
    model_config = _FROZEN

    kind: Literal[CapabilityId.MODIFY_CART] = CapabilityId.MODIFY_CART
    operation: Literal["add", "remove", "set_quantity"]
    item_query: NonEmptyText
    quantity: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def quantity_matches_operation(self) -> ModifyCart:
        if self.operation == "remove" and self.quantity is not None:
            raise ValueError("remove must not carry a quantity")
        if self.operation == "add" and (self.quantity is None or self.quantity < 1):
            raise ValueError("add requires a positive quantity")
        if self.operation == "set_quantity" and self.quantity is None:
            raise ValueError("set_quantity requires a quantity")
        return self


class PlaceOrder(BaseModel):
    model_config = _FROZEN

    kind: Literal[CapabilityId.PLACE_ORDER] = CapabilityId.PLACE_ORDER


class CancelOrders(BaseModel):
    model_config = _FROZEN

    kind: Literal[CapabilityId.CANCEL_ORDERS] = CapabilityId.CANCEL_ORDERS
    target: CancelSelector


class RefundOrder(BaseModel):
    model_config = _FROZEN

    kind: Literal[CapabilityId.REFUND_ORDER] = CapabilityId.REFUND_ORDER
    target: OrderTarget
    amount_usd: float | None = Field(default=None, gt=0)
    destination: RefundDestination | None = None


class ReturnOrder(BaseModel):
    model_config = _FROZEN

    kind: Literal[CapabilityId.RETURN_ORDER] = CapabilityId.RETURN_ORDER
    target: OrderTarget


class ChangeProfile(BaseModel):
    model_config = _FROZEN

    kind: Literal[CapabilityId.CHANGE_PROFILE] = CapabilityId.CHANGE_PROFILE
    field: ProfileField
    new_value: NonEmptyText | None = None


class VerifyIdentity(BaseModel):
    model_config = _FROZEN

    kind: Literal[CapabilityId.VERIFY_IDENTITY] = CapabilityId.VERIFY_IDENTITY


class SwitchAccount(BaseModel):
    model_config = _FROZEN

    kind: Literal[CapabilityId.SWITCH_ACCOUNT] = CapabilityId.SWITCH_ACCOUNT


class RequestPerson(BaseModel):
    model_config = _FROZEN

    kind: Literal[CapabilityId.REQUEST_PERSON] = CapabilityId.REQUEST_PERSON


IntentRequest = Annotated[
    AnswerQuestion
    | SearchCatalog
    | VerifyOrderStatus
    | ListOrders
    | ViewCart
    | ModifyCart
    | PlaceOrder
    | CancelOrders
    | RefundOrder
    | ReturnOrder
    | ChangeProfile
    | VerifyIdentity
    | SwitchAccount
    | RequestPerson,
    Field(discriminator="kind"),
]

ClarificationReason = Literal[
    "ambiguous_intent",
    "missing_target",
    "missing_value",
    "unsupported_workflow",
    "unsupported_capability",
    "invalid_output",
]


class RouteDecision(BaseModel):
    """Validated Milestone-1 route: exactly one direct request or one clarification."""

    model_config = _FROZEN

    decision: Literal["clarify", "direct"]
    request: IntentRequest | None = None
    clarification_reason: ClarificationReason | None = None

    @model_validator(mode="after")
    def payload_matches_decision(self) -> RouteDecision:
        if self.decision == "clarify":
            if self.request is not None or self.clarification_reason is None:
                raise ValueError("clarify requires only a clarification_reason")
        elif self.request is None or self.clarification_reason is not None:
            raise ValueError("direct requires only one request")
        return self

    @classmethod
    def clarify(cls, reason: ClarificationReason) -> RouteDecision:
        return cls(decision="clarify", clarification_reason=reason)

    @classmethod
    def direct(cls, request: IntentRequest) -> RouteDecision:
        return cls(decision="direct", request=request)


def route_decision_or_clarify(value: object) -> RouteDecision:
    """Validate model output; shape failures become a non-executable closed decision.

    Transport/provider exceptions do not enter this function and must remain visible to the
    caller. This handles only returned values whose shape is missing, invalid, or incompatible.
    """

    try:
        return RouteDecision.model_validate(value)
    except ValidationError:
        return RouteDecision.clarify("invalid_output")


class RoutingContext(BaseModel):
    """Bounded, authority-free input projected for the semantic router."""

    model_config = _FROZEN

    utterance: NonEmptyText
    bound_customer: StrictBool
    active_flow: Literal["cart", "support", "identity"] | None = None
    pending_action: (
        Literal[
            "place_order",
            "refund_order",
            "cancel_orders",
            "return_order",
            "change_profile",
            "verify_identity",
        ]
        | None
    ) = None
    recent_effect: (
        Literal[
            "cart_changed",
            "order_placed",
            "orders_cancelled",
            "refund_created",
            "return_created",
            "profile_changed",
            "identity_bound",
        ]
        | None
    ) = None
    available_capabilities: tuple[CapabilityId, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def capabilities_are_unique(self) -> RoutingContext:
        if len(set(self.available_capabilities)) != len(self.available_capabilities):
            raise ValueError("available_capabilities must not contain duplicates")
        return self


class CapabilityOutcome(BaseModel):
    """Base for code-authored, capability-specific terminal outcome DTOs.

    Registry entries must declare a concrete subclass carrying the evidence their response needs;
    this base is not itself a registrable success result.
    """

    model_config = _FROZEN

    status: Literal["completed", "needs_input", "declined", "failed", "human_required"]


class VerificationProof(BaseModel):
    """One committed proof that may initialize a freshly rotated principal context."""

    model_config = _FROZEN

    proof_id: NonEmptyText = Field(default_factory=lambda: uuid.uuid4().hex)
    method: Literal["otp"] = "otp"
    raised_to: Literal[2] = 2


class PrincipalTransition(BaseModel):
    """Allowlisted payload carried outside the retired reasoning checkpoint."""

    model_config = _FROZEN

    transition_id: NonEmptyText = Field(default_factory=lambda: uuid.uuid4().hex)
    customer_ref: NonEmptyText
    masked_contact: NonEmptyText
    fresh_proof: VerificationProof
    continuation: IntentRequest | None = None
