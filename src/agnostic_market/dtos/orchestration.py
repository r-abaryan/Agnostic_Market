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
OrderContextOperation = Literal["read", "list", "place", "cancel", "refund", "return"]


class IntentRequestModel(BaseModel):
    """Typed intent request; slot completeness is not authorization or effect authority."""

    model_config = _FROZEN

    def is_slot_complete(self) -> bool:
        """Fail closed unless the concrete request declares its completeness rule."""

        return False


class _CompleteIntentRequest(IntentRequestModel):
    """Request whose required Pydantic fields fully describe its non-authority slots."""

    def is_slot_complete(self) -> bool:
        return True


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
    VIEW_IDENTITY_STATUS = "view_identity_status"
    DISCLOSE_AI_IDENTITY = "disclose_ai_identity"
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


class AnswerQuestion(_CompleteIntentRequest):
    model_config = _FROZEN

    kind: Literal[CapabilityId.ANSWER_QUESTION] = CapabilityId.ANSWER_QUESTION
    topic: Literal["policy", "general"]


class SearchCatalog(IntentRequestModel):
    model_config = _FROZEN

    kind: Literal[CapabilityId.SEARCH_CATALOG] = CapabilityId.SEARCH_CATALOG
    query: NonEmptyText | None = None

    def is_slot_complete(self) -> bool:
        return self.query is not None


class VerifyOrderStatus(IntentRequestModel):
    model_config = _FROZEN

    kind: Literal[CapabilityId.VERIFY_ORDER_STATUS] = CapabilityId.VERIFY_ORDER_STATUS
    target: OrderStatusSelector | None = None

    def is_slot_complete(self) -> bool:
        return self.target is not None


class ListOrders(_CompleteIntentRequest):
    model_config = _FROZEN

    kind: Literal[CapabilityId.LIST_ORDERS] = CapabilityId.LIST_ORDERS
    scope: Literal["session", "account"]


class ViewCart(_CompleteIntentRequest):
    model_config = _FROZEN

    kind: Literal[CapabilityId.VIEW_CART] = CapabilityId.VIEW_CART


class ModifyCart(IntentRequestModel):
    model_config = _FROZEN

    kind: Literal[CapabilityId.MODIFY_CART] = CapabilityId.MODIFY_CART
    operation: Literal["add", "remove", "set_quantity"]
    item_query: NonEmptyText | None = None
    quantity: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def quantity_matches_operation(self) -> ModifyCart:
        if self.operation == "remove" and self.quantity is not None:
            raise ValueError("remove must not carry a quantity")
        if self.operation == "add" and self.quantity is not None and self.quantity < 1:
            raise ValueError("add quantity must be positive")
        return self

    def is_slot_complete(self) -> bool:
        if self.item_query is None:
            return False
        return self.operation == "remove" or self.quantity is not None


class PlaceOrder(_CompleteIntentRequest):
    model_config = _FROZEN

    kind: Literal[CapabilityId.PLACE_ORDER] = CapabilityId.PLACE_ORDER


class CancelOrders(IntentRequestModel):
    model_config = _FROZEN

    kind: Literal[CapabilityId.CANCEL_ORDERS] = CapabilityId.CANCEL_ORDERS
    target: CancelSelector | None = None

    def is_slot_complete(self) -> bool:
        return self.target is not None


class RefundOrder(IntentRequestModel):
    model_config = _FROZEN

    kind: Literal[CapabilityId.REFUND_ORDER] = CapabilityId.REFUND_ORDER
    target: OrderTarget | None = None
    amount_usd: float | None = Field(default=None, gt=0)
    destination: RefundDestination | None = None

    def is_slot_complete(self) -> bool:
        return (
            self.target is not None and self.amount_usd is not None and self.destination is not None
        )


class ReturnOrder(IntentRequestModel):
    model_config = _FROZEN

    kind: Literal[CapabilityId.RETURN_ORDER] = CapabilityId.RETURN_ORDER
    target: OrderTarget | None = None

    def is_slot_complete(self) -> bool:
        return self.target is not None


class ChangeProfile(IntentRequestModel):
    model_config = _FROZEN

    kind: Literal[CapabilityId.CHANGE_PROFILE] = CapabilityId.CHANGE_PROFILE
    field: ProfileField
    new_value: NonEmptyText | None = None

    def is_slot_complete(self) -> bool:
        return self.new_value is not None


class VerifyIdentity(_CompleteIntentRequest):
    model_config = _FROZEN

    kind: Literal[CapabilityId.VERIFY_IDENTITY] = CapabilityId.VERIFY_IDENTITY


class SwitchAccount(_CompleteIntentRequest):
    model_config = _FROZEN

    kind: Literal[CapabilityId.SWITCH_ACCOUNT] = CapabilityId.SWITCH_ACCOUNT


class ViewIdentityStatus(_CompleteIntentRequest):
    model_config = _FROZEN

    kind: Literal[CapabilityId.VIEW_IDENTITY_STATUS] = CapabilityId.VIEW_IDENTITY_STATUS


class DiscloseAiIdentity(_CompleteIntentRequest):
    model_config = _FROZEN

    kind: Literal[CapabilityId.DISCLOSE_AI_IDENTITY] = CapabilityId.DISCLOSE_AI_IDENTITY


class RequestPerson(_CompleteIntentRequest):
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
    | ViewIdentityStatus
    | DiscloseAiIdentity
    | RequestPerson,
    Field(discriminator="kind"),
]

ClarificationReason = Literal[
    "ambiguous_intent",
    "missing_target",
    "missing_value",
    "unsupported_workflow",
    "unsupported_capability",
]


class RouteDecision(BaseModel):
    """Model-authored semantic route: clarify, one direct request, or current-owner continuation."""

    model_config = _FROZEN

    decision: Literal["clarify", "direct", "continue"]
    request: IntentRequest | None = None
    clarification_reason: ClarificationReason | None = None

    @model_validator(mode="after")
    def payload_matches_decision(self) -> RouteDecision:
        if self.decision == "clarify":
            if self.request is not None or self.clarification_reason is None:
                raise ValueError("clarify requires only a clarification_reason")
        elif self.decision == "direct":
            if self.request is None or self.clarification_reason is not None:
                raise ValueError("direct requires only one request")
            if isinstance(self.request, ChangeProfile) and self.request.new_value is not None:
                raise ValueError("profile values are gathered by the owning capability")
        elif self.request is not None or self.clarification_reason is not None:
            raise ValueError("continue carries no payload")
        return self

    @classmethod
    def clarify(cls, reason: ClarificationReason) -> RouteDecision:
        return cls(decision="clarify", clarification_reason=reason)

    @classmethod
    def direct(cls, request: IntentRequest) -> RouteDecision:
        return cls(decision="direct", request=request)

    @classmethod
    def continue_current(cls) -> RouteDecision:
        return cls(decision="continue")


RoutingFailureReason = Literal["invalid_output", "routing_unavailable"]


class RoutingFailure(BaseModel):
    """Code-authored, non-executable failure to obtain a trustworthy semantic route."""

    model_config = _FROZEN

    outcome: Literal["routing_failure"] = "routing_failure"
    reason: RoutingFailureReason


RouteResolution = RouteDecision | RoutingFailure


def validate_route_output(value: object) -> RouteResolution:
    """Validate returned model output; provider exceptions are classified by the caller."""

    try:
        return RouteDecision.model_validate(value)
    except ValidationError:
        return RoutingFailure(reason="invalid_output")


class RoutingContext(BaseModel):
    """Bounded, authority-free input projected for the semantic router."""

    model_config = _FROZEN

    utterance: NonEmptyText
    bound_customer: StrictBool
    active_capability: CapabilityId | None = None
    recent_order_operation: OrderContextOperation | None = None
    recent_order_count: int = Field(default=0, ge=0)
    cart_state: Literal["empty", "nonempty"]
    available_capabilities: tuple[CapabilityId, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def context_is_coherent(self) -> RoutingContext:
        if len(set(self.available_capabilities)) != len(self.available_capabilities):
            raise ValueError("available_capabilities must not contain duplicates")
        if (
            self.active_capability is not None
            and self.active_capability not in self.available_capabilities
        ):
            raise ValueError("active_capability must be available")
        if (self.recent_order_operation is None) != (self.recent_order_count == 0):
            raise ValueError("recent order operation and count must be present together")
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


class PrincipalTransitionInspection(BaseModel):
    """Live, non-checkpointed verdict over a published principal transition."""

    model_config = _FROZEN

    outcome: Literal["none", "coherent", "inconsistent"]
    transition: PrincipalTransition | None = None

    @model_validator(mode="after")
    def _transition_matches_outcome(self) -> PrincipalTransitionInspection:
        if (self.outcome == "none") != (self.transition is None):
            raise ValueError("only the none outcome may omit a transition")
        return self
