"""Typed contracts shared by routing, continuation, and capability dispatch.

A request carries caller intent, never authority: slot completeness is not authorization, and
the owning flow re-resolves and re-authorizes every target against the live principal. There
are no planner DTOs: a multi-step plan is an unsupported workflow, not a gap to fill here.
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
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic.json_schema import SkipJsonSchema

from agnostic_market.dtos.confirmation import ProfileField, RefundDestination
from agnostic_market.dtos.money import PositiveUsdAmount

_FROZEN = ConfigDict(extra="forbid", frozen=True)
NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
CancelScope = Literal["all_cancellable", "both_cancellable"]
OrderContextOperation = Literal["read", "list", "place", "cancel", "refund", "return"]
AnswerTopic = Literal["policy", "general"]
ListOrderScope = Literal["session", "account"]
CartOperation = Literal["add", "remove", "set_quantity"]
OrderStatusRouteSelector = Literal["explicit", "focused", "recent"]


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
    ABORT_CURRENT = "abort_current"
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


OrderStatusSelector = Annotated[
    ExplicitOrderSet | FocusedOrderSet | RecentOrderSet,
    Field(discriminator="selector"),
]


class OrderTargetProposal(BaseModel):
    """Ephemeral model proposal; shape validation is not target resolution or authority."""

    model_config = _FROZEN

    relationship: Literal["single", "plural", "alternative", "ambiguous"]
    order_refs: tuple[NonEmptyText, ...] = ()

    @model_validator(mode="after")
    def _reference_count_matches_relationship(self) -> OrderTargetProposal:
        count = len(self.order_refs)
        if self.relationship == "single" and count != 1:
            raise ValueError("a single target proposal requires exactly one reference")
        if self.relationship in {"plural", "alternative"} and count < 2:
            raise ValueError(
                f"an {self.relationship} target proposal requires at least two references"
            )
        return self


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
    topic: AnswerTopic


class AnswerResponse(BaseModel):
    """Tool-incapable answer-owner output; no-answer decisions carry no model prose."""

    model_config = _FROZEN

    decision: Literal["answer", "clarify", "unsupported"]
    answer: NonEmptyText | None = None

    @model_validator(mode="after")
    def _answer_matches_decision(self) -> AnswerResponse:
        if (self.decision == "answer") != (self.answer is not None):
            raise ValueError("only the answer decision may carry a non-empty answer")
        return self


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
    explicit_target_turn_id: SkipJsonSchema[NonEmptyText | None] = None
    explicit_target_confirmed: SkipJsonSchema[StrictBool] = False

    @model_validator(mode="after")
    def code_owned_target_evidence_is_coherent(self) -> VerifyOrderStatus:
        if (
            self.explicit_target_turn_id is not None or self.explicit_target_confirmed
        ) and not isinstance(self.target, ExplicitOrderSet):
            raise ValueError("only an explicit order-status target can carry caller evidence")
        if self.explicit_target_confirmed and self.explicit_target_turn_id is None:
            raise ValueError("order-status confirmation requires its admitted source turn")
        return self

    def with_explicit_target_turn(self, turn_id: str) -> VerifyOrderStatus:
        if not isinstance(self.target, ExplicitOrderSet):
            raise ValueError("order-status target evidence requires an explicit target")
        normalized_turn_id = turn_id.strip()
        if (
            self.explicit_target_turn_id is not None
            and self.explicit_target_turn_id != normalized_turn_id
        ):
            raise ValueError("order-status target evidence turn cannot be replaced")
        return VerifyOrderStatus(
            target=self.target,
            explicit_target_turn_id=normalized_turn_id,
            explicit_target_confirmed=self.explicit_target_confirmed,
        )

    def with_confirmed_explicit_target(self) -> VerifyOrderStatus:
        if not isinstance(self.target, ExplicitOrderSet) or self.explicit_target_turn_id is None:
            raise ValueError("order-status confirmation requires explicit target evidence")
        return VerifyOrderStatus(
            target=self.target,
            explicit_target_turn_id=self.explicit_target_turn_id,
            explicit_target_confirmed=True,
        )

    def is_slot_complete(self) -> bool:
        return self.target is not None


class ListOrders(_CompleteIntentRequest):
    model_config = _FROZEN

    kind: Literal[CapabilityId.LIST_ORDERS] = CapabilityId.LIST_ORDERS
    scope: ListOrderScope


class ViewCart(_CompleteIntentRequest):
    model_config = _FROZEN

    kind: Literal[CapabilityId.VIEW_CART] = CapabilityId.VIEW_CART


class CartItemQuery(BaseModel):
    model_config = _FROZEN

    selector: Literal["query"] = "query"
    query: NonEmptyText


class CartItemChoices(BaseModel):
    """Code-owned ambiguous SKU set retained until the caller selects one live option."""

    model_config = _FROZEN

    selector: Literal["choices"] = "choices"
    skus: tuple[NonEmptyText, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def skus_are_unique(self) -> CartItemChoices:
        if len(set(self.skus)) != len(self.skus):
            raise ValueError("cart item choices must be unique")
        return self


class ResolvedCartItemRef(BaseModel):
    model_config = _FROZEN

    selector: Literal["resolved"] = "resolved"
    sku: NonEmptyText


CartItemSelector = Annotated[
    CartItemQuery | CartItemChoices | ResolvedCartItemRef,
    Field(discriminator="selector"),
]


class ModifyCart(IntentRequestModel):
    model_config = _FROZEN

    kind: Literal[CapabilityId.MODIFY_CART] = CapabilityId.MODIFY_CART
    operation: CartOperation
    item: CartItemSelector | None = None
    quantity: int | None = Field(default=None, strict=True, ge=0)

    @model_validator(mode="after")
    def quantity_matches_operation(self) -> ModifyCart:
        if self.operation == "remove" and self.quantity is not None:
            raise ValueError("remove must not carry a quantity")
        if self.operation == "add" and self.quantity is not None and self.quantity < 1:
            raise ValueError("add quantity must be positive")
        return self

    def is_slot_complete(self) -> bool:
        if self.item is None or isinstance(self.item, CartItemChoices):
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
    amount_usd: PositiveUsdAmount | None = None
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


class AbortCurrent(_CompleteIntentRequest):
    model_config = _FROZEN

    kind: Literal[CapabilityId.ABORT_CURRENT] = CapabilityId.ABORT_CURRENT


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
    | AbortCurrent
    | DiscloseAiIdentity
    | RequestPerson,
    Field(discriminator="kind"),
]
_INTENT_REQUEST_ADAPTER = TypeAdapter(IntentRequest)


def _revalidate_intent_request(value: object) -> IntentRequest:
    if isinstance(value, IntentRequestModel):
        value = value.model_dump(warnings=False)
    return _INTENT_REQUEST_ADAPTER.validate_python(value)


class ActiveInvocation(BaseModel):
    """One code-opened capability request retained across deterministic continuation."""

    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")

    invocation_id: NonEmptyText = Field(default_factory=lambda: uuid.uuid4().hex)
    request: IntentRequest
    opened_turn_id: NonEmptyText

    @field_validator("request", mode="before")
    @classmethod
    def request_is_fully_validated(cls, value: object) -> IntentRequest:
        # A model instance may have been built past validation (`model_construct`, an
        # unchecked copy), so re-validate the payload rather than trusting the instance.
        return _revalidate_intent_request(value)

    @property
    def capability(self) -> CapabilityId:
        return self.request.kind

    def with_request(self, request: IntentRequest) -> ActiveInvocation:
        """Validate a same-capability replacement while retaining invocation identity."""

        replacement = ActiveInvocation.model_validate(
            {
                "invocation_id": self.invocation_id,
                "request": request,
                "opened_turn_id": self.opened_turn_id,
            }
        )
        if replacement.capability != self.capability:
            raise ValueError("active invocation request cannot change capability")
        return replacement


class InvocationClarificationOwner(BaseModel):
    """Clarification budget owned by one retained capability invocation."""

    model_config = _FROZEN

    kind: Literal["invocation"] = "invocation"
    invocation_id: NonEmptyText


class RouterClarificationOwner(BaseModel):
    """Clarification budget owned by consecutive ordinary router no-action turns."""

    model_config = _FROZEN

    kind: Literal["router"] = "router"
    clarification_id: NonEmptyText = Field(default_factory=lambda: uuid.uuid4().hex)


ClarificationOwner = Annotated[
    InvocationClarificationOwner | RouterClarificationOwner,
    Field(discriminator="kind"),
]


class CapabilityDispatchEnvelope(BaseModel):
    """One admitted direct or continuation decision awaiting registry dispatch."""

    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")

    turn_id: NonEmptyText
    mode: Literal["direct", "continue"]
    request: IntentRequest | None = None
    observed_invocation_id: NonEmptyText | None = None

    @field_validator("request", mode="before")
    @classmethod
    def request_is_fully_validated(cls, value: object) -> IntentRequest | None:
        if value is None:
            return None
        return _revalidate_intent_request(value)

    @model_validator(mode="after")
    def payload_matches_mode(self) -> CapabilityDispatchEnvelope:
        if self.mode == "direct":
            if self.request is None or self.observed_invocation_id is not None:
                raise ValueError("direct dispatch requires only one request")
            RouteDecision.direct(self.request)
        elif self.request is not None or self.observed_invocation_id is None:
            raise ValueError("continuation dispatch requires only the observed invocation ID")
        return self


ClarificationReason = Literal[
    "ambiguous_intent",
    "missing_target",
    "missing_value",
    "unsupported_workflow",
    "unsupported_capability",
]


class RouteProposal(BaseModel):
    """Provider wire route; coarse ownership only, never authority or fine slots."""

    model_config = _FROZEN

    decision: Literal["clarify", "direct", "continue"]
    capability: CapabilityId | None = None
    clarification_reason: ClarificationReason | None = None
    answer_topic: AnswerTopic | None = None
    list_scope: ListOrderScope | None = None
    cart_operation: CartOperation | None = None
    profile_field: ProfileField | None = None
    order_status_selector: OrderStatusRouteSelector | None = None


class RouteDecision(BaseModel):
    """Validated internal route consumed by telemetry and capability dispatch."""

    model_config = _FROZEN

    decision: Literal["clarify", "direct", "continue"]
    request: IntentRequest | None = None
    clarification_reason: ClarificationReason | None = None

    @field_validator("request", mode="before")
    @classmethod
    def request_is_fully_validated(cls, value: object) -> IntentRequest | None:
        if value is None:
            return None
        return _revalidate_intent_request(value)

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
            if isinstance(self.request, ModifyCart) and isinstance(
                self.request.item, (CartItemChoices, ResolvedCartItemRef)
            ):
                raise ValueError("code-owned cart selectors are minted by the owning capability")
            if isinstance(self.request, VerifyOrderStatus) and (
                self.request.explicit_target_turn_id is not None
                or self.request.explicit_target_confirmed
            ):
                raise ValueError("order-status target evidence is minted by its owning flow")
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


RoutingFailureReason = Literal[
    "invalid_output",
    "routing_unavailable",
    "context_invalid",
    "decision_rejected",
]


class RoutingFailure(BaseModel):
    """Code-authored, non-executable failure to obtain a trustworthy semantic route."""

    model_config = _FROZEN

    outcome: Literal["routing_failure"] = "routing_failure"
    reason: RoutingFailureReason


RouterNoActionReason = ClarificationReason | RoutingFailureReason


class RouterNoActionEnvelope(BaseModel):
    """One admitted non-executable route awaiting code-authored speech."""

    model_config = _FROZEN

    turn_id: NonEmptyText
    owner: ClarificationOwner
    reason: RouterNoActionReason


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

    routing_scope: Literal["ordinary", "confirmation_escape"] = "ordinary"
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


class VerificationProof(BaseModel):
    """One committed proof that may initialize a freshly rotated principal context."""

    model_config = _FROZEN

    proof_id: NonEmptyText = Field(default_factory=lambda: uuid.uuid4().hex)
    method: Literal["otp"] = "otp"
    raised_to: Literal[2] = 2


PrincipalCompletionKind = Literal[
    "continue_request",
    "switch_account",
    "verify_identity",
]


def _is_principal_transition_continuation(request: IntentRequest) -> bool:
    if isinstance(request, ListOrders):
        return request.scope == "account"
    if isinstance(request, CancelOrders):
        return isinstance(request.target, CancellableOrderScope | ExplicitOrderSet)
    if isinstance(request, RefundOrder | ReturnOrder):
        return isinstance(request.target, ExplicitOrderTarget)
    return isinstance(request, ChangeProfile)


class PrincipalTransitionProjection(BaseModel):
    """One total interpretation of an allowlisted principal-transition request."""

    model_config = _FROZEN

    continuation: IntentRequest | None
    completion_kind: PrincipalCompletionKind

    @model_validator(mode="after")
    def continuation_matches_completion(self) -> PrincipalTransitionProjection:
        if self.completion_kind == "continue_request":
            if self.continuation is None or not _is_principal_transition_continuation(
                self.continuation
            ):
                raise ValueError("continue_request requires an allowlisted continuation")
        elif self.continuation is not None:
            raise ValueError("only continue_request may carry a continuation")
        return self


def project_principal_transition(request: IntentRequest) -> PrincipalTransitionProjection:
    """Project one allowlisted request into its post-rotation continuation and outcome."""

    if request is None:
        raise ValueError("principal transition requires an initiating request")
    if isinstance(request, SwitchAccount):
        return PrincipalTransitionProjection(
            continuation=None,
            completion_kind="switch_account",
        )
    if isinstance(request, VerifyIdentity):
        return PrincipalTransitionProjection(
            continuation=None,
            completion_kind="verify_identity",
        )
    if _is_principal_transition_continuation(request):
        return PrincipalTransitionProjection(
            continuation=request,
            completion_kind="continue_request",
        )
    raise ValueError(f"{request.kind.value} cannot continue across a principal transition")


class PrincipalTransition(BaseModel):
    """Allowlisted payload carried outside the retired reasoning checkpoint."""

    model_config = _FROZEN

    transition_id: NonEmptyText = Field(default_factory=lambda: uuid.uuid4().hex)
    customer_ref: NonEmptyText
    masked_contact: NonEmptyText
    fresh_proof: VerificationProof
    initiating_request: IntentRequest

    @property
    def projection(self) -> PrincipalTransitionProjection:
        return project_principal_transition(self.initiating_request)

    @property
    def continuation(self) -> IntentRequest | None:
        return self.projection.continuation

    @property
    def completion_kind(self) -> PrincipalCompletionKind:
        return self.projection.completion_kind

    @model_validator(mode="after")
    def initiating_request_is_allowlisted(self) -> PrincipalTransition:
        project_principal_transition(self.initiating_request)
        return self


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
