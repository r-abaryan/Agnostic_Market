"""Routing/continuation DTO contracts: request shapes, invocation revalidation, registry
resolution, and the principal-transition projection. Pure contracts: no graph, no network."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from pydantic import BaseModel, ValidationError

from agnostic_market.agents.capabilities import (
    CapabilityEntry,
    CapabilityRegistry,
    CapabilityRegistryError,
    CapabilitySpec,
)
from agnostic_market.dtos.orchestration import (
    ActiveInvocation,
    CancellableOrderScope,
    CancelOrders,
    CapabilityId,
    ChangeProfile,
    DiscloseAiIdentity,
    ExplicitOrderSet,
    ExplicitOrderTarget,
    FocusedOrderTarget,
    IntentRequestModel,
    ListOrders,
    ModifyCart,
    PrincipalTransition,
    PrincipalTransitionProjection,
    RefundOrder,
    ReturnOrder,
    RouteDecision,
    RoutingContext,
    RoutingFailure,
    SearchCatalog,
    SwitchAccount,
    VerificationProof,
    VerifyIdentity,
    VerifyOrderStatus,
    ViewCart,
    ViewIdentityStatus,
    project_principal_transition,
    validate_route_output,
)
from agnostic_market.dtos.state import ReasoningState, open_active_invocation


class _RogueRequest(BaseModel):
    kind: CapabilityId = CapabilityId.SEARCH_CATALOG

    def is_slot_complete(self) -> bool:
        return True


def _cancel_request() -> CancelOrders:
    return CancelOrders(target=CancellableOrderScope(scope="all_cancellable"))


def test_active_invocation_is_minimal_derived_and_freshly_identified() -> None:
    first = ActiveInvocation(request=ViewCart(), opened_turn_id="turn-1")
    second = ActiveInvocation(request=ViewCart(), opened_turn_id="turn-1")

    assert first.capability == CapabilityId.VIEW_CART
    assert "capability" not in first.model_dump()
    assert first.invocation_id != second.invocation_id
    with pytest.raises(ValidationError):
        ActiveInvocation(request=ViewCart(), opened_turn_id=" ")


def test_active_invocation_replaces_only_the_same_capability_request() -> None:
    invocation = ActiveInvocation(request=CancelOrders(), opened_turn_id="turn-1")
    completed = invocation.with_request(_cancel_request())

    assert completed.request == _cancel_request()
    assert completed.invocation_id == invocation.invocation_id
    assert completed.opened_turn_id == invocation.opened_turn_id
    with pytest.raises(ValueError, match="cannot change capability"):
        invocation.with_request(SwitchAccount())
    with pytest.raises(ValidationError):
        invocation.with_request(object())  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        ActiveInvocation.model_validate(
            {
                "request": {"kind": "view_cart"},
                "opened_turn_id": "turn-1",
                "unexpected": True,
            }
        )


@pytest.mark.parametrize(
    "corrupt_request",
    [
        RefundOrder(
            target=ExplicitOrderTarget(order_ref="ORD-1001"),
            amount_usd=10.0,
            destination="original",
        ).model_copy(update={"amount_usd": -5.0}),
        RefundOrder(
            target=ExplicitOrderTarget(order_ref="ORD-1001"),
            amount_usd=10.0,
            destination="original",
        ).model_copy(update={"target": "not-a-target"}),
        ChangeProfile(field="address", new_value="New address").model_copy(
            update={"new_value": ""}
        ),
    ],
)
def test_active_invocation_revalidates_nested_request_instances(corrupt_request: object) -> None:
    with pytest.raises(ValidationError):
        ActiveInvocation(request=corrupt_request, opened_turn_id="turn-1")


def test_reasoning_state_revalidates_existing_active_invocation_instance() -> None:
    valid = ActiveInvocation(request=_cancel_request(), opened_turn_id="turn-1")
    corrupt = valid.model_copy(
        update={
            "request": {
                "kind": "refund_order",
                "target": "not-a-target",
                "amount_usd": -5.0,
                "destination": None,
            }
        }
    )

    with pytest.raises(ValidationError):
        ReasoningState.model_validate(
            {
                "consumed_turn_ids": ("turn-1",),
                "active_invocation": corrupt,
            }
        )


def test_invocation_opening_uses_only_the_admitted_ledger_tail() -> None:
    with pytest.raises(ValueError, match="admitted turn"):
        open_active_invocation(ViewCart(), consumed_turn_ids=())

    invocation = open_active_invocation(
        ViewCart(),
        consumed_turn_ids=("turn-1", "turn-2"),
    )
    assert invocation.opened_turn_id == "turn-2"
    assert (
        ReasoningState(
            consumed_turn_ids=("turn-1", "turn-2"),
            active_invocation=invocation,
        ).active_invocation
        == invocation
    )

    retained = ActiveInvocation(request=ViewCart(), opened_turn_id="turn-1")
    assert (
        ReasoningState(
            consumed_turn_ids=("turn-1", "turn-2"),
            active_invocation=retained,
        ).active_invocation
        == retained
    )
    with pytest.raises(ValidationError, match="opening turn was not admitted"):
        ReasoningState(
            consumed_turn_ids=("turn-1",),
            active_invocation=invocation,
        )


def test_principal_transition_projection_is_closed_and_context_free() -> None:
    allowed = (
        ListOrders(scope="account"),
        CancelOrders(target=CancellableOrderScope(scope="all_cancellable")),
        CancelOrders(target=ExplicitOrderSet(order_refs=("ORD-1002",))),
        RefundOrder(target=ExplicitOrderTarget(order_ref="ORD-1002")),
        ReturnOrder(target=ExplicitOrderTarget(order_ref="ORD-1002")),
        ChangeProfile(field="address"),
    )
    with pytest.raises(ValueError, match="initiating request"):
        project_principal_transition(None)  # type: ignore[arg-type]
    switch = PrincipalTransition(
        customer_ref="CUST-001",
        masked_contact="number ending 0119",
        fresh_proof=VerificationProof(),
        initiating_request=SwitchAccount(),
    )
    assert switch.continuation is None
    assert switch.completion_kind == "switch_account"
    verification = PrincipalTransition(
        customer_ref="CUST-001",
        masked_contact="number ending 0119",
        fresh_proof=VerificationProof(),
        initiating_request=VerifyIdentity(),
    )
    assert verification.continuation is None
    assert verification.completion_kind == "verify_identity"
    for request in allowed:
        projection = project_principal_transition(request)
        assert projection.continuation == request
        assert projection.completion_kind == "continue_request"
        transition = PrincipalTransition(
            customer_ref="CUST-001",
            masked_contact="number ending 0119",
            fresh_proof=VerificationProof(),
            initiating_request=request,
        )
        assert transition.continuation == request
        assert transition.initiating_request == request
        assert transition.completion_kind == "continue_request"

    rejected = (
        ListOrders(scope="session"),
        CancelOrders(),
        CancelOrders(target={"selector": "focused"}),
        RefundOrder(target=FocusedOrderTarget()),
        ReturnOrder(target=FocusedOrderTarget()),
        ViewCart(),
    )
    for request in rejected:
        with pytest.raises(ValueError, match="cannot continue"):
            project_principal_transition(request)
        with pytest.raises(ValidationError):
            PrincipalTransition(
                customer_ref="CUST-001",
                masked_contact="number ending 0119",
                fresh_proof=VerificationProof(),
                initiating_request=request,
            )

    with pytest.raises(ValidationError, match="continue_request"):
        PrincipalTransitionProjection(
            continuation=None,
            completion_kind="continue_request",
        )
    with pytest.raises(ValidationError, match="only continue_request"):
        PrincipalTransitionProjection(
            continuation=SwitchAccount(),
            completion_kind="switch_account",
        )
    for forbidden in (SwitchAccount(), VerifyIdentity(), ListOrders(scope="session")):
        with pytest.raises(ValidationError, match="allowlisted continuation"):
            PrincipalTransitionProjection(
                continuation=forbidden,
                completion_kind="continue_request",
            )


def test_every_capability_id_has_one_valid_intent_shape() -> None:
    payloads: dict[CapabilityId, dict[str, object]] = {
        CapabilityId.ANSWER_QUESTION: {"kind": "answer_question", "topic": "policy"},
        CapabilityId.SEARCH_CATALOG: {"kind": "search_catalog", "query": "running shoes"},
        CapabilityId.VERIFY_ORDER_STATUS: {
            "kind": "verify_order_status",
            "target": {"selector": "focused"},
        },
        CapabilityId.LIST_ORDERS: {"kind": "list_orders", "scope": "session"},
        CapabilityId.VIEW_CART: {"kind": "view_cart"},
        CapabilityId.MODIFY_CART: {
            "kind": "modify_cart",
            "operation": "remove",
            "item_query": "running shoes",
        },
        CapabilityId.PLACE_ORDER: {"kind": "place_order"},
        CapabilityId.CANCEL_ORDERS: {
            "kind": "cancel_orders",
            "target": {"selector": "focused"},
        },
        CapabilityId.REFUND_ORDER: {
            "kind": "refund_order",
            "target": {"selector": "focused"},
        },
        CapabilityId.RETURN_ORDER: {
            "kind": "return_order",
            "target": {"selector": "focused"},
        },
        CapabilityId.CHANGE_PROFILE: {"kind": "change_profile", "field": "address"},
        CapabilityId.VERIFY_IDENTITY: {"kind": "verify_identity"},
        CapabilityId.SWITCH_ACCOUNT: {"kind": "switch_account"},
        CapabilityId.VIEW_IDENTITY_STATUS: {"kind": "view_identity_status"},
        CapabilityId.DISCLOSE_AI_IDENTITY: {"kind": "disclose_ai_identity"},
        CapabilityId.REQUEST_PERSON: {"kind": "request_person"},
    }

    assert set(payloads) == set(CapabilityId)
    for payload in payloads.values():
        decision = RouteDecision.model_validate({"decision": "direct", "request": payload})
        assert decision.request is not None
        assert decision.request.kind == CapabilityId(payload["kind"])


def test_direct_route_has_exactly_one_typed_request() -> None:
    decision = RouteDecision.direct(_cancel_request())

    assert decision.decision == "direct"
    assert isinstance(decision.request, CancelOrders)
    assert decision.clarification_reason is None


def test_continue_route_carries_no_model_selected_request_or_invocation() -> None:
    decision = RouteDecision.continue_current()

    assert decision.decision == "continue"
    assert decision.request is None
    assert decision.clarification_reason is None
    for payload in (
        {"decision": "continue", "request": {"kind": "view_cart"}},
        {"decision": "continue", "clarification_reason": "ambiguous_intent"},
    ):
        with pytest.raises(ValidationError):
            RouteDecision.model_validate(payload)


@pytest.mark.parametrize(
    "value",
    [
        None,
        "not a route",
        {},
        {"decision": "direct", "request": None},
        {"decision": "plan", "request": {"kind": "view_cart"}},
        {
            "decision": "clarify",
            "clarification_reason": "missing_target",
            "request": {"kind": "view_cart"},
        },
        {"decision": "clarify", "clarification_reason": "invalid_output"},
    ],
)
def test_invalid_missing_and_incompatible_routes_fail_to_non_executable_result(
    value: object,
) -> None:
    result = validate_route_output(value)

    assert result == RoutingFailure(reason="invalid_output")


def test_valid_clarification_carries_no_executable_request() -> None:
    decision = validate_route_output(
        {"decision": "clarify", "clarification_reason": "unsupported_workflow"}
    )

    assert isinstance(decision, RouteDecision)
    assert decision.decision == "clarify"
    assert decision.request is None


def test_routing_unavailable_is_code_authored_and_not_a_model_clarification() -> None:
    failure = RoutingFailure(reason="routing_unavailable")

    assert failure.outcome == "routing_failure"
    with pytest.raises(ValidationError):
        RouteDecision.model_validate(
            {"decision": "clarify", "clarification_reason": "routing_unavailable"}
        )


def test_cancel_selector_cannot_widen_from_missing_or_empty_targets() -> None:
    incomplete = CancelOrders()
    assert incomplete.target is None
    assert not incomplete.is_slot_complete()

    with pytest.raises(ValidationError):
        CancelOrders.model_validate(
            {"kind": "cancel_orders", "target": {"selector": "explicit_set", "order_refs": []}}
        )
    with pytest.raises(ValidationError):
        CancelOrders.model_validate(
            {
                "kind": "cancel_orders",
                "target": {"selector": "cancellable_scope", "scope": "everything"},
            }
        )


@pytest.mark.parametrize(
    ("payload", "valid_shape", "slot_complete"),
    [
        ({"operation": "add", "item_query": "shampoo", "quantity": 1}, True, True),
        ({"operation": "add", "item_query": "shampoo"}, True, False),
        ({"operation": "remove", "item_query": "shampoo"}, True, True),
        ({"operation": "remove", "item_query": "shampoo", "quantity": 1}, False, False),
        ({"operation": "set_quantity", "item_query": "shampoo", "quantity": 0}, True, True),
        ({"operation": "set_quantity", "item_query": "shampoo"}, True, False),
        ({"item_query": "shampoo"}, False, False),
        ({"item_query": "shampoo", "quantity": 1}, False, False),
    ],
)
def test_cart_mutation_requires_operation_specific_fields(
    payload: dict[str, object],
    valid_shape: bool,
    slot_complete: bool,
) -> None:
    if valid_shape:
        assert ModifyCart.model_validate(payload).is_slot_complete() is slot_complete
    else:
        with pytest.raises(ValidationError):
            ModifyCart.model_validate(payload)


@pytest.mark.parametrize(
    ("intent_request", "complete"),
    [
        (SearchCatalog(), False),
        (SearchCatalog(query="trail shoes"), True),
        (VerifyOrderStatus(), False),
        (VerifyOrderStatus(target={"selector": "focused"}), True),
        (RefundOrder(), False),
        (
            RefundOrder(
                target=ExplicitOrderTarget(order_ref="ORD-1002"),
                amount_usd=10,
                destination="original",
            ),
            True,
        ),
        (ReturnOrder(), False),
        (ReturnOrder(target=ExplicitOrderTarget(order_ref="ORD-1002")), True),
        (ChangeProfile(field="address"), False),
        (ChangeProfile(field="address", new_value="10 High Street"), True),
        (ViewIdentityStatus(), True),
        (DiscloseAiIdentity(), True),
    ],
)
def test_request_slot_completeness_is_explicit(
    intent_request: IntentRequestModel,
    complete: bool,
) -> None:
    assert intent_request.is_slot_complete() is complete


def test_profile_value_cannot_be_copied_into_a_direct_router_decision() -> None:
    with pytest.raises(ValidationError, match="owning capability"):
        RouteDecision.direct(ChangeProfile(field="address", new_value="10 High Street"))

    decision = RouteDecision.direct(ChangeProfile(field="address"))
    assert isinstance(decision.request, ChangeProfile)
    assert decision.request.new_value is None


@pytest.mark.parametrize(
    "route_request",
    (
        {"kind": "list_orders"},
        {"kind": "modify_cart", "item_query": "shampoo"},
        {"kind": "change_profile", "new_value": "10 High Street"},
    ),
)
def test_scope_and_operation_fields_remain_required(route_request: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        RouteDecision.model_validate({"decision": "direct", "request": route_request})


def test_routing_context_is_bounded_and_authority_free() -> None:
    context = RoutingContext(
        utterance="cancel all my orders",
        bound_customer=True,
        active_capability=CapabilityId.CANCEL_ORDERS,
        recent_order_operation="place",
        recent_order_count=2,
        cart_state="nonempty",
        available_capabilities=(CapabilityId.CANCEL_ORDERS, CapabilityId.VERIFY_ORDER_STATUS),
    )

    assert set(context.model_dump()) == {
        "utterance",
        "bound_customer",
        "active_capability",
        "recent_order_operation",
        "recent_order_count",
        "cart_state",
        "available_capabilities",
    }
    with pytest.raises(ValidationError):
        RoutingContext(
            utterance="cancel all my orders",
            bound_customer=True,
            cart_state="empty",
            available_capabilities=(CapabilityId.CANCEL_ORDERS, CapabilityId.CANCEL_ORDERS),
        )
    with pytest.raises(ValidationError):
        RoutingContext(
            utterance="cancel all my orders",
            bound_customer=True,
            recent_order_count=-1,
            cart_state="empty",
            available_capabilities=(CapabilityId.CANCEL_ORDERS,),
        )
    for inconsistent in (
        {"recent_order_operation": "list", "recent_order_count": 0},
        {"recent_order_count": 1},
    ):
        with pytest.raises(ValidationError, match="present together"):
            RoutingContext(
                utterance="cancel all my orders",
                bound_customer=True,
                cart_state="empty",
                available_capabilities=(CapabilityId.CANCEL_ORDERS,),
                **inconsistent,
            )
    with pytest.raises(ValidationError, match="active_capability must be available"):
        RoutingContext(
            utterance="change my address",
            bound_customer=True,
            active_capability=CapabilityId.CHANGE_PROFILE,
            cart_state="empty",
            available_capabilities=(CapabilityId.SEARCH_CATALOG, CapabilityId.VIEW_CART),
        )


def test_registry_resolves_complete_and_incomplete_requests_to_an_opaque_entry() -> None:
    entry = CapabilityEntry("support_capability_entry")
    registry = CapabilityRegistry(
        [
            CapabilitySpec(
                capability_id=CapabilityId.CANCEL_ORDERS,
                request_type=CancelOrders,
                entry=entry,
            )
        ]
    )

    assert registry.resolve(_cancel_request()) is entry
    assert registry.resolve(CancelOrders()) is entry
    assert registry.capability_ids == (CapabilityId.CANCEL_ORDERS,)


def test_registry_is_immutable_and_exposes_no_executor_or_planner_contract() -> None:
    entry = CapabilityEntry("support_capability_entry")
    spec = CapabilitySpec(
        capability_id=CapabilityId.CANCEL_ORDERS,
        request_type=CancelOrders,
        entry=entry,
    )
    registry = CapabilityRegistry([spec])

    with pytest.raises(TypeError):
        registry.specs[CapabilityId.CANCEL_ORDERS] = spec  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        registry._capability_ids = ()
    with pytest.raises(FrozenInstanceError):
        registry._entry_nodes = ()
    with pytest.raises(FrozenInstanceError):
        registry._specs = {}  # type: ignore[assignment]
    assert not hasattr(registry, "register")
    assert not hasattr(registry, "execute")
    assert not hasattr(spec, "adapter")
    assert not hasattr(spec, "outcome_type")
    assert not hasattr(spec, "planner_ready")


def test_registry_rejects_duplicate_or_inconsistent_contracts() -> None:
    entry = CapabilityEntry("support_capability_entry")
    spec = CapabilitySpec(CapabilityId.CANCEL_ORDERS, CancelOrders, entry)

    with pytest.raises(CapabilityRegistryError, match="already registered"):
        CapabilityRegistry((spec, spec))
    with pytest.raises(CapabilityRegistryError, match="request_type kind"):
        CapabilitySpec(
            capability_id=CapabilityId.VIEW_CART,
            request_type=CancelOrders,
            entry=entry,
        )
    with pytest.raises(CapabilityRegistryError, match="capability_id must be"):
        CapabilitySpec(
            capability_id="cancel_orders",  # type: ignore[arg-type]
            request_type=CancelOrders,
            entry=entry,
        )
    with pytest.raises(CapabilityRegistryError, match="normalized node name"):
        CapabilityEntry(" support_capability_entry ")
    with pytest.raises(CapabilityRegistryError, match="CapabilityEntry"):
        CapabilitySpec(
            CapabilityId.CANCEL_ORDERS,
            CancelOrders,
            "support_capability_entry",  # type: ignore[arg-type]
        )
    with pytest.raises(CapabilityRegistryError, match="CapabilitySpec"):
        CapabilityRegistry((object(),))  # type: ignore[arg-type]


def test_registry_rejects_a_base_model_lookalike_that_has_a_valid_kind() -> None:
    with pytest.raises(CapabilityRegistryError, match="IntentRequestModel subclass"):
        CapabilitySpec(
            capability_id=CapabilityId.SEARCH_CATALOG,
            request_type=_RogueRequest,  # type: ignore[arg-type]
            entry=CapabilityEntry("catalog_entry"),
        )


def test_registry_rejects_untyped_unregistered_and_incompatible_requests() -> None:
    registry = CapabilityRegistry(
        (
            CapabilitySpec(
                CapabilityId.CANCEL_ORDERS,
                CancelOrders,
                CapabilityEntry("support_capability_entry"),
            ),
        )
    )

    with pytest.raises(CapabilityRegistryError, match="typed IntentRequest"):
        registry.resolve(_RogueRequest())  # type: ignore[arg-type]
    with pytest.raises(CapabilityRegistryError, match="not registered"):
        registry.resolve(ViewCart())
    incompatible = ViewCart().model_copy(update={"kind": CapabilityId.CANCEL_ORDERS})
    with pytest.raises(CapabilityRegistryError, match="incompatible request type"):
        registry.resolve(incompatible)
