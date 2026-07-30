"""Milestone-1 routing DTO, capability registry, and fail-closed contract tests."""

from __future__ import annotations

from typing import Literal

import pytest
from pydantic import BaseModel, ValidationError

from agnostic_market.agents.capabilities import (
    CapabilityRegistry,
    CapabilityRegistryError,
    CapabilitySpec,
)
from agnostic_market.dtos.orchestration import (
    CancellableOrderScope,
    CancelOrders,
    CapabilityId,
    CapabilityOutcome,
    ChangeProfile,
    DiscloseAiIdentity,
    ExplicitOrderTarget,
    IntentRequestModel,
    ModifyCart,
    RefundOrder,
    ReturnOrder,
    RouteDecision,
    RoutingContext,
    RoutingFailure,
    SearchCatalog,
    VerifyOrderStatus,
    ViewCart,
    ViewIdentityStatus,
    validate_route_output,
)


class _CancelCompleted(CapabilityOutcome):
    status: Literal["completed"] = "completed"
    order_refs: tuple[str, ...]


class _WrongOutcome(CapabilityOutcome):
    status: Literal["failed"] = "failed"


class _RogueRequest(BaseModel):
    kind: CapabilityId = CapabilityId.SEARCH_CATALOG

    def is_slot_complete(self) -> bool:
        return True


def _cancel_request() -> CancelOrders:
    return CancelOrders(target=CancellableOrderScope(scope="all_cancellable"))


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


async def test_registry_execution_returns_only_the_declared_outcome() -> None:
    seen: list[BaseModel] = []

    async def adapter(request: BaseModel) -> BaseModel:
        seen.append(request)
        return _CancelCompleted(order_refs=("ORD-TEST-1",))

    registry = CapabilityRegistry(
        [
            CapabilitySpec(
                capability_id=CapabilityId.CANCEL_ORDERS,
                request_type=CancelOrders,
                outcome_type=_CancelCompleted,
                adapter=adapter,
                effect="write",
                write_serialization_key="orders",
                planner_ready=True,
            )
        ]
    )

    request = _cancel_request()
    outcome = await registry.execute(request)

    assert seen == [request]
    assert outcome == _CancelCompleted(order_refs=("ORD-TEST-1",))
    assert registry.capability_ids == (CapabilityId.CANCEL_ORDERS,)
    assert registry.planner_ready_ids == (CapabilityId.CANCEL_ORDERS,)


async def test_registry_rejects_an_adapter_outcome_mismatch() -> None:
    async def adapter(_: BaseModel) -> BaseModel:
        return _WrongOutcome()

    registry = CapabilityRegistry(
        [
            CapabilitySpec(
                capability_id=CapabilityId.CANCEL_ORDERS,
                request_type=CancelOrders,
                outcome_type=_CancelCompleted,
                adapter=adapter,
                effect="write",
                write_serialization_key="orders",
            )
        ]
    )

    with pytest.raises(CapabilityRegistryError, match="incompatible outcome"):
        await registry.execute(_cancel_request())


async def test_registry_rejects_incomplete_requests_before_calling_the_adapter() -> None:
    seen: list[BaseModel] = []

    async def adapter(request: BaseModel) -> BaseModel:
        seen.append(request)
        return _CancelCompleted(order_refs=("ORD-TEST-1",))

    registry = CapabilityRegistry(
        [
            CapabilitySpec(
                capability_id=CapabilityId.CANCEL_ORDERS,
                request_type=CancelOrders,
                outcome_type=_CancelCompleted,
                adapter=adapter,
                effect="write",
                write_serialization_key="orders",
            )
        ]
    )

    with pytest.raises(CapabilityRegistryError, match="incomplete request"):
        await registry.execute(CancelOrders())
    assert seen == []


async def test_registry_rejects_an_untyped_request_without_dereferencing_kind() -> None:
    with pytest.raises(CapabilityRegistryError, match="typed IntentRequest"):
        await CapabilityRegistry().execute(IntentRequestModel())


def test_registry_rejects_duplicate_or_inconsistent_contracts() -> None:
    async def adapter(_: BaseModel) -> BaseModel:
        return _CancelCompleted(order_refs=("ORD-TEST-1",))

    spec = CapabilitySpec(
        capability_id=CapabilityId.CANCEL_ORDERS,
        request_type=CancelOrders,
        outcome_type=_CancelCompleted,
        adapter=adapter,
        effect="write",
        write_serialization_key="orders",
    )
    registry = CapabilityRegistry([spec])

    with pytest.raises(CapabilityRegistryError, match="already registered"):
        registry.register(spec)
    with pytest.raises(CapabilityRegistryError, match="request_type kind"):
        CapabilitySpec(
            capability_id=CapabilityId.VIEW_CART,
            request_type=CancelOrders,
            outcome_type=_CancelCompleted,
            adapter=adapter,
            effect="read",
        )
    with pytest.raises(CapabilityRegistryError, match="concrete"):
        CapabilitySpec(
            capability_id=CapabilityId.VIEW_CART,
            request_type=ViewCart,
            outcome_type=CapabilityOutcome,
            adapter=adapter,
            effect="read",
        )


def test_registry_rejects_a_base_model_lookalike_that_has_a_valid_kind() -> None:
    async def adapter(_: BaseModel) -> BaseModel:
        return _CancelCompleted(order_refs=("ORD-TEST-1",))

    with pytest.raises(CapabilityRegistryError, match="IntentRequestModel subclass"):
        CapabilitySpec(
            capability_id=CapabilityId.SEARCH_CATALOG,
            request_type=_RogueRequest,  # type: ignore[arg-type]
            outcome_type=_CancelCompleted,
            adapter=adapter,
            effect="read",
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"effect": "write", "write_serialization_key": None},
        {"effect": "write", "write_serialization_key": "  "},
        {"effect": "write", "write_serialization_key": " orders "},
        {"effect": "read", "write_serialization_key": "orders"},
    ],
)
def test_registry_enforces_write_serialization_contract(kwargs: dict[str, object]) -> None:
    async def adapter(_: BaseModel) -> BaseModel:
        return _CancelCompleted(order_refs=("ORD-TEST-1",))

    with pytest.raises(CapabilityRegistryError):
        CapabilitySpec(
            capability_id=CapabilityId.CANCEL_ORDERS,
            request_type=CancelOrders,
            outcome_type=_CancelCompleted,
            adapter=adapter,
            **kwargs,  # type: ignore[arg-type]
        )
