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
    ModifyCart,
    RouteDecision,
    RoutingContext,
    ViewCart,
    route_decision_or_clarify,
)


class _CancelCompleted(CapabilityOutcome):
    status: Literal["completed"] = "completed"
    order_refs: tuple[str, ...]


class _WrongOutcome(CapabilityOutcome):
    status: Literal["failed"] = "failed"


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
    ],
)
def test_invalid_missing_and_incompatible_routes_fail_to_clarification(value: object) -> None:
    decision = route_decision_or_clarify(value)

    assert decision == RouteDecision.clarify("invalid_output")
    assert decision.request is None


def test_valid_clarification_carries_no_executable_request() -> None:
    decision = route_decision_or_clarify(
        {"decision": "clarify", "clarification_reason": "unsupported_workflow"}
    )

    assert decision.decision == "clarify"
    assert decision.request is None


def test_cancel_selector_cannot_widen_from_missing_or_empty_targets() -> None:
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
    ("payload", "valid"),
    [
        ({"operation": "add", "item_query": "shampoo", "quantity": 1}, True),
        ({"operation": "add", "item_query": "shampoo"}, False),
        ({"operation": "remove", "item_query": "shampoo"}, True),
        ({"operation": "remove", "item_query": "shampoo", "quantity": 1}, False),
        ({"operation": "set_quantity", "item_query": "shampoo", "quantity": 0}, True),
        ({"operation": "set_quantity", "item_query": "shampoo"}, False),
    ],
)
def test_cart_mutation_requires_operation_specific_fields(
    payload: dict[str, object], valid: bool
) -> None:
    if valid:
        ModifyCart.model_validate(payload)
    else:
        with pytest.raises(ValidationError):
            ModifyCart.model_validate(payload)


def test_routing_context_is_bounded_and_authority_free() -> None:
    context = RoutingContext(
        utterance="cancel all my orders",
        bound_customer=True,
        active_flow=None,
        pending_action=None,
        recent_effect="order_placed",
        available_capabilities=(CapabilityId.CANCEL_ORDERS, CapabilityId.VERIFY_ORDER_STATUS),
    )

    assert set(context.model_dump()) == {
        "utterance",
        "bound_customer",
        "active_flow",
        "pending_action",
        "recent_effect",
        "available_capabilities",
    }
    with pytest.raises(ValidationError):
        RoutingContext(
            utterance="cancel all my orders",
            bound_customer=True,
            available_capabilities=(CapabilityId.CANCEL_ORDERS, CapabilityId.CANCEL_ORDERS),
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
