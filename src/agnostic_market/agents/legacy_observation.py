"""Closed, turn-local observation of the legacy routing path during shadow evaluation."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

from agnostic_market.dtos.orchestration import CapabilityId

LegacyObservationSource = Literal["gate", "model", "tool", "typed_owner"]

LEGACY_READ_TOOL_CAPABILITIES: Mapping[str, tuple[CapabilityId, ...]] = MappingProxyType(
    {
        "order_status": (CapabilityId.VERIFY_ORDER_STATUS,),
        "list_orders": (CapabilityId.LIST_ORDERS,),
        "catalog_search": (CapabilityId.SEARCH_CATALOG,),
        "view_cart": (CapabilityId.VIEW_CART,),
    }
)
LEGACY_FLOW_TOOL_CAPABILITIES: Mapping[str, tuple[CapabilityId, ...]] = MappingProxyType(
    {
        "propose_refund": (CapabilityId.REFUND_ORDER,),
        "propose_cancel": (CapabilityId.CANCEL_ORDERS,),
        "propose_return": (CapabilityId.RETURN_ORDER,),
        "propose_profile_change": (CapabilityId.CHANGE_PROFILE,),
        "add_to_cart": (CapabilityId.MODIFY_CART,),
        "remove_from_cart": (CapabilityId.MODIFY_CART,),
        "set_quantity": (CapabilityId.MODIFY_CART,),
        "review_cart": (CapabilityId.VIEW_CART,),
        "buy_now": (CapabilityId.MODIFY_CART, CapabilityId.PLACE_ORDER),
        "go_to_checkout": (CapabilityId.PLACE_ORDER,),
        "propose_identity": (CapabilityId.VERIFY_IDENTITY, CapabilityId.SWITCH_ACCOUNT),
    }
)
LEGACY_HANDOVER_CAPABILITIES: Mapping[str, tuple[CapabilityId, ...]] = MappingProxyType(
    {
        "address_change": (CapabilityId.CHANGE_PROFILE,),
        "contact_change": (CapabilityId.CHANGE_PROFILE,),
        "cancel_order": (CapabilityId.CANCEL_ORDERS,),
        "refund": (CapabilityId.REFUND_ORDER, CapabilityId.RETURN_ORDER),
        "cart_write": (CapabilityId.MODIFY_CART, CapabilityId.PLACE_ORDER),
        "list_orders": (CapabilityId.LIST_ORDERS,),
        "verification_required": (CapabilityId.VERIFY_IDENTITY,),
        "switch_account": (CapabilityId.SWITCH_ACCOUNT,),
    }
)
LEGACY_FLOW_CAPABILITIES: Mapping[str, tuple[CapabilityId, ...]] = MappingProxyType(
    {
        "support": (
            CapabilityId.CANCEL_ORDERS,
            CapabilityId.REFUND_ORDER,
            CapabilityId.RETURN_ORDER,
            CapabilityId.CHANGE_PROFILE,
        ),
        "cart": (CapabilityId.MODIFY_CART, CapabilityId.PLACE_ORDER),
        "identity": (
            CapabilityId.VERIFY_IDENTITY,
            CapabilityId.SWITCH_ACCOUNT,
            CapabilityId.LIST_ORDERS,
        ),
    }
)


@dataclass(slots=True)
class LegacyRouteObservation:
    """Mutable collector shared by graph tasks for one admitted ordinary turn."""

    turn_id: str
    capability_candidates: set[CapabilityId] = field(default_factory=set)
    owner_sources: set[LegacyObservationSource] = field(default_factory=set)
    answer_sources: set[LegacyObservationSource] = field(default_factory=set)

    def as_event(self) -> dict[str, object]:
        capabilities = sorted(capability.value for capability in self.capability_candidates)
        answer_sources = sorted(self.answer_sources)
        return {
            "event": "semantic_route_legacy_observation",
            "turn_id": self.turn_id,
            "decision_source": "legacy",
            "capability_candidates": capabilities,
            "owner_sources": sorted(self.owner_sources),
            "answer_sources": answer_sources,
            "answered": bool(answer_sources),
            "non_tool_ai_text_without_owner": bool(answer_sources)
            and not capabilities
            and self.answer_sources == {"model"},
        }


_CURRENT_OBSERVATION: ContextVar[LegacyRouteObservation | None] = ContextVar(
    "legacy_route_observation",
    default=None,
)


@contextmanager
def legacy_route_observation_scope(turn_id: str) -> Iterator[LegacyRouteObservation]:
    """Open the sole legacy-observation collector for one shadowed ordinary turn."""

    if not turn_id.strip():
        raise ValueError("legacy observation requires a nonblank turn id")
    if _CURRENT_OBSERVATION.get() is not None:
        raise RuntimeError("legacy observation scopes cannot be nested")
    observation = LegacyRouteObservation(turn_id=turn_id)
    token = _CURRENT_OBSERVATION.set(observation)
    try:
        yield observation
    finally:
        _CURRENT_OBSERVATION.reset(token)


def observe_legacy_capabilities(
    capabilities: Sequence[CapabilityId],
    *,
    source: LegacyObservationSource,
) -> None:
    observation = _CURRENT_OBSERVATION.get()
    if observation is None or not capabilities:
        return
    observation.capability_candidates.update(capabilities)
    observation.owner_sources.add(source)


def observe_legacy_answer(*, source: LegacyObservationSource) -> None:
    observation = _CURRENT_OBSERVATION.get()
    if observation is not None:
        observation.answer_sources.add(source)


def observe_legacy_model_tool_calls(tool_calls: Sequence[Mapping[str, object]]) -> None:
    """Record closed legacy flow nominations at the model-output boundary."""

    for call in tool_calls:
        name = call.get("name")
        if isinstance(name, str):
            observe_legacy_capabilities(
                LEGACY_FLOW_TOOL_CAPABILITIES.get(name, ()),
                source="model",
            )
