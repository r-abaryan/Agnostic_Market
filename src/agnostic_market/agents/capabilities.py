"""Single typed capability registry and executor contract.

Milestone 1 defines and tests the seam without changing live routing or registering duplicate
adapters for the existing flows. Per-session adapters land with direct-routing integration.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel

from agnostic_market.dtos.orchestration import CapabilityId, CapabilityOutcome, IntentRequest

CapabilityAdapter = Callable[[BaseModel], Awaitable[BaseModel]]


class CapabilityRegistryError(RuntimeError):
    """A capability registration or execution violated the typed contract."""


@dataclass(frozen=True, slots=True)
class CapabilitySpec:
    capability_id: CapabilityId
    request_type: type[BaseModel]
    outcome_type: type[CapabilityOutcome]
    adapter: CapabilityAdapter
    effect: Literal["read", "write"]
    write_serialization_key: str | None = None
    planner_ready: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.request_type, type) or not issubclass(self.request_type, BaseModel):
            raise CapabilityRegistryError("request_type must be a BaseModel subclass")
        if not isinstance(self.outcome_type, type) or not issubclass(
            self.outcome_type, CapabilityOutcome
        ):
            raise CapabilityRegistryError("outcome_type must be a CapabilityOutcome subclass")
        if not callable(self.adapter):
            raise CapabilityRegistryError("adapter must be callable")
        if self.effect not in ("read", "write"):
            raise CapabilityRegistryError("effect must be 'read' or 'write'")
        if not isinstance(self.planner_ready, bool):
            raise CapabilityRegistryError("planner_ready must be a bool")
        kind = self.request_type.model_fields.get("kind")
        if kind is None or kind.default != self.capability_id:
            raise CapabilityRegistryError(
                "request_type kind must equal the registered capability_id"
            )
        if self.outcome_type is CapabilityOutcome:
            raise CapabilityRegistryError("outcome_type must be a concrete CapabilityOutcome")
        if self.effect == "read" and self.write_serialization_key is not None:
            raise CapabilityRegistryError("read capabilities cannot have a write serialization key")
        if self.effect == "write" and (
            self.write_serialization_key is None
            or not self.write_serialization_key.strip()
            or self.write_serialization_key != self.write_serialization_key.strip()
        ):
            raise CapabilityRegistryError(
                "write capabilities require a non-empty normalized serialization key"
            )


class CapabilityRegistry:
    def __init__(self, specs: Iterable[CapabilitySpec] = ()) -> None:
        self._specs: dict[CapabilityId, CapabilitySpec] = {}
        for spec in specs:
            self.register(spec)

    @property
    def capability_ids(self) -> tuple[CapabilityId, ...]:
        return tuple(sorted(self._specs, key=str))

    @property
    def planner_ready_ids(self) -> tuple[CapabilityId, ...]:
        return tuple(
            capability_id
            for capability_id in self.capability_ids
            if self._specs[capability_id].planner_ready
        )

    def register(self, spec: CapabilitySpec) -> None:
        if spec.capability_id in self._specs:
            raise CapabilityRegistryError(
                f"capability {spec.capability_id.value!r} is already registered"
            )
        self._specs[spec.capability_id] = spec

    def get(self, capability_id: CapabilityId) -> CapabilitySpec:
        try:
            return self._specs[capability_id]
        except KeyError as exc:
            raise CapabilityRegistryError(
                f"capability {capability_id.value!r} is not registered"
            ) from exc

    async def execute(self, request: IntentRequest) -> CapabilityOutcome:
        if not isinstance(request, BaseModel) or not isinstance(
            getattr(request, "kind", None), CapabilityId
        ):
            raise CapabilityRegistryError("executor requires a typed IntentRequest")
        spec = self.get(request.kind)
        if not isinstance(request, spec.request_type):
            raise CapabilityRegistryError(
                f"capability {request.kind.value!r} received an incompatible request type"
            )
        outcome = await spec.adapter(request)
        if not isinstance(outcome, spec.outcome_type):
            raise CapabilityRegistryError(
                f"capability {request.kind.value!r} returned an incompatible outcome type"
            )
        return outcome
