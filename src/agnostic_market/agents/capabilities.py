"""Immutable per-session capability-to-graph-entry registry."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from agnostic_market.dtos.orchestration import (
    CapabilityId,
    IntentRequest,
    IntentRequestModel,
)


class CapabilityRegistryError(RuntimeError):
    """Capability assembly or resolution violated the typed contract."""


@dataclass(frozen=True, slots=True)
class CapabilityEntry:
    """Opaque graph-entry token owned by per-session graph assembly."""

    node_name: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.node_name, str)
            or not self.node_name.strip()
            or self.node_name != self.node_name.strip()
        ):
            raise CapabilityRegistryError("capability entry requires a normalized node name")


@dataclass(frozen=True, slots=True)
class CapabilitySpec:
    """One capability's declared request type and the node that owns it.

    The id must be the enum member, not its string value: `CapabilityId` is a `StrEnum`, so a
    raw string would compare equal to `kind` and leak an untyped id into `capability_ids`.
    """

    capability_id: CapabilityId
    request_type: type[IntentRequestModel]
    entry: CapabilityEntry

    def __post_init__(self) -> None:
        if not isinstance(self.capability_id, CapabilityId):
            raise CapabilityRegistryError("capability_id must be a CapabilityId")
        if not isinstance(self.entry, CapabilityEntry):
            raise CapabilityRegistryError("entry must be a CapabilityEntry")
        if not isinstance(self.request_type, type) or not issubclass(
            self.request_type, IntentRequestModel
        ):
            raise CapabilityRegistryError("request_type must be an IntentRequestModel subclass")
        kind = self.request_type.model_fields.get("kind")
        if kind is None or kind.default != self.capability_id:
            raise CapabilityRegistryError(
                "request_type kind must equal the registered capability_id"
            )


@dataclass(frozen=True, slots=True, init=False)
class CapabilityRegistry:
    """The session's whole capability set, fixed at graph assembly.

    `resolve()` validates a typed request against its registered type and returns the owning
    entry; the dispatcher does nothing else. Frozen because availability is projected to the
    router and must not change mid-turn; an unregistered id resolves to nothing, never to a
    fallback owner.
    """

    _specs: Mapping[CapabilityId, CapabilitySpec]
    _capability_ids: tuple[CapabilityId, ...]

    def __init__(self, specs: Iterable[CapabilitySpec]) -> None:
        by_id: dict[CapabilityId, CapabilitySpec] = {}
        for spec in specs:
            if not isinstance(spec, CapabilitySpec):
                raise CapabilityRegistryError("registry accepts only CapabilitySpec entries")
            if spec.capability_id in by_id:
                raise CapabilityRegistryError(
                    f"capability {spec.capability_id.value!r} is already registered"
                )
            by_id[spec.capability_id] = spec
        object.__setattr__(self, "_specs", MappingProxyType(by_id))
        object.__setattr__(self, "_capability_ids", tuple(by_id))

    @property
    def capability_ids(self) -> tuple[CapabilityId, ...]:
        return self._capability_ids

    @property
    def specs(self) -> Mapping[CapabilityId, CapabilitySpec]:
        return self._specs

    def resolve(self, request: IntentRequest) -> CapabilityEntry:
        if not isinstance(request, IntentRequestModel) or not isinstance(
            getattr(request, "kind", None), CapabilityId
        ):
            raise CapabilityRegistryError("resolver requires a typed IntentRequest")
        try:
            spec = self._specs[request.kind]
        except KeyError as exc:
            raise CapabilityRegistryError(
                f"capability {request.kind.value!r} is not registered"
            ) from exc
        if not isinstance(request, spec.request_type):
            raise CapabilityRegistryError(
                f"capability {request.kind.value!r} received an incompatible request type"
            )
        return spec.entry
