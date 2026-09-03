"""Resolve trusted inbound signals to a configured merchant identity.

Inbound calls resolve from their canonical DID. Approved server-side dispatches resolve
from an explicit merchant identity. Unknown or malformed signals fail closed.
"""

from __future__ import annotations

from pydantic import TypeAdapter, ValidationError

from agnostic_market.config.registry import ConfigRegistry
from agnostic_market.dtos.config import E164PhoneNumber

_INBOUND_NUMBER_ADAPTER = TypeAdapter(E164PhoneNumber)


class TenantResolutionError(RuntimeError):
    """An inbound signal did not resolve to a known merchant."""


def _canonical_inbound_number(value: str) -> str:
    try:
        return _INBOUND_NUMBER_ADAPTER.validate_python(value, strict=True)
    except ValidationError as exc:
        raise TenantResolutionError("inbound number must use canonical E.164 format") from exc


class TenantResolver:
    """Map an inbound DID or trusted merchant identity through the registry."""

    def __init__(self, registry: ConfigRegistry) -> None:
        self._registry = registry
        self._by_inbound_number: dict[str, str] = {}
        self._build_indexes()

    def _build_indexes(self) -> None:
        """Index inbound numbers -> merchant_id from resolved configs. Reject collisions."""
        by_number: dict[str, str] = {}
        for merchant_id in self._registry.merchant_ids:
            config = self._registry.get(merchant_id).config
            number = _canonical_inbound_number(config.telephony.inbound_number)
            if number in by_number:
                raise TenantResolutionError(
                    f"inbound number {number} is claimed by both "
                    f"{by_number[number]!r} and {merchant_id!r}"
                )
            by_number[number] = merchant_id
        self._by_inbound_number = by_number

    def resolve_by_number(self, inbound_number: str) -> str:
        """Inbound DID -> merchant_id (the primary inbound path)."""
        canonical_number = _canonical_inbound_number(inbound_number)
        merchant_id = self._by_inbound_number.get(canonical_number)
        if merchant_id is None:
            raise TenantResolutionError(f"no merchant for inbound number {canonical_number!r}")
        return merchant_id

    def resolve_by_id(self, merchant_id: str) -> str:
        """Validate an explicit merchant identity against the registry."""
        if merchant_id not in self._registry.merchant_ids:
            raise TenantResolutionError(f"unknown merchant_id {merchant_id!r}")
        return merchant_id
