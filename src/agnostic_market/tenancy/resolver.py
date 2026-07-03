"""Resolve an inbound signal (phone# / domain / explicit button) to a merchant_id.

Backed by the ConfigRegistry: the inbound-number index is built from each merchant's
`telephony.inbound_number`. An explicit button press carries a merchant_id directly
(validated against the registry). Unknown signals fail loudly — a caller is never silently
mapped to the wrong tenant.

DID (inbound number) is the isolation anchor for inbound calls (DESIGN_REVIEW L6). ANI /
caller-ID is NOT used for tenant resolution of sensitive actions (SECURITY §2a).
"""

from __future__ import annotations

from agnostic_market.config.registry import ConfigRegistry


class TenantResolutionError(RuntimeError):
    """An inbound signal did not resolve to a known merchant."""


class TenantResolver:
    """Map phone# / domain / explicit merchant_id -> merchant_id, via the registry."""

    def __init__(self, registry: ConfigRegistry) -> None:
        self._registry = registry
        self._by_inbound_number: dict[str, str] = {}
        self._build_indexes()

    def _build_indexes(self) -> None:
        """Index inbound numbers -> merchant_id from resolved configs. Reject collisions."""
        by_number: dict[str, str] = {}
        for merchant_id in self._registry.merchant_ids:
            config = self._registry.get(merchant_id).config
            number = config.telephony.inbound_number
            if number in by_number:
                raise TenantResolutionError(
                    f"inbound number {number} is claimed by both "
                    f"{by_number[number]!r} and {merchant_id!r}"
                )
            by_number[number] = merchant_id
        self._by_inbound_number = by_number

    def resolve_by_number(self, inbound_number: str) -> str:
        """Inbound DID -> merchant_id (the primary inbound path)."""
        merchant_id = self._by_inbound_number.get(inbound_number)
        if merchant_id is None:
            raise TenantResolutionError(f"no merchant for inbound number {inbound_number!r}")
        return merchant_id

    def resolve_by_button(self, merchant_id: str) -> str:
        """Explicit merchant selection (UI button) -> validated merchant_id."""
        if merchant_id not in self._registry.merchant_ids:
            raise TenantResolutionError(f"unknown merchant_id {merchant_id!r}")
        return merchant_id
