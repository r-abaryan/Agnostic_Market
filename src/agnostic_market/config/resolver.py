"""3-layer config resolution + the safety-lock check (the load-bearing security control).

Resolution order (BUILD_PLAN "Merchant config — 3-layer resolution"):
    platform base  (safety-locked)  ->  vertical template  ->  merchant override
Deep-merged last-wins, THEN validated into `MerchantConfig` (dtos/config.py).

Safety-lock: the base layer declares `_safety_locked` — a list of dotted key-paths that are
platform-enforced and NOT merchant-editable (guardrails, PCI/payment, verification tiers,
committed-only auth, egress, CRITICAL_CONFIRMATION_FIELDS, tenant-isolation). The resolver
REJECTS any template/override that sets a locked path (or a descendant of one). Lock is
declared in the locked base (data) and enforced here (code) — "platform code + locked base".
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from agnostic_market.dtos.config import MerchantConfig

_SAFETY_LOCK_DECLARATION_KEY = "_safety_locked"
# Platform-only sections: present in base for the safety-lock to protect, but NOT fields of
# MerchantConfig (which models only the merchant-editable surface). Stripped before validation.
_PLATFORM_ONLY_KEYS = ("_safety_locked", "_platform")


class SafetyLockViolationError(RuntimeError):
    """A template/override attempted to set a platform-safety-locked key path."""


class ConfigResolutionError(RuntimeError):
    """The resolved config failed schema validation."""


def _iter_key_paths(data: dict[str, Any], prefix: str = "") -> list[str]:
    """All dotted key-paths present in a nested mapping (leaves and interior nodes)."""
    paths: list[str] = []
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        paths.append(path)
        if isinstance(value, dict):
            paths.extend(_iter_key_paths(value, path))
    return paths


def _is_locked(path: str, locked: frozenset[str]) -> bool:
    """True if `path` is a locked key or a descendant of one (e.g. `guardrails.caps.x`)."""
    if path in locked:
        return True
    return any(path.startswith(f"{lock}.") for lock in locked)


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursive last-wins merge; dict values merge, scalars/lists overwrite."""
    merged = dict(base)
    for key, value in overlay.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def _assert_no_locked_keys(
    layer: dict[str, Any], locked: frozenset[str], *, layer_name: str
) -> None:
    """Reject a layer touching any safety-locked path. Precise error (which key, which layer)."""
    for path in _iter_key_paths(layer):
        if _is_locked(path, locked):
            raise SafetyLockViolationError(
                f"{layer_name} layer attempts to set safety-locked key '{path}' "
                f"- platform-enforced keys are not merchant-editable"
            )


def resolve_merchant_config(
    base: dict[str, Any],
    template: dict[str, Any],
    override: dict[str, Any],
) -> MerchantConfig:
    """Resolve base -> template -> override into a validated MerchantConfig.

    Raises SafetyLockViolationError if template/override touches a locked key, or
    ConfigResolutionError if the merged config fails schema validation.
    """
    locked = frozenset(base.get(_SAFETY_LOCK_DECLARATION_KEY, []))

    # Neither non-base layer may touch a locked path.
    _assert_no_locked_keys(template, locked, layer_name="template")
    _assert_no_locked_keys(override, locked, layer_name="override")

    merged = _deep_merge(_deep_merge(base, template), override)
    # Platform-only sections (the lock declaration + the `_platform` safety block) are
    # directives, not MerchantConfig fields — drop them before validation. The DTO forbids
    # extras, so this also keeps them out of the effective merchant config.
    for platform_key in _PLATFORM_ONLY_KEYS:
        merged.pop(platform_key, None)

    try:
        return MerchantConfig.model_validate(merged)
    except ValidationError as exc:
        raise ConfigResolutionError(f"resolved config failed validation:\n{exc}") from exc
