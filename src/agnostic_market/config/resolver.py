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
# The always-locked floor — enforced in CODE, independent of what base.yaml declares. The
# base's `_safety_locked` list only EXTENDS this floor; it can never define or lower it, so a
# base fixture that forgets to list `_platform` can't accidentally make it merchant-overridable
# ("authority in code, not data"). `_safety_locked` itself is locked: an override/template that
# ships one is tampering with the lock and is rejected.
_ALWAYS_LOCKED: frozenset[str] = frozenset(_PLATFORM_ONLY_KEYS)


class SafetyLockViolationError(RuntimeError):
    """A template/override attempted to set a platform-safety-locked key path."""


class ConfigResolutionError(RuntimeError):
    """The resolved config failed schema validation."""


def _iter_key_paths(data: dict[str, Any], prefix: str = "") -> list[str]:
    """All dotted key-paths in a nested mapping, ANCESTOR-BEFORE-DESCENDANT.

    The ordering matters for the safety-lock: a locked interior node (e.g. `_platform`) is
    always yielded before its children, so `_assert_no_locked_keys` catches an override of a
    locked subtree at the ancestor — an exact-match check on each path suffices (no need to
    also prefix-match descendants).
    """
    paths: list[str] = []
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        paths.append(path)  # ancestor first
        if isinstance(value, dict):
            paths.extend(_iter_key_paths(value, path))
    return paths


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
        if path in locked:
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
    # The lock floor is fixed in code; base may only EXTEND it, never shrink it.
    locked = _ALWAYS_LOCKED | frozenset(base.get(_SAFETY_LOCK_DECLARATION_KEY, []))

    # Neither non-base layer may touch a locked path (this includes `_safety_locked` itself,
    # so an override can't ship its own lock declaration to weaken enforcement).
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
