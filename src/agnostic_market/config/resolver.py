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


class PolicyBoundsViolationError(RuntimeError):
    """A merchant policy value exceeds a platform ceiling (`_platform.limits`), i.e. tried to
    disable a guard rather than tune it within bounds."""


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


def _assert_policy_within_bounds(merged: dict[str, Any]) -> None:
    """Reject a merchant policy that exceeds a platform ceiling (the "within bounds" half of
    policy-within-bounds; the lock-check is the "can't touch locked keys" half).

    Reads `_platform.limits` from the MERGED config (before `_platform` is stripped) — this
    is the ONE place that sees both the effective merchant policy and the locked ceilings.
    Loud at config-load (like SafetyLockViolationError), not a silent clamp: a merchant learns
    at onboarding that a value is out of bounds, never gets a quietly-lowered guard mid-call.
    """
    limits = merged.get("_platform", {}).get("limits", {})
    policies = merged.get("policies", {})
    refunds = policies.get("refunds", {})

    require_human = refunds.get("require_human_above_usd")
    ceiling = limits.get("refund_require_human_ceiling_usd")
    if require_human is not None and ceiling is not None and require_human > ceiling:
        raise PolicyBoundsViolationError(
            f"policies.refunds.require_human_above_usd={require_human} exceeds the platform "
            f"ceiling {ceiling} - a merchant may lower the human-review threshold, never disable it"
        )

    # Ordering sanity: auto-approve can't sit above the human-review line (would be incoherent).
    auto = refunds.get("auto_approve_under_usd")
    if auto is not None and require_human is not None and auto > require_human:
        raise PolicyBoundsViolationError(
            f"policies.refunds.auto_approve_under_usd={auto} is above "
            f"require_human_above_usd={require_human} - auto-approve cannot exceed the human line"
        )

    returnless = refunds.get("returnless_under_usd")
    returnless_ceiling = limits.get("refund_returnless_ceiling_usd")
    if (
        returnless is not None
        and returnless_ceiling is not None
        and returnless > returnless_ceiling
    ):
        raise PolicyBoundsViolationError(
            f"policies.refunds.returnless_under_usd={returnless} exceeds the platform ceiling "
            f"{returnless_ceiling} - the returnless window may be widened only up to the "
            "platform bound; above it, shipped refunds are return-first"
        )

    ttl = policies.get("pending_confirmation_ttl_seconds")
    ttl_max = limits.get("pending_confirmation_ttl_max_seconds")
    if ttl is not None and ttl_max is not None and ttl > ttl_max:
        raise PolicyBoundsViolationError(
            f"policies.pending_confirmation_ttl_seconds={ttl} exceeds the platform ceiling "
            f"{ttl_max} - the confirmation window may be shortened, never made unbounded"
        )

    window = policies.get("returns", {}).get("window_days")
    window_max = limits.get("return_window_max_days")
    if window is not None and window_max is not None and window > window_max:
        raise PolicyBoundsViolationError(
            f"policies.returns.window_days={window} exceeds the platform ceiling "
            f"{window_max} - the return window may be widened only up to the platform bound"
        )


def resolve_merchant_config(
    base: dict[str, Any],
    template: dict[str, Any],
    override: dict[str, Any],
) -> MerchantConfig:
    """Resolve base -> template -> override into a validated MerchantConfig.

    Raises SafetyLockViolationError if template/override touches a locked key,
    PolicyBoundsViolationError if a merchant policy value exceeds a platform ceiling, or
    ConfigResolutionError if the merged config fails schema validation.
    """
    # The lock floor is fixed in code; base may only EXTEND it, never shrink it.
    locked = _ALWAYS_LOCKED | frozenset(base.get(_SAFETY_LOCK_DECLARATION_KEY, []))

    # Neither non-base layer may touch a locked path (this includes `_safety_locked` itself,
    # so an override can't ship its own lock declaration to weaken enforcement).
    _assert_no_locked_keys(template, locked, layer_name="template")
    _assert_no_locked_keys(override, locked, layer_name="override")

    merged = _deep_merge(_deep_merge(base, template), override)
    # Bounds check runs on the MERGED config while `_platform.limits` is still present.
    _assert_policy_within_bounds(merged)
    # Platform-only sections (the lock declaration + the `_platform` safety block) are
    # directives, not MerchantConfig fields — drop them before validation. The DTO forbids
    # extras, so this also keeps them out of the effective merchant config.
    for platform_key in _PLATFORM_ONLY_KEYS:
        merged.pop(platform_key, None)

    try:
        return MerchantConfig.model_validate(merged)
    except ValidationError as exc:
        raise ConfigResolutionError(f"resolved config failed validation:\n{exc}") from exc
