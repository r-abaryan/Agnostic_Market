"""In-memory tenant config registry — resolve all merchants once, serve by merchant_id.

Loads the config tree (base + templates + merchants), resolves each merchant through the
3 layers, validates it, and caches the effective MerchantConfig plus its config_version.
`get(merchant_id)` returns the resolved config; unknown ids fail loudly.

Hot-reload is a documented stub for Phase 0 — the full swap-under-live-traffic semantics
(with per-session config_version pinning so an in-flight call is unaffected, DESIGN_REVIEW
M4) land when the registry actually serves live traffic.

Expected config tree (BUILD_PLAN repo layout):
    <root>/base/base.yaml
    <root>/templates/<vertical>/template.yaml
    <root>/merchants/<merchant_id>.yaml   (override; declares `extends_template`)
"""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agnostic_market.config.loader import ConfigError, config_version, load_yaml_layer
from agnostic_market.config.resolver import resolve_merchant_config
from agnostic_market.dtos.config import MerchantConfig

# `extends_template` comes from the merchant override (semi-trusted, merchant-editable) and
# is used to build a filesystem path — so it must be a bare name, never a path fragment.
# This whitelist blocks traversal ('.', '/', '\\' are all excluded) and matches our naming.
_TEMPLATE_NAME_RE = re.compile(r"^[a-z0-9_-]+$")


class UnknownMerchantError(KeyError):
    """No config resolved for the requested merchant_id."""


@dataclass(frozen=True)
class ResolvedConfig:
    """A merchant's effective config plus the version hash pinned per session."""

    config: MerchantConfig
    config_version: str


def _resolve_merchant_override(
    root: Path,
    base: dict[str, Any],
    override: dict[str, Any],
    *,
    source: str | Path,
) -> ResolvedConfig:
    template_name = override.get("extends_template")
    if not template_name:
        raise ConfigError(f"{source} is missing required 'extends_template'")
    if not _TEMPLATE_NAME_RE.match(str(template_name)):
        raise ConfigError(
            f"{source} has invalid extends_template {template_name!r} "
            f"(must match {_TEMPLATE_NAME_RE.pattern} - no path separators)"
        )
    template = load_yaml_layer(root / "templates" / str(template_name) / "template.yaml")
    config = resolve_merchant_config(base, template, override)
    return ResolvedConfig(
        config=config,
        config_version=config_version(config.model_dump(mode="json")),
    )


def resolve_merchant_override(
    root: Path,
    override: dict[str, Any],
    *,
    source: str | Path = "merchant override",
) -> ResolvedConfig:
    """Resolve one in-memory merchant override through the production three-layer path."""

    base = load_yaml_layer(root / "base" / "base.yaml")
    return _resolve_merchant_override(root, base, override, source=source)


class ConfigRegistry:
    """Resolve and hold every merchant's effective config, keyed by merchant_id."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._by_id: dict[str, ResolvedConfig] = {}
        self._source_overrides: dict[str, dict[str, Any]] = {}

    @property
    def merchant_ids(self) -> frozenset[str]:
        return frozenset(self._by_id)

    def load(self) -> ConfigRegistry:
        """Load + resolve every merchant override file under <root>/merchants/. Fails loudly."""
        base = load_yaml_layer(self._root / "base" / "base.yaml")
        merchants_dir = self._root / "merchants"
        if not merchants_dir.is_dir():
            raise ConfigError(f"merchants directory not found: {merchants_dir}")

        resolved: dict[str, ResolvedConfig] = {}
        source_overrides: dict[str, dict[str, Any]] = {}
        for override_path in sorted(merchants_dir.glob("*.yaml")):
            override = load_yaml_layer(override_path)
            merchant = _resolve_merchant_override(
                self._root,
                base,
                override,
                source=override_path,
            )
            if merchant.config.merchant_id in resolved:
                raise ConfigError(
                    f"duplicate merchant_id '{merchant.config.merchant_id}' in {override_path}"
                )
            resolved[merchant.config.merchant_id] = merchant
            source_overrides[merchant.config.merchant_id] = override

        self._by_id = resolved
        self._source_overrides = source_overrides
        return self

    def get(self, merchant_id: str) -> ResolvedConfig:
        try:
            return self._by_id[merchant_id]
        except KeyError as exc:
            raise UnknownMerchantError(f"no config for merchant_id '{merchant_id}'") from exc

    def source_override(self, merchant_id: str) -> dict[str, Any]:
        """Return a detached copy of the source layer resolved for one merchant."""

        try:
            return deepcopy(self._source_overrides[merchant_id])
        except KeyError as exc:
            raise UnknownMerchantError(f"no config for merchant_id '{merchant_id}'") from exc

    def reload(self) -> ConfigRegistry:
        """Phase-0 stub: full reload (no live-traffic swap semantics yet).

        Later: swap under live traffic with per-session config_version pinning so an
        in-flight checkout keeps its pinned version (DESIGN_REVIEW M4). For now, just
        re-resolve everything.
        """
        return self.load()
