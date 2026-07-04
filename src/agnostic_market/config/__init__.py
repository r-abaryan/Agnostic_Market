"""Config plane — 3-layer resolution (base -> template -> merchant override).

- loader.py   : safe YAML load (with the implicit-bool guard) + config_version hashing.
- resolver.py : deep-merge the 3 layers + reject any override touching a safety-locked key.
- registry.py : in-memory lookup of resolved MerchantConfig by merchant_id.

Models live in dtos/config.py (single source of truth) — this package holds behavior only,
never model definitions (there is deliberately no config/schemas.py).
"""

from agnostic_market.config.loader import ConfigError, config_version, load_yaml_layer
from agnostic_market.config.registry import (
    ConfigRegistry,
    ResolvedConfig,
    UnknownMerchantError,
)
from agnostic_market.config.resolver import (
    ConfigResolutionError,
    SafetyLockViolationError,
    resolve_merchant_config,
)

__all__ = [
    "ConfigError",
    "ConfigRegistry",
    "ConfigResolutionError",
    "ResolvedConfig",
    "SafetyLockViolationError",
    "UnknownMerchantError",
    "config_version",
    "load_yaml_layer",
    "resolve_merchant_config",
]
