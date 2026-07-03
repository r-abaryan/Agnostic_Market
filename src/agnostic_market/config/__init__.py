"""Config plane — 3-layer resolution (base -> template -> merchant override).

- loader.py   : safe YAML load (with the implicit-bool guard) + config_version hashing.
- resolver.py : deep-merge the 3 layers + reject any override touching a safety-locked key.
- registry.py : in-memory lookup of resolved MerchantConfig by merchant_id.

Models live in dtos/config.py (single source of truth) — this package holds behavior only,
never model definitions (there is deliberately no config/schemas.py).
"""

from agnostic_market.config.loader import ConfigError, load_yaml_layer
from agnostic_market.config.registry import ConfigRegistry
from agnostic_market.config.resolver import SafetyLockViolationError, resolve_merchant_config

__all__ = [
    "ConfigError",
    "ConfigRegistry",
    "SafetyLockViolationError",
    "load_yaml_layer",
    "resolve_merchant_config",
]
