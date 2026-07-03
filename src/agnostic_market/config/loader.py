"""YAML loading with the implicit-bool guard, plus config_version hashing.

Why a custom loader (Phase-0 plan, Step 4): PyYAML resolves the implicit-bool tokens
(yes/no/on/off/true/false/y/n) to Python `bool` AT PARSE TIME, before any schema sees the
value. So a string field whose value is `no`/`off`/`NO` (e.g. Norway's country code) would
be silently turned into `False`, and strict Pydantic downstream can't reject what was
already coerced. The fix must be at the loader: we strip the implicit-bool resolver so those
tokens stay strings unless the author writes a canonical `true`/`false` (which we resolve).
Combined with StrictBool on the DTOs, booleans must be explicit and unambiguous.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import yaml


class ConfigError(RuntimeError):
    """A config file is missing or is not valid YAML / not a mapping."""


class _NoImplicitBoolLoader(yaml.SafeLoader):
    """SafeLoader that does NOT implicitly resolve bool tokens.

    A subclass so we mutate a private copy of the resolver table, never PyYAML's global
    SafeLoader (which would leak the behavior process-wide).
    """


def _strip_implicit_bool_resolvers(loader_cls: type[yaml.SafeLoader]) -> None:
    """Remove only the bool implicit resolvers from a loader's resolver table.

    `yaml_implicit_resolvers` maps a first-character -> list of (tag, regex). We drop the
    bool entries and keep everything else (int, float, null, timestamp, ...) intact.
    """
    bool_tag = "tag:yaml.org,2002:bool"
    # Copy-then-reassign per first-char so we don't mutate an inherited shared list.
    loader_cls.yaml_implicit_resolvers = {
        ch: [(tag, regex) for (tag, regex) in mappings if tag != bool_tag]
        for ch, mappings in loader_cls.yaml_implicit_resolvers.items()
    }


_strip_implicit_bool_resolvers(_NoImplicitBoolLoader)

# Canonical booleans are still accepted: re-add ONLY `true`/`false` (and capitalized forms),
# so intentional booleans work while ambiguous tokens (yes/no/on/off/y/n) stay strings.
_NoImplicitBoolLoader.add_implicit_resolver(
    "tag:yaml.org,2002:bool",
    re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"),
    list("tTfF"),
)


def load_yaml_layer(path: Path) -> dict[str, Any]:
    """Load one YAML file into a dict, with the implicit-bool guard. Fails loudly."""
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        text = path.read_text(encoding="utf-8")
        data = yaml.load(text, Loader=_NoImplicitBoolLoader)  # noqa: S506 — custom SafeLoader subclass
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(
            f"config file {path} must be a mapping at the top level, got {type(data).__name__}"
        )
    return data


def config_version(resolved: dict[str, Any]) -> str:
    """Stable content hash of a resolved config bundle (pinned per session, AGENTS §A1).

    Deterministic: sorted keys + compact separators, so the same bundle always hashes the
    same and any change flips the hash (DESIGN_REVIEW M4).
    """
    canonical = json.dumps(resolved, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
