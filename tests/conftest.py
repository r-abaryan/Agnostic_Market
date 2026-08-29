"""Shared test fixtures — the real config tree under ./config is the Phase-0 test data."""

from __future__ import annotations

from pathlib import Path

import pytest

from agnostic_market.config.registry import ConfigRegistry

# repo root = two levels up from this file (tests/ -> repo).
_REPO_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_ROOT = _REPO_ROOT / "config"


@pytest.fixture
def config_root() -> Path:
    return _CONFIG_ROOT


@pytest.fixture
def registry(config_root: Path) -> ConfigRegistry:
    return ConfigRegistry(config_root).load()
