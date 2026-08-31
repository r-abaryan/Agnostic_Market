"""Shared test fixtures backed by a complete synthetic configuration overlay."""

from __future__ import annotations

from pathlib import Path
from shutil import copy2, copytree, ignore_patterns

import pytest

from agnostic_market.config.registry import ConfigRegistry

# repo root = two levels up from this file (tests/ -> repo).
_REPO_ROOT = Path(__file__).resolve().parents[1]
_REPO_CONFIG_ROOT = _REPO_ROOT / "config"
_TEST_ROOT = Path(__file__).resolve().parent
_COMMITTED_FIXTURES = {
    family: _REPO_CONFIG_ROOT / "fixtures" / family / "acme_store.yaml"
    for family in ("orders", "profiles")
}
_SYNTHETIC_FIXTURES = {
    "customers": _TEST_ROOT / "synthetic_customers.yaml",
    "payment_instruments": _TEST_ROOT / "synthetic_payment_instruments.yaml",
    "verification": _TEST_ROOT / "synthetic_verification.yaml",
}


@pytest.fixture(scope="session")
def config_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("synthetic-config")
    copytree(
        _REPO_CONFIG_ROOT,
        root,
        dirs_exist_ok=True,
        ignore=ignore_patterns("fixtures", "telemetry", "reports.json"),
    )
    for family, source in (_COMMITTED_FIXTURES | _SYNTHETIC_FIXTURES).items():
        target = root / "fixtures" / family / "acme_store.yaml"
        target.parent.mkdir(parents=True)
        copy2(source, target)
    return root


@pytest.fixture
def registry(config_root: Path) -> ConfigRegistry:
    return ConfigRegistry(config_root).load()
