"""Loading for versioned synthetic merchant scenario datasets."""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from agnostic_market.config.loader import ConfigError, load_yaml_layer
from agnostic_market.management.contracts import MerchantScenarioDataset


def load_merchant_scenario_dataset(path: Path) -> MerchantScenarioDataset:
    """Load one exact YAML dataset through the strict management contract."""

    try:
        return MerchantScenarioDataset.model_validate(load_yaml_layer(path))
    except ValidationError as exc:
        raise ConfigError(f"merchant scenario dataset {path} failed validation") from exc
