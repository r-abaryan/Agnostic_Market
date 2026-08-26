"""Exact monetary value types shared across configuration, state, and commerce."""

from __future__ import annotations

from decimal import Decimal
from typing import Annotated

from pydantic import Field, TypeAdapter

UsdAmount = Annotated[
    Decimal,
    Field(ge=Decimal("0"), decimal_places=2, allow_inf_nan=False),
]
PositiveUsdAmount = Annotated[
    Decimal,
    Field(gt=Decimal("0"), decimal_places=2, allow_inf_nan=False),
]

_USD_AMOUNT_ADAPTER = TypeAdapter(UsdAmount)


def validate_usd(value: object) -> Decimal:
    """Validate an external store argument as a non-negative USD amount."""

    return _USD_AMOUNT_ADAPTER.validate_python(value)
