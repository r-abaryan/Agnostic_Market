"""Tenant-scoped catalog retrieval contracts and the validated fixture adapter."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agnostic_market.dtos.money import UsdAmount
from agnostic_market.tenancy.context import TenantBound, normalize_tenant_id

_FROZEN = ConfigDict(extra="forbid", frozen=True)
CATALOG_RESULT_MAX_ITEMS = 10


class _NamedItem(Protocol):
    name: str


class _CandidateItem(_NamedItem, Protocol):
    sku: str
    price_usd: UsdAmount


class CatalogProduct(BaseModel):
    model_config = _FROZEN

    sku: str = Field(min_length=1)
    name: str = Field(min_length=1)
    price_usd: UsdAmount


class CatalogFixture(BaseModel):
    """Validated product data shared by fixture-backed commerce adapters."""

    model_config = _FROZEN

    products: tuple[CatalogProduct, ...]

    @model_validator(mode="after")
    def product_skus_are_unique(self) -> CatalogFixture:
        seen: set[str] = set()
        duplicates: set[str] = set()
        for product in self.products:
            if product.sku in seen:
                duplicates.add(product.sku)
            seen.add(product.sku)
        if duplicates:
            raise ValueError(f"product SKUs must be unique: {sorted(duplicates)!r}")
        return self


class CatalogProductSet(BaseModel):
    """One bounded product result returned by a catalog operation."""

    model_config = _FROZEN

    products: tuple[CatalogProduct, ...] = Field(max_length=CATALOG_RESULT_MAX_ITEMS)

    @model_validator(mode="after")
    def product_skus_are_unique(self) -> CatalogProductSet:
        if len({product.sku for product in self.products}) != len(self.products):
            raise ValueError("catalog result contains duplicate product SKUs")
        return self


class Candidate(BaseModel):
    """One transient code-numbered catalog or cart option."""

    model_config = _FROZEN

    key: str = Field(min_length=1)
    sku: str = Field(min_length=1)
    name: str = Field(min_length=1)
    price_usd: UsdAmount


@runtime_checkable
class CatalogPort(TenantBound, Protocol):
    """The catalog reads required by graph capability owners."""

    def search(self, query: str) -> CatalogProductSet: ...

    def browse(self) -> CatalogProductSet: ...

    def resolve_products(self, skus: Sequence[str]) -> tuple[CatalogProduct | None, ...]: ...


def _word_tokens(value: str) -> tuple[str, ...]:
    tokens: list[str] = []
    current: list[str] = []
    for character in value.casefold():
        if character.isalnum():
            current.append(character)
        elif current:
            tokens.append("".join(current))
            current.clear()
    if current:
        tokens.append("".join(current))
    return tuple(tokens)


def _contains_token_sequence(haystack: tuple[str, ...], needle: tuple[str, ...]) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    width = len(needle)
    return any(
        haystack[index : index + width] == needle for index in range(len(haystack) - width + 1)
    )


def _rank_named_items[T: _NamedItem](items: Sequence[T], query: str) -> list[tuple[int, T]]:
    query_tokens = _word_tokens(query)
    if not query_tokens:
        return []
    query_set = frozenset(query_tokens)
    ranked: list[tuple[int, int, T]] = []
    for index, item in enumerate(items):
        name_tokens = _word_tokens(item.name)
        overlap = len(query_set.intersection(name_tokens))
        minimum_overlap = 1 if len(query_tokens) == 1 or len(name_tokens) == 1 else 2
        if overlap < minimum_overlap:
            continue
        exact_sequence_bonus = int(
            _contains_token_sequence(query_tokens, name_tokens)
            or _contains_token_sequence(name_tokens, query_tokens)
        )
        ranked.append((overlap + exact_sequence_bonus, index, item))
    ranked.sort(key=lambda candidate: (-candidate[0], candidate[1]))
    return [(score, item) for score, _index, item in ranked]


def match_named_items[T: _NamedItem](items: Sequence[T], query: str) -> list[T]:
    """Return the equally best lexical name matches for deterministic selection."""
    ranked = _rank_named_items(items, query)
    if not ranked:
        return []
    best_score = ranked[0][0]
    return [item for score, item in ranked if score == best_score]


def number_candidates(items: Sequence[_CandidateItem]) -> list[Candidate]:
    """Render live items as local model-facing keys."""
    return [
        Candidate(key=str(index), sku=item.sku, name=item.name, price_usd=item.price_usd)
        for index, item in enumerate(items, start=1)
    ]


class FixtureCatalog:
    """Validated in-memory catalog used by development and offline evaluation."""

    def __init__(self, tenant_id: str, fixture: CatalogFixture) -> None:
        self.tenant_id = normalize_tenant_id(tenant_id, boundary="fixture catalog")
        self._products = fixture.products
        self._by_sku = {product.sku: product for product in fixture.products}

    def search(self, query: str) -> CatalogProductSet:
        ranked = _rank_named_items(self._products, query)
        return CatalogProductSet(
            products=tuple(product for _score, product in ranked[:CATALOG_RESULT_MAX_ITEMS])
        )

    def browse(self) -> CatalogProductSet:
        return CatalogProductSet(products=self._products[:CATALOG_RESULT_MAX_ITEMS])

    def resolve_products(self, skus: Sequence[str]) -> tuple[CatalogProduct | None, ...]:
        return tuple(self._by_sku.get(sku) for sku in skus)
