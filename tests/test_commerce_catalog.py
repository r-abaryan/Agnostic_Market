"""Catalog port contracts and fixture-adapter retrieval behavior."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from agnostic_market.commerce.catalog import (
    CATALOG_RESULT_MAX_ITEMS,
    Candidate,
    CatalogFixture,
    CatalogProduct,
    CatalogProductSet,
    FixtureCatalog,
    match_named_items,
    number_candidates,
)
from agnostic_market.commerce.orders import load_orders_fixture
from agnostic_market.dtos.state import CartLine


def _catalog(config_root: Path) -> FixtureCatalog:
    fixture = load_orders_fixture(config_root, "acme_store")
    return FixtureCatalog("acme_store", fixture)


def _candidate(key: str, name: str) -> Candidate:
    return Candidate(key=key, sku=f"SKU-{key}", name=name, price_usd=1.0)


def test_match_named_items_accepts_a_product_name_inside_natural_utterance() -> None:
    items = [_candidate("1", "trail running shoes"), _candidate("2", "rain jacket")]

    assert match_named_items(items, "I want the trail running shoes") == [items[0]]


@pytest.mark.parametrize("query", ("", "   ", "zzz-nothing"))
def test_match_named_items_returns_empty_without_an_actual_match(query: str) -> None:
    items = [_candidate("1", "trail running shoes"), _candidate("2", "rain jacket")]

    assert match_named_items(items, query) == []


def test_match_named_items_preserves_ambiguity_for_the_caller_to_narrow() -> None:
    items = [_candidate("1", "trail running shoes"), _candidate("2", "trail hiking shoes")]

    assert match_named_items(items, "trail") == items


@pytest.mark.parametrize(
    ("query", "product_name"),
    (
        ("guarantee delivery", "Tee"),
        ("escape this", "Cap"),
        ("recap my order", "Cap"),
    ),
)
def test_match_named_items_rejects_substrings_inside_unrelated_words(
    query: str,
    product_name: str,
) -> None:
    item = _candidate("1", product_name)

    assert match_named_items([item], query) == []


def test_fixture_catalog_browse_is_explicit_and_bounded(config_root: Path) -> None:
    catalog = _catalog(config_root)

    result = catalog.browse()

    assert 0 < len(result.products) <= CATALOG_RESULT_MAX_ITEMS
    assert catalog.search("").products == ()


@pytest.mark.parametrize(
    ("query", "matched_names"),
    (
        ("running", ("trail running shoes",)),
        ("  RUNNING  ", ("trail running shoes",)),
        ("I want the trail running shoes", ("trail running shoes",)),
        ("Do you carry the waterproof rain jacket?", ("waterproof rain jacket",)),
        ("walking shoes", ()),
        ("", ()),
    ),
)
def test_fixture_catalog_search_handles_natural_product_phrasing(
    config_root: Path,
    query: str,
    matched_names: tuple[str, ...],
) -> None:
    result = _catalog(config_root).search(query)

    assert tuple(product.name for product in result.products) == matched_names


@pytest.mark.parametrize(
    ("query", "matched_names"),
    (
        ("Do you have hiking socks?", ("merino hiking socks",)),
        ("Can you show me the rain jacket?", ("waterproof rain jacket",)),
        ("I am looking for running shoes", ("trail running shoes",)),
    ),
)
def test_fixture_catalog_search_recalls_partial_product_phrases(
    config_root: Path,
    query: str,
    matched_names: tuple[str, ...],
) -> None:
    result = _catalog(config_root).search(query)

    assert tuple(product.name for product in result.products) == matched_names


def test_fixture_catalog_ranks_before_applying_the_result_limit() -> None:
    products = (
        *(
            CatalogProduct(sku=f"SKU-{index}", name="trail running", price_usd=1.0)
            for index in range(CATALOG_RESULT_MAX_ITEMS)
        ),
        CatalogProduct(sku="SKU-BEST", name="trail running shoes", price_usd=80.0),
    )
    catalog = FixtureCatalog("ranked_store", CatalogFixture(products=products))

    result = catalog.search("trail running shoes")

    assert result.products[0].sku == "SKU-BEST"


def test_fixture_catalog_resolves_only_requested_live_skus(config_root: Path) -> None:
    catalog = _catalog(config_root)

    products = catalog.resolve_products(("SKU-GRN-15", "UNKNOWN", "SKU-RED-42"))

    assert tuple(product.sku if product is not None else None for product in products) == (
        "SKU-GRN-15",
        None,
        "SKU-RED-42",
    )


def test_number_candidates_is_shared_by_catalog_and_live_cart_items() -> None:
    lines = (
        CartLine(sku="SKU-1", name="trail shoes", price_usd=79.0, quantity=2),
        CartLine(sku="SKU-2", name="rain jacket", price_usd=129.0, quantity=1),
    )

    assert number_candidates(lines) == [
        Candidate(key="1", sku="SKU-1", name="trail shoes", price_usd=79.0),
        Candidate(key="2", sku="SKU-2", name="rain jacket", price_usd=129.0),
    ]


def test_catalog_fixture_requires_unique_skus_but_allows_duplicate_names() -> None:
    products = [
        {"sku": "SKU-1", "name": "trail runner", "price_usd": 80.0},
        {"sku": "SKU-2", "name": "trail runner", "price_usd": 95.0},
    ]

    fixture = CatalogFixture.model_validate({"products": products})
    assert [product.sku for product in fixture.products] == ["SKU-1", "SKU-2"]

    products[1]["sku"] = "SKU-1"
    with pytest.raises(ValidationError, match="product SKUs must be unique"):
        CatalogFixture.model_validate({"products": products})


def test_catalog_result_accepts_an_empty_product_set() -> None:
    assert CatalogProductSet(products=()).products == ()


def test_fixture_catalog_handles_an_empty_catalog() -> None:
    catalog = FixtureCatalog("empty_store", CatalogFixture(products=()))

    assert catalog.browse().products == ()
    assert catalog.search("anything").products == ()


def test_catalog_result_rejects_duplicate_skus() -> None:
    product = CatalogProduct(sku="SKU-1", name="trail runner", price_usd=80.0)

    with pytest.raises(ValidationError, match="duplicate product SKUs"):
        CatalogProductSet(products=(product, product))


def test_catalog_result_rejects_more_than_the_platform_bound() -> None:
    products = tuple(
        CatalogProduct(sku=f"SKU-{index}", name=f"item {index}", price_usd=1.0)
        for index in range(CATALOG_RESULT_MAX_ITEMS + 1)
    )

    with pytest.raises(ValidationError, match="at most"):
        CatalogProductSet(products=products)


def test_fixture_catalog_requires_a_tenant_identity(config_root: Path) -> None:
    fixture = load_orders_fixture(config_root, "acme_store")

    with pytest.raises(ValueError, match="requires a tenant id"):
        FixtureCatalog(" ", fixture)
