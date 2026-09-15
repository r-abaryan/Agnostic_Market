"""Deployment-owned tenant lifecycle inventory contracts."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from agnostic_market.config.registry import ConfigRegistry
from agnostic_market.durability.tenant_lifecycle import (
    TenantDrainResult,
    TenantDurableRowCounts,
    TenantLifecycleEntry,
    TenantLifecycleInventory,
    TenantLifecycleInventoryError,
    TenantLifecycleState,
    build_tenant_drain_result,
    load_tenant_drain_result,
    load_tenant_lifecycle_inventory,
    tenant_lifecycle_inventory_fingerprint,
    validate_tenant_lifecycle_inventory,
    validate_tenant_lifecycle_transition,
    write_tenant_drain_result,
)


def _inventory(
    *,
    revision: int = 1,
    previous: TenantLifecycleInventory | None = None,
    entries: tuple[TenantLifecycleEntry, ...] | None = None,
) -> TenantLifecycleInventory:
    if entries is None:
        entries = (
            TenantLifecycleEntry(
                tenant_id="acme_store",
                state=TenantLifecycleState.ACTIVE,
            ),
            TenantLifecycleEntry(
                tenant_id="demo_shop",
                state=TenantLifecycleState.ACTIVE,
            ),
        )
    return TenantLifecycleInventory(
        schema_version=1,
        revision=revision,
        previous_revision=None if previous is None else previous.revision,
        previous_fingerprint=(
            None if previous is None else tenant_lifecycle_inventory_fingerprint(previous)
        ),
        entries=entries,
    )


def _drain(
    inventory: TenantLifecycleInventory,
    tenant_id: str = "acme_store",
) -> TenantDrainResult:
    return TenantDrainResult(
        schema_version=1,
        tenant_id=tenant_id,
        deployment_id="deployment-a",
        runtime_contract_fingerprint="b" * 64,
        inventory_revision=inventory.revision,
        inventory_fingerprint=tenant_lifecycle_inventory_fingerprint(inventory),
        observed_at=datetime.now(UTC),
        platform_sessions=0,
        platform_session_operations=0,
        platform_checkpoint_generations=0,
        platform_checkpoint_write_manifests=0,
        raw_checkpoint_cleanup_verified=True,
    )


def test_inventory_loads_strict_canonical_yaml_and_has_a_stable_fingerprint(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tenant-lifecycle.yaml"
    path.write_text(
        """schema_version: 1
revision: 1
previous_revision: null
previous_fingerprint: null
entries:
  - tenant_id: acme_store
    state: active
  - tenant_id: demo_shop
    state: retiring
""",
        encoding="utf-8",
    )

    loaded = load_tenant_lifecycle_inventory(path)

    assert loaded.revision == 1
    assert loaded.active_tenant_ids == ("acme_store",)
    assert loaded.retiring_tenant_ids == ("demo_shop",)
    assert tenant_lifecycle_inventory_fingerprint(loaded) == (
        tenant_lifecycle_inventory_fingerprint(loaded.model_copy())
    )


def test_example_inventory_exactly_matches_the_repository_merchants(
    config_root: Path,
    registry: ConfigRegistry,
) -> None:
    inventory = load_tenant_lifecycle_inventory(
        config_root / "platform" / "tenant_lifecycle.example.yaml"
    )

    validate_tenant_lifecycle_inventory(inventory, registry.merchant_ids)


@pytest.mark.parametrize(
    "update",
    (
        {"revision": 2},
        {"previous_revision": 1},
        {"previous_fingerprint": "a" * 64},
        {
            "entries": (
                TenantLifecycleEntry(tenant_id="demo_shop", state=TenantLifecycleState.ACTIVE),
                TenantLifecycleEntry(tenant_id="acme_store", state=TenantLifecycleState.ACTIVE),
            )
        },
        {
            "entries": (
                TenantLifecycleEntry(tenant_id="acme_store", state=TenantLifecycleState.ACTIVE),
                TenantLifecycleEntry(tenant_id="acme_store", state=TenantLifecycleState.RETIRING),
            )
        },
    ),
)
def test_inventory_rejects_unbound_or_noncanonical_content(update: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        TenantLifecycleInventory.model_validate(_inventory().model_dump() | update)


def test_inventory_active_set_must_exactly_match_loaded_merchants() -> None:
    inventory = _inventory(
        entries=(
            TenantLifecycleEntry(tenant_id="acme_store", state=TenantLifecycleState.ACTIVE),
            TenantLifecycleEntry(tenant_id="old_shop", state=TenantLifecycleState.RETIRING),
        )
    )

    validate_tenant_lifecycle_inventory(inventory, frozenset({"acme_store"}))

    with pytest.raises(TenantLifecycleInventoryError, match="active tenant set"):
        validate_tenant_lifecycle_inventory(
            inventory,
            frozenset({"acme_store", "demo_shop"}),
        )

    with pytest.raises(TenantLifecycleInventoryError, match="retiring tenant"):
        validate_tenant_lifecycle_inventory(
            inventory,
            frozenset({"acme_store", "old_shop"}),
        )


def test_transition_moves_an_active_tenant_to_retiring_without_dropping_authority() -> None:
    previous = _inventory()
    current = _inventory(
        revision=2,
        previous=previous,
        entries=(
            TenantLifecycleEntry(tenant_id="acme_store", state=TenantLifecycleState.RETIRING),
            TenantLifecycleEntry(tenant_id="demo_shop", state=TenantLifecycleState.ACTIVE),
        ),
    )

    validate_tenant_lifecycle_transition(previous, current)


def test_transition_rejects_direct_active_tenant_removal() -> None:
    previous = _inventory()
    current = _inventory(
        revision=2,
        previous=previous,
        entries=(TenantLifecycleEntry(tenant_id="acme_store", state=TenantLifecycleState.ACTIVE),),
    )

    with pytest.raises(TenantLifecycleInventoryError, match="cannot be removed directly"):
        validate_tenant_lifecycle_transition(previous, current)


def test_transition_rejects_retiring_tenant_reactivation() -> None:
    previous = _inventory(
        entries=(TenantLifecycleEntry(tenant_id="acme_store", state=TenantLifecycleState.RETIRING),)
    )
    current = _inventory(
        revision=2,
        previous=previous,
        entries=(TenantLifecycleEntry(tenant_id="acme_store", state=TenantLifecycleState.ACTIVE),),
    )

    with pytest.raises(TenantLifecycleInventoryError, match="cannot return directly to active"):
        validate_tenant_lifecycle_transition(previous, current)


def test_retiring_tenant_removal_requires_exact_zero_state_drain_evidence() -> None:
    previous = _inventory(
        entries=(
            TenantLifecycleEntry(tenant_id="acme_store", state=TenantLifecycleState.RETIRING),
            TenantLifecycleEntry(tenant_id="demo_shop", state=TenantLifecycleState.ACTIVE),
        )
    )
    current = _inventory(
        revision=2,
        previous=previous,
        entries=(TenantLifecycleEntry(tenant_id="demo_shop", state=TenantLifecycleState.ACTIVE),),
    )

    with pytest.raises(TenantLifecycleInventoryError, match="drain result"):
        validate_tenant_lifecycle_transition(previous, current)

    with pytest.raises(TenantLifecycleInventoryError, match="expected deployment"):
        validate_tenant_lifecycle_transition(
            previous,
            current,
            drain_results=(_drain(previous),),
        )

    validate_tenant_lifecycle_transition(
        previous,
        current,
        drain_results=(_drain(previous),),
        expected_deployment_id="deployment-a",
        expected_runtime_contract_fingerprint="b" * 64,
    )

    stale = _drain(previous).model_copy(update={"inventory_fingerprint": "a" * 64})
    with pytest.raises(TenantLifecycleInventoryError, match="does not bind"):
        validate_tenant_lifecycle_transition(
            previous,
            current,
            drain_results=(stale,),
            expected_deployment_id="deployment-a",
            expected_runtime_contract_fingerprint="b" * 64,
        )

    wrong_deployment = _drain(previous).model_copy(update={"deployment_id": "deployment-b"})
    with pytest.raises(TenantLifecycleInventoryError, match="does not bind"):
        validate_tenant_lifecycle_transition(
            previous,
            current,
            drain_results=(wrong_deployment,),
            expected_deployment_id="deployment-a",
            expected_runtime_contract_fingerprint="b" * 64,
        )


def test_final_retiring_tenant_can_be_removed_after_exact_drain() -> None:
    previous = _inventory(
        entries=(TenantLifecycleEntry(tenant_id="acme_store", state=TenantLifecycleState.RETIRING),)
    )
    current = _inventory(revision=2, previous=previous, entries=())

    validate_tenant_lifecycle_transition(
        previous,
        current,
        drain_results=(_drain(previous),),
        expected_deployment_id="deployment-a",
        expected_runtime_contract_fingerprint="b" * 64,
    )
    validate_tenant_lifecycle_inventory(current, frozenset())


def test_loader_wraps_missing_invalid_and_extra_fields(tmp_path: Path) -> None:
    with pytest.raises(TenantLifecycleInventoryError, match="cannot be loaded"):
        load_tenant_lifecycle_inventory(tmp_path / "missing.yaml")

    path = tmp_path / "tenant-lifecycle.yaml"
    path.write_text("schema_version: 1\nunknown: true\n", encoding="utf-8")
    with pytest.raises(TenantLifecycleInventoryError, match="cannot be loaded"):
        load_tenant_lifecycle_inventory(path)


def test_loader_wraps_yaml_values_that_cannot_be_serialized(tmp_path: Path) -> None:
    path = tmp_path / "tenant-lifecycle.yaml"
    path.write_text(
        "schema_version: 1\n"
        "revision: 2026-01-01\n"
        "previous_revision: null\n"
        "previous_fingerprint: null\n"
        "entries: []\n",
        encoding="utf-8",
    )

    with pytest.raises(TenantLifecycleInventoryError, match="cannot be loaded"):
        load_tenant_lifecycle_inventory(path)


def test_drain_result_is_built_only_from_observed_zero_state_for_a_retiring_tenant(
    tmp_path: Path,
) -> None:
    inventory = _inventory(
        entries=(TenantLifecycleEntry(tenant_id="acme_store", state=TenantLifecycleState.RETIRING),)
    )
    counts = TenantDurableRowCounts(
        tenant_id="acme_store",
        observed_at=datetime.now(UTC),
        platform_sessions=0,
        platform_session_operations=0,
        platform_checkpoint_generations=0,
        platform_checkpoint_write_manifests=0,
    )

    result = build_tenant_drain_result(
        inventory,
        counts,
        deployment_id="deployment-a",
        runtime_contract_fingerprint="b" * 64,
    )
    path = tmp_path / "drain.json"
    write_tenant_drain_result(path, result)

    assert load_tenant_drain_result(path) == result
    with pytest.raises(FileExistsError, match="already exists"):
        write_tenant_drain_result(path, result)

    with pytest.raises(TenantLifecycleInventoryError, match="still owns durable rows"):
        build_tenant_drain_result(
            inventory,
            counts.model_copy(update={"platform_sessions": 1}),
            deployment_id="deployment-a",
            runtime_contract_fingerprint="b" * 64,
        )


def test_drain_result_cannot_be_built_for_an_active_tenant() -> None:
    inventory = _inventory(
        entries=(TenantLifecycleEntry(tenant_id="acme_store", state=TenantLifecycleState.ACTIVE),)
    )
    counts = TenantDurableRowCounts(
        tenant_id="acme_store",
        observed_at=datetime.now(UTC),
        platform_sessions=0,
        platform_session_operations=0,
        platform_checkpoint_generations=0,
        platform_checkpoint_write_manifests=0,
    )

    with pytest.raises(TenantLifecycleInventoryError, match="only be produced for a retiring"):
        build_tenant_drain_result(
            inventory,
            counts,
            deployment_id="deployment-a",
            runtime_contract_fingerprint="b" * 64,
        )
