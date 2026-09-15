"""Versioned deployment authority for active and retiring durable tenants."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from agnostic_market.config.loader import ConfigError, load_yaml_layer
from agnostic_market.dtos.platform import ConfigIdentifier
from agnostic_market.durability.evidence import write_immutable_evidence

_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


class TenantLifecycleState(StrEnum):
    ACTIVE = "active"
    RETIRING = "retiring"


class TenantLifecycleEntry(BaseModel):
    model_config = _STRICT

    tenant_id: ConfigIdentifier
    state: TenantLifecycleState


class TenantLifecycleInventory(BaseModel):
    """Canonical tenant sweep authority chained to its predecessor revision."""

    model_config = _STRICT

    schema_version: Literal[1]
    revision: int = Field(ge=1)
    previous_revision: int | None
    previous_fingerprint: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    entries: tuple[TenantLifecycleEntry, ...]

    @model_validator(mode="after")
    def validate_canonical_inventory(self) -> Self:
        tenant_ids = tuple(entry.tenant_id for entry in self.entries)
        if len(set(tenant_ids)) != len(tenant_ids):
            raise ValueError("tenant lifecycle inventory ids must be unique")
        if tenant_ids != tuple(sorted(tenant_ids)):
            raise ValueError("tenant lifecycle inventory entries must be sorted by tenant id")
        if self.revision == 1:
            if self.previous_revision is not None or self.previous_fingerprint is not None:
                raise ValueError("initial tenant lifecycle inventory cannot name a predecessor")
        elif self.previous_revision != self.revision - 1 or self.previous_fingerprint is None:
            raise ValueError(
                "tenant lifecycle inventory must bind its immediately preceding revision"
            )
        return self

    @property
    def active_tenant_ids(self) -> tuple[str, ...]:
        return tuple(
            entry.tenant_id for entry in self.entries if entry.state is TenantLifecycleState.ACTIVE
        )

    @property
    def retiring_tenant_ids(self) -> tuple[str, ...]:
        return tuple(
            entry.tenant_id
            for entry in self.entries
            if entry.state is TenantLifecycleState.RETIRING
        )

    @property
    def sweep_tenant_ids(self) -> tuple[str, ...]:
        return tuple(entry.tenant_id for entry in self.entries)


class TenantDrainResult(BaseModel):
    """Zero-state observation authorizing removal of one retiring tenant."""

    model_config = _STRICT

    schema_version: Literal[1]
    tenant_id: ConfigIdentifier
    deployment_id: ConfigIdentifier
    runtime_contract_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    inventory_revision: int = Field(ge=1)
    inventory_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    observed_at: datetime
    platform_sessions: Literal[0]
    platform_session_operations: Literal[0]
    platform_checkpoint_generations: Literal[0]
    platform_checkpoint_write_manifests: Literal[0]
    raw_checkpoint_cleanup_verified: Literal[True]

    @model_validator(mode="after")
    def validate_timestamp(self) -> Self:
        if self.observed_at.tzinfo is None:
            raise ValueError("tenant drain timestamp must be timezone-aware")
        return self


class TenantDurableRowCounts(BaseModel):
    """Tenant-scoped durable rows observed through the application RLS role."""

    model_config = _STRICT

    tenant_id: ConfigIdentifier
    observed_at: datetime
    platform_sessions: int = Field(ge=0)
    platform_session_operations: int = Field(ge=0)
    platform_checkpoint_generations: int = Field(ge=0)
    platform_checkpoint_write_manifests: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_timestamp(self) -> Self:
        if self.observed_at.tzinfo is None:
            raise ValueError("tenant durable-state timestamp must be timezone-aware")
        return self


class TenantLifecycleInventoryError(RuntimeError):
    """The deployment inventory or its transition is not authoritative."""


def tenant_lifecycle_inventory_fingerprint(inventory: TenantLifecycleInventory) -> str:
    validated = TenantLifecycleInventory.model_validate_json(inventory.model_dump_json())
    canonical = json.dumps(
        validated.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_tenant_lifecycle_inventory(path: Path) -> TenantLifecycleInventory:
    try:
        payload = json.dumps(load_yaml_layer(path))
        return TenantLifecycleInventory.model_validate_json(payload)
    except (ConfigError, TypeError, ValueError) as exc:
        raise TenantLifecycleInventoryError(
            f"tenant lifecycle inventory cannot be loaded: {path}"
        ) from exc


def load_tenant_drain_result(path: Path) -> TenantDrainResult:
    try:
        return TenantDrainResult.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        raise TenantLifecycleInventoryError(
            f"tenant drain result cannot be loaded: {path}"
        ) from exc


def write_tenant_drain_result(path: Path, result: TenantDrainResult) -> None:
    try:
        validated = TenantDrainResult.model_validate_json(result.model_dump_json())
    except ValidationError as exc:
        raise TenantLifecycleInventoryError("tenant drain result is invalid") from exc
    write_immutable_evidence(path, validated)


def build_tenant_drain_result(
    inventory: TenantLifecycleInventory,
    counts: TenantDurableRowCounts,
    *,
    deployment_id: str,
    runtime_contract_fingerprint: str,
) -> TenantDrainResult:
    validated_inventory = TenantLifecycleInventory.model_validate_json(inventory.model_dump_json())
    validated_counts = TenantDurableRowCounts.model_validate_json(counts.model_dump_json())
    if validated_counts.tenant_id not in validated_inventory.retiring_tenant_ids:
        raise TenantLifecycleInventoryError(
            "tenant drain evidence can only be produced for a retiring inventory entry"
        )
    observed_counts = (
        validated_counts.platform_sessions,
        validated_counts.platform_session_operations,
        validated_counts.platform_checkpoint_generations,
        validated_counts.platform_checkpoint_write_manifests,
    )
    if any(observed_counts):
        raise TenantLifecycleInventoryError(
            "retiring tenant still owns durable rows: "
            f"sessions={observed_counts[0]}, operations={observed_counts[1]}, "
            f"generations={observed_counts[2]}, manifests={observed_counts[3]}"
        )
    return TenantDrainResult(
        schema_version=1,
        tenant_id=validated_counts.tenant_id,
        deployment_id=deployment_id,
        runtime_contract_fingerprint=runtime_contract_fingerprint,
        inventory_revision=validated_inventory.revision,
        inventory_fingerprint=tenant_lifecycle_inventory_fingerprint(validated_inventory),
        observed_at=validated_counts.observed_at,
        platform_sessions=0,
        platform_session_operations=0,
        platform_checkpoint_generations=0,
        platform_checkpoint_write_manifests=0,
        raw_checkpoint_cleanup_verified=True,
    )


def validate_tenant_lifecycle_inventory(
    inventory: TenantLifecycleInventory,
    merchant_ids: frozenset[str],
) -> None:
    try:
        validated = TenantLifecycleInventory.model_validate_json(inventory.model_dump_json())
    except ValidationError as exc:
        raise TenantLifecycleInventoryError("tenant lifecycle inventory is invalid") from exc
    active = frozenset(validated.active_tenant_ids)
    retiring = frozenset(validated.retiring_tenant_ids)
    retiring_merchants = retiring & merchant_ids
    if retiring_merchants:
        raise TenantLifecycleInventoryError(
            "retiring tenant remains present in the admitted merchant registry: "
            + ", ".join(sorted(retiring_merchants))
        )
    if active != merchant_ids:
        missing = sorted(merchant_ids - active)
        unexpected = sorted(active - merchant_ids)
        raise TenantLifecycleInventoryError(
            "tenant lifecycle active tenant set does not match the admitted merchant registry "
            f"(missing={missing}, unexpected={unexpected})"
        )


def validate_tenant_lifecycle_transition(
    previous: TenantLifecycleInventory,
    current: TenantLifecycleInventory,
    *,
    drain_results: tuple[TenantDrainResult, ...] = (),
    expected_deployment_id: str | None = None,
    expected_runtime_contract_fingerprint: str | None = None,
) -> None:
    try:
        prior = TenantLifecycleInventory.model_validate_json(previous.model_dump_json())
        replacement = TenantLifecycleInventory.model_validate_json(current.model_dump_json())
        drains = tuple(
            TenantDrainResult.model_validate_json(result.model_dump_json())
            for result in drain_results
        )
    except ValidationError as exc:
        raise TenantLifecycleInventoryError("tenant lifecycle transition input is invalid") from exc

    expected_fingerprint = tenant_lifecycle_inventory_fingerprint(prior)
    if (
        replacement.revision != prior.revision + 1
        or replacement.previous_revision != prior.revision
        or replacement.previous_fingerprint != expected_fingerprint
    ):
        raise TenantLifecycleInventoryError(
            "tenant lifecycle inventory does not bind the deployed predecessor"
        )

    prior_by_id = {entry.tenant_id: entry.state for entry in prior.entries}
    replacement_by_id = {entry.tenant_id: entry.state for entry in replacement.entries}
    for tenant_id, prior_state in prior_by_id.items():
        replacement_state = replacement_by_id.get(tenant_id)
        if prior_state is TenantLifecycleState.ACTIVE and replacement_state is None:
            raise TenantLifecycleInventoryError(
                f"active tenant cannot be removed directly: {tenant_id}"
            )
        if (
            prior_state is TenantLifecycleState.RETIRING
            and replacement_state is TenantLifecycleState.ACTIVE
        ):
            raise TenantLifecycleInventoryError(
                f"retiring tenant cannot return directly to active: {tenant_id}"
            )
    for tenant_id, replacement_state in replacement_by_id.items():
        if tenant_id not in prior_by_id and replacement_state is not TenantLifecycleState.ACTIVE:
            raise TenantLifecycleInventoryError(
                f"new tenant must enter the lifecycle as active: {tenant_id}"
            )

    removed_retiring = {
        tenant_id
        for tenant_id, state in prior_by_id.items()
        if state is TenantLifecycleState.RETIRING and tenant_id not in replacement_by_id
    }
    drain_by_id = {result.tenant_id: result for result in drains}
    if len(drain_by_id) != len(drains):
        raise TenantLifecycleInventoryError("tenant drain results must have unique tenant ids")
    if set(drain_by_id) != removed_retiring:
        raise TenantLifecycleInventoryError(
            "retiring tenant removal requires one exact drain result per removed tenant"
        )
    if removed_retiring and (
        expected_deployment_id is None or expected_runtime_contract_fingerprint is None
    ):
        raise TenantLifecycleInventoryError(
            "retiring tenant removal requires the expected deployment and runtime identity"
        )
    for tenant_id, drain in drain_by_id.items():
        if (
            drain.deployment_id != expected_deployment_id
            or drain.runtime_contract_fingerprint != expected_runtime_contract_fingerprint
            or drain.inventory_revision != prior.revision
            or drain.inventory_fingerprint != expected_fingerprint
        ):
            raise TenantLifecycleInventoryError(
                f"tenant drain result does not bind the retiring inventory: {tenant_id}"
            )
