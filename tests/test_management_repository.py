"""Transactional repository contracts for merchant configuration management."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from threading import Barrier

import pytest

import agnostic_market.management.contracts as management_contracts
from agnostic_market.commerce.catalog import load_catalog_fixture
from agnostic_market.commerce.identity import load_customers_fixture
from agnostic_market.commerce.orders import load_orders_fixture
from agnostic_market.commerce.payment_instruments import load_payment_instruments_fixture
from agnostic_market.commerce.profile import load_profile_fixture
from agnostic_market.commerce.verification import load_verification_fixture
from agnostic_market.config.loader import config_version, load_yaml_layer
from agnostic_market.config.registry import resolve_merchant_override
from agnostic_market.management.contracts import (
    MerchantDraft,
    MerchantFixtureBundle,
    MerchantPublicationRequest,
    MerchantRetirementRequest,
    MerchantRollbackRequest,
    PublishedMerchantVersion,
    ResolvedMerchantPreview,
    management_contract_schema_fingerprint,
    merchant_draft_fingerprint,
    merchant_fixture_bundle_fingerprint,
    merchant_preview_fingerprint,
)
from agnostic_market.management.datasets import load_merchant_scenario_dataset
from agnostic_market.management.repository import (
    ManagementRepositoryConflictError,
    ManagementRepositoryDataError,
    ManagementRepositoryNotFoundError,
    ManagementRepositoryReplayConflictError,
    SqliteMerchantConfigurationRepository,
)

_NOW = datetime(2026, 9, 19, 13, tzinfo=UTC)


def _bundle(config_root: Path) -> MerchantFixtureBundle:
    return MerchantFixtureBundle(
        catalog=load_catalog_fixture(config_root, "acme_store"),
        orders=load_orders_fixture(config_root, "acme_store"),
        customers=load_customers_fixture(config_root, "acme_store"),
        payment_instruments=load_payment_instruments_fixture(config_root, "acme_store"),
        profiles=load_profile_fixture(config_root, "acme_store"),
        verification=load_verification_fixture(config_root, "acme_store"),
    )


def _draft(
    config_root: Path,
    *,
    revision: int = 1,
    request_id: str = "draft-request-1",
    marker: str = "first",
) -> MerchantDraft:
    merchant_override = load_yaml_layer(config_root / "merchants" / "acme_store.yaml")
    merchant_override["display_name"] = f"Acme Store {marker}"
    return MerchantDraft(
        tenant_id="acme_store",
        draft_id="draft-1",
        revision=revision,
        actor_id="operator-1",
        request_id=request_id,
        created_at=_NOW,
        updated_at=_NOW,
        merchant_override=merchant_override,
        fixtures=_bundle(config_root),
    )


def _preview(config_root: Path, *, draft_revision: int = 1) -> ResolvedMerchantPreview:
    source_draft = _draft(
        config_root,
        revision=draft_revision,
        request_id=f"draft-request-{draft_revision}",
        marker="first" if draft_revision == 1 else "second",
    )
    resolved = resolve_merchant_override(
        config_root,
        source_draft.merchant_override,
        source="repository test draft",
    )
    fixtures = source_draft.fixtures
    source_draft_fingerprint = merchant_draft_fingerprint(source_draft)
    fixture_fingerprint = merchant_fixture_bundle_fingerprint(fixtures)
    schema_fingerprint = management_contract_schema_fingerprint()
    return ResolvedMerchantPreview(
        tenant_id="acme_store",
        draft_id="draft-1",
        draft_revision=draft_revision,
        resolved_at=_NOW,
        config=resolved.config,
        fixtures=fixtures,
        source_draft_fingerprint=source_draft_fingerprint,
        config_fingerprint=resolved.config_version,
        fixture_fingerprint=fixture_fingerprint,
        schema_fingerprint=schema_fingerprint,
        preview_fingerprint=merchant_preview_fingerprint(
            tenant_id="acme_store",
            draft_id="draft-1",
            draft_revision=draft_revision,
            source_draft_fingerprint=source_draft_fingerprint,
            config_fingerprint=resolved.config_version,
            fixture_fingerprint=fixture_fingerprint,
            schema_fingerprint=schema_fingerprint,
        ),
    )


def _repository(
    tmp_path: Path,
    config_root: Path,
    *,
    database_name: str = "management.sqlite3",
    version_ids: tuple[str, ...] = ("version-1", "version-2", "version-3"),
) -> SqliteMerchantConfigurationRepository:
    ids = iter(version_ids)
    return SqliteMerchantConfigurationRepository(
        tmp_path / database_name,
        active_config_root=config_root,
        clock=lambda: _NOW,
        version_id_factory=ids.__next__,
    )


def _publication(
    config_root: Path,
    *,
    request_id: str = "publish-request-1",
    expected_active_version_id: str | None = None,
    draft_revision: int = 1,
    actor_id: str = "operator-1",
) -> MerchantPublicationRequest:
    return MerchantPublicationRequest(
        tenant_id="acme_store",
        expected_active_version_id=expected_active_version_id,
        actor_id=actor_id,
        request_id=request_id,
        preview=_preview(config_root, draft_revision=draft_revision),
    )


def test_repository_refuses_storage_inside_active_configuration(config_root: Path) -> None:
    with pytest.raises(ValueError, match="outside the active config tree"):
        SqliteMerchantConfigurationRepository(
            config_root / "management.sqlite3",
            active_config_root=config_root,
        )


def test_repository_rounds_positive_busy_timeout_up_to_one_millisecond(
    tmp_path: Path, config_root: Path
) -> None:
    repository = SqliteMerchantConfigurationRepository(
        tmp_path / "submillisecond-timeout.sqlite3",
        active_config_root=config_root,
        busy_timeout_seconds=0.0004,
    )

    assert repository._busy_timeout_ms == 1
    with repository._connect() as connection:
        assert connection.execute("PRAGMA busy_timeout").fetchone()[0] == 1


def test_draft_write_is_compare_and_swap_and_exact_retry_is_idempotent(
    tmp_path: Path, config_root: Path
) -> None:
    repository = _repository(tmp_path, config_root)
    first = _draft(config_root)

    assert repository.save_draft(first, expected_revision=0) == first
    assert repository.save_draft(first, expected_revision=0) == first
    second = _draft(
        config_root,
        revision=2,
        request_id="draft-request-2",
        marker="second",
    )
    assert repository.save_draft(second, expected_revision=1) == second
    assert repository.get_draft("acme_store", "draft-1") == second

    with pytest.raises(ManagementRepositoryConflictError, match="original expected revision"):
        repository.save_draft(second, expected_revision=0)

    with pytest.raises(ManagementRepositoryConflictError, match="expected revision"):
        repository.save_draft(
            _draft(
                config_root,
                revision=2,
                request_id="draft-request-stale",
                marker="stale",
            ),
            expected_revision=1,
        )


def test_repository_rejects_removing_dataset_provenance_inside_the_transaction(
    tmp_path: Path, config_root: Path
) -> None:
    repository = _repository(tmp_path, config_root)
    dataset = load_merchant_scenario_dataset(config_root / "datasets" / "fashion-service-v1.yaml")
    first = _draft(config_root).model_copy(
        update={
            "fixtures": dataset.fixtures,
            "dataset_manifest": dataset.manifest,
        }
    )
    repository.save_draft(first, expected_revision=0)
    without_provenance = first.model_copy(
        update={
            "revision": 2,
            "request_id": "draft-request-2",
            "dataset_manifest": None,
        }
    )

    with pytest.raises(ManagementRepositoryConflictError, match="provenance cannot be removed"):
        repository.save_draft(without_provenance, expected_revision=1)

    assert repository.get_draft("acme_store", "draft-1") == first


def test_concurrent_draft_writers_cannot_silently_overwrite_each_other(
    tmp_path: Path, config_root: Path
) -> None:
    database_name = "draft-race.sqlite3"
    repository = _repository(tmp_path, config_root, database_name=database_name)
    repository.save_draft(_draft(config_root), expected_revision=0)
    barrier = Barrier(2)

    def write(marker: str) -> str:
        contender = _repository(tmp_path, config_root, database_name=database_name)
        barrier.wait()
        try:
            contender.save_draft(
                _draft(
                    config_root,
                    revision=2,
                    request_id=f"draft-request-{marker}",
                    marker=marker,
                ),
                expected_revision=1,
            )
        except ManagementRepositoryConflictError:
            return "conflict"
        return "committed"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = tuple(executor.map(write, ("left", "right")))

    assert sorted(outcomes) == ["committed", "conflict"]
    stored = repository.get_draft("acme_store", "draft-1")
    assert stored is not None
    assert stored.merchant_override["display_name"] in {
        "Acme Store left",
        "Acme Store right",
    }


def test_publication_atomically_creates_an_immutable_version_and_active_pointer(
    tmp_path: Path, config_root: Path
) -> None:
    repository = _repository(tmp_path, config_root)
    repository.save_draft(_draft(config_root), expected_revision=0)

    receipt = repository.publish(_publication(config_root))
    active = repository.get_active_version("acme_store")

    assert receipt.replayed is False
    assert active is not None
    assert active.version_id == receipt.version_id == "version-1"
    assert active.version_number == receipt.version_number == 1
    assert active.previous_version_id is None
    assert repository.get_version("acme_store", "version-1") == active


def test_published_version_read_requires_the_current_schema_fingerprint(
    tmp_path: Path,
    config_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _repository(tmp_path, config_root)
    repository.save_draft(_draft(config_root), expected_revision=0)
    receipt = repository.publish(_publication(config_root))
    published = repository.get_version("acme_store", receipt.version_id)
    assert published is not None

    monkeypatch.setattr(
        management_contracts,
        "management_contract_schema_fingerprint",
        lambda: "0" * 64,
    )

    with pytest.raises(ManagementRepositoryDataError, match="stored active merchant version"):
        repository.get_active_version("acme_store")


def test_published_version_read_rejects_raw_payload_tampering(
    tmp_path: Path,
    config_root: Path,
) -> None:
    repository = _repository(tmp_path, config_root)
    repository.save_draft(_draft(config_root), expected_revision=0)
    receipt = repository.publish(_publication(config_root))

    with sqlite3.connect(repository.database_path) as connection:
        row = connection.execute(
            "SELECT payload FROM management_versions WHERE version_id = ?",
            (receipt.version_id,),
        ).fetchone()
        assert row is not None
        payload = json.loads(row[0])
        payload["config"]["display_name"] = "Tampered Store"
        connection.execute(
            "UPDATE management_versions SET payload = ? WHERE version_id = ?",
            (json.dumps(payload), receipt.version_id),
        )

    with pytest.raises(ManagementRepositoryDataError, match="stored active merchant version"):
        repository.get_active_version("acme_store")


def test_publication_rejects_preview_fixtures_that_do_not_match_its_draft(
    tmp_path: Path, config_root: Path
) -> None:
    repository = _repository(tmp_path, config_root)
    repository.save_draft(_draft(config_root), expected_revision=0)
    preview_payload = _preview(config_root).model_dump()
    preview_payload["fixtures"]["catalog"]["products"][0]["price_usd"] = "90.00"
    changed_fixtures = MerchantFixtureBundle.model_validate(preview_payload["fixtures"])
    preview_payload["fixture_fingerprint"] = merchant_fixture_bundle_fingerprint(changed_fixtures)
    preview_payload["preview_fingerprint"] = merchant_preview_fingerprint(
        tenant_id=preview_payload["tenant_id"],
        draft_id=preview_payload["draft_id"],
        draft_revision=preview_payload["draft_revision"],
        source_draft_fingerprint=preview_payload["source_draft_fingerprint"],
        config_fingerprint=preview_payload["config_fingerprint"],
        fixture_fingerprint=preview_payload["fixture_fingerprint"],
        schema_fingerprint=preview_payload["schema_fingerprint"],
    )
    changed_preview = ResolvedMerchantPreview.model_validate(preview_payload)

    with pytest.raises(ManagementRepositoryConflictError, match="fixtures do not match"):
        repository.publish(
            MerchantPublicationRequest(
                tenant_id="acme_store",
                actor_id="operator-1",
                request_id="publish-request-1",
                preview=changed_preview,
            )
        )

    assert repository.get_active_version("acme_store") is None


def test_publication_rejects_preview_config_that_did_not_resolve_from_its_draft(
    tmp_path: Path, config_root: Path
) -> None:
    repository = _repository(tmp_path, config_root)
    repository.save_draft(_draft(config_root), expected_revision=0)
    preview_payload = _preview(config_root).model_dump()
    preview_payload["config"]["display_name"] = "Forged but valid"
    preview_payload["config_fingerprint"] = config_version(preview_payload["config"])
    preview_payload["preview_fingerprint"] = merchant_preview_fingerprint(
        tenant_id=preview_payload["tenant_id"],
        draft_id=preview_payload["draft_id"],
        draft_revision=preview_payload["draft_revision"],
        source_draft_fingerprint=preview_payload["source_draft_fingerprint"],
        config_fingerprint=preview_payload["config_fingerprint"],
        fixture_fingerprint=preview_payload["fixture_fingerprint"],
        schema_fingerprint=preview_payload["schema_fingerprint"],
    )
    changed_preview = ResolvedMerchantPreview.model_validate(preview_payload)

    with pytest.raises(ManagementRepositoryConflictError, match="config does not match"):
        repository.publish(
            MerchantPublicationRequest(
                tenant_id="acme_store",
                actor_id="operator-1",
                request_id="publish-request-1",
                preview=changed_preview,
            )
        )

    assert repository.get_active_version("acme_store") is None


def test_publication_replay_returns_the_original_receipt_and_rejects_divergence(
    tmp_path: Path, config_root: Path
) -> None:
    repository = _repository(tmp_path, config_root)
    repository.save_draft(_draft(config_root), expected_revision=0)
    request = _publication(config_root)

    committed = repository.publish(request)
    replayed = repository.publish(request)

    assert replayed == committed.model_copy(update={"replayed": True})
    with pytest.raises(ManagementRepositoryReplayConflictError, match="different parameters"):
        repository.publish(request.model_copy(update={"actor_id": "operator-2"}))


def test_publication_replay_rejects_a_receipt_that_does_not_match_its_ledger_key(
    tmp_path: Path, config_root: Path
) -> None:
    repository = _repository(tmp_path, config_root)
    repository.save_draft(_draft(config_root), expected_revision=0)
    request = _publication(config_root)
    repository.publish(request)

    with sqlite3.connect(repository.database_path) as connection:
        row = connection.execute(
            """
            SELECT receipt_payload FROM management_publication_requests
            WHERE tenant_id = ? AND request_id = ?
            """,
            (request.tenant_id, request.request_id),
        ).fetchone()
        assert row is not None
        payload = row[0].replace(request.request_id, "different-request", 1)
        connection.execute(
            """
            UPDATE management_publication_requests SET receipt_payload = ?
            WHERE tenant_id = ? AND request_id = ?
            """,
            (payload, request.tenant_id, request.request_id),
        )

    with pytest.raises(ManagementRepositoryDataError, match="repository key"):
        repository.publish(request)


@pytest.mark.parametrize("interruption", (RuntimeError("failed"), asyncio.CancelledError()))
def test_interrupted_publication_leaves_no_version_receipt_or_active_pointer(
    tmp_path: Path,
    config_root: Path,
    interruption: BaseException,
) -> None:
    class InterruptingRepository(SqliteMerchantConfigurationRepository):
        def _write_active_pointer(
            self,
            connection: sqlite3.Connection,
            version: PublishedMerchantVersion,
        ) -> None:
            del connection, version
            raise interruption

    database_path = tmp_path / "interrupted.sqlite3"
    ids = iter(("interrupted-version",))
    repository = InterruptingRepository(
        database_path,
        active_config_root=config_root,
        clock=lambda: _NOW,
        version_id_factory=ids.__next__,
    )
    repository.save_draft(_draft(config_root), expected_revision=0)

    expected_message = "failed" if isinstance(interruption, RuntimeError) else None
    with pytest.raises(type(interruption), match=expected_message):
        repository.publish(_publication(config_root))

    observer = SqliteMerchantConfigurationRepository(
        database_path,
        active_config_root=config_root,
        clock=lambda: _NOW,
        version_id_factory=lambda: "replacement-version",
    )
    assert observer.get_active_version("acme_store") is None
    assert observer.get_version("acme_store", "interrupted-version") is None
    assert tuple(record.event for record in observer.list_audit_records("acme_store")) == (
        "draft_created",
    )
    receipt = observer.publish(_publication(config_root))
    assert receipt.version_id == "replacement-version"


def test_concurrent_publications_admit_only_one_expected_active_version(
    tmp_path: Path, config_root: Path
) -> None:
    database_name = "publication-race.sqlite3"
    repository = _repository(tmp_path, config_root, database_name=database_name)
    repository.save_draft(_draft(config_root), expected_revision=0)
    barrier = Barrier(2)

    def publish(contender_id: str) -> str:
        contender = _repository(
            tmp_path,
            config_root,
            database_name=database_name,
            version_ids=(f"version-{contender_id}",),
        )
        barrier.wait()
        try:
            contender.publish(
                _publication(config_root, request_id=f"publish-request-{contender_id}")
            )
        except ManagementRepositoryConflictError:
            return "conflict"
        return "committed"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = tuple(executor.map(publish, ("left", "right")))

    assert sorted(outcomes) == ["committed", "conflict"]
    active = repository.get_active_version("acme_store")
    assert active is not None
    assert active.version_id in {"version-left", "version-right"}


def test_rollback_publishes_a_new_version_without_mutating_history(
    tmp_path: Path, config_root: Path
) -> None:
    repository = _repository(tmp_path, config_root)
    repository.save_draft(_draft(config_root), expected_revision=0)
    first_receipt = repository.publish(_publication(config_root))
    first_version = repository.get_active_version("acme_store")
    assert first_version is not None

    repository.save_draft(
        _draft(
            config_root,
            revision=2,
            request_id="draft-request-2",
            marker="second",
        ),
        expected_revision=1,
    )
    second_receipt = repository.publish(
        _publication(
            config_root,
            request_id="publish-request-2",
            expected_active_version_id=first_receipt.version_id,
            draft_revision=2,
        )
    )
    request = MerchantRollbackRequest(
        tenant_id="acme_store",
        expected_active_version_id=second_receipt.version_id,
        source_version_id=first_receipt.version_id,
        actor_id="operator-2",
        request_id="rollback-request-1",
    )
    rollback = repository.rollback(request)
    replayed = repository.rollback(request)
    active = repository.get_active_version("acme_store")

    assert rollback.kind == "rollback"
    assert rollback.source_version_id == first_version.version_id
    assert rollback.version_number == 3
    assert active is not None
    assert active.version_id == rollback.version_id == "version-3"
    assert active.previous_version_id == second_receipt.version_id
    assert active.config_fingerprint == first_version.config_fingerprint
    assert active.fixture_fingerprint == first_version.fixture_fingerprint
    assert repository.get_version("acme_store", first_version.version_id) == first_version
    assert replayed == rollback.model_copy(update={"replayed": True})


def test_rollback_keeps_version_numbers_monotonic_if_active_pointer_lags_history(
    tmp_path: Path, config_root: Path
) -> None:
    repository = _repository(tmp_path, config_root)
    repository.save_draft(_draft(config_root), expected_revision=0)
    first = repository.publish(_publication(config_root))
    repository.save_draft(
        _draft(
            config_root,
            revision=2,
            request_id="draft-request-2",
            marker="second",
        ),
        expected_revision=1,
    )
    repository.publish(
        _publication(
            config_root,
            request_id="publish-request-2",
            expected_active_version_id=first.version_id,
            draft_revision=2,
        )
    )
    with repository._connect() as connection:
        connection.execute(
            """
            UPDATE management_active_versions
            SET version_id = ?, version_number = ?
            WHERE tenant_id = ?
            """,
            (first.version_id, first.version_number, "acme_store"),
        )

    rollback = repository.rollback(
        MerchantRollbackRequest(
            tenant_id="acme_store",
            expected_active_version_id=first.version_id,
            source_version_id=first.version_id,
            actor_id="operator-2",
            request_id="rollback-after-pointer-repair",
        )
    )

    assert rollback.version_number == 3
    active = repository.get_active_version("acme_store")
    assert active is not None
    assert active.version_number == 3
    assert active.previous_version_id == first.version_id


def test_retirement_removes_only_the_active_pointer_and_republication_stays_monotonic(
    tmp_path: Path, config_root: Path
) -> None:
    repository = _repository(tmp_path, config_root)
    repository.save_draft(_draft(config_root), expected_revision=0)
    first = repository.publish(_publication(config_root))
    request = MerchantRetirementRequest(
        tenant_id="acme_store",
        expected_active_version_id=first.version_id,
        actor_id="operator-1",
        request_id="retire-request-1",
    )

    retired = repository.retire(request)
    replayed = repository.retire(request)

    assert retired.retired_version_id == first.version_id
    assert retired.replayed is False
    assert replayed == retired.model_copy(update={"replayed": True})
    assert repository.get_active_version("acme_store") is None
    assert repository.get_version("acme_store", first.version_id) is not None
    with pytest.raises(ManagementRepositoryReplayConflictError, match="different parameters"):
        repository.retire(request.model_copy(update={"actor_id": "operator-2"}))

    second = repository.publish(
        _publication(
            config_root,
            request_id="publish-request-2",
            expected_active_version_id=None,
        )
    )
    active = repository.get_active_version("acme_store")
    assert second.version_number == 2
    assert active is not None
    assert active.previous_version_id == first.version_id


def test_retire_and_rollback_without_an_active_version_report_not_found(
    tmp_path: Path, config_root: Path
) -> None:
    repository = _repository(tmp_path, config_root)

    with pytest.raises(ManagementRepositoryNotFoundError, match="requires an active"):
        repository.retire(
            MerchantRetirementRequest(
                tenant_id="acme_store",
                expected_active_version_id="version-missing",
                actor_id="operator-1",
                request_id="retire-request-1",
            )
        )
    with pytest.raises(ManagementRepositoryNotFoundError, match="requires an active"):
        repository.rollback(
            MerchantRollbackRequest(
                tenant_id="acme_store",
                expected_active_version_id="version-missing",
                source_version_id="version-source",
                actor_id="operator-1",
                request_id="rollback-request-1",
            )
        )


def test_previous_repository_schema_is_rejected_without_mutation(
    tmp_path: Path,
    config_root: Path,
) -> None:
    database_path = tmp_path / "unsupported-management.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE management_repository_schema (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                schema_version INTEGER NOT NULL
            )
            """
        )
        connection.execute(
            "INSERT INTO management_repository_schema (singleton, schema_version) VALUES (1, 5)"
        )
    with sqlite3.connect(database_path) as connection:
        before = connection.execute(
            """
            SELECT type, name, sql FROM sqlite_master
            WHERE name NOT LIKE 'sqlite_%'
            ORDER BY type, name
            """
        ).fetchall()

    with pytest.raises(ManagementRepositoryDataError, match="schema version is unsupported"):
        SqliteMerchantConfigurationRepository(
            database_path,
            active_config_root=config_root,
        )

    with sqlite3.connect(database_path) as connection:
        after = connection.execute(
            """
            SELECT type, name, sql FROM sqlite_master
            WHERE name NOT LIKE 'sqlite_%'
            ORDER BY type, name
            """
        ).fetchall()
    assert after == before
