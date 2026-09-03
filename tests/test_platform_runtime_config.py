"""Deployment-owned durable runtime configuration contracts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agnostic_market.dtos.platform import (
    DurableSessionConfig,
    PlatformDatabaseConfig,
    PlatformRuntimeConfig,
    SessionEncryptionConfig,
)


def _valid_config() -> dict[str, object]:
    return {
        "schema_version": 1,
        "database": {
            "application_dsn_ref": {
                "provider": "env",
                "locator": "PLATFORM_POSTGRES_DSN",
            },
            "minimum_pool_size": 2,
            "maximum_pool_size": 12,
            "connection_timeout_seconds": 3.0,
            "pool_acquisition_timeout_seconds": 0.5,
            "statement_timeout_seconds": 1.0,
            "transaction_timeout_seconds": 2.0,
            "operation_timeout_seconds": 3.0,
            "expected_schema_version": 1,
        },
        "sessions": {
            "lease_duration_seconds": 30.0,
            "lease_renewal_interval_seconds": 10.0,
            "session_retention_seconds": 86_400,
            "closed_tombstone_retention_seconds": 604_800,
        },
        "encryption": {
            "envelope_format": "aes_256_gcm_v1",
            "key_ref": {
                "provider": "env",
                "locator": "PLATFORM_SESSION_KEY",
            },
            "key_version": "key-2026-09",
        },
    }


def test_platform_runtime_config_is_strict_and_uses_structured_secret_references() -> None:
    config = PlatformRuntimeConfig.model_validate(_valid_config())

    assert config.database.application_dsn_ref.uri == "env://PLATFORM_POSTGRES_DSN"
    assert config.encryption.key_ref.uri == "env://PLATFORM_SESSION_KEY"
    assert config.sessions.lease_renewal_interval_seconds < config.sessions.lease_duration_seconds

    with pytest.raises(ValidationError, match="extra_forbidden"):
        PlatformRuntimeConfig.model_validate({**_valid_config(), "merchant_id": "acme_store"})


def test_platform_runtime_config_has_no_unmeasured_defaults() -> None:
    assert all(field.is_required() for field in PlatformRuntimeConfig.model_fields.values())
    assert all(
        field.is_required()
        for section in (
            PlatformDatabaseConfig,
            DurableSessionConfig,
            SessionEncryptionConfig,
        )
        for field in section.model_fields.values()
    )


@pytest.mark.parametrize(
    "section, field_name, value",
    (
        ("database", "application_dsn_ref", "postgresql://user:secret@database/service"),
        ("database", "application_dsn_ref", {"provider": "postgresql", "locator": "user:secret"}),
        ("encryption", "key_ref", "raw-key-material"),
        ("encryption", "key_ref", {"provider": "env", "locator": "secret value"}),
    ),
)
def test_platform_runtime_config_rejects_scalar_or_noncanonical_secret_references(
    section: str,
    field_name: str,
    value: object,
) -> None:
    candidate = _valid_config()
    candidate[section] = {**candidate[section], field_name: value}  # type: ignore[misc]

    with pytest.raises(ValidationError):
        PlatformRuntimeConfig.model_validate(candidate)


def test_platform_runtime_config_allows_a_lazy_database_pool() -> None:
    candidate = _valid_config()
    candidate["database"] = {  # type: ignore[assignment]
        **candidate["database"],  # type: ignore[misc]
        "minimum_pool_size": 0,
    }

    config = PlatformRuntimeConfig.model_validate(candidate)

    assert config.database.minimum_pool_size == 0


@pytest.mark.parametrize(
    "field_name",
    (
        "connection_timeout_seconds",
        "pool_acquisition_timeout_seconds",
        "statement_timeout_seconds",
        "transaction_timeout_seconds",
        "operation_timeout_seconds",
    ),
)
def test_platform_runtime_config_rejects_unbounded_database_budgets(
    field_name: str,
) -> None:
    candidate = _valid_config()
    database = dict(candidate["database"])  # type: ignore[arg-type]
    database[field_name] = float("inf")
    candidate["database"] = database

    with pytest.raises(ValidationError, match="finite_number"):
        PlatformRuntimeConfig.model_validate(candidate)


@pytest.mark.parametrize(
    "section, changes, expected_error",
    (
        (
            "database",
            {"minimum_pool_size": 8, "maximum_pool_size": 4},
            "maximum_pool_size",
        ),
        (
            "database",
            {"statement_timeout_seconds": 2.5, "transaction_timeout_seconds": 2.0},
            "statement timeout",
        ),
        (
            "database",
            {"transaction_timeout_seconds": 3.5, "operation_timeout_seconds": 3.0},
            "transaction timeout",
        ),
        (
            "sessions",
            {"lease_duration_seconds": 10.0, "lease_renewal_interval_seconds": 10.0},
            "renewal interval",
        ),
        (
            "sessions",
            {"lease_duration_seconds": 30.0, "session_retention_seconds": 20},
            "session retention",
        ),
    ),
)
def test_platform_runtime_config_rejects_incoherent_operational_budgets(
    section: str,
    changes: dict[str, object],
    expected_error: str,
) -> None:
    candidate = _valid_config()
    candidate[section] = {**candidate[section], **changes}  # type: ignore[misc]

    with pytest.raises(ValidationError, match=expected_error):
        PlatformRuntimeConfig.model_validate(candidate)
