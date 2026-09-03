"""Deployment-owned configuration for the durable application runtime."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from agnostic_market.secrets.reference import SecretReference

_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)

ConfigIdentifier = Annotated[
    str,
    StringConstraints(
        min_length=1,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    ),
]


class PlatformDatabaseConfig(BaseModel):
    model_config = _STRICT

    application_dsn_ref: SecretReference
    minimum_pool_size: int = Field(ge=0)
    maximum_pool_size: int = Field(ge=1)
    connection_timeout_seconds: float = Field(gt=0)
    pool_acquisition_timeout_seconds: float = Field(gt=0)
    statement_timeout_seconds: float = Field(gt=0)
    transaction_timeout_seconds: float = Field(gt=0)
    operation_timeout_seconds: float = Field(gt=0)
    expected_schema_version: int = Field(ge=1)

    @model_validator(mode="after")
    def _validate_budgets(self) -> Self:
        if self.maximum_pool_size < self.minimum_pool_size:
            raise ValueError("maximum_pool_size must be at least minimum_pool_size")
        if self.statement_timeout_seconds > self.transaction_timeout_seconds:
            raise ValueError("statement timeout must not exceed transaction timeout")
        if self.transaction_timeout_seconds > self.operation_timeout_seconds:
            raise ValueError("transaction timeout must not exceed operation timeout")
        return self


class DurableSessionConfig(BaseModel):
    model_config = _STRICT

    lease_duration_seconds: float = Field(gt=0)
    lease_renewal_interval_seconds: float = Field(gt=0)
    session_retention_seconds: int = Field(gt=0)
    closed_tombstone_retention_seconds: int = Field(gt=0)

    @model_validator(mode="after")
    def _validate_lifecycle_windows(self) -> Self:
        if self.lease_renewal_interval_seconds >= self.lease_duration_seconds:
            raise ValueError("lease renewal interval must be shorter than the lease duration")
        if self.session_retention_seconds <= self.lease_duration_seconds:
            raise ValueError("session retention must be longer than the lease duration")
        return self


class SessionEncryptionConfig(BaseModel):
    model_config = _STRICT

    envelope_format: Literal["aes_256_gcm_v1"]
    key_ref: SecretReference
    key_version: ConfigIdentifier


class PlatformRuntimeConfig(BaseModel):
    """Validated deployment settings that merchant configuration cannot override."""

    model_config = _STRICT

    schema_version: Literal[1]
    database: PlatformDatabaseConfig
    sessions: DurableSessionConfig
    encryption: SessionEncryptionConfig
