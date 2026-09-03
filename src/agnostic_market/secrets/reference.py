"""Structured references for secrets owned by a deployment provider."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, StringConstraints

SecretProvider = Annotated[
    str,
    StringConstraints(
        min_length=1,
        pattern=r"^[a-z][a-z0-9-]*$",
    ),
]
SecretLocator = Annotated[
    str,
    StringConstraints(
        min_length=1,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._/-]*$",
    ),
]


class SecretReference(BaseModel):
    """Structured provider lookup resolved outside application configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    provider: SecretProvider
    locator: SecretLocator

    @property
    def uri(self) -> str:
        return f"{self.provider}://{self.locator}"
