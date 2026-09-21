"""Structured references for secrets owned by a deployment provider."""

from __future__ import annotations

from typing import Annotated, Self

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

    @classmethod
    def from_uri(cls, value: str) -> Self:
        """Parse one provider reference without resolving its secret value."""

        provider, separator, locator = value.partition("://")
        if not separator:
            raise ValueError("secret reference must use provider://locator syntax")
        return cls(provider=provider, locator=locator)

    @property
    def uri(self) -> str:
        return f"{self.provider}://{self.locator}"
