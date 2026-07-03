"""The `SecretResolver` seam — the stable interface every backend implements."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


class SecretResolutionError(RuntimeError):
    """A `secrets_ref` could not be resolved (unknown scheme, missing value, etc.)."""


@runtime_checkable
class SecretResolver(Protocol):
    """Resolve a `secrets_ref` string into its secret value.

    A ref is `<scheme>://<locator>` (e.g. `env://ORDER_API_KEY`, later `vault://...`).
    Implementations must raise `SecretResolutionError` on failure — never return a
    placeholder or empty string (a silent-empty secret is a security footgun).
    """

    def resolve(self, ref: str) -> str: ...
