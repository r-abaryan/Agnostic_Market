"""Dev secret resolver — reads `env://NAME` refs from the environment / a .env file.

DEV ONLY. Production uses a Vault/cloud-KMS resolver (Phase 4/5) behind the same
`SecretResolver` Protocol. This resolver handles the `env://` scheme; any other scheme
(e.g. `vault://`) raises, so a prod ref can't silently fall through to the dev path.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from agnostic_market.secrets.base import SecretResolutionError

_ENV_SCHEME = "env://"


class EnvSecretResolver:
    """Resolve `env://NAME` against environment variables (optionally seeded from .env)."""

    def __init__(self, *, load_dotenv_file: bool = True) -> None:
        # Load .env into the process env once, if present. Non-fatal if absent.
        if load_dotenv_file:
            load_dotenv()

    def resolve(self, ref: str) -> str:
        if not ref.startswith(_ENV_SCHEME):
            raise SecretResolutionError(
                f"EnvSecretResolver only handles '{_ENV_SCHEME}' refs, got: {ref!r}"
            )
        name = ref[len(_ENV_SCHEME) :]
        if not name:
            raise SecretResolutionError(f"empty env var name in ref: {ref!r}")
        value = os.environ.get(name)
        if value is None:
            raise SecretResolutionError(f"env var {name!r} (from ref {ref!r}) is not set")
        return value
