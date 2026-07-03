"""Secret resolution seam.

Config holds a `secrets_ref` (e.g. `env://ORDER_API_KEY`), never an inline secret
(SECURITY §5). A `SecretResolver` turns a ref into the actual secret at use time. Phase 0
ships an env/dotenv dev resolver; a Vault/cloud-KMS resolver plugs in at Phase 4/5 behind
the same Protocol, with no config-schema change.
"""

from agnostic_market.secrets.base import SecretResolutionError, SecretResolver
from agnostic_market.secrets.env_resolver import EnvSecretResolver

__all__ = ["EnvSecretResolver", "SecretResolutionError", "SecretResolver"]
