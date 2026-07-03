# Agnostic Market

Multi-tenant, LLM-agnostic voice commerce agent platform.

**Status: Phase 0** — config + tenancy foundation. Design docs are the source of truth: see [`docs/`](./docs) (start with [ARCHITECTURE.md](./docs/ARCHITECTURE.md) and [BUILD_PLAN.md](./docs/BUILD_PLAN.md)).

## Phase 0 scope
The foundational plumbing everything else sits on:
- `dtos/` — the single Pydantic v2 source of truth for shared shapes.
- `config/` — 3-layer config resolution (platform base → vertical template → merchant override) with safety-locked keys enforced in code.
- `tenancy/` — resolve a caller/request to a `merchant_id` + an immutable per-session tenant context.
- `secrets/` — a pluggable `SecretResolver` seam (env/dotenv dev impl; Vault/KMS later).

Later planes (voice / reasoning / agents / stores / …) are built in their own phases (BUILD_PLAN.md).

## Develop
```bash
uv sync --extra dev
uv run ruff check
uv run ruff format --check
uv run pytest
uv run python scripts/smoke.py   # exercises the Phase-0 exit end-to-end
```
