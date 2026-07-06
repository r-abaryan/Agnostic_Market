# Agnostic Market

Multi-tenant, LLM-agnostic voice commerce agent platform.

**Status: Phase 2** — minimal voice loop (LiveKit + Deepgram + LangGraph + Cartesia), on top of:
- **Phase 0** — `dtos/` (single Pydantic v2 source of truth) · `config/` (3-layer resolution, safety-locked keys enforced in code) · `tenancy/` (merchant resolution + immutable session context) · `secrets/` (pluggable `SecretResolver`).
- **Phase 1** — `llm/` (provider-agnostic gateway + fail-closed conformance gate: a model serves commerce turns only after passing certification).
- **Phase 2** — `voice/` (config-driven STT/TTS engine factories, read-only tools, minimal graph behind LiveKit's `LLMAdapter`, call-start AI disclosure played first in code).

## Develop

```bash
uv sync
uv run ruff check
uv run ruff format --check
uv run pytest        # zero-network; no keys needed
```

Secrets are never committed: copy `.env.example` to `.env` and fill in keys; config
files hold `env://NAME` refs only.
