# Agnostic Market

Multi-tenant, LLM-agnostic voice commerce agent platform.

**Status: Phase 3 commerce graph — implemented and under review.** The voice loop runs the
frontline, cart, support, returns, profile, and identity flows behind `ReasoningEngine`; the
Phase-3d planner remains conditional on a trace proving that the direct flows are insufficient.
The current build is intentionally fixture-backed for commerce and verification, with an
in-memory per-session checkpointer. Real system integration, durable checkpointing, shared-store
tenant enforcement, and production OTPs remain Phase 4 work; payments and SIP warm transfer are
Phase 5.

Completed hardening in the current build: fixture-validated temporary OTP configuration (no
source-code default), deterministic live status re-checks for “is it cancelled?” follow-ups,
code-rendered order status, and an exact `max_tool_hops` ceiling with no dangling tool calls.

Built foundations:

- **Phase 0** — `dtos/` (single Pydantic v2 source of truth) · `config/` (3-layer resolution, safety-locked keys enforced in code) · `tenancy/` (merchant resolution + immutable session context) · `secrets/` (pluggable `SecretResolver`).
- **Phase 1** — `llm/` (provider-agnostic gateway + fail-closed conformance gate: a model serves commerce turns only after passing certification).
- **Phase 2** — `voice/` (config-driven STT/TTS engine factories, read-only tools, minimal graph behind LiveKit's `LLMAdapter`, call-start AI disclosure played first in code).
- **Phase 3** — gated commerce graph: fixture-backed catalog/orders/profile, cart placement, cancel/refund/return/profile flows, risk-gated OTP step-up, object-bound order reads, OTP-bound enumeration, deterministic read rendering, and the human-on-ramp context package.

## Develop

```bash
uv sync
uv run ruff check
uv run ruff format --check
uv run pytest        # zero-network; no keys needed
```

Secrets are never committed: copy `.env.example` to `.env` and fill in keys; config
files hold `env://NAME` refs only.
