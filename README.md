# Agnostic Market

Multi-tenant, LLM-agnostic voice commerce agent. The model proposes; code authorizes and executes.

A caller talks to a voice agent that can search a catalog, build a cart, place an order, and handle
cancellations, refunds, returns, and profile changes. Every irreversible action passes through
deterministic guardrails, a spoken readback, and explicit consent before any effect is committed.
The model never holds effect authority.

## Design rules

Enforced in code, not by convention.

**Two planes.** The model proposes a typed request. Code resolves the target, re-checks
authorization against live state, and executes. A tool call is a proposal, never a command.

**One author per turn.** Graph topology decides who may speak. A node that invokes a model cannot
author caller-facing text, and graph construction fails if those two sets intersect.

**Authorization is not authentication.** Mutating an order requires an OTP-bound identity or an
order placed in the current session. A contact match grants reads only.

**No existence oracle.** An unknown order and someone else's order produce the same response, so
the system cannot be probed for which orders exist.

**Policy within bounds.** Merchants tune refund thresholds and return windows; platform ceilings
clamp them. The spoken policy summary is derived from the enforced values, so a model cannot state
a threshold the guardrail would refuse.

**Effects survive failure.** Pending actions are serialized, idempotent, revalidated against the
store before commit, and resumed from a checkpoint rather than reinterpreted as a fresh turn.

**Providers are swappable.** STT, TTS, and LLM providers are config-driven. A model serves commerce
turns only after passing a fail-closed conformance gate.

## Layout

```
src/agnostic_market/
  dtos/       Pydantic contracts, the single source of truth for shared types
  config/     three-layer merchant resolution, safety-locked keys enforced in code
  tenancy/    merchant resolution and immutable per-session context
  secrets/    pluggable SecretResolver; config holds env:// refs, never values
  llm/        provider-agnostic gateway and the conformance gate
  voice/      STT/TTS factories, LiveKit adapter, call-start disclosure
  commerce/   fixture-backed catalog, orders, cart, profile, verification
  agents/     the tiered reasoning graph and its engine
scripts/      worker entrypoint, evaluator, conformance, transport-fault and smoke runners
tests/        38 test modules, zero network
```

Inside `agents/`, each gated flow is a package pairing graph logic with its model-facing prompt.
`engine.py` owns ordinary-turn semantic recognition plus thread and resume lifecycle,
`frontline/graph.py` dispatches typed requests, `recovery.py` owns per-node failure policy, and
`capabilities.py` provides the immutable per-session registry. `cart/`, `support/`, `identity/`,
and `frontline/read_flow.py` own execution. `routing.py` understands intent but carries no live
authority.

## Run

Requires Python 3.12 or newer, and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run ruff check
uv run ruff format --check
uv run pytest                # no network and no API keys, but see below
```

The suite needs merchant fixture data under `config/fixtures/`, which is not committed. Those
files stand in for a system of record the code does not ship with, so the commerce, identity, and
verification suites cannot run from a clean checkout.

To run the voice worker, copy `.env.example` to `.env`, fill in the provider keys, then:

```bash
uv run python scripts/voice_agent.py console   # terminal, no LiveKit room
uv run python scripts/voice_agent.py dev       # connect to a LiveKit room
```

Voice startup is fail-closed. In addition to provider conformance, the configured routing model
must have a current passing cutover report at
`config/telemetry/semantic_routing_report.json`. The report must match the active model, structured
output method, route schema, prompt, registry, context projector, runtime limits, and frozen corpus.
The preserved rejected report does not satisfy that contract.

Secrets are never committed. Config files hold `env://NAME` references and the resolver reads the
value at use time.

## Status

Phases 0 through 3 are built: config and tenancy, the LLM gateway and conformance gate, the voice
loop, and the gated commerce graph. The current feature branch has completed the source-level
migration from keyword gates to typed capability requests resolved through one immutable
per-session registry. Ordinary human and abort requests are semantic capabilities; deterministic
code still owns consent, authorization, effects, and speech from authoritative outcomes.

This architecture is not production-qualified. The prior generative candidate remains rejected,
and voice startup now rejects it before constructing the recognizer. Activation still requires an
approved recognizer, per-domain acceptance, latency and outage evidence, live shadow review, and
the separate disclosure owner.

Deliberately not built yet:

- Commerce, profile, and verification data are fixture-backed. There is no real system of record.
- Checkpointing is in memory, per session.
- No payment processing, inventory, tax, shipping, or live human transfer.
- Tenant scoping is enforced at the tool wrapper, not yet at a shared store.

Real integrations, durable checkpointing, shared-store tenant enforcement, and production OTP
delivery are Phase 4 work. Payments and SIP warm transfer are Phase 5.

## License

Apache License 2.0. See [LICENSE](LICENSE).

Copyright 2026 Rasoul Abaryan.
