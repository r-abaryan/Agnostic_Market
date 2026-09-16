# Agnostic Market

Multi-tenant, provider-agnostic voice commerce with typed semantic routing and code-owned effects.

A production-shaped voice agent for catalog discovery, cart management, order placement, order
support, identity verification, and account changes. A model may understand the caller and propose
typed work. It never grants authority or commits an effect.

> Status: the semantic-routing migration is merged and the Phase 4C durable multi-tenant runtime is
> in progress. Not production-qualified and not authorized for real merchant traffic. Synthetic
> evidence is the current development authority, used to deepen tests, not to weaken rubrics or to
> claim real-caller accuracy.

## Architecture

Three planes, each owning one responsibility:

| Plane | Responsibility | Implementation |
|---|---|---|
| Voice | VAD, STT, TTS, turn admission, barge-in, disclosure | LiveKit Agents behind the voice adapter |
| Reasoning | semantic recognition, typed dispatch, deterministic owners, HITL, recovery | LangGraph behind `ReasoningEngine` |
| Data | tenant services, session authority, effects, receipts, checkpoints, telemetry | fixture-backed ports and in-memory session state; PostgreSQL composition lands in Phase 4C |

One ordinary committed turn follows a single ownership path:

```text
caller speech
    -> voice pipeline
    -> ReasoningEngine
    -> one semantic recognizer
    -> typed dispatch or bounded no-action envelope
    -> immutable capability registry (16 typed entries)
    -> one deterministic capability owner
    -> live authorization, policy, consent, and effect boundary
    -> validated caller-facing result
```

`ReasoningEngine` owns language recognition before graph execution. The graph entry accepts only an
engine-authored dispatch, bounded no-action work, recovery work, or terminal state. There is no
regex intent router, model-authored handover, backup recognizer, or runtime semantic fallback.

`build_application_session()` is the composition boundary. It binds one immutable tenant context,
one `TenantServices` bundle, and one `ApplicationSessionState` bundle before constructing the graph,
router, engine, telemetry, and caller lifecycle. Tenant mismatches fail before an owner can run.

Call-start AI disclosure is owned by the voice lifecycle, not by ordinary semantic routing. Payment
card capture, inventory, promotions, tax, shipping, fulfilment, and live SIP transfer are not
implemented capabilities.

## Repository structure

```text
src/agnostic_market/
  application.py     tenant services and application-session composition
  checkpoints.py     strict checkpoint namespace, schema, serializer, and I/O boundary
  session.py         caller authority, lifecycle, close, and principal transition
  agents/
    engine.py        turn admission, semantic routing, replay, and recovery orchestration
    capabilities.py  immutable typed capability registry
    routing.py       recognizer-neutral semantic routing boundary
    frontline/       dispatcher, typed read owners, and caller-facing graph assembly
    cart/            cart mutation and placement flow
    support/         cancel, refund, return, and profile-change flow
    identity/        factor-bound identity flow
  commerce/          service ports, fixture adapters, effects, receipts, and renderers
  config/            validated base, template, merchant, policy, and provider resolution
  dtos/              strict Pydantic state, routing, confirmation, and money contracts
  durability/        encryption, migrations, session registry, leases, and revisioned state
  llm/               provider gateway and model conformance
  secrets/           environment-backed secret resolution
  tenancy/           immutable tenant identity and resolution
  voice/             trusted tenant admission, LiveKit pipeline, disclosure, speech transport
config/              base, merchant, policy, fixture, eval, and telemetry artifacts
scripts/             worker, evaluators, smoke checks, recovery tools, PostgreSQL harness
tests/               synthetic unit, integration, adversarial, lifecycle, backend contracts
assets/audio/        recorded utterances used by the latency measurement harness
.github/workflows/   locked verification workflow
```

## Development setup

Requirements: Python 3.12 or newer and [uv](https://docs.astral.sh/uv/). Docker is needed only for
the default CI PostgreSQL harness; native binaries and a remote DSN also work. Provider credentials
are needed only for voice, live conformance, or credentialed evaluation.

Install the locked environment:

```bash
uv sync --frozen
```

Run the offline quality gates, which need no API keys and no network:

```bash
uv run --no-sync ruff format --check .
uv run --no-sync ruff check .
uv run --no-sync pytest -m "not postgres"
```

The offline suite assembles a complete synthetic configuration from committed fixture and test
artifacts. Runtime voice sessions require the fixture families under `config/fixtures/` until
durable service adapters replace them.

Run the disposable-container PostgreSQL checkpoint harness used by CI:

```bash
uv run --no-sync python scripts/postgres_checkpoint_harness.py
```

To run the voice worker, copy `.env.example` to `.env` and supply provider and LiveKit credentials.
`VOICE_AGENT_DEPLOYMENT_ID` must identify the immutable deployed artifact, and console mode also
requires an explicit `VOICE_AGENT_MERCHANT_ID`. Network workers additionally require
`VOICE_AGENT_PLATFORM_CONFIG`, `VOICE_AGENT_CERTIFICATION_CONFIG`, and
`VOICE_AGENT_BUILD_ARTIFACT_DIGEST`, and activation requires schema-5 voice evidence matching the
deployed runtime. See `.env.example` for the complete set.

## License

Apache License 2.0. See [LICENSE](LICENSE).

Copyright 2026 R-Abaryan.
