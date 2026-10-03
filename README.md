# Agnostic Market

Multi-tenant, provider-agnostic voice commerce with typed semantic routing and code-owned effects.

A production-shaped voice agent for catalog discovery, cart management, order placement, order
support, identity verification, and account changes. A model may understand the caller and propose
typed work. It never grants authority or commits an effect.

> Status: the semantic-routing migration is merged and the Phase 4C durable multi-tenant runtime is
> in progress. Not production-qualified and not authorized for real merchant traffic. Synthetic
> evidence is the current development authority, used to deepen tests, not to weaken rubrics or to
> claim real-caller accuracy. Routing promotion now awaits one source-disjoint cutover package;
> schema-5 voice-processing certification follows after that package exists.

## Architecture

Three planes, each owning one responsibility:

| Plane | Responsibility | Implementation |
|---|---|---|
| Voice | VAD, STT, TTS, turn admission, barge-in, disclosure | LiveKit Agents behind the voice adapter |
| Reasoning | semantic recognition, typed dispatch, deterministic owners, HITL, recovery | LangGraph behind `ReasoningEngine` |
| Data | tenant services, session authority, effects, receipts, checkpoints, telemetry | fixture-backed service ports; PostgreSQL session registry and lifecycle, completing in Phase 4C |

One ordinary committed turn follows a single ownership path:

```text
caller speech
    -> voice pipeline
    -> ReasoningEngine
    -> one semantic recognizer
    -> typed dispatch or bounded no-action envelope
    -> immutable capability registry (18 typed entries)
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
  agents/       engine, semantic routing, capability registry, and the frontline, cart,
                support and identity flows
  commerce/     service ports, fixture adapters, effects, receipts
  config/       base, template, merchant, policy and provider resolution
  dtos/         strict Pydantic state, routing, confirmation and money contracts
  durability/   encryption, migrations, session registry, leases
  llm/          provider gateway and model conformance
  voice/        tenant admission, LiveKit pipeline, disclosure
config/         base, merchant, template, fixtures, eval and qualification artifacts
scripts/        worker, evaluators, smoke checks, PostgreSQL harness
tests/          synthetic unit, integration, adversarial and lifecycle tests
```

## Development setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/). Docker is needed only for the
PostgreSQL harness, Node 20+ only for the browser-client tests, and provider credentials only for
voice and live evaluation.

```bash
uv sync --frozen
uv run --no-sync ruff format --check .
uv run --no-sync ruff check .
uv run --no-sync pytest -m "not postgres"   # offline: no API keys, no network
```

### Merchant workbench

```bash
uv run --no-sync python scripts/management_api.py \
  --database .local/merchant-management.sqlite3 \
  --actor-id local-operator
```

Open `http://127.0.0.1:8000/admin` (API docs at `/docs`). It has no authentication, so keep it on
loopback. The SQLite file is disposable: if startup rejects it after a config change, stop the
server, delete the file, restart, and republish. Text simulations run in the same process; each
pins one publication and uses in-memory session state. Synthetic scenario bundles in
`config/datasets/` import through the draft dataset endpoint.

### PostgreSQL harness

```bash
uv run --no-sync python scripts/postgres_checkpoint_harness.py
```

### Voice worker

Copy `.env.example` to `.env` and fill in the provider and LiveKit credentials; the file lists
every variable, including the extra ones production workers need. For a LiveKit Cloud
development session, set `VOICE_AGENT_MERCHANT_ID` and run:

```bash
uv run python scripts/voice_agent_development.py dev --no-reload --log-level debug
```

## License

Apache License 2.0. See [LICENSE](LICENSE).

Copyright 2026 R-Abaryan.
