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
config/              base, merchant, template, fixtures, eval, platform, telemetry, qualification,
                     and conformance artifacts
scripts/             worker, evaluators, smoke checks, recovery tools, PostgreSQL harness
tests/               synthetic unit, integration, adversarial, lifecycle, backend contracts
assets/audio/        the pipeline thinking beep and the latency harness utterances
.github/workflows/   locked verification workflow
```

## Development setup

Requirements: Python 3.12 or newer and [uv](https://docs.astral.sh/uv/). Node 20 or newer is used
only by the dependency-free browser-client tests; running the workbench does not require npm.
Docker is needed only for the default CI PostgreSQL harness; native binaries and a remote DSN also
work. Provider credentials are needed only for voice, live conformance, or credentialed evaluation.

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

Run the local merchant administration API with an explicit operator identity and local SQLite
state path:

```bash
uv run --no-sync python scripts/management_api.py \
  --database .local/merchant-management.sqlite3 \
  --actor-id local-operator
```

The development adapter is fixed to `127.0.0.1:8000`. Open the merchant workbench at
`http://127.0.0.1:8000/admin` or inspect the OpenAPI document at
`http://127.0.0.1:8000/docs`. It has no network authentication boundary and must not be exposed
beyond loopback. Tenant and actor authority are derived from URL scope and process configuration,
not accepted from write request bodies.

The SQLite repository is disposable development state, not a compatibility surface. Contract
changes advance its repository schema and reject an older database at startup. Stop the local
server, remove the explicitly supplied `--database` file, and restart to create the current schema;
drafts and publications must then be recreated through the management API.

The same loopback process exposes the development text-simulation API. A simulation starts from one
active or explicitly selected immutable publication, remains pinned to that publication across
turns and resets, and uses isolated in-memory session state. Provider credentials remain server-side
and are resolved from the existing environment-backed provider configuration only when a turn uses
them. Turn results and the dedicated `/state` resource expose a bounded workbench projection of the
cart, totals, order-context counts, committed receipt counts, turn count, and session revision. They
do not expose identity bindings, order references, receipt payloads, checkpoint values, prompts, or
secrets. The workbench simulator controls start or resume a named isolated session, send caller
turns, render only caller-facing events, reset to the same immutable publication, and close the
session. This development path does not authorize production routing or telephony.

Versioned synthetic scenario bundles live under `config/datasets/`. Each bundle contains one
complete tenant fixture snapshot plus a strict manifest that binds its tenant, revision, source,
entity counts, intended capabilities, scenario tags, and fixture fingerprint. The management API
imports the complete bundle atomically through
`/v1/merchants/{tenant_id}/drafts/{draft_id}/dataset`; a partial family update or a manifest whose
counts, tenant, or fingerprint do not match is rejected. The committed fashion and grocery bundles
intentionally reuse several SKU, order, and customer identifiers with different tenant-owned data.
They also include processing, shipped, delivered, and cancelled history plus customers with
intentionally missing dependent profile or payment data, so their tests prove scoping and
fail-closed availability behavior rather than depending on globally unique or uniformly complete
fixtures.

Simulator state includes a portable, value-free `committed_receipts` projection supplied through
the commerce ports. It reports cumulative committed cart, order, and profile receipt counts without
returning idempotency keys, customer references, order references, or receipt payloads. This is
development inspection evidence, not a business-ledger export or production activation signal.

Run the disposable-container PostgreSQL checkpoint harness used by CI:

```bash
uv run --no-sync python scripts/postgres_checkpoint_harness.py
```

To run the voice worker, copy `.env.example` to `.env` and supply provider and LiveKit credentials.
`VOICE_AGENT_DEPLOYMENT_ID` must identify the immutable deployed artifact, and console mode also
requires an explicit `VOICE_AGENT_MERCHANT_ID`. Network workers additionally require
`VOICE_AGENT_PLATFORM_CONFIG`, `VOICE_AGENT_CERTIFICATION_CONFIG`, and
`VOICE_AGENT_BUILD_ARTIFACT_DIGEST`, and the absolute `VOICE_AGENT_LATENCY_METHODOLOGY` and
`VOICE_AGENT_LATENCY_REPORT` paths carrying schema-5 voice evidence for the deployed runtime.
Production composition also requires the issued
`config/qualification/semantic_routing_release.json`; a standalone mutable routing report is not
activation authority. See `.env.example` for the complete set.

For a metadata-free LiveKit Cloud development session, set `VOICE_AGENT_MERCHANT_ID` and run the
isolated development worker:

```bash
uv run python scripts/voice_agent_development.py dev --no-reload --log-level debug
```

It registers as `<production-agent-name>-development`, accepts only the LiveKit `dev` command and a
standard participant, refuses production dispatch metadata, and uses in-memory session state. It
uses the configured semantic recognizer without claiming immutable routing qualification, and does
not exercise or authorize the durable platform. Production and certification workers retain their
strict dispatch metadata, routing-release package, immutable build identity, and deployment-evidence
gates.

## License

Apache License 2.0. See [LICENSE](LICENSE).

Copyright 2026 R-Abaryan.
