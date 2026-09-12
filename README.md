# Agnostic Market

Multi-tenant, provider-agnostic voice commerce with typed semantic routing and code-owned effects.

Agnostic Market is a production-shaped voice agent for catalog discovery, cart management, order
placement, order support, identity verification, and account changes. A model may understand the
caller and propose typed work. It never grants authority or commits an effect.

> Current status: the semantic-routing migration is merged. The Phase 4C durable multi-tenant
> runtime is in progress. The system is not production-qualified or authorized for real merchant
> traffic.

## Capabilities

The current immutable registry contains 16 typed entries covering:

- catalog search and bounded general or policy answers;
- cart view, add, remove, quantity change, and whole-cart placement;
- order status, session or account order listing, cancellation, refund, and return;
- identity verification, identity-status read, and account switching;
- profile changes, current-request abort, and the human on-ramp.

Call-start AI disclosure is owned by the voice lifecycle, not ordinary semantic routing. Payment
card capture, inventory, promotions, tax, shipping, fulfilment, and live SIP transfer are not
implemented capabilities.

## Architecture

The system separates real-time voice, reasoning, and data responsibilities:

| Plane | Responsibility | Current implementation |
|---|---|---|
| Voice | VAD, STT, TTS, turn admission, barge-in, disclosure | LiveKit Agents behind the voice adapter |
| Reasoning | semantic recognition, typed dispatch, deterministic owners, HITL, recovery | LangGraph behind `ReasoningEngine` |
| Data | tenant services, session authority, effects, receipts, checkpoints, telemetry | fixture-backed ports and in-memory session state; production PostgreSQL composition begins in Phase 4C Milestone 3 |

An ordinary committed turn follows one ownership path:

```text
caller speech
    -> voice pipeline
    -> ReasoningEngine
    -> one semantic recognizer
    -> typed dispatch or bounded no-action envelope
    -> immutable capability registry
    -> one deterministic capability owner
    -> live authorization, policy, consent, and effect boundary
    -> validated caller-facing result
```

`ReasoningEngine` owns ordinary language recognition before graph execution. The graph entry accepts
only an engine-authored dispatch, bounded no-action work, recovery work, or terminal state. There is
no legacy regex intent router, model-authored inter-flow handover, backup recognizer, or runtime
semantic fallback.

`build_application_session()` is the composition boundary. It binds one immutable tenant context,
one `TenantServices` bundle, and one `ApplicationSessionState` bundle before constructing the graph,
router, engine, telemetry, and caller lifecycle. Tenant mismatches fail before an owner can run.

## Safety and correctness

The main contracts are structural rather than prompt-only:

- **A typed request is not authority.** Owners re-resolve live targets and re-check identity,
  tenant, policy, and effect eligibility.
- **Consent is deterministic.** Irreversible actions require a complete code-authored readback and
  explicit consent. Negation wins and ambiguous language re-asks.
- **Intent has one semantic owner.** Open-language intent is never inferred with regexes, keyword
  lists, or substring matching. Deterministic classifiers are limited to closed-domain controls.
- **Speech has declared provenance.** Authoritative commerce outcomes are code-rendered. Dedicated
  catalog and answer model outputs are buffered, validated, and released only from declared
  model-speech nodes. Transactional model nodes cannot speak.
- **Authentication and authorization are separate.** A contact match may grant one order read. It
  cannot authorize mutation or bind an account. Protected account work uses fresh factor-bound
  verification.
- **Effects are replay-safe.** Proposals are checkpointed before consent, effect keys are stable,
  stores arbitrate idempotency, and recovery renders authoritative receipts instead of replaying a
  mutator blindly.
- **Failures converge safely.** Per-node recovery policies, terminal latching, execution
  quiescence, principal rotation, and verified checkpoint deletion prevent late work from reviving
  retired authority.
- **Configuration fails closed.** Platform safety limits constrain merchant policy, and production
  voice startup requires a current qualification report matching the exact routing runtime.

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
  voice/             trusted tenant admission, LiveKit pipeline, disclosure, and speech transport
scripts/             worker, evaluators, smoke checks, recovery tools, PostgreSQL harness
tests/               synthetic unit, integration, adversarial, lifecycle, and backend contracts
.github/workflows/   locked verification workflow
```

## Development setup

Requirements:

- Python 3.12 or newer;
- [uv](https://docs.astral.sh/uv/);
- Docker only for the default CI PostgreSQL harness, with native binaries and a remote DSN
  supported for local runs;
- provider credentials only for voice, live conformance, or credentialed evaluation.

Install the locked environment:

```bash
uv sync --frozen
```

Run the offline quality gates:

```bash
uv run --no-sync ruff format --check .
uv run --no-sync ruff check .
uv run --no-sync pytest -m "not postgres"
```

The offline suite assembles a complete synthetic configuration from committed fixture and test
artifacts. It requires no API keys or network access. Runtime voice sessions require the complete
local fixture families under `config/fixtures/` until durable service adapters replace them.

## PostgreSQL checkpoint harness

Phase 4C Milestone 1 selects PostgreSQL 18 as the only new durable service. The branch pins PostgreSQL 18.6,
`langgraph-checkpoint-postgres` 3.1.2, and Psycopg 3.3.4. The asynchronous saver remains behind the
same strict schema, namespace, serializer, deletion, and whole-operation deadline boundary as the
development saver.

Run the canonical disposable-container harness used by CI:

```bash
uv run --no-sync python scripts/postgres_checkpoint_harness.py
```

The harness exercises two independent pools, setup, read, list, checkpoint and pending-write
persistence, deletion, pool exhaustion, deadline, cancellation, and post-cancellation recovery.

Run the frozen controlled crash/concurrency matrix against the same isolated backend and record a
redacted, immutable result labelled with an operator-declared implementation identifier:

```bash
uv run --no-sync python scripts/postgres_checkpoint_harness.py \
  --crash-report config/telemetry/durable-crash-<run-id>.json \
  --implementation-id <commit-or-build-id>
```

The matrix runs exact existing architecture contracts; it does not duplicate them. Each case
declares its execution surface, so the pinned PostgreSQL version covers only backend cases, and each
result records the observed pytest collection counts, so a skipped or expected-failure case cannot
be recorded as a pass. After writing, the run is reloaded and verified against the registered
methodology; schema 1 artifacts predate these fields and are refused by identity rather than
silently compared.

The artifact carries the run envelope, the embedded methodology it was run against, and one result
per case holding only case identity, boundary, execution surface, outcome, observed counts, and
duration. It never contains a DSN, checkpoint payload, transcript, or caller data. The
implementation identifier is operator-declared metadata, not build attestation; trusted build
identity belongs to release assembly. A passing controlled report explicitly does not certify live
transport or authorize activation.

For a Docker-free local run, set exactly one alternative before invoking the same command:

- `PHASE4C_POSTGRES_DSN` uses a PostgreSQL 18.6 service. The harness creates a uniquely named
  schema, binds every test connection to it with `search_path`, and removes only that schema.
- `PHASE4C_POSTGRES_BIN` names an extracted PostgreSQL 18.6 binary directory containing `initdb`,
  `pg_ctl`, and `postgres`. The harness creates, starts, stops, and removes a fresh local cluster.

The native path verifies the server executable reports exactly PostgreSQL 18.6. Release provenance
still depends on obtaining and checksum-verifying the archive from the PostgreSQL Windows download
provider. Setting both alternatives is rejected as ambiguous configuration.

Network-worker composition now requires encrypted fenced PostgreSQL checkpoints. Encrypted session
payloads, registry leases and fencing, revisioned reconstruction, and shared durable close/reaping
are implemented and backend-tested through the common application builder. A concrete graph-latency
runner creates fresh job-scoped durable resources, executes the frozen synthetic journey corpus
through the real graph, consumes the graph's existing turn-span measurement, and writes one
immutable, redacted result. Its schema-3 methodology binds the whole canonical journey corpus, the
redacted effective platform configuration and PostgreSQL endpoint, and the `reasoning_graph`
measurement surface. This result characterizes the durable graph path; it cannot authorize a
network worker because it excludes LiveKit voice processing and TTS. The runner is explicit and is
never launched by worker startup:

```bash
uv run python scripts/durable_latency_certification.py \
  --platform-config <absolute-platform-yaml> \
  --methodology <pre-registered-methodology-yaml> \
  --report <new-result-json> \
  --deployment-id <immutable-deployment-id>
```

The methodology must bind the three journey IDs in
`config/eval/durable_latency_journeys.yaml` to these current SHA-256 contract fingerprints:

- `simple-cart-read`: `a9304596004c7bd0515380f0cbab5ef5caf62bbf8b6ef607328304a1556bbe67`
- `commerce-cart-confirmation`: `d2da9a766025b7e196688531453fc0babe455cbe39d0227eb26b0b786af97083`
- `checkout-placement-readback`: `0c57f13de9e35722c199b81e204a8aee2f518a2acec3c9e181f3cf1e5bb458d3`

Choose and freeze the deployment-owned backend location, startup threshold, concurrency, sample
count, watchdog, and required operation coverage before the first run. Do not copy a local result
into deployment activation. Scheduling of the operational reaper, complete crash/concurrency
certification, the actual deployment-shaped latency result, and live transport evidence remain
open. The voice pipeline also correlates LiveKit's existing user
`end_of_turn_delay` and assistant `e2e_latency`, retains both, and derives processing-only latency
without adding another timer; missing, ambiguous, or contradictory pairs produce no measurement.
Warmup failure produces a versioned, redacted aborted-run artifact rather than disappearing or
becoming a partial report. Controlled evidence is not eligible to authorize deployment activation.
Turn-component evidence excludes setup and cleanup; cleanup failures still fail the gate, and
cleanup-only measurements cannot satisfy turn coverage.
Latency methodology schema 3 requires non-empty, unique operation coverage for startup and each
journey. It also binds the full journey-corpus fingerprint, a non-secret fingerprint of the exact
platform runtime and database endpoint, and the measured surface. These requirements enter the
methodology fingerprint and are checked per sample, not across the run as a whole. Concrete
deployment probes must also assert the intended journey behaviour; operation names alone are not
proof of correct execution.
If fresh registration succeeds but subsequent session setup fails, its confirmed lease is closed
through the existing lifecycle coordinator. The original error (including cancellation) is retained
if cleanup also fails. Unacknowledged registration does not authorize guessed cleanup or reacquisition;
unresolved rows remain subject to fenced reaping.
The checkpoint and session-registry contracts have passed through
a schema-isolated remote service, a fresh native Windows cluster, and the digest-pinned Docker path
in the repository workflow. Each provisioner removes only the database resources it owns.

Run one bounded expiry and tombstone-cleanup cycle with a deployment-owned platform configuration:

```bash
uv run --no-sync python scripts/session_reaper.py --platform-config <path> --once
```

Omit `--once` only in the dedicated reaper process. Platform configuration schema version 2
requires explicit reaper cadence and batch-size values. The command enumerates trusted tenant IDs
from the merchant registry and reuses the fenced close coordinator; it does not run inside a voice
job.

Milestone 3 now has native async application construction, registry and encryption contracts,
lease supervision, and revisioned session-state adapters. The same harness also exercises registry
fencing, encrypted operation replay, and committed-but-unacknowledged principal retirement.
These are component and lifecycle contracts, not proof of a production durable voice session.
The next boundary is the frozen crash, concurrency, latency, and live transport certification that
decides whether the durable production path can be enabled.
The production UK DID/SIP carrier is intentionally not selected yet. The merchant fixture's
current Telnyx value is a development configuration, not deployment evidence or a carrier lock-in.
Durable verification later owns full principal recovery. Durable commerce owns
post-effect receipt reconciliation and the transactional commerce-success outbox. Telemetry
delivery drains that outbox asynchronously and never decides whether an effect committed.
Network admission now reserves the server-issued LiveKit dispatch ID for logical session identity
and binds the room SID, job ID, and worker ID as physical transport authority. Room identity alone
never authorizes recovery. Worker-loss recovery remains terminate-call until the live crash matrix
proves dispatch continuity, old-transport retirement, and one replacement speaker.

## Voice worker

Copy `.env.example` to `.env` and provide the required provider and LiveKit credentials.
`VOICE_AGENT_DEPLOYMENT_ID` must identify the immutable deployed artifact. Console mode also
requires an explicit `VOICE_AGENT_MERCHANT_ID`; there is no default merchant.
`VOICE_AGENT_NAME` is required for network worker commands and must match the named LiveKit
dispatch target. Network workers also require `VOICE_AGENT_PLATFORM_CONFIG` to be an absolute
path to the deployment-owned schema-2 platform runtime YAML. Console and model-file download
commands do not require either network setting. The YAML resolves `PLATFORM_POSTGRES_DSN` as the
restricted application-role connection and `PLATFORM_SESSION_KEY` as a base64-encoded 32-byte
AES-256 key. Do not reuse the migration or Phase 4C harness administrator connection as the
application DSN. Worker activation additionally requires a schema-3 latency methodology and report
whose measurement surface is `voice_processing`; the repository's current graph-only certification
runner deliberately cannot produce that activation artifact.

On Windows, network-worker prewarm selects the psycopg-compatible selector loop inside each
LiveKit job process; the running supervisor loop is unchanged. Console mode uses LiveKit's thread
executor and keeps the platform-default Proactor loop because it does not open durable PostgreSQL
resources. The installed LiveKit SDK exposes no job `loop_factory`, so network jobs use Python's
policy API, deprecated in 3.14 and scheduled for removal in 3.16. Replace that bridge with explicit
SDK loop selection before the upgrade. The reaper owns its loop directly and already uses
`loop_factory`.

```bash
uv run --no-sync python scripts/voice_agent.py console
uv run --no-sync python scripts/voice_agent.py dev
```

Every network job uses named explicit dispatch. A strict metadata preflight resolves the tenant,
then model certification, platform secret resolution, PostgreSQL pool creation, and read-only
schema and role gates finish before the worker joins the room. Participant admission binds the
transport before the worker atomically acquires a fresh session lease. The fenced encrypted
checkpointer and application session are built only after that lease succeeds. Existing session
rows remain fail-closed; this path does not enable replacement-worker takeover.
Standard participants require metadata shaped
as `{"schema_version":1,"merchant_id":"<configured merchant>","participant_kind":"standard","participant_identity":"<participant identity>"}`.
SIP participants require metadata shaped as
`{"schema_version":1,"merchant_id":"<configured merchant>","participant_kind":"sip"}`; an optional
`participant_identity` may bind a specific caller. SIP admission also reads LiveKit's
server-populated `sip.trunkPhoneNumber` and requires it to resolve to the metadata tenant. Caller
ANI is never a tenant authority. Inbound numbers must use canonical E.164 form. Unknown,
malformed, missing, duplicate, or conflicting admission signals fail the job. Room connection and
participant binding share the safety-locked `runtime.voice_admission_timeout_seconds` budget.
Participant or DID rejection happens before the voice session starts or any owner executes.

Voice startup is fail-closed. The configured routing model requires a current passing report at
`config/telemetry/semantic_routing_report.json`. The report must match the active model, structured
output method, route schema, prompt, registry, context projector, runtime limits, and frozen corpus.
A stale or failed report prevents recognizer construction. Qualification is evaluated against the
exact graph capability registry before platform resources are opened or the worker joins a room.

Secrets are never stored in configuration files. Config holds `env://NAME` references and resolves
their values at use time. `.env` is ignored by Git.

## Current delivery boundary

Implemented and exercised offline:

- typed semantic routing and the complete executable capability registry;
- deterministic authorization, consent, effect, receipt, and speech boundaries;
- cart, order-support, identity, profile, recovery, replay, and lifecycle journeys;
- trusted voice tenant admission and session composition with fixture implementations behind
  narrow ports;
- asynchronous PostgreSQL checkpoint conformance against a real PostgreSQL 18.6 service;
- encrypted session registry, lease/fence enforcement, revisioned state and operation receipts;
- network-worker composition from admitted transport authority through a fresh durable lease,
  encrypted fenced checkpoints, lease supervision, and ordered resource teardown;
- local teardown that discards caller state on durable failure without claiming durable deletion.

Still required before production activation:

- a qualified semantic recognizer, latency and outage evidence, and reviewed live shadow;
- deployment-shaped durable-path latency and crash evidence, live transport certification, and
  operational reaper activation;
- durable business, verification, abuse-control, and operational telemetry adapters;
- hard shared-store tenant isolation and production restore and failover evidence;
- payment, inventory, tax, shipping, fulfilment, and live human-transfer contracts where required.

Synthetic evidence is the current development authority. It is used to deepen tests and
evaluations, not to weaken rubrics or claim real-caller population accuracy.

## License

Apache License 2.0. See [LICENSE](LICENSE).

Copyright 2026 Rasoul Abaryan.
