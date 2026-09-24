import assert from "node:assert/strict";
import test from "node:test";

import {
  ManagementApi,
  ManagementApiError,
  buildDraftUpdate,
  canRollback,
  parseCatalog,
  selectMerchantWorkspace,
  simulationDiagnostics,
  simulationTurnRequest,
  simulationMessages,
} from "../../src/agnostic_market/management/ui/assets/client.js";
import {
  recognitionFailure,
  spokenText,
  supportSummary,
} from "../../src/agnostic_market/management/ui/assets/voice.js";

test("API errors expose bounded status and code without reflecting response text", async () => {
  const api = new ManagementApi(async () =>
    new Response(JSON.stringify({ schema_version: 1, code: "state_conflict" }), {
      status: 409,
      headers: { "content-type": "application/json" },
    }),
  );

  await assert.rejects(
    api.getDraft("tenant a", "draft/1"),
    (error) =>
      error instanceof ManagementApiError &&
      error.status === 409 &&
      error.code === "state_conflict" &&
      !error.message.includes("tenant a"),
  );
});

test("non-JSON failures preserve the status in a bounded API error", async () => {
  const api = new ManagementApi(async () =>
    new Response("Internal Server Error", {
      status: 500,
      headers: { "content-type": "text/plain" },
    }),
  );

  await assert.rejects(
    api.listMerchants(),
    (error) =>
      error instanceof ManagementApiError &&
      error.status === 500 &&
      error.code === "service_unavailable" &&
      !error.message.includes("Internal Server Error"),
  );
});

test("tenant and draft identifiers are encoded into request paths", async () => {
  const calls = [];
  const api = new ManagementApi(async (path, options) => {
    calls.push({ path, options });
    return new Response(JSON.stringify({ tenant_id: "tenant a", revision: 1 }), {
      status: 200,
      headers: { "content-type": "application/json" },
    });
  });

  await api.getDraft("tenant a", "draft/1");

  assert.equal(calls[0].path, "/v1/merchants/tenant%20a/drafts/draft%2F1");
  assert.equal(calls[0].options.headers.accept, "application/json");
});

test("draft updates send editable content without server-owned timestamps", () => {
  const current = {
    revision: 4,
    created_at: "2026-09-20T10:00:00Z",
    fixtures: { catalog: { products: [] } },
    dataset_manifest: { schema_version: 1, dataset_id: "fashion-service" },
  };

  const update = buildDraftUpdate(current, { merchant_id: "acme_store" }, "request-5");

  assert.equal(update.expectedRevision, 4);
  assert.equal(update.payload.revision, 5);
  assert.equal("created_at" in update.payload, false);
  assert.equal("updated_at" in update.payload, false);
  assert.deepEqual(update.payload.fixtures, current.fixtures);
  assert.deepEqual(update.payload.dataset_manifest, current.dataset_manifest);
});

test("catalog parsing handles JSON syntax but leaves catalog schema to the API", () => {
  const catalog = parseCatalog(
    JSON.stringify({
      products: [
        { sku: "SKU-1", name: "First", price_usd: "10.00" },
        { sku: "SKU-2", name: "Second", price_usd: "12.50" },
      ],
    }),
  );
  assert.equal(catalog.products.length, 2);

  assert.deepEqual(parseCatalog('{"future_schema_field":true}'), { future_schema_field: true });
  assert.throws(() => parseCatalog("not json"), /valid JSON/);
  assert.throws(() => parseCatalog("[]"), /JSON object/);
});

test("rollback controls reject the active version and missing authority", () => {
  assert.equal(canRollback("version-2", "version-1"), true);
  assert.equal(canRollback("version-2", "version-2"), false);
  assert.equal(canRollback(null, "version-1"), false);
});

test("simulation client methods preserve tenant and session scope", async () => {
  const calls = [];
  const api = new ManagementApi(async (path, options) => {
    calls.push({ path, options });
    if (options?.method === "DELETE") return new Response(null, { status: 204 });
    return new Response(JSON.stringify({ schema_version: 1 }), {
      status: 200,
      headers: { "content-type": "application/json" },
    });
  });

  await api.startSimulation("tenant a", "session/1", "version-2");
  await api.sendSimulationTurn("tenant a", "session/1", "turn-1", "hello", true);
  await api.inspectSimulation("tenant a", "session/1");
  await api.resetSimulation("tenant a", "session/1");
  await api.closeSimulation("tenant a", "session/1");

  assert.equal(
    calls[0].path,
    "/v1/merchants/tenant%20a/simulations/session%2F1",
  );
  assert.deepEqual(JSON.parse(calls[0].options.body), {
    schema_version: 1,
    version_id: "version-2",
  });
  assert.equal(calls[1].path.endsWith("/turns"), true);
  assert.deepEqual(JSON.parse(calls[1].options.body), {
    schema_version: 1,
    request_id: "turn-1",
    text: "hello",
    readback_interrupted: true,
  });
  assert.equal(calls[2].path.endsWith("/state"), true);
  assert.equal(calls[3].path.endsWith("/reset"), true);
  assert.equal(calls[4].options.method, "DELETE");
});

test("simulation event projection exposes only caller-facing messages", () => {
  assert.deepEqual(
    simulationMessages([
      { kind: "token", text: "Your " },
      { kind: "token", text: "cart is empty." },
      { kind: "spoken_message", text: "I can help with that.", node: "frontline" },
      { kind: "interrupt", prompt: "Should I place the order?" },
      { kind: "future_internal_event", secret: "must not render" },
    ]),
    [
      { kind: "assistant", text: "Your cart is empty." },
      { kind: "assistant", text: "I can help with that." },
      { kind: "confirmation", text: "Should I place the order?" },
    ],
  );
});

test("simulation diagnostics expose only the closed operational fields", () => {
  assert.deepEqual(
    simulationDiagnostics({
      routing_records: [
        {
          event: "semantic_route",
          attributes: {
            decision: "direct",
            capability: "answer_question",
            provider_call_outcome: "completed",
            latency_ms: 1339.94,
            prompt: "must not render",
          },
        },
      ],
      operational_records: [
        {
          event: "turn_failed",
          attributes: {
            reason: "node_exception",
            node: "answer_response",
            action: "safe_abort",
            exception_message: "must not render",
          },
        },
      ],
    }),
    [
      "semantic_route | decision=direct | capability=answer_question | provider_call_outcome=completed | latency_ms=1340",
      "turn_failed | reason=node_exception | node=answer_response | action=safe_abort",
    ],
  );
});

test("changing merchants clears every tenant-bound workspace value", () => {
  const state = {
    merchantId: "acme_store",
    draftId: "working",
    draft: { tenant_id: "acme_store" },
    preview: { tenant_id: "acme_store" },
    validation: { tenant_id: "acme_store" },
    activeVersion: { tenant_id: "acme_store" },
    versions: [{ tenant_id: "acme_store" }],
    audit: [{ tenant_id: "acme_store" }],
    simulation: { tenant_id: "acme_store" },
    simulationState: { tenant_id: "acme_store" },
    simulationMessages: [{ kind: "caller", text: "hello" }],
    simulationDiagnostics: ["turn_failed | node=answer_response"],
    simulationLatency: { total_seconds: 0.1 },
    pendingSimulationTurn: { requestId: "turn-1" },
    merchants: [{ tenant_id: "acme_store" }, { tenant_id: "demo_shop" }],
    busy: false,
  };

  selectMerchantWorkspace(state, "demo_shop");

  assert.equal(state.merchantId, "demo_shop");
  assert.equal(state.draftId, "working");
  assert.deepEqual(state.merchants, [
    { tenant_id: "acme_store" },
    { tenant_id: "demo_shop" },
  ]);
  assert.equal(state.busy, false);
  for (const key of [
    "draft",
    "preview",
    "validation",
    "activeVersion",
    "simulation",
    "simulationState",
    "simulationLatency",
    "pendingSimulationTurn",
  ]) {
    assert.equal(state[key], null, key);
  }
  assert.deepEqual(state.versions, []);
  assert.deepEqual(state.audit, []);
  assert.deepEqual(state.simulationMessages, []);
  assert.deepEqual(state.simulationDiagnostics, []);
});

test("an unchanged simulation turn reuses its request id until completion", () => {
  let sequence = 0;
  const nextId = () => `simulation-turn-${++sequence}`;
  const input = {
    tenantId: "acme_store",
    simulationId: "simulation-1",
    text: "add one shirt",
    readbackInterrupted: false,
  };

  const first = simulationTurnRequest(null, input, nextId);
  const retry = simulationTurnRequest(first, input, nextId);
  const changed = simulationTurnRequest(first, { ...input, text: "add two shirts" }, nextId);

  assert.strictEqual(retry, first);
  assert.equal(retry.requestId, "simulation-turn-1");
  assert.equal(changed.requestId, "simulation-turn-2");
});

test("spoken text collapses whitespace and is bounded before it reaches the speech queue", () => {
  assert.equal(spokenText("  We have   trail\n running shoes. "), "We have trail running shoes.");
  assert.equal(spokenText(null), "");
  assert.equal(spokenText(undefined), "");

  const long = spokenText("a".repeat(2000));
  assert.equal(long.length, 1203);
  assert.ok(long.endsWith("..."));
});

test("speech support is reported per capability rather than as one on-or-off claim", () => {
  assert.match(supportSummary({ input: true, output: true }), /Microphone and playback/);
  assert.match(supportSummary({ input: false, output: true }), /Playback only/);
  assert.match(supportSummary({ input: true, output: false }), /Microphone only/);
  assert.match(supportSummary({ input: false, output: false }), /neither/);
});

test("a blocked speech backend is named, not reported as a generic failure", () => {
  // Brave and some Chromium builds expose webkitSpeechRecognition while stripping the service,
  // so the constructor check passes and the first start() is the only honest signal.
  assert.match(recognitionFailure("network"), /blocks the speech service/);
  assert.match(recognitionFailure("service-not-allowed"), /refused the speech service/);
  assert.match(recognitionFailure("not-allowed"), /permission was refused/);
  assert.match(recognitionFailure("audio-capture"), /No microphone/);

  // Transient outcomes must not latch the control off.
  assert.equal(recognitionFailure("no-speech"), null);
  assert.equal(recognitionFailure("aborted"), null);
});
