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
  CAPTURE_SAMPLE_RATE,
  concatSamples,
  describeEngines,
  resampleTo,
  supportSummary,
  toPcm16,
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

test("capture is converted to the exact PCM shape the STT engine accepts", () => {
  // Signed 16-bit little-endian mono. A float outside [-1, 1] must clamp rather than wrap,
  // which would turn a loud sample into the opposite polarity.
  const pcm = new DataView(toPcm16(Float32Array.from([0, 1, -1, 2, -2])));
  assert.equal(pcm.byteLength, 10);
  assert.equal(pcm.getInt16(0, true), 0);
  assert.equal(pcm.getInt16(2, true), 32767);
  assert.equal(pcm.getInt16(4, true), -32767);
  assert.equal(pcm.getInt16(6, true), 32767);
  assert.equal(pcm.getInt16(8, true), -32767);
});

test("capture is resampled only when the browser ignored the requested rate", () => {
  const samples = Float32Array.from([0, 0.25, 0.5, 0.75]);
  assert.equal(resampleTo(samples, CAPTURE_SAMPLE_RATE), samples);

  const halved = resampleTo(Float32Array.from([0, 1, 0, 1]), CAPTURE_SAMPLE_RATE * 2);
  assert.equal(halved.length, 2);

  assert.equal(resampleTo(new Float32Array(0), 48000).length, 0);
});

test("capture blocks join in arrival order", () => {
  const merged = concatSamples([Float32Array.from([1, 2]), Float32Array.from([3])]);
  assert.deepEqual([...merged], [1, 2, 3]);
});

test("microphone activity drives the visual level and resets when capture ends", async () => {
  let captureNode;
  fakeMedia({ onNode: (node) => { captureNode = node; } });
  const module = await import(
    `../../src/agnostic_market/management/ui/assets/voice.js?stub=capture-level`
  );
  const levels = [];
  const voice = module.createVoiceController({
    api: {},
    tenantId: () => "t",
    versionId: () => "v",
    onLevel: (level) => levels.push(level),
  });

  assert.equal(await voice.listen(), true);
  captureNode.port.onmessage({ data: Float32Array.from({ length: 160 }, () => 0.1) });
  assert.ok(levels.some((level) => level > 0));
  await voice.cancelListening();
  assert.equal(levels.at(-1), 0);
});

test("spoken playback drives the visual level without leaving a running animation", async () => {
  fakeMedia();
  const frames = new Map();
  let nextFrame = 0;
  globalThis.requestAnimationFrame = (callback) => {
    frames.set(++nextFrame, callback);
    return nextFrame;
  };
  globalThis.cancelAnimationFrame = (frame) => frames.delete(frame);
  globalThis.AudioContext = class {
    state = "running";
    destination = {};
    async decodeAudioData() { return {}; }
    createAnalyser() {
      return {
        fftSize: 32,
        connect() {},
        disconnect() {},
        getByteTimeDomainData(samples) {
          samples.fill(128);
          samples[0] = 255;
        },
      };
    }
    createBufferSource() {
      return { connect() {}, start() {}, stop() {} };
    }
  };
  const module = await import(
    `../../src/agnostic_market/management/ui/assets/voice.js?stub=playback-level`
  );
  const levels = [];
  const voice = module.createVoiceController({
    api: { synthesizeSpeech: async () => ({ arrayBuffer: async () => new ArrayBuffer(8) }) },
    tenantId: () => "t",
    versionId: () => "v",
    onLevel: (level) => levels.push(level),
  });

  assert.equal(await voice.speak("hello"), true);
  assert.ok(frames.size > 0);
  const [frame, callback] = frames.entries().next().value;
  frames.delete(frame);
  callback();
  assert.ok(levels.some((level) => level > 0));
  voice.stopSpeaking();
  assert.equal(frames.size, 0);
  assert.equal(levels.at(-1), 0);
  delete globalThis.requestAnimationFrame;
  delete globalThis.cancelAnimationFrame;
});

test("the panel names the engines a preview uses rather than leaving them to be inferred", () => {
  const described = describeEngines({
    tts_provider: "cartesia",
    tts_model: "sonic-3.5-2026-05-04",
    stt_provider: "deepgram",
    stt_model: "nova-3",
  });
  assert.match(described, /cartesia sonic-3\.5-2026-05-04/);
  assert.match(described, /deepgram nova-3/);
  assert.equal(describeEngines(null), "");
});

test("audio support is reported per capability because the halves fail independently", () => {
  assert.match(supportSummary({ input: true, output: true }), /Microphone and playback/);
  assert.match(supportSummary({ input: false, output: true }), /Playback only/);
  assert.match(supportSummary({ input: true, output: false }), /Capture only/);
  assert.match(supportSummary({ input: false, output: false }), /neither/);
});

function fakeMedia({ onNode } = {}) {
  const tracks = [];
  // Node exposes navigator as a getter-only property, so the stub is defined rather than assigned.
  Object.defineProperty(globalThis, "navigator", {
    configurable: true,
    value: {
      language: "en-US",
      mediaDevices: {
        async getUserMedia() {
          const track = { stopped: false, stop() { this.stopped = true; } };
          tracks.push(track);
          // Resolve on a later tick so a second call can start while this one is pending.
          await new Promise((resolve) => setTimeout(resolve, 5));
          return { getTracks: () => [track] };
        },
      },
    },
  });
  globalThis.AudioWorkletNode = class {
    constructor() {
      // Deliver one block once the handler is attached, so a capture has audio to transcribe.
      let handler = null;
      const port = {};
      Object.defineProperty(port, "onmessage", {
        get: () => handler,
        set(fn) {
          handler = fn;
          setTimeout(() => fn({ data: new Float32Array(128) }), 0);
        },
      });
      this.port = port;
      onNode?.(this);
    }
    disconnect() {}
  };
  globalThis.AudioContext = class {
    constructor() {
      this.sampleRate = CAPTURE_SAMPLE_RATE;
      this.state = "running";
      this.audioWorklet = { async addModule() {} };
    }
    createMediaStreamSource() {
      return { connect() {} };
    }
    async close() {}
    async resume() {}
  };
  return tracks;
}

test("hands-free capture waits for speech and sends once after a pause", async () => {
  let captureNode;
  fakeMedia({ onNode: (node) => { captureNode = node; } });
  const module = await import(
    `../../src/agnostic_market/management/ui/assets/voice.js?stub=hands-free`
  );
  const heard = [];
  let transcriptions = 0;
  const voice = module.createVoiceController({
    api: {
      transcribeCapture: async () => {
        transcriptions += 1;
        return { text: "show me jackets" };
      },
    },
    tenantId: () => "t",
    versionId: () => "v",
    onTranscript: (text) => heard.push(text),
  });

  assert.equal(await voice.listen({ autoSendOnSilence: true }), true);
  const quiet = new Float32Array(160);
  const speech = Float32Array.from({ length: 160 }, () => 0.1);
  for (let index = 0; index < 120; index += 1) captureNode.port.onmessage({ data: quiet });
  assert.equal(transcriptions, 0, "silence alone must not submit a turn");
  for (let index = 0; index < 15; index += 1) captureNode.port.onmessage({ data: speech });
  for (let index = 0; index < 100; index += 1) captureNode.port.onmessage({ data: quiet });
  await new Promise((resolve) => setTimeout(resolve, 0));

  assert.equal(transcriptions, 1);
  assert.deepEqual(heard, ["show me jackets"]);
  await voice.reset();
});

test("hands-free speech interrupts playback without mistaking low-level echo for speech", async () => {
  let captureNode;
  const tracks = fakeMedia({ onNode: (node) => { captureNode = node; } });
  let playback;
  globalThis.AudioContext.prototype.decodeAudioData = async () => ({});
  globalThis.AudioContext.prototype.createBufferSource = () => {
    playback = { stopped: false, connect() {}, start() {}, stop() { this.stopped = true; } };
    return playback;
  };
  globalThis.AudioContext.prototype.destination = {};
  const module = await import(
    `../../src/agnostic_market/management/ui/assets/voice.js?stub=spoken-barge-in`
  );
  const interrupted = [];
  const heard = [];
  const voice = module.createVoiceController({
    api: {
      synthesizeSpeech: async () => ({ arrayBuffer: async () => new ArrayBuffer(8) }),
      transcribeCapture: async () => ({ text: "another question" }),
    },
    tenantId: () => "t",
    versionId: () => "v",
    onBargeIn: () => interrupted.push(true),
    onTranscript: (text) => heard.push(text),
  });

  assert.equal(await voice.speak("the answer"), true);
  assert.equal(await voice.listen({ autoSendOnSilence: true, monitorPlayback: true }), true);
  assert.equal(voice.state, "barge-ready");
  assert.equal(playback.stopped, false);
  const echo = Float32Array.from({ length: 160 }, () => 0.02);
  for (let index = 0; index < 40; index += 1) captureNode.port.onmessage({ data: echo });
  assert.equal(playback.stopped, false);
  const speech = Float32Array.from({ length: 160 }, () => 0.12);
  for (let index = 0; index < 30; index += 1) captureNode.port.onmessage({ data: speech });
  assert.equal(playback.stopped, true);
  assert.equal(voice.state, "listening");
  assert.equal(interrupted.length, 1);
  for (let index = 0; index < 90; index += 1) {
    captureNode.port.onmessage({ data: new Float32Array(160) });
  }
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.deepEqual(heard, ["another question"]);
  await voice.reset();
  assert.ok(tracks.every((track) => track.stopped));
});

test("a completed reply continues on the already-open hands-free microphone", async () => {
  let captureNode;
  const tracks = fakeMedia({ onNode: (node) => { captureNode = node; } });
  let playback;
  globalThis.AudioContext.prototype.decodeAudioData = async () => ({});
  globalThis.AudioContext.prototype.createBufferSource = () => {
    playback = { connect() {}, start() {}, stop() {}, onended: null };
    return playback;
  };
  globalThis.AudioContext.prototype.destination = {};
  const module = await import(
    `../../src/agnostic_market/management/ui/assets/voice.js?stub=barge-complete`
  );
  const voice = module.createVoiceController({
    api: { synthesizeSpeech: async () => ({ arrayBuffer: async () => new ArrayBuffer(8) }) },
    tenantId: () => "t",
    versionId: () => "v",
  });

  await voice.speak("the answer");
  await voice.listen({ autoSendOnSilence: true, monitorPlayback: true });
  assert.equal(voice.state, "barge-ready");
  playback.onended();
  assert.equal(voice.state, "listening");
  assert.equal(tracks.length, 1);
  assert.ok(captureNode.port.onmessage);
  await voice.cancelListening();
  assert.ok(tracks[0].stopped);
});

test("playback completing during microphone permission leaves one live listener", async () => {
  const tracks = fakeMedia();
  let playback;
  globalThis.AudioContext.prototype.decodeAudioData = async () => ({});
  globalThis.AudioContext.prototype.createBufferSource = () => {
    playback = { connect() {}, start() {}, stop() {}, onended: null };
    return playback;
  };
  globalThis.AudioContext.prototype.destination = {};
  const module = await import(
    `../../src/agnostic_market/management/ui/assets/voice.js?stub=barge-permission-race`
  );
  const voice = module.createVoiceController({
    api: { synthesizeSpeech: async () => ({ arrayBuffer: async () => new ArrayBuffer(8) }) },
    tenantId: () => "t",
    versionId: () => "v",
  });

  await voice.speak("the answer");
  const listening = voice.listen({ autoSendOnSilence: true, monitorPlayback: true });
  playback.onended();
  assert.equal(await listening, true);
  assert.equal(voice.state, "listening");
  assert.equal(tracks.length, 1);
  await voice.reset();
  assert.ok(tracks[0].stopped);
});

test("stopping hands-free monitoring leaves current playback alone", async () => {
  const tracks = fakeMedia();
  let playback;
  globalThis.AudioContext.prototype.decodeAudioData = async () => ({});
  globalThis.AudioContext.prototype.createBufferSource = () => {
    playback = { stopped: false, connect() {}, start() {}, stop() { this.stopped = true; } };
    return playback;
  };
  globalThis.AudioContext.prototype.destination = {};
  const module = await import(
    `../../src/agnostic_market/management/ui/assets/voice.js?stub=barge-disarm`
  );
  const voice = module.createVoiceController({
    api: { synthesizeSpeech: async () => ({ arrayBuffer: async () => new ArrayBuffer(8) }) },
    tenantId: () => "t",
    versionId: () => "v",
  });

  await voice.speak("the answer");
  await voice.listen({ autoSendOnSilence: true, monitorPlayback: true });
  await voice.cancelListening();
  assert.equal(playback.stopped, false);
  assert.equal(voice.state, "speaking");
  assert.ok(tracks[0].stopped);
  voice.stopSpeaking();
});

test("manually sending a captured turn does not wait for silence", async () => {
  let captureNode;
  fakeMedia({ onNode: (node) => { captureNode = node; } });
  const module = await import(
    `../../src/agnostic_market/management/ui/assets/voice.js?stub=manual-send`
  );
  const voice = module.createVoiceController({
    api: { transcribeCapture: async () => ({ text: "two jackets" }) },
    tenantId: () => "t",
    versionId: () => "v",
  });

  assert.equal(await voice.listen(), true);
  captureNode.port.onmessage({ data: Float32Array.from({ length: 160 }, () => 0.1) });
  assert.equal(await voice.stopListening(), "two jackets");
});

test("stopping hands-free during transcription discards the pending turn", async () => {
  let captureNode;
  fakeMedia({ onNode: (node) => { captureNode = node; } });
  const module = await import(
    `../../src/agnostic_market/management/ui/assets/voice.js?stub=cancel-transcription`
  );
  let release;
  let started;
  const inFlight = new Promise((resolve) => { started = resolve; });
  const heard = [];
  const voice = module.createVoiceController({
    api: { transcribeCapture: () => new Promise((resolve) => {
      release = () => resolve({ text: "old turn" });
      started();
    }) },
    tenantId: () => "t",
    versionId: () => "v",
    onTranscript: (text) => heard.push(text),
  });

  assert.equal(await voice.listen({ autoSendOnSilence: true }), true);
  const speech = Float32Array.from({ length: 160 }, () => 0.1);
  const quiet = new Float32Array(160);
  for (let index = 0; index < 15; index += 1) captureNode.port.onmessage({ data: speech });
  for (let index = 0; index < 100; index += 1) captureNode.port.onmessage({ data: quiet });
  await inFlight;
  await voice.cancelListening();
  release();
  await new Promise((resolve) => setTimeout(resolve, 0));

  assert.deepEqual(heard, []);
  assert.equal(voice.state, "idle");
});

test("starting a new capture interrupts synthesis before old audio can play", async () => {
  fakeMedia();
  const module = await import(
    `../../src/agnostic_market/management/ui/assets/voice.js?stub=interrupt-synthesis`
  );
  let release;
  const voice = module.createVoiceController({
    api: { synthesizeSpeech: () => new Promise((resolve) => {
      release = () => resolve({ arrayBuffer: async () => new ArrayBuffer(8) });
    }) },
    tenantId: () => "t",
    versionId: () => "v",
  });

  const pending = voice.speak("old reply");
  assert.equal(voice.playbackActive, true);
  assert.equal(await voice.listen(), true);
  release();
  assert.equal(await pending, false);
  assert.equal(voice.state, "listening");
  assert.equal(voice.playbackActive, false);
  await voice.reset();
});

test("submitting a typed turn stops a reply already playing", async () => {
  fakeMedia();
  let source;
  globalThis.AudioContext = class {
    state = "running";
    destination = {};
    async decodeAudioData() { return {}; }
    createBufferSource() {
      source = { stopped: false, connect() {}, start() {}, stop() { this.stopped = true; } };
      return source;
    }
  };
  const module = await import(
    `../../src/agnostic_market/management/ui/assets/voice.js?stub=typed-interrupt`
  );
  const voice = module.createVoiceController({
    api: { synthesizeSpeech: async () => ({ arrayBuffer: async () => new ArrayBuffer(8) }) },
    tenantId: () => "t",
    versionId: () => "v",
  });

  assert.equal(await voice.speak("old reply"), true);
  await voice.think();
  assert.equal(source.stopped, true);
  assert.equal(voice.state, "thinking");
});

// captureSupport is frozen at module load, so the stubs must exist before the module is
// evaluated. A distinct query string gives a fresh instance rather than the cached one.
let instance = 0;
async function freshController() {
  const module = await import(
    `../../src/agnostic_market/management/ui/assets/voice.js?stub=${(instance += 1)}`
  );
  return module.createVoiceController({ api: {}, tenantId: () => "t", versionId: () => "v" });
}

test("a second listen while permission is pending cannot leak the first microphone", async () => {
  // Two quick clicks previously acquired two streams while the state was still idle; only the
  // last was retained, so the first track stayed live for the rest of the session.
  const tracks = fakeMedia();
  const voice = await freshController();

  const [first, second] = await Promise.all([voice.listen(), voice.listen()]);

  assert.equal(first, true);
  assert.equal(second, false, "the second call must be refused, not race for the device");
  assert.equal(tracks.length, 1, "only one stream should ever be acquired");

  await voice.reset();
  assert.ok(tracks.every((track) => track.stopped), "every acquired track must be stopped");
});

test("leaving the listening state for a typed turn stops the microphone", async () => {
  const tracks = fakeMedia();
  const voice = await freshController();
  assert.equal(await voice.listen(), true);

  // A typed turn calls think while capture is open. Previously the state left "listening",
  // stopListening then refused to run, and the track stayed live.
  await voice.think();

  assert.equal(voice.state, "thinking");
  assert.ok(tracks.every((track) => track.stopped), "think must tear capture down");
});

test("a transcript from a superseded capture never reaches the new session", async () => {
  // reset() bumps the capture token, but the awaited STT result was delivered without checking
  // it, so an utterance spoken into the closed session arrived as the next session's turn.
  const tracks = fakeMedia();
  let release = null;
  let started = null;
  const inFlight = new Promise((resolve) => {
    started = resolve;
  });
  const api = {
    transcribeCapture: () =>
      new Promise((resolve) => {
        release = () => resolve({ text: "cancel all my orders" });
        started();
      }),
  };
  const module = await import(
    `../../src/agnostic_market/management/ui/assets/voice.js?stub=transcript`
  );
  const heard = [];
  const voice = module.createVoiceController({
    api,
    tenantId: () => "t",
    versionId: () => "v",
    onTranscript: (text) => heard.push(text),
  });

  assert.equal(await voice.listen(), true);
  // The worklet delivers its block on a macrotask, so let it land before capture stops.
  await new Promise((resolve) => setTimeout(resolve, 10));
  const pending = voice.stopListening();
  await inFlight;
  await voice.reset();
  release();
  await pending;

  assert.deepEqual(heard, [], "a superseded transcript must be discarded");
  assert.ok(tracks.every((track) => track.stopped));
});

test("a superseded playback completion cannot orphan newer audio", async () => {
  // A stopped source still fires onended. Clearing the shared reference from that callback
  // left the newer source unreachable, so stopSpeaking could no longer stop it.
  const module = await import(
    `../../src/agnostic_market/management/ui/assets/voice.js?stub=playback`
  );
  const started = [];
  globalThis.AudioContext = class {
    constructor() {
      this.state = "running";
      this.destination = {};
    }
    async resume() {}
    async decodeAudioData() {
      return {};
    }
    createBufferSource() {
      const source = {
        buffer: null,
        onended: null,
        stopped: false,
        connect() {},
        start() {},
        stop() {
          this.stopped = true;
        },
      };
      started.push(source);
      return source;
    }
  };
  const api = { synthesizeSpeech: async () => ({ arrayBuffer: async () => new ArrayBuffer(8) }) };
  const voice = module.createVoiceController({ api, tenantId: () => "t", versionId: () => "v" });

  await voice.speak("first reply");
  const first = started[0];
  await voice.speak("second reply");
  const second = started[1];

  // The first source was stopped by the second speak; its late completion must be ignored.
  first.onended?.();
  voice.stopSpeaking();

  assert.equal(second.stopped, true, "the newer source must still be stoppable");
});

test("stopping capture twice during teardown preserves the first transcript", async () => {
  fakeMedia();
  let releaseClose;
  const closing = new Promise((resolve) => { releaseClose = resolve; });
  globalThis.AudioContext.prototype.close = () => closing;
  const module = await import(
    `../../src/agnostic_market/management/ui/assets/voice.js?stub=double-stop`
  );
  const heard = [];
  const voice = module.createVoiceController({
    api: { transcribeCapture: async () => ({ text: "hello" }) },
    tenantId: () => "t",
    versionId: () => "v",
    onTranscript: (text) => heard.push(text),
  });

  assert.equal(await voice.listen(), true);
  await new Promise((resolve) => setTimeout(resolve, 10));
  const first = voice.stopListening();
  const second = voice.stopListening();
  releaseClose();

  assert.deepEqual(await Promise.all([first, second]), ["hello", ""]);
  assert.deepEqual(heard, ["hello"]);
});

test("a superseded decode failure cannot change newer playback state", async () => {
  let rejectOld;
  let decodeStarted;
  const decoding = new Promise((resolve) => { decodeStarted = resolve; });
  let decodeCount = 0;
  globalThis.AudioContext = class {
    state = "running";
    destination = {};
    decodeAudioData() {
      decodeCount += 1;
      if (decodeCount === 1) {
        return new Promise((resolve, reject) => {
          rejectOld = reject;
          decodeStarted();
        });
      }
      return Promise.resolve({});
    }
    createBufferSource() {
      return { connect() {}, start() {}, stop() {} };
    }
  };
  const module = await import(
    `../../src/agnostic_market/management/ui/assets/voice.js?stub=stale-decode`
  );
  const errors = [];
  const voice = module.createVoiceController({
    api: { synthesizeSpeech: async () => ({ arrayBuffer: async () => new ArrayBuffer(8) }) },
    tenantId: () => "t",
    versionId: () => "v",
    onError: (message) => errors.push(message),
  });

  const first = voice.speak("first reply");
  await decoding;
  await voice.speak("second reply");
  rejectOld(new Error("old decode failed"));
  await first;

  assert.equal(voice.state, "speaking");
  assert.deepEqual(errors, []);
});

test("a superseded synthesis failure cannot report an error over newer playback", async () => {
  let rejectOld;
  let calls = 0;
  const module = await import(
    `../../src/agnostic_market/management/ui/assets/voice.js?stub=stale-synthesis`
  );
  const errors = [];
  const voice = module.createVoiceController({
    api: {
      synthesizeSpeech: () => {
        calls += 1;
        if (calls === 1) return new Promise((resolve, reject) => { rejectOld = reject; });
        return Promise.resolve({ arrayBuffer: async () => new ArrayBuffer(8) });
      },
    },
    tenantId: () => "t",
    versionId: () => "v",
    onError: (message) => errors.push(message),
  });

  const first = voice.speak("first reply");
  await voice.speak("second reply");
  rejectOld(new Error("old synthesis failed"));
  await first;

  assert.equal(voice.state, "speaking");
  assert.deepEqual(errors, []);
});
