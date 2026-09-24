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
} from "./client.js";
import {
  captureSupport,
  createVoiceController,
  describeEngines,
  supportSummary,
} from "./voice.js";

const api = new ManagementApi();

const state = {
  merchantId: "",
  draftId: "working",
  draft: null,
  preview: null,
  validation: null,
  activeVersion: null,
  versions: [],
  audit: [],
  merchants: [],
  simulation: null,
  simulationState: null,
  simulationMessages: [],
  simulationDiagnostics: [],
  simulationLatency: null,
  pendingSimulationTurn: null,
  busy: false,
  voiceMode: "manual",
  handsFreeArmed: false,
  voiceAwaitingReply: false,
};

const elements = Object.fromEntries(
  [
    "merchant-select",
    "draft-id",
    "refresh-merchants",
    "load-draft",
    "seed-draft",
    "merchant-identity",
    "draft-status",
    "publication-status",
    "conflict-banner",
    "reload-conflict",
    "global-status",
    "global-error",
    "empty-state",
    "workspace",
    "section-nav",
    "overview-simulator-link",
    "metric-revision",
    "metric-products",
    "metric-versions",
    "merchant-override",
    "save-configuration",
    "catalog-json",
    "catalog-table",
    "save-catalog",
    "validate-draft",
    "preview-draft",
    "publish-draft",
    "validation-result",
    "preview-fingerprint",
    "config-fingerprint",
    "fixture-fingerprint",
    "simulation-status",
    "simulation-id",
    "simulation-version",
    "start-simulation",
    "reset-simulation",
    "close-simulation",
    "simulation-transcript",
    "simulation-turn",
    "readback-interrupted",
    "send-simulation-turn",
    "simulation-publication",
    "simulation-turn-count",
    "simulation-revision",
    "simulation-cart-total",
    "simulation-receipts",
    "simulation-latency",
    "simulation-cart",
    "simulation-diagnostics",
    "refresh-history",
    "version-list",
    "base-version",
    "target-version",
    "compare-versions",
    "version-diff",
    "audit-list",
    "confirmation-dialog",
    "confirmation-title",
    "confirmation-copy",
    "voice-orb",
    "voice-state-label",
    "voice-support",
    "voice-listen",
    "voice-send-capture",
    "voice-mode-manual",
    "voice-mode-auto",
    "voice-speak-replies",
  ].map((id) => [id, document.getElementById(id)]),
);

const VOICE_STATE_LABELS = {
  idle: "Idle",
  arming: "Starting microphone",
  "barge-ready": "Listening while speaking",
  listening: "Listening",
  thinking: "Working",
  speaking: "Speaking",
  acknowledged: "Received",
};

const voice = createVoiceController({
  api,
  tenantId: () => state.merchantId,
  // The version the simulation pinned, so speech matches the publication under test.
  versionId: () => state.simulation?.publication_version_id ?? "",
  onState(next) {
    const orb = elements["voice-orb"];
    const previous = orb.dataset.state;
    orb.dataset.state = next;
    const label = VOICE_STATE_LABELS[next] ?? "Idle";
    orb.setAttribute("aria-label", `Voice assistant state: ${label.toLowerCase()}`);
    elements["voice-state-label"].textContent = label;
    syncVoiceAvailability();
    if (next === "speaking") queueHandsFreeMonitor();
    if (next === "listening" && previous === "barge-ready") {
      state.voiceAwaitingReply = false;
    }
    if (next === "acknowledged" || next === "idle") {
      if (next === "acknowledged") state.voiceAwaitingReply = false;
      queueHandsFreeListen();
    }
  },
  onLevel(level) {
    const orb = elements["voice-orb"];
    orb.style.setProperty("--voice-scale", (1 + level * 0.2).toFixed(3));
    orb.style.setProperty("--voice-glow", (0.32 + level * 0.5).toFixed(3));
    orb.style.setProperty("--voice-wave-amplitude", (0.35 + level * 0.9).toFixed(3));
  },
  onBargeIn() {
    noteInterruptedReadback();
    state.voiceAwaitingReply = false;
  },
  onTranscript(heard) {
    // Heard speech fills the same composer the text path uses, so one send path stays authoritative.
    elements["simulation-turn"].value = heard;
    announce(`Heard: ${heard}`);
    if (state.simulation && !state.busy) void sendSimulationTurn();
  },
  onError(message, category) {
    // The announce channel is sr-only, so a sighted operator saw nothing when the microphone
    // failed. Voice problems belong on the voice panel, where the control that failed lives.
    elements["voice-support"].textContent = message;
    elements["voice-support"].classList.add("voice-support-alert");
    announce(message);
    if (category === "capture" || category === "transcription") state.handsFreeArmed = false;
    syncVoiceAvailability();
  },
});

function speakLatestReply(replies) {
  if (!elements["voice-speak-replies"].checked) {
    voice.acknowledge();
    return;
  }
  // Only the assistant's own lines are spoken; caller echoes would read the operator back to itself.
  const spoken = [...replies]
    .reverse()
    .find((message) => message.kind !== "caller");
  void voice.speak(spoken?.text ?? "");
  syncVoiceAvailability();
}

function noteInterruptedReadback() {
  if (state.simulationMessages.at(-1)?.kind === "confirmation") {
    elements["readback-interrupted"].checked = true;
  }
}

function syncVoiceAvailability() {
  const active = Boolean(state.simulation);
  const automatic = state.voiceMode === "auto";
  const capturing = voice.acquiring || voice.state === "listening" ||
    voice.state === "barge-ready";
  elements["voice-listen"].disabled =
    state.busy || !active || !captureSupport.input || voice.inputBlocked;
  if (automatic) {
    elements["voice-listen"].textContent = state.handsFreeArmed
      ? "Stop hands-free"
      : "Start hands-free";
  } else {
    elements["voice-listen"].textContent = voice.acquiring
      ? "Cancel microphone request"
      : voice.state === "listening" ? "Cancel listening" : "Start listening";
  }
  elements["voice-listen"].setAttribute(
    "aria-pressed",
    String(automatic ? state.handsFreeArmed : capturing),
  );
  elements["voice-send-capture"].hidden = automatic;
  elements["voice-send-capture"].disabled = state.busy || voice.state !== "listening";
  elements["voice-mode-manual"].disabled = state.busy;
  elements["voice-mode-auto"].disabled = state.busy;
  elements["voice-speak-replies"].disabled = !captureSupport.output;
  if (!captureSupport.output) elements["voice-speak-replies"].checked = false;
}

function queueHandsFreeListen() {
  if (!state.handsFreeArmed || state.voiceAwaitingReply || state.busy || !state.simulation) return;
  queueMicrotask(() => {
    if (!state.handsFreeArmed || state.voiceAwaitingReply || state.busy || !state.simulation) return;
    if (voice.state !== "idle" && voice.state !== "acknowledged") return;
    void voice.listen({ autoSendOnSilence: true });
  });
}

function queueHandsFreeMonitor() {
  if (!state.handsFreeArmed || state.busy || !state.simulation) return;
  queueMicrotask(() => {
    if (state.handsFreeArmed && !state.busy && state.simulation &&
        voice.state === "speaking" && !voice.acquiring) {
      void voice.listen({ autoSendOnSilence: true, monitorPlayback: true });
    }
  });
}

function requestId(prefix) {
  return `${prefix}-${crypto.randomUUID()}`;
}

function announce(message) {
  elements["global-status"].textContent = message;
}

function clearError() {
  elements["global-error"].hidden = true;
  elements["global-error"].textContent = "";
}

function showError(error) {
  const messages = {
    invalid_request: "The request is incomplete or contains an invalid value.",
    not_found: "The requested managed record does not exist.",
    scope_violation: "The request crossed its selected merchant scope.",
    state_conflict: "The stored state changed. Reload before saving again.",
    replay_conflict: "This request identifier was already used for different work.",
    draft_invalid: "The draft did not pass validation.",
    service_unavailable: "The management service could not complete the request.",
  };
  const code = error instanceof ManagementApiError ? error.code : "service_unavailable";
  elements["global-error"].textContent = messages[code] ?? messages.service_unavailable;
  elements["global-error"].hidden = false;
  elements["conflict-banner"].hidden = !["state_conflict", "replay_conflict"].includes(code);
  if (error instanceof ManagementApiError && error.validation) {
    state.validation = error.validation;
    renderValidation();
  }
}

function syncButtonStates() {
  for (const button of document.querySelectorAll("button")) {
    if (!button.closest("dialog")) button.disabled = state.busy;
  }
  elements["publish-draft"].disabled = state.busy || !state.preview;
  elements["compare-versions"].disabled = state.busy || state.versions.length < 1;
  const simulationActive = Boolean(state.simulation);
  elements["start-simulation"].disabled =
    state.busy || simulationActive || state.versions.length < 1;
  elements["reset-simulation"].disabled = state.busy || !simulationActive;
  elements["close-simulation"].disabled = state.busy || !simulationActive;
  elements["send-simulation-turn"].disabled = state.busy || !simulationActive;
  elements["simulation-turn"].disabled = state.busy || !simulationActive;
  elements["readback-interrupted"].disabled = state.busy || !simulationActive;
  elements["simulation-id"].disabled = state.busy || simulationActive;
  elements["simulation-version"].disabled = state.busy || simulationActive;
  elements["merchant-select"].disabled = state.busy || simulationActive;
  elements["draft-id"].disabled = state.busy || simulationActive;
  syncVoiceAvailability();
  for (const button of document.querySelectorAll("[data-rollback-version]")) {
    button.disabled =
      state.busy || !canRollback(state.activeVersion?.version_id, button.dataset.rollbackVersion);
  }
}

function setBusy(busy, message = "") {
  state.busy = busy;
  document.body.dataset.busy = String(busy);
  syncButtonStates();
  if (message) announce(message);
}

async function runAction(message, action) {
  clearError();
  setBusy(true, message);
  try {
    return await action();
  } catch (error) {
    showError(error);
    return null;
  } finally {
    setBusy(false);
  }
}

function option(value, label) {
  const node = document.createElement("option");
  node.value = value;
  node.textContent = label;
  return node;
}

function replaceOptions(select, options, selected = "") {
  select.replaceChildren(...options);
  if (selected) select.value = selected;
}

function formatFingerprint(value) {
  return value ? `${value.slice(0, 12)}...${value.slice(-8)}` : "-";
}

function formatTimestamp(value) {
  return value ? new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }).format(new Date(value)) : "-";
}

function renderMerchants() {
  const choices = state.merchants.map((merchant) => {
    const status = merchant.managed ? "managed" : "configured";
    return option(merchant.tenant_id, `${merchant.tenant_id} | ${status}`);
  });
  if (!choices.length) choices.push(option("", "No configured merchants"));
  replaceOptions(elements["merchant-select"], choices, state.merchantId);
}

function renderCatalog() {
  const products = state.draft?.fixtures?.catalog?.products ?? [];
  elements["catalog-table"].replaceChildren(
    ...products.map((product) => {
      const row = document.createElement("tr");
      for (const value of [product.sku, product.name, String(product.price_usd)]) {
        const cell = document.createElement("td");
        cell.textContent = value;
        row.append(cell);
      }
      return row;
    }),
  );
  elements["catalog-json"].value = JSON.stringify(
    state.draft?.fixtures?.catalog ?? { products: [] },
    null,
    2,
  );
}

function renderValidation() {
  const container = elements["validation-result"];
  container.replaceChildren();
  if (!state.validation) {
    container.className = "result-empty";
    container.textContent = "Not validated yet.";
    return;
  }
  if (!state.validation.findings.length) {
    container.className = "result-pass";
    container.textContent = `Passed revision ${state.validation.draft_revision}.`;
    return;
  }
  container.className = "finding-list";
  const list = document.createElement("ul");
  for (const finding of state.validation.findings) {
    const item = document.createElement("li");
    item.textContent = `${finding.severity}: ${finding.code} at ${finding.path.join(".")}`;
    list.append(item);
  }
  container.append(list);
}

function renderPreview() {
  elements["preview-fingerprint"].textContent = formatFingerprint(state.preview?.preview_fingerprint);
  elements["config-fingerprint"].textContent = formatFingerprint(state.preview?.config_fingerprint);
  elements["fixture-fingerprint"].textContent = formatFingerprint(state.preview?.fixture_fingerprint);
  elements["publish-draft"].disabled = state.busy || !state.preview;
}

function renderVersions() {
  const versionList = elements["version-list"];
  versionList.replaceChildren();
  if (!state.versions.length) {
    versionList.className = "result-empty";
    versionList.textContent = "No published versions.";
  } else {
    versionList.className = "version-stack";
    for (const version of state.versions) {
      const article = document.createElement("article");
      const heading = document.createElement("strong");
      heading.textContent = `Version ${version.version_number}`;
      const meta = document.createElement("span");
      meta.textContent = `${version.version_id} | ${formatTimestamp(version.published_at)}`;
      const rollback = document.createElement("button");
      rollback.type = "button";
      rollback.className = "text-button";
      rollback.textContent = "Roll back to this version";
      rollback.dataset.rollbackVersion = version.version_id;
      rollback.addEventListener("click", () => rollbackTo(version));
      article.append(heading, meta, rollback);
      versionList.append(article);
    }
  }
  const choices = state.versions.map((version) =>
    option(version.version_id, `v${version.version_number} | ${version.version_id}`),
  );
  replaceOptions(elements["base-version"], choices.map((item) => item.cloneNode(true)));
  replaceOptions(elements["target-version"], choices, state.activeVersion?.version_id ?? "");
  replaceOptions(
    elements["simulation-version"],
    state.versions.map((version) =>
      option(version.version_id, `v${version.version_number} - ${version.version_id}`),
    ),
    state.activeVersion?.version_id ?? state.versions[0]?.version_id ?? "",
  );
  elements["compare-versions"].disabled = state.busy || state.versions.length < 1;
  syncButtonStates();
}

function renderSimulation() {
  const active = Boolean(state.simulation);
  elements["simulation-status"].textContent = active ? "Running" : "Stopped";
  elements["simulation-status"].className = `pill ${active ? "published" : "neutral"}`;

  const transcript = elements["simulation-transcript"];
  transcript.replaceChildren();
  if (!state.simulationMessages.length) {
    const item = document.createElement("li");
    item.className = "result-empty";
    item.textContent = active ? "Send a caller turn." : "Start a simulation to send a turn.";
    transcript.append(item);
  } else {
    for (const message of state.simulationMessages) {
      const item = document.createElement("li");
      item.className = message.kind;
      item.textContent = message.text;
      transcript.append(item);
    }
    transcript.scrollTop = transcript.scrollHeight;
  }

  const projection = state.simulationState;
  elements["simulation-publication"].textContent = projection?.publication_version_id ?? "-";
  elements["simulation-turn-count"].textContent = String(projection?.turn_count ?? 0);
  elements["simulation-revision"].textContent = String(projection?.session_revision ?? 0);
  elements["simulation-cart-total"].textContent = `$${Number(projection?.cart_total_usd ?? 0).toFixed(2)}`;
  const receipts = projection?.committed_receipts;
  const orderReceipts = receipts
    ? Object.values(receipts.orders).reduce((total, count) => total + count, 0)
    : 0;
  elements["simulation-receipts"].textContent = receipts
    ? `cart ${receipts.cart.mutations}, orders ${orderReceipts}, profiles ${receipts.profiles.changes}`
    : "cart 0, orders 0, profiles 0";
  elements["simulation-latency"].textContent = state.simulationLatency
    ? `${Math.round(state.simulationLatency.total_seconds * 1000)} ms`
    : "-";

  const cart = elements["simulation-cart"];
  cart.replaceChildren();
  for (const line of projection?.cart_lines ?? []) {
    const item = document.createElement("li");
    item.textContent = `${line.quantity} x ${line.name} - $${Number(line.line_total).toFixed(2)}`;
    cart.append(item);
  }
  if (!cart.children.length) {
    const item = document.createElement("li");
    item.className = "result-empty";
    item.textContent = "Cart is empty.";
    cart.append(item);
  }

  const diagnostics = elements["simulation-diagnostics"];
  diagnostics.replaceChildren();
  for (const line of state.simulationDiagnostics) {
    const item = document.createElement("li");
    item.textContent = line;
    diagnostics.append(item);
  }
  if (!diagnostics.children.length) {
    const item = document.createElement("li");
    item.className = "result-empty";
    item.textContent = "No turn diagnostics yet.";
    diagnostics.append(item);
  }
  syncButtonStates();
}

function renderAudit() {
  elements["audit-list"].replaceChildren(
    ...state.audit.map((record) => {
      const item = document.createElement("li");
      const title = document.createElement("strong");
      title.textContent = record.event.replaceAll("_", " ");
      const meta = document.createElement("span");
      meta.textContent = `${formatTimestamp(record.occurred_at)} | ${record.actor_id}`;
      item.append(title, meta);
      return item;
    }),
  );
  if (!state.audit.length) {
    const item = document.createElement("li");
    item.className = "result-empty";
    item.textContent = "No audit events.";
    elements["audit-list"].append(item);
  }
}

function renderWorkspace() {
  const hasDraft = Boolean(state.draft);
  elements["empty-state"].hidden = hasDraft;
  elements.workspace.hidden = !hasDraft;
  elements["overview-simulator-link"].hidden = !hasDraft;
  for (const link of elements["section-nav"].querySelectorAll("[data-needs-draft]")) {
    link.hidden = !hasDraft;
  }
  scheduleNavigationUpdate();
  elements["merchant-identity"].textContent = state.merchantId || "No merchant selected";
  elements["draft-status"].textContent = hasDraft ? `Draft r${state.draft.revision}` : "No draft loaded";
  elements["draft-status"].className = `pill ${hasDraft ? "active" : "neutral"}`;
  elements["publication-status"].textContent = state.activeVersion
    ? `Published v${state.activeVersion.version_number}`
    : "No publication";
  elements["publication-status"].className = `pill ${state.activeVersion ? "published" : "neutral"}`;
  elements["metric-revision"].textContent = hasDraft ? String(state.draft.revision) : "-";
  elements["metric-products"].textContent = hasDraft
    ? String(state.draft.fixtures.catalog.products.length)
    : "-";
  elements["metric-versions"].textContent = String(state.versions.length);
  if (hasDraft) {
    elements["merchant-override"].value = JSON.stringify(state.draft.merchant_override, null, 2);
    renderCatalog();
  }
  renderValidation();
  renderPreview();
  renderVersions();
  renderAudit();
  renderSimulation();
}

async function refreshMerchants() {
  const response = await runAction("Refreshing merchants", () => api.listMerchants());
  if (!response) return;
  state.merchants = response.merchants;
  if (!state.merchantId && state.merchants.length) {
    state.merchantId = state.merchants[0].tenant_id;
  }
  renderMerchants();
  renderWorkspace();
  announce(`Loaded ${state.merchants.length} merchants.`);
}

async function loadHistory() {
  if (!state.merchantId) return;
  const [versions, audit, active] = await Promise.all([
    api.listVersions(state.merchantId),
    api.listAudit(state.merchantId),
    api.getActiveVersion(state.merchantId).catch((error) => {
      if (error instanceof ManagementApiError && error.code === "not_found") return null;
      throw error;
    }),
  ]);
  state.versions = versions.versions;
  state.audit = audit.records;
  state.activeVersion = active;
}

async function loadWorkspace() {
  state.merchantId = elements["merchant-select"].value;
  state.draftId = elements["draft-id"].value.trim();
  if (!state.merchantId || !state.draftId) {
    showError(new ManagementApiError(422, "invalid_request"));
    return;
  }
  const loaded = await runAction("Loading merchant workspace", async () => {
    const draft = await api.getDraft(state.merchantId, state.draftId).catch((error) => {
      if (error instanceof ManagementApiError && error.code === "not_found") return null;
      throw error;
    });
    await loadHistory();
    return draft;
  });
  state.draft = loaded;
  state.preview = null;
  state.validation = null;
  elements["conflict-banner"].hidden = true;
  renderWorkspace();
  announce(loaded ? `Loaded draft revision ${loaded.revision}.` : "No stored draft was found.");
}

async function seedDraft() {
  state.merchantId = elements["merchant-select"].value;
  state.draftId = elements["draft-id"].value.trim();
  if (!state.merchantId || !state.draftId) {
    showError(new ManagementApiError(422, "invalid_request"));
    return;
  }
  const draft = await runAction("Creating the first draft", () =>
    api.seedDraft(state.merchantId, state.draftId, requestId("seed")),
  );
  if (!draft) return;
  state.draft = draft;
  state.preview = null;
  state.validation = null;
  await refreshMerchants();
  renderWorkspace();
  announce("Created draft revision 1 from validated development configuration.");
}

async function saveConfiguration() {
  if (!state.draft) return;
  let merchantOverride;
  try {
    merchantOverride = JSON.parse(elements["merchant-override"].value);
  } catch {
    showError(new ManagementApiError(422, "invalid_request"));
    return;
  }
  const update = buildDraftUpdate(state.draft, merchantOverride, requestId("draft"));
  const draft = await runAction("Saving the next draft revision", () =>
    api.saveDraft(state.merchantId, state.draftId, update.expectedRevision, update.payload),
  );
  if (!draft) return;
  state.draft = draft;
  state.preview = null;
  state.validation = null;
  renderWorkspace();
  announce(`Saved draft revision ${draft.revision}.`);
}

async function saveCatalog() {
  if (!state.draft) return;
  let catalog;
  try {
    catalog = parseCatalog(elements["catalog-json"].value);
  } catch {
    showError(new ManagementApiError(422, "invalid_request"));
    return;
  }
  const draft = await runAction("Importing the catalog", () =>
    api.importCatalog(
      state.merchantId,
      state.draftId,
      state.draft.revision,
      requestId("catalog"),
      catalog,
    ),
  );
  if (!draft) return;
  state.draft = draft;
  state.preview = null;
  state.validation = null;
  renderWorkspace();
  announce(`Imported catalog into draft revision ${draft.revision}.`);
}

async function validateDraft() {
  if (!state.draft) return;
  const validation = await runAction("Validating the draft", () =>
    api.validateDraft(
      state.merchantId,
      state.draftId,
      state.draft.revision,
      requestId("validate"),
    ),
  );
  if (!validation) return;
  state.validation = validation;
  renderValidation();
  announce(`Validated draft revision ${state.draft.revision}.`);
}

async function previewDraft() {
  if (!state.draft) return;
  state.preview = await runAction("Resolving the publication preview", () =>
    api.previewDraft(state.merchantId, state.draftId, state.draft.revision),
  );
  renderPreview();
  if (state.preview) announce("Resolved and fingerprinted the current draft.");
}

async function publishDraft() {
  if (!state.draft || !state.preview) return;
  const receipt = await runAction("Publishing the approved preview", () =>
    api.publishDraft(
      state.merchantId,
      state.draftId,
      state.draft.revision,
      state.preview.preview_fingerprint,
      state.activeVersion?.version_id ?? null,
      requestId("publish"),
    ),
  );
  if (!receipt) return;
  await runAction("Refreshing immutable history", loadHistory);
  renderWorkspace();
  announce(`Published version ${receipt.version_number}.`);
}

async function compareVersions() {
  const base = elements["base-version"].value;
  const target = elements["target-version"].value;
  if (!base || !target) return;
  const difference = await runAction("Comparing immutable versions", () =>
    api.compareVersions(state.merchantId, base, target),
  );
  if (!difference) return;
  const changes = [
    ...difference.config_changes,
    ...difference.fixture_changes,
    ...difference.dataset_changes,
  ];
  elements["version-diff"].className = changes.length ? "finding-list" : "result-pass";
  elements["version-diff"].textContent = changes.length
    ? changes.map((change) => `${change.kind}: ${change.path.join(".")}`).join("\n")
    : "No differences.";
}

async function startSimulation() {
  const simulationId = elements["simulation-id"].value.trim();
  const versionId = elements["simulation-version"].value;
  if (!state.merchantId || !simulationId || !versionId) {
    showError(new ManagementApiError(422, "invalid_request"));
    return;
  }
  const opened = await runAction("Starting the isolated simulation", async () => {
    try {
      return {
        status: await api.startSimulation(state.merchantId, simulationId, versionId),
        resumed: false,
      };
    } catch (error) {
      if (!(error instanceof ManagementApiError) || error.code !== "state_conflict") throw error;
      const existing = await api.getSimulation(state.merchantId, simulationId);
      if (existing.publication_version_id !== versionId) throw error;
      return { status: existing, resumed: true };
    }
  });
  if (!opened) return;
  state.simulation = opened.status;
  state.simulationState = await runAction("Inspecting simulation state", () =>
    api.inspectSimulation(state.merchantId, simulationId),
  );
  state.simulationMessages = [];
  state.simulationDiagnostics = [];
  state.simulationLatency = null;
  state.pendingSimulationTurn = null;
  renderSimulation();
  // The bound voice belongs to the pinned publication, so it is resolved once one exists.
  await refreshVoiceIdentity();
  announce(
    `${opened.resumed ? "Resumed" : "Started"} simulation on ${opened.status.publication_version_id}.`,
  );
}

async function sendSimulationTurn() {
  if (!state.simulation) return;
  const text = elements["simulation-turn"].value.trim();
  if (!text) {
    showError(new ManagementApiError(422, "invalid_request"));
    return;
  }
  if (voice.playbackActive) noteInterruptedReadback();
  const turn = simulationTurnRequest(
    state.pendingSimulationTurn,
    {
      tenantId: state.merchantId,
      simulationId: state.simulation.simulation_id,
      text,
      readbackInterrupted: elements["readback-interrupted"].checked,
    },
    () => requestId("simulation-turn"),
  );
  state.pendingSimulationTurn = turn;
  state.voiceAwaitingReply = state.handsFreeArmed;
  void voice.think();
  const result = await runAction("Running the caller turn", () =>
    api.sendSimulationTurn(
      turn.tenantId,
      turn.simulationId,
      turn.requestId,
      turn.text,
      turn.readbackInterrupted,
    ),
  );
  if (!result) {
    state.handsFreeArmed = false;
    state.voiceAwaitingReply = false;
    void voice.reset();
    return;
  }
  state.pendingSimulationTurn = null;
  const replies = simulationMessages(result.events);
  state.simulationMessages.push(
    { kind: "caller", text },
    ...replies,
  );
  state.simulationDiagnostics = simulationDiagnostics(result);
  state.simulationState = result.state;
  state.simulationLatency = result.latency.at(-1) ?? null;
  elements["simulation-turn"].value = "";
  elements["readback-interrupted"].checked = false;
  renderSimulation();
  speakLatestReply(replies);
  announce(`Completed simulation turn ${result.turn_number}.`);
}

async function resetSimulation() {
  if (!state.simulation) return;
  const reset = await runAction("Resetting the isolated simulation", async () => {
    state.handsFreeArmed = false;
    state.voiceAwaitingReply = false;
    // Stop local capture even if the remote reset is slow or fails.
    await voice.reset();
    return api.resetSimulation(state.merchantId, state.simulation.simulation_id);
  });
  if (!reset) return;
  state.simulation = reset;
  state.simulationState = await runAction("Inspecting reset state", () =>
    api.inspectSimulation(state.merchantId, reset.simulation_id),
  );
  state.simulationMessages = [];
  state.simulationDiagnostics = [];
  state.simulationLatency = null;
  state.pendingSimulationTurn = null;
  renderSimulation();
  announce(`Reset simulation on ${reset.publication_version_id}.`);
}

async function closeSimulation() {
  if (!state.simulation) return;
  const simulationId = state.simulation.simulation_id;
  const closed = await runAction("Closing the isolated simulation", async () => {
    state.handsFreeArmed = false;
    state.voiceAwaitingReply = false;
    await voice.reset();
    await api.closeSimulation(state.merchantId, simulationId);
    return true;
  });
  if (!closed) return;
  state.simulation = null;
  state.simulationState = null;
  state.simulationMessages = [];
  state.simulationDiagnostics = [];
  state.simulationLatency = null;
  state.pendingSimulationTurn = null;
  renderSimulation();
  await refreshVoiceIdentity();
  announce("Closed the isolated simulation.");
}

function confirmOperation(title, copy) {
  const dialog = elements["confirmation-dialog"];
  elements["confirmation-title"].textContent = title;
  elements["confirmation-copy"].textContent = copy;
  dialog.showModal();
  return new Promise((resolve) => {
    dialog.addEventListener("close", () => resolve(dialog.returnValue === "confirm"), {
      once: true,
    });
  });
}

async function rollbackTo(version) {
  if (!state.activeVersion) return;
  const confirmed = await confirmOperation(
    `Roll back to version ${version.version_number}?`,
    "Rollback publishes a new immutable version. It does not rewrite or delete history.",
  );
  if (!confirmed) return;
  const receipt = await runAction("Publishing the rollback", () =>
    api.rollback(
      state.merchantId,
      state.activeVersion.version_id,
      version.version_id,
      requestId("rollback"),
    ),
  );
  if (!receipt) return;
  await runAction("Refreshing immutable history", loadHistory);
  renderWorkspace();
  announce(`Published rollback version ${receipt.version_number}.`);
}

elements["refresh-merchants"].addEventListener("click", refreshMerchants);
elements["merchant-select"].addEventListener("change", () => {
  selectMerchantWorkspace(state, elements["merchant-select"].value);
  renderWorkspace();
});
elements["load-draft"].addEventListener("click", loadWorkspace);
elements["seed-draft"].addEventListener("click", seedDraft);
elements["reload-conflict"].addEventListener("click", loadWorkspace);
elements["save-configuration"].addEventListener("click", saveConfiguration);
elements["save-catalog"].addEventListener("click", saveCatalog);
elements["validate-draft"].addEventListener("click", validateDraft);
elements["preview-draft"].addEventListener("click", previewDraft);
elements["publish-draft"].addEventListener("click", publishDraft);
elements["refresh-history"].addEventListener("click", async () => {
  await runAction("Refreshing immutable history", loadHistory);
  renderWorkspace();
});
elements["compare-versions"].addEventListener("click", compareVersions);
elements["start-simulation"].addEventListener("click", startSimulation);
elements["send-simulation-turn"].addEventListener("click", sendSimulationTurn);
elements["simulation-turn"].addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendSimulationTurn();
  }
});
elements["reset-simulation"].addEventListener("click", resetSimulation);
elements["close-simulation"].addEventListener("click", closeSimulation);
elements["voice-listen"].addEventListener("click", () => {
  if (state.voiceMode === "auto") {
    state.handsFreeArmed = !state.handsFreeArmed;
    if (state.handsFreeArmed) {
      if (voice.state === "speaking") queueHandsFreeMonitor();
      else queueHandsFreeListen();
    }
    else void voice.cancelListening();
  } else if (voice.state === "listening" || voice.acquiring) {
    void voice.cancelListening();
  } else {
    if (voice.playbackActive) noteInterruptedReadback();
    void voice.listen();
  }
  syncVoiceAvailability();
});
elements["voice-send-capture"].addEventListener("click", () => {
  void voice.stopListening();
});
for (const mode of ["voice-mode-manual", "voice-mode-auto"]) {
  elements[mode].addEventListener("change", () => {
    state.voiceMode = elements["voice-mode-auto"].checked ? "auto" : "manual";
    state.handsFreeArmed = false;
    void voice.cancelListening();
    syncVoiceAvailability();
  });
}
elements["voice-speak-replies"].addEventListener("change", () => {
  if (!elements["voice-speak-replies"].checked) {
    voice.stopSpeaking();
    state.voiceAwaitingReply = false;
    queueHandsFreeListen();
  }
  syncVoiceAvailability();
});

elements["voice-support"].textContent = supportSummary();
syncVoiceAvailability();

const navigationLinks = [...elements["section-nav"].querySelectorAll("a[href^='#']")];
let navigationUpdateQueued = false;

function scheduleNavigationUpdate() {
  if (navigationUpdateQueued) return;
  navigationUpdateQueued = true;
  requestAnimationFrame(() => {
    navigationUpdateQueued = false;
    const cutoff = Math.min(200, window.innerHeight * 0.3);
    let current = navigationLinks[0];
    for (const link of navigationLinks) {
      if (link.hidden) continue;
      const section = document.getElementById(link.hash.slice(1));
      if (section?.getClientRects().length && section.getBoundingClientRect().top <= cutoff) {
        current = link;
      }
    }
    for (const link of navigationLinks) {
      if (link === current) link.setAttribute("aria-current", "location");
      else link.removeAttribute("aria-current");
    }
  });
}

window.addEventListener("scroll", scheduleNavigationUpdate, { passive: true });
window.addEventListener("resize", scheduleNavigationUpdate);

async function refreshVoiceIdentity() {
  // Name the engines the preview actually uses, so nobody has to infer them from config.
  // Identity belongs to a publication, so it resolves once a simulation has pinned one.
  const versionId = state.simulation?.publication_version_id;
  if (!state.merchantId || !versionId) {
    elements["voice-support"].textContent = `${supportSummary()} Start a simulation to bind a voice.`;
    return;
  }
  try {
    const identity = await api.getVoiceIdentity(state.merchantId, versionId);
    elements["voice-support"].textContent = `${supportSummary()} ${describeEngines(identity)}`;
    elements["voice-support"].classList.remove("voice-support-alert");
  } catch {
    elements["voice-support"].textContent =
      "Voice preview is not configured on this server. The text path is unaffected.";
  }
}

await refreshMerchants();
if (state.merchantId) await loadWorkspace();
await refreshVoiceIdentity();
elements["merchant-select"].addEventListener("change", refreshVoiceIdentity);
