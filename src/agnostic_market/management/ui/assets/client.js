export class ManagementApiError extends Error {
  constructor(status, code, validation = null) {
    super(`Management request failed with ${code}`);
    this.name = "ManagementApiError";
    this.status = status;
    this.code = code;
    this.validation = validation;
  }
}

function encoded(value) {
  return encodeURIComponent(value);
}

export class ManagementApi {
  constructor(fetchImpl = globalThis.fetch.bind(globalThis)) {
    this.fetchImpl = fetchImpl;
  }

  async request(path, options = {}) {
    const headers = { accept: "application/json", ...options.headers };
    if (options.body !== undefined) {
      headers["content-type"] = "application/json";
    }
    const response = await this.fetchImpl(path, {
      cache: "no-store",
      credentials: "same-origin",
      ...options,
      headers,
    });
    let payload = null;
    if (response.status !== 204) {
      try {
        payload = await response.json();
      } catch {
        throw new ManagementApiError(response.status, "service_unavailable");
      }
    }
    if (!response.ok) {
      throw new ManagementApiError(
        response.status,
        payload?.code ?? "service_unavailable",
        payload?.validation ?? null,
      );
    }
    return payload;
  }

  getVoiceIdentity(tenantId, versionId) {
    return this.request(
      `/v1/merchants/${encoded(tenantId)}/versions/${encoded(versionId)}/voice`,
    );
  }

  // Audio, not JSON, so these two bypass `request` while keeping its error semantics: the
  // response body is never reflected into the thrown error.
  async synthesizeSpeech(tenantId, versionId, text) {
    const path = `/v1/merchants/${encoded(tenantId)}/versions/${encoded(versionId)}/voice/speech`;
    const response = await this.fetchImpl(path, {
      method: "POST",
      cache: "no-store",
      credentials: "same-origin",
      headers: { accept: "audio/wav", "content-type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (!response.ok) {
      throw new ManagementApiError(response.status, "service_unavailable");
    }
    return response.blob();
  }

  async transcribeCapture(tenantId, versionId, pcm) {
    const path = `/v1/merchants/${encoded(tenantId)}/versions/${encoded(versionId)}/voice/transcript`;
    const response = await this.fetchImpl(path, {
      method: "POST",
      cache: "no-store",
      credentials: "same-origin",
      headers: { accept: "application/json", "content-type": "application/octet-stream" },
      body: pcm,
    });
    let payload = null;
    try {
      payload = await response.json();
    } catch {
      throw new ManagementApiError(response.status, "service_unavailable");
    }
    if (!response.ok) {
      throw new ManagementApiError(response.status, payload?.code ?? "service_unavailable");
    }
    return payload;
  }

  listMerchants() {
    return this.request("/v1/merchants");
  }

  getDraft(tenantId, draftId) {
    return this.request(`/v1/merchants/${encoded(tenantId)}/drafts/${encoded(draftId)}`);
  }

  seedDraft(tenantId, draftId, requestId) {
    return this.request(
      `/v1/merchants/${encoded(tenantId)}/drafts/${encoded(draftId)}/seed`,
      {
        method: "POST",
        body: JSON.stringify({ schema_version: 1, request_id: requestId }),
      },
    );
  }

  saveDraft(tenantId, draftId, expectedRevision, payload) {
    const query = new URLSearchParams({ expected_revision: String(expectedRevision) });
    return this.request(
      `/v1/merchants/${encoded(tenantId)}/drafts/${encoded(draftId)}?${query}`,
      { method: "PUT", body: JSON.stringify(payload) },
    );
  }

  importCatalog(tenantId, draftId, expectedRevision, requestId, catalog) {
    return this.request(
      `/v1/merchants/${encoded(tenantId)}/drafts/${encoded(draftId)}/catalog`,
      {
        method: "PUT",
        body: JSON.stringify({
          schema_version: 1,
          expected_draft_revision: expectedRevision,
          request_id: requestId,
          catalog,
        }),
      },
    );
  }

  validateDraft(tenantId, draftId, draftRevision, requestId) {
    return this.request(
      `/v1/merchants/${encoded(tenantId)}/drafts/${encoded(draftId)}/validation`,
      {
        method: "POST",
        body: JSON.stringify({
          schema_version: 1,
          draft_revision: draftRevision,
          request_id: requestId,
        }),
      },
    );
  }

  previewDraft(tenantId, draftId, draftRevision) {
    return this.request(
      `/v1/merchants/${encoded(tenantId)}/drafts/${encoded(draftId)}/preview`,
      {
        method: "POST",
        body: JSON.stringify({ schema_version: 1, draft_revision: draftRevision }),
      },
    );
  }

  publishDraft(tenantId, draftId, draftRevision, fingerprint, activeVersionId, requestId) {
    return this.request(
      `/v1/merchants/${encoded(tenantId)}/drafts/${encoded(draftId)}/publication`,
      {
        method: "POST",
        body: JSON.stringify({
          schema_version: 1,
          draft_revision: draftRevision,
          expected_preview_fingerprint: fingerprint,
          expected_active_version_id: activeVersionId,
          request_id: requestId,
        }),
      },
    );
  }

  listVersions(tenantId) {
    return this.request(`/v1/merchants/${encoded(tenantId)}/versions`);
  }

  getActiveVersion(tenantId) {
    return this.request(`/v1/merchants/${encoded(tenantId)}/versions/active`);
  }

  compareVersions(tenantId, baseVersionId, targetVersionId) {
    const query = new URLSearchParams({
      base_version_id: baseVersionId,
      target_version_id: targetVersionId,
    });
    return this.request(`/v1/merchants/${encoded(tenantId)}/version-diff?${query}`);
  }

  listAudit(tenantId) {
    return this.request(`/v1/merchants/${encoded(tenantId)}/audit`);
  }

  rollback(tenantId, expectedActiveVersionId, sourceVersionId, requestId) {
    return this.request(`/v1/merchants/${encoded(tenantId)}/rollbacks`, {
      method: "POST",
      body: JSON.stringify({
        schema_version: 1,
        expected_active_version_id: expectedActiveVersionId,
        source_version_id: sourceVersionId,
        request_id: requestId,
      }),
    });
  }

  startSimulation(tenantId, simulationId, versionId = null) {
    return this.request(
      `/v1/merchants/${encoded(tenantId)}/simulations/${encoded(simulationId)}`,
      {
        method: "POST",
        body: JSON.stringify({ schema_version: 1, version_id: versionId }),
      },
    );
  }

  getSimulation(tenantId, simulationId) {
    return this.request(
      `/v1/merchants/${encoded(tenantId)}/simulations/${encoded(simulationId)}`,
    );
  }

  sendSimulationTurn(tenantId, simulationId, requestId, text, readbackInterrupted = false) {
    return this.request(
      `/v1/merchants/${encoded(tenantId)}/simulations/${encoded(simulationId)}/turns`,
      {
        method: "POST",
        body: JSON.stringify({
          schema_version: 1,
          request_id: requestId,
          text,
          readback_interrupted: readbackInterrupted,
        }),
      },
    );
  }

  inspectSimulation(tenantId, simulationId) {
    return this.request(
      `/v1/merchants/${encoded(tenantId)}/simulations/${encoded(simulationId)}/state`,
    );
  }

  resetSimulation(tenantId, simulationId) {
    return this.request(
      `/v1/merchants/${encoded(tenantId)}/simulations/${encoded(simulationId)}/reset`,
      { method: "POST" },
    );
  }

  closeSimulation(tenantId, simulationId) {
    return this.request(
      `/v1/merchants/${encoded(tenantId)}/simulations/${encoded(simulationId)}`,
      { method: "DELETE" },
    );
  }
}

export function buildDraftUpdate(current, merchantOverride, requestId) {
  return {
    expectedRevision: current.revision,
    payload: {
      schema_version: 1,
      revision: current.revision + 1,
      request_id: requestId,
      merchant_override: merchantOverride,
      fixtures: current.fixtures,
      dataset_manifest: current.dataset_manifest ?? null,
    },
  };
}

export function canRollback(activeVersionId, sourceVersionId) {
  return Boolean(activeVersionId && sourceVersionId && activeVersionId !== sourceVersionId);
}

export function selectMerchantWorkspace(state, merchantId) {
  Object.assign(state, {
    merchantId,
    draft: null,
    preview: null,
    validation: null,
    activeVersion: null,
    versions: [],
    audit: [],
    simulation: null,
    simulationState: null,
    simulationMessages: [],
    simulationDiagnostics: [],
    simulationLatency: null,
    pendingSimulationTurn: null,
  });
}

export function simulationTurnRequest(current, input, requestIdFactory) {
  if (
    current?.tenantId === input.tenantId &&
    current.simulationId === input.simulationId &&
    current.text === input.text &&
    current.readbackInterrupted === input.readbackInterrupted
  ) {
    return current;
  }
  return { ...input, requestId: requestIdFactory() };
}

export function simulationMessages(events) {
  const messages = [];
  let streamed = "";
  const flushStream = () => {
    if (streamed) messages.push({ kind: "assistant", text: streamed });
    streamed = "";
  };
  for (const event of events) {
    if (event.kind === "token") {
      streamed += event.text;
      continue;
    }
    flushStream();
    if (event.kind === "spoken_message") {
      messages.push({ kind: "assistant", text: event.text });
    } else if (event.kind === "interrupt") {
      messages.push({ kind: "confirmation", text: event.prompt });
    }
  }
  flushStream();
  return messages;
}

const DIAGNOSTIC_ATTRIBUTE_KEYS = [
  "decision",
  "capability",
  "clarification_reason",
  "failure_reason",
  "provider_call_outcome",
  "provider_error_category",
  "reason",
  "node",
  "action",
  "disposition",
  "latency_ms",
];

export function simulationDiagnostics(result) {
  const records = [...(result.routing_records ?? []), ...(result.operational_records ?? [])];
  return records.map((record) => {
    const fields = [record.event];
    const attributes = record.attributes ?? {};
    for (const key of DIAGNOSTIC_ATTRIBUTE_KEYS) {
      const value = attributes[key];
      if (value === null || value === undefined) continue;
      const rendered = key === "latency_ms" ? Math.round(Number(value)) : String(value);
      fields.push(`${key}=${rendered}`);
    }
    return fields.join(" | ");
  });
}

export function parseCatalog(source) {
  let catalog;
  try {
    catalog = JSON.parse(source);
  } catch {
    throw new Error("Catalog must be valid JSON.");
  }
  if (!catalog || typeof catalog !== "object" || Array.isArray(catalog)) {
    throw new Error("Catalog must be a JSON object.");
  }
  return catalog;
}
