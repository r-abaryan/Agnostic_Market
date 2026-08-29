from __future__ import annotations

from agnostic_market.agents.telemetry import (
    InMemoryTelemetrySink,
    SessionTelemetry,
    TenantTelemetry,
)


def make_tenant_telemetry(tenant_id: str) -> TenantTelemetry:
    return TenantTelemetry(
        tenant_id,
        InMemoryTelemetrySink(),
        InMemoryTelemetrySink(),
    )


def make_session_telemetry(tenant_id: str, session_id: str) -> SessionTelemetry:
    return make_tenant_telemetry(tenant_id).bind_session(session_id)
