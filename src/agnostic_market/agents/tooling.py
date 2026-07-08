"""Tool wrapper — the enforcement + audit seam every agent tool call passes through.

Phase 3a lands the SEAM and the audit line (AGENTS.md §A5: "every tool call is wrapped —
validate tenant, stamp it, audit"). The heavier enforcement surface (shared-store tenant
stamping, policy checks, OTel spans) grows here in Phase 4 when there are real stores to
stamp. Kept deliberately thin now — the point is that the wrapper EXISTS and every tool
goes through it, not that it does everything yet.

Audit discipline: log tool name, tenant, outcome, duration — NEVER argument values (a
caller's order id / query is PII-adjacent; SECURITY §6 logging rule).
"""

from __future__ import annotations

import logging
import time
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool

logger = logging.getLogger("agnostic_market.agents.audit")


def wrap_readonly_tool(tool: BaseTool, tenant_id: str) -> BaseTool:
    """Return an audited delegate of a read-only tool, tenant-stamped.

    The delegate preserves the tool's name/description/schema (so the model sees an
    identical tool) and logs one audit line per call with NO argument values.
    """

    def _audited(**kwargs: Any) -> Any:
        start = time.monotonic()
        try:
            result = tool.invoke(kwargs)
        except Exception:
            logger.warning(
                "tool call failed | tenant=%s tool=%s duration_ms=%.0f",
                tenant_id,
                tool.name,
                (time.monotonic() - start) * 1000,
            )
            raise
        logger.info(
            "tool call | tenant=%s tool=%s ok duration_ms=%.0f",
            tenant_id,
            tool.name,
            (time.monotonic() - start) * 1000,
        )
        return result

    return StructuredTool.from_function(
        func=_audited,
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
    )
