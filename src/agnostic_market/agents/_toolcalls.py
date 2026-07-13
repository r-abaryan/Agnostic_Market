"""Assemble-node tool-call hygiene, shared by the gated flows (cart + support).

Control/terminal and money proposals act ONE per turn (`tool_calls[0]`); the cart flow's
reversible mutations batch instead (its assemble answers each call itself). Either way a
model that emits several calls in one response must get a tool_result for EVERY tool_use:
the assemble's new messages persist into the shared thread history, and a dangling
tool_use/tool_result pair fails provider-side validation on every later model call in the
session — one bad turn would poison the whole call.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, ToolMessage


def ack_extra_tool_calls(response: AIMessage, new_messages: list) -> None:
    """Answer every tool call after the first with an explicit 'ignored' ToolMessage."""
    for extra in response.tool_calls[1:]:
        new_messages.append(
            ToolMessage(
                "ignored - one action per turn; only the first call was used",
                tool_call_id=extra["id"],
            )
        )
