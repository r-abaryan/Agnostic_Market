"""SpeakableTokens — the filter between the frontline graph and livekit's LLMAdapter.

The frontline graph (agents/frontline.py) is the live reasoning graph; this module is only
the transport-side filter that decides which streamed items become spoken audio.

Why it exists (livekit-plugins-langchain 1.6.4, live-observed 2026-07-06): the adapter's
`stream_mode="messages"` path turns EVERY streamed item into speakable text — including
ToolMessages (raw tool output was spoken verbatim). We pass only text the caller should
hear:
  - `AIMessageChunk`  — the model node's streamed answer tokens (real provider streaming);
  - node-authored full `AIMessage` from the `handover` node — the deferral line, which is
    NOT streamed (a node return, emitted whole). Without this, gate-trip turns (model never
    invoked) and model-handover turns would be SILENT.
Everything else (ToolMessage, Human/System, the model node's non-streamed echo) is dropped.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk

# Graph nodes whose full (non-streamed) AIMessage output must still be spoken.
_SPEAKABLE_MESSAGE_NODES = frozenset({"handover"})


class SpeakableTokens:
    """`astream` pass-through yielding only caller-audible items (see module docstring)."""

    def __init__(self, graph: Any) -> None:
        self._graph = graph

    def astream(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        source = self._graph.astream(*args, **kwargs)

        async def _speakable() -> AsyncIterator[Any]:
            async for item in source:
                token = item[0] if isinstance(item, tuple) and len(item) == 2 else item
                meta = item[1] if isinstance(item, tuple) and len(item) == 2 else {}
                # Streamed model tokens (the normal answer path).
                if isinstance(token, AIMessageChunk):
                    yield item
                    continue
                # A node-authored full AIMessage (e.g. the deferral) — speak it only from a
                # node we designate, so the model node's own non-streamed echo isn't doubled.
                if isinstance(token, AIMessage) and not isinstance(token, AIMessageChunk):
                    node = meta.get("langgraph_node") if isinstance(meta, dict) else None
                    if node in _SPEAKABLE_MESSAGE_NODES:
                        yield item

        return _speakable()
