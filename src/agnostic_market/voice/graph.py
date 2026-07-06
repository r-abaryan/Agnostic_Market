"""The Phase-2 minimal LangGraph graph behind livekit's LLMAdapter.

This is NOT the frontline agent (that is Phase 3a, docs/AGENTS.md) — it is the
pipe-cleaner that proves voice -> STT -> graph -> TTS end-to-end with read-only tools.
Built with `langchain.agents.create_agent` (the current factory; `create_react_agent`
is deprecated in LangGraph v1), which returns the compiled Pregel graph that
`livekit.plugins.langchain.LLMAdapter` requires.

The system prompt is NOT set here: the livekit `Agent(instructions=...)` carries it and
LLMAdapter maps it to the graph's SystemMessage — one prompt source, not two.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessageChunk
from langchain_core.tools import BaseTool
from langgraph.pregel import Pregel


def build_voice_graph(chat_model: BaseChatModel, tools: Sequence[BaseTool]) -> Pregel:
    """Compile the minimal tool-calling graph for the voice loop."""
    return create_agent(chat_model, list(tools))


class SpeakableTokens:
    """`astream` pass-through that yields ONLY LLM token chunks (AIMessageChunk).

    Workaround for livekit-plugins-langchain 1.6.4: its stream path converts EVERY item
    from `stream_mode="messages"` into speakable text — including ToolMessages — so the
    raw tool return string was spoken verbatim ahead of the model's actual answer
    (observed on the live call, 2026-07-06). LLMAdapter exercises only `astream`, so a
    filtering wrapper at our seam is the whole fix. Assumes the adapter's defaults
    (single `stream_mode="messages"`, `subgraphs=False` — items are `(token, metadata)`
    tuples). Remove when upstream filters non-token messages.
    """

    def __init__(self, graph: Pregel) -> None:
        self._graph = graph

    def astream(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        source = self._graph.astream(*args, **kwargs)

        async def _tokens_only() -> AsyncIterator[Any]:
            async for item in source:
                token = item[0] if isinstance(item, tuple) and len(item) == 2 else item
                if isinstance(token, AIMessageChunk):
                    yield item

        return _tokens_only()
