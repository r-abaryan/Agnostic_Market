"""The minimal voice graph: tool-calling loop over the fixture + LLMAdapter compatibility."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessageChunk, HumanMessage, ToolMessage
from livekit.plugins import langchain as lk_langchain
from llm_fakes import CONFORMANT_ARGS, FakeChatModel

from agnostic_market.voice.graph import SpeakableTokens, build_voice_graph
from agnostic_market.voice.tools import build_voice_tools, load_orders_fixture

# The fake picks the first bound tool when none is named in the prompt, so order_status
# (first in build_voice_tools) gets called with these canned args.
_VOICE_ARGS: dict[str, dict[str, Any]] = {
    **CONFORMANT_ARGS,
    "order_status": {"order_id": "ORD-1001"},
}


def _graph(config_root: Path, fake: FakeChatModel):
    tools = build_voice_tools(load_orders_fixture(config_root, "acme_store"))
    return build_voice_graph(fake, tools)


async def test_graph_answers_order_status_through_the_tool(config_root: Path) -> None:
    fake = FakeChatModel(tool_call_limit=1, canned_args=_VOICE_ARGS)
    graph = _graph(config_root, fake)
    result = await graph.ainvoke(
        {"messages": [HumanMessage("What is the status of order ORD-1001?")]}
    )
    messages = result["messages"]
    tool_outputs = [m.content for m in messages if m.type == "tool"]
    assert any("shipped" in out for out in tool_outputs)  # the tool actually ran
    assert messages[-1].type == "ai" and messages[-1].content  # loop ended with a spoken answer


def test_llm_adapter_accepts_the_wrapped_graph(config_root: Path) -> None:
    # Pins the compatibility claim exactly as production wires it (pipeline.py).
    adapter = lk_langchain.LLMAdapter(SpeakableTokens(_graph(config_root, FakeChatModel())))
    assert adapter is not None


class _ScriptedGraph:
    """Graph double whose astream replays the item shapes langgraph's DEFAULT (v1)
    messages handler emits — which, per langgraph pregel/_messages.py, still includes
    ToolMessages. That is exactly what livekit's LLMAdapter turned into spoken text on
    the 2026-07-06 live call (raw tool string voiced before the model's answer)."""

    def __init__(self, items: list) -> None:
        self._items = items

    def astream(self, *args, **kwargs):
        async def _gen():
            for item in self._items:
                yield item

        return _gen()


async def test_raw_tool_output_is_never_speakable() -> None:
    meta = {"langgraph_node": "model"}
    scripted = _ScriptedGraph(
        [
            (AIMessageChunk(content=""), meta),  # cycle 1: the tool-call chunk
            (ToolMessage("Order ORD-1001: status shipped", tool_call_id="c1"), meta),
            (AIMessageChunk(content="Your order is on "), meta),
            (AIMessageChunk(content="its way."), meta),
        ]
    )
    survived = [item async for item in SpeakableTokens(scripted).astream({}, None)]
    tokens = [item[0] for item in survived]
    assert all(isinstance(t, AIMessageChunk) for t in tokens)  # ToolMessage dropped
    spoken = "".join(t.text for t in tokens)
    assert "shipped" not in spoken  # the tool's raw return string is never spoken ...
    assert "Your order is on its way." in spoken  # ... while the model's tokens all are
