"""SpeakableTokens — the transport filter: what streams to TTS and what is dropped."""

from __future__ import annotations

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage

from agnostic_market.voice.graph import SpeakableTokens


class _ScriptedGraph:
    """Graph double whose astream replays the item shapes langgraph's DEFAULT (v1)
    messages handler emits — (message, metadata) tuples, including ToolMessages and
    node-authored full AIMessages. Lets us pin the filter without a live model."""

    def __init__(self, items: list) -> None:
        self._items = items

    def astream(self, *args, **kwargs):
        async def _gen():
            for item in self._items:
                yield item

        return _gen()


async def _speak(items: list) -> list:
    return [item[0] async for item in SpeakableTokens(_ScriptedGraph(items)).astream({}, None)]


async def test_raw_tool_output_is_never_speakable() -> None:
    # The 2026-07-06 live bug: ToolMessages were spoken verbatim. They must be dropped.
    model_meta = {"langgraph_node": "model"}
    survived = await _speak(
        [
            (AIMessageChunk(content=""), model_meta),
            (ToolMessage("Order ORD-1001: status shipped", tool_call_id="c1"), model_meta),
            (AIMessageChunk(content="Your order is on "), model_meta),
            (AIMessageChunk(content="its way."), model_meta),
        ]
    )
    assert all(isinstance(t, AIMessageChunk) for t in survived)
    spoken = "".join(t.text for t in survived)
    assert "shipped" not in spoken  # raw tool string never spoken ...
    assert "Your order is on its way." in spoken  # ... model's tokens all are


async def test_handover_deferral_message_is_spoken() -> None:
    # The deferral is a node-authored FULL AIMessage (not streamed) — without passing it,
    # gate-trip and model-handover turns would be SILENT.
    survived = await _speak(
        [(AIMessage("I'll make sure it reaches our support team."), {"langgraph_node": "handover"})]
    )
    assert len(survived) == 1
    assert "support team" in survived[0].content


async def test_non_handover_full_message_is_not_doubled() -> None:
    # A full AIMessage from the MODEL node (the non-streamed echo) must NOT be re-spoken —
    # the model's answer already streamed as chunks.
    survived = await _speak(
        [
            (AIMessageChunk(content="the answer"), {"langgraph_node": "model"}),
            (AIMessage("the answer"), {"langgraph_node": "model"}),
        ]
    )
    assert len(survived) == 1
    assert isinstance(survived[0], AIMessageChunk)


async def test_human_and_system_messages_dropped() -> None:
    from langchain_core.messages import SystemMessage

    survived = await _speak(
        [
            (HumanMessage("hi"), {"langgraph_node": "model"}),
            (SystemMessage("prompt"), {"langgraph_node": "model"}),
            (AIMessageChunk(content="hello"), {"langgraph_node": "model"}),
        ]
    )
    assert len(survived) == 1
    assert survived[0].content == "hello"
