"""Shared zero-network test fakes: a scripted BaseChatModel + a recording SecretResolver.

Used by the conformance suite, the gateway tests, and the voice tests. One configurable
chat fake: the default flags produce a fully conformant model; each broken variant flips
one flag; `tool_call_limit` bounds agent loops. The fake picks the bound tool whose name
appears in the prompt (falling back to the first bound tool — which is how the
with_structured_output schema-tool gets selected), then emits the canned args for it.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import PrivateAttr


class RecordingResolver:
    """SecretResolver test double — records refs, returns a dummy key."""

    def __init__(self) -> None:
        self.resolved: list[str] = []

    def resolve(self, ref: str) -> str:
        self.resolved.append(ref)
        return "sk-test-dummy"


# Valid args per probe tool / schema (names from llm/providers.py's suite).
CONFORMANT_ARGS: dict[str, dict[str, Any]] = {
    "get_order_status": {"order_id": "ORD-1001"},
    "search_catalog": {"query": "running shoes"},
    "CheckoutQuote": {
        "items": [{"sku": "SKU-RED-42", "quantity": 2}],
        "currency": "USD",
        "total": 19.99,
    },
}

# Fails CheckoutQuote validation (bad types + missing required field).
BROKEN_QUOTE_ARGS: dict[str, dict[str, Any]] = {
    **CONFORMANT_ARGS,
    "CheckoutQuote": {"items": "not-a-list", "currency": "GBP"},
}

_TEXT_RESPONSE = "A light, cushioned running shoe. It grips well on wet roads."


class FakeChatModel(BaseChatModel):
    """Deterministic fake; flip flags to produce each broken variant."""

    emit_tool_calls: bool = True
    pick_wrong_tool: bool = False
    canned_args: dict[str, dict[str, Any]] = CONFORMANT_ARGS
    stream_chunks: int = 3
    raise_transport: bool = False
    # None = tool-call on every tools-bound invoke (conformance-suite behavior). An int
    # bounds it so an agent LOOP (model -> tool -> model) terminates with a text answer.
    tool_call_limit: int | None = None
    # Deterministically emit a specific tool by name (bypasses _pick_tool) — for tests that
    # must exercise a tool the prompt wouldn't naturally name (e.g. request_handover).
    force_tool: str | None = None
    # Emit a SECOND identical tool call alongside the first (a misbehaving model) — drives
    # the assemble nodes' extra-call ack (a dangling tool_use poisons the thread history).
    double_tool_calls: bool = False
    # Fully scripted tool-call responses: each tools-bound invoke pops the next list of
    # (name, args) pairs and emits them as ONE multi-call message — drives the cart flow's
    # mutation batching ("one from each" = N calls in one response). Exhausted → normal
    # behavior. Overrides force_tool/canned_args while entries remain.
    scripted_calls: list[list[tuple[str, dict[str, Any]]]] | None = None
    # Capture every prompt this fake is invoked with — leak pins assert on what the model
    # SAW (e.g. the support candidate list must contain no unauthorized order data).
    record_prompts: bool = False
    _tool_calls_made: int = PrivateAttr(default=0)
    _script_index: int = PrivateAttr(default=0)
    _seen_prompts: list[str] = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fake-conformance"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Runnable:
        return self.bind(tools=[convert_to_openai_tool(t) for t in tools], **kwargs)

    def _pick_tool(self, tools: list[dict[str, Any]], messages: list[BaseMessage]) -> str:
        names = [t["function"]["name"] for t in tools]
        prompt = " ".join(str(m.content) for m in messages)
        mentioned = [n for n in names if n in prompt]
        correct = mentioned[0] if mentioned else names[0]
        if self.pick_wrong_tool:
            others = [n for n in names if n != correct]
            return others[0] if others else correct
        return correct

    def _respond(self, messages: list[BaseMessage], **kwargs: Any) -> AIMessage:
        if self.record_prompts:
            self._seen_prompts.append("\n".join(str(m.content) for m in messages))
        if self.raise_transport:
            raise ConnectionError("fake transport failure (simulated 429/timeout)")
        tools = kwargs.get("tools") or []
        if tools and self.scripted_calls and self._script_index < len(self.scripted_calls):
            scripted = self.scripted_calls[self._script_index]
            self._script_index += 1
            self._tool_calls_made += 1
            return AIMessage(
                content="",
                tool_calls=[
                    {"name": n, "args": a, "id": f"call_{i + 1}", "type": "tool_call"}
                    for i, (n, a) in enumerate(scripted)
                ],
            )
        budget_left = self.tool_call_limit is None or self._tool_calls_made < self.tool_call_limit
        if tools and self.emit_tool_calls and budget_left:
            self._tool_calls_made += 1
            name = self.force_tool or self._pick_tool(tools, messages)
            calls = [
                {
                    "name": name,
                    "args": self.canned_args.get(name, {}),
                    "id": "call_1",
                    "type": "tool_call",
                }
            ]
            if self.double_tool_calls:
                calls.append({**calls[0], "id": "call_2"})
            return AIMessage(content="", tool_calls=calls)
        return AIMessage(content=_TEXT_RESPONSE)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._respond(messages, **kwargs))])

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        message = self._respond(messages, **kwargs)
        text = str(message.content)
        if self.stream_chunks <= 1 or not text:
            yield ChatGenerationChunk(message=AIMessageChunk(content=text))
            return
        size = max(len(text) // self.stream_chunks, 1)
        for start in range(0, len(text), size):
            yield ChatGenerationChunk(message=AIMessageChunk(content=text[start : start + size]))
