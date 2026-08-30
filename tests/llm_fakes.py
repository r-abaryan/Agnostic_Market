"""Shared zero-network test fakes: a scripted BaseChatModel + a recording SecretResolver.

Used by the conformance suite, the gateway tests, and the voice tests. One configurable
chat fake: the default flags produce a fully conformant model; each broken variant flips
one flag; `tool_call_limit` bounds agent loops. The fake picks the bound tool whose name
appears in the prompt (falling back to the first bound tool — which is how the
with_structured_output schema-tool gets selected), then emits the canned args for it.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import PrivateAttr

from agnostic_market.dtos.llm import StructuredOutputMethod

TEST_STRUCTURED_OUTPUT_METHOD: StructuredOutputMethod = "function_calling"
TEST_CALLER_AUDIBLE_MODEL_TEXT_MAX_CHARS = 500


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
    "RouteProposal": {"decision": "direct", "capability": "search_catalog"},
}

CONFORMANT_STRUCTURED_ARGS: dict[str, tuple[dict[str, Any], ...]] = {
    "RouteProposal": (
        CONFORMANT_ARGS["RouteProposal"],
        {
            "decision": "clarify",
            "clarification_reason": "ambiguous_intent",
        },
        {"decision": "continue"},
        {
            "decision": "direct",
            "capability": "answer_question",
            "answer_topic": "policy",
        },
        {
            "decision": "direct",
            "capability": "list_orders",
            "list_scope": "account",
        },
        {
            "decision": "direct",
            "capability": "modify_cart",
            "cart_operation": "add",
        },
        {
            "decision": "direct",
            "capability": "change_profile",
            "profile_field": "address",
        },
        {
            "decision": "direct",
            "capability": "verify_order_status",
            "order_status_selector": "explicit",
        },
    ),
    "AnswerResponse": (
        {"decision": "answer", "answer": "Returns are accepted within 30 days."},
        {"decision": "clarify", "answer": None},
        {"decision": "unsupported", "answer": None},
    ),
    "OrderTargetProposal": (
        {"relationship": "single", "order_refs": ["ORD-1002"]},
        {"relationship": "plural", "order_refs": ["ORD-1001", "ORD-1002"]},
        {"relationship": "alternative", "order_refs": ["ORD-1001", "ORD-1002"]},
        {"relationship": "ambiguous", "order_refs": []},
    ),
}

_TEXT_RESPONSE = "A light, cushioned running shoe. It grips well on wet roads."


class FakeChatModel(BaseChatModel):
    """Deterministic fake; flip flags to produce each broken variant."""

    emit_tool_calls: bool = True
    pick_wrong_tool: bool = False
    canned_args: dict[str, dict[str, Any]] = CONFORMANT_ARGS
    structured_args: dict[str, tuple[dict[str, Any], ...]] | None = CONFORMANT_STRUCTURED_ARGS
    stream_chunks: int = 3
    raise_transport: bool = False
    # None = tool-call on every tools-bound invoke (conformance-suite behavior). An int
    # bounds it so an agent LOOP (model -> tool -> model) terminates with a text answer.
    tool_call_limit: int | None = None
    # Deterministically emit a specific bound tool by name when a test must bypass selection.
    force_tool: str | None = None
    # Emit a SECOND identical tool call alongside the first (a misbehaving model) — drives
    # the assemble nodes' extra-call ack (a dangling tool_use poisons the thread history).
    double_tool_calls: bool = False
    # Fully scripted tool-call responses: each tools-bound invoke pops the next list of
    # (name, args) pairs and emits them as ONE multi-call message — drives the cart flow's
    # mutation batching ("one from each" = N calls in one response). Exhausted → normal
    # behavior. Overrides force_tool/canned_args while entries remain.
    scripted_calls: list[list[tuple[str, dict[str, Any]]]] | None = None
    text_response: str = _TEXT_RESPONSE
    # Capture every prompt this fake is invoked with — leak pins assert on what the model
    # SAW (e.g. the support candidate list must contain no unauthorized order data).
    record_prompts: bool = False
    _tool_calls_made: int = PrivateAttr(default=0)
    _script_index: int = PrivateAttr(default=0)
    _invoke_count: int = PrivateAttr(default=0)
    _seen_prompts: list[str] = PrivateAttr(default_factory=list)
    _emitted_messages: list[AIMessage] = PrivateAttr(default_factory=list)
    _bound_tools: dict[str, dict[str, Any]] = PrivateAttr(default_factory=dict)
    _structured_indexes: dict[str, int] = PrivateAttr(default_factory=dict)
    _structured_methods: list[object] = PrivateAttr(default_factory=list)

    @property
    def invoke_count(self) -> int:
        return self._invoke_count

    @property
    def emitted_messages(self) -> tuple[AIMessage, ...]:
        return tuple(self._emitted_messages)

    @property
    def bound_tools(self) -> dict[str, dict[str, Any]]:
        return dict(self._bound_tools)

    @property
    def structured_methods(self) -> tuple[object, ...]:
        return tuple(self._structured_methods)

    @property
    def _llm_type(self) -> str:
        return "fake-conformance"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Runnable:
        converted = [convert_to_openai_tool(tool) for tool in tools]
        self._bound_tools.update({schema["function"]["name"]: schema for schema in converted})
        return self.bind(tools=converted, **kwargs)

    def with_structured_output(
        self,
        schema: dict[str, Any] | type,
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable:
        self._structured_methods.append(kwargs.get("method"))
        return super().with_structured_output(schema, include_raw=include_raw, **kwargs)

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
        self._invoke_count += 1
        if self.record_prompts:
            self._seen_prompts.append("\n".join(str(m.content) for m in messages))
        if self.raise_transport:
            raise ConnectionError("fake transport failure (simulated 429/timeout)")
        tools = kwargs.get("tools") or []
        if tools and self.scripted_calls and self._script_index < len(self.scripted_calls):
            scripted = self.scripted_calls[self._script_index]
            self._script_index += 1
            self._tool_calls_made += 1
            response = AIMessage(
                content="",
                tool_calls=[
                    {"name": n, "args": a, "id": f"call_{i + 1}", "type": "tool_call"}
                    for i, (n, a) in enumerate(scripted)
                ],
            )
            self._emitted_messages.append(response)
            return response
        budget_left = self.tool_call_limit is None or self._tool_calls_made < self.tool_call_limit
        if tools and self.emit_tool_calls and budget_left:
            self._tool_calls_made += 1
            name = self.force_tool or self._pick_tool(tools, messages)
            args = self.canned_args.get(name, {})
            if self.structured_args is not None and name in self.structured_args:
                payloads = self.structured_args[name]
                index = self._structured_indexes.get(name, 0)
                if index >= len(payloads):
                    raise RuntimeError(f"no structured payload remains for {name}")
                args = payloads[index]
                self._structured_indexes[name] = index + 1
            calls = [
                {
                    "name": name,
                    "args": args,
                    "id": "call_1",
                    "type": "tool_call",
                }
            ]
            if self.double_tool_calls:
                calls.append({**calls[0], "id": "call_2"})
            response = AIMessage(content="", tool_calls=calls)
            self._emitted_messages.append(response)
            return response
        response = AIMessage(content=self.text_response)
        self._emitted_messages.append(response)
        return response

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


class NativeAsyncOnlyFakeChatModel(FakeChatModel):
    """Fake that fails if production falls back to the synchronous model surface."""

    def _generate(self, *_args: Any, **_kwargs: Any) -> ChatResult:
        raise AssertionError("model node used the synchronous provider API")

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager
        await asyncio.sleep(0)
        return ChatResult(generations=[ChatGeneration(message=self._respond(messages, **kwargs))])


class NativeAsyncBlockingFakeChatModel(NativeAsyncOnlyFakeChatModel):
    """Native-async model that blocks until its caller cancels the provider task."""

    _started: asyncio.Event = PrivateAttr(default_factory=asyncio.Event)
    _cancelled: asyncio.Event = PrivateAttr(default_factory=asyncio.Event)

    @property
    def started(self) -> asyncio.Event:
        return self._started

    @property
    def cancelled(self) -> asyncio.Event:
        return self._cancelled

    async def _agenerate(self, *_args: Any, **_kwargs: Any) -> ChatResult:
        self._started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self._cancelled.set()
            raise


class ExplodingOnceFakeChatModel(FakeChatModel):
    """Raise once as a provider outage, then use the configured fake response."""

    _exploded: bool = PrivateAttr(default=False)

    def _respond(self, messages: list[BaseMessage], **kwargs: Any) -> AIMessage:
        if not self._exploded:
            self._exploded = True
            raise RuntimeError("simulated provider 529 overloaded")
        return super()._respond(messages, **kwargs)
