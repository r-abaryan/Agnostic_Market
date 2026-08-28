"""GraphVoiceAdapter — the Plane-1 side of the ReasoningEngine seam (evolved from the
Phase-2/3a `SpeakableTokens` filter).

Sits where LiveKit's `LLMAdapter` expects a langgraph: presents `.astream(state, ...)`.
Per turn it:
  - extracts the NEW committed user turn from the transport input (the adapter passes the
    full chat_ctx history each call; the engine's thread checkpoint carries history, so
    only the last user message is fed — feeding the full list would duplicate state);
  - gathers the §4a perception fact: whether the caller barged over the pending
    confirmation readback (`ChatMessage.interrupted` on the last assistant history item —
    consent over a truncated readback is not consent, VOICE_PIPELINE §4a);
  - calls `engine.stream_turn(CommittedTurn(...), TurnFacts(...))` and renders TurnEvents as plain
    strings (LLMAdapter's `_to_chat_chunk` accepts str — verified from plugin source).

ALL LiveKit knowledge lives here; the engine imports nothing from the voice plane. The
session is attached after construction (`attach_session`) because the AgentSession is
built around this adapter.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import HumanMessage

from agnostic_market.agents.engine import ReasoningEngine
from agnostic_market.dtos.events import CommittedTurn, TurnFacts

logger = logging.getLogger(__name__)


class GraphVoiceAdapter:
    """`astream`-compatible facade over a ReasoningEngine, for LiveKit's LLMAdapter."""

    def __init__(self, engine: ReasoningEngine) -> None:
        self._engine = engine
        self._session: Any = None

    def attach_session(self, session: Any) -> None:
        """Bind the live AgentSession (post-construction; the session wraps this adapter)."""
        self._session = session

    @property
    def engine(self) -> ReasoningEngine:
        return self._engine

    def _last_user_turn(self, state: dict[str, Any]) -> CommittedTurn | None:
        for msg in reversed(state.get("messages", [])):
            if isinstance(msg, HumanMessage):
                return CommittedTurn(
                    text=str(msg.content),
                    message_id=msg.id,
                )
        return None

    def _readback_interrupted(self) -> bool:
        """§4a fact: was the last agent utterance (the pending readback) barged over?

        Only meaningful while the engine is paused at a confirmation interrupt; reads
        `interrupted` from the most recent assistant item in the session history.
        """
        if self._session is None:
            return False
        try:
            items = list(self._session.history.items)
        except AttributeError:
            return False
        for item in reversed(items):
            is_message = getattr(item, "type", None) == "message"
            if is_message and getattr(item, "role", "") == "assistant":
                return bool(getattr(item, "interrupted", False))
        return False

    def astream(self, state: dict[str, Any], *args: Any, **kwargs: Any) -> AsyncIterator[str]:
        """The LLMAdapter entry point. `state` is the chat_ctx-derived message dict; the
        adapter's config/stream_mode args are ignored — the engine owns thread + modes."""
        turn = self._last_user_turn(state)

        async def _events() -> AsyncIterator[str]:
            if turn is None:
                logger.warning("voice adapter: no user message in transport input; empty turn")
                return
            facts = TurnFacts(readback_interrupted=self._readback_interrupted())
            async for event in self._engine.stream_turn(turn, facts):
                # Token / spoken-message / interrupt prompt — all graph-authored text.
                yield event.text if event.kind != "interrupt" else event.prompt

        return _events()
