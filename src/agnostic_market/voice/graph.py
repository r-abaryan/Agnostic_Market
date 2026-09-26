"""GraphVoiceAdapter — the Plane-1 side of the ReasoningEngine seam (evolved from the
Phase-2/3a `SpeakableTokens` filter).

Sits where LiveKit's `LLMAdapter` expects a langgraph: presents `.astream(state, ...)`.
Per turn it:
  - extracts the NEW committed user turn from the transport input (the adapter passes the
    full chat_ctx history each call; the engine's thread checkpoint carries history, so
    only the last user message is fed — feeding the full list would duplicate state);
  - gathers the playback fact for the assistant reply to the preceding caller turn;
    unknown or interrupted playback cannot authorize consent (VOICE_PIPELINE section 4a);
  - calls `engine.stream_turn(CommittedTurn(...), TurnFacts(...))` and renders TurnEvents as plain
    strings (LLMAdapter's `_to_chat_chunk` accepts str — verified from plugin source).

ALL LiveKit knowledge lives here; the engine imports nothing from the voice plane. The
session is attached after construction (`attach_session`) because the AgentSession is
built around this adapter.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from agnostic_market.agents.engine import ReasoningEngine
from agnostic_market.dtos.events import CommittedTurn, TurnEvent, TurnFacts

logger = logging.getLogger(__name__)


class GraphVoiceAdapter:
    """`astream`-compatible facade over a ReasoningEngine, for LiveKit's LLMAdapter."""

    def __init__(
        self,
        engine: ReasoningEngine,
        *,
        turn_started_observer: Callable[[], None] | None = None,
        turn_event_observer: Callable[[TurnEvent], None] | None = None,
        observer_failure_observer: Callable[[Exception], None] | None = None,
    ) -> None:
        self._engine = engine
        self._session: Any = None
        self._turn_started_observer = turn_started_observer
        self._turn_event_observer = turn_event_observer
        self._observer_failure_observer = observer_failure_observer

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

    def _readback_interrupted(self, state: dict[str, Any], turn: CommittedTurn) -> bool | None:
        """Read playback only for the reply to the preceding caller turn."""
        messages = [
            message
            for message in state.get("messages", [])
            if isinstance(message, (AIMessage, HumanMessage))
        ]
        if (
            len(messages) < 3
            or not isinstance(messages[-1], HumanMessage)
            or messages[-1].id != turn.message_id
        ):
            logger.debug("assistant playback status unavailable: turn history incomplete")
            return None
        previous_user_index = next(
            (
                index
                for index in range(len(messages) - 2, -1, -1)
                if isinstance(messages[index], HumanMessage)
            ),
            None,
        )
        if previous_user_index is None:
            logger.debug("assistant playback status unavailable: preceding caller turn absent")
            return None
        replies = [
            message
            for message in messages[previous_user_index + 1 : -1]
            if isinstance(message, AIMessage)
        ]
        if len(replies) != 1 or not replies[0].id:
            logger.warning(
                "assistant playback status unavailable: preceding reply absent or ambiguous"
            )
            return None
        reply_id = replies[0].id
        if self._session is None:
            logger.warning("assistant playback status unavailable: session unattached")
            return None
        try:
            items = list(self._session.history.items)
        except AttributeError:
            logger.warning("assistant playback status unavailable: history inaccessible")
            return None
        matches = [item for item in items if getattr(item, "id", None) == reply_id]
        if (
            len(matches) != 1
            or getattr(matches[0], "type", None) != "message"
            or getattr(matches[0], "role", None) != "assistant"
        ):
            logger.warning("assistant playback status unavailable: reply not in history")
            return None
        interrupted = getattr(matches[0], "interrupted", None)
        if isinstance(interrupted, bool):
            logger.debug("assistant playback status correlated: interrupted=%s", interrupted)
            return interrupted
        logger.warning("assistant playback status unavailable: interruption flag invalid")
        return None

    def _observe(self, observer: Callable[..., None] | None, *values: object) -> None:
        if observer is None:
            return
        try:
            observer(*values)
        except Exception as exc:
            logger.exception("voice diagnostic observer failed")
            if self._observer_failure_observer is None:
                return
            try:
                self._observer_failure_observer(exc)
            except Exception:
                logger.exception("voice diagnostic failure observer failed")

    def astream(self, state: dict[str, Any], *args: Any, **kwargs: Any) -> AsyncIterator[str]:
        """The LLMAdapter entry point. `state` is the chat_ctx-derived message dict; the
        adapter's config/stream_mode args are ignored — the engine owns thread + modes."""
        turn = self._last_user_turn(state)

        async def _events() -> AsyncIterator[str]:
            if turn is None:
                logger.warning("voice adapter: no user message in transport input; empty turn")
                return
            self._observe(self._turn_started_observer)
            facts = TurnFacts(readback_interrupted=self._readback_interrupted(state, turn))
            async for event in self._engine.stream_turn(turn, facts):
                self._observe(self._turn_event_observer, event)
                # Token / spoken-message / interrupt prompt — all graph-authored text.
                yield event.text if event.kind != "interrupt" else event.prompt

        return _events()
