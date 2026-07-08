"""AgentSession assembly — the Phase-2 minimal voice loop (BUILD_PLAN Phase 2).

Per-merchant, all from config: STT/TTS via the engine factories, the LLM via the
Phase-1 gateway (`config.llm.routing`), tools from the orders fixture. VAD and audio
turn detection are livekit-agents 1.6 session DEFAULTS — deliberately not configured
here (adopt built, VOICE_PIPELINE §1b); the built-in audio turn detector replaced the
deprecated text turn-detector plugin (VOICE_PIPELINE §2 pending an update pass).

The call-start disclosure (COMPLIANCE §2, EU AI Act Art. 50(1)) is played by the agent's
own `on_enter` hook — STRUCTURAL ordering (it fires the moment the agent enters the
session, before any user turn can be answered), not reliant on the worker calling say()
in time. Wording comes from merchant config; that it plays, plays first, and cannot be
barged over is enforced here in code.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from livekit.agents import Agent, AgentSession, ConversationItemAddedEvent
from livekit.plugins import langchain as lk_langchain

from agnostic_market.agents.frontline import build_frontline_graph
from agnostic_market.agents.tooling import wrap_readonly_tool
from agnostic_market.config.registry import ResolvedConfig
from agnostic_market.dtos.llm import ProviderCredentialsConfig
from agnostic_market.llm.gateway import LLMGateway
from agnostic_market.secrets.base import SecretResolver
from agnostic_market.voice.graph import SpeakableTokens
from agnostic_market.voice.stt_engine import build_stt
from agnostic_market.voice.tools import build_voice_tools, load_orders_fixture
from agnostic_market.voice.tts_engine import build_tts

logger = logging.getLogger(__name__)


class DisclosureFirstAgent(Agent):
    """Agent that speaks the call-start disclosure the moment it enters the session.

    `on_enter` is the session lifecycle hook — it runs before any user turn can be
    answered, so disclosure-first is guaranteed by structure (COMPLIANCE §2), not by the
    worker racing to call say() after session.start(). Uninterruptible: a barged-over
    disclosure was not fully played.
    """

    def __init__(self, *, instructions: str, disclosure: str) -> None:
        super().__init__(instructions=instructions)
        self.disclosure = disclosure

    async def on_enter(self) -> None:
        await self.session.say(self.disclosure, allow_interruptions=False)


@dataclass(frozen=True)
class VoiceLoop:
    """Everything the worker needs to serve one merchant's call."""

    session: AgentSession
    agent: DisclosureFirstAgent


def build_voice_loop(
    resolved: ResolvedConfig,
    credentials: ProviderCredentialsConfig,
    secrets: SecretResolver,
    *,
    config_root: Path,
) -> VoiceLoop:
    """Assemble the per-merchant session: engines, graph, tools, disclosure — all from config."""
    config = resolved.config
    fixture = load_orders_fixture(config_root, config.merchant_id)
    chat_model = LLMGateway(credentials, secrets).chat_model(config.llm.routing)
    # Read-only tools pass through the audit/tenant wrapper; the frontline graph owns its
    # own system prompt + few-shot (F1), so the Agent below carries NO instructions.
    tools = [wrap_readonly_tool(t, config.merchant_id) for t in build_voice_tools(fixture)]
    graph = build_frontline_graph(chat_model, tools, display_name=config.display_name)

    session = AgentSession(
        stt=build_stt(config.voice.stt, credentials, secrets),
        # SpeakableTokens: only the model's answer tokens + the handover deferral reach TTS
        # — never raw tool output (see graph.py).
        llm=lk_langchain.LLMAdapter(SpeakableTokens(graph)),
        tts=build_tts(config.voice.tts, credentials, secrets),
    )
    _attach_turn_metrics_logger(session)

    agent = DisclosureFirstAgent(
        # Prompt lives in the graph (F1); empty here is dropped by the adapter, no duplicate.
        instructions="",
        # `.replace`, not `.format`: merchant-authored wording may contain other braces.
        disclosure=config.compliance.call_start_disclosure.replace(
            "{display_name}", config.display_name
        ),
    )
    return VoiceLoop(session=session, agent=agent)


def _attach_turn_metrics_logger(session: AgentSession) -> None:
    """Log per-turn latency (BUILD_PLAN Phase 2 'measure turn latency'; OTel backend = Phase 6).

    `ChatMessage.metrics` is the current per-turn source (`metrics_collected` is
    deprecated): user messages carry `end_of_turn_delay`/`transcription_delay`;
    assistant messages carry `llm_node_ttft`/`tts_node_ttfb`/`e2e_latency`.
    """

    @session.on("conversation_item_added")
    def _log_turn_metrics(ev: ConversationItemAddedEvent) -> None:
        metrics = getattr(ev.item, "metrics", None) or {}
        if not metrics:
            return
        fields = ", ".join(
            f"{name}={value:.3f}s"
            for name in (
                "end_of_turn_delay",
                "transcription_delay",
                "llm_node_ttft",
                "tts_node_ttfb",
                "e2e_latency",
            )
            if (value := metrics.get(name)) is not None
        )
        if fields:
            logger.info("turn metrics [%s]: %s", ev.item.role, fields)
