"""Assemble one configured voice session around the reasoning engine.

STT, TTS, response and reasoning models, merchant stores, and the qualified routing recognizer
are explicit session dependencies. LiveKit owns VAD and audio turn-detection defaults. The
agent's `on_enter` hook owns disclosure ordering before any caller turn can be answered.
"""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass
from pathlib import Path

from livekit import rtc
from livekit.agents import Agent, AgentSession, ConversationItemAddedEvent, JobContext
from livekit.agents.voice.background_audio import AudioConfig, BackgroundAudioPlayer
from livekit.plugins import langchain as lk_langchain

from agnostic_market.agents.capabilities import CapabilityRegistry
from agnostic_market.agents.engine import ReasoningEngine
from agnostic_market.agents.routing_activation import RoutingRecognizerFactory
from agnostic_market.application import (
    ApplicationModels,
    ApplicationSession,
    ApplicationSettings,
    TenantServices,
    build_application_session,
)
from agnostic_market.config.registry import ResolvedConfig
from agnostic_market.dtos.llm import ProviderCredentialsConfig
from agnostic_market.llm.gateway import LLMGateway
from agnostic_market.secrets.base import SecretResolver
from agnostic_market.session import CallerContext
from agnostic_market.tenancy.context import TenantContext
from agnostic_market.voice.graph import GraphVoiceAdapter
from agnostic_market.voice.stt_engine import build_stt
from agnostic_market.voice.tts_engine import build_tts

logger = logging.getLogger(__name__)


class DisclosureFirstAgent(Agent):
    """Agent that speaks the call-start disclosure the moment it enters the session.

    `on_enter` is the session lifecycle hook — it runs before any user turn can be
    answered, so disclosure-first is guaranteed by structure (COMPLIANCE §2), not by the
    worker racing to call say() after session.start(). Uninterruptible: a barged-over
    disclosure was not fully played.

    Dead-air on a slow turn is covered by the non-verbal thinking-SOUND earcon
    (BackgroundAudioPlayer, started in the worker entrypoint), NOT a verbal filler: two
    LiveKit-native verbal approaches (say() from the LLM adapter, and this pre-reply hook)
    were tried and neither played reliably in the live runtime — the earcon is the mechanism
    LiveKit actually drives off the agent 'thinking' state.
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
    application: ApplicationSession
    # Background "thinking" earcon — a subtle typing sound while the graph works, masking the
    # LLM/tool dead-air on turns the deterministic read renderer can't shortcut (catalog
    # search, multi-intent). CONSTRUCTED here from config; STARTED in the worker entrypoint
    # where the room lives (`background_audio.start(room=..., agent_session=...)`).
    background_audio: BackgroundAudioPlayer

    @property
    def engine(self) -> ReasoningEngine:
        return self.application.engine

    @property
    def capability_registry(self) -> CapabilityRegistry:
        return self.application.assembly.capability_registry

    def register_shutdown(self, job_context: JobContext) -> None:
        """Make job shutdown await the caller lifecycle's idempotent teardown."""
        job_context.add_shutdown_callback(self.application.state.caller_context.aclose_session)


async def retire_voice_transport(
    session: AgentSession,
    room: rtc.Room,
    caller_context: CallerContext,
    *,
    timeout_seconds: float,
) -> None:
    """Stop caller work and output, then retire the current room transport."""
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("transport retirement timeout must be positive")
    caller_context.stop_turn_admission()
    try:
        try:
            session.shutdown(drain=False)
        finally:
            async with asyncio.timeout(timeout_seconds):
                await room.disconnect()
    finally:
        await caller_context.aclose_session()


# The thinking earcon: a subtle double-blip (assets/audio/, reproducible via its generator
# script). Not a LiveKit built-in — those are keyboard-typing / ambience / hold-music, none of
# which fits a short "working on it" beat; a soft tone reads warmer under a voice call.
_THINKING_SOUND_PATH = (
    Path(__file__).resolve().parents[3] / "assets" / "audio" / "thinking_beep.wav"
)


def _build_background_audio() -> BackgroundAudioPlayer:
    """The thinking-sound earcon (live-call dead-air fix). A subtle beep auto-plays while the
    agent is in the 'thinking' state and stops when speech starts (LiveKit-managed) — so it
    never overlaps the answer or a confirmation readback (§4a). Non-verbal by design: verbal
    filler ("give me a sec") near a sensitive readback is a §4a collision risk, its own pass.
    No ambient sound (a call is not a storefront). If the asset is missing, degrade to no
    thinking sound rather than crash the session."""
    thinking = (
        AudioConfig(str(_THINKING_SOUND_PATH), volume=0.6)
        if _THINKING_SOUND_PATH.exists()
        else None
    )
    return BackgroundAudioPlayer(thinking_sound=thinking)


async def build_voice_loop(
    tenant: TenantContext,
    resolved: ResolvedConfig,
    credentials: ProviderCredentialsConfig,
    secrets: SecretResolver,
    *,
    deployment_id: str,
    tenant_services: TenantServices,
    routing_recognizer_factory: RoutingRecognizerFactory,
) -> VoiceLoop:
    """Assemble the per-merchant session: engines, graph, tools, disclosure — all from config."""
    config = resolved.config
    if tenant.tenant_id != config.merchant_id or tenant.config_version != resolved.config_version:
        raise ValueError("voice tenant context does not match the resolved configuration")
    if tenant.policy != config.policies.to_policy_context():
        raise ValueError("voice tenant policy does not match the resolved configuration")
    gateway = LLMGateway(credentials, secrets)
    application = await build_application_session(
        tenant,
        ApplicationSettings.from_merchant_config(config),
        ApplicationModels(
            response=gateway.chat_model(config.llm.response),
            reasoning=gateway.chat_model(config.llm.reasoning),
            response_structured_output_method=gateway.structured_output_method(config.llm.response),
        ),
        tenant_services,
        deployment_id=deployment_id,
        routing_factory=routing_recognizer_factory,
    )
    adapter = GraphVoiceAdapter(application.engine)

    session = AgentSession(
        stt=build_stt(config.voice.stt, credentials, secrets),
        # Only graph-authored speakable text reaches TTS — model tokens, node-authored
        # lines, and the confirmation readback; never raw tool output (see graph.py).
        llm=lk_langchain.LLMAdapter(adapter),
        tts=build_tts(config.voice.tts, credentials, secrets),
        # Preemptive generation MUST stay off (explicit, not default-reliant): LiveKit
        # starts the LLM task on the INTERIM transcript (`schedule_speech=False` defers
        # only TTS), and our "LLM" is the stateful graph — a speculative call resumes a
        # HITL interrupt and fires the place/cancel EFFECT before the caller's turn is
        # committed. A preliminary ASR hypothesis is not consent (live call #9, P1).
        #
        # Endpointing max_delay is trialled DOWN from the streaming default (2.5s): a
        # low-confidence end-of-turn prediction waits the full max_delay before committing,
        # which is where the multi-second dead pauses on uncertain turns came from (live
        # call #10). 1.5s is a deliberate floor — the caller uses natural multi-clause
        # speech ("Looks good. Let's go ahead and place the order"), so a tighter delay
        # would cut mid-thought. min_delay stays at the streaming default (0.3s) so
        # confident turns are still snappy. `endpointing`/`preemptive_generation` are the
        # current knobs; the flat `min/max_endpointing_delay` kwargs are deprecated in 1.6.
        turn_handling={
            "preemptive_generation": {"enabled": False},
            "endpointing": {"max_delay": 1.5},
        },
    )
    adapter.attach_session(session)  # §4a fact source (readback-interrupted flag)
    _attach_turn_metrics_logger(session)
    _attach_thread_reaper(session, application.state.caller_context)

    agent = DisclosureFirstAgent(
        # Prompt lives in the graph (F1); empty here is dropped by the adapter, no duplicate.
        instructions="",
        # `.replace`, not `.format`: merchant-authored wording may contain other braces.
        disclosure=config.compliance.call_start_disclosure.replace(
            "{display_name}", config.display_name
        ),
    )
    return VoiceLoop(
        session=session,
        agent=agent,
        application=application,
        background_audio=_build_background_audio(),
    )


def _attach_thread_reaper(session: AgentSession, caller_context: CallerContext) -> None:
    """Clock B (AGENTS §A10 rule 4): on session close, the caller context is torn down
    UNCONDITIONALLY — a dropped call must never leave a resumable placement/refund, a stale
    level, cart, recent-order context, verified identity, or guest order for a reattaching
    session. The per-session stores already die with it; this is belt-and-suspenders. The whole
    teardown lives in `CallerContext.aclose_session` (Fix 5), which is itself idempotent; the
    reaper adds the session-close-specific `flow_abandoned` telemetry (pending interrupt) and a
    re-entrant guard so a double-fired close acts at most once. Nothing is spoken (the caller is
    gone); expiry of a still-connected caller's pending action is Clock A (the confirm node)."""
    reaped = False
    close_tasks: set[asyncio.Task[None]] = set()

    @session.on("close")
    def _reap(_event: object) -> None:
        nonlocal reaped
        if reaped:
            return
        reaped = True

        async def close() -> None:
            await caller_context.aclose_session()
            if caller_context.close_had_pending_interrupt:
                caller_context.telemetry.record(
                    {"event": "flow_abandoned", "reason": "session_closed"}
                )

        def completed(task: asyncio.Task[None]) -> None:
            close_tasks.discard(task)
            if not task.cancelled() and (error := task.exception()) is not None:
                logger.critical(
                    "caller-context close failed",
                    exc_info=(type(error), error, error.__traceback__),
                )

        task = asyncio.create_task(close())
        close_tasks.add(task)
        task.add_done_callback(completed)


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
