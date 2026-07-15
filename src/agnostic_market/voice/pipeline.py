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
import uuid
from dataclasses import dataclass
from pathlib import Path

from livekit.agents import Agent, AgentSession, ConversationItemAddedEvent
from livekit.agents.voice.background_audio import AudioConfig, BackgroundAudioPlayer
from livekit.plugins import langchain as lk_langchain

from agnostic_market.agents.engine import ReasoningEngine, build_checkpointer
from agnostic_market.agents.frontline import build_frontline_graph
from agnostic_market.agents.telemetry import write_event
from agnostic_market.agents.tooling import wrap_readonly_tool
from agnostic_market.commerce.cart import CartStore
from agnostic_market.commerce.orders import LastOrderPointer, OrderStore, load_orders_fixture
from agnostic_market.commerce.profile import ProfileStore, load_profile_fixture
from agnostic_market.commerce.verification import OtpProvider, RiskProvider, VerificationStore
from agnostic_market.config.registry import ResolvedConfig
from agnostic_market.dtos.llm import ProviderCredentialsConfig
from agnostic_market.llm.gateway import LLMGateway
from agnostic_market.secrets.base import SecretResolver
from agnostic_market.voice.graph import GraphVoiceAdapter
from agnostic_market.voice.stt_engine import build_stt
from agnostic_market.voice.tools import build_voice_tools
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
    engine: ReasoningEngine
    # Background "thinking" earcon — a subtle typing sound while the graph works, masking the
    # LLM/tool dead-air on turns the deterministic read renderer can't shortcut (catalog
    # search, multi-intent). CONSTRUCTED here from config; STARTED in the worker entrypoint
    # where the room lives (`background_audio.start(room=..., agent_session=...)`).
    background_audio: BackgroundAudioPlayer


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


def build_voice_loop(
    resolved: ResolvedConfig,
    credentials: ProviderCredentialsConfig,
    secrets: SecretResolver,
    *,
    config_root: Path,
) -> VoiceLoop:
    """Assemble the per-merchant session: engines, graph, tools, disclosure — all from config."""
    config = resolved.config
    store = OrderStore(load_orders_fixture(config_root, config.merchant_id))
    # The session cart (Group B): mutable working state, built once here like OrderStore and
    # reaped with the session. The SAME instance feeds BOTH the read-only view_cart tool and
    # the cart flow (below) — pass one instance to both, or the frontline reads a different
    # cart than the flow mutates (split-brain).
    cart_store = CartStore()
    # Per-session step-up seams (AGENTS §A4a): built once here, like OrderStore, and torn
    # down with the session (no cross-session state — the durable/keyed form lands in Phase
    # 4). The store shares the SAME otp the dispatch node uses (verify + dispatch agree).
    otp = OtpProvider()
    verification_store = VerificationStore(otp)
    risk = RiskProvider()
    # Per-session profile SoR (Group C) — fixture-backed like OrderStore; the profile-change
    # flow's effect target. Dies with the session.
    profile_store = ProfileStore(load_profile_fixture(config_root, config.merchant_id))
    # Session "that order" pointer (Group C L4): the SAME instance feeds the order_status
    # tool (set on a found read) and the flows (set on place/cancel/return) — one instance
    # to both, or references resolve against different state (split-brain, like the cart).
    pointer = LastOrderPointer()
    gateway = LLMGateway(credentials, secrets)
    # Read-only tools pass through the audit/tenant wrapper; the graph owns its own
    # system prompts + few-shot (F1), so the Agent below carries NO instructions.
    tools = [
        wrap_readonly_tool(t, config.merchant_id)
        for t in build_voice_tools(store, cart_store, pointer)
    ]
    graph = build_frontline_graph(
        chat_model=gateway.chat_model(config.llm.routing),
        read_only_tools=tools,
        display_name=config.display_name,
        tenant_id=config.merchant_id,
        # Checkout runs on the reasoning tier (AGENTS §A11: big model for gated flows).
        reasoning_model=gateway.chat_model(config.llm.reasoning),
        store=store,
        cart_store=cart_store,
        # The ONE config->runtime policy mapping (dtos/config.py to_policy_context) — a new
        # policy field lands there once, never per-site.
        policy=config.policies.to_policy_context(),
        # Support step-up seams (3c): the verification level lives in verification_store and
        # is read LIVE inside the support guardrail — never a checkpointed channel, so a
        # replayed checkpoint can't re-grant a level (§A388).
        verification_store=verification_store,
        otp=otp,
        risk=risk,
        profile_store=profile_store,
        pointer=pointer,
        # The checkout/support HITL interrupts need a durable thread (in-memory for the build
        # phase; the Redis saver is a constructor swap at deploy). The serde trusts our
        # checkpointed DTOs (build_checkpointer) — no 'unregistered type' warning.
        checkpointer=build_checkpointer(),
    )
    engine = ReasoningEngine(graph, thread_id=uuid.uuid4().hex)
    adapter = GraphVoiceAdapter(engine)

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
    _attach_thread_reaper(session, engine, verification_store, cart_store, pointer)

    agent = DisclosureFirstAgent(
        # Prompt lives in the graph (F1); empty here is dropped by the adapter, no duplicate.
        instructions="",
        # `.replace`, not `.format`: merchant-authored wording may contain other braces.
        disclosure=config.compliance.call_start_disclosure.replace(
            "{display_name}", config.display_name
        ),
    )
    return VoiceLoop(
        session=session, agent=agent, engine=engine,
        background_audio=_build_background_audio(),
    )


def _attach_thread_reaper(
    session: AgentSession,
    engine: ReasoningEngine,
    verification_store: VerificationStore,
    cart_store: CartStore,
    pointer: LastOrderPointer,
) -> None:
    """Clock B (AGENTS §A10 rule 4): on session close, the thread is reaped UNCONDITIONALLY
    — a dropped call must never leave a resumable placement/refund. Re-entrant-safe: a
    double-fired close deletes once and emits the abandoned event at most once. Nothing is
    spoken (the caller is gone); expiry of a still-connected caller's pending action is
    Clock A, owned by the graph's confirm node. The verification grant AND the cart are
    cleared too, so a reattaching session can never inherit a stale L2 or a stale cart
    (belt-and-suspenders — the per-session stores already die with the session)."""
    reaped = False

    @session.on("close")
    def _reap(_event: object) -> None:
        nonlocal reaped
        if reaped:
            return
        reaped = True
        if engine.pending_interrupt():
            write_event({"event": "flow_abandoned", "reason": "session_closed"})
        engine.delete_thread()
        verification_store.clear()
        cart_store.clear()
        pointer.clear()


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
