"""Phase-2 voice worker — serve one merchant's minimal voice loop (BUILD_PLAN Phase 2).

Run:
    uv run python scripts/voice_agent.py console   # local mic/speaker dev loop
    uv run python scripts/voice_agent.py dev       # LiveKit Cloud -> Playground (the live call)

Needs in .env: DEEPGRAM_API_KEY, CARTESIA_API_KEY, ANTHROPIC_API_KEY (routing LLM),
LIVEKIT_URL/LIVEKIT_API_KEY/LIVEKIT_API_SECRET (dev mode; read by the LiveKit SDK).
Merchant served: env VOICE_AGENT_MERCHANT_ID (default "acme_store").

Exit check (Phase 2): the call opens with the disclosure (played first, uninterruptible,
COMPLIANCE 2), answers "what's the status of order ORD-1001" from the fixture, and logs
per-turn latency (see voice/pipeline.py). ASCII-only output.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from livekit import agents

from agnostic_market.config.registry import ConfigRegistry
from agnostic_market.llm.gateway import load_provider_credentials
from agnostic_market.llm.providers import (
    ConformanceRegistry,
    check_llm_certification,
    load_conformance_targets,
)
from agnostic_market.secrets.env_resolver import EnvSecretResolver
from agnostic_market.voice.pipeline import build_voice_loop

if __package__:
    from .close_evidence_recorder import (
        CloseEvidenceRecorder,
        load_close_certification_request,
    )
else:
    from close_evidence_recorder import (
        CloseEvidenceRecorder,
        load_close_certification_request,
    )

# .env must be in the process env BEFORE the LiveKit worker starts (it reads LIVEKIT_URL
# at startup) and BEFORE the module-level env read below — so load at import, not in main
# (job subprocesses import this module without executing the __main__ block).
load_dotenv()

_CONFIG_ROOT = Path(__file__).resolve().parents[1] / "config"
_MERCHANT_ID = os.environ.get("VOICE_AGENT_MERCHANT_ID", "acme_store")

logger = logging.getLogger("voice_agent")


async def entrypoint(ctx: agents.JobContext) -> None:
    close_certification = load_close_certification_request(_CONFIG_ROOT)
    secrets = EnvSecretResolver()
    credentials = load_provider_credentials(_CONFIG_ROOT / "base" / "providers.yaml")
    registry = ConfigRegistry(_CONFIG_ROOT).load()
    resolved = registry.get(_MERCHANT_ID)

    # Config-time certification check (warn-only in Phase 1/2; Phase 4 gate makes it blocking).
    targets = load_conformance_targets(_CONFIG_ROOT / "conformance" / "targets.yaml")
    conformance = ConformanceRegistry(
        _CONFIG_ROOT / "conformance" / "reports.json",
        max_report_age_days=targets.max_report_age_days,
    )
    for warning in check_llm_certification(resolved.config, conformance):
        logger.warning("certification: %s", warning)

    loop = build_voice_loop(resolved, credentials, secrets, config_root=_CONFIG_ROOT)
    logger.info(
        "serving merchant %s (config_version %s)", _MERCHANT_ID, resolved.config_version[:12]
    )

    await ctx.connect()
    if close_certification is not None:
        # Certification is intentionally single-participant and opt-in. Resolve the linked
        # participant before AgentSession.start so disconnect evidence never depends on
        # LiveKit's set-backed close-listener ordering.
        participant = await ctx.wait_for_participant()
        close_recorder = CloseEvidenceRecorder(
            close_certification,
            merchant_id=_MERCHANT_ID,
        )
        close_recorder.attach(
            session=loop.session,
            room=ctx.room,
            engine=loop.engine,
            linked_participant_identity=participant.identity,
        )
        ctx.add_shutdown_callback(close_recorder.wait_for_completion)
    # The disclosure (COMPLIANCE 2 / EU AI Act Art. 50(1)) plays via the agent's own
    # on_enter hook - structurally first, before any user turn can be answered.
    await loop.session.start(loop.agent, room=ctx.room)
    # The thinking-sound earcon needs the room (a runtime concern); start it after the
    # session. Auto-plays while the agent is 'thinking', stops when it speaks (no overlap with
    # the answer or a readback). No-op/warn in console mode (LiveKit-managed).
    await loop.background_audio.start(room=ctx.room, agent_session=loop.session)
    # Stop the earcon's mixer on shutdown, or its background task throws
    # "Event loop is closed" when the loop tears down (a dangling task on disconnect).
    ctx.add_shutdown_callback(loop.background_audio.aclose)


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
