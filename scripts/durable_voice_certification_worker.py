"""Run the isolated synthetic worker that produces durable voice evidence."""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from pathlib import Path

from dotenv import load_dotenv
from livekit import agents

from agnostic_market.durability.voice_certification import (
    VoiceCertificationTarget,
    answer_certification_request,
    load_voice_certification_target,
)

if __package__:
    from . import voice_agent
else:
    import voice_agent

load_dotenv()

_CERTIFICATION_CONFIG_ENV = "VOICE_AGENT_CERTIFICATION_CONFIG"


def _is_configuration_free_command(arguments: Sequence[str]) -> bool:
    return (
        not arguments
        or any(argument in {"-h", "--help"} for argument in arguments)
        or arguments[0] == "download-files"
    )


def _certification_target(arguments: Sequence[str]) -> VoiceCertificationTarget | None:
    if _is_configuration_free_command(arguments):
        return None
    if arguments[0] == "console":
        raise RuntimeError("voice certification requires a network worker command")
    raw_path = os.environ.get(_CERTIFICATION_CONFIG_ENV, "").strip()
    if not raw_path:
        raise RuntimeError(f"{_CERTIFICATION_CONFIG_ENV} must identify an absolute file path")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        raise RuntimeError(f"{_CERTIFICATION_CONFIG_ENV} must be an absolute path")
    target = load_voice_certification_target(path)
    return target


def _worker_options(arguments: Sequence[str]) -> agents.WorkerOptions:
    target = _certification_target(arguments)
    if target is None:
        return agents.WorkerOptions(
            entrypoint_fnc=voice_agent.entrypoint,
            prewarm_fnc=voice_agent._prewarm_for_arguments(arguments),
            agent_name=voice_agent._agent_name(arguments),
        )

    async def certification_entrypoint(ctx: agents.JobContext) -> None:
        await voice_agent.entrypoint(ctx, certification_target=target)

    async def certification_request(request: agents.JobRequest) -> None:
        await answer_certification_request(request, target)

    return agents.WorkerOptions(
        entrypoint_fnc=certification_entrypoint,
        request_fnc=certification_request,
        prewarm_fnc=voice_agent._prewarm_for_arguments(arguments),
        agent_name=target.certification_agent_name,
    )


def main() -> None:
    agents.cli.run_app(_worker_options(tuple(sys.argv[1:])))


if __name__ == "__main__":
    main()
