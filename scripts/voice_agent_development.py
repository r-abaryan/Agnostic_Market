"""Run the isolated, non-authorizing LiveKit development voice worker."""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence

from livekit import agents

if __package__:
    from . import voice_agent
else:
    import voice_agent

_DEVELOPMENT_AGENT_SUFFIX = "-development"


def _require_development_command(arguments: Sequence[str]) -> None:
    if not arguments or arguments[0] != "dev":
        raise RuntimeError("the development voice worker accepts only the LiveKit dev command")
    if not os.environ.get("VOICE_AGENT_MERCHANT_ID", "").strip():
        raise RuntimeError("VOICE_AGENT_MERCHANT_ID is required for development network admission")


def _worker_options(arguments: Sequence[str]) -> agents.WorkerOptions:
    _require_development_command(arguments)
    production_name = voice_agent._agent_name(arguments)
    return agents.WorkerOptions(
        entrypoint_fnc=voice_agent.development_network_entrypoint,
        prewarm_fnc=voice_agent._prewarm_for_arguments(arguments),
        agent_name=f"{production_name}{_DEVELOPMENT_AGENT_SUFFIX}",
    )


def main() -> None:
    arguments = tuple(sys.argv[1:])
    agents.cli.run_app(_worker_options(arguments))


if __name__ == "__main__":
    main()
