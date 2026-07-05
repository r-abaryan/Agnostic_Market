"""LLM plane — agnostic gateway + provider conformance gate (BUILD_PLAN Phase 1).

- gateway.py   : provider:model selection -> a LangChain chat model, keys via SecretResolver.
- providers.py : conformance suite (tool-call + structured-output + streaming) + the
                 fail-closed commerce gate the Phase-3 router will call (AGENTS §A11).
"""

from agnostic_market.llm.gateway import GatewayError, LLMGateway, load_provider_credentials
from agnostic_market.llm.providers import (
    SUITE_VERSION,
    ChatOnlyModelError,
    ConformanceRegistry,
    ConformanceRunError,
    check_llm_certification,
    load_conformance_targets,
    run_conformance,
)

__all__ = [
    "SUITE_VERSION",
    "ChatOnlyModelError",
    "ConformanceRegistry",
    "ConformanceRunError",
    "GatewayError",
    "LLMGateway",
    "check_llm_certification",
    "load_conformance_targets",
    "load_provider_credentials",
    "run_conformance",
]
