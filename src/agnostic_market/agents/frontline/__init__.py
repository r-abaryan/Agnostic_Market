"""The frontline agent (Tier 1) + the reasoning graph that wires in every gated flow.

- graph.py  : builds the graph and its capability registry, returned together as one
              `FrontlineGraphAssembly` so runtime and evaluator share one registry instance.
              Also owns the handover tool, deferral copy, and speakable-node declaration. CODE.
- prompt.py : the frontline model-facing instructions + contrastive few-shot. PROMPT.
"""

from agnostic_market.agents.frontline.graph import (
    FRONTLINE_SPEAKABLE_NODES,
    MODEL_SPEECH_NODES,
    TRANSACTIONAL_MODEL_NODES,
    FrontlineGraphAssembly,
    build_frontline_graph,
)

__all__ = [
    "FRONTLINE_SPEAKABLE_NODES",
    "MODEL_SPEECH_NODES",
    "TRANSACTIONAL_MODEL_NODES",
    "FrontlineGraphAssembly",
    "build_frontline_graph",
]
