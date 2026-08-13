"""Pure, config-bound admissibility rules for model-authored caller text.

This module validates text only. Graph assembly grants speech authority to nodes, the
stream boundary interprets messages and tool calls, and capability owners establish factual
grounding.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CallerAudibleModelTextPolicy:
    """Reject completed model text that is unsafe or unusable for caller output."""

    max_chars: int

    def __post_init__(self) -> None:
        if self.max_chars <= 0:
            raise ValueError("caller-audible model-text limit must be positive")

    def validate(self, text: str) -> None:
        if len(text) > self.max_chars:
            raise ValueError("caller-audible model text exceeded the configured limit")
        if not any(character.isalnum() for character in text):
            raise ValueError("caller-audible model text contained no lexical content")
