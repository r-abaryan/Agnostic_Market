"""TurnEvent / TurnFacts — the ReasoningEngine <-> voice-plane contract (AGENTS §A0).

The engine (Plane 2) yields TurnEvents; the voice adapter (Plane 1) renders them to
speech. Events carry TEXT the GRAPH authored — the engine never composes a spoken string
(one-author rule). Facts flow the other way: the voice plane passes TurnFacts in with
each turn; the engine relays them into the graph (e.g. into a Command resume payload) and
never fetches anything from the voice plane itself — the dependency arrow is voice->engine
only, which is what keeps the engine mockable.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

_FROZEN = ConfigDict(extra="forbid", frozen=True)


class TokenEvent(BaseModel):
    """A streamed model token (the normal answer path) — speak as it arrives."""

    model_config = _FROZEN

    kind: Literal["token"] = "token"
    text: str


class SpokenMessageEvent(BaseModel):
    """A node-authored, non-streamed caller-facing line (deferral, checkout outcome).

    Emitted only for nodes the graph declares speakable — `node` names the author.
    """

    model_config = _FROZEN

    kind: Literal["spoken_message"] = "spoken_message"
    text: str
    node: str = Field(min_length=1)


class InterruptEvent(BaseModel):
    """The graph paused for HITL confirmation; `prompt` is the graph-authored readback
    line (the `interrupt()` payload) — speak it, then feed the caller's next committed
    turn back as a resume."""

    model_config = _FROZEN

    kind: Literal["interrupt"] = "interrupt"
    prompt: str = Field(min_length=1)


TurnEvent = TokenEvent | SpokenMessageEvent | InterruptEvent


class TurnFacts(BaseModel):
    """Perception-layer facts the voice plane asserts about THIS turn's input.

    `readback_interrupted`: the caller barged over the pending confirmation readback
    before it finished playing — a confirmation given over a truncated readback is not
    consent (VOICE_PIPELINE §4a); the confirm node re-confirms instead of placing.
    """

    model_config = _FROZEN

    readback_interrupted: bool = False
