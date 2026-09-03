"""Platform authority passed from trusted admission to session ownership."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, StringConstraints

AuthorityIdentifier = Annotated[
    str,
    StringConstraints(
        min_length=1,
        pattern=r"^\S+$",
    ),
]


class TransportAuthority(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    provider: Literal["livekit"]
    assignment_id: AuthorityIdentifier
    worker_id: AuthorityIdentifier


class AdmittedSessionAuthority(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    logical_session_id: AuthorityIdentifier
    transport: TransportAuthority
