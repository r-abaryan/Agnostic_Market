"""Shared evidence-artifact field types and immutable persistence."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, StringConstraints

# Exception class names are Python identifiers, so private and dunder-adjacent names
# from third-party internals are recordable. A stricter pattern would turn the failure
# an evidence envelope exists to record into a validation error that destroys it.
ExceptionTypeName = Annotated[
    str,
    StringConstraints(
        min_length=1,
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*$",
    ),
]


def write_immutable_evidence(path: Path, evidence: BaseModel) -> None:
    """Atomically create one evidence file without replacing an existing result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError("evidence path already exists")
    descriptor, raw_temp = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temp_path = Path(raw_temp)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(evidence.model_dump_json(indent=2))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # A same-directory hard link publishes the complete inode atomically and
        # fails rather than replacing an artifact won by a concurrent writer.
        os.link(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)
