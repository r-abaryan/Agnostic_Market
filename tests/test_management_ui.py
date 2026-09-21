"""Executable browser-client contracts using Node's dependency-free test runner."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def test_management_ui_javascript_contracts() -> None:
    node = shutil.which("node")
    assert node is not None, "Node 20 or newer is required for management UI tests"
    test_file = Path(__file__).parent / "js" / "test_management_ui.mjs"

    completed = subprocess.run(  # noqa: S603
        (node, "--test", str(test_file)),
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
