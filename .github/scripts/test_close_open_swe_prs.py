"""Pytest shim for the open-swe PR cleanup Node.js tests."""

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_close_open_swe_prs_node_tests() -> None:
    """Run native Node.js tests for the GitHub workflow helper."""
    subprocess.run(
        ["node", "--test", ".github/scripts/close-open-swe-prs.test.js"],
        cwd=ROOT,
        check=True,
        text=True,
    )
