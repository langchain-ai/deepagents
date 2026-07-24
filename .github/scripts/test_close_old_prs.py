"""Pytest shim for the old PR cleanup Node.js tests."""

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_close_old_prs_node_tests() -> None:
    """Run native Node.js tests for the GitHub workflow helper."""
    subprocess.run(
        ["node", "--test", ".github/scripts/close-old-prs.test.js"],
        cwd=ROOT,
        check=True,
        text=True,
    )
