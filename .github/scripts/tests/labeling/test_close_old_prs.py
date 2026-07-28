"""Pytest shim for the old PR cleanup Node.js tests."""

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]


def test_close_old_prs_node_tests() -> None:
    """Run native Node.js tests for the GitHub workflow helper."""
    subprocess.run(
        ["node", "--test", ".github/scripts/tests/labeling/close-old-prs.test.js"],
        cwd=ROOT,
        check=True,
        text=True,
    )
