"""Pytest shim for the PR labeler helper Node.js tests."""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def test_pr_labeler_node_tests() -> None:
    """Run native Node.js tests for the shared PR labeling helpers."""
    subprocess.run(
        ["node", "--test", ".github/scripts/tests/labeling/pr-labeler.test.js"],
        cwd=ROOT,
        check=True,
        text=True,
    )
