"""Pytest shim for the priority label sync Node.js tests."""

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]


def test_priority_labels_node_tests() -> None:
    """Run native Node.js tests for sync_priority_labels.yml."""
    subprocess.run(
        ["node", "--test", ".github/scripts/tests/labeling/priority-labels.test.js"],
        cwd=ROOT,
        check=True,
        text=True,
    )
