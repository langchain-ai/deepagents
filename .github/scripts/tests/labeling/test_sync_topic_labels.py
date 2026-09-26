"""Pytest shim for topic manifest generation and workflow tests."""

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]


def test_sync_topic_labels_node_tests() -> None:
    """Run native Node.js tests for the topic manifest workflow."""
    subprocess.run(
        ["node", "--test", ".github/scripts/tests/labeling/sync-topic-labels.test.js"],
        cwd=ROOT,
        check=True,
        text=True,
    )
