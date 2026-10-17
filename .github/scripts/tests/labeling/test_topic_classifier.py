"""Pytest shim for topic classifier Node.js tests."""

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]


def test_topic_classifier_node_tests() -> None:
    """Run native Node.js tests for topic classification."""
    subprocess.run(
        [
            "node",
            "--test",
            ".github/scripts/tests/labeling/topic-classifier.test.js",
            ".github/scripts/tests/labeling/semif-topic-classifier.test.js",
        ],
        cwd=ROOT,
        check=True,
        text=True,
    )
