"""Pytest shim for the PR lifecycle Node.js tests."""

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]


def test_lifecycle_workflows_node_tests() -> None:
    """Run native Node.js tests for the GitHub lifecycle workflows."""
    subprocess.run(
        ["node", "--test", ".github/scripts/tests/labeling/lifecycle-workflows.test.js"],
        cwd=ROOT,
        check=True,
        text=True,
    )
