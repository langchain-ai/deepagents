"""Pytest shim for the project README gate Node.js tests."""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def test_readme_gate_node_tests() -> None:
    """Run native Node.js tests for the README gate helper."""
    subprocess.run(
        ["node", "--test", ".github/scripts/tests/checks/readme-gate.test.js"],
        cwd=ROOT,
        check=True,
        text=True,
    )
