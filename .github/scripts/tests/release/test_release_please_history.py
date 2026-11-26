"""Pytest shim for the release-please history Node.js tests."""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def test_release_please_history_node_tests() -> None:
    """Run the native Node.js tests for `.github/release-please/history.mjs`."""
    result = subprocess.run(
        ["node", "--test", ".github/scripts/tests/release/release-please-history.test.mjs"],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"Node tests failed (exit {result.returncode}):\n"
            f"{result.stdout}\n{result.stderr}"
        )
