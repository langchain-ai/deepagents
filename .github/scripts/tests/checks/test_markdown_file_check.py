"""Pytest shim for the Markdown-file gate Node.js tests."""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def test_markdown_file_check_node_tests() -> None:
    """Run native Node.js tests for the Markdown-file gate detector."""
    result = subprocess.run(
        ["node", "--test", ".github/scripts/tests/checks/markdown_file_check.test.js"],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    # Surface the node output instead of a bare CalledProcessError with no
    # context, matching test_release_notes.py.
    if result.returncode != 0:
        raise AssertionError(
            f"node --test failed ({result.returncode})\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
