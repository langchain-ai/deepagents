"""Pytest shim for the Markdown-file gate Node.js tests."""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def test_markdown_file_check_node_tests() -> None:
    subprocess.run(
        ["node", "--test", ".github/scripts/tests/checks/markdown_file_check.test.js"],
        cwd=ROOT,
        check=True,
        text=True,
    )
