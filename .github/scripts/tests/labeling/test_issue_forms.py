"""Pytest shim for the issue form label Node.js tests."""

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]


def test_issue_forms_node_tests() -> None:
    """Run native Node.js tests binding issue forms to their label consumers."""
    subprocess.run(
        ["node", "--test", ".github/scripts/tests/labeling/issue-forms.test.js"],
        cwd=ROOT,
        check=True,
        text=True,
    )
