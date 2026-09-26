"""Run release-label compatibility tests, including the workflow shell scripts."""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def test_release_label_compatibility() -> None:
    """Exercise legacy, canonical, and mixed release states without a network."""
    subprocess.run(
        ["node", "--test", ".github/scripts/tests/release/release-labels.test.js"],
        cwd=ROOT,
        check=True,
        text=True,
    )
