"""Tests for the release-please workflow."""

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/release-please.yml"


def test_trigger_releases_can_comment_on_release_pr() -> None:
    """Grant only the permissions needed to dispatch and report releases."""
    workflow = yaml.safe_load(WORKFLOW.read_text())

    assert workflow["jobs"]["trigger-releases"]["permissions"] == {
        "actions": "write",
        "issues": "write",
        "pull-requests": "write",
    }
