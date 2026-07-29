"""Tests for the release-please workflow."""

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[4]
WORKFLOW = ROOT / ".github/workflows/release-please.yml"


def _load_workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text())


def _needs(job: dict) -> set[str]:
    needs = job.get("needs", [])
    if isinstance(needs, str):
        return {needs}
    return set(needs)


def _condition(job: dict) -> str:
    return " ".join(str(job.get("if", "")).split())


def test_trigger_releases_can_comment_on_release_pr() -> None:
    """Grant only the permissions needed to dispatch and report releases."""
    workflow = _load_workflow()

    assert workflow["jobs"]["trigger-releases"]["permissions"] == {
        "actions": "write",
        "issues": "read",
        "pull-requests": "write",
    }


def test_release_dispatch_precedes_release_please_maintenance() -> None:
    """Publish dispatch stays first; maintenance waits after successful dispatch."""
    workflow = _load_workflow()
    jobs = workflow["jobs"]

    assert "concurrency" not in workflow

    trigger = jobs["trigger-releases"]
    assert _needs(trigger) == {"detect-release-commit"}

    guard = jobs["guard-pending-release"]
    assert _needs(guard) == {
        "guard-empty-commit",
        "detect-release-commit",
        "trigger-releases",
    }

    guard_if = _condition(guard)
    assert "!cancelled()" in guard_if
    assert "needs.guard-empty-commit.result == 'success'" in guard_if
    assert "needs.detect-release-commit.result == 'success'" in guard_if
    assert "needs.detect-release-commit.outputs.release-commit == 'false'" in guard_if
    assert "needs.detect-release-commit.outputs.release-commit == 'true'" in guard_if
    assert "needs.trigger-releases.result == 'success'" in guard_if
    # Malformed detector output must fail closed (no loose != 'true').
    assert "release-commit != 'true'" not in guard_if

    release_please = jobs["release-please"]
    assert "guard-pending-release" in _needs(release_please)
    assert "trigger-releases" not in _needs(release_please)
    assert _condition(release_please) == (
        "needs.guard-pending-release.outputs.skip == 'false'"
    )
    # Maintenance may run after release commits once the guard authorizes it.
    assert "release-commit" not in _condition(release_please)
    assert "skip != 'true'" not in _condition(release_please)
    assert "||" not in _condition(release_please)

    assert "release-please" in _needs(jobs["update-lockfiles"])
