"""Pytest shim for the curated release-notes Node.js tests."""

import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
AUTOMATION_WORKFLOW = ROOT / ".github/workflows/dcode_release_notes.yml"
CHECK_WORKFLOW = ROOT / ".github/workflows/dcode_release_notes_check.yml"
RELEASE_PLEASE_WORKFLOW = ROOT / ".github/workflows/release-please.yml"


def test_dcode_release_notes_node_tests() -> None:
    """Run native Node.js tests for the GitHub workflow helper."""
    result = subprocess.run(
        ["node", "--test", ".github/scripts/dcode-release-notes.test.js"],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    # Surface the node output (including failing test names) in the pytest report
    # instead of a bare CalledProcessError with no context.
    if result.returncode != 0:
        raise AssertionError(
            f"Node tests failed (exit {result.returncode}):\n"
            f"{result.stdout}\n{result.stderr}"
        )


def _load_workflow(path: Path) -> dict:
    workflow = yaml.safe_load(path.read_text())
    workflow["on"] = workflow.pop(True, workflow.get("on"))
    return workflow


def test_required_check_uses_default_branch_workflow_definition() -> None:
    """Keep the required gate controlled by the default branch."""
    workflow = _load_workflow(CHECK_WORKFLOW)

    assert set(workflow["on"]) == {
        "issue_comment",
        "pull_request_target",
        "workflow_dispatch",
    }
    assert set(workflow["on"]["issue_comment"]["types"]) == {
        "created",
        "edited",
        "deleted",
    }
    assert workflow["permissions"] == {
        "checks": "write",
        "contents": "read",
        "issues": "write",
        "pull-requests": "read",
    }
    job = workflow["jobs"]["curated-release-notes"]
    checkout = job["steps"][0]
    assert checkout["with"]["ref"] == "main"
    assert checkout["with"]["persist-credentials"] is False
    assert "environment" not in job
    check_workflow = CHECK_WORKFLOW.read_text()
    assert "head_sha: pr.head.sha" in check_workflow
    assert "expectedHead: pr.head.sha" in check_workflow
    assert "name: 'curated release notes'" in check_workflow
    # The check reports the `curated release notes` context on every PR (a pass for
    # non-release PRs, full validation for the release PR) so it can be a required
    # status check on main without blocking unrelated PRs.
    assert "isReleaseBranchPr(pr)" in check_workflow
    assert check_workflow.count("checks.create") >= 2
    release_please = RELEASE_PLEASE_WORKFLOW.read_text()
    assert "--ref main" in release_please
    assert "needs.update-lockfiles.result == 'success'" not in release_please


def test_mutation_workflow_commands_are_target_only() -> None:
    """Prevent untrusted fork comments from reaching repository mutations."""
    workflow = _load_workflow(AUTOMATION_WORKFLOW)

    triggers = workflow["on"]
    assert set(triggers) == {"pull_request_target", "issue_comment"}
    assert set(triggers["pull_request_target"]["types"]) == {"ready_for_review"}
    assert workflow["jobs"]["validate"]["permissions"] == {
        "contents": "read",
        "issues": "write",
        "pull-requests": "read",
    }

    automation = AUTOMATION_WORKFLOW.read_text()
    # Both draft and apply failures must surface on the PR, not only as a red run,
    # so a maintainer who issued the command learns why it did not take effect.
    assert "postDraftFailure" in automation
    assert "postApplyFailure" in automation
    # Untrusted release content is fetched at the validated SHA; it is never
    # checked out into a privileged job or passed to shell Git commands.
    assert "path: release-pr" not in automation
    assert "working-directory: release-pr" not in automation
    assert "git push" not in automation
    assert "createApplyCommit" in automation
