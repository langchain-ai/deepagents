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
        "contents": "read",
        "issues": "write",
        "pull-requests": "read",
    }
    job = workflow["jobs"]["curated-release-notes"]
    assert job["name"] == "curated release notes"
    checkout = job["steps"][0]
    assert checkout["with"]["ref"] == "main"
    assert checkout["with"]["persist-credentials"] is False
    assert "environment" not in job
    check_workflow = CHECK_WORKFLOW.read_text()
    assert "expectedHead: pr.head.sha" in check_workflow
    assert "head_sha: pr.head.sha" not in check_workflow
    assert "github.rest.checks" not in check_workflow
    # The workflow job itself reports the required `curated release notes` status.
    # Non-release PRs exit successfully, including fork PRs whose head SHAs cannot
    # receive base-repository check runs through the Checks API.
    assert "isReleaseBranchPr(pr)" in check_workflow
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

    # The privileged draft/apply jobs must stay gated on the validate job's
    # should-run output, pinned to the release-dcode environment, and read-only for
    # contents. Dropping the gate would let the App-token jobs run without the
    # permission/identity check; widening permissions would be privilege escalation.
    for job_name in ("draft", "apply"):
        job = workflow["jobs"][job_name]
        assert "needs.validate.outputs.should-run == 'true'" in job["if"]
        assert job["environment"] == "release-dcode"
        assert job["permissions"] == {"contents": "read"}
        app_token = next(
            step for step in job["steps"] if step.get("id") == "app-token"
        )
        assert app_token["uses"] == (
            "actions/create-github-app-token@"
            "bcd2ba49218906704ab6c1aa796996da409d3eb1"
        )
        assert app_token["with"] == {
            "client-id": "${{ secrets.ORG_MEMBERSHIP_APP_CLIENT_ID }}",
            "private-key": "${{ secrets.ORG_MEMBERSHIP_APP_PRIVATE_KEY }}",
            "permission-contents": "write",
            "permission-issues": "write",
            "permission-pull-requests": "write",
        }
        privileged_steps = [
            step
            for step in job["steps"]
            if step.get("uses", "").startswith("actions/github-script@")
            and "github-token" in step.get("with", {})
        ]
        assert privileged_steps
        assert all(
            step["with"]["github-token"] == "${{ steps.app-token.outputs.token }}"
            for step in privileged_steps
        )
        assert all(
            step["env"]["APP_SLUG"] == "${{ steps.app-token.outputs.app-slug }}"
            for step in privileged_steps
        )
        assert all(
            "appSlug: process.env.APP_SLUG" in step["with"]["script"]
            for step in privileged_steps
        )

    assert "DCODE_RELEASE_BOT_TOKEN" not in automation

    # The drafting agent must stay sandboxed: no GitHub token and shell/MCP/
    # interpreter/memory off. Flipping shell_allow_list to a non-empty value would
    # hand a prompt-injected agent shell access, so assert the containment by value.
    agent_step = next(
        step
        for step in workflow["jobs"]["draft"]["steps"]
        if step.get("uses") == "./trusted-source"
    )
    agent_with = agent_step["with"]
    assert agent_with["github_token"] == ""
    assert agent_with["shell_allow_list"] == ""
    assert agent_with["no_mcp"] == "true"
    assert agent_with["interpreter"] == "false"
    assert agent_with["enable_memory"] == "false"
