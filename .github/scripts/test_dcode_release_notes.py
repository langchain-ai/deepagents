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
        [
            "node",
            "--test",
            ".github/scripts/dcode-release-notes.test.js",
            ".github/scripts/draft-dcode-release-notes.test.js",
        ],
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


def test_required_check_is_attached_to_the_validated_pr_head() -> None:
    """Attach native PR checks and explicit refreshes to the release head."""
    workflow = _load_workflow(CHECK_WORKFLOW)

    assert set(workflow["on"]) == {
        "issue_comment",
        "pull_request",
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
    assert "github.event_name == 'pull_request'" in job["name"]
    assert "curated release notes" in job["name"]
    assert job["timeout-minutes"] == 15
    checkout = job["steps"][0]
    assert checkout["with"]["ref"] == "main"
    assert checkout["with"]["persist-credentials"] is False
    assert "environment" not in job
    check_workflow = CHECK_WORKFLOW.read_text()
    assert "expectedHead: pr.head.sha" in check_workflow
    assert "initialDraftPollAttempts: context.eventName === 'issue_comment' ? 0 : 72" in check_workflow
    # A cancelled/timed-out poll must not leave the refresh check spinning forever:
    # an always() finalizer closes an interrupted in_progress check.
    assert "if: always() && steps.validate.outputs.refresh_check_id != ''" in check_workflow
    assert "github.rest.checks.get" in check_workflow
    assert "head_sha: pr.head.sha" in check_workflow
    assert "github.rest.checks.create" in check_workflow
    assert "github.rest.checks.update" in check_workflow
    # Pull-request runs publish the native required status. Refresh triggers only
    # create a head check for the same-repository release PR, after the strict
    # release-branch identity check has passed.
    assert check_workflow.index("if (!isReleaseBranchPr(pr))") < check_workflow.index(
        "github.rest.checks.create"
    )
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

    # Untrusted release text goes through a deterministic one-request helper, not
    # dcode's agent/tool loop. Only the selected model key is placed in that
    # process, under a provider-neutral variable that the model never sees.
    draft_step = next(
        step
        for step in workflow["jobs"]["draft"]["steps"]
        if step.get("id") == "draft-model"
    )
    assert "uses" not in draft_step
    assert draft_step["run"] == (
        "node ./trusted-source/.github/scripts/draft-dcode-release-notes.js"
    )
    assert set(draft_step["env"]) == {
        "INPUT_FILE",
        "MODEL_API_KEY",
        "MODEL_SPEC",
        "OUTPUT_FILE",
    }
    selected_key = draft_step["env"]["MODEL_API_KEY"]
    assert "secrets.OPENAI_API_KEY" in selected_key
    assert "secrets.ANTHROPIC_API_KEY" in selected_key
    assert "secrets.GOOGLE_API_KEY" in selected_key
    assert "./trusted-source" not in {
        step.get("uses") for step in workflow["jobs"]["draft"]["steps"]
    }
    helper = (ROOT / ".github/scripts/draft-dcode-release-notes.js").read_text()
    assert "child_process" not in helper
    assert "https://api.openai.com/v1/chat/completions" in helper
    assert "https://api.anthropic.com/v1/messages" in helper
    assert "https://generativelanguage.googleapis.com/" in helper
