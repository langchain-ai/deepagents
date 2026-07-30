"""Pytest shim for the curated release-notes Node.js tests."""

import json
import re
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[4]
AUTOMATION_WORKFLOW = ROOT / ".github/workflows/release_notes.yml"
CHECK_WORKFLOW = ROOT / ".github/workflows/release_notes_check.yml"
RELEASE_PLEASE_WORKFLOW = ROOT / ".github/workflows/release-please.yml"


def test_release_notes_node_tests() -> None:
    """Run native Node.js tests for the GitHub workflow helper."""
    result = subprocess.run(
        [
            "node",
            "--test",
            ".github/scripts/tests/release/release-notes.test.js",
            ".github/scripts/tests/release/draft-release-notes.test.js",
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


def test_workflows_reference_helper_scripts_that_exist() -> None:
    """Catch a helper rename that misses a workflow reference.

    Both workflows load their helpers from a `trusted-source` checkout of `main`,
    so a path that does not exist fails at runtime rather than at lint time. The
    check workflow is the worst case: a bad path there breaks a required check.
    """
    referenced = set()
    for workflow in (AUTOMATION_WORKFLOW, CHECK_WORKFLOW):
        for match in re.finditer(
            r"\./trusted-source/(\.github/scripts/\S+?\.js)\b", workflow.read_text()
        ):
            referenced.add(match.group(1))
    assert referenced, "expected the workflows to reference helper scripts"
    missing = sorted(path for path in referenced if not (ROOT / path).is_file())
    assert not missing, f"workflows reference helper scripts that do not exist: {missing}"


def test_release_notes_cover_every_release_please_component() -> None:
    """Keep the curated-notes gate component-agnostic across the whole repo."""
    dispatch = yaml.safe_load(RELEASE_PLEASE_WORKFLOW.read_text())["jobs"][
        "dispatch-release-notes-check"
    ]
    step = next(
        step for step in dispatch["steps"] if "release_notes_check.yml" in step["run"]
    )
    # A hardcoded component here would silently skip every other package's release
    # PR, which is exactly the scoping this workflow moved away from. Only the
    # executable lines matter; comments may name a component as an example.
    code = "\n".join(
        line
        for line in step["run"].splitlines()
        if not line.lstrip().startswith("#")
    )
    config = json.loads((ROOT / "release-please-config.json").read_text())
    for component in (
        meta["component"] for meta in config["packages"].values()
    ):
        assert component not in code, (
            f"dispatch filter is scoped to {component}; it must match any component"
        )

    # The helper derives targets from release-please-config.json, so every managed
    # component must resolve to its own changelog and release branch.
    resolved = json.loads(
        subprocess.run(
            [
                "node",
                "-e",
                "const m = require('./.github/scripts/release/release-notes.js');"
                "const out = {};"
                "for (const [k, v] of m.componentRegistry()) out[k] = v;"
                "process.stdout.write(JSON.stringify(out));",
            ],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        ).stdout
    )
    expected = {
        meta["component"]: f"{path}/{meta.get('changelog-path', 'CHANGELOG.md')}"
        for path, meta in config["packages"].items()
    }
    assert {k: v["changelogPath"] for k, v in resolved.items()} == expected
    for component, target in resolved.items():
        assert target["releaseBranch"] == (
            f"release-please--branches--main--components--{component}"
        )


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
    # should-run output, pinned to the release-bot environment, and read-only for
    # contents. Dropping the gate would let the App-token jobs run without the
    # permission/identity check; widening permissions would be privilege escalation.
    for job_name in ("draft", "apply"):
        job = workflow["jobs"][job_name]
        assert "needs.validate.outputs.should-run == 'true'" in job["if"]
        assert job["environment"] == "release-bot"
        assert job["permissions"] == {"contents": "read"}
        app_token = next(
            step for step in job["steps"] if step.get("id") == "app-token"
        )
        assert app_token["uses"] == (
            "actions/create-github-app-token@"
            "bcd2ba49218906704ab6c1aa796996da409d3eb1"
        )
        assert app_token["with"] == {
            "client-id": "${{ vars.ORG_MEMBERSHIP_APP_CLIENT_ID }}",
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

    # No long-lived bot PAT: repository mutations go through short-lived App tokens.
    # This also covers the old DCODE_RELEASE_BOT_TOKEN name as a substring.
    assert "RELEASE_BOT_TOKEN" not in automation

    # Untrusted release text goes through a deterministic one-request helper, not
    # an agent/tool loop. Only the selected model key is placed in that
    # process, under a provider-neutral variable that the model never sees.
    draft_step = next(
        step
        for step in workflow["jobs"]["draft"]["steps"]
        if step.get("id") == "draft-model"
    )
    assert "uses" not in draft_step
    # The step wraps the helper only to capture its stderr into a step output, so
    # the drafting-failure comment can report why the run failed instead of just
    # "outcome: failure". That wrapper must not become a place to run other work:
    # assert the helper is the single command that does anything externally
    # visible, and that the only thing written is the step-output file.
    draft_run = draft_step["run"]
    helper_invocation = (
        "node ./trusted-source/.github/scripts/release/draft-release-notes.js"
    )
    assert draft_run.count(helper_invocation) == 1
    draft_commands = "\n".join(
        line
        for line in draft_run.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    assert draft_commands.count("node ") == 1
    for forbidden in ("git ", "curl", "wget", "npm ", "npx ", "pip ", "eval ", "source "):
        assert forbidden not in draft_commands
    # Only $GITHUB_OUTPUT is written; nothing is appended to the environment or
    # the step summary, and no other file is created.
    assert draft_commands.count(">>") == 1
    assert '>> "$GITHUB_OUTPUT"' in draft_commands
    assert "GITHUB_ENV" not in draft_commands
    assert "GITHUB_PATH" not in draft_commands
    # The captured message reaches the failure comment through this output.
    failure_step = next(
        step
        for step in workflow["jobs"]["draft"]["steps"]
        if step.get("name") == "Comment on drafting failure"
    )
    assert failure_step["env"]["DRAFT_ERROR"] == (
        "${{ steps.draft-model.outputs.error }}"
    )
    assert "process.env.DRAFT_ERROR" in failure_step["with"]["script"]
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
    helper = (ROOT / ".github/scripts/release/draft-release-notes.js").read_text()
    assert "child_process" not in helper
    assert "https://api.openai.com/v1/chat/completions" in helper
    assert "https://api.anthropic.com/v1/messages" in helper
    assert "https://generativelanguage.googleapis.com/" in helper
