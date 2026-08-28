"""Contracts for the new-Markdown-file pull request gate."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[4]
WORKFLOW = ROOT / ".github" / "workflows" / "markdown_file_check.yml"
GITHUB_SCRIPT_PIN = "actions/github-script@3a2844b7e9c422d3c10d287c895573f7108da1b3"
CHECKOUT_PIN = "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1"


def _load_workflow() -> dict:
    workflow = yaml.safe_load(WORKFLOW.read_text())
    workflow["on"] = workflow.pop(True, workflow.get("on"))
    return workflow


def _steps() -> list[dict]:
    return _load_workflow()["jobs"]["markdown-file-check"]["steps"]


def _script() -> str:
    return _steps()[1]["with"]["script"]


def test_markdown_gate_reruns_for_title_file_and_label_changes() -> None:
    workflow = _load_workflow()

    assert set(workflow["on"]["pull_request_target"]["types"]) == {
        "opened",
        "edited",
        "synchronize",
        "reopened",
        "labeled",
        "unlabeled",
    }
    assert workflow["permissions"] == {"contents": "read", "pull-requests": "write"}


def test_markdown_gate_runs_only_base_revision_code() -> None:
    checkout, check = _steps()

    assert checkout["uses"] == CHECKOUT_PIN
    assert checkout["with"] == {
        "ref": "${{ github.event.pull_request.base.sha }}",
        "persist-credentials": False,
        "sparse-checkout": ".github/scripts/checks/markdown_file_check.js",
    }
    assert check["uses"] == GITHUB_SCRIPT_PIN
    assert "require('./.github/scripts/checks/markdown_file_check.js')" in _script()


def test_markdown_gate_uses_live_files_labels_and_fail_closed_checks() -> None:
    script = _script()

    assert "github.paginate(github.rest.pulls.listFiles" in script
    assert "typeof pullRequest.changed_files !== 'number'" in script
    assert "files.length !== pullRequest.changed_files" in script
    assert "markdown-added: acknowledged" in script
    assert "listLabelsOnIssue" in script
    assert "<!-- markdown-file-check -->" in script


def test_markdown_gate_sets_failure_before_comment_write() -> None:
    script = _script()

    failure = script.index("core.setFailed(`Non-docs PR adds")
    comment = script.index("await upsertStickyComment(body)", failure)
    assert failure < comment
