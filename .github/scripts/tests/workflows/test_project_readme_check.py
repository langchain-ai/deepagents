"""Contract tests for the blocking project README gate."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[4]
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"


def _workflow() -> dict:
    """Load the CI workflow while normalizing PyYAML's YAML 1.1 `on` key."""
    workflow = yaml.safe_load(CI_WORKFLOW.read_text())
    workflow["on"] = workflow.pop(True, workflow.get("on"))
    return workflow


def _gate() -> dict:
    """Return the project README job."""
    return _workflow()["jobs"]["project-readme-check"]


def test_gate_is_part_of_required_ci_success() -> None:
    """The repository's required aggregate check cannot pass while this gate is red."""
    workflow = _workflow()
    assert "project-readme-check" in workflow["jobs"]["ci_success"]["needs"]
    assert _gate()["if"] == "github.event_name == 'pull_request'"
    assert set(workflow["on"]["pull_request"]["types"]) >= {
        "edited",
        "labeled",
        "unlabeled",
    }


def test_gate_executes_only_trusted_base_code() -> None:
    """The detector is sparsely checked out from the base revision."""
    steps = _gate()["steps"]
    checkout = next(
        step for step in steps if step.get("uses", "").startswith("actions/checkout@")
    )
    assert checkout["with"] == {
        "ref": "${{ github.event.pull_request.base.sha }}",
        "path": ".readme-check-base",
        "sparse-checkout": ".github/scripts/checks/check_project_readmes.py\n",
        "persist-credentials": False,
    }
    run = "\n".join(step.get("run", "") for step in steps)
    assert (
        'detector=".readme-check-base/.github/scripts/checks/check_project_readmes.py"'
        in run
    )


def test_gate_collects_complete_rename_aware_paths() -> None:
    """The changed-file producer fails closed and includes old rename paths."""
    text = CI_WORKFLOW.read_text()
    assert "files.length !== pr.changed_files" in text
    assert "core.setFailed(`Changed-file list is incomplete" in text
    assert "file.previous_filename" in text
    assert "changed_files.json" in text


def test_gate_reads_live_labels_and_reconciles_sticky() -> None:
    """Label state comes from the API and one sticky comment tracks the result."""
    text = CI_WORKFLOW.read_text()
    assert "github.rest.issues.listLabelsOnIssue" in text
    assert "labels.some(label => label.name === ACKNOWLEDGMENT_LABEL)" in text
    assert "<!-- project-readme-check -->" in text
    assert "github.rest.issues.createComment" in text
    assert "github.rest.issues.updateComment" in text
    assert "github.rest.issues.deleteComment" in text
    assert "readme: acknowledged" in text
    assert "Project README edits require a docs PR" in text
