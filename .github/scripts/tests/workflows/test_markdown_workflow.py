"""Structural contracts for the new-Markdown-file pull request gate.

The gate's *behavior* is tested in
`.github/scripts/tests/checks/markdown_file_check.test.js`, which runs the real
`run()` against a fake octokit. This module covers only what lives in YAML and
therefore cannot be executed: the trigger list, the permissions, the trusted
checkout, and the thinness of the inline script itself.
"""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[4]
WORKFLOW = ROOT / ".github" / "workflows" / "markdown_file_check.yml"
DETECTOR = ".github/scripts/checks/markdown_file_check.js"
GITHUB_SCRIPT_PIN = "actions/github-script@3a2844b7e9c422d3c10d287c895573f7108da1b3"
CHECKOUT_PIN = "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1"


def _load_workflow() -> dict:
    workflow = yaml.safe_load(WORKFLOW.read_text())
    # PyYAML resolves the bare `on:` key to the boolean True.
    workflow["on"] = workflow.pop(True, workflow.get("on"))
    return workflow


def _steps() -> list[dict]:
    return _load_workflow()["jobs"]["markdown-file-check"]["steps"]


def _script() -> str:
    return _steps()[1]["with"]["script"]


def test_markdown_gate_reruns_for_title_file_and_label_changes() -> None:
    """Without `labeled`/`unlabeled` the bypass label needs a push to take effect."""
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
    """The detector must come from base, never from the PR's own head.

    This job holds `pull-requests: write` under `pull_request_target`. A head
    checkout would both run PR-authored code against that token and let a PR
    edit the detector to pass itself.
    """
    checkout, check = _steps()

    assert checkout["uses"] == CHECKOUT_PIN
    assert checkout["with"]["ref"] == "${{ github.event.pull_request.base.sha }}"
    assert checkout["with"]["persist-credentials"] is False
    assert checkout["with"]["sparse-checkout"].strip() == DETECTOR
    # Cone mode is for directory prefixes; declare the single-file intent.
    assert checkout["with"]["sparse-checkout-cone-mode"] is False
    assert check["uses"] == GITHUB_SCRIPT_PIN


def test_markdown_gate_delegates_to_the_tested_module() -> None:
    """Logic in the YAML string cannot be executed by any test, so keep it out.

    Anything that moves back into the `script:` block silently loses its
    coverage: an inverted condition there still ships green.
    """
    script = _script()

    assert "require(detector)" in script
    assert DETECTOR in script
    assert "await run({ github, context, core })" in script

    # Allow the existence guard and the require, nothing resembling a decision.
    for forbidden in ("setFailed", "listFiles", "listLabelsOnIssue", "createComment"):
        assert forbidden not in script, (
            f"{forbidden} belongs in {DETECTOR}, where it can be tested"
        )


def test_markdown_gate_tolerates_a_base_without_the_detector() -> None:
    """Every PR open when this merges has a base.sha predating the detector.

    Without the guard those runs die on an unhandled module-resolution error:
    an opaque red X clearable only by rebasing.
    """
    script = _script()

    assert "fs.existsSync(detector)" in script
    # Warning, not notice: a detector renamed on base without updating this
    # path silently disarms the gate, which must surface in the Checks UI.
    assert "core.warning(" in script
