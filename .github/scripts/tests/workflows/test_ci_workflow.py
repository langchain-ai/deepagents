"""Static contracts for the main CI workflow."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[4]
WORKFLOWS = ROOT / ".github" / "workflows"
CI_WORKFLOW = WORKFLOWS / "ci.yml"
TEST_WORKFLOW = WORKFLOWS / "_test.yml"
RELEASE_WORKFLOW = WORKFLOWS / "release.yml"
RIPGREP_COMMENT_WORKFLOW = WORKFLOWS / "ripgrep_timeout_comment.yml"


def _load_workflow(path: Path) -> dict[str, Any]:
    """Load a workflow while normalizing PyYAML's YAML 1.1 `on` key."""
    workflow = yaml.safe_load(path.read_text())
    workflow["on"] = workflow.pop(True, workflow.get("on"))
    return workflow


def _find_step(workflow: dict[str, Any], *, job: str, name: str) -> dict[str, Any]:
    """Return one named workflow step."""
    matches = [step for step in workflow["jobs"][job]["steps"] if step.get("name") == name]
    assert len(matches) == 1, f"expected one {name!r} step, found {len(matches)}"
    return matches[0]


def test_deepagents_code_collects_coverage_on_python_3_14() -> None:
    """Keep all supported runtimes while collecting coverage on Python 3.14."""
    workflow = _load_workflow(CI_WORKFLOW)
    config = workflow["jobs"]["test-code"]["with"]

    assert json.loads(config["python-versions"]) == ["3.12", "3.13", "3.14"]
    assert config["coverage-python-version"] == "3.14"


def test_non_release_pr_ripgrep_install_has_two_minute_timeout() -> None:
    """Ordinary PRs may continue only when ripgrep installation times out."""
    workflow = _load_workflow(TEST_WORKFLOW)
    step = _find_step(
        workflow,
        job="build",
        name="🔍 Install ripgrep (non-release PR)",
    )

    assert "github.event_name == 'pull_request'" in step["if"]
    assert "!startsWith(github.head_ref, 'release-please--')" in step["if"]
    assert "!startsWith(github.event.pull_request.title, 'release(')" in step["if"]
    assert "timeout --signal=TERM --kill-after=10s 120s" in step["run"]
    assert 'if [ "$status" -eq 124 ]' in step["run"]
    assert "continue-on-error" not in step
    assert "timeout-minutes" not in step
    upload = _find_step(
        workflow,
        job="build",
        name="📤 Record ripgrep install timeout",
    )
    assert upload["if"] == "steps.ripgrep-install.outputs.timed-out == 'true'"
    assert upload["with"]["name"] == "${{ steps.ripgrep-install.outputs.artifact }}"


@pytest.mark.parametrize(
    ("timeout_status", "expected_status", "expected_timeout"),
    [(0, 0, False), (42, 42, False), (124, 0, True)],
)
def test_non_release_ripgrep_install_only_softens_timeout(
    tmp_path: Path,
    timeout_status: int,
    expected_status: int,
    expected_timeout: bool,
) -> None:
    """The install script preserves success and errors but softens exit 124."""
    workflow = _load_workflow(TEST_WORKFLOW)
    step = _find_step(
        workflow,
        job="build",
        name="🔍 Install ripgrep (non-release PR)",
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    timeout_stub = bin_dir / "timeout"
    timeout_stub.write_text(f"#!/usr/bin/env bash\nexit {timeout_status}\n")
    timeout_stub.chmod(0o755)
    output = tmp_path / "github-output"
    output.touch()
    runner_temp = tmp_path / "runner-temp"
    runner_temp.mkdir()

    result = subprocess.run(
        ["bash", "-c", step["run"]],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "GITHUB_OUTPUT": str(output),
            "MATRIX_OS": "ubuntu-latest",
            "MATRIX_PYTHON": "3.14",
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "RUNNER_TEMP": str(runner_temp),
            "WORKING_DIRECTORY": "libs/code",
        },
    )

    assert result.returncode == expected_status
    output_lines = output.read_text().splitlines()
    assert ("timed-out=true" in output_lines) is expected_timeout
    if expected_timeout:
        assert "::warning::" in result.stdout
        marker_lines = [line for line in output_lines if line.startswith("marker=")]
        assert len(marker_lines) == 1
        assert Path(marker_lines[0].partition("=")[2]).is_file()
    else:
        assert not any(line.startswith("artifact=") for line in output_lines)


def test_release_and_non_pr_ripgrep_install_is_strict() -> None:
    """Release PRs, pushes, and merge queues use the strict install path."""
    workflow = _load_workflow(TEST_WORKFLOW)
    step = _find_step(workflow, job="build", name="🔍 Install ripgrep (strict)")

    assert "github.event_name != 'pull_request'" in step["if"]
    assert "startsWith(github.head_ref, 'release-please--')" in step["if"]
    assert "startsWith(github.event.pull_request.title, 'release(')" in step["if"]
    assert step["run"] == "sudo apt-get update && sudo apt-get install -y ripgrep"
    assert "continue-on-error" not in step
    assert "timeout-minutes" not in step


def test_release_workflow_requires_ripgrep_before_unit_tests() -> None:
    """Release artifact tests install ripgrep without a soft timeout."""
    workflow = _load_workflow(RELEASE_WORKFLOW)
    steps = workflow["jobs"]["pre-release-checks"]["steps"]
    install = _find_step(workflow, job="pre-release-checks", name="Install ripgrep")
    tests = _find_step(workflow, job="pre-release-checks", name="Run unit tests")

    assert steps.index(install) < steps.index(tests)
    assert install["run"] == "sudo apt-get update && sudo apt-get install -y ripgrep"
    assert "continue-on-error" not in install
    assert "timeout-minutes" not in install


def test_ripgrep_timeout_comment_uses_isolated_workflow_run() -> None:
    """PR comments run after CI without exposing write access to tested code."""
    workflow = _load_workflow(RIPGREP_COMMENT_WORKFLOW)
    assert workflow["on"]["workflow_run"]["workflows"] == ["🔧 CI"]
    assert workflow["permissions"] == {
        "actions": "read",
        "issues": "write",
        "pull-requests": "read",
    }
    job = workflow["jobs"]["manage-comment"]
    assert job["if"] == "github.event.workflow_run.event == 'pull_request'"
    assert all("actions/checkout" not in str(step.get("uses", "")) for step in job["steps"])
    script = job["steps"][0]["with"]["script"]
    assert "listPullRequestsAssociatedWithCommit" in script
    assert "pullRequest.head.sha !== run.head_sha" in script
    assert "listWorkflowRunArtifacts" in script
    assert "ripgrep-timeout-" in script
    assert "createComment" in script
    assert "updateComment" in script
    assert "deleteComment" in script
