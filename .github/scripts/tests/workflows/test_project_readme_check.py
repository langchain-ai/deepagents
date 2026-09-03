"""Contract tests for the blocking project README gate workflow.

The gate's *behavior* is tested in
`.github/scripts/tests/checks/readme-gate.test.js` against a fake octokit.
What cannot be tested there, and is tested here, is the wiring: the triggers
the gate needs to be re-runnable, the checkout that keeps a PR from editing
its own detector, and the shell step that chooses which detector to run.

These deliberately parse the YAML rather than grepping it. A substring
assertion cannot tell a working gate from one whose `core.setFailed` became
`core.warning`, which is the exact mutation that silently disables the check.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[4]
WORKFLOW = ROOT / ".github" / "workflows" / "project_readme_check.yml"
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
CHECKS_DIR = ROOT / ".github" / "scripts" / "checks"


def _load(path: Path) -> dict:
    """Load a workflow while normalizing PyYAML's YAML 1.1 `on` key.

    PyYAML parses a bare `on:` as the boolean True, not the string "on".
    """
    workflow = yaml.safe_load(path.read_text())
    workflow["on"] = workflow.pop(True, workflow.get("on"))
    return workflow


def _job() -> dict:
    """Return the project README job definition."""
    return _load(WORKFLOW)["jobs"]["project-readme-check"]


def _step(name_fragment: str) -> dict:
    """Return the first step whose name contains `name_fragment`."""
    return next(
        step for step in _job()["steps"] if name_fragment in step.get("name", "")
    )


def test_gate_subscribes_to_the_events_that_clear_it() -> None:
    """Both escape hatches must re-run the check.

    The gate is cleared by retitling the PR to `docs:` (`edited`) or by
    applying the acknowledgment label (`labeled`), and re-armed by removing it
    (`unlabeled`). Without these trigger types a contributor applies the label
    the failure message names, nothing re-runs, and the required check stays
    red with no way to clear it short of an empty commit.
    """
    types = set(_load(WORKFLOW)["on"]["pull_request"]["types"])
    assert {"edited", "labeled", "unlabeled"} <= types
    assert {"opened", "synchronize", "reopened"} <= types


def test_gate_lives_outside_ci_yml() -> None:
    """The gate must not drag its trigger types into the main CI workflow.

    `on:` is workflow-wide. Putting this gate in ci.yml would make all of its
    lint/test jobs fire on every label change repo-wide, and because ci.yml
    sets `cancel-in-progress` on a group keyed by `github.ref` (shared by
    every event on a PR), applying the acknowledgment label would cancel and
    restart the whole in-flight matrix.
    """
    ci = _load(CI_WORKFLOW)
    assert "project-readme-check" not in ci["jobs"]
    assert "project-readme-check" not in ci["jobs"]["ci_success"]["needs"]
    # Default types only: no `edited`/`labeled`/`unlabeled` on the big workflow.
    assert ci["on"]["pull_request"] is None


def test_gate_runs_only_trusted_base_code() -> None:
    """The detector is sparse-checked-out from the base branch.

    Under `pull_request`, `actions/checkout` defaults to the untrusted PR
    merge commit. Reading the detector from there would let a PR edit
    `check_project_readmes.py` to print an empty result and self-bypass the
    gate -- defeating it for the exact PRs it exists to block.
    """
    checkout = _step("Checkout detector from trusted base branch")
    assert checkout["with"]["ref"] == "${{ github.base_ref }}", (
        "resolve the base branch tip, not the payload's pinned base.sha: "
        "label/edit events do not refresh that snapshot"
    )
    assert checkout["with"]["persist-credentials"] is False
    sparse = checkout["with"]["sparse-checkout"]
    assert "check_project_readmes.py" in sparse
    assert "readme-gate.js" in sparse


def test_pr_title_is_never_interpolated_into_the_shell() -> None:
    """An author-controlled title must reach the detector via env, not `${{ }}`."""
    detect = _step("Detect protected README edits")
    assert detect["env"]["PR_TITLE"] == "${{ github.event.pull_request.title }}"
    assert "${{" not in detect["run"]
    assert '"$PR_TITLE"' in detect["run"]


def _run_resolve_step(tmp_path: Path, *, base: bool, pr: bool) -> subprocess.CompletedProcess:
    """Execute the "Resolve gate sources" step with the given sources present."""
    for present, prefix in ((base, ".readme-check-base"), (pr, ".readme-check-pr")):
        if not present:
            continue
        target = tmp_path / prefix / ".github" / "scripts" / "checks"
        target.mkdir(parents=True)
        for name in ("check_project_readmes.py", "readme-gate.js"):
            shutil.copy(CHECKS_DIR / name, target / name)
    return subprocess.run(
        ["bash", "-c", _step("Resolve gate sources")["run"]],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin"},
    )


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell script")
def test_resolve_prefers_base_sources(tmp_path: Path) -> None:
    """With the detector on the base branch, the PR's copy is never used."""
    result = _run_resolve_step(tmp_path, base=True, pr=True)
    assert result.returncode == 0, result.stderr
    assert "bootstrap window" not in result.stdout
    assert (tmp_path / ".readme-gate" / "check_project_readmes.py").is_file()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell script")
def test_resolve_falls_back_to_pr_sources_during_bootstrap(tmp_path: Path) -> None:
    """Before the gate lands on main, the PR's own copy runs -- and warns."""
    result = _run_resolve_step(tmp_path, base=False, pr=True)
    assert result.returncode == 0, result.stderr
    assert "::warning::" in result.stdout
    assert (tmp_path / ".readme-gate" / "readme-gate.js").is_file()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell script")
def test_resolve_fails_closed_when_the_detector_is_gone(tmp_path: Path) -> None:
    """A deleted or renamed detector reds the check instead of passing silently.

    This is the bound on the bootstrap fallback: without it, removing
    `check_project_readmes.py` from the repo would disable the gate for every
    future PR behind nothing but a `::warning::` nobody reads.
    """
    result = _run_resolve_step(tmp_path, base=False, pr=False)
    assert result.returncode != 0
    assert "::error::" in result.stdout
    assert not (tmp_path / ".readme-gate").exists()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell script")
def test_detect_step_writes_the_detector_verdict_to_github_output(
    tmp_path: Path,
) -> None:
    """The detect step's heredoc lands a parseable verdict in `$GITHUB_OUTPUT`."""
    gate_dir = tmp_path / ".readme-gate"
    gate_dir.mkdir()
    shutil.copy(CHECKS_DIR / "check_project_readmes.py", gate_dir)
    (tmp_path / "changed_files.json").write_text(json.dumps(["README.md"]))
    output = tmp_path / "github-output"
    output.touch()

    result = subprocess.run(
        ["bash", "-c", _step("Detect protected README edits")["run"]],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        env={
            "GITHUB_OUTPUT": str(output),
            "PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin",
            "PR_TITLE": "feat(code): touch the landing page",
        },
    )
    assert result.returncode == 0, result.stderr
    lines = output.read_text().splitlines()
    assert lines[0] == "result<<__README_EOF__"
    assert json.loads(lines[1]) == {"pr_type": "feat", "readmes": ["README.md"]}
    assert lines[2] == "__README_EOF__"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell script")
def test_detect_step_fails_closed_on_malformed_input(tmp_path: Path) -> None:
    """`set -euo pipefail` must propagate the detector's exit 2 into a red step."""
    gate_dir = tmp_path / ".readme-gate"
    gate_dir.mkdir()
    shutil.copy(CHECKS_DIR / "check_project_readmes.py", gate_dir)
    (tmp_path / "changed_files.json").write_text('{"README.md": true}')
    output = tmp_path / "github-output"
    output.touch()

    result = subprocess.run(
        ["bash", "-c", _step("Detect protected README edits")["run"]],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        env={
            "GITHUB_OUTPUT": str(output),
            "PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin",
            "PR_TITLE": "fix(code): whatever",
        },
    )
    assert result.returncode != 0
    assert output.read_text() == "", "no verdict may reach GITHUB_OUTPUT"
