"""Contracts for the CI workflows.

Mostly static assertions over workflow YAML, plus an executable check of the
bounded ripgrep install script: `test_non_release_ripgrep_install_*` runs the
step's real `run:` body against stubbed `sudo`/`timeout`/`dpkg`/`rg` binaries.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
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
FILESYSTEM_TESTS = (
    ROOT / "libs/deepagents/tests/unit_tests/backends/test_filesystem_backend.py"
)

STRICT_STEP = "🔍 Install ripgrep (strict)"
SOFT_STEP = "🔍 Install ripgrep (non-release PR)"
APT_INSTALL = "apt-get update && sudo apt-get install -y ripgrep"
# Shared by the shell producer in `_test.yml` and the JS consumer in
# `ripgrep_timeout_comment.yml`; nothing else couples the two files.
ARTIFACT_PREFIX = "ripgrep-timeout-"
EXPECTED_ENV = "DEEPAGENTS_RIPGREP_EXPECTED=1"


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


def _starts_with(value: Any, prefix: str) -> bool:
    """Mimic the GitHub Actions `startsWith`, which coerces null to ''."""
    return str(value or "").startswith(prefix)


# Longest first: a shorter path must not rewrite part of a longer one.
_CONTEXT_PATHS = sorted(
    (
        "github.event.pull_request.title",
        "github.event_name",
        "github.head_ref",
        "runner.os",
    ),
    key=len,
    reverse=True,
)


def _eval_condition(expression: str, context: dict[str, Any]) -> bool:
    """Evaluate the subset of GitHub Actions expression syntax used by these steps.

    Supports context lookups, `startsWith`, `!`, `&&`, `||`, and parentheses —
    enough to check that the two install steps really are mutually exclusive,
    which substring assertions cannot do.
    """
    expr = " ".join(expression.split())
    for path in _CONTEXT_PATHS:
        expr = expr.replace(path, repr(context.get(path)))
    expr = expr.replace("&&", " and ").replace("||", " or ")
    # Only unary `!`; the lookahead keeps `!=` intact.
    expr = re.sub(r"!\s*(?=startsWith|\()", " not ", expr)
    expr = expr.replace("startsWith(", "_starts_with(")
    return bool(eval(expr, {"__builtins__": {}}, {"_starts_with": _starts_with}))


def _context(
    *,
    event_name: str,
    head_ref: str | None = None,
    title: str | None = None,
    runner_os: str = "Linux",
) -> dict[str, Any]:
    return {
        "github.event_name": event_name,
        "github.head_ref": head_ref,
        "github.event.pull_request.title": title,
        "runner.os": runner_os,
    }


# (label, context, expected step) — "expected step" is the one that must run.
SELECTION_CASES = [
    ("push to main", _context(event_name="push"), STRICT_STEP),
    ("merge queue", _context(event_name="merge_group"), STRICT_STEP),
    (
        "release-please PR",
        _context(
            event_name="pull_request",
            head_ref="release-please--branches--main--components--deepagents",
            title="chore: release main",
        ),
        STRICT_STEP,
    ),
    (
        "manual release PR by title",
        _context(
            event_name="pull_request",
            head_ref="mdrxy/release-prep",
            title="release(deepagents): 1.2.3",
        ),
        STRICT_STEP,
    ),
    (
        "ordinary PR",
        _context(
            event_name="pull_request",
            head_ref="mdrxy/ci/soft-timeout-ripgrep",
            title="fix(code): tighten grep bounds",
        ),
        SOFT_STEP,
    ),
    (
        "ordinary PR on Windows",
        _context(
            event_name="pull_request",
            head_ref="mdrxy/ci/soft-timeout-ripgrep",
            title="fix(code): tighten grep bounds",
            runner_os="Windows",
        ),
        None,
    ),
]


def test_deepagents_code_collects_coverage_on_python_3_14() -> None:
    """Keep all supported runtimes while collecting coverage on Python 3.14."""
    workflow = _load_workflow(CI_WORKFLOW)
    config = workflow["jobs"]["test-code"]["with"]

    assert json.loads(config["python-versions"]) == ["3.12", "3.13", "3.14"]
    assert config["coverage-python-version"] == "3.14"


@pytest.mark.parametrize(
    ("label", "context", "expected"),
    SELECTION_CASES,
    ids=[case[0] for case in SELECTION_CASES],
)
def test_exactly_one_ripgrep_install_step_runs(
    label: str,
    context: dict[str, Any],
    expected: str | None,
) -> None:
    """The strict and soft install paths are mutually exclusive and exhaustive.

    Guards the invariant that substring assertions miss: flipping the strict
    step's `||` to `&&` makes it unsatisfiable, so release runs would install
    no ripgrep at all while every `in step["if"]` assertion still passed.
    """
    workflow = _load_workflow(TEST_WORKFLOW)
    selected = [
        name
        for name in (STRICT_STEP, SOFT_STEP)
        if _eval_condition(_find_step(workflow, job="build", name=name)["if"], context)
    ]

    assert selected == ([expected] if expected else []), (
        f"{label}: expected {expected!r} to run, got {selected!r}"
    )


def test_non_release_pr_ripgrep_install_has_two_minute_timeout() -> None:
    """Ordinary PRs may continue only when ripgrep installation times out."""
    workflow = _load_workflow(TEST_WORKFLOW)
    step = _find_step(workflow, job="build", name=SOFT_STEP)

    # The upload step and the comment workflow both key off this id; a rename
    # would resolve to an empty string and silently disable the whole warning.
    assert step["id"] == "ripgrep-install"
    # `timeout` must run under `sudo`, or it cannot signal the root-owned
    # `apt-get` and the bound silently does nothing.
    assert "sudo timeout --signal=TERM --kill-after=10s 120s" in step["run"]
    assert "continue-on-error" not in step
    assert "timeout-minutes" not in step

    upload = _find_step(
        workflow, job="build", name="📤 Record ripgrep install timeout"
    )
    assert upload["if"] == "steps.ripgrep-install.outputs.timed-out == 'true'"
    assert upload["with"]["name"] == "${{ steps.ripgrep-install.outputs.artifact }}"


def test_only_two_ripgrep_install_steps_exist() -> None:
    """No third install path can drift in alongside the strict/soft pair."""
    workflow = _load_workflow(TEST_WORKFLOW)
    installs = [
        step
        for step in workflow["jobs"]["build"]["steps"]
        if "apt-get install -y ripgrep" in str(step.get("run", ""))
    ]

    assert [step["name"] for step in installs] == [STRICT_STEP, SOFT_STEP]


def _run_install_script(
    script: str,
    tmp_path: Path,
    *,
    timeout_status: int,
    rg_available: bool = False,
) -> tuple[subprocess.CompletedProcess[str], list[str], list[str]]:
    """Execute the install step's `run:` body against stubbed binaries."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    def stub(name: str, body: str) -> None:
        path = bin_dir / name
        path.write_text(f"#!/usr/bin/env bash\n{body}\n")
        path.chmod(0o755)

    # `sudo` execs its arguments so the stubbed `timeout`/`dpkg` are reached.
    # A real `sudo` must never run here: it would hit a password prompt, or
    # actually touch the developer's package manager.
    stub("sudo", 'exec "$@"')
    stub("timeout", f"exit {timeout_status}")
    stub("dpkg", "exit 0")
    stub("rg", "exit 0" if rg_available else "exit 127")

    output = tmp_path / "github-output"
    output.touch()
    env_file = tmp_path / "github-env"
    env_file.touch()
    runner_temp = tmp_path / "runner-temp"
    runner_temp.mkdir()

    result = subprocess.run(
        ["bash", "-c", script],
        check=False,
        capture_output=True,
        text=True,
        # Explicit, minimal environment: inheriting os.environ would let a
        # developer's BASH_ENV/SHELLOPTS leak into the script under test.
        env={
            "GITHUB_ENV": str(env_file),
            "GITHUB_OUTPUT": str(output),
            "MATRIX_OS": "ubuntu-latest",
            "MATRIX_PYTHON": "3.14",
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "RUNNER_TEMP": str(runner_temp),
            "WORKING_DIRECTORY": "libs/code",
        },
    )
    return result, output.read_text().splitlines(), env_file.read_text().splitlines()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell script")
@pytest.mark.parametrize(
    ("timeout_status", "expected_status", "expected_timeout"),
    [
        (0, 0, False),
        (42, 42, False),
        (124, 0, True),
        # `--kill-after` escalates to SIGKILL when apt defers the TERM, which
        # coreutils reports as 137 rather than 124. Softening only 124 would
        # hard-fail on exactly the wedged install this bound exists to absorb.
        (137, 0, True),
    ],
)
def test_non_release_ripgrep_install_only_softens_timeout(
    tmp_path: Path,
    timeout_status: int,
    expected_status: int,
    expected_timeout: bool,
) -> None:
    """The install script preserves success and errors but softens 124 and 137."""
    workflow = _load_workflow(TEST_WORKFLOW)
    step = _find_step(workflow, job="build", name=SOFT_STEP)
    result, output_lines, env_lines = _run_install_script(
        step["run"], tmp_path, timeout_status=timeout_status
    )

    assert result.returncode == expected_status

    # Written exactly once: a duplicated key would leave the upload step
    # depending on undocumented last-write-wins parsing.
    timed_out = [line for line in output_lines if line.startswith("timed-out=")]
    assert timed_out == [f"timed-out={str(expected_timeout).lower()}"]

    if expected_timeout:
        assert "::warning::" in result.stdout
        # Pins the slash-to-dash package transform and the prefix the comment
        # workflow filters on.
        assert f"artifact={ARTIFACT_PREFIX}libs-code-ubuntu-latest-3.14" in output_lines
        marker_lines = [line for line in output_lines if line.startswith("marker=")]
        assert len(marker_lines) == 1
        assert Path(marker_lines[0].partition("=")[2]).is_file()
        # A runner that lost ripgrep must not also claim it is guaranteed.
        assert EXPECTED_ENV not in env_lines
    else:
        assert "::warning::" not in result.stdout
        assert not any(line.startswith("artifact=") for line in output_lines)
        # Only a genuinely successful install promises ripgrep to the tests.
        assert (EXPECTED_ENV in env_lines) is (expected_status == 0)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell script")
def test_timeout_with_usable_ripgrep_is_not_reported(tmp_path: Path) -> None:
    """A bound hit that still left a working `rg` must not warn.

    "The bound was hit" and "ripgrep is missing" are different facts; only the
    second one should reach the PR.
    """
    workflow = _load_workflow(TEST_WORKFLOW)
    step = _find_step(workflow, job="build", name=SOFT_STEP)
    result, output_lines, env_lines = _run_install_script(
        step["run"], tmp_path, timeout_status=124, rg_available=True
    )

    assert result.returncode == 0
    assert [line for line in output_lines if line.startswith("timed-out=")] == [
        "timed-out=false"
    ]
    assert "::warning::" not in result.stdout
    assert "::notice::" in result.stdout
    assert EXPECTED_ENV in env_lines


def test_release_and_non_pr_ripgrep_install_is_strict() -> None:
    """Release PRs, pushes, and merge queues install ripgrep with no timeout."""
    workflow = _load_workflow(TEST_WORKFLOW)
    step = _find_step(workflow, job="build", name=STRICT_STEP)

    assert APT_INSTALL in step["run"]
    assert "timeout" not in step["run"]
    assert "continue-on-error" not in step
    assert "timeout-minutes" not in step
    # Promises ripgrep to the tests, turning a missing `rg` into a failure
    # instead of a silent skip.
    assert EXPECTED_ENV in step["run"]


def test_release_workflow_requires_ripgrep_before_unit_tests() -> None:
    """Release artifact tests install ripgrep without a soft timeout."""
    workflow = _load_workflow(RELEASE_WORKFLOW)
    steps = workflow["jobs"]["pre-release-checks"]["steps"]
    install = _find_step(workflow, job="pre-release-checks", name="Install ripgrep")
    tests = _find_step(workflow, job="pre-release-checks", name="Run unit tests")

    assert steps.index(install) < steps.index(tests)
    assert APT_INSTALL in install["run"]
    assert "timeout" not in install["run"]
    assert "continue-on-error" not in install
    assert "timeout-minutes" not in install
    assert EXPECTED_ENV in install["run"]


def test_missing_ripgrep_fails_loudly_when_ci_promised_it() -> None:
    """The rg-gated tests must not silently skip on a runner that installed rg.

    Those tests are the only coverage of the real-binary grep contract, and one
    of them guards symlink containment, so a silent skip would let a
    containment regression merge green.
    """
    source = FILESYSTEM_TESTS.read_text()

    assert "def require_ripgrep()" in source
    assert 'os.environ.get("DEEPAGENTS_RIPGREP_EXPECTED") == "1"' in source
    assert "pytest.fail(" in source
    # Every rg-gated test routes through the helper rather than skipping inline.
    assert 'pytest.skip("ripgrep not installed")' in source
    assert source.count("require_ripgrep()") >= 6  # definition + call sites


def test_ripgrep_timeout_comment_uses_isolated_workflow_run() -> None:
    """PR comments run after CI without exposing write access to tested code."""
    workflow = _load_workflow(RIPGREP_COMMENT_WORKFLOW)
    assert workflow["on"]["workflow_run"]["workflows"] == ["🔧 CI"]
    assert workflow["permissions"] == {
        "actions": "read",
        "contents": "read",
        "issues": "write",
        "pull-requests": "read",
    }

    job = workflow["jobs"]["manage-comment"]
    # Checking out PR-authored code here would hand it an `issues: write` token.
    assert all(
        "actions/checkout" not in str(step.get("uses", "")) for step in job["steps"]
    )
    assert "github.event.workflow_run.event == 'pull_request'" in job["if"]
    # Cancelled and startup-failed runs produce no artifacts, which would be
    # misread as "no timeout" and delete a valid warning.
    assert "conclusion == 'success'" in job["if"]
    assert "conclusion == 'failure'" in job["if"]

    script = job["steps"][0]["with"]["script"]
    assert "pullRequest.head.sha !== run.head_sha" in script
    assert "listWorkflowRunArtifacts" in script
    assert ARTIFACT_PREFIX in script
    assert "createComment" in script
    assert "updateComment" in script
    assert "deleteComment" in script
    # Fork PRs carry no `workflow_run.pull_requests`; resolving by head ref is
    # the only lookup that works for them.
    assert "pulls.list" in script
    assert "run.head_repository.owner.login" in script
    # An unreportable timeout is a broken mechanism, not a no-op.
    assert "core.setFailed" in script


def test_ripgrep_timeout_comment_concurrency_is_keyed_on_head_sha() -> None:
    """Runs for different commits must not cancel each other.

    Keyed on the branch, a newer run could cancel an older one that was about
    to post, and then discard itself via the stale-SHA guard.
    """
    workflow = _load_workflow(RIPGREP_COMMENT_WORKFLOW)
    group = workflow["concurrency"]["group"]

    assert "github.event.workflow_run.head_sha" in group
    assert "head_branch" not in group
