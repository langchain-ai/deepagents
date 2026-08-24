"""Contracts for the CI workflows.

Mostly static assertions over workflow YAML. The parts that can silently
disable a gate are executed instead of grepped: every ripgrep install step
(`_test.yml`'s soft and strict steps, and `release.yml`'s) has its real `run:`
body driven through `_run_install_script` against stubbed
`sudo`/`timeout`/`dpkg`/`rg`/`apt-get` binaries, and the label step that arms
the strict step's bypass runs against a stubbed `gh` in
`test_ripgrep_bypass_step_behaviour`.
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
RESOLVE_STEP = "🏷️ Resolve ripgrep bypass"
# What makes a PR release-sensitive. Spelled once here and asserted against
# every place that re-implements it: two `if:` conditions in `_test.yml` and
# the JS predicate in `ripgrep_timeout_comment.yml`, which decides whether the
# posted comment describes a timeout or a bypass.
RELEASE_PR_PREDICATES = ("release-please--", "release(")
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
    matches = [
        step for step in workflow["jobs"][job]["steps"] if step.get("name") == name
    ]
    assert len(matches) == 1, f"expected one {name!r} step, found {len(matches)}"
    return matches[0]


def _assert_install_is_unbounded(run: str) -> None:
    """No `timeout` may wrap the apt install itself.

    Both strict steps legitimately contain a `timeout` — the bounded
    `dpkg --configure -a` unwind on the failure path — so the absence of any
    one literal spelling proves nothing. Everything before `status=$?` is the
    install; that region must have no `timeout` at all, whatever the duration.
    """
    install, separator, _ = run.partition("status=$?")
    assert separator, "expected the install to capture its status in `status=$?`"
    assert "timeout" not in install, (
        f"the apt install is bounded by a timeout:\n{install}"
    )


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
    """Ordinary PRs may continue only when ripgrep installation times out.

    Also pins the shared upload step, which is wired to *both* install steps:
    the assertions about `ripgrep-strict` below are load-bearing, not leftovers.
    """
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

    upload = _find_step(workflow, job="build", name="📤 Record ripgrep install timeout")
    # Both the soft step (genuine timeout) and the strict step (bypassed
    # release-PR failure) feed the same marker artifact to the comment workflow.
    assert "steps.ripgrep-install.outputs.timed-out == 'true'" in upload["if"]
    assert "steps.ripgrep-strict.outputs.timed-out == 'true'" in upload["if"]
    assert (
        upload["with"]["name"]
        == "${{ steps.ripgrep-install.outputs.artifact || steps.ripgrep-strict.outputs.artifact }}"
    )


def test_only_two_ripgrep_install_steps_exist() -> None:
    """No third install path can drift in alongside the strict/soft pair."""
    workflow = _load_workflow(TEST_WORKFLOW)
    installs = [
        step
        for step in workflow["jobs"]["build"]["steps"]
        if "apt-get install -y ripgrep" in str(step.get("run", ""))
    ]

    assert [step["name"] for step in installs] == [STRICT_STEP, SOFT_STEP]


def _summary_path(tmp_path: Path) -> Path:
    """Where `_run_install_script` points `GITHUB_STEP_SUMMARY`."""
    return tmp_path / "github-step-summary"


def _run_install_script(
    script: str,
    tmp_path: Path,
    *,
    timeout_status: int,
    rg_available: bool = False,
    apt_status: int | None = None,
    bypass: str = "",
    skip_ripgrep_check: str = "",
) -> tuple[subprocess.CompletedProcess[str], list[str], list[str]]:
    """Execute the install step's `run:` body against stubbed binaries.

    The soft step routes apt through `sudo timeout ... bash -c '...apt-get...'`,
    so `timeout_status` drives its outcome. The strict step calls
    `sudo apt-get ...` directly, so `apt_status` (when set) stubs `apt-get`
    itself; `bypass` feeds the step's `BYPASS` env (the resolved PR label).
    `release.yml`'s step is the same shape, driven by `SKIP_RIPGREP_CHECK`
    instead; its step-summary writes land in `summary_path`.
    """
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
    if apt_status is not None:
        stub("apt-get", f"exit {apt_status}")

    output = tmp_path / "github-output"
    output.touch()
    env_file = tmp_path / "github-env"
    env_file.touch()
    runner_temp = tmp_path / "runner-temp"
    runner_temp.mkdir()
    summary = _summary_path(tmp_path)
    summary.touch()

    result = subprocess.run(
        ["bash", "-c", script],
        check=False,
        capture_output=True,
        text=True,
        # Explicit, minimal environment: inheriting os.environ would let a
        # developer's BASH_ENV/SHELLOPTS leak into the script under test.
        env={
            "BYPASS": bypass,
            "GITHUB_ENV": str(env_file),
            "GITHUB_OUTPUT": str(output),
            "GITHUB_STEP_SUMMARY": str(summary),
            "MATRIX_OS": "ubuntu-latest",
            "MATRIX_PYTHON": "3.14",
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "RUNNER_TEMP": str(runner_temp),
            "SKIP_RIPGREP_CHECK": skip_ripgrep_check,
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


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell script")
@pytest.mark.parametrize(
    (
        "bypass",
        "apt_status",
        "rg_available",
        "expected_status",
        "expect_env",
        "expect_artifact",
        "expect_annotation",
    ),
    [
        # The overwhelmingly common case, and the one that arms the rg-gated
        # tests for every release and merge-queue run: apt succeeds, the tests
        # are promised ripgrep, and nothing is reported.
        ("", 0, False, 0, True, False, None),
        ("true", 0, False, 0, True, False, None),
        # Unlabeled: any apt failure fails the job, ripgrep never promised.
        ("", 1, False, 1, False, False, "::error::"),
        ("", 100, False, 100, False, False, "::error::"),
        # An empty `BYPASS` is what `push`/`merge_group` see, because the label
        # step is skipped there and a skipped step's output is the empty string.
        # Anything other than an exact "true" must enforce.
        ("false", 1, False, 1, False, False, "::error::"),
        ("True", 1, False, 1, False, False, "::error::"),
        # Labeled: a failed install is tolerated; without a usable `rg` the leg
        # reports the timeout artifact and does not promise ripgrep to the tests.
        ("true", 1, False, 0, False, True, "::warning::"),
        # Labeled but `rg` is actually present anyway: promise ripgrep, keep the
        # leg's coverage, and report nothing to the comment workflow.
        ("true", 1, True, 0, True, False, "::notice::"),
    ],
    ids=[
        "unlabeled-succeeds",
        "labeled-succeeds",
        "unlabeled-fails",
        "unlabeled-fails-100",
        "bypass-literal-false",
        "bypass-wrong-case",
        "labeled-bypasses",
        "labeled-rg-present",
    ],
)
def test_strict_ripgrep_install_bypass(
    tmp_path: Path,
    bypass: str,
    apt_status: int,
    rg_available: bool,
    expected_status: int,
    expect_env: bool,
    expect_artifact: bool,
    expect_annotation: str | None,
) -> None:
    """The strict step fails on apt errors unless the PR carries the bypass label.

    Every row asserts the full outcome — exit status, whether ripgrep is
    promised to the tests, whether the leg is reported to the comment workflow,
    and which annotation is emitted — so no case can pass by falling through.
    """
    workflow = _load_workflow(TEST_WORKFLOW)
    step = _find_step(workflow, job="build", name=STRICT_STEP)
    result, output_lines, env_lines = _run_install_script(
        step["run"],
        tmp_path,
        timeout_status=0,
        apt_status=apt_status,
        rg_available=rg_available,
        bypass=bypass,
    )

    assert result.returncode == expected_status
    assert (EXPECTED_ENV in env_lines) is expect_env

    timed_out = [line for line in output_lines if line.startswith("timed-out=")]
    artifact = f"artifact={ARTIFACT_PREFIX}libs-code-ubuntu-latest-3.14"
    if expect_artifact:
        # Only a bypassed failure with no usable rg reports a leg; reporting a
        # leg that kept its coverage would put a false warning on the PR.
        assert timed_out == ["timed-out=true"]
        assert artifact in output_lines
    else:
        assert timed_out == []
        assert artifact not in output_lines

    for annotation in ("::error::", "::warning::", "::notice::"):
        assert (annotation in result.stdout) is (annotation == expect_annotation), (
            f"unexpected {annotation} handling in: {result.stdout}"
        )


def test_ripgrep_bypass_step_runs_only_where_the_strict_step_does() -> None:
    """The label step must not annotate legs that have no strict install.

    Its `if:` is the intersection of `pull_request` and the strict step's own
    condition. Widen it and every ordinary PR collects a per-leg `::error::`
    about a check that is not enforced there; narrow it and a release PR
    silently loses the bypass.
    """
    workflow = _load_workflow(TEST_WORKFLOW)
    resolve = _find_step(workflow, job="build", name=RESOLVE_STEP)
    strict = _find_step(workflow, job="build", name=STRICT_STEP)

    condition = " ".join(resolve["if"].split())
    assert condition == (
        "runner.os == 'Linux' && github.event_name == 'pull_request' && "
        "(startsWith(github.head_ref, 'release-please--') || "
        "startsWith(github.event.pull_request.title, 'release('))"
    )
    # Whatever the release predicate is, both steps must spell it the same way.
    for predicate in RELEASE_PR_PREDICATES:
        assert predicate in " ".join(resolve["if"].split())
        assert predicate in " ".join(strict["if"].split())


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell script")
@pytest.mark.parametrize(
    ("labels", "gh_exit", "expected_bypass", "expect_error"),
    [
        ("bypass-ripgrep-check", 0, "true", False),
        ("dependencies\nlgtm", 0, "false", False),
        ("", 0, "false", False),
        # Neither a superstring nor a prefixed variant may arm the bypass.
        ("bypass-ripgrep-check-v2", 0, "false", False),
        ("no-bypass-ripgrep-check", 0, "false", False),
        # Fail closed and say so. Stdout carrying the label is deliberately
        # ignored when the call itself failed: a partial read must never arm a
        # bypass, and a silent enforce is indistinguishable from "label absent".
        ("bypass-ripgrep-check", 1, "false", True),
    ],
    ids=[
        "label-present",
        "label-absent",
        "no-labels",
        "label-superstring",
        "label-prefixed",
        "api-failure-fails-closed",
    ],
)
def test_ripgrep_bypass_step_behaviour(
    tmp_path: Path,
    labels: str,
    gh_exit: int,
    expected_bypass: str,
    expect_error: bool,
) -> None:
    """Run the label step's real shell against a stubbed `gh`.

    This step is the single switch that can disarm the strict install on a
    release PR, so its polarity is executed rather than grepped.
    """
    workflow = _load_workflow(TEST_WORKFLOW)
    step = _find_step(workflow, job="build", name=RESOLVE_STEP)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gh_stub = bin_dir / "gh"
    gh_stub.write_text(
        f'#!/usr/bin/env bash\nprintf "%s\\n" "{labels}"\nexit {gh_exit}\n'
    )
    gh_stub.chmod(0o755)

    github_output = tmp_path / "github-output"
    github_output.touch()

    result = subprocess.run(
        ["bash", "-c", step["run"]],
        check=False,
        capture_output=True,
        text=True,
        env={
            "GH_TOKEN": "stub-token",
            "GITHUB_OUTPUT": str(github_output),
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "PR": "1234",
            "REPO": "langchain-ai/deepagents",
        },
    )

    # A non-zero exit would fail the job outright, which is a different (and
    # much louder) outcome than resolving the bypass to false.
    assert result.returncode == 0, result.stderr
    assert github_output.read_text().splitlines() == [f"bypass={expected_bypass}"]
    assert ("::error::" in result.stdout) is expect_error


def test_release_and_non_pr_ripgrep_install_is_strict() -> None:
    """Release runs install ripgrep with no soft timeout, and only bypass on a label.

    The install itself is unbounded — the only `timeout` in the step is the
    dpkg unwind that runs *after* a bypassed failure, never around the install.
    The bypass is read from the live PR label, and the tests are promised
    ripgrep only when a usable `rg` is present, whether or not apt said so.
    """
    workflow = _load_workflow(TEST_WORKFLOW)
    step = _find_step(workflow, job="build", name=STRICT_STEP)

    assert APT_INSTALL in step["run"]
    assert "continue-on-error" not in step
    assert "timeout-minutes" not in step
    _assert_install_is_unbounded(step["run"])
    # Bypass comes from the resolved PR label, not the event payload.
    assert step["env"]["BYPASS"] == "${{ steps.ripgrep-bypass.outputs.bypass }}"
    # Promises ripgrep to the tests, turning a missing `rg` into a failure
    # instead of a silent skip. Asserted on the *success* branch specifically:
    # the step sets it twice, so a bare `in step["run"]` is satisfied by the
    # bypass branch alone and would not notice the success branch losing it.
    success_branch, separator, _ = step["run"].partition("# apt failed.")
    assert separator, "expected the success branch to precede the apt-failure comment"
    assert EXPECTED_ENV in success_branch


def test_release_workflow_requires_ripgrep_before_unit_tests() -> None:
    """Release artifact tests install ripgrep without a soft timeout."""
    workflow = _load_workflow(RELEASE_WORKFLOW)
    steps = workflow["jobs"]["pre-release-checks"]["steps"]
    install = _find_step(workflow, job="pre-release-checks", name="Install ripgrep")
    tests = _find_step(workflow, job="pre-release-checks", name="Run unit tests")

    assert steps.index(install) < steps.index(tests)
    assert APT_INSTALL in install["run"]
    _assert_install_is_unbounded(install["run"])
    assert "continue-on-error" not in install
    assert "timeout-minutes" not in install
    # The dangerous bypass must come from the dispatch input, never a default.
    assert (
        install["env"]["SKIP_RIPGREP_CHECK"]
        == "${{ inputs.dangerous-skip-ripgrep-check }}"
    )
    success_branch, separator, _ = install["run"].partition("SKIP_RIPGREP_CHECK")
    assert separator
    assert EXPECTED_ENV in success_branch


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell script")
@pytest.mark.parametrize(
    (
        "skip",
        "apt_status",
        "rg_available",
        "expected_status",
        "expect_env",
        "expect_summary",
        "expect_annotation",
    ),
    [
        # apt succeeds: the artifact tests are held to the real binary.
        ("false", 0, False, 0, True, False, None),
        ("true", 0, False, 0, True, False, None),
        # No dispatch flag: an apt failure fails the publish run, exit code kept.
        ("false", 1, False, 1, False, False, "::error::"),
        ("false", 100, True, 100, False, False, "::error::"),
        # The input arrives as a string; only an exact "true" may soften.
        ("", 1, False, 1, False, False, "::error::"),
        ("True", 1, False, 1, False, False, "::error::"),
        # Flagged and `rg` really is missing: continue, and record the dropped
        # coverage in the run summary rather than in a step log alone.
        ("true", 1, False, 0, False, True, "::warning::"),
        # Flagged but a usable `rg` is present: keep full coverage instead of
        # discarding it, and publish nothing about missing coverage.
        ("true", 1, True, 0, True, False, "::notice::"),
    ],
    ids=[
        "unflagged-succeeds",
        "flagged-succeeds",
        "unflagged-fails",
        "unflagged-fails-100",
        "flag-empty",
        "flag-wrong-case",
        "flagged-bypasses",
        "flagged-rg-present",
    ],
)
def test_release_workflow_ripgrep_bypass(
    tmp_path: Path,
    skip: str,
    apt_status: int,
    rg_available: bool,
    expected_status: int,
    expect_env: bool,
    expect_summary: bool,
    expect_annotation: str | None,
) -> None:
    """`release.yml`'s install tolerates apt only under the dispatch flag.

    This step publishes to PyPI, so it is executed rather than grepped: a
    static check cannot tell `if [ "$SKIP_RIPGREP_CHECK" = "true" ]` from
    `if true`.
    """
    workflow = _load_workflow(RELEASE_WORKFLOW)
    install = _find_step(workflow, job="pre-release-checks", name="Install ripgrep")
    result, _, env_lines = _run_install_script(
        install["run"],
        tmp_path,
        timeout_status=0,
        apt_status=apt_status,
        rg_available=rg_available,
        skip_ripgrep_check=skip,
    )

    assert result.returncode == expected_status
    assert (EXPECTED_ENV in env_lines) is expect_env

    summary = _summary_path(tmp_path).read_text()
    assert ("Published without ripgrep coverage" in summary) is expect_summary

    for annotation in ("::error::", "::warning::", "::notice::"):
        assert (annotation in result.stdout) is (annotation == expect_annotation), (
            f"unexpected {annotation} handling in: {result.stdout}"
        )


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
    assert "listLabelsOnIssue" in script
    assert "bypass-ripgrep-check" in script
    assert "createComment" in script
    assert "updateComment" in script
    assert "deleteComment" in script
    # Fork PRs carry no `workflow_run.pull_requests`; resolving by head ref is
    # the only lookup that works for them.
    assert "pulls.list" in script
    assert "run.head_repository.owner.login" in script
    # An unresolvable PR is a normal race (closed PR / deleted head branch
    # between CI completing and this workflow firing), so the job warns in the
    # run log instead of failing: the CI run's own `::warning::` annotation is
    # the record of the timeout, and there is no conversation left to post to.
    assert "core.setFailed" not in script
    assert "core.warning(`${message} A ripgrep timeout goes unreported" in script


def test_ripgrep_comment_wording_is_keyed_on_pr_kind_not_the_label() -> None:
    """Which failure the comment describes must not depend on a label read.

    On a release PR the only producer of these artifacts is the strict step's
    bypass path, which has no timeout — so the soft-timeout wording ("took more
    than two minutes") is false there. Keying the branch on the label instead
    reintroduces that: a failed `listLabelsOnIssue`, or a label removed between
    CI finishing and this workflow running, would post the wrong explanation
    plus "No action is needed to merge."
    """
    workflow = _load_workflow(RIPGREP_COMMENT_WORKFLOW)
    script = workflow["jobs"]["manage-comment"]["steps"][0]["with"]["script"]

    assert "const body = isReleasePullRequest\n" in script
    assert "isReleasePullRequest && hasBypassLabel" not in script
    # The label may still colour one sentence, and must fail closed to
    # "unconfirmed" rather than asserting either outcome.
    assert "let hasBypassLabel = false;" in script
    assert "hasBypassLabel = true" not in script


def test_release_pr_predicate_matches_between_ci_and_comment_workflow() -> None:
    """`_test.yml` produces the artifacts; the comment workflow explains them.

    Each spells the release-PR test in its own language against its own data
    source. If they drift, a PR gets artifacts from one failure mode described
    as the other, and nothing else in CI notices.
    """
    test_workflow = _load_workflow(TEST_WORKFLOW)
    strict = _find_step(test_workflow, job="build", name=STRICT_STEP)
    soft = _find_step(test_workflow, job="build", name=SOFT_STEP)
    script = _load_workflow(RIPGREP_COMMENT_WORKFLOW)["jobs"]["manage-comment"][
        "steps"
    ][0]["with"]["script"]

    for predicate in RELEASE_PR_PREDICATES:
        assert predicate in strict["if"]
        assert predicate in soft["if"]
        assert f"'{predicate}'" in script


def test_ripgrep_timeout_comment_concurrency_is_keyed_on_head_sha() -> None:
    """Runs for different commits must not cancel each other.

    Keyed on the branch, a newer run could cancel an older one that was about
    to post, and then discard itself via the stale-SHA guard.
    """
    workflow = _load_workflow(RIPGREP_COMMENT_WORKFLOW)
    group = workflow["concurrency"]["group"]

    assert "github.event.workflow_run.head_sha" in group
    assert "head_branch" not in group
