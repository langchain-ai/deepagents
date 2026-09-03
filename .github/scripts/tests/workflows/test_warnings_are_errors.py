"""Contract for the warnings-as-errors test policy.

Every package under `libs/` opts in by putting `"error"` first in its pytest
`filterwarnings`; `_test.yml` keeps the `bypass-warnings-check` label wiring
that demotes the policy for a labeled PR.

The bypass step is the single switch that can disable the policy repo-wide, so
it is not merely grepped for -- `test_bypass_step_behaviour` extracts its shell
and runs it against a stubbed `gh` to pin the polarity and the fail-closed
contract.
"""

from __future__ import annotations

import shutil
import subprocess
import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[4]
WORKFLOWS = ROOT / ".github" / "workflows"
TEST_WORKFLOW = WORKFLOWS / "_test.yml"
CI_WORKFLOW = WORKFLOWS / "ci.yml"

_EXCLUDED_PARTS = frozenset({".venv", "node_modules", "build", "dist", ".tox"})

# Filter actions that re-demote warnings. An entry using one of these bare (no
# message and no category) undoes the leading `"error"`, because later
# `filterwarnings` entries take precedence over earlier ones.
_DEMOTING_ACTIONS = frozenset({"default", "always", "module", "once", "ignore"})

# Categories broad enough that ignoring them re-demotes essentially everything.
_TOO_BROAD_CATEGORIES = frozenset({"Warning", "DeprecationWarning", "UserWarning"})


def _discover_package_pyprojects() -> list[Path]:
    """Return every package `pyproject.toml` under `libs/`, skipping build trees."""
    return sorted(
        path
        for path in (ROOT / "libs").rglob("pyproject.toml")
        if not _EXCLUDED_PARTS & set(path.parts)
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    """Parse a workflow YAML file."""
    return yaml.safe_load(path.read_text())


def _build_steps(workflow: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the steps of the `build` job in `_test.yml`."""
    return workflow["jobs"]["build"]["steps"]


def _bypass_step(workflow: dict[str, Any]) -> dict[str, Any]:
    """Return the single `warnings` step that resolves the bypass label."""
    steps = [step for step in _build_steps(workflow) if step.get("id") == "warnings"]
    assert len(steps) == 1, f"expected exactly one `warnings` step, found {len(steps)}"
    return steps[0]


def _tested_working_directories() -> set[str]:
    """Return the `working-directory` of every `ci.yml` job that calls `_test.yml`."""
    jobs = _load_yaml(CI_WORKFLOW)["jobs"].values()
    return {
        job["with"]["working-directory"]
        for job in jobs
        if isinstance(job, dict) and str(job.get("uses", "")).endswith("_test.yml")
    }


@pytest.fixture(scope="module")
def test_workflow() -> dict[str, Any]:
    """Parse `_test.yml` once for all tests in this module."""
    return _load_yaml(TEST_WORKFLOW)


def test_every_discovered_package_is_wired_into_ci() -> None:
    """Discovery matches the packages `ci.yml` actually runs tests for.

    Pins both directions: a broken glob cannot silently shrink the set this
    module checks, and a package added under `libs/` without a `ci.yml` test job
    cannot slip in unnoticed.
    """
    discovered = {
        str(path.parent.relative_to(ROOT)) for path in _discover_package_pyprojects()
    }
    assert discovered == _tested_working_directories()


@pytest.mark.parametrize(
    "pyproject",
    _discover_package_pyprojects(),
    ids=lambda path: str(path.relative_to(ROOT)),
)
def test_package_promotes_warnings_to_errors(pyproject: Path) -> None:
    """Each package's first `filterwarnings` entry is exactly `"error"`."""
    with pyproject.open("rb") as f:
        data = tomllib.load(f)
    filters = data["tool"]["pytest"]["ini_options"]["filterwarnings"]
    assert isinstance(filters, list), "`filterwarnings` must be a list, not a string"
    assert filters[0] == "error"


@pytest.mark.parametrize(
    "pyproject",
    _discover_package_pyprojects(),
    ids=lambda path: str(path.relative_to(ROOT)),
)
def test_allowlist_does_not_re_demote_everything(pyproject: Path) -> None:
    """No allowlist entry undoes the leading `"error"`.

    `filters[0] == "error"` alone is satisfied by a list ending in `"ignore"` or
    `"ignore::Warning"`, which would silently switch the whole policy back off.
    """
    with pyproject.open("rb") as f:
        data = tomllib.load(f)
    for entry in data["tool"]["pytest"]["ini_options"]["filterwarnings"][1:]:
        action, _, rest = entry.partition(":")
        assert not (action in _DEMOTING_ACTIONS and not rest.strip(":")), (
            f"{entry!r} is a bare {action!r} and re-demotes every warning"
        )
        # Spec is `action:message:category:module:lineno`.
        fields = entry.split(":")
        message, category = (fields + ["", ""])[1:3]
        assert not (
            action in _DEMOTING_ACTIONS
            and not message
            and category in _TOO_BROAD_CATEGORIES
        ), f"{entry!r} matches every {category} and re-demotes the policy"


def test_test_steps_consume_bypass_flag(test_workflow: dict[str, Any]) -> None:
    """Both test steps export `WARNINGS_FLAG` from the step output and pass it on."""
    consumers = [
        step
        for step in _build_steps(test_workflow)
        if "WARNINGS_FLAG" in (step.get("env") or {})
    ]
    assert len(consumers) == 2
    for step in consumers:
        assert step["env"]["WARNINGS_FLAG"] == "${{ steps.warnings.outputs.flag }}"
        assert "$WARNINGS_FLAG" in step["run"]


def test_bypass_step_runs_only_for_pull_requests(test_workflow: dict[str, Any]) -> None:
    """`push`/`merge_group` runs must not be able to resolve a bypass flag."""
    assert _bypass_step(test_workflow)["if"] == "github.event_name == 'pull_request'"


@pytest.mark.parametrize("workflow_path", [CI_WORKFLOW, TEST_WORKFLOW])
def test_workflow_grants_label_read_permission(workflow_path: Path) -> None:
    """Live label reads require `pull-requests: read` on the token."""
    workflow = _load_yaml(workflow_path)
    assert workflow["permissions"]["pull-requests"] == "read"


@pytest.mark.parametrize(
    ("labels", "gh_exit", "expected_flag", "expect_error_annotation"),
    [
        pytest.param("bypass-warnings-check", 0, "-W default", False, id="label-present"),
        pytest.param("dependencies\nlgtm", 0, "", False, id="label-absent"),
        pytest.param("", 0, "", False, id="no-labels"),
        # A substring or superstring of the label must not trigger the bypass.
        pytest.param("bypass-warnings-check-v2", 0, "", False, id="label-superstring"),
        pytest.param("no-bypass-warnings-check", 0, "", False, id="label-prefixed"),
        # Fail closed, and say so: an unreadable label list must enforce the
        # policy *and* annotate, or a maintainer cannot tell why the label had
        # no effect. Labels on stdout are ignored when the call itself failed.
        pytest.param("bypass-warnings-check", 1, "", True, id="api-failure-fails-closed"),
    ],
)
@pytest.mark.skipif(shutil.which("bash") is None, reason="requires bash")
def test_bypass_step_behaviour(
    test_workflow: dict[str, Any],
    tmp_path: Path,
    labels: str,
    gh_exit: int,
    expected_flag: str,
    expect_error_annotation: bool,
) -> None:
    """Run the step's real shell against a stubbed `gh` and pin its decisions."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gh_stub = bin_dir / "gh"
    gh_stub.write_text(f'#!/usr/bin/env bash\nprintf "%s\\n" "{labels}"\nexit {gh_exit}\n')
    gh_stub.chmod(0o755)

    script = tmp_path / "step.sh"
    script.write_text(_bypass_step(test_workflow)["run"])
    github_output = tmp_path / "github_output"
    github_output.touch()

    result = subprocess.run(  # noqa: S603
        ["bash", str(script)],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
        env={
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "GITHUB_OUTPUT": str(github_output),
            "GH_TOKEN": "stub-token",
            "REPO": "langchain-ai/deepagents",
            "PR": "1234",
        },
    )

    assert result.returncode == 0, result.stderr
    assert github_output.read_text().strip() == f"flag={expected_flag}"
    assert ("::error::" in result.stdout) is expect_error_annotation
