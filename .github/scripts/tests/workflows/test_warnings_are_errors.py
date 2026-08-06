"""Static contract for the warnings-as-errors test policy.

Every package under `libs/` opts in by putting `"error"` first in its pytest
`filterwarnings`; `_test.yml` keeps the `bypass-warnings-check` label wiring
that demotes the policy for a labeled PR.
"""

from __future__ import annotations

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


@pytest.fixture(scope="module")
def test_workflow() -> dict[str, Any]:
    """Parse `_test.yml` once for all tests in this module."""
    return _load_yaml(TEST_WORKFLOW)


def test_discovery_finds_all_packages() -> None:
    """Guard against vacuous passes if the discovery glob breaks."""
    assert len(_discover_package_pyprojects()) >= 10


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
    assert filters[0] == "error"


def test_label_bypass_step_is_wired(test_workflow: dict[str, Any]) -> None:
    """The `warnings` step exists and checks for the bypass label."""
    steps = [step for step in _build_steps(test_workflow) if step.get("id") == "warnings"]
    assert len(steps) == 1
    assert "bypass-warnings-check" in steps[0]["run"]


def test_test_steps_consume_bypass_flag(test_workflow: dict[str, Any]) -> None:
    """Both test steps export `WARNINGS_FLAG` from the step output and pass it to pytest."""
    consumers = [
        step
        for step in _build_steps(test_workflow)
        if "WARNINGS_FLAG" in (step.get("env") or {})
    ]
    assert len(consumers) == 2
    for step in consumers:
        assert step["env"]["WARNINGS_FLAG"] == "${{ steps.warnings.outputs.flag }}"
        assert "$WARNINGS_FLAG" in step["run"]


@pytest.mark.parametrize("workflow_path", [CI_WORKFLOW, TEST_WORKFLOW])
def test_workflow_grants_label_read_permission(workflow_path: Path) -> None:
    """Live label reads require `pull-requests: read` on the token."""
    workflow = _load_yaml(workflow_path)
    assert workflow["permissions"]["pull-requests"] == "read"
