"""Contracts for the release workflow's Python matrix."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest
import yaml
from resolve_python_matrix import (
    SUPPORTED_PYTHON_VERSIONS,
    resolve_from_pyproject,
    resolve_python_versions,
)


ROOT = Path(__file__).resolve().parents[4]
CI_WORKFLOW = ROOT / ".github/workflows/ci.yml"
RELEASE_WORKFLOW = ROOT / ".github/workflows/release.yml"
TEST_WORKFLOW_REF = "./.github/workflows/_test.yml"


def _ci_test_jobs() -> dict[str, dict]:
    """Return every CI job using the reusable test workflow."""
    workflow = yaml.safe_load(CI_WORKFLOW.read_text())
    return {
        name: job
        for name, job in workflow["jobs"].items()
        if job.get("uses") == TEST_WORKFLOW_REF
    }


def test_ci_matrices_match_requires_python() -> None:
    """Keep PR and release test matrices aligned."""
    for name, job in _ci_test_jobs().items():
        working_directory = job["with"]["working-directory"]
        pyproject = ROOT / working_directory / "pyproject.toml"
        expected = resolve_from_pyproject(pyproject.read_text())
        declared = json.loads(job["with"]["python-versions"])
        assert declared == expected, (
            f"{name} tests {declared}, but {working_directory} supports {expected}"
        )


def test_release_uses_resolved_matrix() -> None:
    """Fan out release checks over the versions resolved by `setup`."""
    workflow = yaml.safe_load(RELEASE_WORKFLOW.read_text())
    jobs = workflow["jobs"]

    assert jobs["setup"]["outputs"]["python-versions"] == (
        "${{ steps.python-matrix.outputs.python-versions }}"
    )
    strategy = jobs["pre-release-checks"]["strategy"]
    assert strategy["fail-fast"] is False
    assert strategy["matrix"]["python-version"] == (
        "${{ fromJSON(needs.setup.outputs.python-versions) }}"
    )


@pytest.mark.parametrize(
    ("requires_python", "expected"),
    [
        (">=3.11", ["3.11", "3.12", "3.13", "3.14"]),
        (">=3.11,<4.0", ["3.11", "3.12", "3.13", "3.14"]),
        (">=3.12,<3.14", ["3.12", "3.13"]),
        (">=3.11,!=3.12", ["3.11", "3.13", "3.14"]),
        ("==3.13", ["3.13"]),
        (">3.11,<=3.13", ["3.12", "3.13"]),
    ],
)
def test_resolve_python_versions(requires_python: str, expected: list[str]) -> None:
    assert resolve_python_versions(requires_python) == expected


@pytest.mark.parametrize(
    "requires_python",
    ["", "~=3.11", "==3.12.*", ">=3.15", ">=3.11,<3.11"],
)
def test_resolve_python_versions_rejects_unusable_specifiers(
    requires_python: str,
) -> None:
    """Unknown syntax and empty matrices must fail closed."""
    with pytest.raises(ValueError):
        resolve_python_versions(requires_python)


def test_supported_versions_are_ordered_and_unique() -> None:
    versions = list(SUPPORTED_PYTHON_VERSIONS)
    expected = sorted(
        set(versions), key=lambda value: tuple(map(int, value.split(".")))
    )
    assert versions == expected


def test_every_release_package_resolves_a_matrix() -> None:
    """Each release option must yield at least one test interpreter."""
    workflow = yaml.safe_load(RELEASE_WORKFLOW.read_text())
    options = workflow[True]["workflow_dispatch"]["inputs"]["package"]["options"]

    pyprojects: dict[str, Path] = {}
    for pyproject in (ROOT / "libs").rglob("pyproject.toml"):
        relative = pyproject.parent.relative_to(ROOT / "libs").parts
        if len(relative) == 1 or (len(relative) == 2 and relative[0] == "partners"):
            with open(pyproject, "rb") as file:
                name = tomllib.load(file).get("project", {}).get("name")
            if name:
                pyprojects[name] = pyproject

    for option in options:
        assert resolve_from_pyproject(pyprojects[option].read_text())
