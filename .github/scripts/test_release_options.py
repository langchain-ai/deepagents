"""Verify release.yml dropdown options match actual package names on disk."""

import tomllib
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def _get_release_options() -> list[str]:
    workflow = REPO_ROOT / ".github" / "workflows" / "release.yml"
    with open(workflow) as f:
        data = yaml.safe_load(f)
    try:
        # PyYAML (YAML 1.1) parses the bare key `on` as boolean True
        return data[True]["workflow_dispatch"]["inputs"]["package"]["options"]
    except (KeyError, TypeError) as e:
        msg = f"Could not find workflow_dispatch package options in {workflow}: {e}"
        raise AssertionError(msg) from e


def _get_package_names() -> set[str]:
    """Read project.name from every pyproject.toml under libs/."""
    libs = REPO_ROOT / "libs"
    names: set[str] = set()
    for pyproject in libs.rglob("pyproject.toml"):
        # Only consider direct children of libs/ and libs/partners/
        rel = pyproject.parent.relative_to(libs)
        parts = rel.parts
        if len(parts) == 1 or (len(parts) == 2 and parts[0] == "partners"):
            with open(pyproject, "rb") as f:
                data = tomllib.load(f)
            name = data.get("project", {}).get("name")
            if name:
                names.add(name)
    return names


def test_release_options_match_packages() -> None:
    options = set(_get_release_options())
    packages = _get_package_names()
    missing_from_dropdown = packages - options
    extra_in_dropdown = options - packages
    assert not missing_from_dropdown, (
        f"Packages on disk missing from release.yml dropdown: {missing_from_dropdown}"
    )
    assert not extra_in_dropdown, (
        f"Dropdown options with no matching package: {extra_in_dropdown}"
    )
