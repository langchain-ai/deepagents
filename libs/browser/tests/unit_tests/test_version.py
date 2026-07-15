"""Version metadata tests."""

import tomllib
from pathlib import Path

from deepagents_browser import __version__


def test_version_matches_pyproject() -> None:
    """The public version matches package metadata."""
    project_root = Path(__file__).parents[2]
    with (project_root / "pyproject.toml").open("rb") as file:
        project_version = tomllib.load(file)["project"]["version"]

    assert __version__ == project_version
