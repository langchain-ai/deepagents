"""Test version consistency between _version.py and pyproject.toml."""

import tomllib
from pathlib import Path

from deepagents_cli._version import __version__


def test_version_matches_pyproject() -> None:
    """Verify that __version__ in _version.py matches version in pyproject.toml."""
    # Get the project root directory
    project_root = Path(__file__).parent.parent.parent
    pyproject_path = project_root / "pyproject.toml"

    # Read the version from pyproject.toml
    with pyproject_path.open("rb") as f:
        pyproject_data = tomllib.load(f)
    pyproject_version = pyproject_data["project"]["version"]

    # Compare versions
    assert __version__ == pyproject_version, (
        f"Version mismatch: _version.py has '{__version__}' "
        f"but pyproject.toml has '{pyproject_version}'"
    )
