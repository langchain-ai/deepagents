"""Tests for version-related functionality."""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

from deepagents_cli._version import __version__


def test_version_matches_pyproject() -> None:
    """`__version__` in `_version.py` must match the version in `pyproject.toml`."""
    project_root = Path(__file__).parent.parent.parent
    pyproject_path = project_root / "pyproject.toml"

    with pyproject_path.open("rb") as f:
        pyproject_data = tomllib.load(f)
    pyproject_version = pyproject_data["project"]["version"]

    assert __version__ == pyproject_version, (
        f"Version mismatch: _version.py has '{__version__}' "
        f"but pyproject.toml has '{pyproject_version}'"
    )


def test_cli_version_flag() -> None:
    """`--version` prints the package version and exits 0."""
    result = subprocess.run(
        [sys.executable, "-m", "deepagents_cli", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert f"deepagents-cli {__version__}" in result.stdout


def test_cli_bare_invocation_redirects_to_deepagents_code() -> None:
    """Bare `deepagents` invocation prints the deprecation notice and exits non-zero."""
    result = subprocess.run(
        [sys.executable, "-m", "deepagents_cli"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "deepagents-code" in result.stderr
