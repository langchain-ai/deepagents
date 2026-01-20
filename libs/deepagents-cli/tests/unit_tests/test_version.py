"""Tests for version-related functionality."""

import subprocess
import sys
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


def test_cli_version_flag() -> None:
    """Verify that --version flag outputs the correct version."""
    result = subprocess.run(
        [sys.executable, "-m", "deepagents_cli.main", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    # argparse exits with 0 for --version
    assert result.returncode == 0
    assert f"deepagents {__version__}" in result.stdout


def test_version_slash_command_message_format() -> None:
    """Verify the /version slash command message format matches expected output."""
    # This tests the exact message format used in app.py's _handle_command for /version
    expected_message = f"deepagents version: {__version__}"
    assert "deepagents version:" in expected_message
    assert __version__ in expected_message


def test_help_mentions_version_flag() -> None:
    """Verify that the CLI help text mentions --version."""
    result = subprocess.run(
        [sys.executable, "-m", "deepagents_cli.main", "help"],
        capture_output=True,
        text=True,
        check=False,
    )
    # Help command should succeed
    assert result.returncode == 0
    # Help output should mention --version
    assert "--version" in result.stdout
