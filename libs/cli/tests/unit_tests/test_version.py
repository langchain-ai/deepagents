"""Tests for version-related functionality."""

import subprocess
import sys
import tomllib
from importlib.metadata import version as pkg_version
from pathlib import Path
from unittest.mock import patch

import pytest

from deepagents_cli._version import __version__


def test_version_matches_pyproject() -> None:
    """Verify `__version__` in `_version.py` matches version in `pyproject.toml`."""
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
    """Verify that `--version` flag outputs the correct version."""
    result = subprocess.run(
        [sys.executable, "-m", "deepagents_cli.main", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    # argparse exits with 0 for --version
    assert result.returncode == 0
    assert f"deepagents-cli {__version__}" in result.stdout
    from importlib.metadata import version as pkg_version

    sdk_version = pkg_version("deepagents")
    assert f"deepagents (SDK) {sdk_version}" in result.stdout


@pytest.mark.asyncio
async def test_version_slash_command_message_format() -> None:
    """Verify the `/version` slash command outputs both CLI and SDK versions."""
    from deepagents_cli.app import DeepAgentsApp
    from deepagents_cli.widgets.messages import AppMessage

    sdk_version = pkg_version("deepagents")

    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        await app._handle_command("/version")
        await pilot.pause()

        app_msgs = app.query(AppMessage)
        content = str(app_msgs[-1]._content)
        assert f"deepagents-cli version: {__version__}" in content
        assert f"deepagents (SDK) version: {sdk_version}" in content


@pytest.mark.asyncio
async def test_version_slash_command_sdk_unavailable() -> None:
    """Verify `/version` shows 'unknown' when SDK package metadata is missing."""
    from importlib.metadata import PackageNotFoundError

    from deepagents_cli.app import DeepAgentsApp
    from deepagents_cli.widgets.messages import AppMessage

    def patched_version(name: str) -> str:
        if name == "deepagents":
            raise PackageNotFoundError(name)
        return pkg_version(name)

    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        with patch("importlib.metadata.version", side_effect=patched_version):
            await app._handle_command("/version")
        await pilot.pause()

        app_msgs = app.query(AppMessage)
        content = str(app_msgs[-1]._content)
        assert f"deepagents-cli version: {__version__}" in content
        assert "deepagents (SDK) version: unknown" in content


@pytest.mark.asyncio
async def test_version_slash_command_cli_version_unavailable() -> None:
    """Verify `/version` shows 'unknown' when CLI _version module is missing."""
    from deepagents_cli.app import DeepAgentsApp
    from deepagents_cli.widgets.messages import AppMessage

    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        # Setting a module to None in sys.modules causes ImportError on import
        with patch.dict(sys.modules, {"deepagents_cli._version": None}):
            await app._handle_command("/version")
        await pilot.pause()

        app_msgs = app.query(AppMessage)
        content = str(app_msgs[-1]._content)
        assert "deepagents-cli version: unknown" in content


def test_help_mentions_version_flag() -> None:
    """Verify that the CLI help text mentions `--version` and SDK."""
    result = subprocess.run(
        [sys.executable, "-m", "deepagents_cli.main", "help"],
        capture_output=True,
        text=True,
        check=False,
    )
    # Help command should succeed
    assert result.returncode == 0
    # Help output should mention --version and SDK
    assert "--version" in result.stdout
    assert "SDK" in result.stdout


def test_cli_help_flag() -> None:
    """Verify that `--help` flag shows help and exits with code 0."""
    result = subprocess.run(
        [sys.executable, "-m", "deepagents_cli.main", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    # --help should exit with 0
    assert result.returncode == 0
    # Help output should mention key options
    assert "--version" in result.stdout
    assert "--agent" in result.stdout


def test_cli_help_flag_short() -> None:
    """Verify that `-h` flag shows help and exits with code 0."""
    result = subprocess.run(
        [sys.executable, "-m", "deepagents_cli.main", "-h"],
        capture_output=True,
        text=True,
        check=False,
    )
    # -h should exit with 0
    assert result.returncode == 0
    # Help output should mention key options
    assert "--version" in result.stdout
    assert "--agent" in result.stdout


def test_help_excludes_interactive_features() -> None:
    """Verify that `--help` does not contain Interactive Features section."""
    result = subprocess.run(
        [sys.executable, "-m", "deepagents_cli.main", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    # Help should succeed
    assert result.returncode == 0
    # Help should NOT contain Interactive Features section
    assert "Interactive Features" not in result.stdout
