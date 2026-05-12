"""Tests for version-related functionality."""

from __future__ import annotations

import subprocess
import sys
import tomllib
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from deepagents_cli._version import __version__

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def _block_sdk_pypi_fetch(tmp_path: Path) -> Iterator[None]:
    """Prevent `/version` tests from hitting real PyPI for CLI or SDK release age.

    The `DeepAgentsApp` background `_check_for_updates()` worker calls
    `is_update_available()` on startup, which makes a live PyPI request.
    Without blocking this, a newly published CLI version on PyPI mutates
    `app._update_available` mid-test, breaking assertions that assume the
    initial `(False, None)` state.

    Tests that exercise SDK release-age behavior directly override
    `CACHE_FILE` themselves; this fixture only ensures tests that don't care
    about that field never make a network request on cache miss.
    """
    cache_path = tmp_path / "latest_version.json"
    with (
        patch("deepagents_cli.update_check.CACHE_FILE", cache_path),
        patch("deepagents_cli.update_check.get_sdk_release_time", return_value=None),
        patch(
            "deepagents_cli.update_check.is_update_available",
            return_value=(False, None),
        ),
    ):
        yield


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
    """Verify that `--version` flag outputs the correct version and extras."""
    result = subprocess.run(
        [sys.executable, "-m", "deepagents_cli.main", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    # argparse exits with 0 for --version
    assert result.returncode == 0
    assert f"deepagents-cli {__version__}" in result.stdout
    sdk_version = pkg_version("deepagents")
    assert f"deepagents (SDK) {sdk_version}" in result.stdout
    # Extras block is plain-text (no markdown table or headings).
    assert "Installed optional dependencies:" in result.stdout
    assert "langchain-anthropic" in result.stdout
    assert "| Extra" not in result.stdout
    assert "###" not in result.stdout


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
        plain = [m for m in app_msgs if not m._is_markdown]
        content = str(plain[-1]._content)
        assert f"deepagents-cli version: {__version__}" in content
        assert f"deepagents (SDK) version: {sdk_version}" in content


async def test_version_slash_command_includes_optional_dependencies() -> None:
    """Verify `/version` mounts a markdown message with the extras table."""
    from deepagents_cli.app import DeepAgentsApp
    from deepagents_cli.widgets.messages import AppMessage

    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        await app._handle_command("/version")
        await pilot.pause()

        md_msgs = [m for m in app.query(AppMessage) if m._is_markdown]
        assert md_msgs
        source = str(md_msgs[-1]._content)
        assert "### Installed optional dependencies" in source
        assert "| Extra" in source
        assert "| Package" in source
        assert "| Version" in source
        assert "langchain-anthropic" in source


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

        app_msgs = [m for m in app.query(AppMessage) if not m._is_markdown]
        content = str(app_msgs[-1]._content)
        assert f"deepagents-cli version: {__version__}" in content
        assert "deepagents (SDK) version: unknown" in content


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

        app_msgs = [m for m in app.query(AppMessage) if not m._is_markdown]
        content = str(app_msgs[-1]._content)
        assert "deepagents-cli version: unknown" in content


async def test_version_slash_command_includes_release_age(tmp_path) -> None:
    """Verify `/version` appends the cached release age for the CLI version."""
    import json
    import time
    from datetime import UTC, datetime, timedelta

    from deepagents_cli.app import DeepAgentsApp
    from deepagents_cli.widgets.messages import AppMessage

    cache_path = tmp_path / "latest_version.json"
    iso = (datetime.now(tz=UTC) - timedelta(days=3)).isoformat()
    cache_path.write_text(
        json.dumps(
            {
                "release_times": {__version__: iso},
                "checked_at": time.time(),
            }
        )
    )

    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        with patch("deepagents_cli.update_check.CACHE_FILE", cache_path):
            await app._handle_command("/version")
        await pilot.pause()

        app_msgs = [m for m in app.query(AppMessage) if not m._is_markdown]
        content = str(app_msgs[-1]._content)
        assert f"deepagents-cli version: {__version__}, released " in content
        assert "ago" in content


async def test_version_slash_command_includes_sdk_release_age() -> None:
    """Verify `/version` appends the cached release age for the installed SDK."""
    from deepagents_cli.app import DeepAgentsApp
    from deepagents_cli.widgets.messages import AppMessage

    sdk_version = pkg_version("deepagents")

    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        # Override the autouse stub to simulate a populated cache.
        with (
            patch(
                "deepagents_cli.update_check.get_sdk_release_time",
                return_value="2026-04-10T12:00:00Z",
            ),
            patch(
                "deepagents_cli.sessions.format_relative_timestamp",
                return_value="1w ago",
            ),
        ):
            await app._handle_command("/version")
        await pilot.pause()

        app_msgs = [m for m in app.query(AppMessage) if not m._is_markdown]
        content = str(app_msgs[-1]._content)
        assert f"deepagents (SDK) version: {sdk_version}, released 1w ago" in content


async def test_version_slash_command_mentions_update_available() -> None:
    """Verify `/version` appends an update-available hint when one was detected."""
    from deepagents_cli.app import DeepAgentsApp
    from deepagents_cli.widgets.messages import AppMessage

    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app._update_available = (True, "99.99.99")
        await app._handle_command("/version")
        await pilot.pause()

        app_msgs = [m for m in app.query(AppMessage) if not m._is_markdown]
        content = str(app_msgs[-1]._content)
        assert "Update available: v99.99.99" in content
        assert "Run: " in content


async def test_version_slash_command_omits_update_hint_when_up_to_date() -> None:
    """Verify `/version` does not add the update hint when none is pending."""
    from deepagents_cli.app import DeepAgentsApp
    from deepagents_cli.widgets.messages import AppMessage

    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        # Default state — no update detected by the background check.
        assert app._update_available == (False, None)
        await app._handle_command("/version")
        await pilot.pause()

        app_msgs = [m for m in app.query(AppMessage) if not m._is_markdown]
        content = str(app_msgs[-1]._content)
        assert "Update available" not in content


async def test_update_slash_command_editable_install_short_circuits() -> None:
    """Editable install must not invoke `perform_upgrade` from the TUI.

    A regression here would run `pip install --upgrade deepagents-cli` on
    an editable dev checkout and overwrite the local install.
    """
    from unittest.mock import AsyncMock

    from deepagents_cli.app import DeepAgentsApp
    from deepagents_cli.widgets.messages import AppMessage

    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        with (
            patch(
                "deepagents_cli.config._is_editable_install",
                return_value=True,
            ),
            patch("deepagents_cli.update_check.is_update_available") as is_update_mock,
            patch(
                "deepagents_cli.update_check.perform_upgrade",
                new_callable=AsyncMock,
            ) as perform_upgrade_mock,
        ):
            await app._handle_command("/update")
            await pilot.pause()

        is_update_mock.assert_not_called()
        perform_upgrade_mock.assert_not_awaited()
        app_msgs = [m for m in app.query(AppMessage) if not m._is_markdown]
        content = str(app_msgs[-1]._content)
        assert "Updates are not available for editable installs" in content
        assert f"Currently on v{__version__}" in content


async def test_update_slash_command_pypi_unreachable_short_circuits() -> None:
    """`latest is None` from `is_update_available` must not run upgrade.

    Regression guard: collapsing this branch into the up-to-date message
    would tell users they're current when the check actually failed.
    """
    from unittest.mock import AsyncMock

    from deepagents_cli.app import DeepAgentsApp
    from deepagents_cli.widgets.messages import AppMessage

    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        with (
            patch(
                "deepagents_cli.config._is_editable_install",
                return_value=False,
            ),
            patch(
                "deepagents_cli.update_check.is_update_available",
                return_value=(False, None),
            ),
            patch(
                "deepagents_cli.update_check.perform_upgrade",
                new_callable=AsyncMock,
            ) as perform_upgrade_mock,
        ):
            await app._handle_command("/update")
            await pilot.pause()

        perform_upgrade_mock.assert_not_awaited()
        app_msgs = [m for m in app.query(AppMessage) if not m._is_markdown]
        content = str(app_msgs[-1]._content)
        assert "Could not determine the latest version" in content


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
