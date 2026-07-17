"""Tests for version-related functionality."""

from __future__ import annotations

import asyncio
import subprocess
import sys
import tomllib
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, Mock, patch

import pytest

from deepagents_code._version import __version__

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
        patch("deepagents_code.update_check.CACHE_FILE", cache_path),
        patch("deepagents_code.update_check.get_sdk_release_time", return_value=None),
        patch(
            "deepagents_code.update_check.is_update_available",
            return_value=(False, None),
        ),
        patch(
            "deepagents_code.update_check.release_requires_prereleases",
            return_value=False,
        ),
        # Pin the post-upgrade shadow check to a clean "no shadow" for the
        # whole module. Several `/update` tests pin `detect_install_method`
        # to `"uv"` to exercise pre-release handling, which would otherwise
        # make the real `detect_shadowed_dcode` run against the test
        # runner's filesystem and replace the expected success message
        # with the shadow warning. The dedicated shadow-present tests in
        # this file override this patch with their own.
        patch(
            "deepagents_code.update_check.detect_shadowed_dcode",
            return_value=None,
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
        [sys.executable, "-m", "deepagents_code.main", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    # argparse exits with 0 for --version
    assert result.returncode == 0
    assert f"deepagents-code {__version__}" in result.stdout
    sdk_version = pkg_version("deepagents")
    assert f"deepagents (SDK) {sdk_version}" in result.stdout
    # Extras block is plain-text (no markdown table or headings).
    assert "Installed optional dependencies:" in result.stdout
    assert "langchain-anthropic" in result.stdout
    assert "| Extra" not in result.stdout
    assert "###" not in result.stdout


async def test_version_slash_command_message_format() -> None:
    """Verify the `/version` slash command outputs both CLI and SDK versions."""
    from deepagents_code.app import DeepAgentsApp
    from deepagents_code.tui.widgets.messages import AppMessage

    sdk_version = pkg_version("deepagents")

    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        await app._handle_command("/version")
        await pilot.pause()

        app_msgs = app.query(AppMessage)
        plain = [m for m in app_msgs if not m._is_markdown]
        content = str(plain[-1]._content)
        assert f"deepagents-code version: {__version__}" in content
        assert f"deepagents (SDK) version: {sdk_version}" in content


async def test_version_slash_command_includes_optional_dependencies() -> None:
    """Verify `/version` mounts a markdown message with the extras table."""
    from deepagents_code.app import DeepAgentsApp
    from deepagents_code.tui.widgets.messages import AppMessage

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

    from deepagents_code.app import DeepAgentsApp
    from deepagents_code.tui.widgets.messages import AppMessage

    def patched_version(name: str) -> str:
        if name == "deepagents":
            raise PackageNotFoundError(name)
        return pkg_version(name)

    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        with patch(
            "deepagents_code.extras_info.pkg_version", side_effect=patched_version
        ):
            await app._handle_command("/version")
        await pilot.pause()

        app_msgs = [m for m in app.query(AppMessage) if not m._is_markdown]
        content = str(app_msgs[-1]._content)
        assert f"deepagents-code version: {__version__}" in content
        assert "deepagents (SDK) version: unknown" in content


async def test_version_slash_command_cli_version_unavailable() -> None:
    """Verify `/version` shows 'unknown' when CLI _version module is missing."""
    from deepagents_code.app import DeepAgentsApp
    from deepagents_code.tui.widgets.messages import AppMessage

    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        # Setting a module to None in sys.modules causes ImportError on import
        with patch.dict(sys.modules, {"deepagents_code._version": None}):
            await app._handle_command("/version")
        await pilot.pause()

        app_msgs = [m for m in app.query(AppMessage) if not m._is_markdown]
        content = str(app_msgs[-1]._content)
        assert "deepagents-code version: unknown" in content


async def test_version_slash_command_includes_release_age(tmp_path) -> None:
    """Verify `/version` appends the cached release age for the CLI version."""
    import json
    import time
    from datetime import UTC, datetime, timedelta

    from deepagents_code.app import DeepAgentsApp
    from deepagents_code.tui.widgets.messages import AppMessage

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
        with patch("deepagents_code.update_check.CACHE_FILE", cache_path):
            await app._handle_command("/version")
        await pilot.pause()

        app_msgs = [m for m in app.query(AppMessage) if not m._is_markdown]
        content = str(app_msgs[-1]._content)
        assert f"deepagents-code version: {__version__}, released " in content
        assert "ago" in content


async def test_version_slash_command_includes_sdk_release_age() -> None:
    """Verify `/version` appends the cached release age for the installed SDK."""
    from deepagents_code.app import DeepAgentsApp
    from deepagents_code.tui.widgets.messages import AppMessage

    sdk_version = pkg_version("deepagents")

    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        # Override the autouse stub to simulate a populated cache.
        with (
            patch(
                "deepagents_code.update_check.get_sdk_release_time",
                return_value="2026-04-10T12:00:00Z",
            ),
            patch(
                "deepagents_code.sessions.format_relative_timestamp",
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
    from deepagents_code.app import DeepAgentsApp
    from deepagents_code.tui.widgets.messages import AppMessage

    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app._update_available = (True, "99.99.99")
        with (
            patch(
                "deepagents_code.update_check.detect_install_method",
                return_value="uv",
            ),
            patch(
                "deepagents_code.update_check.is_auto_update_enabled",
                return_value=True,
            ),
        ):
            await app._handle_command("/version")
        await pilot.pause()

        app_msgs = [m for m in app.query(AppMessage) if not m._is_markdown]
        content = str(app_msgs[-1]._content)
        assert "Update available: v99.99.99" in content
        assert "relaunch dcode to install it automatically" in content
        # The auto-update hint must not also carry the manual fallback.
        assert "/update" not in content
        assert "uv tool install" not in content


async def test_version_slash_command_update_hint_when_auto_update_disabled() -> None:
    """Verify `/version` points to `/update` when auto-update is disabled."""
    from deepagents_code.app import DeepAgentsApp
    from deepagents_code.tui.widgets.messages import AppMessage

    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app._update_available = (True, "99.99.99")
        with (
            patch(
                "deepagents_code.update_check.detect_install_method",
                return_value="uv",
            ),
            patch(
                "deepagents_code.update_check.is_auto_update_enabled",
                return_value=False,
            ),
        ):
            await app._handle_command("/version")
        await pilot.pause()

        app_msgs = [m for m in app.query(AppMessage) if not m._is_markdown]
        content = str(app_msgs[-1]._content)
        assert "Update available: v99.99.99" in content
        assert "/update" in content
        assert "dcode update" in content
        # The manual hint must not falsely promise an automatic install.
        assert "automatically" not in content
        assert "uv tool install" not in content


async def test_version_slash_command_does_not_promise_unsupported_update() -> None:
    """Verify `/version` does not promise auto-update for unknown installers."""
    from deepagents_code.app import DeepAgentsApp
    from deepagents_code.tui.widgets.messages import AppMessage

    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app._update_available = (True, "99.99.99")
        with (
            patch(
                "deepagents_code.update_check.detect_install_method",
                return_value="other",
            ),
            patch(
                "deepagents_code.update_check.is_auto_update_enabled",
                return_value=True,
            ) as auto_update_enabled,
        ):
            await app._handle_command("/version")
        await pilot.pause()

        app_msgs = [m for m in app.query(AppMessage) if not m._is_markdown]
        content = str(app_msgs[-1]._content)
        assert "Update available: v99.99.99" in content
        assert "method that installed this copy of dcode" in content
        assert "automatically" not in content
        auto_update_enabled.assert_not_called()


async def test_version_slash_command_falls_back_when_preference_lookup_fails() -> None:
    """Verify `/version` survives an unreadable auto-update preference."""
    from deepagents_code.app import DeepAgentsApp
    from deepagents_code.tui.widgets.messages import AppMessage

    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app._update_available = (True, "99.99.99")
        with (
            patch(
                "deepagents_code.update_check.detect_install_method",
                return_value="uv",
            ),
            patch(
                "deepagents_code.update_check.is_auto_update_enabled",
                side_effect=AttributeError("malformed update config"),
            ),
        ):
            await app._handle_command("/version")
        await pilot.pause()

        app_msgs = [m for m in app.query(AppMessage) if not m._is_markdown]
        content = str(app_msgs[-1]._content)
        assert "Update available: v99.99.99" in content
        assert "/update" in content
        assert "automatically" not in content


async def test_version_slash_command_omits_update_hint_when_up_to_date() -> None:
    """Verify `/version` does not add the update hint when none is pending."""
    from deepagents_code.app import DeepAgentsApp
    from deepagents_code.tui.widgets.messages import AppMessage

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


async def test_version_slash_command_indicates_editable_install() -> None:
    """Verify `/version` reports editable mode and lists core dependencies."""
    from deepagents_code.app import DeepAgentsApp
    from deepagents_code.tui.widgets.messages import AppMessage

    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        with (
            patch(
                "deepagents_code.config._is_editable_install",
                return_value=True,
            ),
            patch(
                "deepagents_code.config._get_editable_install_path",
                return_value="~/src/deepagents/libs/code",
            ),
        ):
            await app._handle_command("/version")
        await pilot.pause()

        msgs = app.query(AppMessage)
        plain = str([m for m in msgs if not m._is_markdown][-1]._content)
        assert "Editable install: ~/src/deepagents/libs/code" in plain

        md_sources = [str(m._content) for m in msgs if m._is_markdown]
        core = [s for s in md_sources if "### Core dependencies" in s]
        assert core
        assert "langchain-core" in core[-1]
        assert "langgraph" in core[-1]
        assert "langsmith" in core[-1]


async def test_version_slash_command_omits_editable_info_when_not_editable() -> None:
    """Verify `/version` hides editable info and core deps for normal installs."""
    from deepagents_code.app import DeepAgentsApp
    from deepagents_code.tui.widgets.messages import AppMessage

    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        with patch(
            "deepagents_code.config._is_editable_install",
            return_value=False,
        ):
            await app._handle_command("/version")
        await pilot.pause()

        msgs = app.query(AppMessage)
        plain = str([m for m in msgs if not m._is_markdown][-1]._content)
        assert "Editable install" not in plain
        md_sources = [str(m._content) for m in msgs if m._is_markdown]
        assert not any("### Core dependencies" in s for s in md_sources)


def test_build_version_text_includes_editable_core_deps() -> None:
    """Verify the `--version` text reports editable mode and core deps."""
    from deepagents_code.main import build_version_text

    with (
        patch(
            "deepagents_code.config._is_editable_install",
            return_value=True,
        ),
        patch(
            "deepagents_code.config._get_editable_install_path",
            return_value="~/src/deepagents/libs/code",
        ),
    ):
        text = build_version_text()

    assert f"deepagents-code {__version__}" in text
    assert "Editable install: ~/src/deepagents/libs/code" in text
    assert "Core dependencies:" in text
    assert "langchain-core" in text
    assert "langsmith" in text


def test_build_version_text_omits_core_deps_when_not_editable() -> None:
    """Verify the `--version` text hides editable info for normal installs."""
    from deepagents_code.main import build_version_text

    with patch(
        "deepagents_code.config._is_editable_install",
        return_value=False,
    ):
        text = build_version_text()

    assert "Editable install" not in text
    assert "Core dependencies:" not in text


def test_format_core_dependencies_lists_known_packages() -> None:
    """Verify `format_core_dependencies` renders a row for each core package."""
    from deepagents_code.extras_info import (
        CORE_DEPENDENCIES,
        format_core_dependencies,
    )

    rendered = format_core_dependencies()
    assert rendered.startswith("### Core dependencies")
    for name in CORE_DEPENDENCIES:
        assert f"| {name} |" in rendered


def test_get_core_dependency_versions_marks_missing_as_none() -> None:
    """Verify missing core packages resolve to `None` rather than raising."""
    from importlib.metadata import PackageNotFoundError

    from deepagents_code.extras_info import get_core_dependency_versions

    def patched_version(name: str) -> str:
        if name == "langgraph-sdk":
            raise PackageNotFoundError(name)
        return "1.2.3"

    with patch("deepagents_code.extras_info.pkg_version", side_effect=patched_version):
        result = dict(get_core_dependency_versions())

    assert result["langgraph-sdk"] is None
    assert result["langchain"] == "1.2.3"


async def test_update_slash_command_editable_install_short_circuits() -> None:
    """Editable install must not invoke `perform_upgrade` from the TUI.

    A regression here would run `uv tool upgrade deepagents-code` on an
    editable dev checkout and clobber the local install with a PyPI copy.
    """
    from deepagents_code.app import DeepAgentsApp
    from deepagents_code.tui.widgets.messages import AppMessage

    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        with (
            patch(
                "deepagents_code.config._is_editable_install",
                return_value=True,
            ),
            patch("deepagents_code.update_check.is_update_available") as is_update_mock,
            patch(
                "deepagents_code.update_check.perform_upgrade",
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
    from deepagents_code.app import DeepAgentsApp
    from deepagents_code.tui.widgets.messages import AppMessage

    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        with (
            patch(
                "deepagents_code.config._is_editable_install",
                return_value=False,
            ),
            patch(
                "deepagents_code.update_check.is_update_available",
                return_value=(False, None),
            ) as is_update_mock,
            patch(
                "deepagents_code.update_check.perform_upgrade",
                new_callable=AsyncMock,
            ) as perform_upgrade_mock,
        ):
            await app._handle_command("/update")
            await pilot.pause()

        is_update_mock.assert_called_once_with(
            bypass_cache=True,
            include_prereleases=None,
        )
        perform_upgrade_mock.assert_not_awaited()
        app_msgs = [m for m in app.query(AppMessage) if not m._is_markdown]
        content = str(app_msgs[-1]._content)
        assert "Could not determine the latest version" in content


async def test_update_slash_command_omitted_prerelease_preserves_channel() -> None:
    """`/update` lets update helpers infer the channel from the installed version."""
    from deepagents_code.app import DeepAgentsApp
    from deepagents_code.tui.widgets.messages import AppMessage

    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        with (
            patch(
                "deepagents_code.config._is_editable_install",
                return_value=False,
            ),
            patch(
                "deepagents_code.update_check.is_update_available",
                return_value=(True, "99.0.0"),
            ) as is_update_mock,
            patch(
                "deepagents_code.update_check.perform_upgrade",
                new_callable=AsyncMock,
                return_value=(True, ""),
            ) as perform_upgrade_mock,
        ):
            await app._handle_command("/update")
            await pilot.pause()

        is_update_mock.assert_called_once_with(
            bypass_cache=True,
            include_prereleases=None,
        )
        perform_upgrade_mock.assert_awaited_once_with(
            include_prereleases=None,
            target_version="99.0.0",
        )
        app_msgs = [m for m in app.query(AppMessage) if not m._is_markdown]
        assert "Updated to v99.0.0" in str(app_msgs[-1]._content)


async def test_update_slash_command_stable_prerelease_deps_keep_intent_none() -> None:
    """Stable releases with pre-release deps let `perform_upgrade` pin the app."""
    from deepagents_code.app import DeepAgentsApp

    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        with (
            patch(
                "deepagents_code.config._is_editable_install",
                return_value=False,
            ),
            patch(
                "deepagents_code.update_check.is_update_available",
                return_value=(True, "99.0.0"),
            ),
            patch(
                "deepagents_code.update_check.release_requires_prereleases",
                return_value=True,
            ),
            patch(
                "deepagents_code.update_check.detect_install_method",
                return_value="uv",
            ),
            patch(
                "deepagents_code.update_check.perform_upgrade",
                new_callable=AsyncMock,
                return_value=(True, ""),
            ) as perform_upgrade_mock,
        ):
            await app._handle_command("/update")
            await pilot.pause()

    perform_upgrade_mock.assert_awaited_once_with(
        include_prereleases=None,
        target_version="99.0.0",
    )


async def test_update_slash_command_prerelease_updates_channel() -> None:
    """`/update --prerelease` opts into alpha/beta/rc releases."""
    from deepagents_code.app import DeepAgentsApp
    from deepagents_code.tui.widgets.messages import AppMessage, UserMessage

    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        with (
            patch(
                "deepagents_code.config._is_editable_install",
                return_value=False,
            ),
            # `--prerelease` is only honored on uv installs; pin the detected
            # method so the precheck doesn't refuse based on the test runner's
            # own environment.
            patch(
                "deepagents_code.update_check.detect_install_method",
                return_value="uv",
            ),
            patch(
                "deepagents_code.update_check.is_update_available",
                return_value=(True, "99.0.0rc1"),
            ) as is_update_mock,
            patch(
                "deepagents_code.update_check.perform_upgrade",
                new_callable=AsyncMock,
                return_value=(True, ""),
            ) as perform_upgrade_mock,
        ):
            await app._handle_command("/update --prerelease")
            await pilot.pause()

        is_update_mock.assert_called_once_with(
            bypass_cache=True,
            include_prereleases=True,
        )
        perform_upgrade_mock.assert_awaited_once_with(
            include_prereleases=True,
            target_version="99.0.0rc1",
        )
        user_msgs = list(app.query(UserMessage))
        assert str(user_msgs[-1]._content) == "/update --prerelease"
        app_msgs = [m for m in app.query(AppMessage) if not m._is_markdown]
        assert "Updated to v99.0.0rc1" in str(app_msgs[-1]._content)


async def test_update_slash_command_replaces_success_with_shadow_warning() -> None:
    """A shadowed `dcode` after a successful upgrade swaps success line for warning.

    Regression guard for the inverted-conditional bug class: showing the
    user a green "relaunch to use the new version" line and then *also*
    warning that relaunching will keep the old version is the exact UX
    this branch exists to prevent. The shadow path must mount an
    `ErrorMessage` with the warning and skip the success `AppMessage`
    entirely; a regression that flipped the `if/else` (or kept the
    success line unconditionally) would ship a reassuring success line
    over a broken upgrade.
    """
    from deepagents_code.app import DeepAgentsApp
    from deepagents_code.tui.widgets.messages import AppMessage, ErrorMessage
    from deepagents_code.update_check import ShadowedDcode

    shadow = ShadowedDcode(
        shadowing_bin=Path("/opt/stale/bin/dcode"),
        upgraded_bin_dir=Path("/home/user/.local/bin"),
    )
    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        with (
            patch(
                "deepagents_code.config._is_editable_install",
                return_value=False,
            ),
            patch(
                "deepagents_code.update_check.is_update_available",
                return_value=(True, "99.0.0"),
            ),
            patch(
                "deepagents_code.update_check.perform_upgrade",
                new_callable=AsyncMock,
                return_value=(True, ""),
            ),
            # Override the module-level autouse `None` patch with the
            # positive case. Innermost patch wins.
            patch(
                "deepagents_code.update_check.detect_shadowed_dcode",
                return_value=shadow,
            ),
        ):
            await app._handle_command("/update")
            await pilot.pause()

        plain_msgs = [
            str(m._content) for m in app.query(AppMessage) if not m._is_markdown
        ]
        # The shadow path must NOT mount the success line, because relaunching
        # would not actually use the new version. A regression that kept the
        # success line would show both messages, contradicting itself.
        assert not any("Updated to v99.0.0" in m for m in plain_msgs)
        # The warning is mounted as an `ErrorMessage` (red), not a generic
        # `AppMessage`, so it visually stands apart from neutral status text.
        error_msgs = [str(m._content) for m in app.query(ErrorMessage)]
        assert any("/opt/stale/bin/dcode" in m for m in error_msgs)
        assert any("/home/user/.local/bin" in m for m in error_msgs)


async def test_update_slash_command_prerelease_unsupported_install_refuses() -> None:
    """`/update --prerelease` refuses on a non-uv install before hitting PyPI."""
    from deepagents_code.app import DeepAgentsApp
    from deepagents_code.tui.widgets.messages import AppMessage

    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        with (
            patch(
                "deepagents_code.config._is_editable_install",
                return_value=False,
            ),
            # Pin a non-uv install so the precheck's refusal is driven by the
            # test rather than the test runner's own environment.
            patch(
                "deepagents_code.update_check.detect_install_method",
                return_value="brew",
            ),
            patch(
                "deepagents_code.update_check.is_update_available",
            ) as is_update_mock,
            patch(
                "deepagents_code.update_check.perform_upgrade",
                new_callable=AsyncMock,
            ) as perform_upgrade_mock,
        ):
            await app._handle_command("/update --prerelease")
            await pilot.pause()

        # The refusal must short-circuit before promising or attempting an
        # upgrade the install method can't honor.
        is_update_mock.assert_not_called()
        perform_upgrade_mock.assert_not_awaited()
        app_msgs = [m for m in app.query(AppMessage) if not m._is_markdown]
        assert "aren't supported for this install" in str(app_msgs[-1]._content)


async def test_update_slash_command_rejects_unknown_option() -> None:
    """`/update` surfaces typo'd options instead of silently running stable."""
    from deepagents_code.app import DeepAgentsApp
    from deepagents_code.tui.widgets.messages import AppMessage

    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        with patch(
            "deepagents_code.update_check.is_update_available",
        ) as is_update_mock:
            await app._handle_command("/update --prereleases")
            await pilot.pause()

        # A discarded flag must never silently fall through to an update check.
        is_update_mock.assert_not_called()
        app_msgs = [m for m in app.query(AppMessage) if not m._is_markdown]
        assert "Unknown option(s) for /update: --prereleases" in str(
            app_msgs[-1]._content
        )


async def test_update_deps_refreshes_when_dcode_current() -> None:
    """`/update --deps` re-resolves deps after confirming dcode is current."""
    from deepagents_code.app import DeepAgentsApp
    from deepagents_code.tui.widgets.messages import AppMessage

    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        with (
            patch(
                "deepagents_code.config._is_editable_install",
                return_value=False,
            ),
            patch(
                "deepagents_code.update_check.is_update_available",
                return_value=(False, "1.0.0"),
            ) as is_update_mock,
            patch(
                "deepagents_code.update_check.perform_dependency_refresh",
                new_callable=AsyncMock,
                return_value=(
                    True,
                    " - langchain-openai==1.3.2\n + langchain-openai==1.5.0\n",
                ),
            ) as refresh_mock,
        ):
            await app._handle_command("/update --deps")
            await pilot.pause()

        is_update_mock.assert_called_once_with(
            bypass_cache=True,
            include_prereleases=None,
        )
        refresh_mock.assert_awaited_once_with(include_prereleases=None)
        app_msgs = [m for m in app.query(AppMessage) if not m._is_markdown]
        content = str(app_msgs[-1]._content)
        assert "Refreshed dependencies:" in content
        assert "langchain-openai  1.3.2 -> 1.5.0" in content


async def test_update_deps_reports_when_already_current() -> None:
    """`/update --deps` reports when nothing changed."""
    from deepagents_code.app import DeepAgentsApp
    from deepagents_code.tui.widgets.messages import AppMessage

    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        with (
            patch(
                "deepagents_code.config._is_editable_install",
                return_value=False,
            ),
            patch(
                "deepagents_code.update_check.is_update_available",
                return_value=(False, "1.0.0"),
            ),
            patch(
                "deepagents_code.update_check.perform_dependency_refresh",
                new_callable=AsyncMock,
                return_value=(True, "Resolved 120 packages in 12ms\n"),
            ),
        ):
            await app._handle_command("/update --deps")
            await pilot.pause()

        app_msgs = [m for m in app.query(AppMessage) if not m._is_markdown]
        assert "Dependencies are already up to date." in str(app_msgs[-1]._content)


async def test_update_deps_routes_outdated_dcode_through_regular_update() -> None:
    """`/update --deps` runs the normal update flow when dcode is outdated."""
    from deepagents_code.app import DeepAgentsApp
    from deepagents_code.tui.widgets.messages import AppMessage

    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        with (
            patch(
                "deepagents_code.config._is_editable_install",
                return_value=False,
            ),
            patch(
                "deepagents_code.update_check.is_update_available",
                return_value=(True, "1.1.0"),
            ),
            patch.object(
                DeepAgentsApp,
                "_confirm_update_before_dependency_refresh",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch(
                "deepagents_code.update_check.perform_upgrade",
                new_callable=AsyncMock,
                return_value=(
                    True,
                    " - deepagents-code==1.0.0\n + deepagents-code==1.1.0\n",
                ),
            ),
        ):
            await app._handle_command("/update --deps")
            await pilot.pause()

        app_msgs = [m for m in app.query(AppMessage) if not m._is_markdown]
        content = str(app_msgs[-1]._content)
        assert "Updated to v1.1.0" in content
        assert "Quit and relaunch dcode to use the new version" in content
        assert "Updated deepagents-code:" not in content
        assert "Dependencies are already up to date." not in content


async def test_update_deps_skips_refresh_prompt_when_refresh_unsupported() -> None:
    """Unsupported refresh installs take the normal outdated dcode update path."""
    from deepagents_code.app import DeepAgentsApp
    from deepagents_code.tui.widgets.messages import AppMessage

    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        with (
            patch(
                "deepagents_code.config._is_editable_install",
                return_value=False,
            ),
            patch(
                "deepagents_code.update_check.is_update_available",
                return_value=(True, "1.1.0"),
            ),
            patch(
                "deepagents_code.update_check.dependency_refresh_supported",
                return_value=(False, "Homebrew install detected — ..."),
            ),
            patch.object(
                DeepAgentsApp,
                "_confirm_update_before_dependency_refresh",
                new_callable=AsyncMock,
            ) as confirm_mock,
            patch(
                "deepagents_code.update_check.perform_dependency_refresh",
                new_callable=AsyncMock,
            ) as refresh_mock,
            patch(
                "deepagents_code.update_check.perform_upgrade",
                new_callable=AsyncMock,
                return_value=(
                    True,
                    " - deepagents-code==1.0.0\n + deepagents-code==1.1.0\n",
                ),
            ) as perform_upgrade_mock,
        ):
            await app._handle_command("/update --deps")
            await pilot.pause()

        confirm_mock.assert_not_awaited()
        refresh_mock.assert_not_awaited()
        perform_upgrade_mock.assert_awaited_once_with(
            include_prereleases=None,
            target_version="1.1.0",
        )
        app_msgs = [m for m in app.query(AppMessage) if not m._is_markdown]
        content = str(app_msgs[-1]._content)
        assert "Updated to v1.1.0" in content
        assert "Updated deepagents-code:" not in content
        assert "Dependency refresh failed" not in content


async def test_update_deps_decline_app_update_refreshes_current_deps() -> None:
    """Declining the app update refreshes deps without upgrading dcode."""
    from deepagents_code.app import DeepAgentsApp
    from deepagents_code.tui.widgets.messages import AppMessage

    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        with (
            patch(
                "deepagents_code.config._is_editable_install",
                return_value=False,
            ),
            patch(
                "deepagents_code.update_check.is_update_available",
                return_value=(True, "1.1.0"),
            ),
            patch(
                "deepagents_code.update_check.dependency_refresh_supported",
                return_value=(True, None),
            ),
            patch.object(
                DeepAgentsApp,
                "_confirm_update_before_dependency_refresh",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch(
                "deepagents_code.update_check.perform_dependency_refresh",
                new_callable=AsyncMock,
                return_value=(
                    True,
                    " - langchain-openai==1.3.2\n + langchain-openai==1.5.0\n",
                ),
            ) as refresh_mock,
            patch(
                "deepagents_code.update_check.perform_upgrade",
                new_callable=AsyncMock,
            ) as perform_upgrade_mock,
        ):
            await app._handle_command("/update --deps")
            await pilot.pause()

        refresh_mock.assert_awaited_once_with(include_prereleases=None)
        perform_upgrade_mock.assert_not_awaited()
        app_msgs = [m for m in app.query(AppMessage) if not m._is_markdown]
        content = str(app_msgs[-1]._content)
        assert "Refreshed dependencies:" in content
        assert "langchain-openai  1.3.2 -> 1.5.0" in content
        assert "A deepagents-code update is available: v1.1.0." in content
        assert "Updated to v1.1.0" not in content


async def test_update_deps_decline_app_update_reports_no_new_deps() -> None:
    """`/update --deps` reports current deps even when dcode has an update."""
    from deepagents_code.app import DeepAgentsApp
    from deepagents_code.tui.widgets.messages import AppMessage

    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        with (
            patch(
                "deepagents_code.config._is_editable_install",
                return_value=False,
            ),
            patch(
                "deepagents_code.update_check.is_update_available",
                return_value=(True, "1.1.0"),
            ),
            patch(
                "deepagents_code.update_check.dependency_refresh_supported",
                return_value=(True, None),
            ),
            patch.object(
                DeepAgentsApp,
                "_confirm_update_before_dependency_refresh",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch(
                "deepagents_code.update_check.perform_dependency_refresh",
                new_callable=AsyncMock,
                return_value=(True, "Resolved 120 packages in 12ms\n"),
            ),
            patch(
                "deepagents_code.update_check.perform_upgrade",
                new_callable=AsyncMock,
            ) as perform_upgrade_mock,
        ):
            await app._handle_command("/update --deps")
            await pilot.pause()

        perform_upgrade_mock.assert_not_awaited()
        app_msgs = [m for m in app.query(AppMessage) if not m._is_markdown]
        content = str(app_msgs[-1]._content)
        assert "Dependencies are already up to date." in content
        assert "A deepagents-code update is available: v1.1.0." in content
        assert "Updated to v1.1.0" not in content


async def test_update_already_current_prompts_and_refreshes_on_confirm() -> None:
    """Plain `/update` offers a dep refresh when dcode is already current."""
    from deepagents_code.app import DeepAgentsApp
    from deepagents_code.tui.widgets.messages import AppMessage

    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        with (
            patch(
                "deepagents_code.config._is_editable_install",
                return_value=False,
            ),
            patch(
                "deepagents_code.update_check.is_update_available",
                return_value=(False, "1.0.0"),
            ),
            patch(
                "deepagents_code.update_check.dependency_refresh_supported",
                return_value=(True, None),
            ),
            patch(
                "deepagents_code.update_check.perform_dependency_refresh_dry_run",
                new_callable=AsyncMock,
                return_value=(
                    True,
                    " - langchain-openai==1.3.2\n + langchain-openai==1.5.0\n",
                ),
            ) as dry_run_mock,
            patch.object(
                DeepAgentsApp,
                "_confirm_refresh_dependencies",
                new_callable=AsyncMock,
                return_value=True,
            ) as confirm_mock,
            patch(
                "deepagents_code.update_check.perform_dependency_refresh",
                new_callable=AsyncMock,
                return_value=(
                    True,
                    " - langchain-openai==1.3.2\n + langchain-openai==1.5.0\n",
                ),
            ) as refresh_mock,
        ):
            await app._handle_command("/update")
            await pilot.pause()

        dry_run_mock.assert_awaited_once_with(include_prereleases=None)
        confirm_mock.assert_awaited_once_with(
            planned_changes="  langchain-openai  1.3.2 -> 1.5.0",
        )
        refresh_mock.assert_awaited_once_with(include_prereleases=None)
        app_msgs = [m for m in app.query(AppMessage) if not m._is_markdown]
        assert "Refreshed dependencies:" in str(app_msgs[-1]._content)


async def test_update_already_current_skips_refresh_on_decline() -> None:
    """Declining the prompt leaves the install untouched."""
    from deepagents_code.app import DeepAgentsApp
    from deepagents_code.tui.widgets.messages import AppMessage

    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        with (
            patch(
                "deepagents_code.config._is_editable_install",
                return_value=False,
            ),
            patch(
                "deepagents_code.update_check.is_update_available",
                return_value=(False, "1.0.0"),
            ),
            patch(
                "deepagents_code.update_check.dependency_refresh_supported",
                return_value=(True, None),
            ),
            patch(
                "deepagents_code.update_check.perform_dependency_refresh_dry_run",
                new_callable=AsyncMock,
                return_value=(
                    True,
                    " - langchain-openai==1.3.2\n + langchain-openai==1.5.0\n",
                ),
            ) as dry_run_mock,
            patch.object(
                DeepAgentsApp,
                "_confirm_refresh_dependencies",
                new_callable=AsyncMock,
                return_value=False,
            ) as confirm_mock,
            patch(
                "deepagents_code.update_check.perform_dependency_refresh",
                new_callable=AsyncMock,
            ) as refresh_mock,
        ):
            await app._handle_command("/update")
            await pilot.pause()

        dry_run_mock.assert_awaited_once_with(include_prereleases=None)
        confirm_mock.assert_awaited_once_with(
            planned_changes="  langchain-openai  1.3.2 -> 1.5.0",
        )
        refresh_mock.assert_not_awaited()
        app_msgs = [m for m in app.query(AppMessage) if not m._is_markdown]
        assert "Dependency refresh skipped." in str(app_msgs[-1]._content)


async def test_update_already_current_reports_no_dependency_changes() -> None:
    """Plain `/update` skips the prompt when the dry run finds no changes."""
    from deepagents_code.app import DeepAgentsApp
    from deepagents_code.tui.widgets.messages import AppMessage

    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        with (
            patch(
                "deepagents_code.config._is_editable_install",
                return_value=False,
            ),
            patch(
                "deepagents_code.update_check.is_update_available",
                return_value=(False, "1.0.0"),
            ),
            patch(
                "deepagents_code.update_check.dependency_refresh_supported",
                return_value=(True, None),
            ),
            patch(
                "deepagents_code.update_check.perform_dependency_refresh_dry_run",
                new_callable=AsyncMock,
                return_value=(True, "Resolved 120 packages in 12ms\n"),
            ) as dry_run_mock,
            patch.object(
                DeepAgentsApp,
                "_confirm_refresh_dependencies",
                new_callable=AsyncMock,
            ) as confirm_mock,
            patch(
                "deepagents_code.update_check.perform_dependency_refresh",
                new_callable=AsyncMock,
            ) as refresh_mock,
        ):
            await app._handle_command("/update")
            await pilot.pause()

            dry_run_mock.assert_awaited_once_with(include_prereleases=None)
            confirm_mock.assert_not_awaited()
            refresh_mock.assert_not_awaited()
            app_msgs = [m for m in app.query(AppMessage) if not m._is_markdown]
            assert "Dependencies are already up to date." in str(app_msgs[-1]._content)


async def test_update_already_current_reports_dependency_check_failure() -> None:
    """A failed dry-run check reports the failure without refreshing."""
    from deepagents_code.app import DeepAgentsApp
    from deepagents_code.tui.widgets.messages import AppMessage

    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        with (
            patch(
                "deepagents_code.config._is_editable_install",
                return_value=False,
            ),
            patch(
                "deepagents_code.update_check.is_update_available",
                return_value=(False, "1.0.0"),
            ),
            patch(
                "deepagents_code.update_check.dependency_refresh_supported",
                return_value=(True, None),
            ),
            patch(
                "deepagents_code.update_check.perform_dependency_refresh_dry_run",
                new_callable=AsyncMock,
                return_value=(False, "No solution found"),
            ),
            patch.object(
                DeepAgentsApp,
                "_confirm_refresh_dependencies",
                new_callable=AsyncMock,
            ) as confirm_mock,
            patch(
                "deepagents_code.update_check.perform_dependency_refresh",
                new_callable=AsyncMock,
            ) as refresh_mock,
        ):
            await app._handle_command("/update")
            await pilot.pause()

            confirm_mock.assert_not_awaited()
            refresh_mock.assert_not_awaited()
            app_msgs = [m for m in app.query(AppMessage) if not m._is_markdown]
            content = str(app_msgs[-1]._content)
            assert "Could not check dependency updates" in content
            assert "No solution found" in content


async def test_update_already_current_skips_prompt_when_refresh_unsupported() -> None:
    """brew/other installs aren't prompted for a refresh or prerelease support."""
    from deepagents_code.app import DeepAgentsApp
    from deepagents_code.tui.widgets.messages import AppMessage

    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        with (
            patch(
                "deepagents_code.config._is_editable_install",
                return_value=False,
            ),
            patch(
                "deepagents_code.update_check.is_update_available",
                return_value=(False, "1.0.0"),
            ),
            patch(
                "deepagents_code.update_check.release_requires_prereleases",
                return_value=True,
            ),
            patch(
                "deepagents_code.update_check.prerelease_upgrade_supported",
                return_value=(False, "Pre-release updates aren't supported"),
            ) as prerelease_supported_mock,
            patch(
                "deepagents_code.update_check.dependency_refresh_supported",
                return_value=(False, "Homebrew install detected — ..."),
            ),
            patch.object(
                DeepAgentsApp,
                "_confirm_refresh_dependencies",
                new_callable=AsyncMock,
            ) as confirm_mock,
            patch(
                "deepagents_code.update_check.perform_dependency_refresh",
                new_callable=AsyncMock,
            ) as refresh_mock,
        ):
            await app._handle_command("/update")
            await pilot.pause()

        confirm_mock.assert_not_awaited()
        refresh_mock.assert_not_awaited()
        prerelease_supported_mock.assert_not_called()
        app_msgs = [m for m in app.query(AppMessage) if not m._is_markdown]
        assert "Already on the latest version" in str(app_msgs[-1]._content)


async def test_refresh_dependencies_surfaces_failure() -> None:
    """A failed refresh reports the start of uv's output, not silence."""
    from deepagents_code.app import DeepAgentsApp
    from deepagents_code.tui.widgets.messages import AppMessage

    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        with patch(
            "deepagents_code.update_check.perform_dependency_refresh",
            new_callable=AsyncMock,
            return_value=(False, "No solution found for langchain-openai" + "x" * 400),
        ):
            await app._refresh_dependencies(include_prereleases=None)
            await pilot.pause()

        app_msgs = [m for m in app.query(AppMessage) if not m._is_markdown]
        content = str(app_msgs[-1]._content)
        assert "Dependency refresh failed" in content
        assert "No solution found" in content


async def test_refresh_dependencies_renders_self_changes() -> None:
    """A `deepagents-code` line in the diff renders under its own heading."""
    from deepagents_code.app import DeepAgentsApp
    from deepagents_code.tui.widgets.messages import AppMessage

    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        with patch(
            "deepagents_code.update_check.perform_dependency_refresh",
            new_callable=AsyncMock,
            return_value=(
                True,
                (
                    " - deepagents-code==1.0.0\n + deepagents-code==1.0.1\n"
                    " - langchain-openai==1.3.2\n + langchain-openai==1.5.0\n"
                ),
            ),
        ):
            await app._refresh_dependencies(include_prereleases=None)
            await pilot.pause()

        app_msgs = [m for m in app.query(AppMessage) if not m._is_markdown]
        content = str(app_msgs[-1]._content)
        assert "Updated deepagents-code:" in content
        assert "Refreshed dependencies:" in content
        assert "langchain-openai  1.3.2 -> 1.5.0" in content


async def test_refresh_dependencies_skips_in_debug_mode(monkeypatch) -> None:
    """`DEBUG_UPDATE` short-circuits before shelling out to uv."""
    from deepagents_code._env_vars import DEBUG_UPDATE
    from deepagents_code.app import DeepAgentsApp
    from deepagents_code.tui.widgets.messages import AppMessage

    monkeypatch.setenv(DEBUG_UPDATE, "1")
    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        with patch(
            "deepagents_code.update_check.perform_dependency_refresh",
            new_callable=AsyncMock,
        ) as refresh_mock:
            await app._refresh_dependencies(include_prereleases=None)
            await pilot.pause()

        refresh_mock.assert_not_awaited()
        app_msgs = [m for m in app.query(AppMessage) if not m._is_markdown]
        assert "Skipped dependency refresh (debug mode)." in str(app_msgs[-1]._content)


async def test_confirm_refresh_dependencies_reports_mount_failure() -> None:
    """A modal that fails to mount is surfaced, not silently swallowed."""
    from deepagents_code.app import DeepAgentsApp
    from deepagents_code.tui.widgets.messages import AppMessage

    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        with patch.object(
            DeepAgentsApp,
            "_push_screen_wait",
            Mock(side_effect=RuntimeError("boom")),
        ):
            result = await app._confirm_refresh_dependencies()
            await pilot.pause()

        assert result is False
        app_msgs = [m for m in app.query(AppMessage) if not m._is_markdown]
        assert "Couldn't show the dependency-refresh prompt" in str(
            app_msgs[-1]._content
        )


async def test_confirm_refresh_dependencies_reports_timeout(monkeypatch) -> None:
    """A modal that never resolves is bounded by the watchdog and surfaced."""
    from deepagents_code import app as app_module
    from deepagents_code.app import DeepAgentsApp
    from deepagents_code.tui.widgets.messages import AppMessage

    async def _never(_self: object, _screen: object) -> None:
        await asyncio.sleep(10)

    monkeypatch.setattr(app_module, "_MODAL_WATCHDOG_TIMEOUT_SECONDS", 0.01)
    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        with patch.object(DeepAgentsApp, "_push_screen_wait", _never):
            result = await app._confirm_refresh_dependencies()
            await pilot.pause()

        assert result is False
        app_msgs = [m for m in app.query(AppMessage) if not m._is_markdown]
        assert "timed out" in str(app_msgs[-1]._content)


async def test_confirm_update_before_dependency_refresh_reports_mount_failure() -> None:
    """A failed app-update prompt falls back to refreshing current deps."""
    from deepagents_code.app import DeepAgentsApp
    from deepagents_code.tui.widgets.messages import AppMessage

    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        with patch.object(
            DeepAgentsApp,
            "_push_screen_wait",
            Mock(side_effect=RuntimeError("boom")),
        ):
            result = await app._confirm_update_before_dependency_refresh(
                current="1.0.0",
                latest="1.1.0",
            )
            await pilot.pause()

        assert result is False
        app_msgs = [m for m in app.query(AppMessage) if not m._is_markdown]
        assert "Couldn't show the update prompt" in str(app_msgs[-1]._content)


def test_help_mentions_version_flag() -> None:
    """Verify that the CLI help text mentions `--version` and SDK."""
    result = subprocess.run(
        [sys.executable, "-m", "deepagents_code.main", "help"],
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
        [sys.executable, "-m", "deepagents_code.main", "--help"],
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
        [sys.executable, "-m", "deepagents_code.main", "-h"],
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
        [sys.executable, "-m", "deepagents_code.main", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    # Help should succeed
    assert result.returncode == 0
    # Help should NOT contain Interactive Features section
    assert "Interactive Features" not in result.stdout
