"""Tests for version-related functionality."""

from __future__ import annotations

import asyncio
import subprocess
import sys
import tomllib
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import ANY, AsyncMock, Mock, patch

import pytest

from deepagents_code._version import __version__

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterator

    from deepagents_code.extras_info import VersionReport

    # `conftest`'s fixture protocols cannot be imported here: `tool.ty`
    # `extra-paths` puts `libs/deepagents` on the path too, so
    # `tests.unit_tests.conftest` is ambiguous across packages.
    DrainModalCommands = Callable[..., Awaitable[None]]
    WaitForModal = Callable[..., Awaitable[None]]


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


@pytest.fixture(autouse=True)
def _isolate_update_log_dir(tmp_path: Path) -> Iterator[None]:
    """Keep `/update` tests out of the developer's real update-log directory.

    `UPDATE_LOG_DIR` is a module-level constant baked from `default_cache_dir()`
    at import, so no environment variable redirects it. Any test that reaches
    `create_update_log_path` therefore runs `cleanup_update_logs` against the
    real cache dir, pruning a user's genuine logs past `UPDATE_LOG_MAX_FILES`.
    Tests that assert on a specific path override this with their own patch.
    """
    with patch("deepagents_code.update_check.UPDATE_LOG_DIR", tmp_path / "update_logs"):
        yield


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
    from deepagents_code.extras_info import collect_version_report

    sdk_version = collect_version_report().display_sdk_version
    assert f"deepagents (SDK) {sdk_version}" in result.stdout
    # Extras block is plain-text (no markdown table or headings).
    assert "Installed optional dependencies:" in result.stdout
    assert "langchain-anthropic" in result.stdout
    assert "| Extra" not in result.stdout
    assert "###" not in result.stdout


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


def _editable_exact_pin_version_report(cli_metadata: str = "0.1.40") -> VersionReport:
    """Build a report with editable installs, CLI drift, and a newer SDK pin."""
    from packaging.requirements import Requirement

    from deepagents_code.extras_info import DistributionVersion, VersionReport

    return VersionReport(
        cli=DistributionVersion(
            name="deepagents-code",
            source_version=__version__,
            metadata_version=cli_metadata,
            editable=True,
            source_path="~/src/deepagents/libs/code",
            status="resolved",
        ),
        sdk=DistributionVersion(
            name="deepagents",
            source_version="0.6.12",
            metadata_version="0.6.12",
            editable=True,
            source_path="~/src/deepagents/libs/deepagents",
            status="resolved",
        ),
        sdk_requirement=Requirement("deepagents==0.7.0a7"),
        sdk_requirement_satisfied=True,
    )


def test_build_version_text_reports_editable_drift_and_newer_sdk_pin() -> None:
    """`--version` surfaces CLI stale metadata and the effective SDK pin."""
    from deepagents_code.main import build_version_text

    with patch(
        "deepagents_code.extras_info.collect_version_report",
        return_value=_editable_exact_pin_version_report(),
    ):
        text = build_version_text()

    # Editable status is shown on its own `Editable install:` line, so the CLI
    # version line carries only the source/metadata drift, not `editable`.
    assert f"deepagents-code {__version__} (installed metadata: 0.1.40)" in text
    assert (
        "deepagents (SDK) 0.7.0a7+editable "
        "(workspace HEAD; source marker: 0.6.12)" in text
    )


async def test_version_slash_command_reports_editable_drift_and_newer_sdk_pin() -> None:
    """`/version` renders the same editable pin facts as `--version`."""
    from deepagents_code.app import DeepAgentsApp
    from deepagents_code.tui.widgets.messages import AppMessage

    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        with (
            patch(
                "deepagents_code.extras_info.collect_version_report",
                return_value=_editable_exact_pin_version_report(),
            ),
            patch(
                "deepagents_code.update_check.get_sdk_release_time",
                return_value=None,
            ) as get_sdk_release_time,
        ):
            await app._handle_command("/version")
        await pilot.pause()

        get_sdk_release_time.assert_called_once_with("0.7.0a7")
        content = str(
            [m for m in app.query(AppMessage) if not m._is_markdown][-1]._content
        )
        assert (
            f"deepagents-code version: {__version__} "
            "(installed metadata: 0.1.40)" in content
        )
        assert (
            "deepagents (SDK) version: 0.7.0a7+editable "
            "(workspace HEAD; source marker: 0.6.12)" in content
        )


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


async def test_update_slash_command_links_log_instead_of_listing_dependencies() -> None:
    """A normal app upgrade keeps dependency churn in its exact update log."""
    from deepagents_code.app import DeepAgentsApp
    from deepagents_code.tui.widgets.messages import AppMessage, ErrorMessage

    log_path = Path("/tmp/dcode update.log")
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
                "deepagents_code.update_check.create_update_log_file",
                return_value=log_path,
            ),
            patch(
                "deepagents_code.update_check.perform_upgrade",
                new_callable=AsyncMock,
                return_value=(
                    True,
                    (
                        " - deepagents-code==1.0.0\n"
                        " + deepagents-code==99.0.0\n"
                        " - anthropic==0.120.2\n"
                        " + anthropic==0.122.0\n"
                    ),
                    "99.0.0",
                ),
            ) as perform_upgrade_mock,
        ):
            await app._handle_command("/update")
            await pilot.pause()

        perform_upgrade_mock.assert_awaited_once_with(
            log_path=log_path,
            include_prereleases=None,
            target_version="99.0.0",
        )
        content = "\n".join(
            str(m._content) for m in app.query(AppMessage) if not m._is_markdown
        )
        assert "Update log: tail -f '/tmp/dcode update.log'" in content
        # Pin the success branch. The negative assertions below are only
        # meaningful if the flow actually reached the branch that used to print
        # the dependency summary — a swallowed exception (e.g. a `perform_upgrade`
        # arity change) would otherwise make them pass vacuously. `ErrorMessage`
        # is not an `AppMessage` subclass, so `content` cannot see failures.
        assert "Updated to v99.0.0" in content
        assert not app.query(ErrorMessage)
        assert "Dependencies updated:" not in content
        assert "anthropic" not in content


async def test_update_slash_command_uses_powershell_to_follow_logs() -> None:
    """Windows updates show a PowerShell-compatible log-follow command."""
    from deepagents_code.app import DeepAgentsApp
    from deepagents_code.tui.widgets.messages import AppMessage

    log_path = Path(r"C:\Users\dcode update.log")
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
                "deepagents_code.update_check.create_update_log_file",
                return_value=log_path,
            ),
            patch("deepagents_code.update_check.sys.platform", "win32"),
            patch(
                "deepagents_code.update_check.perform_upgrade",
                new_callable=AsyncMock,
                return_value=(True, "", "99.0.0"),
            ),
        ):
            await app._handle_command("/update")
            await pilot.pause()

        content = "\n".join(
            str(m._content) for m in app.query(AppMessage) if not m._is_markdown
        )
        assert (
            "Update log: Get-Content -Wait -LiteralPath 'C:\\Users\\dcode update.log'"
        ) in content


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
                return_value=(True, "", "99.0.0"),
            ) as perform_upgrade_mock,
        ):
            await app._handle_command("/update")
            await pilot.pause()

    perform_upgrade_mock.assert_awaited_once_with(
        log_path=ANY,
        include_prereleases=None,
        target_version="99.0.0",
    )


async def test_update_slash_command_reports_success_without_a_banner() -> None:
    """A missing splash banner cannot turn a successful `/update` into a failure.

    `_perform_app_upgrade` runs inside the caller's broad
    `except Exception -> _mount_update_failure`, so an unguarded banner lookup
    would report a completed upgrade as "Update failed" and name an internal
    CSS selector at the user. The repaint is cosmetic; the success line is not.
    """
    from deepagents_code.app import DeepAgentsApp
    from deepagents_code.tui.widgets.messages import AppMessage, ErrorMessage
    from deepagents_code.tui.widgets.welcome import WelcomeBanner

    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        await app.query_one("#welcome-banner", WelcomeBanner).remove()
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
                return_value=(True, "", "99.0.0"),
            ),
        ):
            await app._handle_command("/update")
            await pilot.pause()

        plain_msgs = [
            str(m._content) for m in app.query(AppMessage) if not m._is_markdown
        ]
        assert any("Updated to v99.0.0" in m for m in plain_msgs)
        error_msgs = [str(m._content) for m in app.query(ErrorMessage)]
        assert not any("Update failed" in m for m in error_msgs)


async def test_update_slash_command_replaces_success_with_shadow_warning() -> None:
    """A shadowed `dcode` after a successful upgrade swaps success line for warning.

    Regression guard for the inverted-conditional bug class: showing the
    user a green "relaunch to use the new version" line and then *also*
    warning that relaunching will keep the old version is the exact UX
    this branch exists to prevent. The shadow path must mount an
    `ErrorMessage` with the warning and skip the success `AppMessage`
    entirely; a regression that flipped the `if/else` (or kept the
    success line unconditionally) would ship a reassuring success line
    over a broken upgrade. The splash version carries the same guarantee
    for the same reason, and is asserted here too.
    """
    from deepagents_code.app import DeepAgentsApp
    from deepagents_code.tui.widgets.messages import AppMessage, ErrorMessage
    from deepagents_code.tui.widgets.welcome import WelcomeBanner
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
                return_value=(True, "", "99.0.0"),
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
            splash = (
                app.query_one("#welcome-banner", WelcomeBanner)._build_banner().plain
            )

        plain_msgs = [
            str(m._content) for m in app.query(AppMessage) if not m._is_markdown
        ]
        # The shadow path must NOT mount the success line, because relaunching
        # would not actually use the new version. A regression that kept the
        # success line would show both messages, contradicting itself.
        assert not any("Updated to v99.0.0" in m for m in plain_msgs)
        # Same reasoning for the splash: bumping the banner here would promise
        # an upgrade that a relaunch will not actually deliver. This is the
        # only guard on `_update_splash_version` staying inside the
        # `shadow is None` branch, so do not drop it as redundant.
        assert f"v{__version__}" in splash
        assert "v99.0.0" not in splash
        # The warning is mounted as an `ErrorMessage` (red), not a generic
        # `AppMessage`, so it visually stands apart from neutral status text.
        error_msgs = [str(m._content) for m in app.query(ErrorMessage)]
        assert any("/opt/stale/bin/dcode" in m for m in error_msgs)
        assert any("/home/user/.local/bin" in m for m in error_msgs)


async def test_update_deps_routes_outdated_dcode_through_regular_update(
    drain_modal_commands: DrainModalCommands,
) -> None:
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
                    "1.1.0",
                ),
            ),
        ):
            await app._handle_command("/update --deps")
            await drain_modal_commands(app)
            await pilot.pause()

        app_msgs = [m for m in app.query(AppMessage) if not m._is_markdown]
        content = str(app_msgs[-1]._content)
        assert "Updated to v1.1.0" in content
        assert "Quit and relaunch dcode to use the new version" in content
        assert "Updated deepagents-code:" not in content
        assert "Dependencies are already up to date." not in content


async def test_update_deps_skips_refresh_prompt_when_refresh_unsupported(
    drain_modal_commands: DrainModalCommands,
) -> None:
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
                    "1.1.0",
                ),
            ) as perform_upgrade_mock,
        ):
            await app._handle_command("/update --deps")
            await drain_modal_commands(app)
            await pilot.pause()

        confirm_mock.assert_not_awaited()
        refresh_mock.assert_not_awaited()
        perform_upgrade_mock.assert_awaited_once_with(
            log_path=ANY,
            include_prereleases=None,
            target_version="1.1.0",
        )
        app_msgs = [m for m in app.query(AppMessage) if not m._is_markdown]
        content = str(app_msgs[-1]._content)
        assert "Updated to v1.1.0" in content
        assert "Updated deepagents-code:" not in content
        assert "Dependency refresh failed" not in content


async def test_update_deps_decline_app_update_refreshes_current_deps(
    drain_modal_commands: DrainModalCommands,
) -> None:
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
            await drain_modal_commands(app)
            await pilot.pause()

        refresh_mock.assert_awaited_once_with(include_prereleases=None)
        perform_upgrade_mock.assert_not_awaited()
        app_msgs = [m for m in app.query(AppMessage) if not m._is_markdown]
        content = str(app_msgs[-1]._content)
        assert "Refreshed dependencies:" in content
        assert "langchain-openai  1.3.2 -> 1.5.0" in content
        assert "A deepagents-code update is available: v1.1.0." in content
        assert "Updated to v1.1.0" not in content


async def test_update_deps_decline_app_update_reports_no_new_deps(
    drain_modal_commands: DrainModalCommands,
) -> None:
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
            await drain_modal_commands(app)
            await pilot.pause()

        perform_upgrade_mock.assert_not_awaited()
        app_msgs = [m for m in app.query(AppMessage) if not m._is_markdown]
        content = str(app_msgs[-1]._content)
        assert "Dependencies are already up to date." in content
        assert "A deepagents-code update is available: v1.1.0." in content
        assert "Updated to v1.1.0" not in content


async def test_update_already_current_prompts_and_refreshes_on_confirm(
    drain_modal_commands: DrainModalCommands,
) -> None:
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
            await drain_modal_commands(app)
            await pilot.pause()

        dry_run_mock.assert_awaited_once_with(include_prereleases=None)
        confirm_mock.assert_awaited_once_with(
            planned_changes="  langchain-openai  1.3.2 -> 1.5.0",
        )
        refresh_mock.assert_awaited_once_with(include_prereleases=None)
        app_msgs = [m for m in app.query(AppMessage) if not m._is_markdown]
        assert "Refreshed dependencies:" in str(app_msgs[-1]._content)


async def test_update_refresh_continuation_surfaces_unexpected_error(
    drain_modal_commands: DrainModalCommands,
) -> None:
    """A raise inside the detached refresh continuation reaches the user.

    The continuation runs outside `_handle_update_command`'s `try/except`, so
    without its own handler the failure would reach only `_log_task_exception`
    and the user would see no outcome at all.
    """
    from deepagents_code.app import DeepAgentsApp
    from deepagents_code.tui.widgets.messages import ErrorMessage

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
            ),
            patch.object(
                app,
                "_confirm_refresh_dependencies",
                new=AsyncMock(return_value=True),
            ),
            patch.object(
                app,
                "_refresh_dependencies",
                new=AsyncMock(side_effect=RuntimeError("resolver blew up")),
            ),
        ):
            await app._handle_command("/update")
            await drain_modal_commands(app)
            await pilot.pause()

        joined = "\n".join(str(m._content) for m in app.query(ErrorMessage))
        assert "Update failed: RuntimeError" in joined
        assert "resolver blew up" in joined


async def test_update_already_current_skips_refresh_on_decline(
    drain_modal_commands: DrainModalCommands,
) -> None:
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
            await drain_modal_commands(app)
            await pilot.pause()

        dry_run_mock.assert_awaited_once_with(include_prereleases=None)
        confirm_mock.assert_awaited_once_with(
            planned_changes="  langchain-openai  1.3.2 -> 1.5.0",
        )
        refresh_mock.assert_not_awaited()
        app_msgs = [m for m in app.query(AppMessage) if not m._is_markdown]
        assert "Dependency refresh skipped." in str(app_msgs[-1]._content)


async def test_update_refresh_prompt_responsive_through_message_pump(
    drain_modal_commands: DrainModalCommands,
    wait_for_modal: WaitForModal,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dependency-refresh modal must accept keys via the real submit path.

    Regression test: `/update` is dispatched from the App's
    `on_chat_input_submitted` handler, which is awaited inline on the App message
    pump. Previously the confirmation was `await`ed in that chain, so the pump
    stayed blocked while the modal was open and the modal never received the
    Enter/Esc key events it needs to resolve — it appeared frozen until the
    confirmation helper's watchdog fired. The confirmation now runs off the pump
    (`_schedule_off_message_pump`), so submitting through the real
    `ChatInput.Submitted` path and pressing Enter must resolve the modal and run
    the refresh rather than wedging the UI.

    The watchdog is shortened here so a regression fails on a real assertion in
    seconds. At its production length the blocked pump also stalls `run_test()`
    teardown, and the failure surfaces as a bare pytest-timeout that says
    nothing about the cause.
    """
    from deepagents_code import app as app_module
    from deepagents_code.app import DeepAgentsApp

    monkeypatch.setattr(app_module, "_MODAL_WATCHDOG_TIMEOUT_SECONDS", 2.0)
    from deepagents_code.tui.widgets.chat_input import ChatInput
    from deepagents_code.tui.widgets.update_confirm import (
        RefreshDependenciesConfirmScreen,
    )

    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        # Idle session so the submission is processed instead of queued.
        app._agent_running = False
        app._connecting = False
        app._startup_sequence_running = False
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
            ),
            patch(
                "deepagents_code.update_check.perform_dependency_refresh",
                new_callable=AsyncMock,
                return_value=(
                    True,
                    " - langchain-openai==1.3.2\n + langchain-openai==1.5.0\n",
                ),
            ) as refresh_mock,
        ):
            # Submit through the message pump, exactly as the ChatInput widget
            # would, rather than awaiting `_handle_command` directly (which would
            # not exercise the pump-blocking path).
            assert app._chat_input is not None
            app._chat_input.post_message(ChatInput.Submitted("/update", "command"))
            await wait_for_modal(
                pilot,
                RefreshDependenciesConfirmScreen,
                present=True,
            )

            # With the pump free, Enter must reach the modal and resolve it.
            await pilot.press("enter")
            await wait_for_modal(
                pilot,
                RefreshDependenciesConfirmScreen,
                present=False,
            )
            await drain_modal_commands(app)

        refresh_mock.assert_awaited_once_with(include_prereleases=None)


async def test_update_deps_prompt_responsive_through_message_pump(
    drain_modal_commands: DrainModalCommands,
    wait_for_modal: WaitForModal,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The `--deps` app-update modal must accept keys via the real submit path.

    Same pump-blocking regression as the dependency-refresh modal: Esc has to
    reach `UpdateBeforeDependenciesConfirmScreen` for the user to keep the
    current version and refresh its dependencies. The watchdog is shortened for
    the same reason as there — so a regression fails legibly instead of hanging.
    """
    from deepagents_code import app as app_module
    from deepagents_code.app import DeepAgentsApp

    monkeypatch.setattr(app_module, "_MODAL_WATCHDOG_TIMEOUT_SECONDS", 2.0)
    from deepagents_code.tui.widgets.chat_input import ChatInput
    from deepagents_code.tui.widgets.update_confirm import (
        UpdateBeforeDependenciesConfirmScreen,
    )

    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app._agent_running = False
        app._connecting = False
        app._startup_sequence_running = False
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
            patch(
                "deepagents_code.update_check.perform_upgrade",
                new_callable=AsyncMock,
            ) as upgrade_mock,
            patch(
                "deepagents_code.update_check.perform_dependency_refresh",
                new_callable=AsyncMock,
                return_value=(True, ""),
            ) as refresh_mock,
        ):
            assert app._chat_input is not None
            app._chat_input.post_message(
                ChatInput.Submitted("/update --deps", "command"),
            )
            await wait_for_modal(
                pilot,
                UpdateBeforeDependenciesConfirmScreen,
                present=True,
            )

            await pilot.press("escape")
            await wait_for_modal(
                pilot,
                UpdateBeforeDependenciesConfirmScreen,
                present=False,
            )
            await drain_modal_commands(app)

        refresh_mock.assert_awaited_once_with(include_prereleases=None)
        upgrade_mock.assert_not_awaited()


async def test_update_already_current_reports_no_dependency_changes(
    drain_modal_commands: DrainModalCommands,
) -> None:
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
            await drain_modal_commands(app)
            await pilot.pause()

            dry_run_mock.assert_awaited_once_with(include_prereleases=None)
            confirm_mock.assert_not_awaited()
            refresh_mock.assert_not_awaited()
            app_msgs = [m for m in app.query(AppMessage) if not m._is_markdown]
            assert "Dependencies are already up to date." in str(app_msgs[-1]._content)


async def test_update_already_current_reports_dependency_check_failure(
    drain_modal_commands: DrainModalCommands,
) -> None:
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
            await drain_modal_commands(app)
            await pilot.pause()

            confirm_mock.assert_not_awaited()
            refresh_mock.assert_not_awaited()
            app_msgs = [m for m in app.query(AppMessage) if not m._is_markdown]
            content = str(app_msgs[-1]._content)
            assert "Could not check dependency updates" in content
            assert "No solution found" in content


async def test_update_already_current_skips_prompt_when_refresh_unsupported(
    drain_modal_commands: DrainModalCommands,
) -> None:
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
            await drain_modal_commands(app)
            await pilot.pause()

        confirm_mock.assert_not_awaited()
        refresh_mock.assert_not_awaited()
        prerelease_supported_mock.assert_not_called()
        app_msgs = [m for m in app.query(AppMessage) if not m._is_markdown]
        assert "Already on the latest version" in str(app_msgs[-1]._content)


async def test_perform_app_upgrade_failure_surfaces_manual_command() -> None:
    """A failed upgrade reports the detail plus a pinned manual command.

    The `/update` path into this branch had no app-level coverage; the only
    `"Auto-update failed"` assertions live in `test_main.py`, which exercises
    `main._run_startup_auto_update` instead.
    """
    from deepagents_code.app import DeepAgentsApp
    from deepagents_code.tui.widgets.messages import AppMessage

    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        with (
            patch(
                "deepagents_code.update_check.perform_upgrade",
                new_callable=AsyncMock,
                return_value=(False, "resolver: conflict", None),
            ),
            patch(
                "deepagents_code.update_check.upgrade_command",
                return_value="uv tool install -U 'deepagents-code==1.1.0'",
            ),
        ):
            await app._perform_app_upgrade(
                current="1.0.0",
                latest="1.1.0",
                include_prereleases=None,
                upgrade_include_prereleases=None,
                pin_upgrade_version="1.1.0",
            )
            await pilot.pause()

        content = " ".join(
            str(m._content) for m in app.query(AppMessage) if not m._is_markdown
        )
        assert "Auto-update failed" in content
        assert "resolver: conflict" in content
        assert "uv tool install -U 'deepagents-code==1.1.0'" in content


async def test_perform_app_upgrade_failure_repeats_log_path() -> None:
    """A failed upgrade names its log again, not just in the pre-install hint.

    The failure detail is truncated to 200 characters, so the log is the only
    complete record — and the hint mounted before the install has scrolled away
    behind the installer's streamed output by the time the failure lands.
    """
    from deepagents_code.app import DeepAgentsApp
    from deepagents_code.tui.widgets.messages import AppMessage

    log_path = Path("/tmp/dcode failed-update.log")
    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        with (
            patch(
                "deepagents_code.update_check.create_update_log_file",
                return_value=log_path,
            ),
            patch(
                "deepagents_code.update_check.perform_upgrade",
                new_callable=AsyncMock,
                return_value=(False, "resolver: conflict", None),
            ),
            patch(
                "deepagents_code.update_check.upgrade_command",
                return_value="uv tool install -U 'deepagents-code==1.1.0'",
            ),
        ):
            await app._perform_app_upgrade(
                current="1.0.0",
                latest="1.1.0",
                include_prereleases=None,
                upgrade_include_prereleases=None,
                pin_upgrade_version="1.1.0",
            )
            await pilot.pause()

        failures = [
            str(m._content)
            for m in app.query(AppMessage)
            if not m._is_markdown and "Auto-update failed" in str(m._content)
        ]
        assert failures
        assert f"Log: {log_path}" in failures[0]


async def test_perform_app_upgrade_omits_log_hint_when_log_uncreatable() -> None:
    """An uncreatable log is silently skipped, never advertised as a dead path.

    `create_update_log_file` returns `None` when the cache dir cannot be
    written. Pointing the user at `tail -f <path>` for a file that will never
    exist is worse than saying nothing, and the upgrade itself must still run.
    """
    from deepagents_code.app import DeepAgentsApp
    from deepagents_code.tui.widgets.messages import AppMessage, ErrorMessage

    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        with (
            patch(
                "deepagents_code.update_check.create_update_log_file",
                return_value=None,
            ),
            patch(
                "deepagents_code.update_check.perform_upgrade",
                new_callable=AsyncMock,
                return_value=(True, "", "1.1.0"),
            ) as perform_upgrade_mock,
        ):
            await app._perform_app_upgrade(
                current="1.0.0",
                latest="1.1.0",
                include_prereleases=None,
                upgrade_include_prereleases=None,
                pin_upgrade_version="1.1.0",
            )
            await pilot.pause()

        perform_upgrade_mock.assert_awaited_once_with(
            log_path=None,
            include_prereleases=None,
            target_version="1.1.0",
        )
        content = "\n".join(
            str(m._content) for m in app.query(AppMessage) if not m._is_markdown
        )
        assert "Update log:" not in content
        assert "Updated to v1.1.0" in content
        assert not app.query(ErrorMessage)


async def test_perform_app_upgrade_defers_to_another_session() -> None:
    """`/update` must not install while an install is already running.

    `_environment_mutation_lock` only serializes this process, so the update
    lock is what stops two terminals from racing one tool environment. Held
    in-process here for determinism, which exercises the caller's response to
    a refusal; `TestUpdateInstallLock` covers the cross-process guarantee.
    """
    from deepagents_code.app import DeepAgentsApp
    from deepagents_code.tui.widgets.messages import AppMessage
    from deepagents_code.update_check import update_install_lock

    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        with (
            patch(
                "deepagents_code.update_check.perform_upgrade",
                new_callable=AsyncMock,
            ) as upgrade_mock,
            update_install_lock() as holding,
        ):
            assert holding is True
            await app._perform_app_upgrade(
                current="1.0.0",
                latest="1.1.0",
                include_prereleases=None,
                upgrade_include_prereleases=None,
                pin_upgrade_version=None,
            )
            await pilot.pause()

        upgrade_mock.assert_not_awaited()
        app_msgs = [m for m in app.query(AppMessage) if not m._is_markdown]
        assert "An update is already being installed" in str(app_msgs[-1]._content)


async def test_perform_app_upgrade_holds_the_lock_during_install() -> None:
    """`/update` must run the install inside the lock, not after a check.

    Shrinking the `with` to cover only the boolean check would leave the
    install unguarded while the deferral test above still passed. The lock is
    not reentrant, so re-entering from inside `perform_upgrade` proves it is
    held for the real work.
    """
    from deepagents_code.app import DeepAgentsApp
    from deepagents_code.update_check import update_install_lock

    held_during_install: list[bool] = []

    async def _record_lock_state(  # noqa: RUF029
        **_kwargs: object,
    ) -> tuple[bool, str, str | None]:
        with update_install_lock() as holding:
            held_during_install.append(holding)
        return True, "", "1.1.0"

    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        with patch(
            "deepagents_code.update_check.perform_upgrade",
            new=_record_lock_state,
        ):
            await app._perform_app_upgrade(
                current="1.0.0",
                latest="1.1.0",
                include_prereleases=None,
                upgrade_include_prereleases=None,
                pin_upgrade_version=None,
            )
            await pilot.pause()

    assert held_during_install == [False], (
        "the install ran without holding the update lock"
    )


async def test_perform_app_upgrade_skips_install_in_debug_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`DEBUG_UPDATE` short-circuits the app upgrade before shelling out to uv."""
    from deepagents_code._env_vars import DEBUG_UPDATE
    from deepagents_code.app import DeepAgentsApp
    from deepagents_code.tui.widgets.messages import AppMessage

    monkeypatch.setenv(DEBUG_UPDATE, "1")
    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        with patch(
            "deepagents_code.update_check.perform_upgrade",
            new_callable=AsyncMock,
        ) as upgrade_mock:
            await app._perform_app_upgrade(
                current="1.0.0",
                latest="1.1.0",
                include_prereleases=None,
                upgrade_include_prereleases=None,
                pin_upgrade_version=None,
            )
            await pilot.pause()

        upgrade_mock.assert_not_awaited()
        app_msgs = [m for m in app.query(AppMessage) if not m._is_markdown]
        assert "Skipped update install (debug mode)." in str(app_msgs[-1]._content)


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
