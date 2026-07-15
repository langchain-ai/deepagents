"""Unit tests for main entry point."""

import asyncio
import inspect
import os
import signal
import sys
from collections.abc import Callable, Iterator
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
from rich.console import Console

from deepagents_code.app import AppResult, DeepAgentsApp, run_textual_app
from deepagents_code.config import build_langsmith_thread_url, reset_langsmith_url_cache
from deepagents_code.main import (
    _auto_install_ripgrep_cli,
    _handle_termination_signal,
    _install_termination_signal_handlers,
    _is_managed_ripgrep_path,
    _render_teardown_thread_hints,
    _restart_current_process,
    _ripgrep_install_hint,
    _run_startup_auto_update,
    _should_check_teardown_thread,
    _terminal_row_count,
    build_missing_tool_notification,
    check_optional_tools,
    cli_main,
    format_tool_warning_cli,
    run_textual_cli_async,
)
from deepagents_code.mcp_tools import ProjectServerSummary

# Most unit tests set `DEEPAGENTS_CODE_NO_UPDATE_CHECK=1` and patch
# `is_update_check_enabled()` to avoid accidental PyPI/DNS work. This module
# tests startup update behavior itself, so each test must control those values.
pytestmark = pytest.mark.self_managed_update_check


class TestTerminationSignalHandling:
    """Tests for terminating-signal cleanup wiring."""

    def test_posix_installs_unwinding_handler(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """POSIX terminating signals unwind so cleanup can run."""
        monkeypatch.setattr("deepagents_code.main.sys.platform", "linux")
        install = MagicMock()
        monkeypatch.setattr("deepagents_code.main.signal.signal", install)

        _install_termination_signal_handlers()

        assert install.call_args_list == [
            ((signal.SIGHUP, _handle_termination_signal),),
            ((signal.SIGTERM, _handle_termination_signal),),
            ((signal.SIGQUIT, _handle_termination_signal),),
        ]
        for signum in (signal.SIGHUP, signal.SIGTERM, signal.SIGQUIT):
            with pytest.raises(SystemExit) as exc_info:
                _handle_termination_signal(signum, None)
            assert exc_info.value.code == 128 + signum

    def test_windows_does_not_install_termination_signal_handlers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Windows skips the POSIX-only SIGHUP API."""
        monkeypatch.setattr("deepagents_code.main.sys.platform", "win32")
        install = MagicMock()
        monkeypatch.setattr("deepagents_code.main.signal.signal", install)

        _install_termination_signal_handlers()

        install.assert_not_called()


class TestStartupAutoUpdate:
    """Tests for startup auto-update behavior."""

    @pytest.fixture(autouse=True)
    def _no_prerelease_lookup(self) -> Iterator[None]:
        """Stub the pre-release dependency lookup for startup tests.

        The startup auto-update path calls `release_requires_prereleases`
        (e.g. in the restart-loop guard) with `latest`. Unstubbed, that reads
        the real host cache and falls through to a live PyPI request, which is
        non-hermetic and would hit the network under a bare `pytest` run. Pin it
        to `False`; the function's own behavior is covered in `test_update_check`.
        """
        with patch(
            "deepagents_code.update_check.release_requires_prereleases",
            return_value=False,
        ):
            yield

    @pytest.fixture(autouse=True)
    def _ack_auto_update_default(self) -> Iterator[None]:
        """Treat the auto-update default as already acknowledged.

        These tests exercise the install/restart path; the one-time migration
        notice is covered in `TestAutoUpdateDefaultMigration`.
        """
        with patch(
            "deepagents_code.update_check.should_announce_auto_update_default",
            return_value=False,
        ):
            yield

    @pytest.fixture(autouse=True)
    def _no_shadowed_dcode(self) -> Iterator[None]:
        """Default to "no PATH shadow detected" for the success-path tests.

        Without this, every successful-upgrade test would run the real
        `detect_shadowed_dcode` against the host filesystem. That's
        hermetic only by accident — the test runner's editable install
        currently short-circuits at `detect_install_method() != "uv"` — but
        a uv-tool-managed Python or CI image that does match would silently
        re-route every "successful update" test through the new
        `if shadow is not None: return` branch and skip the restart
        assertion. Pin to `None` here so the contract being tested is
        "shadow path is opt-in"; the dedicated shadow-present test below
        patches it explicitly.

        Patches at the source module rather than `deepagents_code.main`
        because `_run_startup_auto_update` lazy-imports it inside the
        function.
        """
        with patch(
            "deepagents_code.update_check.detect_shadowed_dcode",
            return_value=None,
        ):
            yield

    def test_successful_update_restarts_before_launch(self) -> None:
        """A successful startup auto-update should exec a fresh process."""
        console = MagicMock()
        upgrade = AsyncMock(return_value=(True, "updated"))

        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch(
                "deepagents_code.update_check.is_update_check_enabled",
                return_value=True,
            ),
            patch(
                "deepagents_code.update_check.is_auto_update_enabled",
                return_value=True,
            ),
            patch(
                "deepagents_code.update_check.get_cached_update_available",
                return_value=(True, "9.9.9"),
            ),
            patch(
                "deepagents_code.update_check.format_release_age_parenthetical",
                return_value="",
            ),
            patch(
                "deepagents_code.update_check.create_update_log_path",
                return_value=Path("/tmp/dcode-update.log"),
            ),
            patch("deepagents_code.update_check.perform_upgrade", upgrade),
            patch(
                "deepagents_code.update_check.clear_startup_auto_update_failure"
            ) as clear_failure,
            patch(
                "deepagents_code.main._restart_current_process",
                side_effect=SystemExit(0),
            ) as restart,
            pytest.raises(SystemExit),
        ):
            _run_startup_auto_update(console)

        upgrade.assert_awaited_once()
        clear_failure.assert_called_once_with("9.9.9")
        printed = " ".join(str(c.args[0]) for c in console.print.call_args_list)
        assert "tail -f /tmp/dcode-update.log" in printed
        restart.assert_called_once_with()

    def test_successful_update_skips_restart_when_shadowed(self) -> None:
        """Successful upgrade + shadowed dcode must NOT restart into the old binary.

        Regression guard for the critical bug: when a stale `dcode` is
        earlier on PATH than uv's bin dir, re-exec'ing would silently
        re-launch the old version. The pre-launch path must surface a
        warning and return *before* `_restart_current_process` so the user
        sees the message and isn't stranded on the old in-memory version
        with no explanation. Also pins the markup-escape behavior: a path
        containing a Rich-special character must not raise.
        """
        from deepagents_code.update_check import ShadowedDcode

        console = MagicMock()
        upgrade = AsyncMock(return_value=(True, "updated"))
        # Embed `[` in the shadowing path — legal on POSIX filesystems —
        # so a regression that dropped `escape()` would raise a Rich
        # `MarkupError` here instead of silently emitting broken styling.
        shadow = ShadowedDcode(
            shadowing_bin=Path("/opt/old [legacy]/bin/dcode"),
            upgraded_bin_dir=Path("/home/user/.local/bin"),
        )

        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch(
                "deepagents_code.update_check.is_update_check_enabled",
                return_value=True,
            ),
            patch(
                "deepagents_code.update_check.is_auto_update_enabled",
                return_value=True,
            ),
            patch(
                "deepagents_code.update_check.get_cached_update_available",
                return_value=(True, "9.9.9"),
            ),
            patch(
                "deepagents_code.update_check.format_release_age_parenthetical",
                return_value="",
            ),
            patch(
                "deepagents_code.update_check.create_update_log_path",
                return_value=Path("/tmp/dcode-update.log"),
            ),
            patch("deepagents_code.update_check.perform_upgrade", upgrade),
            # Override the autouse `_no_shadowed_dcode` fixture for this
            # single test by re-patching the same name with the positive
            # case. The innermost patch wins, so the autouse fixture's
            # `None` doesn't leak through.
            patch(
                "deepagents_code.update_check.detect_shadowed_dcode",
                return_value=shadow,
            ),
            patch("deepagents_code.main._restart_current_process") as restart,
        ):
            _run_startup_auto_update(console)

        upgrade.assert_awaited_once()
        restart.assert_not_called()
        printed = " ".join(str(c.args[0]) for c in console.print.call_args_list)
        assert "Warning:" in printed
        # The path's `[legacy]` segment must be Rich-escaped (`\[legacy]`)
        # before interpolation under `markup=True`; a regression that
        # dropped `escape()` would either raise `MarkupError` (test fails)
        # or render `[legacy]` as a (broken) style tag. Asserting the
        # escaped form pins the fix.
        assert "/opt/old \\[legacy]/bin/dcode" in printed
        assert "/home/user/.local/bin" in printed
        assert "Continuing with v" in printed

    def test_disabled_update_does_not_check_pypi(self) -> None:
        """Disabled auto-update should not perform network or install work."""
        console = MagicMock()

        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch(
                "deepagents_code.update_check.is_auto_update_enabled",
                return_value=False,
            ),
            patch("deepagents_code.update_check.get_cached_update_available") as check,
            patch("deepagents_code.update_check.perform_upgrade") as upgrade,
        ):
            _run_startup_auto_update(console)

        check.assert_not_called()
        upgrade.assert_not_called()

    def test_disabled_update_check_skips_cached_auto_update(self) -> None:
        """Disabled update checks should block cached startup auto-updates."""
        console = MagicMock()

        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch(
                "deepagents_code.update_check.is_update_check_enabled",
                return_value=False,
            ),
            patch(
                "deepagents_code.update_check.is_auto_update_enabled",
                return_value=True,
            ),
            patch("deepagents_code.update_check.get_cached_update_available") as check,
            patch("deepagents_code.update_check.perform_upgrade") as upgrade,
            patch("deepagents_code.main._restart_current_process") as restart,
        ):
            _run_startup_auto_update(console)

        check.assert_not_called()
        upgrade.assert_not_called()
        restart.assert_not_called()

    def test_restart_uses_module_entrypoint(self) -> None:
        """Restart should reload package code from the updated environment."""
        with (
            patch.object(sys, "executable", "/tool/bin/python"),
            patch.object(sys, "argv", ["dcode", "--model", "openai:gpt-5.5"]),
            patch("os.execv", side_effect=SystemExit(0)) as execv,
            pytest.raises(SystemExit),
        ):
            _restart_current_process()

        execv.assert_called_once_with(
            "/tool/bin/python",
            ["/tool/bin/python", "-m", "deepagents_code", "--model", "openai:gpt-5.5"],
        )

    def test_failed_update_does_not_restart_and_continues(self) -> None:
        """A failed upgrade must not restart; it surfaces the error and returns."""
        console = MagicMock()
        upgrade = AsyncMock(return_value=(False, "pip exploded"))

        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch(
                "deepagents_code.update_check.is_auto_update_enabled",
                return_value=True,
            ),
            patch(
                "deepagents_code.update_check.get_cached_update_available",
                return_value=(True, "9.9.9"),
            ),
            patch(
                "deepagents_code.update_check.format_release_age_parenthetical",
                return_value="",
            ),
            patch(
                "deepagents_code.update_check.create_update_log_path",
                return_value=Path("/tmp/dcode-update.log"),
            ),
            patch(
                "deepagents_code.update_check.upgrade_command",
                return_value="uv tool upgrade deepagents-code",
            ),
            patch("deepagents_code.update_check.perform_upgrade", upgrade),
            patch(
                "deepagents_code.update_check.mark_startup_auto_update_failed"
            ) as mark_failed,
            patch("deepagents_code.main._restart_current_process") as restart,
        ):
            _run_startup_auto_update(console)

        upgrade.assert_awaited_once()
        mark_failed.assert_called_once_with("9.9.9")
        restart.assert_not_called()
        printed = " ".join(str(c.args[0]) for c in console.print.call_args_list)
        assert "Auto-update failed" in printed

    def test_unpersisted_failure_marker_warns_user(self) -> None:
        """An unwritable cooldown marker must be surfaced, not silently dropped."""
        console = MagicMock()
        upgrade = AsyncMock(return_value=(False, "pip exploded"))

        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch(
                "deepagents_code.update_check.is_auto_update_enabled",
                return_value=True,
            ),
            patch(
                "deepagents_code.update_check.get_cached_update_available",
                return_value=(True, "9.9.9"),
            ),
            patch(
                "deepagents_code.update_check.format_release_age_parenthetical",
                return_value="",
            ),
            patch(
                "deepagents_code.update_check.create_update_log_path",
                return_value=Path("/tmp/dcode-update.log"),
            ),
            patch(
                "deepagents_code.update_check.upgrade_command",
                return_value="uv tool upgrade deepagents-code",
            ),
            patch("deepagents_code.update_check.perform_upgrade", upgrade),
            # The marker write fails (e.g. a read-only state dir).
            patch(
                "deepagents_code.update_check.mark_startup_auto_update_failed",
                return_value=False,
            ),
            patch("deepagents_code.main._restart_current_process") as restart,
        ):
            _run_startup_auto_update(console)

        restart.assert_not_called()
        printed = " ".join(str(c.args[0]) for c in console.print.call_args_list)
        assert "could not be recorded" in printed

    def test_exception_during_upgrade_warns_if_failure_marker_is_unpersisted(
        self,
    ) -> None:
        """A raised upgrade must warn when its cooldown marker cannot be saved."""
        console = MagicMock()
        upgrade = AsyncMock(side_effect=RuntimeError("uv wedged"))

        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch(
                "deepagents_code.update_check.is_auto_update_enabled",
                return_value=True,
            ),
            patch(
                "deepagents_code.update_check.get_cached_update_available",
                return_value=(True, "9.9.9"),
            ),
            patch(
                "deepagents_code.update_check.format_release_age_parenthetical",
                return_value="",
            ),
            patch(
                "deepagents_code.update_check.create_update_log_path",
                return_value=Path("/tmp/dcode-update.log"),
            ),
            patch("deepagents_code.update_check.perform_upgrade", upgrade),
            patch(
                "deepagents_code.update_check.mark_startup_auto_update_failed",
                return_value=False,
            ) as mark_failed,
            patch("deepagents_code.main._restart_current_process") as restart,
        ):
            # Must not raise: the fail-soft handler swallows and continues.
            _run_startup_auto_update(console)

        mark_failed.assert_called_once_with("9.9.9")
        restart.assert_not_called()
        printed = " ".join(str(c.args[0]) for c in console.print.call_args_list)
        assert "Auto-update failed before startup" in printed
        assert "could not be recorded" in printed

    def test_recent_failure_cooldown_skips_startup_update(self) -> None:
        """A same-version startup failure cooldown must bypass repeat attempts."""
        console = MagicMock()

        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch(
                "deepagents_code.update_check.is_update_check_enabled",
                return_value=True,
            ),
            patch(
                "deepagents_code.update_check.is_auto_update_enabled",
                return_value=True,
            ),
            patch(
                "deepagents_code.update_check.get_cached_update_available",
                return_value=(True, "9.9.9"),
            ),
            patch(
                "deepagents_code.update_check.should_skip_startup_auto_update_after_failure",
                return_value=True,
            ) as should_skip,
            patch(
                "deepagents_code.update_check.upgrade_command",
                return_value="uv tool install -U deepagents-code",
            ),
            patch("deepagents_code.update_check.perform_upgrade") as upgrade,
            patch("deepagents_code.main._restart_current_process") as restart,
        ):
            _run_startup_auto_update(console)

        should_skip.assert_called_once_with("9.9.9")
        upgrade.assert_not_called()
        restart.assert_not_called()
        printed = " ".join(str(c.args[0]) for c in console.print.call_args_list)
        assert "recent failed attempt" in printed

    def test_editable_install_skips_update(self) -> None:
        """Editable installs must short-circuit before any PyPI/install work."""
        console = MagicMock()

        with (
            patch("deepagents_code.config._is_editable_install", return_value=True),
            patch(
                "deepagents_code.update_check.is_auto_update_enabled",
                return_value=True,
            ),
            patch("deepagents_code.update_check.get_cached_update_available") as check,
            patch("deepagents_code.update_check.perform_upgrade") as upgrade,
        ):
            _run_startup_auto_update(console)

        check.assert_not_called()
        upgrade.assert_not_called()

    def test_no_update_available_returns_early(self) -> None:
        """When already current, nothing is announced, installed, or restarted."""
        console = MagicMock()

        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch(
                "deepagents_code.update_check.is_auto_update_enabled",
                return_value=True,
            ),
            patch(
                "deepagents_code.update_check.get_cached_update_available",
                return_value=(False, None),
            ),
            patch("deepagents_code.update_check.perform_upgrade") as upgrade,
            patch("deepagents_code.main._restart_current_process") as restart,
        ):
            _run_startup_auto_update(console)

        upgrade.assert_not_called()
        restart.assert_not_called()
        console.print.assert_not_called()

    def test_in_session_update_already_installed_skips(self) -> None:
        """An in-session `/update` already on disk must not re-upgrade.

        The cache reports a newer version than the baked-in `__version__`,
        but the on-disk install already satisfies it, so the upgrade and
        restart are skipped silently.
        """
        console = MagicMock()

        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch(
                "deepagents_code.update_check.is_auto_update_enabled",
                return_value=True,
            ),
            patch(
                "deepagents_code.update_check.get_cached_update_available",
                return_value=(True, "9.9.9"),
            ),
            patch(
                "deepagents_code.update_check.is_installed_version_at_least",
                return_value=True,
            ),
            patch("deepagents_code.update_check.perform_upgrade") as upgrade,
            patch("deepagents_code.main._restart_current_process") as restart,
        ):
            _run_startup_auto_update(console)

        upgrade.assert_not_called()
        restart.assert_not_called()
        console.print.assert_not_called()

    def test_debug_update_skips_install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """DEBUG_UPDATE announces the update but skips the actual install."""
        console = MagicMock()
        monkeypatch.setenv("DEEPAGENTS_CODE_DEBUG_UPDATE", "1")

        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch(
                "deepagents_code.update_check.is_auto_update_enabled",
                return_value=True,
            ),
            patch(
                "deepagents_code.update_check.get_cached_update_available",
                return_value=(True, "9.9.9"),
            ),
            patch(
                "deepagents_code.update_check.format_release_age_parenthetical",
                return_value="",
            ),
            patch("deepagents_code.update_check.perform_upgrade") as upgrade,
            patch("deepagents_code.main._restart_current_process") as restart,
        ):
            _run_startup_auto_update(console)

        upgrade.assert_not_called()
        restart.assert_not_called()
        printed = " ".join(str(c.args[0]) for c in console.print.call_args_list)
        assert "debug mode" in printed

    def test_unexpected_error_does_not_block_startup(self) -> None:
        """An error in the update machinery must never block launch."""
        console = MagicMock()

        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch(
                "deepagents_code.update_check.is_auto_update_enabled",
                return_value=True,
            ),
            patch(
                "deepagents_code.update_check.get_cached_update_available",
                side_effect=RuntimeError("boom"),
            ),
            patch("deepagents_code.main._restart_current_process") as restart,
        ):
            # Must swallow the error rather than propagate it.
            _run_startup_auto_update(console)

        restart.assert_not_called()
        printed = " ".join(str(c.args[0]) for c in console.print.call_args_list)
        assert "Auto-update failed before startup" in printed

    def test_restart_loop_guard_skips_repeat_upgrade(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A re-exec that did not change the version must not re-upgrade."""
        console = MagicMock()
        # Simulate the sentinel set by the prior generation before its restart.
        monkeypatch.setenv("DEEPAGENTS_CODE_RESTARTED_AFTER_UPDATE", "9.9.9")

        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch(
                "deepagents_code.update_check.is_auto_update_enabled",
                return_value=True,
            ),
            patch(
                "deepagents_code.update_check.get_cached_update_available",
                return_value=(True, "9.9.9"),
            ),
            patch(
                "deepagents_code.update_check.upgrade_command",
                return_value="uv tool upgrade deepagents-code",
            ),
            patch("deepagents_code.update_check.perform_upgrade") as upgrade,
            patch("deepagents_code.main._restart_current_process") as restart,
        ):
            _run_startup_auto_update(console)

        upgrade.assert_not_called()
        restart.assert_not_called()
        # Sentinel is consumed so a genuine future update is not suppressed.
        assert os.environ.get("DEEPAGENTS_CODE_RESTARTED_AFTER_UPDATE") is None
        printed = " ".join(str(c.args[0]) for c in console.print.call_args_list)
        assert "restart loop" in printed

    def test_restart_failure_after_successful_install_continues(self) -> None:
        """A successful install with a failed re-exec reports an accurate message."""
        console = MagicMock()
        upgrade = AsyncMock(return_value=(True, "updated"))

        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch(
                "deepagents_code.update_check.is_auto_update_enabled",
                return_value=True,
            ),
            patch(
                "deepagents_code.update_check.get_cached_update_available",
                return_value=(True, "9.9.9"),
            ),
            patch(
                "deepagents_code.update_check.format_release_age_parenthetical",
                return_value="",
            ),
            patch(
                "deepagents_code.update_check.create_update_log_path",
                return_value=Path("/tmp/dcode-update.log"),
            ),
            patch("deepagents_code.update_check.perform_upgrade", upgrade),
            patch(
                "deepagents_code.main._restart_current_process",
                side_effect=OSError("exec failed"),
            ) as restart,
        ):
            # Install succeeded; a failed re-exec must not raise or claim the
            # update failed.
            _run_startup_auto_update(console)

        restart.assert_called_once_with()
        # Sentinel is dropped since the restart did not happen.
        assert os.environ.get("DEEPAGENTS_CODE_RESTARTED_AFTER_UPDATE") is None
        printed = " ".join(str(c.args[0]) for c in console.print.call_args_list)
        assert "automatic restart failed" in printed
        assert "Auto-update failed" not in printed

    def test_restart_after_update_clears_transient_launch_status(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The re-exec'd process rewrites `Launching...` to stable update text."""
        stream = StringIO()
        console = Console(file=stream, force_terminal=True, no_color=True, width=80)
        # The prior generation recorded the version it restarted into.
        monkeypatch.setenv("DEEPAGENTS_CODE_RESTARTED_AFTER_UPDATE", "9.9.9")

        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch(
                "deepagents_code.update_check.is_update_check_enabled",
                return_value=True,
            ),
            patch(
                "deepagents_code.update_check.is_auto_update_enabled",
                return_value=True,
            ),
            patch(
                "deepagents_code.update_check.is_installed_version_at_least",
                return_value=True,
            ),
            patch(
                "deepagents_code.update_check.get_cached_update_available",
                return_value=(False, "9.9.9"),
            ),
            patch("deepagents_code.update_check.perform_upgrade") as upgrade,
            patch("deepagents_code.main._restart_current_process") as restart,
            patch.object(console, "control", wraps=console.control) as control,
        ):
            _run_startup_auto_update(console)

        upgrade.assert_not_called()
        restart.assert_not_called()
        # Sentinel is consumed so the confirmation only fires once.
        assert os.environ.get("DEEPAGENTS_CODE_RESTARTED_AFTER_UPDATE") is None
        # The prior line is erased via one control call, then reprinted.
        output = stream.getvalue()
        assert control.call_count == 1
        assert "Updated to v9.9.9." in output
        assert "9.9.9" in output

    def test_update_launch_status_rewrite_handles_narrow_terminal_wrap(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The status rewrite erases every row in narrow terminal panes."""
        stream = StringIO()
        console = Console(file=stream, force_terminal=True, no_color=True, width=10)
        narrow_options = console.options.update_width(10)
        monkeypatch.setenv("DEEPAGENTS_CODE_RESTARTED_AFTER_UPDATE", "9.9.9")

        with (
            patch.object(
                type(console),
                "options",
                new_callable=PropertyMock,
                return_value=narrow_options,
            ),
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch(
                "deepagents_code.update_check.is_update_check_enabled",
                return_value=True,
            ),
            patch(
                "deepagents_code.update_check.is_auto_update_enabled",
                return_value=True,
            ),
            patch(
                "deepagents_code.update_check.is_installed_version_at_least",
                return_value=True,
            ),
            patch(
                "deepagents_code.update_check.get_cached_update_available",
                return_value=(False, "9.9.9"),
            ),
            patch("deepagents_code.update_check.perform_upgrade"),
            patch("deepagents_code.main._restart_current_process"),
            patch.object(console, "control", wraps=console.control) as control,
        ):
            launch_rows = _terminal_row_count(
                console, "Updated to v9.9.9. Launching..."
            )
            _run_startup_auto_update(console)

        output = stream.getvalue()
        assert control.call_count == launch_rows
        assert "Updated to" in output
        assert "9.9.9" in output

    def test_restart_after_update_skips_rewrite_when_not_terminal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Redirected (non-terminal) output is not polluted with escape codes."""
        stream = StringIO()
        # `force_terminal=False` makes `is_terminal` report False, exactly as a
        # redirected stream (pipe/file) would. Asserting on the real stream
        # proves no escape bytes reach redirected output, end-to-end.
        console = Console(file=stream, force_terminal=False, no_color=True, width=80)
        monkeypatch.setenv("DEEPAGENTS_CODE_RESTARTED_AFTER_UPDATE", "9.9.9")

        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch(
                "deepagents_code.update_check.is_update_check_enabled",
                return_value=True,
            ),
            patch(
                "deepagents_code.update_check.is_auto_update_enabled",
                return_value=True,
            ),
            patch(
                "deepagents_code.update_check.is_installed_version_at_least",
                return_value=True,
            ),
            patch(
                "deepagents_code.update_check.get_cached_update_available",
                return_value=(False, "9.9.9"),
            ),
            patch("deepagents_code.main._restart_current_process"),
        ):
            _run_startup_auto_update(console)

        output = stream.getvalue()
        assert "\x1b" not in output
        assert "Updated to v9.9.9." not in output

    def test_failed_restart_does_not_confirm_update(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A re-exec that did not change the version must not confirm the update."""
        console = MagicMock()
        console.is_terminal = True
        console.width = 80
        monkeypatch.setenv("DEEPAGENTS_CODE_RESTARTED_AFTER_UPDATE", "9.9.9")

        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch(
                "deepagents_code.update_check.is_update_check_enabled",
                return_value=True,
            ),
            patch(
                "deepagents_code.update_check.is_auto_update_enabled",
                return_value=True,
            ),
            # The install did not change the running version.
            patch(
                "deepagents_code.update_check.is_installed_version_at_least",
                return_value=False,
            ),
            patch(
                "deepagents_code.update_check.get_cached_update_available",
                return_value=(True, "9.9.9"),
            ),
            patch(
                "deepagents_code.update_check.upgrade_command",
                return_value="uv tool upgrade deepagents-code",
            ),
            patch("deepagents_code.main._restart_current_process") as restart,
        ):
            _run_startup_auto_update(console)

        restart.assert_not_called()
        console.control.assert_not_called()
        printed = " ".join(str(c.args[0]) for c in console.print.call_args_list)
        assert "Updated to v9.9.9." not in printed

    def test_version_check_failure_skips_confirm_in_isolation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The confirm must be gated solely by `is_installed_version_at_least`.

        With nothing available (`(False, None)`) the function returns before the
        restart-loop guard, so the only path that could print the stable update
        status is the confirm block. This pins the
        `is_installed_version_at_least(restarted_for)`
        condition: dropping it would let the confirm fire here and fail the test.
        """
        stream = StringIO()
        console = Console(file=stream, force_terminal=True, no_color=True, width=80)
        monkeypatch.setenv("DEEPAGENTS_CODE_RESTARTED_AFTER_UPDATE", "9.9.9")

        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch(
                "deepagents_code.update_check.is_update_check_enabled",
                return_value=True,
            ),
            patch(
                "deepagents_code.update_check.is_auto_update_enabled",
                return_value=True,
            ),
            # The re-exec did not land on the recorded version.
            patch(
                "deepagents_code.update_check.is_installed_version_at_least",
                return_value=False,
            ),
            patch(
                "deepagents_code.update_check.get_cached_update_available",
                return_value=(False, None),
            ),
            patch("deepagents_code.main._restart_current_process") as restart,
        ):
            _run_startup_auto_update(console)

        restart.assert_not_called()
        output = stream.getvalue()
        assert "\x1b[1A" not in output
        assert "Updated to v9.9.9." not in output

    def test_confirm_update_then_continues_to_available_update(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Confirming the prior update must not short-circuit a newer update.

        The sentinel is an older version (now running), while a newer version is
        available: the function should both rewrite the prior line to stable
        update text and proceed into the upgrade path for the newer version.
        """
        stream = StringIO()
        console = Console(file=stream, force_terminal=True, no_color=True, width=80)
        monkeypatch.setenv("DEEPAGENTS_CODE_RESTARTED_AFTER_UPDATE", "9.9.8")
        upgrade = AsyncMock(return_value=(True, ""))

        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch(
                "deepagents_code.update_check.is_update_check_enabled",
                return_value=True,
            ),
            patch(
                "deepagents_code.update_check.is_auto_update_enabled",
                return_value=True,
            ),
            # The running version satisfies the prior restart (9.9.8) but not the
            # newly available 9.9.9, so the upgrade path must still run.
            patch(
                "deepagents_code.update_check.is_installed_version_at_least",
                side_effect=lambda version: version == "9.9.8",
            ),
            patch(
                "deepagents_code.update_check.get_cached_update_available",
                return_value=(True, "9.9.9"),
            ),
            patch(
                "deepagents_code.update_check.format_release_age_parenthetical",
                return_value="",
            ),
            patch(
                "deepagents_code.update_check.create_update_log_path",
                return_value=Path("/tmp/dcode-update.log"),
            ),
            patch("deepagents_code.update_check.perform_upgrade", upgrade),
            patch(
                "deepagents_code.main._restart_current_process",
                side_effect=SystemExit(0),
            ) as restart,
            pytest.raises(SystemExit),
        ):
            _run_startup_auto_update(console)

        upgrade.assert_awaited_once()
        restart.assert_called_once_with()
        output = stream.getvalue()
        # The prior update is confirmed for the running version...
        assert "Updated to v9.9.8." in output
        assert "v9.9.8. Launched." not in output
        # ...and the newer version still goes through the upgrade path.
        assert "v9.9.9. Launching..." in output

    def test_terminal_row_count_single_row(self) -> None:
        """Text that fits on one line counts as a single row."""
        console = Console(file=StringIO(), force_terminal=True, no_color=True, width=80)
        assert _terminal_row_count(console, "abc") == 1

    def test_terminal_row_count_wraps_to_multiple_rows(self) -> None:
        """Text wider than the pane counts each wrapped row.

        Deliberately left unmocked: this is the canary that should fail if a
        future Rich version changes how it wraps text, so its `options` must
        stay real rather than being pinned to a forced width.
        """
        console = Console(file=StringIO(), force_terminal=True, no_color=True, width=10)
        # 20 characters at width 10 wraps to exactly 2 rows.
        assert _terminal_row_count(console, "abcdefghijklmnopqrst") == 2

    def test_terminal_row_count_floors_at_one(self) -> None:
        """Empty text still reports one row, never zero."""
        console = Console(file=StringIO(), force_terminal=True, no_color=True, width=80)
        assert _terminal_row_count(console, "") == 1

    def test_startup_auto_update_wired_into_interactive_launch(self) -> None:
        """`cli_main` must invoke the startup auto-update on interactive launch.

        Without this guard the feature could be dropped from `cli_main` and
        every other unit test would still pass, silently regressing it to a
        no-op.
        """
        source = inspect.getsource(cli_main)
        assert "clear_resume_auto_update_deferral()" in source
        assert "if not should_defer_startup_auto_update_for_resume():" in source
        assert source.count("_run_startup_auto_update(console)") == 2

    def test_project_mcp_prompt_interrupt_aborts_before_tui(self) -> None:
        """Ctrl+C at the project MCP prompt exits before launching Textual."""
        from deepagents_code.main import _ProjectMcpTrustPromptOutcome

        launch = AsyncMock(return_value=AppResult(return_code=0, thread_id="thread"))
        with (
            patch("sys.argv", ["dcode"]),
            patch("sys.stdin", SimpleNamespace(isatty=lambda: True)),
            patch("deepagents_code.main._run_startup_auto_update"),
            patch("deepagents_code.main._resolve_agent_arg", return_value="agent"),
            patch(
                "deepagents_code.main._resolve_interpreter_enabled", return_value=False
            ),
            patch(
                "deepagents_code.main._check_mcp_project_trust",
                return_value=_ProjectMcpTrustPromptOutcome.INTERRUPTED,
            ),
            patch("deepagents_code.main.run_textual_cli_async", launch),
            pytest.raises(SystemExit) as exc_info,
        ):
            cli_main()

        assert exc_info.value.code == 130
        launch.assert_not_called()

    def test_project_mcp_server_selection_cancel_aborts_before_tui(self) -> None:
        """Esc in the server selector cancels launch before Textual starts."""
        from deepagents_code.main import _ProjectMcpTrustPromptOutcome

        launch = AsyncMock(return_value=AppResult(return_code=0, thread_id="thread"))
        with (
            patch("sys.argv", ["dcode"]),
            patch("sys.stdin", SimpleNamespace(isatty=lambda: True)),
            patch("deepagents_code.main._run_startup_auto_update"),
            patch("deepagents_code.main._resolve_agent_arg", return_value="agent"),
            patch(
                "deepagents_code.main._resolve_interpreter_enabled", return_value=False
            ),
            patch(
                "deepagents_code.main._check_mcp_project_trust",
                return_value=_ProjectMcpTrustPromptOutcome.CANCELLED,
            ),
            patch("deepagents_code.main.run_textual_cli_async", launch),
        ):
            cli_main()

        launch.assert_not_called()


class TestAutoUpdateDefaultMigration:
    """First-run consent/migration notice for the auto-update opt-out default."""

    @pytest.fixture(autouse=True)
    def _no_shadowed_dcode(self) -> Iterator[None]:
        """Default to no PATH shadow — same reasoning as `TestStartupAutoUpdate`."""
        with patch(
            "deepagents_code.update_check.detect_shadowed_dcode",
            return_value=None,
        ):
            yield

    def test_first_run_announces_and_skips_install(self) -> None:
        """An implicit (default) opt-in announces once and skips the install."""
        console = MagicMock()
        upgrade = AsyncMock(return_value=(True, "updated"))

        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch(
                "deepagents_code.update_check.is_update_check_enabled",
                return_value=True,
            ),
            patch(
                "deepagents_code.update_check.is_auto_update_enabled",
                return_value=True,
            ),
            patch(
                "deepagents_code.update_check.get_cached_update_available",
                return_value=(True, "9.9.9"),
            ),
            patch(
                "deepagents_code.update_check.should_announce_auto_update_default",
                return_value=True,
            ),
            patch(
                "deepagents_code.update_check.mark_auto_update_default_acknowledged",
                return_value=True,
            ) as mark,
            patch("deepagents_code.update_check.perform_upgrade", upgrade),
            patch("deepagents_code.main._restart_current_process") as restart,
        ):
            _run_startup_auto_update(console)

        upgrade.assert_not_called()
        restart.assert_not_called()
        mark.assert_called_once_with()
        printed = " ".join(str(c.args[0]) for c in console.print.call_args_list)
        assert "updates automatically by default" in printed
        # A successful persist must not warn about the notice repeating.
        assert "could not be saved" not in printed

    def test_first_run_persist_failure_warns_repeat(self) -> None:
        """A failed acknowledgement persist surfaces that the notice may repeat."""
        console = MagicMock()
        upgrade = AsyncMock(return_value=(True, "updated"))

        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch(
                "deepagents_code.update_check.is_update_check_enabled",
                return_value=True,
            ),
            patch(
                "deepagents_code.update_check.is_auto_update_enabled",
                return_value=True,
            ),
            patch(
                "deepagents_code.update_check.get_cached_update_available",
                return_value=(True, "9.9.9"),
            ),
            patch(
                "deepagents_code.update_check.should_announce_auto_update_default",
                return_value=True,
            ),
            patch(
                "deepagents_code.update_check.mark_auto_update_default_acknowledged",
                return_value=False,
            ),
            patch("deepagents_code.update_check.perform_upgrade", upgrade),
            patch("deepagents_code.main._restart_current_process") as restart,
        ):
            _run_startup_auto_update(console)

        upgrade.assert_not_called()
        restart.assert_not_called()
        printed = " ".join(str(c.args[0]) for c in console.print.call_args_list)
        assert "updates automatically by default" in printed
        assert "could not be saved" in printed

    def test_debug_update_does_not_suppress_first_run_notice(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The consent notice wins over the debug-skip branch on the first run.

        `should_announce_auto_update_default` is checked before the
        `DEBUG_UPDATE` short-circuit, so a first run in debug mode shows the
        migration notice (and records the acknowledgement) rather than the
        "Skipped update install (debug mode)" message.
        """
        monkeypatch.setenv("DEEPAGENTS_CODE_DEBUG_UPDATE", "1")
        console = MagicMock()
        upgrade = AsyncMock(return_value=(True, "updated"))

        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch(
                "deepagents_code.update_check.is_update_check_enabled",
                return_value=True,
            ),
            patch(
                "deepagents_code.update_check.is_auto_update_enabled",
                return_value=True,
            ),
            patch(
                "deepagents_code.update_check.get_cached_update_available",
                return_value=(True, "9.9.9"),
            ),
            patch(
                "deepagents_code.update_check.should_announce_auto_update_default",
                return_value=True,
            ),
            patch(
                "deepagents_code.update_check.mark_auto_update_default_acknowledged",
                return_value=True,
            ) as mark,
            patch("deepagents_code.update_check.perform_upgrade", upgrade),
            patch("deepagents_code.main._restart_current_process") as restart,
        ):
            _run_startup_auto_update(console)

        upgrade.assert_not_called()
        restart.assert_not_called()
        mark.assert_called_once_with()
        printed = " ".join(str(c.args[0]) for c in console.print.call_args_list)
        assert "updates automatically by default" in printed
        assert "debug mode" not in printed

    def test_acknowledged_default_proceeds_with_install(self) -> None:
        """Once acknowledged, the install proceeds normally on later launches."""
        console = MagicMock()
        upgrade = AsyncMock(return_value=(True, "updated"))

        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch(
                "deepagents_code.update_check.is_update_check_enabled",
                return_value=True,
            ),
            patch(
                "deepagents_code.update_check.is_auto_update_enabled",
                return_value=True,
            ),
            patch(
                "deepagents_code.update_check.get_cached_update_available",
                return_value=(True, "9.9.9"),
            ),
            patch(
                "deepagents_code.update_check.should_announce_auto_update_default",
                return_value=False,
            ),
            patch(
                "deepagents_code.update_check.format_release_age_parenthetical",
                return_value="",
            ),
            patch(
                "deepagents_code.update_check.create_update_log_path",
                return_value=Path("/tmp/dcode-update.log"),
            ),
            patch("deepagents_code.update_check.perform_upgrade", upgrade),
            patch(
                "deepagents_code.main._restart_current_process",
                side_effect=SystemExit(0),
            ),
            pytest.raises(SystemExit),
        ):
            _run_startup_auto_update(console)

        upgrade.assert_awaited_once()

    def test_first_run_then_next_launch_end_to_end(self, tmp_path: Path) -> None:
        """Drive the real consent state machine across two launches.

        Unlike the other tests here, this does not patch
        `should_announce_auto_update_default` / `mark_auto_update_default_acknowledged`
        — it exercises the genuine implementations against temp config/state
        files so the wiring (announce-and-skip, then proceed) is verified, not
        just the orchestration around stubbed helpers.
        """
        config_path = tmp_path / "config.toml"
        state_file = tmp_path / "update_state.json"
        console = MagicMock()
        upgrade = AsyncMock(return_value=(True, "updated"))

        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch("deepagents_code.update_check.DEFAULT_CONFIG_PATH", config_path),
            patch("deepagents_code.update_check.UPDATE_STATE_FILE", state_file),
            patch(
                "deepagents_code.update_check.get_cached_update_available",
                return_value=(True, "9.9.9"),
            ),
            patch(
                "deepagents_code.update_check.format_release_age_parenthetical",
                return_value="",
            ),
            patch(
                "deepagents_code.update_check.create_update_log_path",
                return_value=Path("/tmp/dcode-update.log"),
            ),
            patch("deepagents_code.update_check.perform_upgrade", upgrade),
        ):
            # First launch: no explicit choice and no recorded acknowledgement,
            # so the migration notice fires and the install is skipped.
            _run_startup_auto_update(console)
            upgrade.assert_not_called()
            printed = " ".join(str(c.args[0]) for c in console.print.call_args_list)
            assert "updates automatically by default" in printed
            assert state_file.exists()  # acknowledgement persisted

            # Second launch: the acknowledgement is now on disk, so the install
            # proceeds and the process re-execs (simulated via SystemExit).
            with (
                patch(
                    "deepagents_code.main._restart_current_process",
                    side_effect=SystemExit(0),
                ),
                pytest.raises(SystemExit),
            ):
                _run_startup_auto_update(console)

        upgrade.assert_awaited_once()


class TestResumeHintLogic:
    """Test that resume hint logic is correct.

    The actual condition in `cli_main` is::

        thread_id and return_code == 0 and asyncio.run(thread_exists(thread_id))

    These tests mirror the three-part condition. `thread_exists` is
    represented as a boolean to keep the tests as pure unit tests.
    """

    def test_resume_hint_condition_error_case(self) -> None:
        """Resume hint should NOT be shown when return_code is non-zero."""
        thread_id = "test123"
        return_code = 1
        has_checkpoints = True

        show = bool(thread_id) and return_code == 0 and has_checkpoints
        assert not show, "Resume hint should not be shown on error"

    def test_resume_hint_condition_success_case(self) -> None:
        """Resume hint SHOULD be shown on success with checkpoints."""
        thread_id = "test123"
        return_code = 0
        has_checkpoints = True

        show = bool(thread_id) and return_code == 0 and has_checkpoints
        assert show, "Resume hint should be shown on success"

    def test_resume_hint_shown_for_resumed_threads(self) -> None:
        """Resume hint SHOULD be shown for resumed threads too."""
        thread_id = "test123"
        return_code = 0
        has_checkpoints = True

        show = bool(thread_id) and return_code == 0 and has_checkpoints
        assert show, "Resume hint should be shown for resumed threads"

    def test_resume_hint_not_shown_without_checkpoints(self) -> None:
        """Resume hint should NOT appear when thread has no checkpoints."""
        thread_id = "test123"
        return_code = 0
        has_checkpoints = False

        show = bool(thread_id) and return_code == 0 and has_checkpoints
        assert not show, "No hint when thread_exists returns False"


class TestTeardownThreadCheckpointLookup:
    """Test teardown checkpoint lookup guard behavior."""

    def test_checks_fresh_thread_without_requests(self) -> None:
        """Fresh interrupted sessions can checkpoint before usage is recorded."""
        should_check = _should_check_teardown_thread(
            "test123",
            request_count=0,
            resume_thread=None,
        )

        assert should_check

    def test_checks_fresh_thread_after_requests(self) -> None:
        """Sessions that made requests may have checkpointed content."""
        should_check = _should_check_teardown_thread(
            "test123",
            request_count=1,
            resume_thread=None,
        )

        assert should_check

    def test_checks_resumed_thread_without_new_requests(self) -> None:
        """Resumed sessions can already have checkpoints before new requests."""
        should_check = _should_check_teardown_thread(
            "test123",
            request_count=0,
            resume_thread="test123",
        )

        assert should_check

    def test_skips_when_no_thread_id(self) -> None:
        """No final thread means there is nothing to look up."""
        should_check = _should_check_teardown_thread(
            None,
            request_count=1,
            resume_thread="test123",
        )

        assert not should_check


class TestRenderTeardownThreadHints:
    """Test the teardown hint renderer shares one `thread_exists` lookup."""

    def _render(
        self,
        *,
        thread_exists_mock: AsyncMock,
        thread_url: str | None,
        return_code: int = 0,
    ) -> str:
        """Render the hints with patched dependencies, returning the output."""
        buffer = StringIO()
        console = Console(file=buffer, width=200)
        with (
            patch("deepagents_code.sessions.thread_exists", thread_exists_mock),
            patch(
                "deepagents_code.config.build_langsmith_thread_url",
                return_value=thread_url,
            ),
        ):
            _render_teardown_thread_hints(console, "test123", return_code=return_code)
        return buffer.getvalue()

    def test_queries_thread_exists_at_most_once(self) -> None:
        """Both hints must share a single checkpoint lookup, never two.

        Guards against a regression that reintroduces a second
        `asyncio.run(thread_exists(...))` (a fresh event loop + aiosqlite
        connection) during teardown.
        """
        thread_exists_mock = AsyncMock(return_value=True)

        output = self._render(thread_exists_mock=thread_exists_mock, thread_url=None)

        thread_exists_mock.assert_awaited_once()
        assert "Resume this thread with:" in output
        assert "dcode -r test123" in output

    def test_prints_langsmith_link_when_available(self) -> None:
        """A configured LangSmith URL is shown alongside the resume hint."""
        thread_exists_mock = AsyncMock(return_value=True)
        url = "https://smith.langchain.com/o/org/projects/p/proj/t/test123"

        output = self._render(thread_exists_mock=thread_exists_mock, thread_url=url)

        assert "View this thread in LangSmith:" in output
        assert "Resume this thread with:" in output
        thread_exists_mock.assert_awaited_once()

    def test_no_hints_without_checkpoints(self) -> None:
        """No checkpoint means no link and no resume hint."""
        thread_exists_mock = AsyncMock(return_value=False)

        output = self._render(thread_exists_mock=thread_exists_mock, thread_url=None)

        assert output == ""
        thread_exists_mock.assert_awaited_once()

    def test_lookup_failure_is_swallowed(self) -> None:
        """A failed checkpoint lookup must not crash teardown or print hints."""
        thread_exists_mock = AsyncMock(side_effect=RuntimeError("db locked"))

        output = self._render(thread_exists_mock=thread_exists_mock, thread_url=None)

        assert output == ""
        thread_exists_mock.assert_awaited_once()

    def test_resume_hint_omitted_on_error_exit(self) -> None:
        """The resume hint is only shown on a clean exit (return_code 0)."""
        thread_exists_mock = AsyncMock(return_value=True)

        output = self._render(
            thread_exists_mock=thread_exists_mock, thread_url=None, return_code=1
        )

        assert "Resume this thread with:" not in output
        thread_exists_mock.assert_awaited_once()


class TestLangSmithTeardownUrl:
    """Test LangSmith thread URL display logic on teardown."""

    def setup_method(self) -> None:
        """Clear LangSmith URL cache before each test."""
        reset_langsmith_url_cache()

    def test_thread_url_requires_all_components(self) -> None:
        """LangSmith link requires thread_id, project_name, and project_url."""
        thread_url = build_langsmith_thread_url("abc123")
        # Without LangSmith configured, should return None
        assert thread_url is None

    def test_thread_url_not_shown_for_none_thread_id(self) -> None:
        """Guard condition: thread_url and thread_exists both needed."""
        thread_url = None
        thread_exists = True
        show_link = bool(thread_url and thread_exists)
        assert not show_link

    def test_thread_url_not_shown_when_no_checkpoints(self) -> None:
        """Guard condition: thread must have checkpointed content."""
        thread_url = "https://smith.langchain.com/o/org/projects/p/proj/t/abc"
        thread_exists = False
        show_link = bool(thread_url and thread_exists)
        assert not show_link

    def test_thread_url_shown_when_all_conditions_met(self) -> None:
        """Guard condition: both thread_url and thread_exists must be truthy."""
        thread_url = "https://smith.langchain.com/o/org/projects/p/proj/t/abc"
        thread_exists = True
        show_link = bool(thread_url and thread_exists)
        assert show_link


class TestAppResult:
    """Tests for the AppResult dataclass."""

    def test_fields_accessible(self) -> None:
        """AppResult should expose return_code and thread_id."""
        result = AppResult(return_code=0, thread_id="tid-abc")
        assert result.return_code == 0
        assert result.thread_id == "tid-abc"

    def test_thread_id_none(self) -> None:
        """AppResult should accept None for thread_id."""
        result = AppResult(return_code=1, thread_id=None)
        assert result.thread_id is None

    def test_frozen(self) -> None:
        """AppResult should be immutable."""
        from dataclasses import FrozenInstanceError

        result = AppResult(return_code=0, thread_id="tid")
        with pytest.raises(FrozenInstanceError):
            result.return_code = 1  # ty: ignore


class TestRunTextualAppReturnType:
    """Test that run_textual_app returns AppResult."""

    async def test_run_textual_app_returns_app_result(self) -> None:
        """run_textual_app should return an AppResult."""
        sig = inspect.signature(run_textual_app)
        annotation = sig.return_annotation
        assert annotation in (AppResult, "AppResult"), (
            f"run_textual_app should return AppResult, got {annotation}"
        )


class TestRunTextualCliAsyncReturnType:
    """Test that run_textual_cli_async returns AppResult."""

    def test_run_textual_cli_async_returns_app_result(self) -> None:
        """run_textual_cli_async should return an AppResult."""
        sig = inspect.signature(run_textual_cli_async)
        assert sig.return_annotation in (AppResult, "AppResult"), (
            "run_textual_cli_async should return AppResult, "
            f"got {sig.return_annotation}"
        )


class TestThreadMessage:
    """Test thread info display format.

    Thread info is now displayed in the WelcomeBanner widget rather than via
    pre-TUI console output, so we verify the banner receives the thread ID.
    """

    def test_thread_id_forwarded_to_app(self) -> None:
        """run_textual_cli_async passes thread_id to run_textual_app."""
        source = inspect.getsource(run_textual_cli_async)
        assert "thread_id=thread_id" in source, (
            "thread_id should be forwarded to run_textual_app"
        )


class TestRunTextualCliAsyncMcp:
    """Tests for MCP/server kwargs forwarding in interactive server mode.

    Server startup and MCP preload now happen inside the TUI via deferred
    kwargs rather than being invoked directly in `run_textual_cli_async`.
    """

    async def test_passes_server_and_mcp_kwargs_to_textual_app(self) -> None:
        """TUI should receive server, mcp_preload, and model kwargs."""
        app_result = AppResult(return_code=0, thread_id="thread-123")
        captured_kwargs: dict[str, Any] = {}

        async def _run_textual_app_stub(**kwargs: Any) -> AppResult:
            captured_kwargs.update(kwargs)
            await asyncio.sleep(0)
            return app_result

        with patch("deepagents_code.app.run_textual_app", new=_run_textual_app_stub):
            result = await run_textual_cli_async(
                "agent",
                thread_id="thread-123",
                model_name="openai:gpt-5.5",
                initial_goal="add refresh tokens",
            )

        assert result == app_result

        # Server kwargs forwarded for deferred startup inside the TUI
        assert captured_kwargs["server_kwargs"] is not None
        assert captured_kwargs["server_kwargs"]["assistant_id"] == "agent"
        assert captured_kwargs["server_kwargs"]["interactive"] is True
        # auto_approve must NOT be in server_kwargs — the interactive server
        # must always compile with full HITL interrupts so Shift+Tab works.
        assert "auto_approve" not in captured_kwargs["server_kwargs"]

        # MCP preload kwargs forwarded (no_mcp=False by default)
        assert captured_kwargs["mcp_preload_kwargs"] is not None
        assert captured_kwargs["mcp_preload_kwargs"]["no_mcp"] is False

        # Model kwargs forwarded for deferred create_model() inside the TUI
        assert captured_kwargs["model_kwargs"] is not None
        assert captured_kwargs["model_kwargs"]["model_spec"] == "openai:gpt-5.5"
        assert captured_kwargs["model_kwargs"]["extra_kwargs"] is None
        assert captured_kwargs["initial_goal"] == "add refresh tokens"

    async def test_no_mcp_kwargs_when_disabled(self) -> None:
        """mcp_preload_kwargs should be None when no_mcp=True."""
        app_result = AppResult(return_code=0, thread_id="thread-123")
        captured_kwargs: dict[str, Any] = {}

        async def _run_textual_app_stub(**kwargs: Any) -> AppResult:
            captured_kwargs.update(kwargs)
            await asyncio.sleep(0)
            return app_result

        with patch("deepagents_code.app.run_textual_app", new=_run_textual_app_stub):
            await run_textual_cli_async(
                "agent",
                thread_id="thread-123",
                model_name="openai:gpt-5.5",
                no_mcp=True,
            )

        assert captured_kwargs["mcp_preload_kwargs"] is None

    async def test_onboarding_trigger_reaches_textual_app(self) -> None:
        """First-run onboarding state should control the app launch flag."""
        app_result = AppResult(return_code=0, thread_id="thread-123")
        captured_kwargs: dict[str, Any] = {}

        async def _run_textual_app_stub(**kwargs: Any) -> AppResult:
            captured_kwargs.update(kwargs)
            await asyncio.sleep(0)
            return app_result

        with (
            patch("deepagents_code.app.run_textual_app", new=_run_textual_app_stub),
            patch(
                "deepagents_code.onboarding.should_run_onboarding", return_value=True
            ),
        ):
            await run_textual_cli_async(
                "agent",
                thread_id="thread-123",
                model_name="openai:gpt-5.5",
            )

        assert captured_kwargs["launch_init"] is True


class TestServerCleanupLifecycle:
    """Verify server_proc.stop() is guaranteed after the TUI exits."""

    async def test_server_proc_stopped_after_app_exits(self) -> None:
        """run_textual_app must call server_proc.stop() in the finally block."""
        server_proc = SimpleNamespace(stop=MagicMock())

        with patch.object(
            DeepAgentsApp,
            "run_async",
            new_callable=AsyncMock,
        ):
            await run_textual_app(server_proc=server_proc, thread_id="t-1")  # ty: ignore

        server_proc.stop.assert_called_once_with()

    async def test_server_proc_stopped_even_on_crash(self) -> None:
        """server_proc.stop() must fire even when run_async raises."""
        server_proc = SimpleNamespace(stop=MagicMock())

        with (
            patch.object(
                DeepAgentsApp,
                "run_async",
                new_callable=AsyncMock,
                side_effect=RuntimeError("boom"),
            ),
            pytest.raises(RuntimeError, match="boom"),
        ):
            await run_textual_app(server_proc=server_proc, thread_id="t-1")  # ty: ignore

        server_proc.stop.assert_called_once_with()

    async def test_deferred_server_proc_stopped_after_app_exits(self) -> None:
        """server_proc set by the background worker must still be cleaned up."""
        server_proc = SimpleNamespace(stop=MagicMock())

        async def _fake_run_async(self: DeepAgentsApp) -> None:  # noqa: RUF029
            # Simulate the background worker having set _server_proc
            self._server_proc = server_proc

        with patch.object(
            DeepAgentsApp,
            "run_async",
            new=_fake_run_async,
        ):
            await run_textual_app(
                server_kwargs={"assistant_id": "a"},
                thread_id="t-1",
            )

        server_proc.stop.assert_called_once_with()


class TestCheckOptionalTools:
    """Tests for check_optional_tools() function."""

    @pytest.fixture(autouse=True)
    def _tavily_available(self) -> Iterator[None]:
        """Patch settings.has_tavily to True so ripgrep-only tests stay isolated."""
        with patch(
            "deepagents_code.config.settings",
            SimpleNamespace(has_tavily=True),
        ):
            yield

    def test_returns_tool_name_when_rg_not_found(self) -> None:
        """Returns `['ripgrep']` when `rg` is not on PATH."""
        with patch("deepagents_code.main.shutil.which", return_value=None):
            missing = check_optional_tools()

        assert missing == ["ripgrep"]

    def test_returns_empty_when_rg_found(self) -> None:
        """Returns empty list when `rg` is found on PATH."""
        with patch("deepagents_code.main.shutil.which", return_value="/usr/bin/rg"):
            missing = check_optional_tools()

        assert missing == []

    def test_managed_rg_still_requires_validation(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Treat the managed binary as missing so `ensure_ripgrep` validates it."""
        managed = tmp_path / "bin" / "rg"
        monkeypatch.setattr(
            "deepagents_code.managed_tools.managed_rg_path",
            lambda: managed,
        )

        with patch("deepagents_code.main.shutil.which", return_value=str(managed)):
            missing = check_optional_tools()

        assert missing == ["ripgrep"]

    def test_warning_suppressed_via_config(self, tmp_path: Path) -> None:
        """Returns empty list when ripgrep warning is suppressed in config."""
        config_path = tmp_path / "config.toml"
        config_path.write_text('[warnings]\nsuppress = ["ripgrep"]\n')

        with patch("deepagents_code.main.shutil.which", return_value=None):
            missing = check_optional_tools(config_path=config_path)

        assert missing == []

    def test_malformed_config_does_not_suppress(self, tmp_path: Path) -> None:
        """Malformed TOML config degrades gracefully instead of crashing."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("this is not valid toml [[[")

        with patch("deepagents_code.main.shutil.which", return_value=None):
            missing = check_optional_tools(config_path=config_path)

        assert missing == ["ripgrep"]

    def test_non_list_suppress_does_not_crash(self, tmp_path: Path) -> None:
        """Non-list `suppress` value degrades gracefully instead of crashing."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("[warnings]\nsuppress = true\n")

        with patch("deepagents_code.main.shutil.which", return_value=None):
            missing = check_optional_tools(config_path=config_path)

        assert missing == ["ripgrep"]

    def test_unrelated_suppress_key_does_not_suppress(self, tmp_path: Path) -> None:
        """Suppressing a different key does not suppress the ripgrep warning."""
        config_path = tmp_path / "config.toml"
        config_path.write_text('[warnings]\nsuppress = ["something_else"]\n')

        with patch("deepagents_code.main.shutil.which", return_value=None):
            missing = check_optional_tools(config_path=config_path)

        assert missing == ["ripgrep"]

    def test_returns_tavily_when_key_missing(self, tmp_path: Path) -> None:
        """Returns `'tavily'` when TAVILY_API_KEY is not set."""
        config_path = tmp_path / "config.toml"
        with (
            patch("deepagents_code.main.shutil.which", return_value="/usr/bin/rg"),
            patch(
                "deepagents_code.config.settings",
                SimpleNamespace(has_tavily=False),
            ),
        ):
            missing = check_optional_tools(config_path=config_path)

        assert missing == ["tavily"]

    def test_omits_tavily_when_key_present(self) -> None:
        """Does not include `'tavily'` when TAVILY_API_KEY is set."""
        with patch("deepagents_code.main.shutil.which", return_value="/usr/bin/rg"):
            missing = check_optional_tools()

        assert "tavily" not in missing

    def test_tavily_warning_suppressed_via_config(self, tmp_path: Path) -> None:
        """Returns empty list when tavily warning is suppressed in config."""
        config_path = tmp_path / "config.toml"
        config_path.write_text('[warnings]\nsuppress = ["tavily"]\n')

        with (
            patch("deepagents_code.main.shutil.which", return_value="/usr/bin/rg"),
            patch(
                "deepagents_code.config.settings",
                SimpleNamespace(has_tavily=False),
            ),
        ):
            missing = check_optional_tools(config_path=config_path)

        assert missing == []


class TestIsManagedRipgrepPath:
    """Tests for `_is_managed_ripgrep_path`."""

    def test_none_is_not_managed(self) -> None:
        """A missing `rg` (path `None`) is not the managed binary."""
        assert _is_managed_ripgrep_path(None) is False

    def test_managed_path_matches(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The resolved managed path is recognized as managed."""
        managed = tmp_path / "bin" / "rg"
        managed.parent.mkdir(parents=True)
        managed.write_bytes(b"x")
        monkeypatch.setattr(
            "deepagents_code.managed_tools.managed_rg_path", lambda: managed
        )

        assert _is_managed_ripgrep_path(str(managed)) is True

    def test_system_path_is_not_managed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A system `rg` elsewhere on `PATH` is not the managed binary."""
        managed = tmp_path / "bin" / "rg"
        monkeypatch.setattr(
            "deepagents_code.managed_tools.managed_rg_path", lambda: managed
        )

        assert _is_managed_ripgrep_path(str(tmp_path / "usr" / "bin" / "rg")) is False


class TestAutoInstallRipgrepCli:
    """Tests for the headless `_auto_install_ripgrep_cli` helper."""

    def test_success_drops_ripgrep_and_prepends(self) -> None:
        """A successful install prepends `PATH` and drops `ripgrep`."""
        console = MagicMock()
        prepend = MagicMock()
        with (
            patch(
                "deepagents_code.managed_tools.ensure_ripgrep",
                AsyncMock(return_value=Path("/managed/rg")),
            ),
            patch(
                "deepagents_code.managed_tools.managed_rg_path",
                return_value=Path("/managed/rg"),
            ),
            patch(
                "deepagents_code.managed_tools.prepend_managed_bin_to_path",
                prepend,
            ),
        ):
            result = _auto_install_ripgrep_cli(console, ["ripgrep", "tavily"])

        assert result == ["tavily"]
        prepend.assert_called_once()

    def test_system_rg_drops_ripgrep_without_prepending(self) -> None:
        """A system `rg` is usable without prepending the managed binary dir."""
        console = MagicMock()
        prepend = MagicMock()
        with (
            patch(
                "deepagents_code.managed_tools.ensure_ripgrep",
                AsyncMock(return_value=Path("/usr/bin/rg")),
            ),
            patch(
                "deepagents_code.managed_tools.managed_rg_path",
                return_value=Path("/managed/rg"),
            ),
            patch(
                "deepagents_code.managed_tools.prepend_managed_bin_to_path",
                prepend,
            ),
        ):
            result = _auto_install_ripgrep_cli(console, ["ripgrep", "tavily"])

        assert result == ["tavily"]
        prepend.assert_not_called()

    def test_install_returns_none_keeps_ripgrep(self) -> None:
        """A skipped/failed install leaves `ripgrep` in the missing list."""
        console = MagicMock()
        prepend = MagicMock()
        with (
            patch(
                "deepagents_code.managed_tools.ensure_ripgrep",
                AsyncMock(return_value=None),
            ),
            patch(
                "deepagents_code.managed_tools.prepend_managed_bin_to_path",
                prepend,
            ),
        ):
            result = _auto_install_ripgrep_cli(console, ["ripgrep"])

        assert result == ["ripgrep"]
        prepend.assert_not_called()

    def test_checksum_mismatch_keeps_ripgrep_and_reports(self) -> None:
        """A checksum mismatch is reported loudly and is not swallowed silently."""
        from deepagents_code.managed_tools import ChecksumMismatchError

        console = MagicMock()
        with patch(
            "deepagents_code.managed_tools.ensure_ripgrep",
            AsyncMock(side_effect=ChecksumMismatchError("bad")),
        ):
            result = _auto_install_ripgrep_cli(console, ["ripgrep"])

        assert result == ["ripgrep"]
        printed = " ".join(str(c.args[0]) for c in console.print.call_args_list)
        assert "SHA-256" in printed

    def test_managed_tool_unavailable_keeps_ripgrep_and_reports(self) -> None:
        """Permanent managed-tool gaps report remediation and keep fallback active."""
        from deepagents_code.managed_tools import ManagedToolUnavailableError

        message = (
            "Managed ripgrep is not available for this system. "
            "Set DEEPAGENTS_CODE_RIPGREP_INSTALLER=system."
        )
        error = ManagedToolUnavailableError(
            tool="ripgrep",
            reason="unsupported",
            message=message,
        )
        console = MagicMock()
        with patch(
            "deepagents_code.managed_tools.ensure_ripgrep",
            AsyncMock(side_effect=error),
        ):
            result = _auto_install_ripgrep_cli(console, ["ripgrep"])

        assert result == ["ripgrep"]
        printed = " ".join(str(c.args[0]) for c in console.print.call_args_list)
        assert f"[yellow]Warning:[/yellow] {message}" in printed

    def test_unexpected_failure_keeps_ripgrep(self) -> None:
        """An unexpected error degrades gracefully to the missing-tool path."""
        console = MagicMock()
        with patch(
            "deepagents_code.managed_tools.ensure_ripgrep",
            AsyncMock(side_effect=RuntimeError("boom")),
        ):
            result = _auto_install_ripgrep_cli(console, ["ripgrep"])

        assert result == ["ripgrep"]


class TestRipgrepInstallHint:
    """Tests for platform-specific ripgrep install hints."""

    def test_macos_brew(self) -> None:
        """Returns brew command on macOS when brew is available."""

        def _which(cmd: str) -> str | None:
            return "/opt/homebrew/bin/brew" if cmd == "brew" else None

        with (
            patch("deepagents_code.main.sys") as mock_sys,
            patch("deepagents_code.main.shutil.which", side_effect=_which),
        ):
            mock_sys.platform = "darwin"
            assert _ripgrep_install_hint() == "brew install ripgrep"

    def test_macos_port(self) -> None:
        """Falls back to MacPorts when brew is absent."""

        def _which(cmd: str) -> str | None:
            return "/opt/local/bin/port" if cmd == "port" else None

        with (
            patch("deepagents_code.main.sys") as mock_sys,
            patch("deepagents_code.main.shutil.which", side_effect=_which),
        ):
            mock_sys.platform = "darwin"
            assert _ripgrep_install_hint() == "sudo port install ripgrep"

    def test_linux_apt(self) -> None:
        """Returns apt-get command on Debian/Ubuntu."""

        def _which(cmd: str) -> str | None:
            return "/usr/bin/apt-get" if cmd == "apt-get" else None

        with (
            patch("deepagents_code.main.sys") as mock_sys,
            patch("deepagents_code.main.shutil.which", side_effect=_which),
        ):
            mock_sys.platform = "linux"
            assert _ripgrep_install_hint() == "sudo apt-get install ripgrep"

    def test_linux_dnf(self) -> None:
        """Returns dnf command on Fedora/RHEL."""

        def _which(cmd: str) -> str | None:
            return "/usr/bin/dnf" if cmd == "dnf" else None

        with (
            patch("deepagents_code.main.sys") as mock_sys,
            patch("deepagents_code.main.shutil.which", side_effect=_which),
        ):
            mock_sys.platform = "linux"
            assert _ripgrep_install_hint() == "sudo dnf install ripgrep"

    def test_linux_pacman(self) -> None:
        """Returns pacman command on Arch."""

        def _which(cmd: str) -> str | None:
            return "/usr/bin/pacman" if cmd == "pacman" else None

        with (
            patch("deepagents_code.main.sys") as mock_sys,
            patch("deepagents_code.main.shutil.which", side_effect=_which),
        ):
            mock_sys.platform = "linux"
            assert _ripgrep_install_hint() == "sudo pacman -S ripgrep"

    def test_linux_zypper(self) -> None:
        """Returns zypper command on openSUSE."""

        def _which(cmd: str) -> str | None:
            return "/usr/bin/zypper" if cmd == "zypper" else None

        with (
            patch("deepagents_code.main.sys") as mock_sys,
            patch("deepagents_code.main.shutil.which", side_effect=_which),
        ):
            mock_sys.platform = "linux"
            assert _ripgrep_install_hint() == "sudo zypper install ripgrep"

    def test_linux_apk(self) -> None:
        """Returns apk command on Alpine."""

        def _which(cmd: str) -> str | None:
            return "/sbin/apk" if cmd == "apk" else None

        with (
            patch("deepagents_code.main.sys") as mock_sys,
            patch("deepagents_code.main.shutil.which", side_effect=_which),
        ):
            mock_sys.platform = "linux"
            assert _ripgrep_install_hint() == "sudo apk add ripgrep"

    def test_linux_nix(self) -> None:
        """Returns nix-env command on NixOS."""

        def _which(cmd: str) -> str | None:
            if cmd == "nix-env":
                return "/nix/var/nix/profiles/default/bin/nix-env"
            return None

        with (
            patch("deepagents_code.main.sys") as mock_sys,
            patch("deepagents_code.main.shutil.which", side_effect=_which),
        ):
            mock_sys.platform = "linux"
            assert _ripgrep_install_hint() == "nix-env -iA nixpkgs.ripgrep"

    def test_win32_choco(self) -> None:
        """Returns choco command on Windows when available."""

        def _which(cmd: str) -> str | None:
            if cmd == "choco":
                return "C:\\ProgramData\\chocolatey\\bin\\choco.exe"
            return None

        with (
            patch("deepagents_code.main.sys") as mock_sys,
            patch("deepagents_code.main.shutil.which", side_effect=_which),
        ):
            mock_sys.platform = "win32"
            assert _ripgrep_install_hint() == "choco install ripgrep"

    def test_win32_scoop(self) -> None:
        """Returns scoop command on Windows when available."""

        def _which(cmd: str) -> str | None:
            if cmd == "scoop":
                return "C:\\Users\\user\\scoop\\shims\\scoop.exe"
            return None

        with (
            patch("deepagents_code.main.sys") as mock_sys,
            patch("deepagents_code.main.shutil.which", side_effect=_which),
        ):
            mock_sys.platform = "win32"
            assert _ripgrep_install_hint() == "scoop install ripgrep"

    def test_win32_winget(self) -> None:
        """Returns winget command on Windows when available."""

        def _which(cmd: str) -> str | None:
            return "C:\\winget.exe" if cmd == "winget" else None

        with (
            patch("deepagents_code.main.sys") as mock_sys,
            patch("deepagents_code.main.shutil.which", side_effect=_which),
        ):
            mock_sys.platform = "win32"
            assert _ripgrep_install_hint() == "winget install BurntSushi.ripgrep"

    def test_darwin_no_manager_falls_through(self) -> None:
        """Falls through to cross-platform on macOS without brew/port."""

        def _which(cmd: str) -> str | None:
            return "/usr/bin/cargo" if cmd == "cargo" else None

        with (
            patch("deepagents_code.main.sys") as mock_sys,
            patch("deepagents_code.main.shutil.which", side_effect=_which),
        ):
            mock_sys.platform = "darwin"
            assert _ripgrep_install_hint() == "cargo install ripgrep"

    def test_linux_no_manager_falls_through(self) -> None:
        """Falls through to URL on Linux without any package manager."""
        with (
            patch("deepagents_code.main.sys") as mock_sys,
            patch("deepagents_code.main.shutil.which", return_value=None),
        ):
            mock_sys.platform = "linux"
            assert "github.com/BurntSushi/ripgrep" in _ripgrep_install_hint()

    def test_cargo_fallback(self) -> None:
        """Falls back to cargo when no system package manager found."""

        def _which(cmd: str) -> str | None:
            return "/usr/bin/cargo" if cmd == "cargo" else None

        with (
            patch("deepagents_code.main.sys") as mock_sys,
            patch("deepagents_code.main.shutil.which", side_effect=_which),
        ):
            mock_sys.platform = "freebsd"
            assert _ripgrep_install_hint() == "cargo install ripgrep"

    def test_conda_fallback(self) -> None:
        """Falls back to conda when no other manager found."""

        def _which(cmd: str) -> str | None:
            return "/usr/bin/conda" if cmd == "conda" else None

        with (
            patch("deepagents_code.main.sys") as mock_sys,
            patch("deepagents_code.main.shutil.which", side_effect=_which),
        ):
            mock_sys.platform = "freebsd"
            assert _ripgrep_install_hint() == "conda install -c conda-forge ripgrep"

    def test_url_fallback(self) -> None:
        """Returns GitHub URL when nothing is detected."""
        with (
            patch("deepagents_code.main.sys") as mock_sys,
            patch("deepagents_code.main.shutil.which", return_value=None),
        ):
            mock_sys.platform = "freebsd"
            hint = _ripgrep_install_hint()
            assert hint.startswith("https://")
            assert "github.com/BurntSushi/ripgrep" in hint


class TestFormatToolWarnings:
    """Tests for the CLI warning formatter and the notification builder."""

    def test_cli_format_contains_install_hint(self) -> None:
        """CLI format includes a platform-specific install hint."""
        hint_patch = patch(
            "deepagents_code.main._ripgrep_install_hint",
            return_value="brew install ripgrep",
        )
        with hint_patch:
            msg = format_tool_warning_cli("ripgrep")
        assert "brew install ripgrep" in msg

    def test_cli_format_wraps_url_in_rich_link(self) -> None:
        """CLI format wraps URL fallback in Rich `[link]` markup."""
        url = "https://github.com/BurntSushi/ripgrep#installation"
        hint_patch = patch(
            "deepagents_code.main._ripgrep_install_hint",
            return_value=url,
        )
        with hint_patch:
            msg = format_tool_warning_cli("ripgrep")
        assert f"[link={url}]" in msg
        assert "[/link]" in msg

    def test_cli_format_contains_config_hint(self) -> None:
        """CLI format references config.toml for suppression."""
        msg = format_tool_warning_cli("ripgrep")
        assert "config.toml" in msg
        assert 'suppress = \\["ripgrep"]' in msg

    def test_cli_format_unknown_tool_fallback(self) -> None:
        """Unknown tools get a generic CLI message."""
        assert format_tool_warning_cli("foo") == "foo is not installed."

    def test_cli_format_tavily_contains_env_hint(self) -> None:
        """CLI format for tavily mentions the env var with Rich link."""
        msg = format_tool_warning_cli("tavily")
        assert "TAVILY_API_KEY" in msg
        assert "[link=https://tavily.com]" in msg

    def test_cli_format_tavily_contains_config_hint(self) -> None:
        """CLI tavily format references config.toml for suppression."""
        msg = format_tool_warning_cli("tavily")
        assert "config.toml" in msg
        assert 'suppress = \\["tavily"]' in msg


class TestBuildMissingToolNotification:
    """Tests for `build_missing_tool_notification` registry factory."""

    def test_ripgrep_with_package_manager_hint(self) -> None:
        """Ripgrep with install command offers copy + open-website + suppress."""
        from deepagents_code.main import _RIPGREP_URL
        from deepagents_code.notifications import ActionId, MissingDepPayload

        with patch(
            "deepagents_code.main._ripgrep_install_hint",
            return_value="brew install ripgrep",
        ):
            entry = build_missing_tool_notification("ripgrep")
        assert entry.key == "dep:ripgrep"
        assert isinstance(entry.payload, MissingDepPayload)
        assert entry.payload.tool == "ripgrep"
        assert entry.payload.install_command == "brew install ripgrep"
        assert entry.payload.url == _RIPGREP_URL
        action_ids = [a.action_id for a in entry.actions]
        assert action_ids == [
            ActionId.COPY_INSTALL,
            ActionId.OPEN_WEBSITE,
            ActionId.SUPPRESS,
        ]
        assert entry.actions[0].primary is True

    def test_ripgrep_url_fallback_opens_website(self) -> None:
        """Ripgrep with URL fallback offers open-website + suppress."""
        from deepagents_code.notifications import ActionId, MissingDepPayload

        url = "https://github.com/BurntSushi/ripgrep#installation"
        with patch(
            "deepagents_code.main._ripgrep_install_hint",
            return_value=url,
        ):
            entry = build_missing_tool_notification("ripgrep")
        assert isinstance(entry.payload, MissingDepPayload)
        assert entry.payload.url == url
        assert entry.payload.install_command is None
        action_ids = [a.action_id for a in entry.actions]
        assert action_ids == [ActionId.OPEN_WEBSITE, ActionId.SUPPRESS]

    def test_tavily_offers_enter_key_website_and_suppress(self) -> None:
        """Tavily entry offers entering a key, the website, and suppression."""
        from deepagents_code.notifications import ActionId, MissingDepPayload

        entry = build_missing_tool_notification("tavily")
        assert entry.key == "dep:tavily"
        assert isinstance(entry.payload, MissingDepPayload)
        assert entry.payload.tool == "tavily"
        assert entry.payload.url == "https://tavily.com"
        assert entry.payload.install_command is None
        action_ids = [a.action_id for a in entry.actions]
        assert action_ids == [
            ActionId.ENTER_API_KEY,
            ActionId.OPEN_WEBSITE,
            ActionId.SUPPRESS,
        ]
        assert entry.actions[0].primary is True
        assert "Tavily API key" in entry.body

    def test_unknown_tool_only_suppresses_and_logs(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Unknown tools fall back to a bare suppress action and log a warning."""
        import logging

        from deepagents_code.notifications import ActionId, MissingDepPayload

        with caplog.at_level(logging.WARNING, logger="deepagents_code.main"):
            entry = build_missing_tool_notification("foo")
        assert entry.key == "dep:foo"
        assert isinstance(entry.payload, MissingDepPayload)
        assert entry.payload.tool == "foo"
        assert [a.action_id for a in entry.actions] == [ActionId.SUPPRESS]
        assert any("No install hint" in record.message for record in caplog.records)


class TestRunTextualCliAsyncModelConfigError:
    """Verify default model config errors are handled before launching the TUI."""

    async def test_launches_tui_on_no_credentials(self) -> None:
        """Missing default credentials should be recoverable inside the TUI."""
        from deepagents_code.model_config import NoCredentialsConfiguredError

        app_result = AppResult(return_code=0, thread_id="t-1")
        captured_kwargs: dict[str, Any] = {}

        async def _stub(**kwargs: Any) -> AppResult:
            captured_kwargs.update(kwargs)
            await asyncio.sleep(0)
            return app_result

        with (
            patch(
                "deepagents_code.config._get_default_model_spec",
                side_effect=NoCredentialsConfiguredError("No credentials configured"),
            ),
            patch("deepagents_code.app.run_textual_app", new=_stub),
        ):
            result = await run_textual_cli_async("agent")

        assert result == app_result
        assert captured_kwargs["defer_server_start"] is True
        assert captured_kwargs["model_kwargs"] is None
        assert captured_kwargs["server_kwargs"]["model_name"] is None

    async def test_recovery_does_not_rely_on_message_text(self) -> None:
        """`NoCredentialsConfiguredError` triggers deferred start.

        Regardless of the exception message text.
        """
        from deepagents_code.model_config import NoCredentialsConfiguredError

        app_result = AppResult(return_code=0, thread_id="t-2")
        captured_kwargs: dict[str, Any] = {}

        async def _stub(**kwargs: Any) -> AppResult:
            captured_kwargs.update(kwargs)
            await asyncio.sleep(0)
            return app_result

        # Reword the message to prove we no longer string-match on prefix.
        with (
            patch(
                "deepagents_code.config._get_default_model_spec",
                side_effect=NoCredentialsConfiguredError(
                    "Setup required: please run /model"
                ),
            ),
            patch("deepagents_code.app.run_textual_app", new=_stub),
        ):
            result = await run_textual_cli_async("agent")

        assert result == app_result
        assert captured_kwargs["defer_server_start"] is True

    async def test_returns_error_code_on_other_model_config_error(self) -> None:
        """Non-recoverable default model errors should still block startup."""
        from deepagents_code.model_config import ModelConfigError

        with (
            patch(
                "deepagents_code.config._get_default_model_spec",
                side_effect=ModelConfigError("Invalid model config"),
            ),
            patch("deepagents_code.config._get_console") as mock_console_fn,
        ):
            mock_console = MagicMock()
            mock_console_fn.return_value = mock_console

            result = await run_textual_cli_async("agent")

        assert result.return_code == 1
        assert result.thread_id is None

    async def test_no_error_when_model_name_provided(self) -> None:
        """Explicit model_name bypasses _get_default_model_spec."""
        app_result = AppResult(return_code=0, thread_id="t-1")

        async def _stub(**_kwargs: Any) -> AppResult:  # noqa: RUF029  # must be async for run_textual_app signature
            return app_result

        with patch("deepagents_code.app.run_textual_app", new=_stub):
            result = await run_textual_cli_async("agent", model_name="openai:gpt-5.5")

        assert result.return_code == 0


class TestNormalizeCwdFilter:
    """Tests for `_normalize_cwd_filter`."""

    def test_none_returns_none(self) -> None:
        """No flag → no filter."""
        from deepagents_code.main import _normalize_cwd_filter

        assert _normalize_cwd_filter(None) is None

    def test_empty_string_uses_current_cwd(self) -> None:
        """Bare `--cwd` (empty-string sentinel) resolves to current working dir."""
        from deepagents_code.main import _normalize_cwd_filter

        assert _normalize_cwd_filter("") == str(Path.cwd())

    def test_explicit_path_is_made_absolute(self) -> None:
        """A user-supplied path is expanduser'd and made absolute."""
        from deepagents_code.main import _normalize_cwd_filter

        result = _normalize_cwd_filter("~/foo/bar")
        assert result is not None
        assert result == str(Path("~/foo/bar").expanduser().absolute())
        assert Path(result).is_absolute()

    def test_explicit_relative_parent_path_is_normalized(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Explicit relative paths collapse `..` without resolving symlinks."""
        from deepagents_code.main import _normalize_cwd_filter

        project = tmp_path / "project"
        subdir = project / "subdir"
        subdir.mkdir(parents=True)
        monkeypatch.chdir(subdir)

        assert _normalize_cwd_filter("..") == str(project)

    def test_explicit_path_does_not_resolve_symlinks(self, tmp_path: Path) -> None:
        """Lexical normalization (not `.resolve()`) matches storage convention."""
        from deepagents_code.main import _normalize_cwd_filter

        real = tmp_path / "real"
        real.mkdir()
        link = tmp_path / "via_link"
        try:
            link.symlink_to(real)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks unsupported on this platform")

        result = _normalize_cwd_filter(str(link))
        assert result == str(link.absolute())
        # Sanity: resolve() would have collapsed the symlink to `real`.
        assert result != str(link.resolve())

    def test_cwd_unreadable_returns_none(self) -> None:
        """A deleted/unreadable cwd degrades to no filter rather than crashing."""
        from deepagents_code.main import _normalize_cwd_filter

        with patch(
            "deepagents_code.main.Path.cwd",
            side_effect=FileNotFoundError("gone"),
        ):
            assert _normalize_cwd_filter("") is None


class TestThreadsListCwdArgparse:
    """Tests for `--cwd` argparse semantics on `deepagents threads list`."""

    def _parse(self, argv: list[str]) -> Any:  # noqa: ANN401
        from deepagents_code.main import parse_args

        with patch("sys.argv", ["deepagents", *argv]):
            return parse_args()

    def test_cwd_omitted_yields_none(self) -> None:
        """Omitting --cwd leaves the namespace value at `None`."""
        ns = self._parse(["threads", "list"])
        assert getattr(ns, "cwd", "MISSING") is None

    def test_cwd_alone_yields_empty_string_const(self) -> None:
        """Bare `--cwd` stores the `const=""` sentinel for downstream resolution."""
        ns = self._parse(["threads", "list", "--cwd"])
        assert ns.cwd == ""

    def test_cwd_with_value_stores_value(self) -> None:
        """`--cwd /some/path` stores the literal value as-is."""
        ns = self._parse(["threads", "list", "--cwd", "/some/path"])
        assert ns.cwd == "/some/path"


class TestCheckMcpProjectTrustPrompt:
    """The project MCP approval prompt should surface a docs link."""

    def test_debug_env_helper_uses_truthy_parsing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The debug helper treats common falsy strings as disabled."""
        from deepagents_code._env_vars import DEBUG_MCP_PROJECT_TRUST
        from deepagents_code.main import _debug_mcp_project_trust_enabled

        monkeypatch.setenv(DEBUG_MCP_PROJECT_TRUST, "0")

        assert _debug_mcp_project_trust_enabled() is False

        monkeypatch.setenv(DEBUG_MCP_PROJECT_TRUST, "1")

        assert _debug_mcp_project_trust_enabled() is True

    def test_debug_env_forces_prompt_without_project_config(
        self,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The debug env var shows a sample prompt without requiring config files."""
        from deepagents_code._env_vars import DEBUG_MCP_PROJECT_TRUST
        from deepagents_code.main import _check_mcp_project_trust

        project_context = SimpleNamespace(project_root=tmp_path, user_cwd=tmp_path)
        monkeypatch.setenv(DEBUG_MCP_PROJECT_TRUST, "1")

        with (
            patch(
                "deepagents_code.project_utils.ProjectContext.from_user_cwd",
                return_value=project_context,
            ),
            patch(
                "deepagents_code.mcp_tools.discover_mcp_configs",
                return_value=[],
            ),
            patch(
                "deepagents_code.mcp_tools.classify_discovered_configs",
                return_value=([], []),
            ),
            patch("builtins.input", return_value="y"),
        ):
            decision = _check_mcp_project_trust(trust_flag=False)

        assert decision is True
        captured = capsys.readouterr()
        assert "debug-project-mcp" in captured.err

    def test_prompt_is_concise(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        """The prompt keeps required server details without extra explanation."""
        from deepagents_code.main import _check_mcp_project_trust

        project_root = tmp_path / "proj"
        project_root.mkdir()
        project_cfg = project_root / ".mcp.json"
        project_cfg.write_text("{}")

        project_context = SimpleNamespace(
            project_root=project_root, user_cwd=project_root
        )
        with (
            patch(
                "deepagents_code.project_utils.ProjectContext.from_user_cwd",
                return_value=project_context,
            ),
            patch(
                "deepagents_code.mcp_tools.discover_mcp_configs",
                return_value=[project_cfg],
            ),
            patch(
                "deepagents_code.mcp_tools.classify_discovered_configs",
                return_value=([], [project_cfg]),
            ),
            patch(
                "deepagents_code.mcp_tools.load_merged_mcp_configs_lenient",
                return_value={
                    "mcpServers": {"fs": {"command": "node", "args": ["server.js"]}}
                },
            ),
            patch(
                "deepagents_code.mcp_tools.extract_project_server_summaries",
                return_value=[("fs", "stdio", "node server.js")],
            ),
            patch("builtins.input", return_value="n"),
        ):
            decision = _check_mcp_project_trust(trust_flag=False)

        assert decision is False
        captured = capsys.readouterr()
        assert "Approve project MCP servers:" in captured.err
        assert '"fs" (stdio):  node server.js' in captured.err
        assert "Learn more:" not in captured.err
        assert "Remembered approvals apply" not in captured.err

    def test_yes_allows_for_session_only(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        """Answering "y" allows for the session without persisting anything."""
        from deepagents_code.main import _check_mcp_project_trust

        project_root = tmp_path / "proj"
        project_root.mkdir()
        project_cfg = project_root / ".mcp.json"
        project_cfg.write_text("{}")

        project_context = SimpleNamespace(
            project_root=project_root, user_cwd=project_root
        )
        with (
            patch(
                "deepagents_code.project_utils.ProjectContext.from_user_cwd",
                return_value=project_context,
            ),
            patch(
                "deepagents_code.mcp_tools.discover_mcp_configs",
                return_value=[project_cfg],
            ),
            patch(
                "deepagents_code.mcp_tools.classify_discovered_configs",
                return_value=([], [project_cfg]),
            ),
            patch(
                "deepagents_code.mcp_tools.load_merged_mcp_configs_lenient",
                return_value={
                    "mcpServers": {"fs": {"command": "node", "args": ["server.js"]}}
                },
            ),
            patch(
                "deepagents_code.mcp_tools.extract_project_server_summaries",
                return_value=[("fs", "stdio", "node server.js")],
            ),
            patch("builtins.input", return_value="y"),
        ):
            decision = _check_mcp_project_trust(trust_flag=False)

        assert decision is True
        err = capsys.readouterr().err.replace("\n", "")
        assert "could not be saved" not in err
        assert "Allowing 1 project MCP server for this session; remembering 0." in err

    def test_prompt_includes_valid_sibling_of_disabled_malformed_server(
        self,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Session trust never activates a valid server omitted from the prompt."""
        from deepagents_code import model_config
        from deepagents_code.main import _check_mcp_project_trust

        project_root = tmp_path / "proj"
        project_dir = project_root / ".deepagents"
        project_dir.mkdir(parents=True)
        lower_cfg = project_dir / ".mcp.json"
        lower_cfg.write_text(
            '{"mcpServers":{"hidden":{"command":"echo"},"broken":{"args":[]}}}'
        )
        higher_cfg = project_root / ".mcp.json"
        higher_cfg.write_text('{"mcpServers":{"visible":{"command":"echo"}}}')
        user_config = tmp_path / "config.toml"
        user_config.write_text('[mcp]\ndisabled_project_servers = ["broken"]\n')
        monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user_config)
        project_context = SimpleNamespace(
            project_root=project_root, user_cwd=project_root
        )

        with (
            patch(
                "deepagents_code.project_utils.ProjectContext.from_user_cwd",
                return_value=project_context,
            ),
            patch(
                "deepagents_code.mcp_tools.discover_mcp_configs",
                return_value=[lower_cfg, higher_cfg],
            ),
            patch(
                "deepagents_code.mcp_tools.classify_discovered_configs",
                return_value=([], [lower_cfg, higher_cfg]),
            ),
            patch("builtins.input", return_value="y"),
        ):
            decision = _check_mcp_project_trust(trust_flag=False)

        assert decision is True
        err = capsys.readouterr().err
        assert '"hidden"' in err
        assert '"visible"' in err
        assert '"broken"' not in err

    def test_prompt_uses_merged_project_config_before_validation(
        self,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The prompt includes every server that merged runtime trust can load."""
        from deepagents_code import model_config
        from deepagents_code.main import _check_mcp_project_trust

        project_root = tmp_path / "proj"
        project_dir = project_root / ".deepagents"
        project_dir.mkdir(parents=True)
        lower = project_dir / ".mcp.json"
        lower.write_text(
            '{"mcpServers":{"hidden":{"command":"echo","args":["lower"]},'
            '"repaired":{"args":[]}}}'
        )
        higher = project_root / ".mcp.json"
        higher.write_text(
            '{"mcpServers":{"repaired":{"command":"echo","args":["higher"]}}}'
        )
        user_config = tmp_path / "config.toml"
        monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user_config)
        project_context = SimpleNamespace(
            project_root=project_root, user_cwd=project_root
        )

        with (
            patch(
                "deepagents_code.project_utils.ProjectContext.from_user_cwd",
                return_value=project_context,
            ),
            patch(
                "deepagents_code.mcp_tools.discover_mcp_configs",
                return_value=[lower, higher],
            ),
            patch(
                "deepagents_code.mcp_tools.classify_discovered_configs",
                return_value=([], [lower, higher]),
            ),
            patch("builtins.input", return_value="n"),
        ):
            decision = _check_mcp_project_trust(trust_flag=False)

        assert decision is False
        err = capsys.readouterr().err
        assert '"hidden"' in err
        assert "echo lower" in err
        assert '"repaired"' in err
        assert "echo higher" in err

    def test_always_allow_persists_names_to_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Answering "a" persists prompted names to config.toml and allows."""
        from deepagents_code import model_config
        from deepagents_code.main import _check_mcp_project_trust

        project_root = tmp_path / "proj"
        project_root.mkdir()
        project_cfg = project_root / ".mcp.json"
        project_cfg.write_text("{}")

        user_config = tmp_path / "config.toml"
        monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user_config)

        project_context = SimpleNamespace(
            project_root=project_root, user_cwd=project_root
        )
        server_configs = {
            "docs": {"command": "echo"},
            "reference": {"command": "echo"},
        }

        with (
            patch(
                "deepagents_code.project_utils.ProjectContext.from_user_cwd",
                return_value=project_context,
            ),
            patch(
                "deepagents_code.mcp_tools.discover_mcp_configs",
                return_value=[project_cfg],
            ),
            patch(
                "deepagents_code.mcp_tools.classify_discovered_configs",
                return_value=([], [project_cfg]),
            ),
            patch(
                "deepagents_code.mcp_tools.load_merged_mcp_configs_lenient",
                return_value={"mcpServers": server_configs},
            ),
            patch(
                "deepagents_code.mcp_tools.extract_project_server_summaries",
                return_value=[
                    ("docs", "stdio", "echo docs"),
                    ("reference", "stdio", "echo reference"),
                ],
            ),
            patch("builtins.input", return_value="a"),
            patch(
                "deepagents_code.main._run_project_mcp_server_checkbox_picker",
                return_value=["docs", "reference"],
            ),
        ):
            decision = _check_mcp_project_trust(trust_flag=False)

        assert decision is True
        lists = model_config.load_mcp_server_trust_lists(user_config)
        assert lists.enabled == frozenset()
        assert lists.is_enabled(
            "docs", project_root=project_root, server=server_configs["docs"]
        )
        assert lists.is_enabled(
            "reference",
            project_root=project_root,
            server=server_configs["reference"],
        )

    def test_always_allow_warns_when_save_fails(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        """A failed remember write still allows the session but warns."""
        from deepagents_code.main import _check_mcp_project_trust

        project_root = tmp_path / "proj"
        project_root.mkdir()
        project_cfg = project_root / ".mcp.json"
        project_cfg.write_text("{}")

        project_context = SimpleNamespace(
            project_root=project_root, user_cwd=project_root
        )

        with (
            patch(
                "deepagents_code.project_utils.ProjectContext.from_user_cwd",
                return_value=project_context,
            ),
            patch(
                "deepagents_code.mcp_tools.discover_mcp_configs",
                return_value=[project_cfg],
            ),
            patch(
                "deepagents_code.mcp_tools.classify_discovered_configs",
                return_value=([], [project_cfg]),
            ),
            patch(
                "deepagents_code.mcp_tools.load_merged_mcp_configs_lenient",
                return_value={"mcpServers": {"fs": {"command": "node"}}},
            ),
            patch(
                "deepagents_code.mcp_tools.extract_project_server_summaries",
                return_value=[("fs", "stdio", "node")],
            ),
            patch(
                "deepagents_code.model_config.add_enabled_project_mcp_servers",
                return_value=False,
            ),
            patch("builtins.input", return_value="always"),
        ):
            decision = _check_mcp_project_trust(trust_flag=False)

        assert decision is True
        assert "could not be remembered" in capsys.readouterr().err

    def test_always_allow_checkbox_persists_only_selected(
        self,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Checkbox selection persists only the chosen server, not the full list."""
        from deepagents_code import model_config
        from deepagents_code.main import _check_mcp_project_trust

        project_root = tmp_path / "proj"
        project_root.mkdir()
        project_cfg = project_root / ".mcp.json"
        project_cfg.write_text("{}")

        user_config = tmp_path / "config.toml"
        monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user_config)

        project_context = SimpleNamespace(
            project_root=project_root, user_cwd=project_root
        )
        server_configs = {
            "docs": {"command": "echo"},
            "reference": {"command": "echo"},
        }

        with (
            patch(
                "deepagents_code.project_utils.ProjectContext.from_user_cwd",
                return_value=project_context,
            ),
            patch(
                "deepagents_code.mcp_tools.discover_mcp_configs",
                return_value=[project_cfg],
            ),
            patch(
                "deepagents_code.mcp_tools.classify_discovered_configs",
                return_value=([], [project_cfg]),
            ),
            patch(
                "deepagents_code.mcp_tools.load_merged_mcp_configs_lenient",
                return_value={"mcpServers": server_configs},
            ),
            patch(
                "deepagents_code.mcp_tools.extract_project_server_summaries",
                return_value=[
                    ("docs", "stdio", "echo docs"),
                    ("reference", "stdio", "echo reference"),
                ],
            ),
            # Allow -> always; checkbox-select only the 2nd server.
            patch("builtins.input", return_value="a"),
            patch(
                "deepagents_code.main._run_project_mcp_server_checkbox_picker",
                return_value=["reference"],
            ),
        ):
            decision = _check_mcp_project_trust(trust_flag=False)

        assert decision is True
        lists = model_config.load_mcp_server_trust_lists(user_config)
        assert lists.enabled == frozenset()
        assert not lists.is_enabled(
            "docs", project_root=project_root, server=server_configs["docs"]
        )
        assert lists.is_enabled(
            "reference",
            project_root=project_root,
            server=server_configs["reference"],
        )
        summary = capsys.readouterr().err.replace("\n", "")
        assert (
            "Allowing 2 project MCP servers for this session; remembering 1 for "
            "this project." in summary
        )

    def test_always_allow_empty_checkbox_selection_denies(
        self,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Selecting no servers denies rather than implicitly allowing the session."""
        from deepagents_code import model_config
        from deepagents_code.main import _check_mcp_project_trust

        project_root = tmp_path / "proj"
        project_root.mkdir()
        project_cfg = project_root / ".mcp.json"
        project_cfg.write_text("{}")

        user_config = tmp_path / "config.toml"
        monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user_config)

        project_context = SimpleNamespace(
            project_root=project_root, user_cwd=project_root
        )
        server_configs = {
            "docs": {"command": "echo"},
            "reference": {"command": "echo"},
        }

        with (
            patch(
                "deepagents_code.project_utils.ProjectContext.from_user_cwd",
                return_value=project_context,
            ),
            patch(
                "deepagents_code.mcp_tools.discover_mcp_configs",
                return_value=[project_cfg],
            ),
            patch(
                "deepagents_code.mcp_tools.classify_discovered_configs",
                return_value=([], [project_cfg]),
            ),
            patch(
                "deepagents_code.mcp_tools.load_merged_mcp_configs_lenient",
                return_value={"mcpServers": server_configs},
            ),
            patch(
                "deepagents_code.mcp_tools.extract_project_server_summaries",
                return_value=[
                    ("docs", "stdio", "echo docs"),
                    ("reference", "stdio", "echo reference"),
                ],
            ),
            patch("builtins.input", return_value="a"),
            patch(
                "deepagents_code.main._run_project_mcp_server_checkbox_picker",
                return_value=[],
            ),
        ):
            decision = _check_mcp_project_trust(trust_flag=False)

        assert decision is False
        assert not user_config.exists()
        assert (
            "No servers selected; denied 2 project MCP servers"
            in capsys.readouterr().err
        )

    def test_always_allow_picker_cancel_aborts_launch(
        self,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Backing out of the server picker returns a launch cancellation.

        Regression guard: a cancelled picker must not be read as "remember none,
        allow this session" or as a denial that continues into the TUI.
        """
        from deepagents_code import model_config
        from deepagents_code.main import (
            _check_mcp_project_trust,
            _ProjectMcpTrustPromptOutcome,
        )

        project_root = tmp_path / "proj"
        project_root.mkdir()
        project_cfg = project_root / ".mcp.json"
        project_cfg.write_text("{}")

        user_config = tmp_path / "config.toml"
        monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user_config)

        project_context = SimpleNamespace(
            project_root=project_root, user_cwd=project_root
        )
        server_configs = {
            "docs": {"command": "echo"},
            "reference": {"command": "echo"},
        }

        with (
            patch(
                "deepagents_code.project_utils.ProjectContext.from_user_cwd",
                return_value=project_context,
            ),
            patch(
                "deepagents_code.mcp_tools.discover_mcp_configs",
                return_value=[project_cfg],
            ),
            patch(
                "deepagents_code.mcp_tools.classify_discovered_configs",
                return_value=([], [project_cfg]),
            ),
            patch(
                "deepagents_code.mcp_tools.load_merged_mcp_configs_lenient",
                return_value={"mcpServers": server_configs},
            ),
            patch(
                "deepagents_code.mcp_tools.extract_project_server_summaries",
                return_value=[
                    ("docs", "stdio", "echo docs"),
                    ("reference", "stdio", "echo reference"),
                ],
            ),
            patch("builtins.input", return_value="a"),
            patch(
                "deepagents_code.main._run_project_mcp_server_checkbox_picker",
                return_value=_ProjectMcpTrustPromptOutcome.CANCELLED,
            ),
        ):
            decision = _check_mcp_project_trust(trust_flag=False)

        assert decision is _ProjectMcpTrustPromptOutcome.CANCELLED
        assert not user_config.exists()
        assert "Cancelled" in capsys.readouterr().err

    def test_always_allow_all_excludes_disabled_server(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A disabled server is never offered for — or persisted to — the allowlist."""
        from deepagents_code import model_config
        from deepagents_code.main import _check_mcp_project_trust

        project_root = tmp_path / "proj"
        project_root.mkdir()
        project_cfg = project_root / ".mcp.json"
        project_cfg.write_text("{}")

        # The user's home config already denies "reference".
        user_config = tmp_path / "config.toml"
        user_config.write_text('[mcp]\ndisabled_project_servers = ["reference"]\n')
        monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user_config)

        project_context = SimpleNamespace(
            project_root=project_root, user_cwd=project_root
        )
        server_configs = {
            "docs": {"command": "echo"},
            "reference": {"command": "echo"},
        }

        with (
            patch(
                "deepagents_code.project_utils.ProjectContext.from_user_cwd",
                return_value=project_context,
            ),
            patch(
                "deepagents_code.mcp_tools.discover_mcp_configs",
                return_value=[project_cfg],
            ),
            patch(
                "deepagents_code.mcp_tools.classify_discovered_configs",
                return_value=([], [project_cfg]),
            ),
            patch(
                "deepagents_code.mcp_tools.load_merged_mcp_configs_lenient",
                return_value={"mcpServers": server_configs},
            ),
            patch(
                "deepagents_code.mcp_tools.extract_project_server_summaries",
                return_value=[
                    ("docs", "stdio", "echo docs"),
                    ("reference", "stdio", "echo reference"),
                ],
            ),
            # "reference" is denied, so only "docs" is promptable — the subset
            # menu is skipped and answering "a" persists just that one name.
            patch("builtins.input", return_value="a"),
        ):
            decision = _check_mcp_project_trust(trust_flag=False)

        assert decision is True
        lists = model_config.load_mcp_server_trust_lists(user_config)
        # The denied server is neither offered nor written to the allowlist, and
        # its deny entry survives (reject precedence still holds).
        assert lists.enabled == frozenset()
        assert lists.is_enabled(
            "docs", project_root=project_root, server=server_configs["docs"]
        )
        assert lists.disabled == frozenset({"reference"})

    def test_all_servers_list_resolved_skip_prompt_without_noise(
        self,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """List-resolved server rows skip the approval prompt without extra output."""
        from deepagents_code import model_config
        from deepagents_code.main import _check_mcp_project_trust

        project_root = tmp_path / "proj"
        project_root.mkdir()
        project_cfg = project_root / ".mcp.json"
        project_cfg.write_text("{}")

        user_config = tmp_path / "config.toml"
        monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user_config)
        server_configs = {
            "docs[/green]": {"command": "echo"},
            "blocked[/red]": {"command": "echo"},
        }
        fingerprint = model_config.fingerprint_mcp_server_config(
            server_configs["docs[/green]"]
        )
        user_config.write_text(
            "[mcp]\n"
            "enabled_project_server_approvals = ["
            f'{{ project_root = "{project_root}", name = "docs[/green]", '
            f'fingerprint = "{fingerprint}" }}]\n'
            'disabled_project_servers = ["blocked[/red]"]\n'
        )

        project_context = SimpleNamespace(
            project_root=project_root, user_cwd=project_root
        )

        def _no_input(_prompt: str = "") -> str:
            msg = "prompt must be skipped when all servers are list-resolved"
            raise AssertionError(msg)

        with (
            patch(
                "deepagents_code.project_utils.ProjectContext.from_user_cwd",
                return_value=project_context,
            ),
            patch(
                "deepagents_code.mcp_tools.discover_mcp_configs",
                return_value=[project_cfg],
            ),
            patch(
                "deepagents_code.mcp_tools.classify_discovered_configs",
                return_value=([], [project_cfg]),
            ),
            patch(
                "deepagents_code.mcp_tools.load_merged_mcp_configs_lenient",
                return_value={"mcpServers": server_configs},
            ),
            patch(
                "deepagents_code.mcp_tools.extract_project_server_summaries",
                return_value=[
                    ("docs[/green]", "stdio[/green]", "echo [/green]"),
                    ("blocked[/red]", "http[/red]", "https://x.test/[/red]"),
                ],
            ),
            patch("builtins.input", _no_input),
        ):
            decision = _check_mcp_project_trust(trust_flag=False)

        assert decision is None
        err = capsys.readouterr().err
        assert err == ""

    def test_prompt_asks_only_about_unlisted_and_hides_preapproved(
        self,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Pre-approved servers stay hidden when unlisted servers are prompted."""
        from deepagents_code import model_config
        from deepagents_code.main import _check_mcp_project_trust

        project_root = tmp_path / "proj"
        project_root.mkdir()
        project_cfg = project_root / ".mcp.json"
        project_cfg.write_text("{}")

        user_config = tmp_path / "config.toml"
        monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user_config)
        server_configs = {
            "docs": {"command": "echo"},
            "other": {"command": "echo"},
        }
        assert model_config.add_enabled_project_mcp_servers(
            ["docs"],
            project_root=project_root,
            server_configs=server_configs,
        )

        project_context = SimpleNamespace(
            project_root=project_root, user_cwd=project_root
        )

        with (
            patch(
                "deepagents_code.project_utils.ProjectContext.from_user_cwd",
                return_value=project_context,
            ),
            patch(
                "deepagents_code.mcp_tools.discover_mcp_configs",
                return_value=[project_cfg],
            ),
            patch(
                "deepagents_code.mcp_tools.classify_discovered_configs",
                return_value=([], [project_cfg]),
            ),
            patch(
                "deepagents_code.mcp_tools.load_merged_mcp_configs_lenient",
                return_value={"mcpServers": server_configs},
            ),
            patch(
                "deepagents_code.mcp_tools.extract_project_server_summaries",
                return_value=[
                    ("docs", "stdio", "echo docs"),
                    ("other", "stdio", "echo other"),
                ],
            ),
            patch("builtins.input", return_value="n"),
        ):
            decision = _check_mcp_project_trust(trust_flag=False)

        assert decision is False
        err = capsys.readouterr().err
        # The unlisted server is the one actually being asked about.
        assert '  "other" (stdio):  echo other' in err
        # The pre-approved server is not repeated in the prompt.
        assert '  "docs" (stdio)' not in err
        assert "Resolved by your config" not in err

    def test_ctrl_c_returns_interrupted_outcome(self, tmp_path: Path) -> None:
        """Ctrl+C at the approval prompt cancels launch instead of denying MCP."""
        from deepagents_code.main import (
            _check_mcp_project_trust,
            _ProjectMcpTrustPromptOutcome,
        )

        project_root = tmp_path / "proj"
        project_root.mkdir()
        project_cfg = project_root / ".mcp.json"
        project_cfg.write_text("{}")

        project_context = SimpleNamespace(
            project_root=project_root, user_cwd=project_root
        )
        with (
            patch(
                "deepagents_code.project_utils.ProjectContext.from_user_cwd",
                return_value=project_context,
            ),
            patch(
                "deepagents_code.mcp_tools.discover_mcp_configs",
                return_value=[project_cfg],
            ),
            patch(
                "deepagents_code.mcp_tools.classify_discovered_configs",
                return_value=([], [project_cfg]),
            ),
            patch(
                "deepagents_code.mcp_tools.load_merged_mcp_configs_lenient",
                return_value={"mcpServers": {"docs": {"command": "echo"}}},
            ),
            patch(
                "deepagents_code.mcp_tools.extract_project_server_summaries",
                return_value=[("docs", "stdio", "echo docs")],
            ),
            patch("builtins.input", side_effect=KeyboardInterrupt),
        ):
            decision = _check_mcp_project_trust(trust_flag=False)

        assert decision is _ProjectMcpTrustPromptOutcome.INTERRUPTED

    def test_eof_at_prompt_denies(self, tmp_path: Path) -> None:
        """EOF (closed stdin) at the approval prompt fails safe to deny.

        A non-interactive/piped stdin must not accidentally allow project MCP
        servers: EOF coerces the answer to empty, which is neither yes nor
        always, so the prompt returns False.
        """
        from deepagents_code.main import _check_mcp_project_trust

        project_root = tmp_path / "proj"
        project_root.mkdir()
        project_cfg = project_root / ".mcp.json"
        project_cfg.write_text("{}")

        project_context = SimpleNamespace(
            project_root=project_root, user_cwd=project_root
        )
        with (
            patch(
                "deepagents_code.project_utils.ProjectContext.from_user_cwd",
                return_value=project_context,
            ),
            patch(
                "deepagents_code.mcp_tools.discover_mcp_configs",
                return_value=[project_cfg],
            ),
            patch(
                "deepagents_code.mcp_tools.classify_discovered_configs",
                return_value=([], [project_cfg]),
            ),
            patch(
                "deepagents_code.mcp_tools.load_merged_mcp_configs_lenient",
                return_value={"mcpServers": {"docs": {"command": "echo"}}},
            ),
            patch(
                "deepagents_code.mcp_tools.extract_project_server_summaries",
                return_value=[("docs", "stdio", "echo docs")],
            ),
            patch("builtins.input", side_effect=EOFError),
        ):
            decision = _check_mcp_project_trust(trust_flag=False)

        assert decision is False

    def test_unreadable_policy_fails_closed_without_prompting(
        self,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A corrupt user config.toml makes the prompt fail closed (return False).

        The allow/deny policy could not be read, so the prompt must not ask (and
        possibly persist trust) under an unknown deny list; it warns and denies,
        matching the loader's fail-closed behavior.
        """
        from deepagents_code import model_config
        from deepagents_code.main import _check_mcp_project_trust

        project_root = tmp_path / "proj"
        project_root.mkdir()
        project_cfg = project_root / ".mcp.json"
        project_cfg.write_text("{}")

        user_config = tmp_path / "config.toml"
        user_config.write_text("[[not valid toml")
        monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user_config)

        project_context = SimpleNamespace(
            project_root=project_root, user_cwd=project_root
        )

        def _no_input(_prompt: str = "") -> str:
            msg = "prompt must be skipped when the trust policy is unreadable"
            raise AssertionError(msg)

        with (
            patch(
                "deepagents_code.project_utils.ProjectContext.from_user_cwd",
                return_value=project_context,
            ),
            patch(
                "deepagents_code.mcp_tools.discover_mcp_configs",
                return_value=[project_cfg],
            ),
            patch(
                "deepagents_code.mcp_tools.classify_discovered_configs",
                return_value=([], [project_cfg]),
            ),
            patch(
                "deepagents_code.mcp_tools.load_merged_mcp_configs_lenient",
                return_value={"mcpServers": {"docs": {"command": "echo"}}},
            ),
            patch(
                "deepagents_code.mcp_tools.extract_project_server_summaries",
                return_value=[("docs", "stdio", "echo docs")],
            ),
            patch("builtins.input", _no_input),
        ):
            decision = _check_mcp_project_trust(trust_flag=False)

        assert decision is False
        err = capsys.readouterr().err
        # Rich may wrap the warning across lines; flatten before matching.
        flattened = err.replace("\n", "")
        assert "treating project MCP servers as untrusted" in flattened
        assert "require approval" not in err


class TestSelectProjectServersToPersist:
    """Tests for the "always allow" subset selection helpers."""

    @pytest.fixture
    def _interactive_picker_terminal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Allow picker tests to run independently of pytest's captured streams."""
        monkeypatch.setattr(
            "deepagents_code.main._project_mcp_picker_has_terminal",
            lambda: True,
        )

    @pytest.mark.parametrize(
        ("raw", "count", "expected"),
        [
            ("1,3", 3, [1, 3]),
            ("1 3", 3, [1, 3]),
            ("2, 2, 1", 3, [2, 1]),  # dedupe, preserve order
            ("0,4,x", 3, []),  # out-of-range and non-numeric ignored
            ("", 3, []),
        ],
    )
    def test_parse_server_number_selection(
        self, raw: str, count: int, expected: list[int]
    ) -> None:
        """The fallback index parser drops invalid tokens and keeps order."""
        from deepagents_code.main import _parse_server_number_selection

        assert _parse_server_number_selection(raw, count) == expected

    def test_checkbox_rows_include_every_server(self) -> None:
        """The inline picker renders each prompted server exactly once."""
        from deepagents_code.config import ASCII_GLYPHS
        from deepagents_code.main import _format_project_mcp_checkbox_rows

        servers = [
            ProjectServerSummary(
                "docs-langchain", "http", "https://docs.langchain.com/mcp"
            ),
            ProjectServerSummary(
                "reference-langchain", "http", "https://reference.langchain.com/mcp"
            ),
        ]

        rows = _format_project_mcp_checkbox_rows(
            servers, {"docs-langchain", "reference-langchain"}, 0, ASCII_GLYPHS
        )

        assert len(rows) == 2
        rendered = "".join(text for _style, text in rows)
        assert "docs-langchain" in rendered
        assert "reference-langchain" in rendered

    def test_single_server_skips_picker(self) -> None:
        """A lone prompted server is returned without asking anything."""
        from rich.console import Console

        from deepagents_code.main import _select_project_servers_to_persist

        def _unexpected_picker(*_args: Any, **_kwargs: Any) -> list[str] | None:
            msg = "no picker expected for a single server"
            raise AssertionError(msg)

        with patch(
            "deepagents_code.main._run_project_mcp_server_checkbox_picker",
            _unexpected_picker,
        ):
            names = _select_project_servers_to_persist(
                [ProjectServerSummary("fs", "stdio", "node")],
                Console(stderr=True),
            )

        assert names == ["fs"]

    @pytest.mark.usefixtures("_interactive_picker_terminal")
    def test_action_picker_is_inline_and_defaults_to_deny(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The unified selector is inline and requires navigation to grant trust."""
        from rich.console import Console

        from deepagents_code.main import (
            _ProjectMcpTrustAction,
            _run_project_mcp_trust_action_picker,
        )

        captured: dict[str, Any] = {}

        class _FakeApplication:
            def __class_getitem__(cls, _item: object) -> type["_FakeApplication"]:
                return cls

            def __init__(self, **kwargs: Any) -> None:
                captured.update(kwargs)

            def run(self) -> _ProjectMcpTrustAction:
                bindings = captured["key_bindings"].bindings
                holder: dict[str, _ProjectMcpTrustAction] = {}
                event = SimpleNamespace(
                    app=SimpleNamespace(
                        exit=lambda *, result: holder.update(value=result)
                    )
                )
                confirm = next(
                    binding.handler
                    for binding in bindings
                    if binding.handler.__name__ == "_confirm"
                )
                confirm(event)
                return holder["value"]

        monkeypatch.setattr("prompt_toolkit.Application", _FakeApplication)
        result = _run_project_mcp_trust_action_picker(Console(stderr=True))

        assert result is _ProjectMcpTrustAction.DENY
        assert captured["full_screen"] is False
        rendered = "".join(
            text for _style, text in captured["layout"].container.content.text()
        )
        assert "Allow once" in rendered
        assert "Allow for this project — until changed" in rendered
        assert "Deny" in rendered
        assert "Choose how to continue" not in rendered

    def test_action_picker_falls_back_when_stderr_is_redirected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A hidden selector is not launched when stderr is not a terminal."""
        from rich.console import Console

        from deepagents_code.main import _run_project_mcp_trust_action_picker

        monkeypatch.setattr(
            "deepagents_code.main.sys.stdin", SimpleNamespace(isatty=lambda: True)
        )
        monkeypatch.setattr(
            "deepagents_code.main.sys.stderr", SimpleNamespace(isatty=lambda: False)
        )

        result = _run_project_mcp_trust_action_picker(Console(stderr=True))

        assert result is None

    def test_checkbox_picker_falls_back_when_stderr_is_redirected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The follow-up selector also avoids hidden rendering to stderr."""
        from rich.console import Console

        from deepagents_code.main import _run_project_mcp_server_checkbox_picker

        monkeypatch.setattr(
            "deepagents_code.main.sys.stdin", SimpleNamespace(isatty=lambda: True)
        )
        monkeypatch.setattr(
            "deepagents_code.main.sys.stderr", SimpleNamespace(isatty=lambda: False)
        )

        result = _run_project_mcp_server_checkbox_picker(
            [ProjectServerSummary("docs", "stdio", "echo docs")],
            Console(stderr=True),
        )

        assert result is None

    @pytest.mark.usefixtures("_interactive_picker_terminal")
    def test_inline_checkbox_picker_does_not_use_full_screen(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The checkbox picker stays inline rather than taking over the terminal."""
        from rich.console import Console

        from deepagents_code.main import _run_project_mcp_server_checkbox_picker

        captured: dict[str, Any] = {}

        class _FakeApplication:
            def __class_getitem__(cls, _item: object) -> type["_FakeApplication"]:
                return cls

            def __init__(self, **kwargs: Any) -> None:
                captured.update(kwargs)

            def run(self) -> list[str]:
                return ["reference"]

        monkeypatch.setattr("prompt_toolkit.Application", _FakeApplication)

        servers = [
            ProjectServerSummary("docs", "stdio", "a"),
            ProjectServerSummary("reference", "stdio", "b"),
        ]
        names = _run_project_mcp_server_checkbox_picker(servers, Console(stderr=True))

        assert names == ["reference"]
        assert captured["full_screen"] is False

    @pytest.mark.usefixtures("_interactive_picker_terminal")
    def test_checkbox_picker_navigation_wraps(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Up at the first row and down at the last row wrap around."""
        from rich.console import Console

        from deepagents_code.config import get_glyphs
        from deepagents_code.main import _run_project_mcp_server_checkbox_picker

        cursor = get_glyphs().cursor
        captured: dict[str, Any] = {}

        class _FakeApplication:
            def __class_getitem__(cls, _item: object) -> type["_FakeApplication"]:
                return cls

            def __init__(self, **kwargs: Any) -> None:
                captured.update(kwargs)

            def run(self) -> list[str]:
                bindings = captured["key_bindings"].bindings
                event = SimpleNamespace(
                    app=SimpleNamespace(exit=lambda **_kwargs: None)
                )

                def _handler_for(key_value: str) -> Callable[[Any], None]:
                    return next(
                        binding.handler
                        for binding in bindings
                        if any(
                            getattr(key, "value", key) == key_value
                            for key in binding.keys
                        )
                    )

                up = _handler_for("s-tab")
                down = _handler_for("c-i")
                rows_control = captured["layout"].container.children[1].content

                up(event)
                rendered = "".join(text for _style, text in rows_control.text())
                assert f"{cursor} [ ] reference" in rendered

                down(event)
                rendered = "".join(text for _style, text in rows_control.text())
                assert f"{cursor} [ ] docs" in rendered
                return ["docs", "reference"]

        monkeypatch.setattr("prompt_toolkit.Application", _FakeApplication)

        servers = [
            ProjectServerSummary("docs", "stdio", "a"),
            ProjectServerSummary("reference", "stdio", "b"),
        ]
        names = _run_project_mcp_server_checkbox_picker(servers, Console(stderr=True))

        assert names == ["docs", "reference"]

    @pytest.mark.usefixtures("_interactive_picker_terminal")
    def test_checkbox_picker_space_toggle_derives_selection(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Space checks the cursor row and Enter confirms the explicit selection.

        The picker starts with nothing selected, so confirming after one toggle
        must return only that server rather than presuming the full list.
        """
        from rich.console import Console

        from deepagents_code.main import _run_project_mcp_server_checkbox_picker

        captured: dict[str, Any] = {}

        class _FakeApplication:
            def __class_getitem__(cls, _item: object) -> type["_FakeApplication"]:
                return cls

            def __init__(self, **kwargs: Any) -> None:
                captured.update(kwargs)

            def run(self) -> list[str]:
                bindings = captured["key_bindings"].bindings
                holder: dict[str, list[str]] = {}
                event = SimpleNamespace(
                    app=SimpleNamespace(
                        exit=lambda *, result: holder.update(value=result)
                    )
                )

                def _named(name: str) -> Callable[[Any], None]:
                    return next(
                        binding.handler
                        for binding in bindings
                        if binding.handler.__name__ == name
                    )

                # Cursor starts on row 0 ("docs"); Space explicitly selects it.
                _named("_toggle")(event)
                _named("_confirm")(event)
                return holder["value"]

        monkeypatch.setattr("prompt_toolkit.Application", _FakeApplication)

        servers = [
            ProjectServerSummary("docs", "stdio", "a"),
            ProjectServerSummary("reference", "stdio", "b"),
        ]
        names = _run_project_mcp_server_checkbox_picker(servers, Console(stderr=True))

        assert names == ["docs"]

    @pytest.mark.usefixtures("_interactive_picker_terminal")
    def test_checkbox_picker_select_all_is_explicit(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The dedicated select-all key checks every server from an empty default."""
        from rich.console import Console

        from deepagents_code.main import _run_project_mcp_server_checkbox_picker

        captured: dict[str, Any] = {}

        class _FakeApplication:
            def __class_getitem__(cls, _item: object) -> type["_FakeApplication"]:
                return cls

            def __init__(self, **kwargs: Any) -> None:
                captured.update(kwargs)

            def run(self) -> list[str]:
                bindings = captured["key_bindings"].bindings
                holder: dict[str, list[str]] = {}
                event = SimpleNamespace(
                    app=SimpleNamespace(
                        exit=lambda *, result: holder.update(value=result)
                    )
                )

                def _named(name: str) -> Callable[[Any], None]:
                    return next(
                        binding.handler
                        for binding in bindings
                        if binding.handler.__name__ == name
                    )

                _named("_select_all")(event)
                _named("_confirm")(event)
                return holder["value"]

        monkeypatch.setattr("prompt_toolkit.Application", _FakeApplication)

        servers = [
            ProjectServerSummary("docs", "stdio", "a"),
            ProjectServerSummary("reference", "stdio", "b"),
        ]
        names = _run_project_mcp_server_checkbox_picker(servers, Console(stderr=True))

        assert names == ["docs", "reference"]

    @pytest.mark.usefixtures("_interactive_picker_terminal")
    def test_checkbox_picker_scrolls_a_bounded_viewport(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Long server lists stay bounded while navigation keeps the cursor visible."""
        from rich.console import Console

        from deepagents_code.main import _run_project_mcp_server_checkbox_picker

        captured: dict[str, Any] = {}

        class _FakeApplication:
            def __class_getitem__(cls, _item: object) -> type["_FakeApplication"]:
                return cls

            def __init__(self, **kwargs: Any) -> None:
                captured.update(kwargs)

            def run(self) -> list[str]:
                bindings = captured["key_bindings"].bindings
                event = SimpleNamespace(
                    app=SimpleNamespace(exit=lambda **_kwargs: None)
                )
                down = next(
                    binding.handler
                    for binding in bindings
                    if binding.handler.__name__ == "_down"
                )
                for _ in range(8):
                    down(event)
                rows_control = captured["layout"].container.children[1].content
                rendered = "".join(text for _style, text in rows_control.text())
                assert "item08 (" in rendered
                assert "item00 (" not in rendered
                assert len(rendered.splitlines()) == 8
                return []

        monkeypatch.setattr("prompt_toolkit.Application", _FakeApplication)

        servers = [
            ProjectServerSummary(f"item{index:02}", "stdio", "echo")
            for index in range(12)
        ]
        _run_project_mcp_server_checkbox_picker(servers, Console(stderr=True))

    @pytest.mark.usefixtures("_interactive_picker_terminal")
    def test_checkbox_picker_escape_cancels(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Esc cancels the approval, distinct from confirming an empty selection.

        Both paths deny rather than silently granting session trust, while Ctrl+C
        remains a separate launch interruption.
        """
        from rich.console import Console

        from deepagents_code.main import (
            _ProjectMcpTrustPromptOutcome,
            _run_project_mcp_server_checkbox_picker,
        )

        captured: dict[str, Any] = {}

        class _FakeApplication:
            def __class_getitem__(cls, _item: object) -> type["_FakeApplication"]:
                return cls

            def __init__(self, **kwargs: Any) -> None:
                captured.update(kwargs)

            def run(self) -> list[str]:
                bindings = captured["key_bindings"].bindings
                holder: dict[str, list[str]] = {}
                event = SimpleNamespace(
                    app=SimpleNamespace(
                        exit=lambda *, result: holder.update(value=result)
                    )
                )
                cancel = next(
                    binding.handler
                    for binding in bindings
                    if binding.handler.__name__ == "_cancel"
                )
                cancel(event)
                return holder["value"]

        monkeypatch.setattr("prompt_toolkit.Application", _FakeApplication)

        servers = [
            ProjectServerSummary("docs", "stdio", "a"),
            ProjectServerSummary("reference", "stdio", "b"),
        ]
        result = _run_project_mcp_server_checkbox_picker(servers, Console(stderr=True))

        assert result is _ProjectMcpTrustPromptOutcome.CANCELLED

    @pytest.mark.usefixtures("_interactive_picker_terminal")
    def test_checkbox_picker_ctrl_c_returns_interrupted(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Ctrl+C in the checkbox picker aborts the launch flow."""
        from rich.console import Console

        from deepagents_code.main import (
            _ProjectMcpTrustPromptOutcome,
            _run_project_mcp_server_checkbox_picker,
        )

        captured: dict[str, Any] = {}

        class _FakeApplication:
            def __class_getitem__(cls, _item: object) -> type["_FakeApplication"]:
                return cls

            def __init__(self, **kwargs: Any) -> None:
                captured.update(kwargs)

            def run(self) -> _ProjectMcpTrustPromptOutcome:
                bindings = captured["key_bindings"].bindings
                holder: dict[str, _ProjectMcpTrustPromptOutcome] = {}
                event = SimpleNamespace(
                    app=SimpleNamespace(
                        exit=lambda *, result: holder.update(value=result)
                    )
                )
                interrupt = next(
                    binding.handler
                    for binding in bindings
                    if any(getattr(key, "value", key) == "c-c" for key in binding.keys)
                )
                interrupt(event)
                return holder["value"]

        monkeypatch.setattr("prompt_toolkit.Application", _FakeApplication)

        servers = [
            ProjectServerSummary("docs", "stdio", "a"),
            ProjectServerSummary("reference", "stdio", "b"),
        ]
        result = _run_project_mcp_server_checkbox_picker(servers, Console(stderr=True))

        assert result is _ProjectMcpTrustPromptOutcome.INTERRUPTED

    @pytest.mark.usefixtures("_interactive_picker_terminal")
    def test_checkbox_picker_eof_cancels(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Ctrl+D (EOF) in the checkbox picker cancels rather than confirming."""
        from rich.console import Console

        from deepagents_code.main import (
            _ProjectMcpTrustPromptOutcome,
            _run_project_mcp_server_checkbox_picker,
        )

        class _FakeApplication:
            def __class_getitem__(cls, _item: object) -> type["_FakeApplication"]:
                return cls

            def __init__(self, **_kwargs: Any) -> None:
                pass

            def run(self) -> list[str]:
                raise EOFError

        monkeypatch.setattr("prompt_toolkit.Application", _FakeApplication)

        servers = [
            ProjectServerSummary("docs", "stdio", "a"),
            ProjectServerSummary("reference", "stdio", "b"),
        ]
        result = _run_project_mcp_server_checkbox_picker(servers, Console(stderr=True))

        assert result is _ProjectMcpTrustPromptOutcome.CANCELLED

    @pytest.mark.usefixtures("_interactive_picker_terminal")
    def test_checkbox_picker_falls_back_when_app_run_fails(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A runtime failure launching the checkbox UI falls back to text input."""
        from rich.console import Console

        from deepagents_code.main import _run_project_mcp_server_checkbox_picker

        class _FakeApplication:
            def __class_getitem__(cls, _item: object) -> type["_FakeApplication"]:
                return cls

            def __init__(self, **_kwargs: Any) -> None:
                pass

            def run(self) -> list[str]:
                msg = "no tty"
                raise RuntimeError(msg)

        monkeypatch.setattr("prompt_toolkit.Application", _FakeApplication)

        servers = [
            ProjectServerSummary("docs", "stdio", "a"),
            ProjectServerSummary("reference", "stdio", "b"),
        ]
        result = _run_project_mcp_server_checkbox_picker(servers, Console(stderr=True))

        # None signals the caller to use the number-based text fallback rather
        # than treating the failure as a trust decision.
        assert result is None

    @pytest.mark.usefixtures("_interactive_picker_terminal")
    def test_action_picker_falls_back_when_app_run_fails(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A runtime failure launching the action picker falls back to text input."""
        from rich.console import Console

        from deepagents_code.main import _run_project_mcp_trust_action_picker

        class _FakeApplication:
            def __class_getitem__(cls, _item: object) -> type["_FakeApplication"]:
                return cls

            def __init__(self, **_kwargs: Any) -> None:
                pass

            def run(self) -> object:
                raise OSError

        monkeypatch.setattr("prompt_toolkit.Application", _FakeApplication)

        result = _run_project_mcp_trust_action_picker(Console(stderr=True))

        assert result is None

    def test_checkbox_selection_returns_selected_names(self) -> None:
        """The checkbox picker decides which prompted servers are remembered."""
        from rich.console import Console

        from deepagents_code.main import _select_project_servers_to_persist

        servers = [
            ProjectServerSummary("docs", "stdio", "a"),
            ProjectServerSummary("reference", "stdio", "b"),
        ]
        with patch(
            "deepagents_code.main._run_project_mcp_server_checkbox_picker",
            return_value=["reference"],
        ):
            names = _select_project_servers_to_persist(servers, Console(stderr=True))

        assert names == ["reference"]

    def test_checkbox_empty_selection_returns_empty(self) -> None:
        """Accepting no checked servers persists nothing."""
        from rich.console import Console

        from deepagents_code.main import _select_project_servers_to_persist

        servers = [
            ProjectServerSummary("docs", "stdio", "a"),
            ProjectServerSummary("reference", "stdio", "b"),
        ]
        with patch(
            "deepagents_code.main._run_project_mcp_server_checkbox_picker",
            return_value=[],
        ):
            names = _select_project_servers_to_persist(servers, Console(stderr=True))

        assert names == []

    def test_fallback_numbers_select_subset(self) -> None:
        """If the checkbox picker cannot run, numbers still select a subset."""
        from rich.console import Console

        from deepagents_code.main import _select_project_servers_to_persist

        servers = [
            ProjectServerSummary("docs", "stdio", "a"),
            ProjectServerSummary("reference", "stdio", "b"),
        ]
        with (
            patch(
                "deepagents_code.main._run_project_mcp_server_checkbox_picker",
                return_value=None,
            ),
            patch("builtins.input", return_value="2"),
        ):
            names = _select_project_servers_to_persist(servers, Console(stderr=True))

        assert names == ["reference"]

    def test_fallback_all_returns_every_name(self) -> None:
        """The text fallback can still remember every prompted server."""
        from rich.console import Console

        from deepagents_code.main import _select_project_servers_to_persist

        servers = [
            ProjectServerSummary("docs", "stdio", "a"),
            ProjectServerSummary("reference", "stdio", "b"),
        ]
        with (
            patch(
                "deepagents_code.main._run_project_mcp_server_checkbox_picker",
                return_value=None,
            ),
            patch("builtins.input", return_value="all"),
        ):
            names = _select_project_servers_to_persist(servers, Console(stderr=True))

        assert names == ["docs", "reference"]

    def test_fallback_interrupt_aborts(self) -> None:
        """A KeyboardInterrupt at the fallback prompt aborts the launch flow."""
        from rich.console import Console

        from deepagents_code.main import (
            _ProjectMcpTrustPromptOutcome,
            _select_project_servers_to_persist,
        )

        servers = [
            ProjectServerSummary("docs", "stdio", "a"),
            ProjectServerSummary("reference", "stdio", "b"),
        ]
        with (
            patch(
                "deepagents_code.main._run_project_mcp_server_checkbox_picker",
                return_value=None,
            ),
            patch("builtins.input", side_effect=KeyboardInterrupt),
        ):
            names = _select_project_servers_to_persist(servers, Console(stderr=True))

        assert names is _ProjectMcpTrustPromptOutcome.INTERRUPTED

    def test_fallback_blank_cancels(self) -> None:
        """Blank fallback input cancels (deny) rather than confirming nothing."""
        from rich.console import Console

        from deepagents_code.main import (
            _ProjectMcpTrustPromptOutcome,
            _select_project_servers_to_persist,
        )

        servers = [
            ProjectServerSummary("docs", "stdio", "a"),
            ProjectServerSummary("reference", "stdio", "b"),
        ]
        with (
            patch(
                "deepagents_code.main._run_project_mcp_server_checkbox_picker",
                return_value=None,
            ),
            patch("builtins.input", return_value="   "),
        ):
            names = _select_project_servers_to_persist(servers, Console(stderr=True))

        assert names is _ProjectMcpTrustPromptOutcome.CANCELLED

    def test_fallback_eof_cancels(self) -> None:
        """EOF (Ctrl+D) at the fallback prompt cancels, not an empty confirm."""
        from rich.console import Console

        from deepagents_code.main import (
            _ProjectMcpTrustPromptOutcome,
            _select_project_servers_to_persist,
        )

        servers = [
            ProjectServerSummary("docs", "stdio", "a"),
            ProjectServerSummary("reference", "stdio", "b"),
        ]
        with (
            patch(
                "deepagents_code.main._run_project_mcp_server_checkbox_picker",
                return_value=None,
            ),
            patch("builtins.input", side_effect=EOFError),
        ):
            names = _select_project_servers_to_persist(servers, Console(stderr=True))

        assert names is _ProjectMcpTrustPromptOutcome.CANCELLED


class TestSelectProjectMcpTrustAction:
    """Text-fallback token mapping for the trust action selector.

    The inline arrow picker is covered separately; this pins the letter tokens
    the text fallback accepts — notably that the advertised `r`/`remember` and
    the `a`/`always` alias both map to REMEMBER.
    """

    @pytest.mark.parametrize(
        ("token", "expected_name"),
        [
            ("y", "ALLOW_ONCE"),
            ("yes", "ALLOW_ONCE"),
            ("r", "REMEMBER"),
            ("remember", "REMEMBER"),
            ("a", "REMEMBER"),
            ("always", "REMEMBER"),
            ("n", "DENY"),
            ("", "DENY"),
        ],
    )
    def test_text_fallback_tokens(
        self, token: str, expected_name: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each accepted token maps to the documented action."""
        from rich.console import Console

        from deepagents_code.main import (
            _ProjectMcpTrustAction,
            _select_project_mcp_trust_action,
        )

        # Force the inline picker to defer to the text prompt.
        monkeypatch.setattr(
            "deepagents_code.main._run_project_mcp_trust_action_picker",
            lambda *_args, **_kwargs: None,
        )
        monkeypatch.setattr("builtins.input", lambda _prompt="": token)

        result = _select_project_mcp_trust_action(Console(stderr=True))

        assert result is _ProjectMcpTrustAction[expected_name]


class TestCheckMcpProjectTrustDedupe:
    """Regression tests for the project MCP approval prompt deduplication.

    When the same server name appears in multiple project-level configs
    (e.g. both `.mcp.json` and `.deepagents/.mcp.json`), the approval
    prompt must list it once — not once per file.
    """

    def _write_config(self, path: Path, servers: dict[str, Any]) -> None:
        import json

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"mcpServers": servers}), encoding="utf-8")

    def _deny_project_mcp(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("builtins.input", lambda _prompt="": "n")

    def _captured_prompt(self, capsys: pytest.CaptureFixture[str]) -> str:
        captured = capsys.readouterr()
        return captured.out + captured.err

    def test_duplicate_server_across_configs_listed_once(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A server defined in both project configs appears once in the prompt."""
        from deepagents_code.main import _check_mcp_project_trust

        server = {
            "fs": {
                "command": "uvx",
                "args": ["mcp-server-filesystem", "/tmp"],
            }
        }
        self._write_config(tmp_path / ".mcp.json", server)
        self._write_config(tmp_path / ".deepagents" / ".mcp.json", server)

        self._deny_project_mcp(tmp_path, monkeypatch)

        result = _check_mcp_project_trust(trust_flag=False)

        assert result is False
        combined = self._captured_prompt(capsys)
        assert combined.count('  "fs" (stdio):') == 1, combined

    def test_duplicate_server_across_configs_uses_project_root_definition(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """The higher-precedence project-root config wins for duplicate names."""
        from deepagents_code.main import _check_mcp_project_trust

        self._write_config(
            tmp_path / ".deepagents" / ".mcp.json",
            {"fs": {"command": "npx", "args": ["subdir-server", "/subdir"]}},
        )
        self._write_config(
            tmp_path / ".mcp.json",
            {"fs": {"command": "uvx", "args": ["root-server", "/root"]}},
        )

        self._deny_project_mcp(tmp_path, monkeypatch)

        result = _check_mcp_project_trust(trust_flag=False)

        assert result is False
        combined = self._captured_prompt(capsys)
        assert combined.count('  "fs" (stdio):') == 1, combined
        assert '  "fs" (stdio):  uvx root-server /root' in combined
        assert "subdir-server" not in combined

    def test_duplicate_remote_server_across_configs_listed_once(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Duplicate remote servers are deduped the same way as stdio servers."""
        from deepagents_code.main import _check_mcp_project_trust

        self._write_config(
            tmp_path / ".deepagents" / ".mcp.json",
            {
                "remote": {
                    "type": "http",
                    "url": "https://subdir.example.com/mcp",
                }
            },
        )
        self._write_config(
            tmp_path / ".mcp.json",
            {
                "remote": {
                    "type": "http",
                    "url": "https://root.example.com/mcp",
                }
            },
        )

        self._deny_project_mcp(tmp_path, monkeypatch)

        result = _check_mcp_project_trust(trust_flag=False)

        assert result is False
        combined = self._captured_prompt(capsys)
        assert combined.count('  "remote" (http):') == 1, combined
        assert '  "remote" (http):  https://root.example.com/mcp' in combined
        assert "subdir.example.com" not in combined

    def test_invalid_project_config_does_not_block_valid_config(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Malformed project configs are skipped while valid configs still prompt."""
        from deepagents_code.main import _check_mcp_project_trust

        invalid = tmp_path / ".deepagents" / ".mcp.json"
        invalid.parent.mkdir(parents=True, exist_ok=True)
        invalid.write_text("{not json", encoding="utf-8")
        self._write_config(
            tmp_path / ".mcp.json",
            {"fs": {"command": "uvx", "args": ["root-server", "/root"]}},
        )

        self._deny_project_mcp(tmp_path, monkeypatch)

        result = _check_mcp_project_trust(trust_flag=False)

        assert result is False
        combined = self._captured_prompt(capsys)
        assert combined.count('  "fs" (stdio):') == 1, combined
        assert '  "fs" (stdio):  uvx root-server /root' in combined

    def test_distinct_servers_across_configs_all_listed(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Different servers from different project configs are all shown."""
        from deepagents_code.main import _check_mcp_project_trust

        self._write_config(
            tmp_path / ".mcp.json",
            {"alpha": {"command": "uvx", "args": ["alpha"]}},
        )
        self._write_config(
            tmp_path / ".deepagents" / ".mcp.json",
            {"beta": {"command": "uvx", "args": ["beta"]}},
        )

        self._deny_project_mcp(tmp_path, monkeypatch)

        result = _check_mcp_project_trust(trust_flag=False)

        assert result is False
        combined = self._captured_prompt(capsys)
        assert combined.count('  "alpha" (stdio):') == 1, combined
        assert combined.count('  "beta" (stdio):') == 1, combined
