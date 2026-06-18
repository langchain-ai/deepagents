"""Tests for `/uninstall <extra>` and `--uninstall <extra>` extra removal.

Covers the `uninstall_extra_command` generator, the `cli_main` `--uninstall`
flow, and the in-app `/uninstall` slash dispatch in `DeepAgentsApp`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deepagents_code.app import DeepAgentsApp
from deepagents_code.extras_info import ExtrasIntrospectionError
from deepagents_code.update_check import (
    ExtraNotInstalledError,
    uninstall_extra_command,
)
from deepagents_code.widgets.messages import AppMessage


def _write_dist_info(
    root: Path,
    name: str,
    *,
    version: str = "1.0.0",
    requires: tuple[str, ...] = (),
) -> None:
    normalized = name.replace("-", "_")
    dist_info = root / f"{normalized}-{version}.dist-info"
    dist_info.mkdir()
    metadata = ["Metadata-Version: 2.1", f"Name: {name}", f"Version: {version}"]
    metadata.extend(f"Requires-Dist: {req}" for req in requires)
    dist_info.joinpath("METADATA").write_text("\n".join(metadata), encoding="utf-8")


class TestUninstallExtraCommand:
    """`uninstall_extra_command` rebuilds the tool with the remaining extras."""

    def test_drops_one_keeps_remaining(self, tmp_path, monkeypatch) -> None:
        _write_dist_info(tmp_path, "definitely-present-dcode-test-ollama")
        _write_dist_info(tmp_path, "definitely-present-dcode-test-quickjs")
        _write_dist_info(
            tmp_path,
            "deepagents-code",
            requires=(
                'definitely-present-dcode-test-ollama; extra == "ollama"',
                'definitely-present-dcode-test-quickjs; extra == "quickjs"',
            ),
        )
        monkeypatch.syspath_prepend(str(tmp_path))

        assert (
            uninstall_extra_command("ollama", distribution_name="deepagents-code")
            == "uv tool install -U 'deepagents-code[quickjs]'"
        )

    def test_last_extra_returns_plain_package(self, tmp_path, monkeypatch) -> None:
        _write_dist_info(tmp_path, "definitely-present-dcode-test-ollama")
        _write_dist_info(
            tmp_path,
            "deepagents-code",
            requires=('definitely-present-dcode-test-ollama; extra == "ollama"',),
        )
        monkeypatch.syspath_prepend(str(tmp_path))

        assert (
            uninstall_extra_command("ollama", distribution_name="deepagents-code")
            == "uv tool install -U deepagents-code"
        )

    def test_absent_extra_raises(self, tmp_path, monkeypatch) -> None:
        _write_dist_info(tmp_path, "definitely-present-dcode-test-ollama")
        _write_dist_info(
            tmp_path,
            "deepagents-code",
            requires=('definitely-present-dcode-test-ollama; extra == "ollama"',),
        )
        monkeypatch.syspath_prepend(str(tmp_path))

        with pytest.raises(ExtraNotInstalledError, match="not installed"):
            uninstall_extra_command("quickjs", distribution_name="deepagents-code")

    def test_invalid_name_raises_before_metadata(self) -> None:
        with pytest.raises(ValueError, match="Invalid extra name"):
            uninstall_extra_command(
                "quickjs']; touch /tmp/pwned; '",
                distribution_name="missing-dcode-test",
            )

    def test_missing_distribution_raises(self) -> None:
        with pytest.raises(ExtrasIntrospectionError, match="cannot preserve"):
            uninstall_extra_command(
                "ollama", distribution_name="missing-dcode-test-xyz"
            )


class TestUninstallExtraSubcommand:
    """Control-flow tests for `dcode --uninstall <extra>`."""

    @staticmethod
    def _run(
        extra: str,
        *,
        editable: bool = False,
        installed: tuple[str, ...] = ("ollama",),
        perform_return: tuple[bool, str] = (True, ""),
    ) -> tuple[int, MagicMock, MagicMock]:
        from deepagents_code.main import cli_main

        argv = ["deepagents", "--uninstall", extra]
        console_mock = MagicMock()
        perform_mock = AsyncMock(return_value=perform_return)
        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = False
        mock_stdin.read.return_value = ""
        with (
            patch.object(sys, "argv", argv),
            patch.object(sys, "stdin", mock_stdin),
            patch("deepagents_code.main.check_cli_dependencies"),
            patch("deepagents_code.config.console", console_mock, create=True),
            patch("deepagents_code.config._is_editable_install", return_value=editable),
            patch(
                "deepagents_code.extras_info.installed_extra_names",
                return_value=set(installed),
            ),
            patch(
                "deepagents_code.update_check.create_update_log_path",
                return_value=Path("/tmp/deepagents-uninstall.log"),
            ),
            patch(
                "deepagents_code.update_check.uninstall_extra_command",
                return_value="uv tool install -U deepagents-code",
            ),
            patch(
                "deepagents_code.update_check.perform_uninstall_extra",
                perform_mock,
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            cli_main()
        return int(exc_info.value.code or 0), perform_mock, console_mock

    @staticmethod
    def _text(console_mock: MagicMock) -> str:
        chunks: list[str] = []
        for call in console_mock.print.call_args_list:
            chunks.extend(str(arg) for arg in call.args)
        return "\n".join(chunks)

    def test_installed_extra_runs(self) -> None:
        code, perform, console = self._run("ollama", installed=("ollama",))
        assert code == 0
        perform.assert_awaited_once()
        assert "Uninstalled extra 'ollama'" in self._text(console)

    def test_absent_extra_skips_uv(self) -> None:
        code, perform, console = self._run("quickjs", installed=("ollama",))
        assert code == 0
        perform.assert_not_awaited()
        assert "Extra 'quickjs' is not installed." in self._text(console)

    def test_editable_install_refuses(self) -> None:
        code, perform, _console = self._run("ollama", editable=True)
        assert code == 1
        perform.assert_not_awaited()

    def test_invalid_name_refuses(self) -> None:
        code, perform, _console = self._run("bad;name")
        assert code == 2
        perform.assert_not_awaited()

    def test_failure_surfaces_log_and_manual_command(self) -> None:
        code, _perform, console = self._run(
            "ollama", perform_return=(False, "resolver: conflict")
        )
        assert code == 1
        text = self._text(console)
        assert "Uninstall failed" in text
        assert "/tmp/deepagents-uninstall.log" in text


async def test_uninstall_slash_usage_when_no_extra() -> None:
    """`/uninstall` with no argument prints a usage hint and does not run uv."""
    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        with patch(
            "deepagents_code.update_check.perform_uninstall_extra",
            new_callable=AsyncMock,
        ) as perform_mock:
            await app._handle_command("/uninstall")
            await pilot.pause()
        perform_mock.assert_not_awaited()
        app_msgs = [m for m in app.query(AppMessage) if not m._is_markdown]
        assert any("Usage: /uninstall" in str(m._content) for m in app_msgs)


async def test_uninstall_slash_provider_extra_recommends_restart_slash() -> None:
    """Provider extras advertise `/restart`, not a full relaunch."""
    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch(
                "deepagents_code.extras_info.installed_extra_names",
                return_value={"ollama"},
            ),
            patch(
                "deepagents_code.update_check.uninstall_extra_command",
                return_value="uv tool install -U deepagents-code",
            ),
            patch(
                "deepagents_code.update_check.perform_uninstall_extra",
                new_callable=AsyncMock,
                return_value=(True, ""),
            ) as perform_mock,
        ):
            await app._handle_command("/uninstall ollama")
            await pilot.pause()
        perform_mock.assert_awaited_once()
        app_msgs = [m for m in app.query(AppMessage) if not m._is_markdown]
        success = next(
            m for m in app_msgs if "Uninstalled extra 'ollama'" in str(m._content)
        )
        assert "/restart" in str(success._content)


async def test_uninstall_slash_standalone_extra_recommends_full_relaunch() -> None:
    """Standalone extras (e.g. quickjs) require a full relaunch, not `/restart`."""
    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch(
                "deepagents_code.extras_info.installed_extra_names",
                return_value={"quickjs"},
            ),
            patch(
                "deepagents_code.update_check.uninstall_extra_command",
                return_value="uv tool install -U deepagents-code",
            ),
            patch(
                "deepagents_code.update_check.perform_uninstall_extra",
                new_callable=AsyncMock,
                return_value=(True, ""),
            ),
        ):
            await app._handle_command("/uninstall quickjs")
            await pilot.pause()
        app_msgs = [m for m in app.query(AppMessage) if not m._is_markdown]
        success = next(
            m for m in app_msgs if "Uninstalled extra 'quickjs'" in str(m._content)
        )
        rendered = str(success._content)
        assert "/restart" not in rendered
        assert "relaunch dcode" in rendered


async def test_uninstall_slash_absent_extra_reports_and_skips_uv() -> None:
    """An extra that is not installed is reported without invoking uv."""
    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch(
                "deepagents_code.extras_info.installed_extra_names",
                return_value={"ollama"},
            ),
            patch(
                "deepagents_code.update_check.perform_uninstall_extra",
                new_callable=AsyncMock,
            ) as perform_mock,
        ):
            await app._handle_command("/uninstall quickjs")
            await pilot.pause()
        perform_mock.assert_not_awaited()
        app_msgs = [m for m in app.query(AppMessage) if not m._is_markdown]
        assert any(
            "Extra 'quickjs' is not installed." in str(m._content) for m in app_msgs
        )


async def test_uninstall_slash_editable_install_refuses() -> None:
    """Editable installs refuse and never call the performer."""
    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        with (
            patch("deepagents_code.config._is_editable_install", return_value=True),
            patch(
                "deepagents_code.update_check.perform_uninstall_extra",
                new_callable=AsyncMock,
            ) as perform_mock,
        ):
            await app._handle_command("/uninstall ollama")
            await pilot.pause()
        perform_mock.assert_not_awaited()
        app_msgs = [m for m in app.query(AppMessage) if not m._is_markdown]
        assert any("Editable install detected" in str(m._content) for m in app_msgs)
