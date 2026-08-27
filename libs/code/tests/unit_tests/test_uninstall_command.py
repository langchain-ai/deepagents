"""Tests for optional-extra removal from the CLI and TUI."""

from __future__ import annotations

import argparse
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

if TYPE_CHECKING:
    from collections.abc import Iterator

import pytest

from deepagents_code._version import __version__
from deepagents_code.app import DeepAgentsApp
from deepagents_code.client.commands.extras import (
    run_uninstall_command,
    run_uninstall_request,
)
from deepagents_code.tui.widgets.messages import AppMessage
from deepagents_code.update_check import (
    ExtraNotInstalledError,
    ToolRequirementIntrospectionError,
    perform_uninstall_extra,
    uninstall_extra_command,
)


def _write_receipt(
    root: Path,
    extras: tuple[str, ...],
    *,
    python: str | None = None,
    with_packages: tuple[str, ...] = (),
) -> None:
    extra_values = ", ".join(f'"{extra}"' for extra in extras)
    requirement = (
        f'{{ name = "deepagents-code", extras = [{extra_values}], '
        f'specifier = "=={__version__}" }}'
    )
    requirements = [requirement]
    requirements.extend(f'{{ name = "{package}" }}' for package in with_packages)
    python_line = f'python = "{python}"\n' if python else ""
    root.joinpath("uv-receipt.toml").write_text(
        f"[tool]\n{python_line}requirements = [{', '.join(requirements)}]\n",
        encoding="utf-8",
    )


class TestUninstallExtraCommand:
    """Receipt-aware command generation removes only selected extras."""

    def test_removes_target_and_preserves_tool_context(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_receipt(
            tmp_path,
            ("ollama", "nvidia"),
            python="/opt/Python 3.13/bin/python",
            with_packages=("langchain-custom",),
        )
        monkeypatch.setattr(sys, "prefix", str(tmp_path))

        command = uninstall_extra_command("ollama")

        assert command == (
            "uv tool install --reinstall -U --python '/opt/Python 3.13/bin/python' "
            f"'deepagents-code[nvidia]=={__version__}' --with langchain-custom"
        )

    def test_last_extra_reinstalls_plain_pinned_package(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_receipt(tmp_path, ("ollama",))
        monkeypatch.setattr(sys, "prefix", str(tmp_path))

        assert uninstall_extra_command("ollama") == (
            f"uv tool install --reinstall -U deepagents-code=={__version__}"
        )

    def test_prerelease_install_keeps_prerelease_resolution_enabled(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_receipt(tmp_path, ("ollama",))
        monkeypatch.setattr(sys, "prefix", str(tmp_path))
        monkeypatch.setattr("deepagents_code.update_check.__version__", "1.2.0rc1")

        assert uninstall_extra_command("ollama") == (
            "uv tool install --reinstall -U "
            "deepagents-code==1.2.0rc1 --prerelease allow"
        )

    def test_absent_extra_raises_without_building_a_command(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_receipt(tmp_path, ("nvidia",))
        monkeypatch.setattr(sys, "prefix", str(tmp_path))

        with pytest.raises(ExtraNotInstalledError, match="not installed"):
            uninstall_extra_command("ollama")

    def test_extra_names_use_pep_508_normalization(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_receipt(tmp_path, ("ollama", "my-extra"))
        monkeypatch.setattr(sys, "prefix", str(tmp_path))

        command = uninstall_extra_command("MY_EXTRA")

        assert f"deepagents-code[ollama]=={__version__}" in command
        assert "my-extra" not in command

    @pytest.mark.parametrize("extra", ["openai", "anthropic", "google-genai"])
    def test_base_provider_is_not_removable_even_when_receipt_lists_it(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        extra: str,
    ) -> None:
        _write_receipt(tmp_path, (extra, "nvidia"))
        monkeypatch.setattr(sys, "prefix", str(tmp_path))

        with pytest.raises(ExtraNotInstalledError, match="base dependency"):
            uninstall_extra_command(extra)

    def test_invalid_name_is_rejected_before_receipt_read(self) -> None:
        with pytest.raises(ValueError, match="Invalid extra name"):
            uninstall_extra_command("ollama']; touch /tmp/pwned; '")

    @pytest.mark.parametrize(
        "main_requirement",
        [
            '{ name = "deepagents-code", extras = "ollama" }',
            '{ name = "deepagents-code", extras = ["ollama;rm -rf /"] }',
            '{ name = "deepagents-code", extras = [1] }',
        ],
    )
    def test_malformed_receipt_extras_fail_closed(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        main_requirement: str,
    ) -> None:
        tmp_path.joinpath("uv-receipt.toml").write_text(
            f"[tool]\nrequirements = [{main_requirement}]\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(sys, "prefix", str(tmp_path))

        with pytest.raises(ToolRequirementIntrospectionError):
            uninstall_extra_command("ollama")


class TestPerformUninstallExtra:
    """The performer refuses unsafe/no-op paths before spawning uv."""

    async def test_invalid_name_never_spawns(self) -> None:
        with patch(
            "deepagents_code.update_check._run_install_subprocess",
            new_callable=AsyncMock,
        ) as run:
            success, output = await perform_uninstall_extra("bad;name")
        assert success is False
        assert "Invalid extra name" in output
        run.assert_not_awaited()

    @pytest.mark.parametrize(
        ("method", "message"),
        [
            ("unknown", "Editable install detected"),
            ("brew", "Homebrew install detected"),
            ("other", "Unsupported install method detected"),
        ],
    )
    async def test_non_uv_install_is_refused_without_spawning(
        self, method: str, message: str
    ) -> None:
        with (
            patch(
                "deepagents_code.update_check.detect_install_method",
                return_value=method,
            ),
            patch(
                "deepagents_code.update_check._run_install_subprocess",
                new_callable=AsyncMock,
            ) as run,
        ):
            success, output = await perform_uninstall_extra("ollama")
        assert success is False
        assert message in output
        run.assert_not_awaited()

    async def test_absent_extra_never_spawns(self) -> None:
        with (
            patch(
                "deepagents_code.update_check.detect_install_method", return_value="uv"
            ),
            patch(
                "deepagents_code.update_check.shutil.which", return_value="/usr/bin/uv"
            ),
            patch(
                "deepagents_code.update_check.uninstall_extra_command",
                side_effect=ExtraNotInstalledError("Extra 'ollama' is not installed."),
            ),
            patch(
                "deepagents_code.update_check._run_install_subprocess",
                new_callable=AsyncMock,
            ) as run,
        ):
            success, output = await perform_uninstall_extra("ollama")
        assert success is False
        assert "not installed" in output
        run.assert_not_awaited()

    async def test_lock_wraps_command_generation_and_subprocess(self) -> None:
        events: list[str] = []

        def run_subprocess(*_args: object, **_kwargs: object) -> tuple[bool, str]:
            events.append("run")
            return True, "done"

        run = AsyncMock(side_effect=run_subprocess)

        @contextmanager
        def lock() -> Iterator[bool]:
            events.append("acquire")
            try:
                yield True
            finally:
                events.append("release")

        def command(_extra: str) -> str:
            events.append("command")
            return "uv tool install safe-command"

        with (
            patch(
                "deepagents_code.update_check.detect_install_method", return_value="uv"
            ),
            patch(
                "deepagents_code.update_check.shutil.which", return_value="/usr/bin/uv"
            ),
            patch("deepagents_code.update_check.update_install_lock", lock),
            patch(
                "deepagents_code.update_check.uninstall_extra_command",
                side_effect=command,
            ),
            patch("deepagents_code.update_check._run_install_subprocess", run),
        ):
            result = await perform_uninstall_extra("ollama", log_path=Path("/tmp/log"))

        assert result == (True, "done")
        assert events == ["acquire", "command", "run", "release"]
        run.assert_awaited_once_with(
            "uv tool install safe-command", progress=None, log_path=Path("/tmp/log")
        )

    async def test_contended_lock_skips_receipt_read_and_subprocess(self) -> None:
        command = MagicMock()
        run = AsyncMock()
        with (
            patch(
                "deepagents_code.update_check.detect_install_method", return_value="uv"
            ),
            patch(
                "deepagents_code.update_check.shutil.which", return_value="/usr/bin/uv"
            ),
            patch(
                "deepagents_code.update_check.update_install_lock",
                return_value=nullcontext(False),
            ),
            patch("deepagents_code.update_check.uninstall_extra_command", command),
            patch("deepagents_code.update_check._run_install_subprocess", run),
        ):
            success, output = await perform_uninstall_extra("ollama")

        assert success is False
        assert "Another dcode session" in output
        command.assert_not_called()
        run.assert_not_awaited()


class TestUninstallCli:
    """CLI command and compatibility flag share the same request handler."""

    def test_subcommand_dispatches_name(self) -> None:
        args = argparse.Namespace(uninstall_target="ollama")
        with patch(
            "deepagents_code.client.commands.extras.run_uninstall_request",
            return_value=0,
        ) as request:
            assert run_uninstall_command(args) == 0
        request.assert_called_once_with(name="ollama")

    def test_missing_subcommand_name_shows_help(self) -> None:
        args = argparse.Namespace(uninstall_target=None)
        with patch("deepagents_code.ui.show_uninstall_help") as show_help:
            assert run_uninstall_command(args) == 2
        show_help.assert_called_once()

    def test_absent_extra_is_successful_noop(self) -> None:
        console = MagicMock()
        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch("deepagents_code.config.console", console, create=True),
            patch(
                "deepagents_code.update_check.uninstall_extra_command",
                side_effect=ExtraNotInstalledError("Extra 'ollama' is not installed."),
            ),
            patch(
                "deepagents_code.update_check.perform_uninstall_extra",
                new_callable=AsyncMock,
            ) as perform,
        ):
            assert run_uninstall_request(name="ollama") == 0
        perform.assert_not_awaited()
        assert "not installed" in " ".join(
            str(arg) for call in console.print.call_args_list for arg in call.args
        )

    def test_success_reports_restart_guidance(self) -> None:
        console = MagicMock()
        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch("deepagents_code.config.console", console, create=True),
            patch(
                "deepagents_code.update_check.uninstall_extra_command",
                return_value="uv tool install safe-command",
            ),
            patch(
                "deepagents_code.update_check.create_update_log_path",
                return_value=Path("/tmp/uninstall.log"),
            ),
            patch(
                "deepagents_code.update_check.perform_uninstall_extra",
                new_callable=AsyncMock,
                return_value=(True, ""),
            ) as perform,
        ):
            assert run_uninstall_request(name="ollama") == 0
        perform.assert_awaited_once()
        text = " ".join(
            str(arg) for call in console.print.call_args_list for arg in call.args
        )
        assert "Uninstalled extra 'ollama'" in text
        assert "Relaunch" in text

    @pytest.mark.parametrize(
        ("argv", "command", "target"),
        [
            (["dcode", "uninstall", "ollama"], "uninstall", "ollama"),
            (["dcode", "--uninstall", "ollama"], None, "ollama"),
        ],
    )
    def test_parser_accepts_subcommand_and_compatibility_flag(
        self, argv: list[str], command: str | None, target: str
    ) -> None:
        from deepagents_code.main import parse_args

        with patch.object(sys, "argv", argv):
            args = parse_args()
        assert args.command == command
        parsed_target = args.uninstall_target if command else args.uninstall
        assert parsed_target == target

    def test_install_and_uninstall_flags_are_mutually_exclusive(self) -> None:
        from deepagents_code.main import parse_args

        with (
            patch.object(
                sys,
                "argv",
                ["dcode", "--install", "ollama", "--uninstall", "nvidia"],
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            parse_args()
        assert exc_info.value.code == 2

    @pytest.mark.parametrize(
        "argv",
        [
            ["dcode", "uninstall", "ollama"],
            ["dcode", "--uninstall", "ollama"],
        ],
    )
    def test_cli_main_dispatches_both_forms(self, argv: list[str]) -> None:
        from deepagents_code.main import cli_main

        stdin = MagicMock()
        stdin.isatty.return_value = False
        stdin.read.return_value = ""
        with (
            patch.object(sys, "argv", argv),
            patch.object(sys, "stdin", stdin),
            patch("deepagents_code.main.check_cli_dependencies"),
            patch(
                "deepagents_code.client.commands.extras.run_uninstall_request",
                return_value=7,
            ) as request,
            pytest.raises(SystemExit) as exc_info,
        ):
            cli_main()
        assert exc_info.value.code == 7
        request.assert_called_once_with(name="ollama")


async def test_uninstall_slash_usage_does_not_invoke_performer() -> None:
    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        with patch(
            "deepagents_code.update_check.perform_uninstall_extra",
            new_callable=AsyncMock,
        ) as perform:
            await app._handle_command("/uninstall")
            await pilot.pause()
        perform.assert_not_awaited()
        messages = [
            message for message in app.query(AppMessage) if not message._is_markdown
        ]
        assert any("Usage: /uninstall" in str(message._content) for message in messages)


async def test_uninstall_slash_absent_extra_is_noop() -> None:
    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch(
                "deepagents_code.update_check.uninstall_extra_command",
                side_effect=ExtraNotInstalledError("Extra 'ollama' is not installed."),
            ),
            patch(
                "deepagents_code.update_check.perform_uninstall_extra",
                new_callable=AsyncMock,
            ) as perform,
        ):
            await app._handle_command("/uninstall ollama")
            await pilot.pause()
        perform.assert_not_awaited()
        messages = [
            message for message in app.query(AppMessage) if not message._is_markdown
        ]
        assert any("not installed" in str(message._content) for message in messages)


async def test_uninstall_slash_serializes_environment_mutation() -> None:
    app = DeepAgentsApp()
    app._mount_message = AsyncMock()
    with (
        patch("deepagents_code.config._is_editable_install", return_value=False),
        patch(
            "deepagents_code.update_check.uninstall_extra_command",
            return_value="uv tool install safe-command",
        ),
        patch(
            "deepagents_code.update_check.perform_uninstall_extra",
            new_callable=AsyncMock,
            return_value=(True, ""),
        ) as perform,
    ):
        await app._handle_uninstall_command("/uninstall ollama")
    perform.assert_awaited_once()
    text = " ".join(
        str(call.args[0]._content) for call in app._mount_message.await_args_list
    )
    assert "Uninstalled extra 'ollama'" in text
