"""Tests for the `dcode extras` command group."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deepagents_code.client.commands.extras import (
    run_extras_command,
    run_install_request,
)


class TestExtrasInstallDispatch:
    """Tests for `dcode extras install` dispatch and shared install helpers."""

    def test_no_subcommand_shows_help(self) -> None:
        args = argparse.Namespace(extras_command=None)
        with patch("deepagents_code.ui.show_extras_help") as show_help:
            code = run_extras_command(args)
        assert code == 0
        show_help.assert_called_once()

    def test_install_dispatches_to_shared_helper(self) -> None:
        args = argparse.Namespace(
            extras_command="install",
            extras_target="daytona",
            extras_package=False,
            extras_yes=True,
            package=False,
            yes=False,
        )
        with patch(
            "deepagents_code.client.commands.extras.run_install_request",
            return_value=0,
        ) as install:
            code = run_extras_command(args)
        assert code == 0
        install.assert_called_once_with(name="daytona", package=False, yes=True)

    def test_install_accepts_global_yes_and_package_flags(self) -> None:
        """Root flags preceding the group (e.g. `dcode --yes extras install`) work."""
        args = argparse.Namespace(
            extras_command="install",
            extras_target="langchain-custom",
            extras_package=False,
            extras_yes=False,
            package=True,
            yes=True,
        )
        with patch(
            "deepagents_code.client.commands.extras.run_install_request",
            return_value=0,
        ) as install:
            code = run_extras_command(args)
        assert code == 0
        install.assert_called_once_with(name="langchain-custom", package=True, yes=True)

    def test_known_extra_runs_install(self) -> None:
        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch(
                "deepagents_code.config.console", MagicMock(), create=True
            ) as console_mock,
            patch(
                "deepagents_code.update_check.create_update_log_path",
                return_value=Path("/tmp/deepagents-install.log"),
            ),
            patch(
                "deepagents_code.update_check.install_extra_command",
                return_value="script",
            ),
            patch(
                "deepagents_code.update_check.perform_install_extra",
                new_callable=AsyncMock,
                return_value=(True, ""),
            ) as perform,
        ):
            code = run_install_request(name="quickjs", package=False, yes=True)
        assert code == 0
        perform.assert_awaited_once()
        printed = " ".join(
            str(arg) for call in console_mock.print.call_args_list for arg in call.args
        )
        assert "Installed extra 'quickjs'" in printed

    def test_package_path_runs_package_install(self) -> None:
        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch("deepagents_code.config.console", MagicMock(), create=True),
            patch(
                "deepagents_code.update_check.create_update_log_path",
                return_value=Path("/tmp/deepagents-install.log"),
            ),
            patch(
                "deepagents_code.update_check.perform_install_package",
                new_callable=AsyncMock,
                return_value=(True, ""),
            ) as perform,
        ):
            code = run_install_request(name="langchain-custom", package=True, yes=True)
        assert code == 0
        perform.assert_awaited_once()


class TestExtrasCliParsing:
    """Parse-level coverage for the extras group."""

    def test_parse_extras_install(self) -> None:
        from deepagents_code.main import parse_args

        with patch.object(
            sys, "argv", ["dcode", "extras", "install", "daytona", "--yes"]
        ):
            args = parse_args()
        assert args.command == "extras"
        assert args.extras_command == "install"
        assert args.extras_target == "daytona"
        assert args.extras_yes is True
        assert args.extras_package is False

    def test_parse_extras_install_package(self) -> None:
        from deepagents_code.main import parse_args

        with patch.object(
            sys,
            "argv",
            ["dcode", "extras", "install", "langchain-custom", "--package", "--yes"],
        ):
            args = parse_args()
        assert args.command == "extras"
        assert args.extras_target == "langchain-custom"
        assert args.extras_package is True
        assert args.extras_yes is True


class TestExtrasCliMain:
    """End-to-end control flow through `cli_main` for the new subcommand."""

    def test_extras_install_via_cli_main(self) -> None:
        from deepagents_code.main import cli_main

        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = False
        mock_stdin.read.return_value = ""
        with (
            patch.object(
                sys, "argv", ["dcode", "extras", "install", "quickjs", "--yes"]
            ),
            patch.object(sys, "stdin", mock_stdin),
            patch("deepagents_code.main.check_cli_dependencies"),
            patch("deepagents_code.config.console", MagicMock(), create=True),
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch(
                "deepagents_code.update_check.create_update_log_path",
                return_value=Path("/tmp/deepagents-install.log"),
            ),
            patch(
                "deepagents_code.update_check.install_extra_command",
                return_value="script",
            ),
            patch(
                "deepagents_code.update_check.perform_install_extra",
                new_callable=AsyncMock,
                return_value=(True, ""),
            ) as perform,
            pytest.raises(SystemExit) as exc_info,
        ):
            cli_main()
        assert int(exc_info.value.code or 0) == 0
        perform.assert_awaited_once()
