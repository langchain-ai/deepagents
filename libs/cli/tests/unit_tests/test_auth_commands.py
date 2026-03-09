import argparse
from collections.abc import Callable, Sequence
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


def _make_help_action(_fn: Callable[[], None]) -> type[argparse.Action]:
    class _Help(argparse.Action):
        def __call__(
            self,
            parser: argparse.ArgumentParser,
            namespace: argparse.Namespace,
            values: str | Sequence[Any] | None,
            option_string: str | None = None,
        ) -> None:
            pass

    return _Help


def _build_parser() -> argparse.ArgumentParser:
    from deepagents_cli.auth.commands import setup_auth_parser

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    setup_auth_parser(subparsers, make_help_action=_make_help_action)
    return parser


class TestSetupAuthParser:
    def test_parser_accepts_auth_login(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["auth", "login", "--provider", "codex"])
        assert args.command == "auth"
        assert args.auth_command == "login"
        assert args.provider == "codex"

    def test_parser_accepts_auth_status(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["auth", "status"])
        assert args.auth_command == "status"
        assert args.provider == "codex"

    def test_parser_accepts_auth_logout(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["auth", "logout", "--provider", "codex"])
        assert args.auth_command == "logout"

    def test_headless_flag(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["auth", "login", "--headless"])
        assert args.headless is True


class TestExecuteAuthCommand:
    @patch("deepagents_cli.auth.commands._handle_login")
    def test_dispatch_login(self, mock_login: MagicMock) -> None:
        from deepagents_cli.auth.commands import execute_auth_command

        args = argparse.Namespace(
            auth_command="login", provider="codex", headless=False
        )
        execute_auth_command(args)
        mock_login.assert_called_once_with("codex", headless=False)

    @patch("deepagents_cli.auth.commands._handle_status")
    def test_dispatch_status(self, mock_status: MagicMock) -> None:
        from deepagents_cli.auth.commands import execute_auth_command

        args = argparse.Namespace(auth_command="status", provider="codex")
        execute_auth_command(args)
        mock_status.assert_called_once_with("codex")

    @patch("deepagents_cli.auth.commands._handle_logout")
    def test_dispatch_logout(self, mock_logout: MagicMock) -> None:
        from deepagents_cli.auth.commands import execute_auth_command

        args = argparse.Namespace(auth_command="logout", provider="codex")
        execute_auth_command(args)
        mock_logout.assert_called_once_with("codex")


class TestHandleLoginImportError:
    @patch.dict("sys.modules", {"deepagents_codex": None})
    def test_login_missing_package(self) -> None:
        from deepagents_cli.auth.commands import _handle_login

        with pytest.raises(SystemExit):
            _handle_login("codex")
