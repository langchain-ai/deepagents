"""Tests for the `deepagents mcp` subcommand group."""

import argparse
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest


def _build_parser():
    from deepagents_cli.mcp_commands import setup_mcp_parsers

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    class _NoOpHelpAction(argparse.Action):
        def __call__(self, *args: object, **kwargs: object) -> None:  # pragma: no cover
            return None

    def _make_help_action(_fn):
        return _NoOpHelpAction

    setup_mcp_parsers(subparsers, make_help_action=_make_help_action)
    return parser


class TestSetupMCPParsers:
    def test_mcp_login_accepts_server_arg(self) -> None:
        parser = _build_parser()
        ns = parser.parse_args(["mcp", "login", "notion"])
        assert ns.command == "mcp"
        assert ns.mcp_command == "login"
        assert ns.server == "notion"


class TestRunMCPLogin:
    async def test_happy_path(self, tmp_path: Path) -> None:
        from deepagents_cli.mcp_commands import run_mcp_login

        config_path = tmp_path / "mcp.json"
        config_path.write_text(
            '{"mcpServers":{"notion":{"transport":"http",'
            '"url":"https://mcp.notion.com/mcp","auth":"oauth"}}}'
        )

        with patch(
            "deepagents_cli.mcp_auth.login",
            new=AsyncMock(return_value=None),
        ) as mocked:
            exit_code = await run_mcp_login(
                server="notion", config_path=str(config_path)
            )
        assert exit_code == 0
        mocked.assert_awaited_once()
        await_args = mocked.await_args
        assert await_args is not None
        kwargs = await_args.kwargs
        assert kwargs["server_name"] == "notion"
        assert kwargs["server_config"]["url"] == "https://mcp.notion.com/mcp"

    async def test_server_not_in_config(self, tmp_path: Path) -> None:
        from deepagents_cli.mcp_commands import run_mcp_login

        config_path = tmp_path / "mcp.json"
        config_path.write_text(
            '{"mcpServers":{"linear":{"transport":"http",'
            '"url":"https://mcp.linear.app/mcp","auth":"oauth"}}}'
        )
        exit_code = await run_mcp_login(server="notion", config_path=str(config_path))
        assert exit_code != 0

    async def test_autodiscover_searches_merged_view(self, tmp_path: Path) -> None:
        """Without `--config`, discovered configs are merged before lookup.

        The target server lives only in the lower-precedence config; login must
        still find it rather than failing with "server not found" (regression
        against `found[-1]` only behaviour).
        """
        from deepagents_cli.mcp_commands import run_mcp_login

        lower = tmp_path / "lower.json"
        lower.write_text(
            '{"mcpServers":{"notion":{"transport":"http",'
            '"url":"https://mcp.notion.com/mcp","auth":"oauth"}}}'
        )
        higher = tmp_path / "higher.json"
        higher.write_text(
            '{"mcpServers":{"linear":{"transport":"http",'
            '"url":"https://mcp.linear.app/mcp","auth":"oauth"}}}'
        )

        with (
            patch(
                "deepagents_cli.mcp_tools.discover_mcp_configs",
                return_value=[lower, higher],
            ),
            patch(
                "deepagents_cli.mcp_auth.login",
                new=AsyncMock(return_value=None),
            ) as mocked,
        ):
            exit_code = await run_mcp_login(server="notion", config_path=None)

        assert exit_code == 0
        mocked.assert_awaited_once()
        kwargs = mocked.await_args.kwargs
        assert kwargs["server_name"] == "notion"
        assert kwargs["server_config"]["url"] == "https://mcp.notion.com/mcp"

    async def test_autodiscover_higher_overrides_lower(self, tmp_path: Path) -> None:
        """When both configs define the same server, the later one wins."""
        from deepagents_cli.mcp_commands import run_mcp_login

        lower = tmp_path / "lower.json"
        lower.write_text(
            '{"mcpServers":{"notion":{"transport":"http",'
            '"url":"https://example.invalid/lower","auth":"oauth"}}}'
        )
        higher = tmp_path / "higher.json"
        higher.write_text(
            '{"mcpServers":{"notion":{"transport":"http",'
            '"url":"https://example.invalid/higher","auth":"oauth"}}}'
        )

        with (
            patch(
                "deepagents_cli.mcp_tools.discover_mcp_configs",
                return_value=[lower, higher],
            ),
            patch(
                "deepagents_cli.mcp_auth.login",
                new=AsyncMock(return_value=None),
            ) as mocked,
        ):
            exit_code = await run_mcp_login(server="notion", config_path=None)

        assert exit_code == 0
        kwargs = mocked.await_args.kwargs
        assert kwargs["server_config"]["url"] == "https://example.invalid/higher"

    async def test_autodiscover_none_found_exits_2(self) -> None:
        from deepagents_cli.mcp_commands import run_mcp_login

        with patch(
            "deepagents_cli.mcp_tools.discover_mcp_configs",
            return_value=[],
        ):
            exit_code = await run_mcp_login(server="notion", config_path=None)
        assert exit_code == 2
