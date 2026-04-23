"""Tests for the `deepagents mcp` command group."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, patch

if TYPE_CHECKING:
    from collections.abc import Callable


def _build_parser() -> argparse.ArgumentParser:
    from deepagents_cli.mcp_commands import setup_mcp_parsers

    def _make_help_action(help_fn: Callable[[], None]) -> type[argparse.Action]:
        class _ShowHelp(argparse.Action):
            def __init__(
                self,
                option_strings: list[str],
                dest: str = argparse.SUPPRESS,
                default: str = argparse.SUPPRESS,
                **kwargs: Any,
            ) -> None:
                super().__init__(
                    option_strings=option_strings,
                    dest=dest,
                    default=default,
                    nargs=0,
                    **kwargs,
                )

            def __call__(  # ty: ignore[invalid-method-override]
                self,
                parser: argparse.ArgumentParser,
                _namespace: argparse.Namespace,
                _values: object,
                _option_string: str | None = None,
            ) -> None:
                help_fn()
                parser.exit()

        return _ShowHelp

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    setup_mcp_parsers(subparsers, make_help_action=_make_help_action)
    return parser


class TestSetupMCPParsers:
    """Argument parser wiring for the `mcp` subcommand."""

    def test_mcp_login_accepts_server_arg(self) -> None:
        """The parser recognizes `deepagents mcp login <server>`."""
        parser = _build_parser()
        ns = parser.parse_args(["mcp", "login", "notion"])
        assert ns.command == "mcp"
        assert ns.mcp_command == "login"
        assert ns.server == "notion"


class TestRunMCPLogin:
    """Behavior of the `mcp login` command handler."""

    async def test_happy_path(self, tmp_path: Path) -> None:
        """Explicit config loads and forwards the target server config."""
        from deepagents_cli.mcp_commands import run_mcp_login

        config_path = tmp_path / "mcp.json"
        config_path.write_text(
            '{"mcpServers":{"notion":{"transport":"http",'
            '"url":"https://mcp.notion.com/mcp","auth":"oauth"}}}'
        )

        with patch("deepagents_cli.mcp_auth.login", new=AsyncMock()) as mock_login:
            exit_code = await run_mcp_login(
                server="notion",
                config_path=str(config_path),
            )

        assert exit_code == 0
        mock_login.assert_awaited_once()
        kwargs = mock_login.await_args_list[0].kwargs
        assert kwargs["server_name"] == "notion"
        assert kwargs["server_config"]["url"] == "https://mcp.notion.com/mcp"

    async def test_server_not_in_config(self, tmp_path: Path) -> None:
        """Unknown server names return exit code 1."""
        from deepagents_cli.mcp_commands import run_mcp_login

        config_path = tmp_path / "mcp.json"
        config_path.write_text(
            '{"mcpServers":{"linear":{"transport":"http",'
            '"url":"https://mcp.linear.app/mcp","auth":"oauth"}}}'
        )

        exit_code = await run_mcp_login(server="notion", config_path=str(config_path))
        assert exit_code == 1

    async def test_autodiscover_searches_merged_view(self, tmp_path: Path) -> None:
        """Auto-discovery merges all discovered configs before lookup."""
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
                "deepagents_cli.mcp_trust.is_project_mcp_trusted",
                return_value=True,
            ),
            patch("deepagents_cli.mcp_auth.login", new=AsyncMock()) as mock_login,
        ):
            exit_code = await run_mcp_login(server="notion", config_path=None)

        assert exit_code == 0
        mock_login.assert_awaited_once()
        assert mock_login.await_args_list[0].kwargs["server_config"]["url"] == (
            "https://mcp.notion.com/mcp"
        )

    async def test_autodiscover_higher_precedence_wins(self, tmp_path: Path) -> None:
        """When two configs define the same server, the later one wins."""
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
                "deepagents_cli.mcp_trust.is_project_mcp_trusted",
                return_value=True,
            ),
            patch("deepagents_cli.mcp_auth.login", new=AsyncMock()) as mock_login,
        ):
            exit_code = await run_mcp_login(server="notion", config_path=None)

        assert exit_code == 0
        mock_login.assert_awaited_once()
        assert mock_login.await_args_list[0].kwargs["server_config"]["url"] == (
            "https://example.invalid/higher"
        )

    async def test_no_config_found_returns_2(self) -> None:
        """No discovered config files yields exit code 2."""
        from deepagents_cli.mcp_commands import run_mcp_login

        with patch(
            "deepagents_cli.mcp_tools.discover_mcp_configs",
            return_value=[],
        ):
            exit_code = await run_mcp_login(server="notion", config_path=None)

        assert exit_code == 2

    async def test_untrusted_project_remote_config_is_usable(
        self,
        tmp_path: Path,
    ) -> None:
        """Untrusted project remote servers stay available for login."""
        from deepagents_cli.mcp_commands import run_mcp_login

        project_cfg = tmp_path / "project.json"
        project_cfg.write_text(
            '{"mcpServers":{"notion":{"transport":"http",'
            '"url":"https://mcp.notion.com/mcp",'
            '"auth":"oauth"}}}'
        )

        with (
            patch(
                "deepagents_cli.mcp_tools.discover_mcp_configs",
                return_value=[project_cfg],
            ),
            patch(
                "deepagents_cli.mcp_trust.is_project_mcp_trusted",
                return_value=False,
            ),
            patch("deepagents_cli.mcp_auth.login", new=AsyncMock()) as mock_login,
        ):
            exit_code = await run_mcp_login(server="notion", config_path=None)

        assert exit_code == 0
        mock_login.assert_awaited_once()
        assert mock_login.await_args_list[0].kwargs["server_config"]["url"] == (
            "https://mcp.notion.com/mcp"
        )

    async def test_untrusted_project_stdio_is_filtered_but_remote_survives(
        self,
        tmp_path: Path,
        capsys,
    ) -> None:
        """Untrusted project configs skip stdio entries without dropping remotes."""
        from deepagents_cli.mcp_commands import run_mcp_login

        project_cfg = tmp_path / "project.json"
        project_cfg.write_text(
            '{"mcpServers":{"filesystem":{"command":"npx","args":["@mcp/fs"]},'
            '"notion":{"transport":"http","url":"https://mcp.notion.com/mcp",'
            '"auth":"oauth"}}}'
        )

        with (
            patch(
                "deepagents_cli.mcp_tools.discover_mcp_configs",
                return_value=[project_cfg],
            ),
            patch(
                "deepagents_cli.mcp_trust.is_project_mcp_trusted",
                return_value=False,
            ),
            patch("deepagents_cli.mcp_auth.login", new=AsyncMock()) as mock_login,
        ):
            exit_code = await run_mcp_login(server="notion", config_path=None)

        assert exit_code == 0
        mock_login.assert_awaited_once()
        assert "Skipping untrusted project stdio MCP servers" in capsys.readouterr().err

    async def test_user_level_config_is_trusted_without_approval(
        self,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        """Configs under `~/.deepagents` are always trusted."""
        from deepagents_cli.mcp_commands import run_mcp_login

        fake_home = tmp_path / "home"
        user_dir = fake_home / ".deepagents"
        user_dir.mkdir(parents=True)
        user_cfg = user_dir / ".mcp.json"
        user_cfg.write_text(
            '{"mcpServers":{"notion":{"transport":"http",'
            '"url":"https://mcp.notion.com/mcp","auth":"oauth"}}}'
        )
        monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))

        with (
            patch(
                "deepagents_cli.mcp_tools.discover_mcp_configs",
                return_value=[user_cfg],
            ),
            patch(
                "deepagents_cli.mcp_trust.is_project_mcp_trusted",
                return_value=False,
            ),
            patch("deepagents_cli.mcp_auth.login", new=AsyncMock()) as mock_login,
        ):
            exit_code = await run_mcp_login(server="notion", config_path=None)

        assert exit_code == 0
        mock_login.assert_awaited_once()
