"""CLI commands for `deepagents mcp`.

Registered via `setup_mcp_parsers` in `main.py`.
"""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING, Any

from deepagents_cli.mcp_auth import login
from deepagents_cli.mcp_tools import discover_mcp_configs, load_mcp_config

if TYPE_CHECKING:
    from collections.abc import Callable


def setup_mcp_parsers(
    subparsers: Any,  # noqa: ANN401
    *,
    make_help_action: Callable[[Callable[[], None]], type[argparse.Action]],
) -> None:
    """Register the `deepagents mcp` command group."""
    mcp_parser = subparsers.add_parser(
        "mcp",
        help="Manage MCP servers",
        add_help=False,
    )
    mcp_parser.add_argument(
        "-h",
        "--help",
        action=make_help_action(lambda: mcp_parser.print_help()),
    )
    mcp_sub = mcp_parser.add_subparsers(dest="mcp_command")

    login_parser = mcp_sub.add_parser(
        "login",
        help="Run the OAuth login flow for an MCP server",
        add_help=False,
    )
    login_parser.add_argument("server", help="Server name from mcpServers config")
    login_parser.add_argument(
        "--config",
        dest="config_path",
        default=None,
        help="Path to an MCP config JSON file (default: auto-discovered)",
    )
    login_parser.add_argument(
        "-h",
        "--help",
        action=make_help_action(lambda: login_parser.print_help()),
    )


async def run_mcp_login(*, server: str, config_path: str | None) -> int:
    """Handler for `deepagents mcp login <server>`. Returns an exit code."""
    if config_path is None:
        found = discover_mcp_configs()
        if not found:
            print(  # noqa: T201
                "No MCP config file found. Pass --config <path>.",
                file=sys.stderr,
            )
            return 2
        config_path = str(found[-1])  # highest-precedence file

    config = load_mcp_config(config_path)
    servers = config.get("mcpServers", {})
    if server not in servers:
        print(  # noqa: T201
            f"Server {server!r} not found in {config_path}. "
            f"Known servers: {sorted(servers)}",
            file=sys.stderr,
        )
        return 1

    try:
        await login(server_name=server, server_config=servers[server])
    except (ValueError, RuntimeError) as exc:
        print(f"Login failed: {exc}", file=sys.stderr)  # noqa: T201
        return 1
    return 0
