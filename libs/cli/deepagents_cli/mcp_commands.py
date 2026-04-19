"""CLI commands for `deepagents mcp`.

Registered via `setup_mcp_parsers` in `main.py`.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable


def _lazy_ui_help(fn_name: str) -> Callable[[], None]:
    """Return a callable that lazily imports and invokes a `ui` help function.

    Defers the `ui` import (which pulls in Rich + config) until the user
    actually triggers the help action, keeping parse-time imports cheap.
    """

    def _show() -> None:
        from deepagents_cli import ui

        getattr(ui, fn_name)()

    return _show


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
        action=make_help_action(_lazy_ui_help("show_mcp_help")),
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
        action=make_help_action(_lazy_ui_help("show_mcp_login_help")),
    )


async def run_mcp_login(*, server: str, config_path: str | None) -> int:
    """Handler for `deepagents mcp login <server>`.

    When ``config_path`` is omitted, every auto-discovered MCP config is merged
    in the same precedence order used by the runtime loader (later discoveries
    override earlier ones). When ``config_path`` is set, that file alone is
    loaded — matching the existing "explicit wins" behaviour.

    Returns:
        Process exit code: 0 on success, 1 on config or login failure,
        2 if no config file could be found.
    """
    from deepagents_cli.mcp_auth import login
    from deepagents_cli.mcp_tools import (
        discover_mcp_configs,
        load_mcp_config,
        load_mcp_config_lenient,
        merge_mcp_configs,
    )

    if config_path is not None:
        try:
            config = load_mcp_config(config_path)
        except (FileNotFoundError, TypeError, ValueError, RuntimeError) as exc:
            print(  # noqa: T201
                f"Failed to load MCP config {config_path}: {exc}",
                file=sys.stderr,
            )
            return 1
        search_label = config_path
    else:
        found = discover_mcp_configs()
        if not found:
            print(  # noqa: T201
                "No MCP config file found. Pass --config <path>.",
                file=sys.stderr,
            )
            return 2
        # Merge all discovered configs so `login` sees the same view the
        # runtime loader builds — later paths override earlier ones.
        configs = [
            cfg
            for cfg in (load_mcp_config_lenient(p) for p in found)
            if cfg is not None
        ]
        if not configs:
            print(  # noqa: T201
                f"No usable MCP config found in: {', '.join(str(p) for p in found)}",
                file=sys.stderr,
            )
            return 1
        config = merge_mcp_configs(configs)
        search_label = ", ".join(str(p) for p in found)

    servers = config.get("mcpServers", {})
    if server not in servers:
        print(  # noqa: T201
            f"Server {server!r} not found in {search_label}. "
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
