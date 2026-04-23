"""CLI commands for `deepagents mcp`."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable


def _lazy_ui_help(fn_name: str) -> Callable[[], None]:
    """Return a callable that lazily imports and invokes a `ui` help function."""

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
    """Handle `deepagents mcp login <server>`.

    Args:
        server: Target server name from `mcpServers`.
        config_path: Optional explicit MCP config path.

    Returns:
        Process exit code: 0 on success, 1 on config or login failure,
        2 if no config file could be found.
    """
    from pathlib import Path

    from deepagents_cli.mcp_auth import login
    from deepagents_cli.mcp_tools import (
        _filter_project_stdio_servers,
        classify_discovered_configs,
        discover_mcp_configs,
        extract_stdio_server_commands,
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

        user_paths, project_paths = classify_discovered_configs(found)
        configs: list[dict[str, Any]] = []
        used_paths: list[Path] = []

        for path in user_paths:
            config = load_mcp_config_lenient(path)
            if config is not None:
                configs.append(config)
                used_paths.append(path)

        if project_paths:
            from deepagents_cli.mcp_trust import (
                compute_config_fingerprint,
                is_project_mcp_trusted,
            )
            from deepagents_cli.project_utils import find_project_root

            project_root = str((find_project_root() or Path.cwd()).resolve())
            fingerprint = compute_config_fingerprint(project_paths)
            allow_project_stdio = is_project_mcp_trusted(project_root, fingerprint)
            for path in project_paths:
                config = load_mcp_config_lenient(path)
                if config is None:
                    continue
                if allow_project_stdio:
                    configs.append(config)
                    used_paths.append(path)
                    continue

                stdio_servers = extract_stdio_server_commands(config)
                filtered = _filter_project_stdio_servers(config)
                if filtered.get("mcpServers"):
                    configs.append(filtered)
                    used_paths.append(path)
                if stdio_servers:
                    skipped = [
                        f"{name}: {cmd} {' '.join(args)}"
                        for name, cmd, args in stdio_servers
                    ]
                    print(  # noqa: T201
                        "Skipping untrusted project stdio MCP servers "
                        "(config changed or not yet approved): "
                        f"{'; '.join(skipped)}. "
                        "Approve them by running `deepagents` in this project.",
                        file=sys.stderr,
                    )

        if not configs:
            found_paths = ", ".join(str(path) for path in found)
            print(  # noqa: T201
                f"No usable MCP config found in: {found_paths}",
                file=sys.stderr,
            )
            return 1

        config = merge_mcp_configs(configs)
        search_label = ", ".join(str(path) for path in used_paths)

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
