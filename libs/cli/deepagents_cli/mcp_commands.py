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
    """Register the `deepagents mcp` command group.

    Args:
        subparsers: The `argparse` subparsers object from the top-level CLI
            parser, onto which the `mcp` command group is attached.
        make_help_action: Factory that wraps a `show_*` callable into an
            `argparse.Action` so `-h/--help` renders the hand-maintained
            help screens from `deepagents_cli.ui` instead of argparse's
            auto-generated text.
    """
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

    When `config_path` is omitted, auto-discovered MCP configs are merged in
    the same precedence order as the runtime loader, with matching trust
    gating: user-level configs are always included, but project-level configs
    are only included when the trust store has a fingerprint match. An
    untrusted project-level config (for example, a `.mcp.json` in a cloned
    repo) is skipped so attacker-controlled `headers` entries cannot exfiltrate
    local secrets during the OAuth handshake. When `config_path` is set, that
    file alone is loaded and treated as explicitly trusted.

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
        _validate_server_config,
        classify_discovered_configs,
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
            if is_project_mcp_trusted(project_root, fingerprint):
                for path in project_paths:
                    config = load_mcp_config_lenient(path)
                    if config is not None:
                        configs.append(config)
                        used_paths.append(path)
            else:
                skipped = ", ".join(str(path) for path in project_paths)
                print(  # noqa: T201
                    "Skipping untrusted project MCP config "
                    f"(not yet approved or config changed): {skipped}. "
                    "Approve it by running `deepagents` in this project, or "
                    "pass --config <path> to use it explicitly.",
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

    # Re-run shape validation on the selected entry. Auto-discovered configs
    # reach this point via `load_mcp_config_lenient`, which intentionally
    # skips validation so one malformed entry doesn't break the whole CLI.
    # Validating here rejects path-traversal server names like "../evil"
    # before they flow into `FileTokenStorage` and onto disk.
    try:
        _validate_server_config(server, servers[server])
    except (TypeError, ValueError) as exc:
        print(f"Invalid MCP server config for {server!r}: {exc}", file=sys.stderr)  # noqa: T201
        return 1

    import httpx
    from pydantic import ValidationError

    try:
        await login(server_name=server, server_config=servers[server])
    except (
        ValueError,
        RuntimeError,
        httpx.HTTPError,
        ValidationError,
        KeyError,
        OSError,
    ) as exc:
        print(f"Login failed: {exc}", file=sys.stderr)  # noqa: T201
        return 1
    return 0
