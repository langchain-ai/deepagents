"""CLI commands of the MCP module."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable

    from deepagents_code.mcp_login_service import ConfigResolutionError


def _lazy_ui_help(fn_name: str) -> Callable[[], None]:
    """Return a callable that lazily imports and invokes a `ui` help function."""

    def _show() -> None:
        from deepagents_code import ui

        getattr(ui, fn_name)()

    return _show


def setup_mcp_parsers(
    subparsers: Any,  # noqa: ANN401
    *,
    make_help_action: Callable[[Callable[[], None]], type[argparse.Action]],
) -> None:
    """Register the `dcode mcp` command group.

    Args:
        subparsers: The `argparse` subparsers object from the top-level CLI
            parser, onto which the `mcp` command group is attached.
        make_help_action: Factory that wraps a `show_*` callable into an
            `argparse.Action` so `-h/--help` renders the hand-maintained
            help screens from `deepagents_code.ui` instead of argparse's
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
    """Handle `dcode mcp login <server>`.

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
    from deepagents_code.mcp_auth import login
    from deepagents_code.mcp_login_service import (
        ConfigErrorKind,
        ConfigResolution,
        ConfigResolutionError,
        format_untrusted_project_notice,
        resolve_mcp_config,
        select_server,
    )
    from deepagents_code.mcp_oauth_ui import CliOAuthInteraction

    resolution = resolve_mcp_config(config_path)
    if isinstance(resolution, ConfigResolutionError):
        _print_resolution_error(resolution)
        return 2 if resolution.kind is ConfigErrorKind.NO_CONFIG_FOUND else 1

    if not isinstance(resolution, ConfigResolution):  # pragma: no cover - safety
        print(  # noqa: T201
            "Internal error: unexpected result from resolve_mcp_config. "
            "Please report this bug.",
            file=sys.stderr,
        )
        return 1

    notice = format_untrusted_project_notice(resolution.untrusted_project_paths)
    if notice:
        print(notice, file=sys.stderr)  # noqa: T201

    selection = select_server(resolution, server)
    if isinstance(selection, ConfigResolutionError):
        print(selection.message, file=sys.stderr)  # noqa: T201
        return 1

    import httpx
    from pydantic import ValidationError

    try:
        await login(
            server_name=selection.server_name,
            server_config=selection.server_config,
            ui=CliOAuthInteraction(),
        )
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


def _print_resolution_error(error: ConfigResolutionError) -> None:
    """Print the untrusted-paths notice (if any) then `error.message` to stderr.

    Note: the untrusted-paths notice is also surfaced independently for
    successful resolutions in `run_mcp_login`.
    """
    from deepagents_code.mcp_login_service import format_untrusted_project_notice

    notice = format_untrusted_project_notice(error.untrusted_project_paths)
    if notice:
        print(notice, file=sys.stderr)  # noqa: T201
    print(error.message, file=sys.stderr)  # noqa: T201
