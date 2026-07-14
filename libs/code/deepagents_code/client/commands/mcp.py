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
        "--mcp-config",
        dest="config_path",
        default=None,
        help="Path to an MCP config JSON file. Falls back to the top-level "
        "`--mcp-config`, then to auto-discovered configs.",
    )
    login_parser.add_argument(
        "-h",
        "--help",
        action=make_help_action(_lazy_ui_help("show_mcp_login_help")),
    )

    config_parser = mcp_sub.add_parser(
        "config",
        help="Show MCP config discovery paths",
        add_help=False,
    )
    config_parser.add_argument(
        "-h",
        "--help",
        action=make_help_action(_lazy_ui_help("show_mcp_config_help")),
    )


# Maintainer note: `deepagents-talon` dynamically imports `run_mcp_login` from
# this module for its `talon mcp login` command. Keep the function name,
# keyword-only signature, async behavior, and integer exit-code contract stable
# unless `deepagents-talon` is migrated in the same change.
async def run_mcp_login(*, server: str, config_path: str | None) -> int:
    """Handle `dcode mcp login <server>`.

    When `config_path` is omitted, auto-discovered MCP configs are merged in
    the same precedence order as the runtime loader, with matching trust
    gating: user-level configs are always included, but project-level configs
    contribute only servers with matching scoped approvals (or the process-wide
    `DANGEROUSLY_ENABLE_PROJECT_MCP_SERVERS` allowlist) and no deny-list entry.
    Untrusted project-level server entries (for example, from a `.mcp.json`
    in a cloned repo) are skipped so attacker-controlled `headers` entries
    cannot exfiltrate local secrets during the OAuth handshake. When
    `config_path` is set, that file alone is loaded and treated as explicitly
    trusted.

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
        format_legacy_env_ignored_notice,
        format_legacy_ignored_notice,
        format_malformed_approvals_notice,
        format_policy_error_notice,
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

    # A policy read failure and an "untrusted project" skip are mutually
    # exclusive reasons for the same dropped servers; surface the policy error
    # (the real, actionable cause) instead of nudging the user to re-approve.
    policy_notice = format_policy_error_notice(resolution.policy_error)
    if policy_notice:
        print(policy_notice, file=sys.stderr)  # noqa: T201
    else:
        notice = format_untrusted_project_notice(resolution.untrusted_project_paths)
        if notice:
            print(notice, file=sys.stderr)  # noqa: T201
    legacy_notice = format_legacy_ignored_notice(resolution.legacy_ignored)
    if legacy_notice:
        print(legacy_notice, file=sys.stderr)  # noqa: T201
    legacy_env_notice = format_legacy_env_ignored_notice(resolution.legacy_env_ignored)
    if legacy_env_notice:
        print(legacy_env_notice, file=sys.stderr)  # noqa: T201
    malformed_notice = format_malformed_approvals_notice(resolution.malformed_approvals)
    if malformed_notice:
        print(malformed_notice, file=sys.stderr)  # noqa: T201

    selection = select_server(resolution, server)
    if isinstance(selection, ConfigResolutionError):
        print(selection.message, file=sys.stderr)  # noqa: T201
        return 1

    import httpx
    from pydantic import ValidationError

    from deepagents_code.mcp_auth import format_login_failure

    try:
        await login(
            server_name=selection.server_name,
            server_config=selection.server_config,
            ui=CliOAuthInteraction(),
        )
    except PermissionError as exc:
        # PermissionError typically means the user's home dir or the
        # ~/.deepagents/.state/mcp-tokens/ tree isn't writable. Retrying
        # without a hint sends users in circles.
        print(  # noqa: T201
            f"Login failed: cannot write to the MCP tokens store ({exc}). "
            f"Check permissions on ~/.deepagents/.state/mcp-tokens/ and "
            f"retry `dcode mcp login {selection.server_name}`.",
            file=sys.stderr,
        )
        return 1
    except (
        ValueError,
        RuntimeError,
        httpx.HTTPError,
        ValidationError,
        KeyError,
        OSError,
    ) as exc:
        print(  # noqa: T201
            f"Login failed: {format_login_failure(exc)}",
            file=sys.stderr,
        )
        return 1
    return 0


def run_mcp_config() -> int:
    """Handle `dcode mcp config`.

    Prints the MCP config discovery paths in precedence order with a
    marker showing which exist on disk. Stat-only; never opens config
    files, so config-trust prompts are not triggered.

    Returns:
        Process exit code: always 0.
    """
    from pathlib import Path

    from deepagents_code.mcp_tools import (
        _resolve_project_config_base,
        discover_mcp_configs,
    )
    from deepagents_code.ui import console

    found = {str(p.resolve()) for p in discover_mcp_configs()}
    user_dir = Path.home() / ".deepagents"
    project_root = _resolve_project_config_base(None)

    rows: list[tuple[str, str, bool]] = []
    for display, label, resolved in (
        ("~/.deepagents/.mcp.json", "user-level", user_dir / ".mcp.json"),
        (
            "<project-root>/.deepagents/.mcp.json",
            "project subdir",
            project_root / ".deepagents" / ".mcp.json",
        ),
        ("<project-root>/.mcp.json", "project root", project_root / ".mcp.json"),
    ):
        exists = str(resolved.resolve()) in found or resolved.is_file()
        rows.append((display, label, exists))

    width = max(len(p) for p, _, _ in rows)
    console.print(
        "MCP config discovery paths (lowest to highest precedence):",
        highlight=False,
    )
    for display, label, exists in rows:
        marker = "found" if exists else "missing"
        console.print(
            f"  [{marker:>7}]  {display:<{width}}  ({label})",
            highlight=False,
            markup=False,
        )
    console.print()
    console.print(
        "<project-root> = nearest ancestor with `.git`, else current directory.",
        highlight=False,
    )
    console.print(
        "Override via `--mcp-config <path>` at the top level or on "
        "`dcode mcp login <server>`.",
        highlight=False,
    )
    return 0


def _print_resolution_error(error: ConfigResolutionError) -> None:
    """Print the untrusted-paths and legacy-migration notices then the message.

    Note: both notices are also surfaced independently for successful
    resolutions in `run_mcp_login`.
    """
    from deepagents_code.mcp_login_service import (
        format_legacy_env_ignored_notice,
        format_legacy_ignored_notice,
        format_malformed_approvals_notice,
        format_untrusted_project_notice,
    )

    # On a policy read failure `error.message` already states the reason, so
    # skip the untrusted-paths notice that would otherwise misattribute the
    # dropped servers to "not yet approved."
    if error.policy_error is None:
        notice = format_untrusted_project_notice(error.untrusted_project_paths)
        if notice:
            print(notice, file=sys.stderr)  # noqa: T201
    legacy_notice = format_legacy_ignored_notice(error.legacy_ignored)
    if legacy_notice:
        print(legacy_notice, file=sys.stderr)  # noqa: T201
    legacy_env_notice = format_legacy_env_ignored_notice(error.legacy_env_ignored)
    if legacy_env_notice:
        print(legacy_env_notice, file=sys.stderr)  # noqa: T201
    malformed_notice = format_malformed_approvals_notice(error.malformed_approvals)
    if malformed_notice:
        print(malformed_notice, file=sys.stderr)  # noqa: T201
    print(error.message, file=sys.stderr)  # noqa: T201
