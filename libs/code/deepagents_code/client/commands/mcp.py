"""CLI commands of the MCP module."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable

    from deepagents_code.mcp_login_service import (
        ConfigResolution,
        ConfigResolutionError,
    )


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
        help="List servers needing login or run an OAuth login flow",
        add_help=False,
    )
    login_parser.add_argument(
        "server",
        nargs="?",
        help="Server name from mcpServers config; omit to list servers needing login",
    )
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


async def run_mcp_login_list(*, config_path: str | None) -> int:
    """List configured OAuth servers without stored tokens.

    Servers are drawn from the same trust-gated resolution as `run_mcp_login`,
    so untrusted project-level entries are excluded from the scan.

    A server counts as needing login when it opted into OAuth and has no
    stored token at all. Expiry is deliberately not consulted, matching the
    runtime's upfront gate in `resolve_and_load_mcp_tools`.

    Returns:
        Process exit code: 0 when every configured server's login state was
            determined — including when some of them need login, which is
            informational rather than a failure; 1 when the config could not
            be resolved or any server's state is unknown (unreadable token
            state, unresolvable config, or a config file that failed to
            load); or 2 when no config file was found.
    """
    from deepagents_code._invocation import invoked_name
    from deepagents_code.mcp_login_service import (
        ConfigErrorKind,
        ConfigResolution,
        ConfigResolutionError,
        resolve_mcp_config,
    )
    from deepagents_code.ui import console

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

    _print_resolution_notices(resolution)
    from deepagents_code.mcp_auth import FileTokenStorage, format_login_failure
    from deepagents_code.mcp_config import resolve_mcp_server_env
    from deepagents_code.mcp_tools import _drop_invalid_mcp_config_servers

    # Defense in depth: `resolve_mcp_config` already validates every source, so
    # `errors` is expected to stay empty. Keep the call anyway — it is what
    # guarantees the `resolved_config["url"]` index and `FileTokenStorage`'s
    # server-name regex below cannot raise, and those run outside the `try`.
    valid_config, errors = _drop_invalid_mcp_config_servers(resolution.config)
    for server_name, error in errors.items():
        print(  # noqa: T201
            f"Invalid MCP server config for {server_name!r}: {error}", file=sys.stderr
        )

    needs_login: list[str] = []
    # Servers whose login state could not be determined. A config file that
    # failed to load counts too: it may have held an OAuth server that never
    # reached the scan, so the picture is incomplete before the loop starts.
    unreadable = len(errors) + len(resolution.load_errors)
    for server_name, server_config in valid_config["mcpServers"].items():
        if server_config.get("auth") != "oauth":
            continue
        try:
            resolved_config = resolve_mcp_server_env(server_name, server_config)
        except (RuntimeError, TypeError) as exc:
            print(  # noqa: T201
                f"Invalid MCP server config for {server_name!r}: {exc}",
                file=sys.stderr,
            )
            unreadable += 1
            continue
        storage = FileTokenStorage(server_name, server_url=resolved_config["url"])
        try:
            tokens = await storage.get_tokens()
        except (OSError, RuntimeError, ValueError) as exc:
            # `FileTokenStorage` raises `OSError`/`RuntimeError` from its own
            # file read, and those messages carry the token path and the
            # "delete it and re-login" remedy — render them verbatim.
            # `format_login_failure` is for exceptions that may embed an
            # `OAuthToken`, which here is only the pydantic `ValidationError`
            # from parsing the stored payload.
            detail = (
                str(exc)
                if isinstance(exc, OSError | RuntimeError)
                else format_login_failure(exc)
            )
            print(  # noqa: T201
                f"Could not read login state for {server_name!r}: {detail}",
                file=sys.stderr,
            )
            unreadable += 1
            continue
        if tokens is None:
            needs_login.append(server_name)

    if needs_login:
        console.print("MCP servers needing login:")
        for server_name in needs_login:
            console.print(f"  {server_name}", markup=False)
        console.print()
        console.print(
            f"Run `{invoked_name()} mcp login <server>` to authenticate.",
            markup=False,
        )
    elif not unreadable:
        console.print("No MCP servers need login.")

    # An undetermined server is not an all-clear: its login state is unknown,
    # so both "no servers need login" and a bare list would overstate what was
    # actually checked. Say so on stdout — the per-server reasons went to
    # stderr, which is easily lost when only stdout is read or piped.
    if unreadable:
        if needs_login:
            console.print()
        console.print(f"{unreadable} server(s) could not be checked; see errors above.")
        return 1
    return 0


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

    _print_resolution_notices(resolution)

    selection = select_server(resolution, server)
    if isinstance(selection, ConfigResolutionError):
        print(selection.message, file=sys.stderr)  # noqa: T201
        return 1

    import httpx
    from pydantic import ValidationError

    from deepagents_code.mcp_auth import format_login_failure, token_store_dir

    try:
        await login(
            server_name=selection.server_name,
            server_config=selection.server_config,
            ui=CliOAuthInteraction(),
        )
    except PermissionError as exc:
        from deepagents_code._paths import PATHS

        token_store = token_store_dir()
        token_store_display = PATHS.display(token_store)
        print(  # noqa: T201
            f"Login failed: cannot write to the MCP tokens store ({exc}). "
            f"Check permissions on {token_store_display} and "
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
    from deepagents_code._paths import PATHS, project_paths
    from deepagents_code.mcp_tools import (
        MCP_CONFIG_DISCOVERY_PATHS,
        _resolve_project_config_base,
    )
    from deepagents_code.ui import console

    project = project_paths(_resolve_project_config_base(None))

    # Same three locations, same order, as `discover_mcp_config_sources`. The
    # user row is rendered from live `PATHS` so a test that patches the
    # snapshot sees a row consistent with the rest of this output; the display
    # constant is frozen at the same import and cannot follow a patch.
    user_config = PATHS.profile.mcp_config_file
    candidates = (
        (PATHS.display(user_config), user_config),
        (MCP_CONFIG_DISCOVERY_PATHS[1][0], project.config_mcp_config_file),
        (MCP_CONFIG_DISCOVERY_PATHS[2][0], project.root_mcp_config_file),
    )
    rows = [
        (display, label, path.is_file())
        for (_, label), (display, path) in zip(
            MCP_CONFIG_DISCOVERY_PATHS, candidates, strict=True
        )
    ]

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


def _print_resolution_notices(resolution: ConfigResolution) -> None:
    """Print notices attached to a successful config resolution."""
    from deepagents_code.mcp_login_service import (
        format_legacy_env_ignored_notice,
        format_legacy_ignored_notice,
        format_load_errors_notice,
        format_malformed_approvals_notice,
        format_policy_error_notice,
        format_untrusted_project_notice,
    )

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
    # `format_load_errors_notice` reports configs that failed to parse/validate
    # but were dropped while another config still loaded — otherwise a broken
    # .mcp.json is invisible on this surface (the runtime loader reports the
    # same failures as error rows).
    notices = (
        format_legacy_ignored_notice(resolution.legacy_ignored),
        format_legacy_env_ignored_notice(resolution.legacy_env_ignored),
        format_malformed_approvals_notice(resolution.malformed_approvals),
        format_load_errors_notice(resolution.load_errors),
    )
    for notice in notices:
        if notice:
            print(notice, file=sys.stderr)  # noqa: T201


def _print_resolution_error(error: ConfigResolutionError) -> None:
    """Print the trust/migration notices, then `error.message`.

    Prints the untrusted-paths notice (suppressed when `policy_error` already
    explains the drop), then the legacy-key, legacy-env, and malformed-approval
    notices. Per-path load errors are not printed here because `error.message`
    already embeds them for `NO_USABLE_CONFIG`. These notices are also surfaced
    independently for successful resolutions by `_print_resolution_notices`.
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
