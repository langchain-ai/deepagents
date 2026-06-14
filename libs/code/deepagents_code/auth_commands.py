"""CLI commands for the `auth` group: manage stored provider credentials.

These subcommands mirror the in-TUI `/auth` modal verbs so credentials can be
managed non-interactively (dotfile bootstrap, CI, remote boxes) without
launching the Textual app:

- `auth list` — one row per known provider with its resolution status.
- `auth set <provider>` — store an API key read from stdin (preferred) or,
    with `--from-env VAR`, copied from a process environment variable.
- `auth remove <provider>` — delete a stored credential (aliases `rm`/`delete`).
- `auth status [<provider>]` — print the resolution source (`stored`,
    `env: VAR`, `missing`, ...) for one or all providers.
- `auth path` — print the resolved `auth.json` path.

Security notes:

- The key value is never echoed, logged, or printed back. `set` defaults to
    reading from stdin so the key never lands in shell history or argv, and it
    refuses an interactive TTY (use `--from-env` instead) so an accidental
    invocation cannot hang waiting on input.
- `set` routes through `auth_store.set_stored_key`, so chmod warnings from the
    same `WriteOutcome` path the TUI uses are surfaced on stderr.

Help rendering for a bare `auth` invocation is served by `ui.show_auth_help`,
which does not import this module. The heavy `model_config` imports here are
function-local so a bare `auth`/`auth -h` invocation stays on the startup fast
path (`parse_args` imports this module only for its light top-level imports).
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable

    from deepagents_code.model_config import ProviderAuthStatus

logger = logging.getLogger(__name__)


def _lazy_ui_help(fn_name: str) -> Callable[[], None]:
    """Return a callable that lazily imports and invokes a `ui` help function."""

    def _show() -> None:
        from deepagents_code import ui

        getattr(ui, fn_name)()

    return _show


def setup_auth_parser(
    subparsers: Any,  # noqa: ANN401
    *,
    make_help_action: Callable[[Callable[[], None]], type[argparse.Action]],
) -> None:
    """Register the `dcode auth` command group.

    Args:
        subparsers: The `argparse` subparsers object from the top-level CLI
            parser, onto which the `auth` command group is attached.
        make_help_action: Factory that wraps a `show_*` callable into an
            `argparse.Action` so `-h/--help` renders the hand-maintained
            help screen from `deepagents_code.ui`.
    """
    auth_parser = subparsers.add_parser(
        "auth",
        help="Manage stored model-provider credentials",
        add_help=False,
    )
    auth_parser.add_argument(
        "-h",
        "--help",
        action=make_help_action(_lazy_ui_help("show_auth_help")),
    )
    auth_sub = auth_parser.add_subparsers(dest="auth_command")

    list_parser = auth_sub.add_parser(
        "list",
        aliases=["ls"],
        help="List providers and their credential status",
        add_help=False,
    )
    list_parser.add_argument(
        "-h",
        "--help",
        action=make_help_action(_lazy_ui_help("show_auth_help")),
    )

    set_parser = auth_sub.add_parser(
        "set",
        help="Store an API key for a provider (key read from stdin)",
        add_help=False,
    )
    set_parser.add_argument("provider", help="Provider name (e.g. anthropic)")
    set_parser.add_argument(
        "--from-env",
        dest="from_env",
        metavar="VAR",
        default=None,
        help="Copy the key from this environment variable instead of stdin",
    )
    set_parser.add_argument(
        "-h",
        "--help",
        action=make_help_action(_lazy_ui_help("show_auth_help")),
    )

    remove_parser = auth_sub.add_parser(
        "remove",
        aliases=["rm", "delete"],
        help="Remove a stored credential for a provider",
        add_help=False,
    )
    remove_parser.add_argument("provider", help="Provider name (e.g. anthropic)")
    remove_parser.add_argument(
        "-h",
        "--help",
        action=make_help_action(_lazy_ui_help("show_auth_help")),
    )

    status_parser = auth_sub.add_parser(
        "status",
        help="Show the credential resolution source for one or all providers",
        add_help=False,
    )
    status_parser.add_argument(
        "provider",
        nargs="?",
        default=None,
        help="Provider name; omit to show every provider",
    )
    status_parser.add_argument(
        "-h",
        "--help",
        action=make_help_action(_lazy_ui_help("show_auth_help")),
    )

    path_parser = auth_sub.add_parser(
        "path",
        help="Print the resolved auth.json path",
        add_help=False,
    )
    path_parser.add_argument(
        "-h",
        "--help",
        action=make_help_action(_lazy_ui_help("show_auth_help")),
    )


def run_auth_command(args: argparse.Namespace) -> int:
    """Dispatch a parsed `dcode auth` invocation.

    Returns:
        Process exit code (`0` on success, non-zero on error).
    """
    command = getattr(args, "auth_command", None)
    if command in {"list", "ls"}:
        return _run_list()
    if command == "set":
        return _run_set(args.provider, from_env=args.from_env)
    if command in {"remove", "rm", "delete"}:
        return _run_remove(args.provider)
    if command == "status":
        return _run_status(getattr(args, "provider", None))
    if command == "path":
        return _run_path()
    from deepagents_code.ui import show_auth_help

    show_auth_help()
    return 0


def _resolution_label(status: ProviderAuthStatus) -> str:
    """Render a provider's auth status as a plain-text resolution source.

    Mirrors the badges the `/auth` modal shows (`stored`, `env: VAR`,
    `missing`) so debugging "why isn't my key picked up?" is one command.

    Returns:
        A short, terminal-safe label describing where credentials resolve from.
    """
    from deepagents_code.model_config import (
        ProviderAuthSource,
        ProviderAuthState,
        resolved_env_var_name,
    )

    state = status.state
    if state is ProviderAuthState.CONFIGURED:
        if status.source is ProviderAuthSource.STORED:
            return "stored"
        if status.source is ProviderAuthSource.ENV and status.env_var:
            return f"env: {resolved_env_var_name(status.env_var)}"
        return "configured"
    if state is ProviderAuthState.MISSING:
        return "missing"
    if state is ProviderAuthState.NOT_REQUIRED:
        return status.detail or "no API key required"
    if state is ProviderAuthState.IMPLICIT:
        return status.detail or "implicit auth"
    if state is ProviderAuthState.MANAGED:
        return status.detail or "custom auth"
    # Catch-all for `ProviderAuthState.UNKNOWN` and any state added later, so a
    # new enum member degrades to a readable label rather than raising here.
    return status.detail or "credentials unknown"


def _known_providers() -> list[str]:
    """Return the providers shown by `list`/`status`, sorted.

    Matches the `/auth` manager's set: well-known providers whose integration
    package is installed, plus any provider with a stored credential or an
    `api_key_env` declared in `config.toml` (always shown so stale keys can be
    cleaned up and explicit declarations stay visible).

    Callers must invoke `_warn_if_store_unreadable` first: a corrupt store
    collapses the stored-credential arm to the empty set below (so a
    stored-only provider silently drops out of the listing), and only that
    warning tells the user the rows may be incomplete. The `logger.warning`
    here is invisible at the default CLI log level, so it is a debugging
    breadcrumb, not the user-facing signal.
    """
    from deepagents_code import auth_store
    from deepagents_code.model_config import (
        PROVIDER_API_KEY_ENV,
        ModelConfig,
        get_available_models,
    )

    try:
        stored = set(auth_store.list_configured_providers())
    except RuntimeError as exc:
        logger.warning("Failed to list stored credentials: %s", exc)
        stored = set()
    config = ModelConfig.load()
    config_providers = {
        name for name, cfg in config.providers.items() if cfg.get("api_key_env")
    }
    installed = set(get_available_models().keys())
    well_known_installed = set(PROVIDER_API_KEY_ENV) & installed
    return sorted(well_known_installed | stored | config_providers)


def _warn_if_store_unreadable() -> None:
    """Print a stderr warning when the credential store is present but corrupt.

    `get_provider_auth_status` (via `model_config._has_stored_credential`)
    swallows a corrupt-store `RuntimeError` and reports `missing`/env-only,
    which would otherwise make `list`/`status` silently misreport a provider
    whose stored key cannot be read. The TUI surfaces this with a banner in the
    `/auth` modal; this is the CLI equivalent so the printed rows are not taken
    as authoritative when the store is broken.
    """
    from deepagents_code import auth_store

    try:
        auth_store.list_configured_providers()
    except RuntimeError as exc:
        # Plain stderr diagnostic (not Rich-styled console output).
        print(f"Warning: {exc}", file=sys.stderr)  # noqa: T201


def _print_rows(providers: list[str]) -> None:
    """Print one `<provider>  <status>` row per provider, column-aligned."""
    from deepagents_code.model_config import get_provider_auth_status

    if not providers:
        return
    width = max(len(name) for name in providers)
    for provider in providers:
        label = _resolution_label(get_provider_auth_status(provider))
        # Plain stdout so rows stay greppable/pipeable, not Rich-styled.
        print(f"{provider.ljust(width)}  {label}")  # noqa: T201


def _run_list() -> int:
    """Print every known provider and its credential status.

    Returns:
        Process exit code (`0`).
    """
    _warn_if_store_unreadable()
    providers = _known_providers()
    if not providers:
        print("No providers found.")  # noqa: T201
        return 0
    _print_rows(providers)
    return 0


def _run_status(provider: str | None) -> int:
    """Print the resolution source for one provider or all of them.

    Returns:
        Process exit code (`0`).
    """
    _warn_if_store_unreadable()
    providers = [provider] if provider else _known_providers()
    if not providers:
        print("No providers found.")  # noqa: T201
        return 0
    _print_rows(providers)
    return 0


def _run_set(provider: str, *, from_env: str | None) -> int:
    """Store an API key for `provider`, reading it from env or stdin.

    Returns:
        Process exit code (`0` on success, `1` on a recoverable input error).
    """
    import os

    from deepagents_code import auth_store

    if from_env is not None:
        key = os.environ.get(from_env)
        if not key or not key.strip():
            print(  # noqa: T201
                f"Error: environment variable {from_env} is not set or is empty.",
                file=sys.stderr,
            )
            return 1
    else:
        if sys.stdin.isatty():
            print(  # noqa: T201
                "Error: refusing to read an API key from an interactive terminal.\n"
                f"Pipe the key via stdin (e.g. `echo $KEY | dcode auth set {provider}`)"
                " or use --from-env VAR.",
                file=sys.stderr,
            )
            return 1
        key = sys.stdin.read()
        if not key.strip():
            # Mirror the `--from-env` empty-var message so an empty pipe gives a
            # specific, actionable error here rather than the generic
            # "API key cannot be empty" `ValueError` raised later by
            # `set_stored_key`. The key value is never echoed.
            print(  # noqa: T201
                "Error: no API key received on stdin.",
                file=sys.stderr,
            )
            return 1

    try:
        outcome = auth_store.set_stored_key(
            provider, key, base_url=auth_store.get_stored_base_url(provider)
        )
    except (ValueError, RuntimeError) as exc:
        # `auth_store` messages never include the secret value. `ValueError`
        # carries a short fragment, `RuntimeError` a full sentence with a hint;
        # print verbatim so the two stay consistent and free of double periods.
        print(f"Error: {exc}", file=sys.stderr)  # noqa: T201
        return 1

    for warning in outcome.warnings:
        print(f"Warning: {warning}", file=sys.stderr)  # noqa: T201
    print(f"Stored credential for {provider}.")  # noqa: T201
    return 0


def _run_remove(provider: str) -> int:
    """Delete a stored credential for `provider`.

    Returns:
        Process exit code (`0`).
    """
    from deepagents_code import auth_store

    try:
        removed = auth_store.delete_stored_key(provider)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)  # noqa: T201
        return 1
    if removed:
        print(f"Removed stored credential for {provider}.")  # noqa: T201
    else:
        print(f"No stored credential for {provider}.")  # noqa: T201
    return 0


def _run_path() -> int:
    """Print the resolved `auth.json` path.

    Returns:
        Process exit code (`0`).
    """
    from deepagents_code import auth_store

    print(auth_store.auth_path())  # noqa: T201
    return 0
