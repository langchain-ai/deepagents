"""CLI commands for the `config` group: inspect the configuration surface.

`config show` (aliased as `config list`/`ls`) resolves each option against the
app credential store (for credentials), the live environment, and `config.toml`,
reporting the effective value and which source provided it, matching what
`git config --list` / `aws configure list` users expect. Adding `--verbose`/`--all`
folds in each option's description and where it can be set (the static catalog).
`config get <key>` does the same for a single option. `config path` prints the
on-disk config locations.

Secret-flagged options (API keys and other credentials) are never printed by
value — `config show`/`config get` report only whether they are set and from
which source, so the output is safe to paste into a bug report.

Help rendering for a bare `config` invocation is served by `ui.show_config_help`,
which does not import this module. The heavy manifest/runtime imports here are
function-local to the subcommands, so a bare `config`/`config -h` invocation
never pulls them onto the startup path (`parse_args` does import this module to
register the subparsers, but only its light top-level imports run then).
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from deepagents_code.output import write_json

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable, Sequence

    from deepagents_code.config_manifest import ConfigOption
    from deepagents_code.output import OutputFormat

logger = logging.getLogger(__name__)


def _lazy_ui_help(fn_name: str) -> Callable[[], None]:
    """Return a callable that lazily imports and invokes a `ui` help function."""

    def _show() -> None:
        from deepagents_code import ui

        getattr(ui, fn_name)()

    return _show


def setup_config_parser(
    subparsers: Any,  # noqa: ANN401
    *,
    make_help_action: Callable[[Callable[[], None]], type[argparse.Action]],
    add_output_args: Callable[..., None],
) -> None:
    """Register the `dcode config` command group.

    Args:
        subparsers: The `argparse` subparsers object from the top-level CLI
            parser, onto which the `config` command group is attached.
        make_help_action: Factory that wraps a `show_*` callable into an
            `argparse.Action` so `-h/--help` renders the hand-maintained
            help screens from `deepagents_code.ui`.
        add_output_args: Helper that adds the shared `--json` flag.
    """
    config_parser = subparsers.add_parser(
        "config",
        help="Inspect configuration options and their sources",
        add_help=False,
    )
    config_parser.add_argument(
        "-h",
        "--help",
        action=make_help_action(_lazy_ui_help("show_config_help")),
    )
    add_output_args(config_parser)
    config_sub = config_parser.add_subparsers(dest="config_command")

    def _add_verbose_arg(parser: Any) -> None:  # noqa: ANN401
        parser.add_argument(
            "-v",
            "--verbose",
            "--all",
            dest="verbose",
            action="store_true",
            help="Also show each option's description and where to set it",
        )

    show_parser = config_sub.add_parser(
        "show",
        aliases=["list", "ls"],
        help="Show effective config values and their source",
        add_help=False,
    )
    show_parser.add_argument(
        "-h",
        "--help",
        action=make_help_action(_lazy_ui_help("show_config_help")),
    )
    _add_verbose_arg(show_parser)
    add_output_args(show_parser)

    get_parser = config_sub.add_parser(
        "get",
        help="Show the effective value and source for one option",
        add_help=False,
    )
    get_parser.add_argument("key", help="Option key (e.g. interpreter.memory_limit_mb)")
    get_parser.add_argument(
        "-h",
        "--help",
        action=make_help_action(_lazy_ui_help("show_config_help")),
    )
    add_output_args(get_parser)

    path_parser = config_sub.add_parser(
        "path",
        help="Show config file locations",
        add_help=False,
    )
    path_parser.add_argument(
        "-h",
        "--help",
        action=make_help_action(_lazy_ui_help("show_config_help")),
    )
    add_output_args(path_parser)


# --- Resolution -------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _StoredCredentialView:
    """Snapshot of the `/auth` credential store for one command invocation.

    Built once per `config show`/`get` so the store is read and parsed a single
    time rather than once per credential option.
    """

    keys: dict[str, str] = field(repr=False)
    """Provider/service name to stored API key, for `api_key` entries only.

    `repr=False` keeps the secret key values out of the dataclass repr, so an
    accidental log/`%r` of the view can't leak them.
    """

    error: str | None = None
    """Secret-free remediation message when the store was unreadable, else `None`.

    Never holds the underlying exception text (which can echo file bytes) or any
    key value.
    """


_STORE_UNREADABLE_HINT = (
    "credential store unreadable; showing env/config.toml resolution instead. "
    "Re-add keys via /auth (or delete a corrupt auth.json)."
)
"""Fixed, secret-free notice surfaced when `auth.json` cannot be read."""


def _load_stored_credentials() -> _StoredCredentialView:
    """Read every `/auth`-stored API key once, degrading a corrupt store to empty.

    Reading the store a single time (rather than once per credential option)
    keeps `config show` to one `auth.json` parse and one warning. A corrupt
    store is logged once and reported via the returned `error`, so resolution
    degrades to env/`config.toml` instead of failing the command — and the
    corruption stays visible in the output rather than masquerading as an empty
    store.

    Returns:
        A `_StoredCredentialView` whose `keys` map holds stored `api_key` values
        by provider, and whose `error` is set only when the store was unreadable.
    """
    from deepagents_code import auth_store

    try:
        creds = auth_store.load_credentials()
    except RuntimeError:
        # Omit the exception text on purpose: it can echo file contents, and the
        # remediation is identical regardless of the specific parse failure.
        logger.warning("Could not read stored credentials; treating as absent")
        return _StoredCredentialView(keys={}, error=_STORE_UNREADABLE_HINT)
    keys = {
        provider: entry["key"]
        for provider, entry in creds.items()
        if entry["type"] == "api_key" and entry["key"]
    }
    return _StoredCredentialView(keys=keys)


def _resolve(
    option: ConfigOption,
    toml_data: dict[str, Any],
    *,
    stored: _StoredCredentialView | None = None,
) -> tuple[bool, str, Any]:
    """Resolve an option for display, reporting what the runtime actually reads.

    Credential options follow runtime precedence: a present `DEEPAGENTS_CODE_`
    env override wins (the model factory reads it via `resolve_env_var` even
    after `apply_stored_credentials` bridges a stored key onto the canonical
    var), then a key stored via `/auth`, then the canonical env/`config.toml`.
    Everything else delegates straight to `config_manifest.resolve_scalar`.

    Args:
        option: The option to resolve.
        toml_data: Parsed `config.toml` contents.
        stored: Pre-loaded credential-store snapshot. When `None`, the store is
            read on demand — fine for one-off calls, but callers resolving many
            options should load it once and pass it so `auth.json` is parsed a
            single time.

    Returns:
        `(is_set, source, value)`, where `is_set` is `False` when the value
        came from the typed default.
    """
    from deepagents_code.config_manifest import resolve_scalar
    from deepagents_code.model_config import ProviderAuthSource

    if (
        option.group == "Credentials"
        and option.provider is not None
        and not _has_prefixed_env_override(option)
    ):
        if stored is None:
            stored = _load_stored_credentials()
        key = stored.keys.get(option.provider)
        if key is not None:
            return True, ProviderAuthSource.STORED.value, key

    value, source = resolve_scalar(option, toml_data=toml_data)
    return source != "default", source, value


def _has_prefixed_env_override(option: ConfigOption) -> bool:
    """Return whether an option's `DEEPAGENTS_CODE_` env var is present."""
    if option.env_var is None:
        return False
    prefix = "DEEPAGENTS_CODE_"
    if option.env_var.startswith(prefix):
        return False
    return f"{prefix}{option.env_var}" in os.environ


def _display_value(option: ConfigOption, *, is_set: bool, value: object) -> str:
    """Render an option value for human output, redacting secrets.

    Returns:
        `configured`/`not configured` for credential options, otherwise the value
            as text.
    """
    if option.group == "Credentials":
        if value is None:
            return _with_availability(option, "not configured")
        if option.redacted:
            status = "configured" if is_set else "not configured"
            return _with_availability(option, status)
    if value is None:
        return "(unset)"
    if option.key == "display.charset" and value == "auto":
        return _charset_display_value()
    text = str(value)
    if option.group == "Credentials":
        text = _with_availability(option, text)
    max_len = 60
    if len(text) > max_len:
        return text[: max_len - 1] + "\N{HORIZONTAL ELLIPSIS}"
    return text


def _source_label(source: str, *, option: ConfigOption | None = None) -> str:
    """Render the source column for human output.

    Returns:
        Source label for the value's origin.
    """
    if option is not None and option.group == "Credentials":
        env = _env_source_name(source)
        if env is not None and env.startswith("DEEPAGENTS_CODE_"):
            return f"{source}; session override"
    return source


def _env_source_name(source: str) -> str | None:
    """Return the env var name from an `env (...)` source label, if present."""
    prefix = "env ("
    if not source.startswith(prefix) or not source.endswith(")"):
        return None
    return source[len(prefix) : -1]


def _with_availability(option: ConfigOption, text: str) -> str:
    """Append provider availability to a credential display value when needed.

    Returns:
        Display text with `, unavailable` appended when the provider integration
        package is missing.
    """
    if _missing_extra_hint(option):
        return f"{text}, unavailable"
    return text


def _charset_display_value() -> str:
    """Return the `display.charset=auto` value with its effective glyph mode."""
    from deepagents_code.config import _detect_charset_mode

    mode = _detect_charset_mode().value
    label = "Unicode" if mode == "unicode" else "ASCII"
    return f"auto (using {label} glyphs)"


def _missing_extra_hint(option: ConfigOption) -> bool:
    """Return whether a credential option's provider integration is unavailable."""
    if option.group != "Credentials" or option.dependency_module is None:
        return False
    return importlib.util.find_spec(option.dependency_module) is None


# --- Commands ---------------------------------------------------------------


def _show_json_row(
    option: ConfigOption,
    *,
    is_set: bool,
    source: str,
    value: object,
    store_error: str | None,
    include_catalog: bool,
) -> dict[str, Any]:
    """Build one show/list --json row, redacting secrets and flagging store errors.

    Returns:
        A JSON-serializable row. Redacted options report presence only (`value`
            is `None`); a `store_error` key is added to credential rows when
            the `/auth` store was unreadable, so a corrupt store is
            distinguishable from an empty one in the bug-report artifact. When
            `include_catalog` is set the static catalog fields (summary, type,
            default, ...) are folded in so `config list --json` consumers stay
            unbroken.
    """
    row: dict[str, Any] = {
        "key": option.key,
        "group": option.group,
        "source": source,
        "set": is_set,
        "redacted": option.redacted,
        # Redact secret values: report presence only.
        "value": None if option.redacted else value,
    }
    if include_catalog:
        row.update(
            {
                "summary": option.summary,
                "type": option.type,
                "default": option.default,
                "env_var": option.env_var,
                "toml_path": option.toml_path,
                "cli_flag": option.cli_flag,
            }
        )
    if store_error and option.group == "Credentials":
        row["store_error"] = store_error
    return row


def _run_show(output_format: OutputFormat, *, verbose: bool, command_label: str) -> int:
    """Resolve every option and print its effective value and source.

    With `verbose`, each option also lists its description and where it can be
    set (the catalog detail formerly served by `config list`).

    Returns:
        Process exit code (`0` on success).
    """
    from deepagents_code.config import _ensure_bootstrap
    from deepagents_code.config_manifest import get_config_options, load_config_toml

    # Load `.env` files into the environment so resolution reflects what the
    # app actually reads, not just shell exports.
    _ensure_bootstrap()
    toml_data = load_config_toml()
    # Read the credential store once; `_resolve` reuses this snapshot rather than
    # re-parsing `auth.json` per credential option.
    stored = _load_stored_credentials()

    options = get_config_options()
    resolved = [(opt, *_resolve(opt, toml_data, stored=stored)) for opt in options]

    if output_format == "json":
        # `config list --json` was the catalog endpoint; keep its catalog fields
        # so existing consumers stay unbroken (now additive alongside effective
        # values). `config show --json` stays effective-only unless `--verbose`.
        include_catalog = verbose or command_label == "config list"
        write_json(
            command_label,
            [
                _show_json_row(
                    opt,
                    is_set=is_set,
                    source=source,
                    value=value,
                    store_error=stored.error,
                    include_catalog=include_catalog,
                )
                for opt, is_set, source, value in resolved
            ],
        )
        return 0

    if verbose:
        _print_show_verbose(resolved, options, store_error=stored.error)
    else:
        _print_show_table(resolved, options, store_error=stored.error)
    return 0


def _print_store_warning(store_error: str | None) -> None:
    """Print a warning when the `/auth` credential store was unreadable."""
    if not store_error:
        return
    from rich.markup import escape

    from deepagents_code.config import console

    console.print(f"[yellow]Warning:[/yellow] {escape(store_error)}", highlight=False)
    console.print()


def _print_show_table(
    resolved: Sequence[tuple[ConfigOption, bool, str, Any]],
    options: Sequence[ConfigOption],
    *,
    store_error: str | None = None,
) -> None:
    """Render the compact effective-value table, grouped by section."""
    from rich.table import Table
    from rich.text import Text

    from deepagents_code.config import console
    from deepagents_code.config_manifest import iter_groups

    console.print()
    _print_store_warning(store_error)
    for group in iter_groups(options):
        console.print(f"[bold]{group}[/bold]")
        table = Table.grid(padding=(0, 2))
        table.add_column()
        table.add_column()
        table.add_column(style="dim")
        for opt, is_set, source, value in resolved:
            if opt.group != group:
                continue
            display = _display_value(opt, is_set=is_set, value=value)
            # `display`/`source` may contain markup from env/TOML; `Text` cells
            # render literally, so values can't break the table.
            table.add_row(
                Text(f"  {opt.key}"),
                Text(display),
                Text(_source_label(source, option=opt)),
            )
        console.print(table, highlight=False)
        console.print()


def _print_show_verbose(
    resolved: Sequence[tuple[ConfigOption, bool, str, Any]],
    options: Sequence[ConfigOption],
    *,
    store_error: str | None = None,
) -> None:
    """Render the effective value plus description and how-to-set per option."""
    from rich.markup import escape

    from deepagents_code.config import console
    from deepagents_code.config_manifest import iter_groups

    console.print()
    _print_store_warning(store_error)
    for group in iter_groups(options):
        console.print(f"[bold]{group}[/bold]")
        for opt, is_set, source, value in resolved:
            if opt.group != group:
                continue
            display = _display_value(opt, is_set=is_set, value=value)
            # `display`/`source` may carry markup from env/TOML; escape them.
            console.print(
                f"  [cyan]{opt.key}[/cyan]  {escape(display)}  "
                f"[dim]{escape(_source_label(source, option=opt))}[/dim]",
                highlight=False,
            )
            console.print(f"    {opt.summary}", highlight=False, style="dim")
            console.print(f"    {_sources_line(opt)}", highlight=False, style="dim")
        console.print()


def _run_get(key: str, output_format: OutputFormat) -> int:
    """Resolve and print a single option by key.

    Returns:
        Process exit code (`0` on success, `1` for an unknown key).
    """
    from deepagents_code.config_manifest import get_option

    option = get_option(key)
    if option is None:
        if output_format == "json":
            write_json("config get", {"key": key, "error": "unknown option"})
        else:
            print(  # noqa: T201
                f"Unknown config option: {key!r}. Run `dcode config list` to "
                "see available keys.",
                file=sys.stderr,
            )
        return 1

    from deepagents_code.config import _ensure_bootstrap
    from deepagents_code.config_manifest import load_config_toml

    _ensure_bootstrap()
    toml_data = load_config_toml()
    # Only credential options consult the store, so skip the read (and its
    # warning) for everything else.
    stored = _load_stored_credentials() if option.group == "Credentials" else None
    is_set, source, value = _resolve(option, toml_data, stored=stored)
    store_error = stored.error if stored is not None else None

    if output_format == "json":
        payload: dict[str, Any] = {
            "key": option.key,
            "source": source,
            "set": is_set,
            "redacted": option.redacted,
            "value": None if option.redacted else value,
        }
        if store_error:
            payload["store_error"] = store_error
        write_json("config get", payload)
        return 0

    from rich.markup import escape

    from deepagents_code.config import console

    display = _display_value(option, is_set=is_set, value=value)
    source_label = _source_label(source, option=option)
    console.print(
        f"{option.key} = {escape(display)}  [dim]({escape(source_label)})[/dim]",
        highlight=False,
    )
    if store_error:
        console.print(
            f"[yellow]Warning:[/yellow] {escape(store_error)}", highlight=False
        )
    return 0


def _run_path(output_format: OutputFormat) -> int:
    """Print the on-disk config file locations and whether they exist.

    Returns:
        Process exit code (`0` on success).
    """
    paths = _config_paths()

    if output_format == "json":
        write_json(
            "config path",
            [
                {"label": label, "path": str(path), "exists": exists}
                for label, path, exists in paths
            ],
        )
        return 0

    from deepagents_code.config import console

    console.print()
    console.print("[bold]Config locations[/bold]")
    for label, path, exists in paths:
        marker = "[green]exists[/green]" if exists else "[dim]missing[/dim]"
        console.print(f"  {label:<22} {path}  ({marker})", highlight=False)
    console.print()
    return 0


def run_config_command(args: argparse.Namespace) -> int:
    """Dispatch a parsed `config` subcommand.

    Returns:
        Process exit code from the dispatched subcommand.
    """
    output_format: OutputFormat = getattr(args, "output_format", "text")
    command = getattr(args, "config_command", None)
    verbose: bool = getattr(args, "verbose", False)

    if command in {"show", "list", "ls"}:
        label = "config list" if command in {"list", "ls"} else "config show"
        return _run_show(output_format, verbose=verbose, command_label=label)
    if command == "get":
        return _run_get(args.key, output_format)
    if command == "path":
        return _run_path(output_format)

    from deepagents_code.ui import show_config_help

    show_config_help()
    return 0


# --- Helpers ----------------------------------------------------------------


def _sources_line(option: ConfigOption) -> str:
    """Render a compact 'set via' line for `config list`.

    Returns:
        A human-readable description of where the option can be set.
    """
    parts: list[str] = []
    if option.env_var:
        parts.append(f"env {option.env_var}")
    if option.toml_path:
        parts.append(f"toml {option.toml_path}")
    if option.cli_flag:
        parts.append(f"cli {option.cli_flag}")
    default = f"default {option.default}" if option.default is not None else ""
    set_via = "set via " + ", ".join(parts) if parts else "managed by the app"
    return f"{set_via}{('  |  ' + default) if default else ''}"


def _config_paths() -> list[tuple[str, Any, bool]]:
    """Collect known config file locations and whether each exists.

    Returns:
        A list of `(label, path, exists)` rows in display order.
    """
    from pathlib import Path

    from deepagents_code.config import _GLOBAL_DOTENV_PATH, _find_dotenv_from_start_path
    from deepagents_code.model_config import (
        DEFAULT_CONFIG_PATH,
        DEFAULT_STATE_DIR,
        RECENT_MODELS_FILENAME,
    )

    base = DEFAULT_CONFIG_PATH.parent
    project_dotenv = _find_dotenv_from_start_path(Path.cwd())

    candidates: list[tuple[str, Path | None]] = [
        ("config.toml", DEFAULT_CONFIG_PATH),
        ("project .env", project_dotenv),
        ("global .env", _GLOBAL_DOTENV_PATH),
        ("hooks.json", base / "hooks.json"),
        ("auth.json", DEFAULT_STATE_DIR / "auth.json"),
        ("recent models", DEFAULT_STATE_DIR / RECENT_MODELS_FILENAME),
    ]

    from deepagents_code._paths import PathState, classify_path

    rows: list[tuple[str, Any, bool]] = []
    for label, path in candidates:
        if path is None:
            continue
        # `classify_path` logs unreadable paths at debug level. `config path`
        # reports a plain exists/missing bool, so unreadable still collapses to
        # missing here while `doctor` can surface it as a problem.
        exists = classify_path(path) is PathState.EXISTS
        rows.append((label, path, exists))
    return rows
