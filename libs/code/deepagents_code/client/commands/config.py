"""CLI commands for inspecting the configuration surface.

Bare `config` resolves each option against the app credential store (for
credentials), the live environment, and `config.toml`, reporting the effective
value and which source provided it. `config get <key>` reports the same for a
single option, and `config get <section>` — a dotted key prefix such as
`credentials`, matched case-insensitively — renders every option under that
prefix using the same grouped view as bare
`config`. Adding `--verbose`/`--all` to any of the three folds in each option's
description and where it can be set (the static catalog). `config path` prints
the on-disk config locations.

Secret-flagged options (API keys and other credentials) are never printed by
value — `config`/`config get` report only whether they are set and from which
source, so the output is safe to paste into a bug report.

Help rendering for `config -h` is served by `ui.show_config_help`. The heavy
manifest/runtime imports here are function-local, so help never pulls them onto
the startup path (`parse_args` imports this module to register the parsers, but
only its light top-level imports run then).
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NamedTuple

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
    from argparse import SUPPRESS

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
    config_parser.add_argument(
        "-v",
        "--verbose",
        "--all",
        dest="verbose",
        action="store_true",
        help="Also show each option's description and where to set it",
    )
    config_sub = config_parser.add_subparsers(dest="config_command")

    get_parser = config_sub.add_parser(
        "get",
        help="Show the effective value and source for one option or section",
        add_help=False,
    )
    # Optional so a bare `config get` reaches our handler with a useful hint
    # (available keys + examples) instead of argparse's terse "the following
    # arguments are required: key".
    get_parser.add_argument(
        "key",
        nargs="?",
        default=None,
        help=(
            "Option key (e.g. interpreter.memory_limit_mb) or key prefix "
            "(e.g. credentials)"
        ),
    )
    get_parser.add_argument(
        "-h",
        "--help",
        action=make_help_action(_lazy_ui_help("show_config_help")),
    )
    # `SUPPRESS` so `config --verbose get <key>` keeps the parent value: argparse
    # copies every key of the subparser namespace over the parent's, so a
    # `store_true` default here would reset it to `False`.
    get_parser.add_argument(
        "-v",
        "--verbose",
        "--all",
        dest="verbose",
        action="store_true",
        default=SUPPRESS,
        help="Also show each option's description and how to set it",
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

    Built once per `config`/`config get` so the store is read and parsed a single
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
    keeps `config` to one `auth.json` parse and one warning. A corrupt
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
) -> tuple[bool, str, object]:
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


class ResolvedOption(NamedTuple):
    """An option paired with its resolved effective value, for display.

    Bundles the four values that always travel together through the render
    helpers as one named record, so they can't be reordered or misaligned at a
    call site the way a bare positional tuple can.
    """

    option: ConfigOption
    """The option being described."""

    is_set: bool
    """`False` when `value` came from the option's typed default."""

    source: str
    """Where the effective value came from (e.g. `env (...)`, `stored`, `default`)."""

    value: object
    """The effective value; `None` when unset.

    Redacted for secrets before display.
    """


# --- Commands ---------------------------------------------------------------


def _catalog_fields(option: ConfigOption) -> dict[str, Any]:
    """Return the static catalog fields `--verbose` folds into a JSON payload.

    Shared by the bare-`config`/section rows and the single-key payload so the
    two shapes cannot drift apart.

    Returns:
        The option's description and where it can be set.
    """
    return {
        "summary": option.summary,
        "type": option.type,
        "default": option.default,
        "env_var": option.env_var,
        "toml_path": option.toml_path,
        "cli_flag": option.cli_flag,
    }


def _config_json_row(
    option: ConfigOption,
    *,
    is_set: bool,
    source: str,
    value: object,
    store_error: str | None,
    include_catalog: bool,
) -> dict[str, Any]:
    """Build one `config --json` row, redacting secrets and flagging errors.

    Returns:
        A JSON-serializable row. Redacted options report presence only (`value`
            is `None`); a `store_error` key is added to credential rows when
            the `/auth` store was unreadable, so a corrupt store is
            distinguishable from an empty one in the bug-report artifact. When
            `include_catalog` is set (i.e. `--verbose`) the static catalog
            fields (summary, type, default, ...) are folded in.
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
        row.update(_catalog_fields(option))
    if store_error and option.group == "Credentials":
        row["store_error"] = store_error
    return row


def _run_config(output_format: OutputFormat, *, verbose: bool) -> int:
    """Resolve every option and print its effective value and source.

    With `verbose`, each option also lists its description and where it can be
    set (the static catalog detail).

    Args:
        output_format: `text` for the rendered view, `json` for a machine-
            readable payload.
        verbose: Fold each option's description and how-to-set into the output.

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

    resolved = [
        ResolvedOption(opt, *_resolve(opt, toml_data, stored=stored))
        for opt in get_config_options()
    ]

    if output_format == "json":
        # `config --json` stays effective-only unless `--verbose` folds in
        # the static catalog fields.
        include_catalog = verbose
        write_json(
            "config",
            [
                _config_json_row(
                    row.option,
                    is_set=row.is_set,
                    source=row.source,
                    value=row.value,
                    store_error=stored.error,
                    include_catalog=include_catalog,
                )
                for row in resolved
            ],
        )
        return 0

    if verbose:
        _print_config_verbose(resolved, store_error=stored.error)
    else:
        _print_config_table(resolved, store_error=stored.error)
    return 0


def _print_store_warning(store_error: str | None) -> None:
    """Print a warning when the `/auth` credential store was unreadable."""
    if not store_error:
        return
    from rich.markup import escape

    from deepagents_code.config import console

    console.print(f"[yellow]Warning:[/yellow] {escape(store_error)}", highlight=False)
    console.print()


def _print_config_table(
    resolved: Sequence[ResolvedOption],
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
    groups = list(iter_groups(row.option for row in resolved))
    for index, group in enumerate(groups):
        if index:
            # Blank line separates groups but never trails the last one, so the
            # output ends on content rather than an empty line.
            console.print()
        console.print(f"[bold]{group}[/bold]")
        table = Table.grid(padding=(0, 2))
        table.add_column()
        table.add_column()
        table.add_column(style="dim")
        for row in resolved:
            if row.option.group != group:
                continue
            display = _display_value(row.option, is_set=row.is_set, value=row.value)
            # `display`/`source` may contain markup from env/TOML; `Text` cells
            # render literally, so values can't break the table.
            table.add_row(
                Text(f"  {row.option.key}"),
                Text(display),
                Text(_source_label(row.source, option=row.option)),
            )
        console.print(table, highlight=False)


def _print_config_verbose(
    resolved: Sequence[ResolvedOption],
    *,
    store_error: str | None = None,
) -> None:
    """Render the effective value plus description and how-to-set per option."""
    from rich.markup import escape

    from deepagents_code.config import console
    from deepagents_code.config_manifest import iter_groups

    console.print()
    _print_store_warning(store_error)
    groups = list(iter_groups(row.option for row in resolved))
    for index, group in enumerate(groups):
        if index:
            console.print()
        console.print(f"[bold]{group}[/bold]")
        for row in resolved:
            if row.option.group != group:
                continue
            display = _display_value(row.option, is_set=row.is_set, value=row.value)
            # `display`/`source` may carry markup from env/TOML; escape them.
            console.print(
                f"  [cyan]{row.option.key}[/cyan]  {escape(display)}  "
                f"[dim]{escape(_source_label(row.source, option=row.option))}[/dim]",
                highlight=False,
            )
            console.print(f"    {row.option.summary}", highlight=False, style="dim")
            console.print(
                f"    {_sources_line(row.option)}", highlight=False, style="dim"
            )


_GET_KEY_EXAMPLE = "interpreter.memory_limit_mb"
"""Illustrative key shown in the missing-key hint.

A unit test asserts this stays a real manifest key so the hint never points at a
key that `config get` would reject.
"""


def _report_missing_get_key(output_format: OutputFormat) -> int:
    """Explain that `config get` needs a key, and point at how to find one.

    Reached when the user runs a bare `config get` (the `key` positional is
    optional so this handler can render a useful hint instead of argparse's
    terse usage error).

    Returns:
        Exit code `2`, matching argparse's convention for a usage error so
        existing scripts see the same code they did before.
    """
    from deepagents_code.config_manifest import option_keys

    if output_format == "json":
        write_json(
            "config get",
            {"error": "missing key", "keys": list(option_keys())},
        )
        return 2

    print(  # noqa: T201
        f"`dcode config get` needs an option key, e.g. `dcode config get "
        f"{_GET_KEY_EXAMPLE}`. Run `dcode config` to list options and their "
        "effective values, or `dcode config --verbose` to see every key.",
        file=sys.stderr,
    )
    return 2


class _Selection(NamedTuple):
    """What a `config get` argument resolved to.

    Attributes:
        options: The named options, in manifest order.
        is_exact: Whether the argument was a full option key. A section can hold
            exactly one option, so the caller cannot infer this from `options`
            alone — hence carrying it rather than re-deriving it.
    """

    options: tuple[ConfigOption, ...]
    is_exact: bool


def _select_options(key: str) -> _Selection | None:
    """Resolve a `config get` argument to the options it names.

    Priority is exact key, then dotted key prefix (`credentials`, with an
    optional trailing dot, case-insensitively). Display group titles are
    deliberately not accepted: `Models` and `Tools` name a different set of
    options than the same word as a prefix, so titles would make one argument
    resolve to two different sections depending on which tier matched. Key
    prefixes are the single section namespace.

    Args:
        key: The raw argument passed to `config get`.

    Returns:
        The matched options and whether the match was an exact key, or `None`
        when nothing matches.
    """
    from deepagents_code.config_manifest import get_option, options_with_key_prefix

    exact = get_option(key)
    if exact is not None:
        return _Selection((exact,), is_exact=True)

    matched = options_with_key_prefix(key.removesuffix("."))
    return _Selection(matched, is_exact=False) if matched else None


def _report_unknown_get_key(key: str, output_format: OutputFormat) -> int:
    """Report that `key` names neither an option nor a section.

    Returns:
        Exit code `1`.
    """
    if output_format == "json":
        write_json("config get", {"key": key, "error": "unknown option or section"})
    else:
        print(  # noqa: T201
            f"Unknown config option or section: {key!r}. Run "
            "`dcode config` to list keys, then "
            "`dcode config get <key>` or `dcode config get <section>`.",
            file=sys.stderr,
        )
    return 1


def _run_get_section(
    options: Sequence[ConfigOption], output_format: OutputFormat, *, verbose: bool
) -> int:
    """Resolve and print every option in a matched section.

    Redaction and source resolution reuse `_config_json_row`/`_resolve` and the
    grouped table/verbose printers rather than reimplementing them, so a section
    renders exactly as the same rows do under bare `config`. JSON is always a
    list here — even for a one-option section — so consumers can tell a section
    response from the single-key object.

    Args:
        options: The section's options, in manifest order.
        output_format: `text` for the rendered view, `json` for a machine-
            readable payload.
        verbose: Fold each option's description and how-to-set into the output.

    Returns:
        Process exit code (`0` on success).
    """
    from deepagents_code.config import _ensure_bootstrap
    from deepagents_code.config_manifest import load_config_toml

    _ensure_bootstrap()
    toml_data = load_config_toml()
    # Only credential options consult the store, so skip the read (and its
    # warning) when the section holds none.
    stored = (
        _load_stored_credentials()
        if any(opt.group == "Credentials" for opt in options)
        else None
    )
    resolved = [
        ResolvedOption(opt, *_resolve(opt, toml_data, stored=stored)) for opt in options
    ]
    store_error = stored.error if stored is not None else None

    if output_format == "json":
        write_json(
            "config get",
            [
                _config_json_row(
                    row.option,
                    is_set=row.is_set,
                    source=row.source,
                    value=row.value,
                    store_error=store_error,
                    include_catalog=verbose,
                )
                for row in resolved
            ],
        )
        return 0

    if verbose:
        _print_config_verbose(resolved, store_error=store_error)
    else:
        _print_config_table(resolved, store_error=store_error)
    return 0


def _run_get(
    key: str | None, output_format: OutputFormat, *, verbose: bool = False
) -> int:
    """Resolve and print one option, or every option in a section.

    Args:
        key: Option key, dotted key prefix, or display group title.
        output_format: `text` for the rendered view, `json` for a machine-
            readable payload.
        verbose: Fold the option's description and how-to-set into the output,
            matching `config --verbose`. For an exact key this adds lines/fields
            to the existing payload; the single-key JSON stays an object.

    Returns:
        Process exit code (`0` on success, `1` for an unknown key or section,
        `2` when no key was given).
    """
    if key is None:
        return _report_missing_get_key(output_format)

    selection = _select_options(key)
    if selection is None:
        return _report_unknown_get_key(key, output_format)
    if not selection.is_exact:
        return _run_get_section(selection.options, output_format, verbose=verbose)

    option = selection.options[0]

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
        # No `group` key even under `--verbose`: an object without one is how
        # consumers tell a single-key response from a section list.
        if verbose:
            payload.update(_catalog_fields(option))
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
    if verbose:
        # Static manifest text, so no markup escaping needed — same as the
        # detail lines `_print_config_verbose` emits.
        console.print(f"  {option.summary}", highlight=False, style="dim")
        console.print(f"  {_sources_line(option)}", highlight=False, style="dim")
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
    """Dispatch a parsed `config` invocation.

    Returns:
        Process exit code from the selected config action.
    """
    output_format: OutputFormat = getattr(args, "output_format", "text")
    command = getattr(args, "config_command", None)
    verbose: bool = getattr(args, "verbose", False)

    if command is None:
        return _run_config(output_format, verbose=verbose)
    if command == "get":
        return _run_get(args.key, output_format, verbose=verbose)
    if command == "path":
        return _run_path(output_format)

    from deepagents_code.ui import show_config_help

    show_config_help()
    return 0


# --- Helpers ----------------------------------------------------------------


def _sources_line(option: ConfigOption) -> str:
    """Render a compact 'set via' line for the verbose (`--verbose`) view.

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
    from deepagents_code.hooks.loading import project_hooks_path
    from deepagents_code.model_config import (
        DEFAULT_CONFIG_PATH,
        DEFAULT_STATE_DIR,
        RECENT_MODELS_FILENAME,
    )
    from deepagents_code.project_utils import ProjectContext

    base = DEFAULT_CONFIG_PATH.parent
    project_dotenv = _find_dotenv_from_start_path(Path.cwd())
    project_context = ProjectContext.from_user_cwd(Path.cwd())
    project_root = project_context.project_root or project_context.user_cwd

    candidates: list[tuple[str, Path | None]] = [
        ("config.toml", DEFAULT_CONFIG_PATH),
        ("project .env", project_dotenv),
        ("global .env", _GLOBAL_DOTENV_PATH),
        ("project hooks.json", project_hooks_path(project_root)),
        ("user hooks.json", base / "hooks.json"),
        ("hooks trust", DEFAULT_STATE_DIR / "hooks_trust.json"),
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
