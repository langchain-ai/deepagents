"""CLI commands for the `config` group: inspect the configuration surface.

`config list` prints the static manifest (every tunable option, its type,
default, and where it can be set). `config show` resolves each option against
the live environment and `config.toml`, reporting the effective value and which
source provided it. `config get <key>` does the same for a single option.
`config path` prints the on-disk config locations.

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
import sys
from typing import TYPE_CHECKING, Any

from deepagents_code.output import write_json

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable

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

    show_parser = config_sub.add_parser(
        "show",
        help="Show effective config values and their source",
        add_help=False,
    )
    show_parser.add_argument(
        "-h",
        "--help",
        action=make_help_action(_lazy_ui_help("show_config_help")),
    )
    add_output_args(show_parser)

    list_parser = config_sub.add_parser(
        "list",
        aliases=["ls"],
        help="List all available config options",
        add_help=False,
    )
    list_parser.add_argument(
        "-h",
        "--help",
        action=make_help_action(_lazy_ui_help("show_config_help")),
    )
    add_output_args(list_parser)

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


def _resolve(option: ConfigOption, toml_data: dict[str, Any]) -> tuple[bool, str, Any]:
    """Resolve an option via the shared manifest resolver.

    Delegates to `config_manifest.resolve_scalar` so `config show`/`get`
    report exactly what the runtime reads.

    Returns:
        `(is_set, source, value)`, where `is_set` is `False` when the value
        came from the typed default.
    """
    from deepagents_code.config_manifest import resolve_scalar

    value, source = resolve_scalar(option, toml_data=toml_data)
    return source != "default", source, value


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


def _source_label(source: str) -> str:
    """Render the source column for human output.

    Returns:
        Source label for the value's origin.
    """
    return source


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


def _run_show(output_format: OutputFormat) -> int:
    """Resolve every option and print its effective value and source.

    Returns:
        Process exit code (`0` on success).
    """
    from deepagents_code.config import _ensure_bootstrap
    from deepagents_code.config_manifest import get_config_options, load_config_toml

    # Load `.env` files into the environment so resolution reflects what the
    # app actually reads, not just shell exports.
    _ensure_bootstrap()
    toml_data = load_config_toml()

    options = get_config_options()
    resolved = [(opt, *_resolve(opt, toml_data)) for opt in options]

    if output_format == "json":
        write_json(
            "config show",
            [
                {
                    "key": opt.key,
                    "group": opt.group,
                    "source": source,
                    "set": is_set,
                    "redacted": opt.redacted,
                    # Redact secret values: report presence only.
                    "value": None if opt.redacted else value,
                }
                for opt, is_set, source, value in resolved
            ],
        )
        return 0

    from rich.markup import escape

    from deepagents_code.config import console
    from deepagents_code.config_manifest import iter_groups

    console.print()
    for group in iter_groups(options):
        console.print(f"[bold]{group}[/bold]")
        for opt, is_set, source, value in resolved:
            if opt.group != group:
                continue
            display = _display_value(opt, is_set=is_set, value=value)
            source_label = _source_label(source)
            # `display` and `source_label` may contain Rich markup from env/TOML
            # or terminal metadata; escape them so values can't break rendering.
            display_text = escape(display)
            source_text = escape(source_label)
            console.print(
                f"  {opt.key:<34} {display_text:<22} [dim]{source_text}[/dim]",
                highlight=False,
            )
        console.print()
    return 0


def _run_list(output_format: OutputFormat) -> int:
    """Print the static catalog of available options (no resolution).

    Returns:
        Process exit code (`0` on success).
    """
    from deepagents_code.config_manifest import get_config_options

    options = get_config_options()
    if output_format == "json":
        write_json(
            "config list",
            [
                {
                    "key": opt.key,
                    "group": opt.group,
                    "summary": opt.summary,
                    "type": opt.type,
                    "default": opt.default,
                    "redacted": opt.redacted,
                    "env_var": opt.env_var,
                    "toml_path": opt.toml_path,
                    "cli_flag": opt.cli_flag,
                }
                for opt in options
            ],
        )
        return 0

    from deepagents_code.config import console
    from deepagents_code.config_manifest import iter_groups

    console.print()
    for group in iter_groups(options):
        console.print(f"[bold]{group}[/bold]")
        for opt in options:
            if opt.group != group:
                continue
            console.print(f"  [cyan]{opt.key}[/cyan]  [dim]({opt.type})[/dim]")
            console.print(f"    {opt.summary}", highlight=False)
            console.print(f"    {_sources_line(opt)}", highlight=False, style="dim")
        console.print()
    return 0


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
    is_set, source, value = _resolve(option, toml_data)

    if output_format == "json":
        write_json(
            "config get",
            {
                "key": option.key,
                "source": source,
                "set": is_set,
                "redacted": option.redacted,
                "value": None if option.redacted else value,
            },
        )
        return 0

    from rich.markup import escape

    from deepagents_code.config import console

    display = _display_value(option, is_set=is_set, value=value)
    source_label = _source_label(source)
    console.print(
        f"{option.key} = {escape(display)}  [dim]({escape(source_label)})[/dim]",
        highlight=False,
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

    if command == "show":
        return _run_show(output_format)
    if command in {"list", "ls"}:
        return _run_list(output_format)
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

    rows: list[tuple[str, Any, bool]] = []
    for label, path in candidates:
        if path is None:
            continue
        try:
            exists = path.exists()
        except OSError:
            # A permission/transient FS error is not the same as "missing"; log
            # it (at debug, to keep normal output clean) so a developer can tell
            # the two apart when triaging a `config path` report.
            logger.debug("Could not stat %s", path, exc_info=True)
            exists = False
        rows.append((label, path, exists))
    return rows
