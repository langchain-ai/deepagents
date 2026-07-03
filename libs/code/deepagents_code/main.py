"""Main entry point and loop."""

# ruff: noqa: E402
# Imports placed after warning filters to suppress deprecation warnings

# Suppress deprecation warnings from langchain_core (e.g., Pydantic V1 on Python 3.14+)
import warnings

warnings.filterwarnings("ignore", module="langchain_core._api.deprecation")

import argparse
import asyncio
import contextlib
import importlib.util
import json
import logging
import os
import shutil
import sys
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

if TYPE_CHECKING:
    from rich.console import Console

    from deepagents_code.app import AppResult
    from deepagents_code.mcp_tools import MCPServerInfo
    from deepagents_code.notifications import PendingNotification

# Suppress Pydantic v1 compatibility warnings from langchain on Python 3.14+
warnings.filterwarnings("ignore", message=".*Pydantic V1.*", category=UserWarning)

from deepagents_code._version import __version__

logger = logging.getLogger(__name__)

_SANDBOX_DEFAULT_SENTINEL = "\x00default"
"""Marker stored by `--sandbox` with no value, resolved to `[sandboxes].default`."""


def build_version_text() -> str:
    """Build the plain-text output for the `--version` CLI flag.

    Includes the CLI and SDK versions and any installed optional
    dependencies. For editable installs it also reports the source path and
    the resolved versions of the core LangChain-ecosystem dependencies.

    Reports the same version facts as the `/version` slash command, in the
    same section order (versions, editable path, core dependencies, optional
    dependencies), but omits its network-dependent release-age suffixes and
    update-available hint so `--version` stays offline.

    Returns:
        Multi-line version string suitable for stdout.
    """
    from deepagents_code.extras_info import resolve_sdk_version

    sdk_version_value, status = resolve_sdk_version()
    sdk_version = sdk_version_value if status == "resolved" else "unknown"

    text = f"deepagents-code {__version__}\ndeepagents (SDK) {sdk_version}"

    editable = False
    try:
        from deepagents_code.config import (
            _get_editable_install_path,
            _is_editable_install,
        )

        editable = _is_editable_install()
        if editable:
            path = _get_editable_install_path()
            text += f"\nEditable install: {path}" if path else "\nEditable install"
    except Exception:
        logger.warning("Unexpected error detecting editable install", exc_info=True)

    # Core dependencies precede optional dependencies to match the section
    # order of the `/version` slash command (see `_handle_version_command`).
    if editable:
        try:
            from deepagents_code.extras_info import format_core_dependencies_plain

            core_text = format_core_dependencies_plain()
        except Exception:
            logger.warning("Failed to collect core dependency versions", exc_info=True)
            core_text = ""
        if core_text:
            text = f"{text}\n\n{core_text}"

    try:
        from deepagents_code.extras_info import (
            format_extras_status_plain,
            get_extras_status,
        )

        extras_text = format_extras_status_plain(get_extras_status())
    except Exception:
        logger.warning("Unexpected error collecting optional deps", exc_info=True)
        extras_text = ""
    if extras_text:
        text = f"{text}\n\n{extras_text}"

    return text


def _restart_current_process() -> NoReturn:
    """Replace the current process with a fresh `deepagents_code` invocation.

    Raises:
        RuntimeError: If process replacement unexpectedly returns.
    """
    argv = [sys.executable, "-m", "deepagents_code", *sys.argv[1:]]
    # Re-exec the trusted interpreter with the user's own argv verbatim; the
    # only "input" is the command the user already ran, so S606's concern
    # (untrusted/unsanitized args to a spawned executable) does not apply.
    os.execv(sys.executable, argv)  # noqa: S606
    msg = "os.execv returned unexpectedly"
    raise RuntimeError(msg)


def _terminal_row_count(console: "Console", text: str) -> int:
    """Return how many terminal rows Rich renders for `text`.

    Args:
        console: The Rich console whose current width determines wrapping.
        text: The string to measure, rendered with no markup.

    Returns:
        The number of visual rows Rich wraps `text` into, at least 1.
    """
    from rich.text import Text

    return max(1, len(console.render_lines(Text(text), console.options)))


def _should_check_teardown_thread(
    thread_id: str | None,
    *,
    request_count: int,
    resume_thread: str | None,
) -> bool:
    """Return whether teardown should query for checkpointed thread content.

    Any session that owns a thread may have persisted a checkpoint, so the only
    gate is whether a thread exists. `request_count` and `resume_thread` are
    accepted and ignored: an interrupted first turn can checkpoint before any
    usage metadata is recorded, so they are not a reliable proxy. They remain in
    the signature so callers need not change if the gate later grows selective.
    """
    del request_count, resume_thread
    return bool(thread_id)


def _render_teardown_thread_hints(
    console: "Console",
    thread_id: str,
    *,
    return_code: int,
) -> None:
    """Print the LangSmith link and resume hint for a checkpointed thread.

    Both hints share a single `thread_exists` lookup to avoid spinning up a
    second event loop and aiosqlite connection during teardown. Every failure is
    logged at debug and swallowed: teardown convenience output must never crash
    the exit path.

    Args:
        console: Console to print the hints to.
        thread_id: Thread whose checkpoints back the hints.
        return_code: Process exit code; the resume hint is shown only on a clean
            exit (`0`).
    """
    from rich.style import Style
    from rich.text import Text

    from deepagents_code.config import build_langsmith_thread_url
    from deepagents_code.sessions import thread_exists

    try:
        thread_has_checkpoints = asyncio.run(thread_exists(thread_id))
    except Exception:
        logger.debug(
            "Could not check thread existence on teardown",
            exc_info=True,
        )
        return

    if not thread_has_checkpoints:
        return

    try:
        thread_url = build_langsmith_thread_url(thread_id)
        if thread_url:
            console.print()
            ls_hint = Text("View this thread in LangSmith: ", style="dim")
            ls_hint.append(thread_url, style=Style(dim=True, link=thread_url))
            console.print(ls_hint)
    except Exception:
        logger.debug(
            "Could not display LangSmith thread URL on teardown",
            exc_info=True,
        )

    if return_code == 0:
        console.print()
        console.print("[dim]Resume this thread with:[/dim]")
        hint = Text("dcode -r ", style="cyan")
        hint.append(str(thread_id), style="cyan")
        console.print(hint)


def _confirm_update_after_restart(console: "Console", version: str) -> None:
    """Rewrite the pre-restart `Launching...` line as a stable update status.

    The `Updated to v{version}. Launching...` line is printed by the previous
    generation right before `os.execv`; this runs in the re-exec'd process to
    clear the transient action once the new version is actually running.

    The in-place rewrite is attempted only on a real terminal: `os.execv` does
    nothing between that print and this process's first output, so the cursor
    is parked on the line directly below it. On non-terminals the escape codes
    would corrupt redirected output, so the line is left as-is.

    The row count is recomputed against the current terminal width, so a resize
    during the upgrade/re-exec window could make the erase loop clear the wrong
    number of wrapped rows. This is a benign visual glitch (no exception), and
    is rare enough not to warrant defending against here.

    Args:
        console: The Rich console used for startup output.
        version: The version now running, used in the confirmation line.
    """
    if not console.is_terminal:
        return
    from rich.control import Control
    from rich.segment import ControlType

    launch_status = f"Updated to v{version}. Launching..."
    launch_rows = _terminal_row_count(console, launch_status)
    # Move up to the bottom row of the old status and erase each rendered row.
    # This preserves the rewrite when Rich wrapped the status in a narrow pane.
    for _ in range(launch_rows):
        console.control(
            Control(
                (ControlType.CURSOR_UP, 1),
                (ControlType.CURSOR_MOVE_TO_COLUMN, 0),
                (ControlType.ERASE_IN_LINE, 2),
            )
        )
    console.print(f"[green]Updated to v{version}.[/green]", highlight=False)


def _run_startup_auto_update(console: "Console") -> None:
    """Apply enabled auto-updates before the TUI and server start.

    On a successful upgrade the process is re-exec'd so the new version is
    loaded. Any failure is fail-soft: the installed version is launched and
    the error is surfaced, never blocking startup.

    Raises:
        SystemExit: Re-raised rather than suppressed by the fail-soft handler,
            so a process-exit request is never swallowed (the `os.execv`
            re-exec is simulated this way under test).
    """
    from rich.markup import escape

    from deepagents_code._env_vars import DEBUG_UPDATE, RESTARTED_AFTER_UPDATE
    from deepagents_code._version import __version__ as cli_version
    from deepagents_code.config import _is_editable_install
    from deepagents_code.update_check import (
        create_update_log_path,
        detect_shadowed_dcode_safe,
        format_release_age_parenthetical,
        format_shadowed_dcode_warning,
        get_cached_update_available,
        is_auto_update_enabled,
        is_installed_version_at_least,
        is_update_check_enabled,
        mark_auto_update_default_acknowledged,
        perform_upgrade,
        release_requires_prereleases,
        should_announce_auto_update_default,
        upgrade_command,
    )

    try:
        if (
            _is_editable_install()
            or not is_update_check_enabled()
            or not is_auto_update_enabled()
        ):
            return
        # Consume the re-exec sentinel recorded before the previous restart.
        restarted_for = os.environ.pop(RESTARTED_AFTER_UPDATE, None)
        if restarted_for is not None and is_installed_version_at_least(restarted_for):
            # The re-exec landed on the upgraded version, so the prior
            # "Launching..." line can be replaced with a stable completed status.
            try:
                _confirm_update_after_restart(console, restarted_for)
            except Exception:
                # The upgrade already succeeded; this rewrite is purely
                # cosmetic. Swallow rendering glitches with their own guard so
                # the outer fail-soft handler does not misreport a successful
                # upgrade as "Auto-update failed". The prior "Launching..."
                # line simply stays.
                logger.debug("Post-restart update confirmation failed", exc_info=True)
        available, latest = get_cached_update_available()
        if not available or latest is None:
            return
        if is_installed_version_at_least(latest):
            # The on-disk install already satisfies `latest` (e.g. the user ran
            # `/update` in-session). `get_cached_update_available` compares the
            # baked-in `__version__`, which lags an in-session upgrade, so
            # re-running the upgrade here would be a redundant no-op and restart.
            return
        if restarted_for == latest:
            # Already restarted after upgrading to this version, yet it still
            # reports as available: the install did not change the running
            # version. Bail out instead of upgrading and restarting forever
            # (this runs before the TUI, so there is no in-app way to stop it).
            update_needs_prereleases = release_requires_prereleases(latest)
            cmd = upgrade_command(
                include_prereleases=True if update_needs_prereleases else None,
                version=latest if update_needs_prereleases else None,
            )
            console.print(
                f"[bold yellow]Warning:[/bold yellow] v{latest} still reports as "
                "available after an automatic update; skipping auto-update to "
                f"avoid a restart loop. Update manually: [cyan]{cmd}[/cyan]\n"
                f"Continuing with v{cli_version}.",
                highlight=False,
            )
            return
        if should_announce_auto_update_default():
            # First-run consent/migration: auto-update is on only because of the
            # opt-out default, not an explicit choice. Announce it once and skip
            # this install so the user can opt out before anything runs.
            #
            # Mark *before* printing so a `console.print` failure cannot leave
            # the notice un-acknowledged and re-firing forever. The inverse risk
            # (mark succeeds, print fails, user never sees it) requires a broken
            # console and is the lesser evil versus an unbounded re-nag.
            acknowledged = mark_auto_update_default_acknowledged()
            message = (
                "[bold]dcode now updates automatically by default.[/bold] "
                f"v{latest} will be installed on the next launch.\n"
                "To opt out, set [cyan][update].auto_update = false[/cyan] in "
                "config.toml or [cyan]DEEPAGENTS_CODE_AUTO_UPDATE=0[/cyan] "
                "(or run [cyan]dcode --auto-update[/cyan] to toggle it off now).\n"
                f"Continuing with v{cli_version} for now."
            )
            if not acknowledged:
                # The acknowledgement could not be persisted (e.g. a read-only
                # state dir). Without this note the identical notice would
                # reappear every launch with no explanation.
                message += (
                    "\n[yellow]Note:[/yellow] this acknowledgement could not be "
                    "saved, so this message may appear again until you opt out "
                    "or the state directory becomes writable."
                )
            console.print(message, highlight=False)
            return
        release_age = format_release_age_parenthetical(latest)
        console.print(
            f"Updating dcode from v{cli_version} to v{latest}{release_age}..."
        )
        if os.environ.get(DEBUG_UPDATE):
            console.print("Skipped update install (debug mode).", style="dim")
            return
        log_path = create_update_log_path()
        console.print(
            f"Progress: tail -f {log_path}",
            style="dim",
            highlight=False,
            markup=False,
        )
        success, output = asyncio.run(
            perform_upgrade(log_path=log_path, target_version=latest)
        )
        if success:
            # If a stale `dcode` is earlier on PATH, the auto-restart would
            # re-exec into the old binary and the user would silently keep
            # running the pre-upgrade version. Detect that *before* the
            # re-exec so the warning isn't immediately wiped by the new
            # process's startup, and skip the restart — re-exec'ing into
            # an unchanged version would also trip the `restarted_for`
            # no-op loop guard on the next launch. Use the never-raises
            # wrapper so a detector defect can't crash startup after an
            # otherwise-successful upgrade.
            shadow = detect_shadowed_dcode_safe()
            if shadow is not None:
                # The warning embeds filesystem paths from `shutil.which`,
                # which can legally contain `[` (macOS/Linux). With
                # `markup=True` those would be parsed as Rich style tags,
                # so escape the warning before interpolation; the sibling
                # auto-update failure branch at the bottom of this function
                # escapes its uv output the same way.
                warning = format_shadowed_dcode_warning(shadow)
                console.print(
                    f"[bold yellow]Warning:[/bold yellow] {escape(warning)}\n"
                    f"Continuing with v{cli_version}.",
                    highlight=False,
                    markup=True,
                )
                return
            console.print(
                f"[green]Updated to v{latest}. Launching...[/green]",
                highlight=False,
            )
            # Record the target version so the re-exec'd process can detect a
            # no-op upgrade and break the loop (see the `restarted_for` guard).
            os.environ[RESTARTED_AFTER_UPDATE] = latest
            try:
                _restart_current_process()
            except (OSError, RuntimeError):
                # Upgrade succeeded but the re-exec did not happen (`os.execv`
                # raised, or returned unexpectedly). Drop the sentinel and
                # continue on the old in-memory code; the user must restart
                # manually to load the new version.
                os.environ.pop(RESTARTED_AFTER_UPDATE, None)
                logger.warning("Restart after update failed", exc_info=True)
                console.print(
                    f"[bold yellow]Warning:[/bold yellow] Updated to v{latest} but "
                    "the automatic restart failed. Restart dcode manually to use "
                    "the new version.",
                    highlight=False,
                )
            return
        update_needs_prereleases = release_requires_prereleases(latest)
        cmd = upgrade_command(
            include_prereleases=True if update_needs_prereleases else None,
            version=latest if update_needs_prereleases else None,
        )
        detail = f": {escape(output[:200])}" if output else ""
        console.print(
            f"[bold red]Auto-update failed{detail}[/bold red]\n"
            f"Run manually: [cyan]{cmd}[/cyan]\n"
            f"Continuing with v{cli_version}.",
            markup=True,
            highlight=False,
        )
    except SystemExit:
        # Process replacement (and test doubles that simulate it) must not be
        # swallowed by the fail-soft handler below.
        raise
    except Exception:
        logger.warning("Startup auto-update failed", exc_info=True)
        console.print(
            "[bold yellow]Warning:[/bold yellow] Auto-update failed before startup; "
            "continuing with the installed version."
        )


def _resolve_agent_arg(args: argparse.Namespace) -> str:
    """Resolve the final agent identifier from parsed CLI args.

    Precedence, highest first:

    1. Explicit `-a <name>` (stored as `args.agent` by argparse).
    2. `-r <thread>` is present → use `DEFAULT_AGENT_NAME`. The real agent is
        inferred later by `_resolve_resume_thread` via thread metadata
        (`get_thread_agent`), so we must NOT pre-seed a stored agent here or
        it would suppress that inference.
    3. `[agents].default` from config — the user's intentional sticky
        default (set via Ctrl+S in the `/agents` picker).
    4. `[agents].recent` from config — the most recently switched-to agent.
    5. `DEFAULT_AGENT_NAME` as the final fallback.

    Both `default` and `recent` are gated by `_recent_agent_is_valid` so a
    stale entry pointing at a deleted agent directory is ignored.

    Extracted from the `cli_main` body so it's unit-testable without
    constructing the full arg tree.

    Args:
        args: Parsed argparse namespace from `parse_args()`.

    Returns:
        The agent identifier to hand downstream.
    """
    from deepagents_code._constants import DEFAULT_AGENT_NAME

    if args.agent is not None:
        return args.agent
    if getattr(args, "resume_thread", None) is not None:
        return DEFAULT_AGENT_NAME

    from deepagents_code.model_config import load_default_agent, load_recent_agent

    default = load_default_agent()
    if default and _recent_agent_is_valid(default):
        return default

    recent = load_recent_agent()
    if recent and _recent_agent_is_valid(recent):
        return recent
    return DEFAULT_AGENT_NAME


def _normalize_cwd_filter(cwd: str | None) -> str | None:
    """Normalize the `threads list --cwd` filter for metadata matching.

    Storage uses `str(Path.cwd())` (absolute, no symlink resolution — see
    `build_run_metadata`). We mirror that here with lexical normalization
    rather than `.resolve()` so a user invoking from a symlinked path doesn't
    get an empty result set due to symlink normalization.

    Args:
        cwd: Parsed `--cwd` value. An empty string means the flag was passed
            without a value and should use the current working directory.

    Returns:
        Absolute path string for filtering, or `None` when no filter was
        requested. Returns `None` if the current working directory cannot
        be determined (bare `--cwd` only).
    """
    if cwd is None:
        return None
    if cwd == "":  # noqa: PLC1901
        try:
            return str(Path.cwd())
        except OSError:
            logger.warning(
                "Could not determine working directory for --cwd; "
                "no cwd filter will be applied",
                exc_info=True,
            )
            return None
    return os.path.normpath(str(Path(cwd).expanduser().absolute()))


def _parse_interpreter_tools_flag(
    raw: str | None,
) -> str | list[str] | None:
    """Parse the `--interpreter-tools` argument into the PTC option shape.

    Args:
        raw: Argparse value: `None` (flag absent), `"safe"`, `"all"`, or a
            comma-separated list of tool names.

    Returns:
        `None` when the flag is absent, the literal string `"safe"`/`"all"`,
        or a list of trimmed tool names. The list may contain `"safe"` as an
        expandable preset (e.g. `"safe,task"` → `["safe", "task"]`).

        Calls `sys.exit(2)` when the value is empty, contains only blank
        tokens, or includes `"all"` inside a list — the CLI treats those as
        usage errors.
    """
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        sys.stderr.write(
            "Error: --interpreter-tools requires a value: 'safe', 'all', or a "
            "comma-separated list of tool names.\n"
        )
        sys.exit(2)
    normalized = text.lower()
    if normalized in {"safe", "all"}:
        return normalized
    names = [token.strip() for token in text.split(",") if token.strip()]
    if not names:
        sys.stderr.write(
            "Error: --interpreter-tools list must contain at least one "
            "non-empty tool name.\n"
        )
        sys.exit(2)
    if any(name.lower() == "all" for name in names):
        sys.stderr.write(
            "Error: --interpreter-tools 'all' cannot be combined with other "
            "tools; use 'all' on its own or list explicit tool names "
            "(optionally with the 'safe' preset).\n"
        )
        sys.exit(2)
    return names


def _resolve_interpreter_enabled(args: argparse.Namespace) -> bool:
    """Return whether the JS interpreter should run for these CLI args.

    Delegates to `_resolve_enable_interpreter` so the CLI pre-flight gate and the
    stored `ServerConfig` share one resolution rule and cannot drift. The default
    comes from `[interpreter].enable_interpreter` in local mode and is disabled
    for remote sandboxes, where `CodeInterpreterMiddleware` is unsupported;
    explicit `--interpreter`/`--no-interpreter` overrides the default.

    `args.sandbox` is already normalized by `parse_args` (the bare-flag sentinel
    is resolved to a provider name), so the resolver sees the concrete value.
    """
    from deepagents_code._server_config import _resolve_enable_interpreter

    return _resolve_enable_interpreter(args.interpreter, args.sandbox)


def _warn_if_interpreter_disabled_by_sandbox(args: argparse.Namespace) -> None:
    """Warn that a remote sandbox suppressed the otherwise-default interpreter.

    With `js_eval` on by default in local mode, a `--sandbox` run silently drops
    it (the middleware is unsupported under a remote sandbox). This prints to
    stderr on the non-interactive (`-n`) path; the interactive TUI surfaces the
    same advisory as a startup notification (see
    `DeepAgentsApp._notify_interpreter_disabled_by_sandbox`).

    Keyed on the raw `args.interpreter` tri-state so an explicit
    `--no-interpreter` opt-out stays silent (the predicate only fires for the
    unset default).
    """
    from deepagents_code._server_config import _interpreter_suppressed_by_sandbox
    from deepagents_code.config import settings

    if not _interpreter_suppressed_by_sandbox(
        enable_interpreter=args.interpreter,
        sandbox_type=args.sandbox,
        local_default=settings.enable_interpreter,
    ):
        return
    from rich.console import Console as _Console

    _Console(stderr=True).print(
        "[yellow]Warning:[/yellow] JS interpreter (`js_eval`) is unavailable "
        "under a remote sandbox; it runs in local mode only."
    )


def _resolve_rubric_text(rubric: str | None) -> str | None:
    """Resolve the rubric from `--rubric` into one string.

    `--rubric` accepts literal text, or `@path` to read a file. File paths
    may be absolute, relative to the `dcode` process working directory, or
    `~`-expanded home paths.

    Args:
        rubric: Value of `--rubric` (literal text or `@path`), or `None`.

    Returns:
        The resolved rubric text, or `None` when the flag was not supplied.

    Raises:
        ValueError: If the rubric is empty, or a referenced file is missing,
            unreadable, or empty.
    """
    if rubric is None:
        return None

    # An `@`-prefixed value is always read as a file path. The path may be
    # absolute, relative to the `dcode` process working directory, or `~`-based.
    # There is no way to pass a literal rubric that begins with `@` (put such
    # text in a file).
    if rubric.startswith("@"):
        path = rubric[1:]
        try:
            text = Path(path).expanduser().read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            # `UnicodeError` (e.g. `UnicodeDecodeError`) subclasses `ValueError`,
            # not `OSError`. Catch it here so a binary/non-UTF-8 file yields the
            # framed "Could not read rubric file" message instead of a raw codec
            # error.
            msg = f"Could not read rubric file {path!r}: {exc}."
            raise ValueError(msg) from exc
        if not text.strip():
            msg = f"Rubric file {path!r} is empty."
            raise ValueError(msg)
        return text.strip()

    if not rubric.strip():
        msg = "--rubric must not be empty."
        raise ValueError(msg)
    return rubric.strip()


def _warn_if_interpreter_tools_without_interpreter(
    args: argparse.Namespace, *, enable_interpreter: bool
) -> None:
    """Warn that `--interpreter-tools` is a no-op without the interpreter.

    This drives the non-interactive (`-n`) path and prints to stderr. The
    interactive TUI surfaces the same advisory as a startup notification (see
    `DeepAgentsApp._notify_interpreter_tools_without_interpreter`).

    Attributes are accessed directly (not via `getattr` defaults) so an argparse
    `dest` rename fails loudly in tests rather than silently disabling the warning.
    """
    if args.interpreter_tools is None:
        return
    if enable_interpreter:
        return
    from rich.console import Console as _Console

    _Console(stderr=True).print(
        "[yellow]Warning:[/yellow] --interpreter-tools has no effect "
        "when the interpreter is disabled."
    )


def _recent_agent_is_valid(name: str) -> bool:
    """Return `True` when `~/.deepagents/<name>/` still exists on disk.

    Used to guard against a stale `[agents].recent` entry pointing at an
    agent the user has since deleted — in that case we silently fall back
    to the hard-coded default instead of failing at server start.

    Path is rebuilt from `Path.home()` rather than `settings.user_deepagents_dir`
    because `settings` is intentionally imported *after* argparse in `cli_main`
    (per the startup-hot-path guidance there), and pulling it in here would
    undo that deferral.

    `is_dir()` is wrapped in `try/except OSError` so permission errors on
    `~/.deepagents` (symlink loops, EACCES) don't crash the launch — we
    treat them the same as "not valid" and fall back to the default.
    """
    from pathlib import Path as _Path

    try:
        return (_Path.home() / ".deepagents" / name).is_dir()
    except OSError:
        logger.warning(
            "Could not validate recent agent %r; falling back to default",
            name,
            exc_info=True,
        )
        return False


def check_cli_dependencies() -> None:
    """Check if optional dependencies are installed."""
    missing = []

    if importlib.util.find_spec("requests") is None:
        missing.append("requests")

    if importlib.util.find_spec("dotenv") is None:
        missing.append("python-dotenv")

    if importlib.util.find_spec("tavily") is None:
        missing.append("tavily-python")

    if importlib.util.find_spec("textual") is None:
        missing.append("textual")

    if missing:
        print("\nMissing required dependencies!")  # noqa: T201  # App output for missing dependencies
        print("\nThe following packages are required to use Deep Agents Code:")  # noqa: T201  # App output for missing dependencies
        for pkg in missing:
            print(f"  - {pkg}")  # noqa: T201  # CLI output for missing dependencies
        print("\nReinstall dcode with the recommended installer:")  # noqa: T201  # CLI output for missing dependencies
        print("  curl -LsSf https://langch.in/dcode | bash")  # noqa: T201  # CLI output for missing dependencies
        print("\nOr install the tool directly via uv:")  # noqa: T201  # CLI output for missing dependencies
        print("  uv tool install -U deepagents-code")  # noqa: T201  # CLI output for missing dependencies
        sys.exit(1)


_RIPGREP_URL = "https://github.com/BurntSushi/ripgrep#installation"
"""Fallback installation URL when no platform package manager is detected."""

_SUPPRESS_HINT_CLI = (
    'To suppress, edit ~/.deepagents/config.toml:\n\\[warnings]\nsuppress = \\["<key>"]'
)
"""Suppression hint for non-interactive output.

Contains a `<key>` placeholder that callers replace with the warning key
(e.g. `"ripgrep"`, `"tavily"`).
"""


def _ripgrep_install_hint() -> str:
    """Return a platform-specific install command for ripgrep.

    Falls back to the GitHub URL when the platform isn't recognized.
    """
    plat = sys.platform
    if plat == "darwin":
        if shutil.which("brew"):
            return "brew install ripgrep"
        if shutil.which("port"):
            return "sudo port install ripgrep"
    elif plat == "linux":
        if shutil.which("apt-get"):
            return "sudo apt-get install ripgrep"
        if shutil.which("dnf"):
            return "sudo dnf install ripgrep"
        if shutil.which("pacman"):
            return "sudo pacman -S ripgrep"
        if shutil.which("zypper"):
            return "sudo zypper install ripgrep"
        if shutil.which("apk"):
            return "sudo apk add ripgrep"
        if shutil.which("nix-env"):
            return "nix-env -iA nixpkgs.ripgrep"
    elif plat == "win32":
        if shutil.which("choco"):
            return "choco install ripgrep"
        if shutil.which("scoop"):
            return "scoop install ripgrep"
        if shutil.which("winget"):
            return "winget install BurntSushi.ripgrep"
    # Cross-platform fallbacks
    if shutil.which("cargo"):
        return "cargo install ripgrep"
    if shutil.which("conda"):
        return "conda install -c conda-forge ripgrep"
    return _RIPGREP_URL


def _is_managed_ripgrep_path(path: str | None) -> bool:
    """Return whether `path` points at the managed `rg` binary."""
    if path is None:
        return False

    from deepagents_code.managed_tools import managed_rg_path

    managed = managed_rg_path()
    return os.path.normcase(str(Path(path).resolve())) == os.path.normcase(
        str(managed.resolve())
    )


def _should_ensure_managed_ripgrep() -> bool:
    """Return whether startup should validate or install managed ripgrep."""
    rg_path = shutil.which("rg")
    return rg_path is None or _is_managed_ripgrep_path(rg_path)


def check_optional_tools(*, config_path: Path | None = None) -> list[str]:
    """Check for recommended external tools and return missing tool names.

    Skips tools that the user has suppressed via
    `[warnings].suppress` in `config.toml`.

    Args:
        config_path: Path to config file.

            Defaults to `~/.deepagents/config.toml`.

    Returns:
        List of missing tool names (e.g. `["ripgrep"]`).
    """
    from deepagents_code.model_config import is_warning_suppressed

    missing: list[str] = []
    if _should_ensure_managed_ripgrep() and not is_warning_suppressed(
        "ripgrep", config_path
    ):
        missing.append("ripgrep")

    from deepagents_code.config import settings

    if not settings.has_tavily and not is_warning_suppressed("tavily", config_path):
        missing.append("tavily")

    return missing


def _auto_install_ripgrep_cli(
    warn_console: "Console", missing_tools: list[str]
) -> list[str]:
    """Attempt the one-shot managed `rg` install for the headless CLI path.

    Mirrors the interactive `DeepAgentsApp._ensure_managed_ripgrep` flow for
    the non-interactive launch, where there is no Textual app to surface
    notices through. A checksum mismatch is reported loudly and a generic
    failure as a warning; both leave `"ripgrep"` in the returned list so the
    caller still prints the standard missing-tool notice and the slow Python
    fallback is used.

    Args:
        warn_console: `rich` console bound to stderr for user-facing notices.
        missing_tools: Tool names reported missing by `check_optional_tools`.

    Returns:
        `missing_tools` with `"ripgrep"` removed once a usable `rg` is
        resolved — the managed binary (with `BIN_DIR` prepended to `PATH`) or a
        system `rg` already on `PATH` — otherwise the list unchanged.
    """
    from deepagents_code.managed_tools import (
        ChecksumMismatchError,
        ensure_ripgrep,
        managed_rg_path,
        prepend_managed_bin_to_path,
    )

    warn_console.print("Installing ripgrep...")
    try:
        installed = asyncio.run(ensure_ripgrep())
    except ChecksumMismatchError:
        logger.exception(
            "ripgrep auto-install aborted: SHA-256 mismatch on downloaded archive"
        )
        warn_console.print(
            "[bold red]Error:[/bold red] ripgrep auto-install aborted: downloaded "
            "archive failed SHA-256 verification. Refusing to install."
        )
        return missing_tools
    except Exception:
        logger.warning("ripgrep auto-install failed unexpectedly", exc_info=True)
        warn_console.print(
            "[yellow]Warning:[/yellow] ripgrep auto-install failed unexpectedly "
            "— see logs."
        )
        return missing_tools

    if installed is None:
        return missing_tools

    if installed == managed_rg_path():
        prepend_managed_bin_to_path()
    return [tool for tool in missing_tools if tool != "ripgrep"]


def build_missing_tool_notification(tool: str) -> "PendingNotification":
    """Build a `PendingNotification` for a missing optional tool.

    The returned entry carries the install hint (or URL) in a typed payload so
    the notification center action handler can copy it / open it without
    re-running platform detection.

    Args:
        tool: Name of the missing tool (e.g. `"ripgrep"`, `"tavily"`).

    Returns:
        A registry entry ready for `NotificationRegistry.add`.
    """
    # Deferred import: keeps `--version` and other hot-path commands off the
    # `deepagents_code.notifications` -> `dataclasses`/`logging` chain.
    from deepagents_code.notifications import (
        ActionId,
        MissingDepPayload,
        NotificationAction,
        PendingNotification,
    )

    suppress_action = NotificationAction(
        ActionId.SUPPRESS, "Don't show notification again"
    )
    if tool == "ripgrep":
        hint = _ripgrep_install_hint()
        if hint.startswith("http"):
            actions: tuple[NotificationAction, ...] = (
                NotificationAction(
                    ActionId.OPEN_WEBSITE, "Open installation guide", primary=True
                ),
                suppress_action,
            )
            payload = MissingDepPayload(tool="ripgrep", url=hint)
        else:
            actions = (
                NotificationAction(
                    ActionId.COPY_INSTALL, "Copy install command", primary=True
                ),
                NotificationAction(ActionId.OPEN_WEBSITE, "Open installation guide"),
                suppress_action,
            )
            payload = MissingDepPayload(
                tool="ripgrep", install_command=hint, url=_RIPGREP_URL
            )
        body = (
            "ripgrep is not installed; the grep tool will use a slower fallback.\n\n"
            f"Install: {hint}"
        )
        return PendingNotification(
            key="dep:ripgrep",
            title="ripgrep is not installed",
            body=body,
            actions=actions,
            payload=payload,
        )
    if tool == "tavily":
        return PendingNotification(
            key="dep:tavily",
            title="Web search disabled",
            body=("Add a Tavily API key to enable web search."),
            actions=(
                NotificationAction(
                    ActionId.ENTER_API_KEY, "Enter API key", primary=True
                ),
                NotificationAction(ActionId.OPEN_WEBSITE, "Open tavily.com"),
                suppress_action,
            ),
            payload=MissingDepPayload(tool="tavily", url="https://tavily.com"),
        )
    logger.warning("No install hint configured for tool %r", tool)
    return PendingNotification(
        key=f"dep:{tool}",
        title=f"{tool} is not installed",
        body=f"{tool} is not installed.",
        actions=(
            NotificationAction(
                ActionId.SUPPRESS, "Don't show notification again", primary=True
            ),
        ),
        payload=MissingDepPayload(tool=tool),
    )


def format_tool_warning_cli(tool: str) -> str:
    """Format a missing-tool warning for non-interactive console output.

    Args:
        tool: Name of the missing tool.

    Returns:
        Warning string suitable for `console.print`.
    """
    if tool == "ripgrep":
        hint = _ripgrep_install_hint()
        if hint.startswith("http"):
            hint = f"[link={hint}]{hint}[/link]"
        suppress = _SUPPRESS_HINT_CLI.replace("<key>", "ripgrep")
        return (
            "ripgrep is not installed; the grep tool will use a slower fallback.\n"
            f"Install: {hint}\n\n"
            f"{suppress}\n"
        )
    if tool == "tavily":
        url = "https://tavily.com"
        suppress = _SUPPRESS_HINT_CLI.replace("<key>", "tavily")
        return (
            "Web search is disabled \u2014 TAVILY_API_KEY is not set.\n"
            f"Get a key at [link={url}]{url}[/link]\n\n"
            f"{suppress}\n"
        )
    return f"{tool} is not installed."


async def _preload_session_mcp_server_info(
    *,
    mcp_config_path: str | None,
    no_mcp: bool,
    trust_project_mcp: bool | None,
) -> list["MCPServerInfo"] | None:
    """Load MCP metadata for the interactive TUI in server mode.

    In server mode the actual MCP tools are created inside the LangGraph server
    process, but the local Textual app still needs MCP metadata for the welcome
    banner and `/mcp` viewer. This preloads the metadata in the app process and
    immediately cleans up any temporary MCP sessions it opened.

    Args:
        mcp_config_path: Optional explicit MCP config path.
        no_mcp: Whether MCP loading is disabled.
        trust_project_mcp: Project-level MCP trust decision.

    Returns:
        MCP server metadata for the TUI, or `None` when MCP is disabled.
    """
    if no_mcp:
        return None

    from deepagents_code.mcp_tools import resolve_and_load_mcp_tools
    from deepagents_code.project_utils import ProjectContext

    session_manager = None
    try:
        try:
            project_context = ProjectContext.from_user_cwd(Path.cwd())
        except OSError:
            logger.warning("Could not determine working directory for MCP preload")
            project_context = None
        _tools, session_manager, server_info = await resolve_and_load_mcp_tools(
            explicit_config_path=mcp_config_path,
            no_mcp=no_mcp,
            trust_project_mcp=trust_project_mcp,
            project_context=project_context,
        )
        return server_info
    finally:
        if session_manager is not None:
            try:
                await session_manager.cleanup()
            except Exception:
                logger.warning(
                    "MCP metadata preload cleanup failed",
                    exc_info=True,
                )


_HELP_SPECS: dict[str, tuple[str | None, str]] = {
    "help": (None, "show_help"),
    "agents": ("agents_command", "show_agents_help"),
    "skills": ("skills_command", "show_skills_help"),
    "threads": ("threads_command", "show_threads_help"),
    "mcp": ("mcp_command", "show_mcp_help"),
    "config": ("config_command", "show_config_help"),
    "auth": ("auth_command", "show_auth_help"),
    "tools": ("tools_command", "show_tools_help"),
}
"""Maps top-level command names to their startup-fast-path help dispatch.

Each value is `(subcommand_dest, ui_help_fn_name)`:

- `subcommand_dest` is the argparse `dest=` for the group's sub-subparsers,
    or `None` for leaf commands like `help`. When non-`None` and the parsed
    namespace has a value at that attribute, a real subcommand was given and
    the fast path declines.
- `ui_help_fn_name` is the attribute on `deepagents_code.ui` invoked to
    render the help screen.

When adding a new top-level command group with sub-subparsers, register it
here and add a corresponding `show_<group>_help` to `ui.py`. The drift
test in `tests/unit_tests/test_startup_fast_paths.py` enforces this.
"""


def _show_bare_command_group_help(args: argparse.Namespace) -> bool:
    """Render help for `help` and bare command groups before the heavy bootstrap.

    Short-circuits before `console`/`settings` are imported so help-only
    invocations stay snappy. Mirrors the dispatch in `cli_main` for the
    `help`, `agents`, `skills`, `threads`, `mcp`, `config`, `auth`, and `tools`
    commands when no subcommand was given.

    Args:
        args: Namespace from `parse_args()`. Only `command` and the per-group
            `<group>_command` attributes are read; both may be absent.

    Returns:
        `True` when help was rendered and the caller should exit; `False`
            when the command requires the full runtime path.
    """
    command = getattr(args, "command", None)
    if not isinstance(command, str):
        return False
    spec = _HELP_SPECS.get(command)
    if spec is None:
        return False

    command_attr, help_fn_name = spec
    if command_attr is not None and getattr(args, command_attr, None) is not None:
        return False

    from deepagents_code import ui

    # 2-arg `getattr` is intentional: a missing/renamed `show_*_help` in
    # `ui.py` is a developer bug and should raise `AttributeError` loudly
    # rather than fall through to a silent no-op.
    getattr(ui, help_fn_name)()
    return True


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace.
    """
    from deepagents_code._constants import DEFAULT_AGENT_NAME
    from deepagents_code.auth_commands import setup_auth_parser
    from deepagents_code.config_commands import setup_config_parser
    from deepagents_code.mcp_commands import setup_mcp_parsers
    from deepagents_code.output import add_json_output_arg
    from deepagents_code.skills import setup_skills_parser

    # Factory that builds an argparse Action whose __call__ invokes the
    # supplied *help_fn* instead of argparse's default help text.  Each
    # subcommand can pass its own Rich-formatted help screen so that
    # `deepagents <subcommand> -h` shows context-specific help.
    def _make_help_action(
        help_fn: Callable[[], None],
    ) -> type[argparse.Action]:
        """Create an argparse Action that displays *help_fn* and exits.

        argparse requires a *class* (not a callable) for custom actions.
        This factory uses a closure: the returned `_ShowHelp` class captures
        *help_fn* from the enclosing scope so that each subcommand can wire `-h`
        to its own Rich help screen.

        Args:
            help_fn: Callable that prints help text to the console.

        Returns:
            An argparse Action class wired to the given help function.
        """

        class _ShowHelp(argparse.Action):
            def __init__(
                self,
                option_strings: list[str],
                dest: str = argparse.SUPPRESS,
                default: str = argparse.SUPPRESS,
                **kwargs: Any,
            ) -> None:
                super().__init__(
                    option_strings=option_strings,
                    dest=dest,
                    default=default,
                    nargs=0,
                    **kwargs,
                )

            def __call__(
                self,
                parser: argparse.ArgumentParser,
                namespace: argparse.Namespace,  # noqa: ARG002  # Required by argparse Action interface
                values: str | Sequence[Any] | None,  # noqa: ARG002  # Required by argparse Action interface
                option_string: str | None = None,  # noqa: ARG002  # Required by argparse Action interface
            ) -> None:
                with contextlib.suppress(BrokenPipeError):
                    help_fn()
                parser.exit()

        return _ShowHelp

    # Lazy wrapper: defers `ui` import until the help action fires (i.e.,
    # only when the user passes `-h`). This avoids pulling in Rich and config at
    # parse time for the common non-help path.
    def _lazy_help(fn_name: str) -> Callable[[], None]:
        def _show() -> None:
            from deepagents_code import ui

            getattr(ui, fn_name)()

        return _show

    def help_parent(help_fn: Callable[[], None]) -> list[argparse.ArgumentParser]:
        parent = argparse.ArgumentParser(add_help=False)
        parent.add_argument("-h", "--help", action=_make_help_action(help_fn))
        return [parent]

    parser = argparse.ArgumentParser(
        description=("Deep Agents - AI Coding Assistant"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    subparsers.add_parser(
        "help",
        help="Show help information",
        add_help=False,
        parents=help_parent(_lazy_help("show_help")),
    )

    agents_parser = subparsers.add_parser(
        "agents",
        help="Manage agents",
        add_help=False,
        parents=help_parent(_lazy_help("show_agents_help")),
    )
    add_json_output_arg(agents_parser)
    agents_sub = agents_parser.add_subparsers(dest="agents_command")

    agents_list = agents_sub.add_parser(
        "list",
        aliases=["ls"],
        help="List all agents",
        add_help=False,
        parents=help_parent(_lazy_help("show_list_help")),
    )
    add_json_output_arg(agents_list)

    agents_reset = agents_sub.add_parser(
        "reset",
        help="Reset an agent's prompt to default",
        add_help=False,
        parents=help_parent(_lazy_help("show_reset_help")),
    )
    add_json_output_arg(agents_reset)
    agents_reset.add_argument("--agent", required=True, help="Name of agent to reset")
    agents_reset.add_argument(
        "--target", dest="source_agent", help="Copy prompt from another agent"
    )
    agents_reset.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without making changes",
    )

    setup_skills_parser(
        subparsers,
        make_help_action=_make_help_action,
        add_output_args=add_json_output_arg,
    )

    setup_mcp_parsers(
        subparsers,
        make_help_action=_make_help_action,
    )

    setup_config_parser(
        subparsers,
        make_help_action=_make_help_action,
        add_output_args=add_json_output_arg,
    )

    setup_auth_parser(
        subparsers,
        make_help_action=_make_help_action,
    )

    threads_parser = subparsers.add_parser(
        "threads",
        help="Manage conversation threads",
        add_help=False,
        parents=help_parent(_lazy_help("show_threads_help")),
    )
    add_json_output_arg(threads_parser)
    threads_sub = threads_parser.add_subparsers(dest="threads_command")

    threads_list = threads_sub.add_parser(
        "list",
        aliases=["ls"],
        help="List threads",
        add_help=False,
        parents=help_parent(_lazy_help("show_threads_list_help")),
    )
    add_json_output_arg(threads_list)
    threads_list.add_argument(
        "--agent", default=None, help="Filter by agent name (default: show all)"
    )
    threads_list.add_argument(
        "-n",
        "--limit",
        type=int,
        default=None,
        help="Max number of threads to display (default: 20)",
    )
    threads_list.add_argument(
        "--sort",
        choices=["created", "updated"],
        default=None,
        help="Sort threads by timestamp (default: from config, or updated)",
    )
    threads_list.add_argument(
        "--branch",
        default=None,
        help="Filter by git branch name",
    )
    threads_list.add_argument(
        "--cwd",
        nargs="?",
        const="",
        default=None,
        help=(
            "Filter by working directory. With no value, uses the current "
            "directory; pass a path to filter by that directory instead."
        ),
    )
    threads_list.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Show all columns (branch, created, prompt)",
    )
    threads_list.add_argument(
        "-r",
        "--relative",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show timestamps as relative time (default: from config, or absolute)",
    )
    threads_delete = threads_sub.add_parser(
        "delete",
        help="Delete a thread",
        add_help=False,
        parents=help_parent(_lazy_help("show_threads_delete_help")),
    )
    add_json_output_arg(threads_delete)
    threads_delete.add_argument("thread_id", help="Thread ID to delete")
    threads_delete.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without making changes",
    )

    update_parser = subparsers.add_parser(
        "update",
        help="Check for and install updates",
        add_help=False,
        parents=help_parent(_lazy_help("show_update_help")),
    )
    update_parser.add_argument(
        "--prerelease",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Include alpha/beta/rc releases when checking for updates",
    )
    add_json_output_arg(update_parser)

    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Print install health and diagnostics",
        add_help=False,
        parents=help_parent(_lazy_help("show_doctor_help")),
    )
    add_json_output_arg(doctor_parser)

    tools_parser = subparsers.add_parser(
        "tools",
        help="Manage managed external tools (e.g. ripgrep)",
        add_help=False,
        parents=help_parent(_lazy_help("show_tools_help")),
    )
    add_json_output_arg(tools_parser)
    tools_sub = tools_parser.add_subparsers(dest="tools_command")

    tools_install = tools_sub.add_parser(
        "install",
        help="Install or repair the managed ripgrep binary",
        add_help=False,
        parents=help_parent(_lazy_help("show_tools_install_help")),
    )
    add_json_output_arg(tools_install)

    tools_list = tools_sub.add_parser(
        "list",
        help="List the tools available to the agent",
        add_help=False,
        parents=help_parent(_lazy_help("show_tools_list_help")),
    )
    add_json_output_arg(tools_list)

    # Default interactive mode — argument order here determines the
    # usage line printed by argparse; keep in sync with ui.show_help().
    parser.add_argument(
        "-r",
        "--resume",
        dest="resume_thread",
        nargs="?",
        const="__MOST_RECENT__",
        default=None,
        metavar="ID",
        help="Resume thread: -r for most recent, -r <ID> for specific thread",
    )

    parser.add_argument(
        "-a",
        "--agent",
        default=None,
        metavar="NAME",
        help=(
            "Agent to use (e.g., coder, researcher). "
            "If omitted, falls back to [agents].default, then "
            "[agents].recent, then "
            f"the '{DEFAULT_AGENT_NAME}' built-in default."
        ),
    )

    parser.add_argument(
        "-M",
        "--model",
        metavar="MODEL",
        help="Model to use (e.g., claude-opus-4-7, gpt-5.5). "
        "Provider is auto-detected from model name.",
    )

    parser.add_argument(
        "--model-params",
        metavar="JSON",
        help="Extra kwargs to pass to the model as a JSON string "
        '(e.g., \'{"temperature": 0.7, "max_tokens": 4096}\'). '
        "These take priority, overriding config file values.",
    )

    from deepagents_code.ui import non_negative_int, positive_int

    parser.add_argument(
        "--max-retries",
        type=non_negative_int,
        default=None,
        metavar="N",
        help="Override max retries for transient model errors.",
    )

    parser.add_argument(
        "--profile-override",
        metavar="JSON",
        help="Override model profile fields as a JSON string "
        "(e.g., '{\"max_input_tokens\": 4096}'). "
        "Merged on top of config file profile overrides.",
    )

    parser.add_argument(
        "--default-model",
        metavar="MODEL",
        nargs="?",
        const="__SHOW__",
        default=None,
        help="Set the default model for future launches "
        "(e.g., anthropic:claude-opus-4-6). "
        "Use --default-model with no argument to show the current default. "
        "Use --clear-default-model to remove it.",
    )

    parser.add_argument(
        "--clear-default-model",
        action="store_true",
        help="Clear the default model, falling back to recent model "
        "or environment auto-detection.",
    )

    parser.add_argument(
        "-m",
        "--message",
        dest="initial_prompt",
        metavar="TEXT",
        help="Initial prompt to auto-submit when session starts",
    )

    parser.add_argument(
        "--skill",
        dest="initial_skill",
        metavar="NAME",
        help="Invoke a skill when the interactive session starts",
    )

    parser.add_argument(
        "--startup-cmd",
        dest="startup_cmd",
        metavar="CMD",
        help="Shell command to run at startup, before the first prompt "
        "(output shown, non-zero exit warns but does not abort)",
    )

    parser.add_argument(
        "-n",
        "--non-interactive",
        dest="non_interactive_message",
        metavar="TEXT",
        help="Run a single task non-interactively and exit "
        "(shell disabled unless --shell-allow-list is set)",
    )

    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Clean output for piping — only the agent's response "
        "goes to stdout. Requires -n or piped stdin.",
    )

    parser.add_argument(
        "--no-stream",
        dest="no_stream",
        action="store_true",
        help="Buffer the full response and write it to stdout at once "
        "instead of streaming token-by-token. Requires -n or piped stdin.",
    )

    parser.add_argument(
        "--max-turns",
        dest="max_turns",
        type=positive_int,
        metavar="N",
        help="Maximum number of agentic turns before stopping (must be >= 1). "
        "Overrides the internal safety default. Useful for CI/CD pipelines "
        "to prevent runaway agents. Requires -n or piped stdin.",
    )

    parser.add_argument(
        "--timeout",
        dest="timeout",
        type=positive_int,
        metavar="SECONDS",
        help="Hard wall-clock timeout in seconds. The agent is cancelled and "
        "the process exits with code 124 if the timeout is reached. "
        "Complements --max-turns (turn count) with a time-based limit; both "
        "use exit code 124 on expiry. Requires -n or piped stdin.",
    )

    parser.add_argument(
        "--goal",
        dest="goal",
        metavar="TEXT",
        help="Goal objective to turn into acceptance criteria. Opens a review "
        "prompt on interactive launch, then runs the accepted goal as the first "
        "task.",
    )
    parser.add_argument(
        "--rubric",
        dest="rubric",
        metavar="TEXT|@PATH",
        help="Acceptance criteria the agent self-evaluates against, looping "
        "until satisfied. Accepts literal text or '@path' to read a file "
        "(relative to the current working directory; '~' supported). "
        "Requires -n or piped stdin.",
    )
    parser.add_argument(
        "--rubric-model",
        dest="rubric_model",
        metavar="MODEL",
        help="Model the rubric grader uses (e.g. anthropic:claude-sonnet-4-6). "
        "Defaults to the main agent model.",
    )
    parser.add_argument(
        "--rubric-max-iterations",
        dest="rubric_max_iterations",
        type=positive_int,
        metavar="N",
        help="Override grader iterations per rubric attempt before stopping "
        "(must be >= 1; defaults to the SDK setting).",
    )

    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read input from stdin explicitly (instead of auto-detection)",
    )

    add_json_output_arg(parser, default="text")

    parser.add_argument(
        "-y",
        "--auto-approve",
        action="store_true",
        help=(
            "Auto-approve all tool calls without prompting "
            "(disables human-in-the-loop). Affected tools: shell "
            "execution, file writes/edits, web search, and URL fetch. "
            "Use with caution — the agent can execute arbitrary commands."
        ),
    )

    parser.add_argument(
        "--sandbox",
        nargs="?",
        const=_SANDBOX_DEFAULT_SENTINEL,
        default="none",
        metavar="TYPE",
        help=(
            "Remote sandbox for code execution (default: none - local only). "
            "Built-ins: agentcore, daytona, langsmith, modal, runloop, vercel. "
            "Third-party and config-declared providers are also accepted. "
            "Pass --sandbox with no value to use [sandboxes].default from "
            "config (keep the bare form last on the command line so a "
            "following subcommand isn't read as its value). langsmith is "
            "bundled; others require installing an extra or package."
        ),
    )

    parser.add_argument(
        "--sandbox-id",
        metavar="ID",
        help="Existing sandbox ID to attach to",
    )

    parser.add_argument(
        "--sandbox-snapshot-name",
        metavar="NAME",
        help="Snapshot (langsmith) or blueprint (runloop) name to use or create",
    )

    parser.add_argument(
        "--sandbox-setup",
        metavar="PATH",
        help="Path to setup script to run in sandbox after creation",
    )
    parser.add_argument(
        "-S",
        "--shell-allow-list",
        metavar="LIST",
        help="Comma-separated list of shell commands to auto-approve, "
        "'recommended' for safe defaults, or 'all' to allow any command. "
        "Applies to both -n and interactive modes.",
    )
    parser.add_argument(
        "--mcp-config",
        help="Path to MCP servers JSON configuration file (Claude Desktop format). "
        "Merged on top of auto-discovered configs (highest precedence). "
        "Run `dcode mcp config` to see discovery paths.",
    )
    parser.add_argument(
        "--no-mcp",
        action="store_true",
        help="Disable all MCP tool loading (skip auto-discovery and explicit config)",
    )
    parser.add_argument(
        "--trust-project-mcp",
        action="store_true",
        help="Trust project-level MCP configs with stdio servers "
        "(skip interactive approval prompt)",
    )
    parser.add_argument(
        "--interpreter",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable the JS interpreter (`js_eval`) middleware on the main agent. "
        "Enabled by default when not using a sandbox; use --no-interpreter to disable.",
    )
    parser.add_argument(
        "--interpreter-tools",
        dest="interpreter_tools",
        metavar="VALUE",
        help="PTC allowlist for `js_eval`: 'safe', 'all', or a comma-separated "
        "list of tool names (which may include the 'safe' preset, e.g. "
        "'safe,task'). Default is 'safe' (read-only file tools).",
    )

    parser.add_argument(
        "--update",
        action="store_true",
        help="Check for and install updates, then exit",
    )
    parser.add_argument(
        "--prerelease",
        action="store_true",
        help="With --update, include alpha/beta/rc releases",
    )
    parser.add_argument(
        "--auto-update",
        action="store_true",
        help="Toggle automatic updates on or off, then exit",
    )
    parser.add_argument(
        "--install",
        metavar="NAME",
        help="Install an optional extra (e.g. daytona, fireworks), then exit",
    )
    parser.add_argument(
        "--package",
        action="store_true",
        help=(
            "With --install, treat NAME as a package added via `uv --with` "
            "(for a custom provider package), not a deepagents-code extra"
        ),
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip interactive confirmation prompts (e.g., for --install)",
    )
    parser.add_argument(
        "--acp",
        action="store_true",
        help="Run as an ACP server over stdio instead of launching the Textual UI",
    )

    # `parse_args` runs on every invocation; keep the import-heavy metadata
    # scan off the hot path unless the user explicitly asked for --version.
    if any(arg in {"-v", "--version"} for arg in sys.argv[1:]):
        version_text = build_version_text()
    else:
        # Never surfaced: argparse only emits `version=` when the flag is
        # actually passed, which takes the `build_version_text()` branch above.
        # This placeholder only exists because `version=` requires a value.
        version_text = f"deepagents-code {__version__}"
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=version_text,
    )
    parser.add_argument(
        "-h",
        "--help",
        action=_make_help_action(_lazy_help("show_help")),
    )

    args = parser.parse_args()
    _resolve_and_validate_sandbox(args, parser)
    return args


def _resolve_and_validate_sandbox(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> None:
    """Resolve `--sandbox` against the registry and validate related flags.

    Handles the bare `--sandbox` form (resolve `[sandboxes].default`), unknown
    providers (with install/config guidance), and the `--sandbox-snapshot-name`
    / `--sandbox-id` flags whose support is driven by provider metadata. Calls
    `parser.error` (which exits) on invalid input.

    Because `--sandbox` takes an optional value (`nargs="?"`), placing it
    immediately before a subcommand (e.g. `dcode --sandbox agents`) makes
    argparse consume the subcommand as the flag's value. Pass an explicit
    provider (`--sandbox daytona`) or keep the bare form last on the command
    line.

    Args:
        args: Parsed namespace; `args.sandbox` is normalized in place.
        parser: The parser, used to emit errors.
    """
    if args.sandbox in {"none", None}:
        if args.sandbox_snapshot_name is not None:
            parser.error("--sandbox-snapshot-name requires a --sandbox provider")
        if args.sandbox_id is not None:
            parser.error("--sandbox-id requires a --sandbox provider")
        return

    from deepagents_code.integrations.sandbox_registry import SandboxRegistry

    registry = SandboxRegistry.load()

    def _config_note() -> str:
        """Build a breadcrumb when the config file failed to parse.

        Returns:
            A note to append to an error message, or an empty string when the
            config parsed cleanly.
        """
        if registry.config_error:
            return (
                f"\n\nNote: ~/.deepagents/config.toml could not be used "
                f"({registry.config_error}); any providers or default it "
                "declares were ignored."
            )
        return ""

    if args.sandbox == _SANDBOX_DEFAULT_SENTINEL:
        default = registry.default
        if not default:
            parser.error(
                "--sandbox was given with no value but no [sandboxes].default "
                "is configured in ~/.deepagents/config.toml. Pass a provider "
                "name explicitly or set [sandboxes].default." + _config_note()
            )
        args.sandbox = default

    if not registry.is_available(args.sandbox):
        available = ", ".join(registry.available_providers())
        parser.error(
            f"Unknown sandbox provider '{args.sandbox}'.\n"
            f"Available providers: {available}.\n\n"
            "If this is a third-party provider, install the package that "
            "publishes it and re-run:\n"
            "  /install <package-name> --package\n"
            f"or declare [sandboxes.providers.{args.sandbox}] in "
            "~/.deepagents/config.toml." + _config_note()
        )

    metadata = registry.get_metadata(args.sandbox)
    if args.sandbox_snapshot_name is not None and (
        metadata is None or not metadata.supports_snapshot_name
    ):
        parser.error(
            f"--sandbox-snapshot-name is not supported by provider '{args.sandbox}'"
        )
    if (
        args.sandbox_id is not None
        and metadata is not None
        and not metadata.supports_sandbox_id
    ):
        parser.error(f"--sandbox-id is not supported by provider '{args.sandbox}'")


async def run_textual_cli_async(
    assistant_id: str,
    *,
    auto_approve: bool = False,
    sandbox_type: str = "none",  # str (not None) to match argparse choices
    sandbox_id: str | None = None,
    sandbox_snapshot_name: str | None = None,
    sandbox_setup: str | None = None,
    model_name: str | None = None,
    model_params: dict[str, Any] | None = None,
    profile_override: dict[str, Any] | None = None,
    thread_id: str | None = None,
    resume_thread: str | None = None,
    initial_prompt: str | None = None,
    initial_skill: str | None = None,
    initial_goal: str | None = None,
    startup_cmd: str | None = None,
    mcp_config_path: str | None = None,
    no_mcp: bool = False,
    trust_project_mcp: bool | None = None,
    enable_interpreter: bool | None = None,
    interpreter_arg: bool | None = None,
    interpreter_ptc: str | list[str] | None = None,
    interpreter_ptc_acknowledge_unsafe: bool = False,
) -> "AppResult":
    """Run the Textual TUI interface (async version).

    Starts a LangGraph server in a subprocess and connects the TUI to it via the
    `langgraph-sdk` client.

    Args:
        assistant_id: Agent identifier for memory storage
        auto_approve: Whether to auto-approve tool usage
        sandbox_type: Type of sandbox
            ("none", "agentcore", "modal", "runloop", "daytona", "langsmith")
        sandbox_id: Optional existing sandbox ID to reuse.
        sandbox_snapshot_name: Snapshot (langsmith) or blueprint (runloop) name.
        sandbox_setup: Optional path to setup script to run in the sandbox
            after creation.
        model_name: Optional model name to use
        model_params: Extra kwargs from `--model-params` to pass to the model.

            These override config file values.
        profile_override: Extra profile fields from `--profile-override`.

            Merged on top of config file profile overrides.
        thread_id: Thread ID for the session.

            `None` when `resume_thread` is provided (the TUI resolves the final
            ID asynchronously).
        resume_thread: Raw resume intent from `-r` flag.

            `'__MOST_RECENT__'` for bare `-r`, a thread ID string for `-r <id>`,
            or `None` for new sessions.

            Resolved asynchronously inside the TUI.
        initial_prompt: Optional prompt to auto-submit when session starts
        initial_skill: Optional skill name to invoke when the session starts.
        initial_goal: Optional goal objective to draft criteria for when the
            session starts.
        startup_cmd: Shell command to run at startup before the first prompt.

            Output is rendered in the transcript; non-zero exits warn but
            do not abort the session.
        mcp_config_path: Optional path to MCP servers JSON configuration file.

            Merged on top of auto-discovered configs (highest precedence).
        no_mcp: Disable all MCP tool loading.
        trust_project_mcp: Controls project-level stdio server trust.

            `True` to allow, `False` to deny, `None` to check trust store.
        enable_interpreter: Enable `CodeInterpreterMiddleware` (`js_eval`) on
            the main agent. `None` defers to the sandbox-aware/config default.
        interpreter_arg: The raw `--interpreter`/`--no-interpreter` tri-state,
            forwarded so the app can tell an explicit opt-out from a
            sandbox-suppressed default when surfacing the disabled-by-sandbox
            advisory.
        interpreter_ptc: Override for `settings.interpreter_ptc` (PTC allowlist
            for `js_eval`).
        interpreter_ptc_acknowledge_unsafe: Explicit acknowledgement for
            `interpreter_ptc="all"` outside of `auto_approve`.

    Returns:
        An `AppResult` with the return code and final thread ID.
    """
    from rich.text import Text

    from deepagents_code.app import AppResult, run_textual_app
    from deepagents_code.config import (
        _get_default_model_spec,
        detect_provider,
        settings,
    )
    from deepagents_code.model_config import (
        ModelConfigError,
        ModelSpec,
        NoCredentialsConfiguredError,
    )
    from deepagents_code.onboarding import should_run_onboarding

    # Resolve display-name cheaply (<1ms, no langchain) so the status
    # bar can show the model on first paint. The expensive create_model()
    # (~560ms) is deferred to a background worker.

    defer_server_start = False
    try:
        resolved_spec = model_name or _get_default_model_spec()
    except NoCredentialsConfiguredError:
        resolved_spec = ""
        defer_server_start = True
    except ModelConfigError as e:
        from rich.markup import escape

        from deepagents_code.config import console

        console.print(f"[bold red]Error:[/bold red] {escape(str(e))}", highlight=False)
        return AppResult(return_code=1, thread_id=None)

    if resolved_spec:
        parsed = ModelSpec.try_parse(resolved_spec)
        if parsed:
            settings.model_provider = parsed.provider
            settings.model_name = parsed.model
        else:
            settings.model_name = resolved_spec
            settings.model_provider = detect_provider(resolved_spec) or ""
    else:
        settings.model_provider = ""
        settings.model_name = ""

    model_kwargs: dict[str, Any] | None = None
    if not defer_server_start:
        model_kwargs = {
            "model_spec": model_name or resolved_spec,
            "extra_kwargs": model_params,
            "profile_overrides": profile_override,
        }

    # Build kwargs for deferred server startup (runs inside the TUI).
    # Never pass auto_approve to the server — the interactive server must
    # always configure full HITL interrupts so that Shift+Tab can toggle
    # approval mode mid-session. The -y flag is handled client-side via
    # session_state.auto_approve in textual_adapter.py.
    server_kwargs: dict[str, Any] = {
        "assistant_id": assistant_id,
        "model_name": model_name or resolved_spec or None,
        "model_params": model_params,
        "sandbox_type": sandbox_type,
        "sandbox_id": sandbox_id,
        "sandbox_snapshot_name": sandbox_snapshot_name,
        "sandbox_setup": sandbox_setup,
        "enable_ask_user": True,
        "enable_interpreter": enable_interpreter,
        "interpreter_ptc": interpreter_ptc,
        "interpreter_ptc_acknowledge_unsafe": interpreter_ptc_acknowledge_unsafe,
        "mcp_config_path": mcp_config_path,
        "no_mcp": no_mcp,
        "trust_project_mcp": trust_project_mcp,
        "interactive": True,
    }

    mcp_preload_kwargs: dict[str, Any] | None = None
    if not no_mcp:
        mcp_preload_kwargs = {
            "mcp_config_path": mcp_config_path,
            "no_mcp": no_mcp,
            "trust_project_mcp": trust_project_mcp,
        }

    try:
        result = await run_textual_app(
            assistant_id=assistant_id,
            backend=None,
            auto_approve=auto_approve,
            cwd=Path.cwd(),
            thread_id=thread_id,
            resume_thread=resume_thread,
            initial_prompt=initial_prompt,
            initial_skill=initial_skill,
            initial_goal=initial_goal,
            startup_cmd=startup_cmd,
            launch_init=should_run_onboarding(),
            profile_override=profile_override,
            server_kwargs=server_kwargs,
            mcp_preload_kwargs=mcp_preload_kwargs,
            model_kwargs=model_kwargs,
            model_explicitly_set=model_name is not None,
            interpreter_arg=interpreter_arg,
            defer_server_start=defer_server_start,
        )
    except Exception as e:
        logger.debug("App error", exc_info=True)
        from deepagents_code.config import console

        error_text = Text("Application error: ", style="red")
        error_text.append(str(e))
        console.print(error_text)
        if logger.isEnabledFor(logging.DEBUG):
            console.print(Text(traceback.format_exc(), style="dim"))
        return AppResult(return_code=1, thread_id=None)

    return result


async def _run_acp_cli_async(
    assistant_id: str,
    *,
    run_acp_agent: Callable[[Any], Any],
    agent_server_cls: type[Any],
    model_name: str | None = None,
    model_params: dict[str, Any] | None = None,
    profile_override: dict[str, Any] | None = None,
    mcp_config_path: str | None = None,
    no_mcp: bool = False,
    trust_project_mcp: bool | None = None,
) -> int:
    """Run ACP server mode and return a process exit code.

    Args:
        assistant_id: Agent identifier to initialize.
        run_acp_agent: ACP server runner function.
        agent_server_cls: ACP server class constructor.
        model_name: Optional model name to use.
        model_params: Extra kwargs from `--model-params` to pass to the model.
        profile_override: Extra profile fields from `--profile-override`.
        mcp_config_path: Optional path to MCP servers JSON configuration file.
        no_mcp: Disable all MCP tool loading.
        trust_project_mcp: Controls project-level stdio server trust.

    Returns:
        Exit code for ACP mode.
    """
    from deepagents_code.agent import create_cli_agent, load_async_subagents
    from deepagents_code.config import create_model, settings
    from deepagents_code.model_config import (
        ModelConfigError,
        save_recent_model,
        touch_recent_model,
    )
    from deepagents_code.tools import fetch_url, get_current_thread_id, web_search

    try:
        model_result = create_model(
            model_name,
            extra_kwargs=model_params,
            profile_overrides=profile_override,
        )
    except ModelConfigError as exc:
        sys.stderr.write(f"Error: {exc}\n")
        sys.stderr.flush()
        return 1
    model_result.apply_to_settings()

    # Persist the resolved model so [models].recent is always populated.
    resolved_spec = f"{model_result.provider}:{model_result.model_name}"
    save_recent_model(resolved_spec)
    touch_recent_model(resolved_spec)

    tools: list[Any] = [fetch_url, get_current_thread_id]
    if settings.has_tavily:
        tools.append(web_search)

    mcp_session_manager = None
    mcp_server_info = None
    try:
        from deepagents_code.mcp_tools import resolve_and_load_mcp_tools

        (
            mcp_tools,
            mcp_session_manager,
            mcp_server_info,
        ) = await resolve_and_load_mcp_tools(
            explicit_config_path=mcp_config_path,
            no_mcp=no_mcp,
            trust_project_mcp=trust_project_mcp,
        )
        tools.extend(mcp_tools)
    except FileNotFoundError as exc:
        msg = f"Error: MCP config file not found: {exc}\n"
        sys.stderr.write(msg)
        sys.stderr.flush()
        return 1
    except RuntimeError as exc:
        msg = f"Error: Failed to load MCP tools: {exc}\n"
        sys.stderr.write(msg)
        sys.stderr.flush()
        return 1

    async_subagents = load_async_subagents() or None

    try:
        from langgraph.checkpoint.memory import InMemorySaver

        agent_graph, _backend = create_cli_agent(
            model=model_result.model,
            assistant_id=assistant_id,
            tools=tools,
            mcp_server_info=mcp_server_info,
            checkpointer=InMemorySaver(),
            async_subagents=async_subagents,
        )
    except Exception as exc:
        sys.stderr.write(f"Error: failed to create agent: {exc}\n")
        sys.stderr.flush()
        logger.debug("ACP agent creation failed", exc_info=True)
        return 1

    server = agent_server_cls(agent_graph)  # Pregel is a CompiledStateGraph at runtime
    exit_code = 0
    try:
        await run_acp_agent(server)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        sys.stderr.write(f"Error: ACP server failed: {exc}\n")
        sys.stderr.flush()
        logger.exception("ACP server crashed")
        exit_code = 1
    finally:
        if mcp_session_manager is not None:
            try:
                await mcp_session_manager.cleanup()
            except Exception:
                logger.warning("MCP session cleanup failed", exc_info=True)
    return exit_code


def apply_stdin_pipe(args: argparse.Namespace) -> None:
    r"""Read piped stdin and merge it into the parsed CLI arguments.

    When stdin is not a TTY (i.e. input is piped), reads all available text
    and applies it to the argument namespace. If stdin is a TTY or the piped
    input is empty/whitespace-only, the function returns without modifying
    `args`. Leading and trailing whitespace is stripped from piped input.

    - If `non_interactive_message` is already set (`-n`), prepends the
        piped text to it (the CLI still runs non-interactively):

        ```bash
        cat context.txt | dcode -n "summarize this"
        # non_interactive_message = "{contents of context.txt}\n\nsummarize this"
        ```

    - If `initial_prompt` is already set (`-m`, but not `-n`), prepends
        the piped text to it (the CLI still runs interactively):

        ```bash
        cat error.log | dcode -m "explain this"
        # initial_prompt = "{contents of error.log}\n\nexplain this"
        ```

    - If `initial_skill` is already set (`--skill`, but not `-n`), stores the
        piped text in `initial_prompt` so the skill receives it as the
        startup request:

        ```bash
        cat diff.txt | dcode --skill code-review
        # initial_prompt = "{contents of diff.txt}"
        ```

    - Otherwise, sets `non_interactive_message` to the piped text, causing
        the CLI to run non-interactively with it as the prompt:

        ```bash
        echo "fix the typo in README.md" | dcode
        # non_interactive_message = "fix the typo in README.md"
        ```

    Args:
        args: The parsed argument namespace (mutated in place).
    """
    from deepagents_code.config import console

    explicit_stdin = args.stdin

    if sys.stdin is None:
        if explicit_stdin:
            console.print(
                "[bold red]Error:[/bold red] --stdin was passed but stdin "
                "is not available."
            )
            sys.exit(1)
        return

    try:
        is_tty = sys.stdin.isatty()
    except (ValueError, OSError):
        if explicit_stdin:
            console.print(
                "[bold red]Error:[/bold red] --stdin was passed but stdin "
                "state could not be determined."
            )
            sys.exit(1)
        return

    if is_tty:
        if explicit_stdin:
            console.print(
                "[bold red]Error:[/bold red] --stdin was passed but stdin "
                "is a terminal. Pipe input or use -n instead.\n"
                "  cat prompt.txt | dcode --stdin -q"
            )
            sys.exit(1)
        return

    max_stdin_bytes = 10 * 1024 * 1024  # 10 MiB

    try:
        stdin_text = sys.stdin.read(max_stdin_bytes + 1)
    except UnicodeDecodeError:
        msg = "Could not read piped input — ensure the input is valid text"
        console.print(f"[bold red]Error:[/bold red] {msg}")
        sys.exit(1)
    except (OSError, ValueError) as exc:
        from rich.markup import escape

        console.print(
            f"[bold red]Error:[/bold red] Failed to read piped input: "
            f"{escape(str(exc))}"
        )
        sys.exit(1)

    if len(stdin_text) > max_stdin_bytes:
        msg = (
            f"Piped input exceeds {max_stdin_bytes // (1024 * 1024)} MiB limit. "
            "Consider writing the content to a file and referencing it instead."
        )
        console.print(f"[bold red]Error:[/bold red] {msg}")
        sys.exit(1)

    stdin_text = stdin_text.strip()

    if not stdin_text:
        return

    # Priority: -n message > -m prompt > --skill (no -m) > fallback to -n.
    # The initial_prompt branch uses `is not None` (not truthiness) so that
    # `-m ""` is distinguished from "no -m at all", allowing stdin to land
    # in initial_prompt even when the explicit value is empty.  The --skill
    # branch only fires when -m was NOT provided; when both -m and --skill
    # are set, stdin merges with the -m value (previous branch).
    if args.non_interactive_message:
        args.non_interactive_message = f"{stdin_text}\n\n{args.non_interactive_message}"
    elif args.initial_prompt is not None:
        if args.initial_prompt:
            args.initial_prompt = f"{stdin_text}\n\n{args.initial_prompt}"
        else:
            args.initial_prompt = stdin_text
    elif getattr(args, "initial_skill", None):
        args.initial_prompt = stdin_text
    else:
        args.non_interactive_message = stdin_text

    # Restore stdin from the real terminal so the interactive Textual app
    # (used by the -m path) can read keyboard/mouse input normally.
    # Textual's driver reads from file descriptor 0 directly (not sys.stdin),
    # so we must replace the underlying fd with /dev/tty using os.dup2.
    try:
        tty_fd = os.open("/dev/tty", os.O_RDONLY)
    except OSError:
        # No controlling terminal (CI, Docker, headless). Non-interactive
        # path still works; interactive -m path will fail later with a
        # clear "not a terminal" error from Textual.
        return

    try:
        os.dup2(tty_fd, 0)
        os.close(tty_fd)
        sys.stdin = open(0, encoding="utf-8", closefd=False)  # noqa: SIM115  # fd 0 requires open() for TTY restoration
    except OSError:
        console.print(
            "[yellow]Warning:[/yellow] TTY restoration failed. "
            "Interactive mode (-m) may not work correctly."
        )
        logger.warning(
            "TTY restoration failed after opening /dev/tty",
            exc_info=True,
        )
        try:
            os.close(tty_fd)
        except OSError:
            logger.warning(
                "Failed to close TTY fd %d during cleanup",
                tty_fd,
                exc_info=True,
            )


def _print_session_stats(stats: Any, console: Any) -> None:  # noqa: ANN401
    """Print a session-level usage stats table to the console on TUI exit.

    Args:
        stats: The cumulative session stats from the Textual app.
        console: Rich console for output.
    """
    from deepagents_code.textual_adapter import SessionStats, print_usage_table

    if not isinstance(stats, SessionStats):
        return
    print_usage_table(stats, stats.wall_time_seconds, console)


def _debug_mcp_project_trust_enabled() -> bool:
    """Return whether the project MCP approval prompt debug path is enabled."""
    from deepagents_code._env_vars import DEBUG_MCP_PROJECT_TRUST, is_env_truthy

    return is_env_truthy(DEBUG_MCP_PROJECT_TRUST)


def _check_mcp_project_trust(*, trust_flag: bool = False) -> bool | None:
    """Check whether project-level MCP servers should be trusted.

    Both stdio and remote (http/sse) project entries require approval —
    remote entries from an attacker-controlled `.mcp.json` can SSRF or
    exfiltrate environment variables via `${VAR}` interpolation in their
    `headers`, so they are gated identically to stdio commands.

    When the project has no servers in project-level configs, returns
    `None` (no gate needed). When `--trust-project-mcp` was passed,
    returns `True`. Otherwise checks the persistent trust store; if
    untrusted, shows an interactive approval prompt.

    Args:
        trust_flag: Whether `--trust-project-mcp` was passed.

    Returns:
        `True` to allow project servers, `False` to deny, or `None`
            when no project servers exist.
    """
    from deepagents_code.mcp_tools import (
        classify_discovered_configs,
        discover_mcp_configs,
        extract_project_server_summaries,
        load_mcp_config_lenient,
        merge_mcp_configs,
    )
    from deepagents_code.project_utils import ProjectContext

    debug_prompt = _debug_mcp_project_trust_enabled()

    try:
        project_context = ProjectContext.from_user_cwd(Path.cwd())
        config_paths = discover_mcp_configs(project_context=project_context)
    except (OSError, RuntimeError):
        return None

    _, project_configs = classify_discovered_configs(config_paths)
    if not project_configs and not debug_prompt:
        return None

    # Merge configs by server name (last wins, matching the loader) so that
    # a server defined in multiple project configs (for example,
    # `.deepagents/.mcp.json` and higher-precedence `.mcp.json`) only shows
    # up once in the prompt.
    loaded_configs = [
        cfg
        for cfg in (load_mcp_config_lenient(path) for path in project_configs)
        if cfg is not None
    ]
    merged_config = merge_mcp_configs(loaded_configs)
    all_servers = extract_project_server_summaries(merged_config)

    if not all_servers and debug_prompt:
        all_servers = [
            (
                "debug-project-mcp",
                "stdio",
                "uvx deepagents-debug-mcp --sample-project-server",
            )
        ]

    if not all_servers:
        return None

    if trust_flag:
        return True

    # Check trust store
    from deepagents_code.mcp_trust import (
        compute_config_fingerprint,
        is_project_mcp_trusted,
        trust_project_mcp,
    )

    project_root = str(
        (project_context.project_root or project_context.user_cwd).resolve()
    )
    fingerprint = compute_config_fingerprint(project_configs)

    if not debug_prompt and is_project_mcp_trusted(project_root, fingerprint):
        return True

    # Interactive prompt
    from rich.console import Console as _Console

    docs_url = (
        "https://docs.langchain.com/oss/python/deepagents/code/"
        "mcp-tools#project-level-trust"
    )
    prompt_console = _Console(stderr=True)
    prompt_console.print()
    prompt_console.print(
        "[bold yellow]Project MCP servers require approval:[/bold yellow]"
    )
    for name, kind, summary in all_servers:
        prompt_console.print(f'  [bold]"{name}"[/bold] ({kind}):  {summary}')
    prompt_console.print()
    prompt_console.print(
        f"[dim]Learn more: [link={docs_url}]{docs_url}[/link][/dim]",
        highlight=False,
    )
    prompt_console.print()

    try:
        answer = input("Allow? [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = ""

    if answer == "y":
        if not debug_prompt and not trust_project_mcp(project_root, fingerprint):
            prompt_console.print(
                "[yellow]Approved for this session, but the decision could not be "
                "saved — you'll be asked again next time.[/yellow]",
                highlight=False,
            )
        return True
    return False


def _verify_interpreter_or_exit() -> None:
    """Run the interpreter pre-flight check; print and exit on failure.

    Called before spawning the langgraph dev server subprocess so a missing
    `langchain-quickjs` dependency surfaces a one-line, actionable hint instead
    of an opaque "Server process exited with code N" downstream. Gated on the
    resolved interpreter state (`_resolve_interpreter_enabled`), not the
    `--interpreter` flag alone, since the interpreter is now on by default.
    """
    from deepagents_code.extras_info import verify_interpreter_deps

    try:
        verify_interpreter_deps()
    except ImportError as exc:
        from rich.markup import escape

        from deepagents_code.config import console

        console.print(f"[bold red]Error:[/bold red] {escape(str(exc))}")
        sys.exit(1)


def cli_main() -> None:
    """Entry point for console script."""
    # Fix for gRPC fork issue on macOS
    # https://github.com/grpc/grpc/issues/37642
    if sys.platform == "darwin":
        os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"

    # Note: LANGSMITH_PROJECT override is handled lazily by config.py's
    # _ensure_bootstrap() (triggered on first access of `settings`).
    # This ensures agent traces use DEEPAGENTS_CODE_LANGSMITH_PROJECT while
    # shell commands use the user's original LANGSMITH_PROJECT.

    # Fast path: print version without loading heavy dependencies
    if len(sys.argv) == 2 and sys.argv[1] in {"-v", "--version"}:  # noqa: PLR2004  # argv length check for fast-path
        print(build_version_text())  # noqa: T201  # Version output
        sys.exit(0)

    # ACP mode does not require Textual, so skip UI dependency checks when
    # the flag is present in raw argv.
    if "--acp" not in sys.argv[1:]:
        check_cli_dependencies()

    try:
        args = parse_args()

        if _show_bare_command_group_help(args):
            return

        # Keep self-contained commands that do not need global settings here, before
        # state migration and settings bootstrap. If a future command only reads
        # local files or delegates bootstrap to specific subcommands, dispatch it here
        # so lightweight diagnostic paths stay fast.
        # Use `getattr` because this fast-path block is for optional top-level
        # subcommands only. ACP/root-mode invocations may not define `command`,
        # and should fall through to the later handlers instead of raising here.
        command = getattr(args, "command", None)
        if command == "config":
            from deepagents_code.config_commands import run_config_command

            sys.exit(run_config_command(args))

        if command == "auth" and getattr(args, "auth_command", None) == "path":
            from deepagents_code.auth_commands import run_auth_command

            sys.exit(run_auth_command(args))

        if command == "doctor":
            from deepagents_code.doctor import run_doctor_command

            sys.exit(run_doctor_command(args))

        if command == "tools":
            from deepagents_code.tools_commands import run_tools_command

            sys.exit(run_tools_command(args))

        # Best-effort, idempotent migration. Placed after parse_args and the
        # bare-help fast path so --help / --version / `deepagents <group>`
        # exit before any I/O. Wrapped broadly so an unexpected non-OSError
        # (e.g., RuntimeError from `Path.home()` when $HOME is unset on a CI
        # runner) cannot crash startup — state migration has zero functional
        # value vs. failing-soft.
        try:
            from deepagents_code.state_migration import migrate_legacy_state

            migrate_legacy_state()
        except Exception:
            logger.warning(
                "Legacy state migration failed unexpectedly; continuing.",
                exc_info=True,
            )

        # Import console/settings AFTER arg parsing and after the bare-help
        # fast path so neither argparse's `--help`/`-h` exit nor
        # `deepagents <group>` pays the settings bootstrap cost.
        from deepagents_code.config import console, settings

        if command == "auth":
            from deepagents_code.auth_commands import run_auth_command

            sys.exit(run_auth_command(args))

        model_params: dict[str, Any] | None = None
        raw_kwargs = getattr(args, "model_params", None)
        if raw_kwargs:
            try:
                model_params = json.loads(raw_kwargs)
            except json.JSONDecodeError as e:
                console.print(
                    f"[bold red]Error:[/bold red] --model-params is not valid JSON: {e}"
                )
                sys.exit(1)
            if not isinstance(model_params, dict):
                console.print(
                    "[bold red]Error:[/bold red] --model-params must be a JSON object"
                )
                sys.exit(1)

        max_retries = getattr(args, "max_retries", None)
        if max_retries is not None:
            from deepagents_code.config import CLI_MAX_RETRIES_KEY

            if model_params is None:
                model_params = {}
            # Carry the flag value under an internal key; `create_model` folds it
            # under the resolved provider's retry-param name (which may not be
            # `max_retries` for custom providers) with top precedence.
            model_params[CLI_MAX_RETRIES_KEY] = max_retries

        profile_override: dict[str, Any] | None = None
        raw_profile = getattr(args, "profile_override", None)
        if raw_profile:
            try:
                profile_override = json.loads(raw_profile)
            except json.JSONDecodeError as e:
                console.print(
                    "[bold red]Error:[/bold red] "
                    f"--profile-override is not valid JSON: {e}"
                )
                sys.exit(1)
            if not isinstance(profile_override, dict):
                console.print(
                    "[bold red]Error:[/bold red] "
                    "--profile-override must be a JSON object"
                )
                sys.exit(1)

        if getattr(args, "acp", False):
            assistant_id = _resolve_agent_arg(args)
            try:
                from acp import run_agent as run_acp_agent
                from deepagents_acp.server import AgentServerACP
            except ImportError as exc:
                msg = (
                    f"ACP dependencies not available: {exc}\n"
                    "Install with: uv tool install --reinstall -U deepagents-code "
                    "--with deepagents-acp\n"
                )
                sys.stderr.write(msg)
                sys.stderr.flush()
                sys.exit(1)

            if getattr(args, "no_mcp", False) and getattr(args, "mcp_config", None):
                msg = (
                    "Error: --no-mcp and --mcp-config are mutually exclusive."
                    " Use one or the other.\n"
                    "  dcode --mcp-config path/to/config.json\n"
                    "  dcode --no-mcp\n"
                )
                sys.stderr.write(msg)
                sys.stderr.flush()
                sys.exit(2)

            exit_code = asyncio.run(
                _run_acp_cli_async(
                    assistant_id=assistant_id,
                    run_acp_agent=run_acp_agent,
                    agent_server_cls=AgentServerACP,
                    model_name=getattr(args, "model", None),
                    model_params=model_params,
                    profile_override=profile_override,
                    mcp_config_path=getattr(args, "mcp_config", None),
                    no_mcp=getattr(args, "no_mcp", False),
                    trust_project_mcp=getattr(args, "trust_project_mcp", False),
                )
            )
            sys.exit(exit_code)

        # Apply shell-allow-list from command line if provided (overrides env var)
        if args.shell_allow_list:
            from deepagents_code.config import parse_shell_allow_list

            settings.shell_allow_list = parse_shell_allow_list(args.shell_allow_list)

        apply_stdin_pipe(args)

        if getattr(args, "no_mcp", False) and getattr(args, "mcp_config", None):
            from rich.console import Console as _Console

            _Console(stderr=True).print(
                "[bold red]Error:[/bold red] --no-mcp and --mcp-config "
                "are mutually exclusive. Use one or the other.\n"
                "  dcode --mcp-config path/to/config.json\n"
                "  dcode --no-mcp"
            )
            sys.exit(2)

        if (
            getattr(args, "initial_skill", None)
            and not args.non_interactive_message
            and (args.quiet or args.no_stream)
        ):
            from rich.console import Console as _Console

            _Console(stderr=True).print(
                "[bold red]Error:[/bold red] --skill requires "
                "--non-interactive (-n) when combined with --quiet or "
                "--no-stream.\n"
                "  dcode --skill code-review -m 'review this patch'\n"
                "  dcode --skill code-review -n 'review this patch'"
            )
            sys.exit(2)

        max_turns_set = getattr(args, "max_turns", None) is not None
        if max_turns_set and not args.non_interactive_message:
            from rich.console import Console as _Console

            _Console(stderr=True).print(
                "[bold red]Error:[/bold red] --max-turns requires "
                "--non-interactive (-n) or piped stdin\n"
                "  dcode -n 'refactor auth module' --max-turns 5"
            )
            sys.exit(2)

        timeout_set = getattr(args, "timeout", None) is not None
        if timeout_set and not args.non_interactive_message:
            from rich.console import Console as _Console

            _Console(stderr=True).print(
                "[bold red]Error:[/bold red] --timeout requires "
                "--non-interactive (-n) or piped stdin\n"
                "  dcode -n 'run the test suite' --timeout 120"
            )
            sys.exit(2)

        # `--goal` conflicts with every rubric flag, not just `--rubric`.
        # `--rubric-model`/`--rubric-max-iterations` also require `-n` (see the
        # non-interactive guard below), so without this check `--goal
        # --rubric-model X` would slip past here and hit a contradictory "add
        # -n" error — and adding `-n` then trips the interactive-only `--goal`
        # guard. Reject the combination up front instead.
        if getattr(args, "goal", None) is not None and any(
            getattr(args, attr, None) is not None
            for attr in ("rubric", "rubric_model", "rubric_max_iterations")
        ):
            from rich.console import Console as _Console

            _Console(stderr=True).print(
                "[bold red]Error:[/bold red] --goal is mutually exclusive with "
                "--rubric/--rubric-model/--rubric-max-iterations. Use --goal to "
                "generate criteria interactively, or --rubric (with -n) to "
                "provide them directly."
            )
            sys.exit(2)

        goal_text = getattr(args, "goal", None)
        if goal_text is not None and not goal_text.strip():
            from rich.console import Console as _Console

            _Console(stderr=True).print(
                "[bold red]Error:[/bold red] --goal must not be empty."
            )
            sys.exit(2)
        if goal_text is not None and args.non_interactive_message:
            from rich.console import Console as _Console

            _Console(stderr=True).print(
                "[bold red]Error:[/bold red] --goal is only supported in "
                "interactive mode for now.\n"
                "  dcode --goal 'add OAuth refresh handling'"
            )
            sys.exit(2)
        if goal_text is not None and (
            getattr(args, "initial_prompt", None) is not None
            or getattr(args, "initial_skill", None)
        ):
            from rich.console import Console as _Console

            _Console(stderr=True).print(
                "[bold red]Error:[/bold red] --goal cannot be combined with "
                "-m/--message or --skill.\n"
                "  dcode --goal 'add OAuth refresh handling'"
            )
            sys.exit(2)

        non_interactive_rubric_set = any(
            getattr(args, attr, None) is not None
            for attr in (
                "rubric",
                "rubric_model",
                "rubric_max_iterations",
            )
        )
        if non_interactive_rubric_set and not args.non_interactive_message:
            from rich.console import Console as _Console

            _Console(stderr=True).print(
                "[bold red]Error:[/bold red] --rubric/--rubric-model/"
                "--rubric-max-iterations require "
                "--non-interactive (-n) or piped stdin\n"
                "  dcode -n 'implement X' --rubric 'tests pass'"
            )
            sys.exit(2)

        if (args.quiet or args.no_stream) and not args.non_interactive_message:
            # Print to stderr (not the module-level stdout console) and exit
            # with code 2 to match the POSIX convention for usage errors, as
            # argparse's parser.error() would.
            from rich.console import Console as _Console

            flags = []
            if args.quiet:
                flags.append("--quiet")
            if args.no_stream:
                flags.append("--no-stream")
            flag = " and ".join(flags)
            _Console(stderr=True).print(
                f"[bold red]Error:[/bold red] {flag} requires "
                "--non-interactive (-n) or piped stdin\n"
                "  dcode -n 'summarize README.md' --quiet"
            )
            sys.exit(2)

        if args.prerelease and not (args.update or args.command == "update"):
            from rich.console import Console as _Console

            _Console(stderr=True).print(
                "[bold red]Error:[/bold red] --prerelease requires --update "
                "or the update subcommand"
            )
            sys.exit(2)

        # Handle --update flag or `update` subcommand (headless, no session)
        if args.update or args.command == "update":
            try:
                from rich.markup import escape

                from deepagents_code._env_vars import DEBUG_UPDATE
                from deepagents_code._version import __version__ as cli_version
                from deepagents_code.config import _is_editable_install
                from deepagents_code.update_check import (
                    _PRERELEASE_UNSUPPORTED_MESSAGE,
                    create_update_log_path,
                    format_age_suffix,
                    format_installed_age_suffix,
                    format_release_age_parenthetical,
                    is_update_available,
                    perform_upgrade,
                    prerelease_upgrade_supported,
                    release_requires_prereleases,
                    upgrade_command,
                )

                if _is_editable_install():
                    age_suffix = format_age_suffix(cli_version)
                    console.print(
                        "[bold yellow]Warning:[/bold yellow] "
                        "Updates are not available for editable installs. "
                        f"Currently on v{cli_version}{age_suffix}."
                    )
                    sys.exit(0)

                include_prereleases = True if args.prerelease else None

                # Refuse pre-release upgrades the install method can't honor
                # before promising an upgrade or hitting PyPI.
                if args.prerelease:
                    supported, reason = prerelease_upgrade_supported()
                    if not supported:
                        console.print(
                            "[bold red]Error:[/bold red] "
                            f"{reason or _PRERELEASE_UNSUPPORTED_MESSAGE}"
                        )
                        sys.exit(1)

                console.print("Checking for updates...", style="dim")
                available, latest = is_update_available(
                    bypass_cache=True,
                    include_prereleases=include_prereleases,
                )
                if latest is None:
                    console.print(
                        "[bold yellow]Warning:[/bold yellow] Could not "
                        "determine the latest version. Check your network "
                        "and try again."
                    )
                    sys.exit(1)
                if not available:
                    age_suffix = format_age_suffix(cli_version)
                    console.print(
                        f"Already on the latest version (v{cli_version}{age_suffix})."
                    )
                    sys.exit(0)

                upgrade_include_prereleases = include_prereleases
                pin_upgrade_version: str | None = None
                if include_prereleases is None and release_requires_prereleases(latest):
                    upgrade_include_prereleases = True
                    pin_upgrade_version = latest
                if upgrade_include_prereleases is True:
                    supported, reason = prerelease_upgrade_supported()
                    if not supported:
                        console.print(
                            "[bold red]Error:[/bold red] "
                            f"{reason or _PRERELEASE_UNSUPPORTED_MESSAGE}"
                        )
                        sys.exit(1)

                release_age = format_release_age_parenthetical(latest)
                installed_age = format_installed_age_suffix(cli_version)
                console.print(
                    f"Update available: v{latest}{release_age}. "
                    f"Currently installed: {cli_version}{installed_age}. "
                    "Upgrading..."
                )
                if os.environ.get(DEBUG_UPDATE):
                    console.print("Skipped update install (debug mode).", style="dim")
                    sys.exit(0)
                log_path = create_update_log_path()
                console.print(
                    f"Update log: {log_path}\nTail progress: tail -f {log_path}",
                    style="dim",
                    highlight=False,
                    markup=False,
                )
                success, output = asyncio.run(
                    perform_upgrade(
                        log_path=log_path,
                        include_prereleases=include_prereleases,
                        target_version=latest,
                    )
                )
                if success:
                    console.print(f"[green]Updated to v{latest}.[/green]")
                else:
                    cmd = upgrade_command(
                        include_prereleases=upgrade_include_prereleases,
                        version=pin_upgrade_version,
                    )
                    detail = f": {escape(output[:200])}" if output else ""
                    console.print(
                        f"[bold red]Auto-update failed{detail}[/bold red]\n"
                        f"Run manually: [cyan]{cmd}[/cyan]"
                    )
                    sys.exit(1)
                sys.exit(0)
            except Exception:
                logger.warning("--update failed", exc_info=True)
                # Preserve the user's pre-release intent in the manual fallback:
                # a `--prerelease` request that crashes unexpectedly must not
                # suggest a stable-only command, which would silently downgrade
                # the channel. Both are module-level string constants, so this
                # import can't fail inside the last-resort handler.
                from deepagents_code.update_check import (
                    _UV_PRERELEASE_UPGRADE_COMMAND,
                    FALLBACK_UPGRADE_COMMAND,
                )

                manual_cmd = (
                    _UV_PRERELEASE_UPGRADE_COMMAND
                    if args.prerelease
                    else FALLBACK_UPGRADE_COMMAND
                )
                console.print(
                    "[bold red]Error:[/bold red] Update failed.\n"
                    f"Run manually: [cyan]{manual_cmd}[/cyan]"
                )
                sys.exit(1)

        if args.package and not args.install:
            console.print(
                "[bold red]Error:[/bold red] --package requires --install <package>.",
            )
            sys.exit(2)

        # Handle --install <package> --package flag (headless, no session).
        # Installs an arbitrary package via `uv --with` for a custom provider,
        # rather than a deepagents-code extra. Always exits.
        if args.install and args.package:
            from rich.markup import escape

            from deepagents_code.config import _is_editable_install
            from deepagents_code.update_check import (
                create_update_log_path,
                editable_package_hint,
                is_valid_package_name,
                perform_install_package,
            )

            package: str = args.install
            pkg_log_path: Path | None = None
            try:
                if not is_valid_package_name(package):
                    # Defense in depth — the package is interpolated into a
                    # shell command. Reject malformed names before any prompt
                    # or uv call, even with --yes.
                    console.print(
                        f"[bold red]Error:[/bold red] "
                        f"Invalid package name '{escape(package)}'. "
                        "Package names must be alphanumeric with `-`, `_`, "
                        "or `.` (PEP 508).",
                        highlight=False,
                    )
                    sys.exit(2)
                if _is_editable_install():
                    console.print(
                        "[bold yellow]Warning:[/bold yellow] "
                        "--install --package is not supported on editable "
                        "installs.\n" + escape(editable_package_hint(package)),
                        highlight=False,
                    )
                    sys.exit(1)

                # Arbitrary packages have no curated allowlist to vet against,
                # so confirm before pulling third-party code into the tool env.
                console.print(
                    f"This will install the package '{escape(package)}' into "
                    "the Deep Agents Code environment (this runs third-party "
                    "code).",
                    highlight=False,
                )
                if not args.yes:
                    if not sys.stdin.isatty():
                        console.print(
                            "[bold red]Error:[/bold red] "
                            "Refusing package install in non-interactive mode. "
                            "Pass --yes to proceed."
                        )
                        sys.exit(2)
                    try:
                        reply = input(f"Install package '{package}'? [y/N] ")
                    except EOFError:
                        console.print("\nAborted.", style="dim")
                        sys.exit(130)
                    if reply.strip().lower() not in {"y", "yes"}:
                        console.print("Aborted.", style="dim")
                        sys.exit(1)

                console.print(f"Installing package '{package}'...")
                pkg_log_path = create_update_log_path()
                console.print(
                    f"Install log: {pkg_log_path}\n"
                    f"Tail progress: tail -f {pkg_log_path}",
                    style="dim",
                    highlight=False,
                    markup=False,
                )
                success, output = asyncio.run(
                    perform_install_package(package, log_path=pkg_log_path)
                )
                if success:
                    console.print(f"[green]Installed package '{package}'.[/green]")
                    sys.exit(0)
                # Tail the last 200 chars — uv prints the resolved error at the
                # end. The full output is in the log.
                detail = f": {output[-200:]}" if output else ""
                console.print(
                    f"[bold red]Install failed[/bold red]{escape(detail)}\n"
                    f"Log: {pkg_log_path}",
                    markup=True,
                    highlight=False,
                )
                sys.exit(1)
            except KeyboardInterrupt:
                console.print("\nAborted.", style="dim")
                sys.exit(130)
            except Exception as exc:
                logger.warning("--install --package failed", exc_info=True)
                log_line = f"\nLog: {pkg_log_path}" if pkg_log_path else ""
                console.print(
                    f"[bold red]Error:[/bold red] "
                    f"{type(exc).__name__}: {escape(str(exc))}"
                    f"{escape(log_line)}",
                    markup=True,
                    highlight=False,
                )
                sys.exit(1)

        # Handle --install <extra> flag (headless, no session)
        if args.install:
            from rich.markup import escape

            from deepagents_code.config import _is_editable_install
            from deepagents_code.extras_info import (
                KNOWN_EXTRAS,
                ExtrasIntrospectionError,
            )
            from deepagents_code.update_check import (
                ToolRequirementIntrospectionError,
                create_update_log_path,
                editable_extra_hint,
                install_extra_command,
                install_extra_recovery_command,
                install_extras_command,
                is_valid_extra_name,
                perform_install_extra,
            )

            extra: str = args.install
            log_path: Path | None = None
            manual_cmd: str | None = None
            try:
                if not is_valid_extra_name(extra):
                    # Defense in depth — the extra is interpolated into a
                    # shell command. Reject malformed names before any
                    # confirmation prompt, even with --yes.
                    console.print(
                        f"[bold red]Error:[/bold red] "
                        f"Invalid extra name '{escape(extra)}'. "
                        "Extra names must be alphanumeric with `-`, `_`, "
                        "or `.` (PEP 508).",
                        highlight=False,
                    )
                    sys.exit(2)
                if _is_editable_install():
                    console.print(
                        "[bold yellow]Warning:[/bold yellow] "
                        "--install is not supported on editable installs.\n"
                        + escape(editable_extra_hint(extra)),
                        highlight=False,
                    )
                    sys.exit(1)

                manual_cmd = install_extra_command(extra)
                # KNOWN_EXTRAS is a curated "did you mean" list, not the
                # authoritative set (that's pyproject, resolved by uv): warn and
                # confirm rather than refuse, since valid-but-unlisted names
                # exist (e.g. all-providers). Malformed names blocked above.
                if extra not in KNOWN_EXTRAS:
                    known = ", ".join(sorted(KNOWN_EXTRAS))
                    console.print(
                        f"[bold yellow]Warning:[/bold yellow] "
                        f"'{extra}' is not a known extra.\n"
                        f"Known extras: {known}",
                        highlight=False,
                    )
                    if not args.yes:
                        if not sys.stdin.isatty():
                            console.print(
                                "[bold red]Error:[/bold red] "
                                "Refusing unknown extra in non-interactive "
                                "mode. Pass --yes to override."
                            )
                            sys.exit(2)
                        reply = input("Continue anyway? [y/N] ").strip().lower()
                        if reply not in {"y", "yes"}:
                            console.print("Aborted.", style="dim")
                            sys.exit(1)

                console.print(f"Installing extra '{extra}'...")
                log_path = create_update_log_path()
                console.print(
                    f"Install log: {log_path}\nTail progress: tail -f {log_path}",
                    style="dim",
                    highlight=False,
                    markup=False,
                )
                success, output = asyncio.run(
                    perform_install_extra(extra, log_path=log_path)
                )
                if success:
                    console.print(f"[green]Installed extra '{extra}'.[/green]")
                    sys.exit(0)
                # Tail the last 200 chars — uv resolver prints the resolved
                # error at the end, not the beginning.
                detail = f": {output[-200:]}" if output else ""
                try:
                    manual_cmd = install_extra_recovery_command(extra)
                except (
                    ExtrasIntrospectionError,
                    ToolRequirementIntrospectionError,
                    ValueError,
                ):
                    logger.warning(
                        "--install recovery command failed (install reported failure)",
                        exc_info=True,
                    )
                    # Keep the install-script command bound above; fall back to a
                    # bare extras command only if that was never set.
                    manual_cmd = manual_cmd or install_extras_command((extra,))
                console.print(
                    f"[bold red]Install failed[/bold red]{escape(detail)}\n"
                    f"Log: {log_path}\n"
                    f"Run manually: [cyan]{escape(manual_cmd)}[/cyan]",
                    markup=True,
                    highlight=False,
                )
                sys.exit(1)
            except KeyboardInterrupt:
                console.print("\nAborted.", style="dim")
                sys.exit(130)
            except Exception as exc:
                logger.warning("--install failed", exc_info=True)
                log_line = f"\nLog: {log_path}" if log_path else ""
                # This is the catch-all for any unexpected install failure, so
                # the recovery-hint guard is intentionally broad too: it must
                # never raise a second error over the original one. `manual_cmd`
                # may be unset here (the failure could predate its assignment),
                # so fall back to a bare extras command.
                try:
                    fallback_cmd = install_extra_recovery_command(extra)
                except Exception:  # best-effort hint, never re-raise here
                    logger.warning(
                        "--install recovery command failed (unexpected error)",
                        exc_info=True,
                    )
                    fallback_cmd = manual_cmd or install_extras_command((extra,))
                console.print(
                    f"[bold red]Error:[/bold red] "
                    f"{type(exc).__name__}: {escape(str(exc))}"
                    f"{escape(log_line)}\n"
                    f"Run manually: [cyan]{escape(fallback_cmd)}[/cyan]",
                    markup=True,
                    highlight=False,
                )
                sys.exit(1)

        # Handle --auto-update flag (headless toggle: reads current state
        # and inverts it, no session)
        if args.auto_update:
            try:
                from deepagents_code.config import _is_editable_install
                from deepagents_code.update_check import (
                    is_auto_update_enabled,
                    set_auto_update,
                )

                if _is_editable_install():
                    console.print(
                        "[bold yellow]Warning:[/bold yellow] "
                        "Auto-updates are not available for editable installs."
                    )
                    sys.exit(1)

                currently_enabled = is_auto_update_enabled()
                new_state = not currently_enabled
                set_auto_update(new_state)
                label = "enabled" if new_state else "disabled"
                console.print(f"Auto-updates {label}.")
            except OSError:
                logger.warning("--auto-update failed: filesystem error", exc_info=True)
                console.print(
                    "[bold red]Error:[/bold red] Failed to toggle auto-updates. "
                    "Check permissions for ~/.deepagents/"
                )
                sys.exit(1)
            except Exception:
                logger.warning("--auto-update failed", exc_info=True)
                console.print(
                    "[bold red]Error:[/bold red] Failed to toggle auto-updates."
                )
                sys.exit(1)
            sys.exit(0)

        # Handle --default-model / --clear-default-model (headless, no session)
        if args.clear_default_model:
            from deepagents_code.model_config import clear_default_model

            if clear_default_model():
                console.print("Default model cleared.")
            else:
                console.print(
                    "[bold red]Error:[/bold red] Could not clear default model. "
                    "Check permissions for ~/.deepagents/"
                )
                sys.exit(1)
            sys.exit(0)

        if args.default_model is not None:
            from deepagents_code.model_config import (
                ModelConfig,
                save_default_model,
            )

            if args.default_model == "__SHOW__":
                config = ModelConfig.load()
                if config.default_model:
                    console.print(f"Default model: {config.default_model}")
                else:
                    console.print("No default model set.")
                sys.exit(0)

            model_spec = args.default_model
            # Auto-detect provider for bare model names
            from deepagents_code.config import detect_provider
            from deepagents_code.model_config import ModelSpec

            parsed = ModelSpec.try_parse(model_spec)
            if not parsed:
                provider = detect_provider(model_spec)
                if provider:
                    model_spec = f"{provider}:{model_spec}"

            if save_default_model(model_spec):
                console.print(f"Default model set to {model_spec}")
            else:
                console.print(
                    "[bold red]Error:[/bold red] Could not save default model. "
                    "Check permissions for ~/.deepagents/"
                )
                sys.exit(1)
            sys.exit(0)

        output_format = getattr(args, "output_format", "text")

        if args.command == "help":
            from deepagents_code.ui import show_help

            show_help()
        elif args.command == "agents":
            from deepagents_code.agent import list_agents, reset_agent
            from deepagents_code.ui import show_agents_help

            # "ls" is an argparse alias for "list"
            if args.agents_command in {"list", "ls"}:
                list_agents(output_format=output_format)
            elif args.agents_command == "reset":
                reset_agent(
                    args.agent,
                    args.source_agent,
                    dry_run=args.dry_run,
                    output_format=output_format,
                )
            else:
                show_agents_help()
        elif args.command == "skills":
            from deepagents_code.skills import execute_skills_command

            execute_skills_command(args)
        elif args.command == "mcp":
            from deepagents_code.mcp_commands import run_mcp_config, run_mcp_login
            from deepagents_code.ui import show_mcp_help

            if args.mcp_command == "login":
                config_path = args.config_path or args.mcp_config
                if config_path and not args.config_path:
                    print(  # noqa: T201
                        f"Using --mcp-config from top-level: {config_path}",
                        file=sys.stderr,
                    )
                sys.exit(
                    asyncio.run(
                        run_mcp_login(
                            server=args.server,
                            config_path=config_path,
                        )
                    )
                )
            if args.mcp_command == "config":
                sys.exit(run_mcp_config())
            show_mcp_help()
        elif args.command == "threads":
            from deepagents_code.sessions import (
                delete_thread_command,
                list_threads_command,
            )
            from deepagents_code.ui import show_threads_help

            # "ls" is an argparse alias for "list" — argparse stores the
            # alias as-is in the namespace, so we must match both values.
            if args.threads_command in {"list", "ls"}:
                raw_cwd = getattr(args, "cwd", None)
                cwd_filter = _normalize_cwd_filter(raw_cwd)
                # Warn (but still query) when the user passed an explicit
                # `--cwd <path>` that does not exist on disk — otherwise a
                # typo would silently return "No threads found" with no hint.
                # Skip the check for the bare-flag (empty string) form, where
                # the path was generated from `Path.cwd()` and is known to
                # exist.
                if (
                    raw_cwd not in {None, ""}
                    and cwd_filter is not None
                    and not Path(cwd_filter).exists()
                ):
                    print(  # noqa: T201
                        f"Warning: --cwd path {cwd_filter!r} does not exist; "
                        "filtering by stored metadata anyway.",
                        file=sys.stderr,
                    )
                asyncio.run(
                    list_threads_command(
                        agent_name=getattr(args, "agent", None),
                        limit=getattr(args, "limit", None),
                        sort_by=getattr(args, "sort", None),
                        branch=getattr(args, "branch", None),
                        cwd=cwd_filter,
                        verbose=getattr(args, "verbose", False),
                        relative=getattr(args, "relative", None),
                        output_format=output_format,
                    )
                )
            elif args.threads_command == "delete":
                asyncio.run(
                    delete_thread_command(
                        args.thread_id,
                        dry_run=args.dry_run,
                        output_format=output_format,
                    )
                )
            else:
                # No subcommand provided, show threads help screen
                show_threads_help()
        elif args.non_interactive_message:
            # Resolve recent-agent fallback only for actual session launches.
            assistant_id = _resolve_agent_arg(args)
            # Check for optional tools before running agent (stderr so
            # --quiet piped output stays clean)
            try:
                from rich.console import Console as _Console
            except ImportError:
                logger.warning(
                    "Could not import rich.console; skipping tool warnings",
                    exc_info=True,
                )
            else:
                warn_console = None
                try:
                    warn_console = _Console(stderr=True)
                    missing_tools = check_optional_tools()
                    if _should_ensure_managed_ripgrep():
                        missing_tools = _auto_install_ripgrep_cli(
                            warn_console, missing_tools
                        )
                    for tool in missing_tools:
                        warn_console.print(
                            f"[yellow]Warning:[/yellow] {format_tool_warning_cli(tool)}"
                        )
                except Exception:
                    logger.warning(
                        "Optional-tools check failed unexpectedly", exc_info=True
                    )
                    # A swallowed failure here must not be fully silent: surface
                    # one stderr line so a degraded grep is at least signposted.
                    if warn_console is not None:
                        with contextlib.suppress(Exception):
                            warn_console.print(
                                "[dim]Tool availability check skipped — see logs.[/dim]"
                            )
            # Validate sandbox provider deps before spawning server subprocess
            if args.sandbox and args.sandbox != "none":
                from deepagents_code.integrations.sandbox_factory import (
                    verify_sandbox_deps,
                )

                try:
                    verify_sandbox_deps(args.sandbox)
                except ImportError as exc:
                    from rich.markup import escape

                    console.print(f"[bold red]Error:[/bold red] {escape(str(exc))}")
                    sys.exit(1)

            enable_interpreter = _resolve_interpreter_enabled(args)
            if enable_interpreter:
                _verify_interpreter_or_exit()

            # Non-interactive mode - execute single task and exit
            from deepagents_code.non_interactive import run_non_interactive

            interpreter_ptc = _parse_interpreter_tools_flag(
                getattr(args, "interpreter_tools", None)
            )
            _warn_if_interpreter_tools_without_interpreter(
                args, enable_interpreter=enable_interpreter
            )
            _warn_if_interpreter_disabled_by_sandbox(args)

            try:
                rubric_text = _resolve_rubric_text(getattr(args, "rubric", None))
            except ValueError as exc:
                from rich.console import Console as _Console

                _Console(stderr=True).print(f"[bold red]Error:[/bold red] {exc}")
                sys.exit(2)

            timeout = getattr(args, "timeout", None)
            try:
                exit_code = asyncio.run(
                    asyncio.wait_for(
                        run_non_interactive(
                            message=args.non_interactive_message,
                            assistant_id=assistant_id,
                            model_name=getattr(args, "model", None),
                            model_params=model_params,
                            profile_override=profile_override,
                            sandbox_type=args.sandbox,
                            sandbox_id=args.sandbox_id,
                            sandbox_snapshot_name=args.sandbox_snapshot_name,
                            sandbox_setup=getattr(args, "sandbox_setup", None),
                            initial_skill=getattr(args, "initial_skill", None),
                            startup_cmd=getattr(args, "startup_cmd", None),
                            quiet=args.quiet,
                            stream=not args.no_stream,
                            mcp_config_path=getattr(args, "mcp_config", None),
                            no_mcp=getattr(args, "no_mcp", False),
                            trust_project_mcp=getattr(args, "trust_project_mcp", False),
                            enable_interpreter=enable_interpreter,
                            interpreter_ptc=interpreter_ptc,
                            max_turns=getattr(args, "max_turns", None),
                            rubric=rubric_text,
                            rubric_model=getattr(args, "rubric_model", None),
                            rubric_max_iterations=getattr(
                                args, "rubric_max_iterations", None
                            ),
                        ),
                        timeout=timeout,
                    )
                )
            except TimeoutError:
                # `asyncio.wait_for` raises `asyncio.TimeoutError`, which is
                # an alias of the builtin on Python >= 3.11 (the project's
                # minimum).
                from rich.console import Console as _Console

                _Console(stderr=True).print(
                    f"[bold red]Error:[/bold red] agent timed out after "
                    f"{timeout}s. Retry with a larger --timeout, or use "
                    "--max-turns for a turn-count limit."
                )
                sys.exit(124)
            except KeyboardInterrupt:
                # `asyncio.run` re-raises `KeyboardInterrupt` past the inner
                # `run_non_interactive` handler when the signal hits during
                # `wait_for`; mirror its exit code 130 here so Ctrl-C is a
                # quiet exit instead of a traceback.
                sys.exit(130)
            sys.exit(exit_code)
        else:
            _run_startup_auto_update(console)
            # Resolve recent-agent fallback only for actual session launches.
            assistant_id = _resolve_agent_arg(args)
            # Interactive mode - handle thread resume
            from rich.text import Text

            from deepagents_code.sessions import generate_thread_id

            # Instead of resolving thread_id here with synchronous asyncio.run()
            # DB calls, pass the raw resume request to the TUI and let it
            # resolve asynchronously during startup.
            resume_thread = args.resume_thread  # "__MOST_RECENT__", "<id>", or None
            thread_id = None if resume_thread else generate_thread_id()

            # Validate sandbox provider deps before spawning server subprocess
            if args.sandbox and args.sandbox != "none":
                from deepagents_code.integrations.sandbox_factory import (
                    verify_sandbox_deps,
                )

                try:
                    verify_sandbox_deps(args.sandbox)
                except ImportError as exc:
                    from rich.markup import escape

                    console.print(f"[bold red]Error:[/bold red] {escape(str(exc))}")
                    sys.exit(1)

            enable_interpreter = _resolve_interpreter_enabled(args)
            if enable_interpreter:
                _verify_interpreter_or_exit()

            # Check project MCP trust before launching TUI
            mcp_trust_decision = _check_mcp_project_trust(
                trust_flag=getattr(args, "trust_project_mcp", False),
            )
            if _debug_mcp_project_trust_enabled():
                sys.exit(0)

            # Run Textual TUI
            return_code = 0
            try:
                interpreter_ptc = _parse_interpreter_tools_flag(
                    getattr(args, "interpreter_tools", None)
                )
                # A stderr warning here would be clobbered by the alternate
                # screen the moment the TUI launches; the app surfaces the
                # advisory as a startup notification instead (see
                # `DeepAgentsApp._notify_interpreter_tools_without_interpreter`).

                result = asyncio.run(
                    run_textual_cli_async(
                        assistant_id=assistant_id,
                        auto_approve=args.auto_approve,
                        sandbox_type=args.sandbox,
                        sandbox_id=args.sandbox_id,
                        sandbox_snapshot_name=args.sandbox_snapshot_name,
                        sandbox_setup=getattr(args, "sandbox_setup", None),
                        model_name=getattr(args, "model", None),
                        model_params=model_params,
                        profile_override=profile_override,
                        thread_id=thread_id,
                        resume_thread=resume_thread,
                        initial_prompt=getattr(args, "initial_prompt", None),
                        initial_skill=getattr(args, "initial_skill", None),
                        initial_goal=getattr(args, "goal", None),
                        startup_cmd=getattr(args, "startup_cmd", None),
                        mcp_config_path=getattr(args, "mcp_config", None),
                        no_mcp=getattr(args, "no_mcp", False),
                        trust_project_mcp=mcp_trust_decision,
                        enable_interpreter=enable_interpreter,
                        interpreter_arg=args.interpreter,
                        interpreter_ptc=interpreter_ptc,
                    )
                )
                return_code = result.return_code
                # The user may have switched threads via /threads during the
                # session; use the final thread ID for teardown messages.
                thread_id = result.thread_id or thread_id
                _print_session_stats(result.session_stats, console)
            except Exception as e:  # noqa: BLE001  # Top-level error handler for the application
                error_msg = Text("\nApplication error: ", style="red")
                error_msg.append(str(e))
                console.print(error_msg)
                console.print(Text(traceback.format_exc(), style="dim"))
                sys.exit(1)

            # Show LangSmith thread link and resume hint for threads with
            # checkpointed content. The `thread_id is not None` check narrows the
            # type to `str` for the helper; `_should_check_teardown_thread` gates
            # whether the teardown lookup runs at all.
            if thread_id is not None and _should_check_teardown_thread(
                thread_id,
                request_count=result.session_stats.request_count,
                resume_thread=args.resume_thread,
            ):
                _render_teardown_thread_hints(
                    console, thread_id, return_code=return_code
                )

            # Warn about available update on exit
            try:
                if result.update_available[0]:
                    from deepagents_code._version import __version__ as cli_version
                    from deepagents_code.update_check import (
                        format_installed_age_suffix,
                        format_release_age_parenthetical,
                        is_auto_update_enabled,
                        is_installed_version_at_least,
                        mark_update_notified,
                        should_notify_update,
                        upgrade_command,
                    )

                    latest = result.update_available[1]
                    if (
                        latest
                        and not is_installed_version_at_least(latest)
                        and should_notify_update(latest)
                    ):
                        console.print()
                        release_age = format_release_age_parenthetical(latest)
                        installed_age = format_installed_age_suffix(cli_version)
                        update_msg = Text(
                            f"Update available: v{latest}", style="yellow bold"
                        )
                        update_msg.append(
                            f"{release_age}. "
                            f"Currently installed: {cli_version}{installed_age}.",
                            style="dim",
                        )
                        console.print(update_msg)
                        cmd_hint = Text("Run: ", style="dim")
                        cmd_hint.append(upgrade_command(), style="cyan")
                        console.print(cmd_hint)
                        if not is_auto_update_enabled():
                            auto_hint = Text("Enable auto-updates: ", style="dim")
                            auto_hint.append("dcode --auto-update", style="cyan")
                            console.print(auto_hint)
                        mark_update_notified(latest)
            except Exception:
                logger.warning("Failed to display exit update banner", exc_info=True)
    except KeyboardInterrupt:
        # Clean exit on Ctrl+C — suppress ugly traceback.
        # `console` may not be bound if Ctrl+C arrives during config import.
        try:
            console.print("\n\n[yellow]Interrupted[/yellow]")
        except NameError:
            sys.stderr.write("\n\nInterrupted\n")
        sys.exit(0)


if __name__ == "__main__":
    cli_main()
