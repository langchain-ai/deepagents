"""CLI install path for optional deepagents-code extras.

`dcode install <name>` installs a curated optional extra (for example a sandbox
or model-provider dependency) into the current `dcode` environment. The legacy
global flags `dcode --install <name>` / `--package` / `--yes` remain as
compatible aliases and call the same implementation.

`tools install` is intentionally separate: that group provisions managed host
binaries (ripgrep), while this command manages Python package extras for dcode.

Help rendering for `dcode install -h` is served by `ui.show_install_help`, which
does not import this module, so the help path stays light.
"""

from __future__ import annotations

import asyncio
import logging
import shlex
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse
    from pathlib import Path

logger = logging.getLogger(__name__)


def _tail_log_command(log_path: Path | str) -> str:
    """Return a copy-pasteable command for following a log file."""
    return f"tail -f {shlex.quote(str(log_path))}"


def run_install_command(args: argparse.Namespace) -> int:
    """Dispatch `dcode install <name>`.

    Args:
        args: Parsed CLI namespace.

    Returns:
        Process exit code.
    """
    name = getattr(args, "install_target", None)
    if not isinstance(name, str) or not name:
        from deepagents_code import ui

        ui.show_install_help()
        return 2
    # Accept modifiers on the subcommand or as global root flags that precede
    # it, e.g. `dcode --yes install daytona`.
    package = bool(
        getattr(args, "install_package", False) or getattr(args, "package", False)
    )
    yes = bool(getattr(args, "install_yes", False) or getattr(args, "yes", False))
    return run_install_request(name=name, package=package, yes=yes)


def run_install_request(*, name: str, package: bool, yes: bool) -> int:
    """Install an optional extra or an arbitrary package via `uv --with`.

    Shared by `dcode install` and the legacy `dcode --install` flag.

    Args:
        name: Extra or package name to install.
        package: When `True`, install `name` as an arbitrary package rather than
            a `deepagents-code` extra.
        yes: Skip interactive confirmation prompts.

    Returns:
        Process exit code.
    """
    if package:
        return _run_install_package(name=name, yes=yes)
    return _run_install_extra(name=name, yes=yes)


def _run_install_package(*, name: str, yes: bool) -> int:
    """Install an arbitrary package via `uv --with` (custom provider path).

    Returns:
        Process exit code.
    """
    from rich.markup import escape

    from deepagents_code.config import _is_editable_install, console
    from deepagents_code.update_check import (
        create_update_log_path,
        editable_package_hint,
        is_valid_package_name,
        perform_install_package,
    )

    package = name
    pkg_log_path: Path | None = None
    try:
        if not is_valid_package_name(package):
            # Defense in depth — the package is interpolated into a shell
            # command. Reject malformed names before any prompt or uv call,
            # even with --yes.
            console.print(
                f"[bold red]Error:[/bold red] "
                f"Invalid package name '{escape(package)}'. "
                "Package names must be alphanumeric with `-`, `_`, "
                "or `.` (PEP 508).",
                highlight=False,
            )
            return 2
        if _is_editable_install():
            console.print(
                "[bold yellow]Warning:[/bold yellow] "
                "Package install is not supported on editable "
                "installs.\n" + escape(editable_package_hint(package)),
                highlight=False,
            )
            return 1

        # Arbitrary packages have no curated allowlist to vet against, so
        # confirm before pulling third-party code into the tool env.
        console.print(
            f"This will install the package '{escape(package)}' into "
            "the dcode environment (this runs third-party "
            "code).",
            highlight=False,
        )
        if not yes:
            if not sys.stdin.isatty():
                console.print(
                    "[bold red]Error:[/bold red] "
                    "Refusing package install in non-interactive mode. "
                    "Pass --yes to proceed."
                )
                return 2
            try:
                reply = input(f"Install package '{package}'? [y/N] ")
            except EOFError:
                console.print("\nAborted.", style="dim")
                return 130
            if reply.strip().lower() not in {"y", "yes"}:
                console.print("Aborted.", style="dim")
                return 1

        console.print(f"Installing package '{package}'...")
        pkg_log_path = create_update_log_path()
        console.print(
            f"Install log: {_tail_log_command(pkg_log_path)}",
            style="dim",
            highlight=False,
            markup=False,
        )
        success, output = asyncio.run(
            perform_install_package(package, log_path=pkg_log_path)
        )
        if success:
            console.print(f"[green]Installed package '{package}'.[/green]")
            return 0
        # Tail the last 200 chars — uv prints the resolved error at the end.
        # The full output is in the log.
        detail = f": {output[-200:]}" if output else ""
        console.print(
            f"[bold red]Install failed[/bold red]{escape(detail)}\nLog: {pkg_log_path}",
            markup=True,
            highlight=False,
        )
    except KeyboardInterrupt:
        console.print("\nAborted.", style="dim")
        return 130
    except Exception as exc:
        logger.warning("install --package failed", exc_info=True)
        log_line = f"\nLog: {pkg_log_path}" if pkg_log_path else ""
        console.print(
            f"[bold red]Error:[/bold red] "
            f"{type(exc).__name__}: {escape(str(exc))}"
            f"{escape(log_line)}",
            markup=True,
            highlight=False,
        )
        return 1
    else:
        return 1


def _run_install_extra(*, name: str, yes: bool) -> int:
    """Install a deepagents-code optional extra into the current env.

    Returns:
        Process exit code.
    """
    from rich.markup import escape

    from deepagents_code.config import _is_editable_install, console
    from deepagents_code.extras_info import KNOWN_EXTRAS
    from deepagents_code.update_check import (
        create_update_log_path,
        editable_extra_hint,
        install_extra_command,
        install_extras_command,
        is_valid_extra_name,
        perform_install_extra,
        safe_install_extra_recovery_command,
    )

    extra = name
    log_path: Path | None = None
    manual_cmd: str | None = None
    try:
        if not is_valid_extra_name(extra):
            # Defense in depth — the extra is interpolated into a shell
            # command. Reject malformed names before any confirmation
            # prompt, even with --yes.
            console.print(
                f"[bold red]Error:[/bold red] "
                f"Invalid extra name '{escape(extra)}'. "
                "Extra names must be alphanumeric with `-`, `_`, "
                "or `.` (PEP 508).",
                highlight=False,
            )
            return 2
        if _is_editable_install():
            console.print(
                "[bold yellow]Warning:[/bold yellow] "
                "Installing extras is not supported on editable installs.\n"
                + escape(editable_extra_hint(extra)),
                highlight=False,
            )
            return 1

        manual_cmd = install_extra_command(extra)
        # KNOWN_EXTRAS is a curated "did you mean" list, not the authoritative
        # set (that's pyproject, resolved by uv): warn and confirm rather than
        # refuse, since valid-but-unlisted names exist (e.g. all-providers).
        # Malformed names blocked above.
        if extra not in KNOWN_EXTRAS:
            known = ", ".join(sorted(KNOWN_EXTRAS))
            console.print(
                f"[bold yellow]Warning:[/bold yellow] "
                f"'{extra}' is not a known extra.\n"
                f"Known extras: {known}",
                highlight=False,
            )
            if not yes:
                if not sys.stdin.isatty():
                    console.print(
                        "[bold red]Error:[/bold red] "
                        "Refusing unknown extra in non-interactive "
                        "mode. Pass --yes to override."
                    )
                    return 2
                reply = input("Continue anyway? [y/N] ").strip().lower()
                if reply not in {"y", "yes"}:
                    console.print("Aborted.", style="dim")
                    return 1

        console.print(f"Installing extra '{extra}'...")
        log_path = create_update_log_path()
        console.print(
            f"Install log: {_tail_log_command(log_path)}",
            style="dim",
            highlight=False,
            markup=False,
        )
        success, output = asyncio.run(perform_install_extra(extra, log_path=log_path))
        if success:
            console.print(f"[green]Installed extra '{extra}'.[/green]")
            return 0
        # Tail the last 200 chars — uv resolver prints the resolved error at
        # the end, not the beginning.
        detail = f": {output[-200:]}" if output else ""
        # Best-effort upgrade of `manual_cmd` (set above via
        # `install_extra_command`) to the install-method-specific recovery
        # command. On failure, keep that already-bound install-script command
        # so the hint is never empty.
        manual_cmd = safe_install_extra_recovery_command(extra, fallback=manual_cmd)
        console.print(
            f"[bold red]Install failed[/bold red]{escape(detail)}\n"
            f"Log: {log_path}\n"
            f"Run manually: [cyan]{escape(manual_cmd)}[/cyan]",
            markup=True,
            highlight=False,
        )
    except KeyboardInterrupt:
        console.print("\nAborted.", style="dim")
        return 130
    except Exception as exc:
        logger.warning("install failed", exc_info=True)
        log_line = f"\nLog: {log_path}" if log_path else ""
        # Catch-all for any unexpected install failure: never let recovery-hint
        # generation raise a second error over the original one. `manual_cmd`
        # may still be unset if the failure predated `install_extra_command`,
        # so fall back to a bare extras command in that case.
        fallback_cmd = safe_install_extra_recovery_command(
            extra, fallback=manual_cmd or install_extras_command((extra,))
        )
        console.print(
            f"[bold red]Error:[/bold red] "
            f"{type(exc).__name__}: {escape(str(exc))}"
            f"{escape(log_line)}\n"
            f"Run manually: [cyan]{escape(fallback_cmd)}[/cyan]",
            markup=True,
            highlight=False,
        )
        return 1
    else:
        return 1
