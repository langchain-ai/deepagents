"""The `dcode tools` command group: provision managed external tools.

Currently this exposes `dcode tools install`, which fetches the pinned,
SHA-256-verified ripgrep binary into `~/.deepagents/bin` (the same managed
path used on first run) and is also handy for repairing a missing or stale
`rg`. The install script calls this verb instead of re-encoding the pinned
version + checksum table in bash.

Help rendering for `dcode tools -h` / `dcode tools install -h` is served by
`ui.show_tools_help` / `ui.show_tools_install_help`, which do not import this
module, so the help path stays light.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from deepagents_code.output import write_json

if TYPE_CHECKING:
    import argparse

logger = logging.getLogger(__name__)


def run_tools_command(args: argparse.Namespace) -> int:
    """Dispatch a `dcode tools` subcommand.

    Args:
        args: Parsed CLI namespace.

    Returns:
        Process exit code.
    """
    subcommand = getattr(args, "tools_command", None)
    if subcommand == "install":
        return _run_tools_install(args)

    # `cli_main`'s bare-group help fast path handles `dcode tools` with no
    # subcommand, so this is only reached for an unexpected value.
    from deepagents_code import ui

    ui.show_tools_help()
    return 0


def _run_tools_install(args: argparse.Namespace) -> int:
    """Install or repair the managed ripgrep binary.

    Honors the same opt-outs as first-run startup (`DEEPAGENTS_CODE_OFFLINE`
    and `DEEPAGENTS_CODE_RIPGREP_INSTALLER=system`) so behavior stays
    consistent across entry points.

    Args:
        args: Parsed CLI namespace. Only `output_format` is read.

    Returns:
        `0` when a usable `rg` is available (installed, already current, or an
        intentional opt-out), `1` when an install was expected but failed.
    """
    from deepagents_code.managed_tools import (
        RIPGREP_VERSION,
        ChecksumMismatchError,
        ensure_ripgrep,
        is_offline,
        managed_rg_path,
        prefers_system_ripgrep,
        prepend_managed_bin_to_path,
    )

    output_format = getattr(args, "output_format", "text")
    managed_target = managed_rg_path()

    try:
        installed = asyncio.run(ensure_ripgrep())
    except ChecksumMismatchError:
        logger.exception(
            "ripgrep install aborted: SHA-256 mismatch on downloaded archive"
        )
        return _emit_install_result(
            output_format,
            status="error",
            ok=False,
            message=(
                "ripgrep install aborted: the downloaded archive failed SHA-256 "
                "verification. Refusing to install."
            ),
        )

    if installed is not None:
        prepend_managed_bin_to_path()
        if installed == managed_target:
            message = f"Managed ripgrep {RIPGREP_VERSION} ready at {installed}"
        else:
            message = f"Using ripgrep already on PATH at {installed}"
        return _emit_install_result(
            output_format,
            status="ok",
            ok=True,
            message=message,
            path=str(installed),
        )

    # `ensure_ripgrep` returned `None`: an intentional opt-out is a success
    # (nothing to do), while an unexpected failure is reported as an error.
    if prefers_system_ripgrep():
        return _emit_install_result(
            output_format,
            status="skipped",
            ok=True,
            message=(
                "Skipped managed ripgrep install: DEEPAGENTS_CODE_RIPGREP_INSTALLER"
                "=system. Install ripgrep with your package manager, or unset the "
                "variable to use the managed binary."
            ),
        )
    if is_offline():
        return _emit_install_result(
            output_format,
            status="skipped",
            ok=True,
            message=(
                "Skipped managed ripgrep install: DEEPAGENTS_CODE_OFFLINE is set. "
                "Unset it to download the managed binary."
            ),
        )
    return _emit_install_result(
        output_format,
        status="error",
        ok=False,
        message=(
            "Could not install ripgrep (unsupported platform or download failure). "
            "See logs, or install ripgrep manually."
        ),
    )


def _emit_install_result(
    output_format: str,
    *,
    status: str,
    ok: bool,
    message: str,
    path: str | None = None,
) -> int:
    """Print the install outcome as text or JSON and return its exit code.

    Args:
        output_format: `"json"` for machine-readable output, else text.
        status: Stable machine token (`"ok"`, `"skipped"`, or `"error"`).
        ok: Whether the outcome is healthy (drives the `0`/`1` exit code).
        message: Human-readable summary line.
        path: Resolved `rg` path when one is available.

    Returns:
        `0` when `ok`, otherwise `1`.
    """
    if output_format == "json":
        payload: dict[str, object] = {"status": status, "ok": ok, "message": message}
        if path is not None:
            payload["path"] = path
        write_json("tools install", payload)
    else:
        from deepagents_code.config import console

        style = "green" if ok else "bold red"
        console.print(message, style=style, markup=False)

    return 0 if ok else 1
