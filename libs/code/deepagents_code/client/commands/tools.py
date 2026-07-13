"""The `dcode tools` command group: provision managed external tools.

`dcode tools install` fetches the pinned, SHA-256-verified ripgrep binary into
`~/.deepagents/bin` (the same managed path used on first run) and is also handy
for repairing a missing or stale `rg`. The install script calls this verb
instead of re-encoding the pinned version + checksum table in bash.

`dcode tools list` prints the tools available to the agent, grouped by source
(built-in tools, then per-server MCP tools), enumerated from the real tool
objects the agent binds so the output never drifts from what the model sees.

Help rendering for `dcode tools -h` / `dcode tools install -h` /
`dcode tools list -h` is served by `ui.show_tools_help` /
`ui.show_tools_install_help` / `ui.show_tools_list_help`, which do not import
this module, so the help path stays light.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Literal

from deepagents_code.output import write_json

if TYPE_CHECKING:
    import argparse

    from deepagents_code.output import OutputFormat
    from deepagents_code.tool_catalog import ToolCatalog, UnavailableServer

logger = logging.getLogger(__name__)

InstallStatus = Literal["ok", "skipped", "error"]
"""Stable machine token for the `tools install` outcome, surfaced via `--json`.

`ok` (installed or already current), `skipped` (intentional opt-out), and
`error` (an install was expected but failed). Only `error` is unhealthy, so it
alone drives a non-zero exit code."""


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
    if subcommand == "list":
        return _run_tools_list(args)

    # `cli_main`'s bare-group help fast path handles `dcode tools` with no
    # subcommand, so this is only reached for an unexpected value.
    from deepagents_code import ui

    ui.show_tools_help()
    return 0


def _run_tools_list(args: argparse.Namespace) -> int:
    """List the tools available to the agent, grouped by source.

    Enumerates the real tool objects the agent binds (see
    `tool_catalog.collect_catalog`) so names and descriptions never drift from
    what the model sees. The same runtime options that shape the agent's tool
    set are honored: the resolved interpreter setting controls whether `js_eval`
    is listed, and the MCP options (`--no-mcp`, `--mcp-config`,
    `--trust-project-mcp`) control MCP discovery. Those are top-level flags, so
    they must precede the subcommand (e.g. `dcode --no-mcp tools list`).

    MCP discovery is best-effort: the built-in tools always render. Servers that
    errored, need login, or are disabled are still reported (not hidden) so a
    user debugging a missing tool can see why it is absent. When discovery fails
    outright while an explicit `--mcp-config` was supplied, the command exits
    non-zero because the user's explicit request could not be satisfied.

    The exit code is not a complete health signal: only a discovery *failure*
    (missing/unparseable config) sets a non-zero code, and only for an explicit
    `--mcp-config`. An explicit config that parses but whose server is merely
    unreachable is surfaced as an `unavailable` entry with exit `0`. Scripts
    that need per-server health must inspect `unavailable`/`mcp_error` in the
    `--json` output, not the exit code alone.

    Args:
        args: Parsed CLI namespace. Reads `output_format`, `agent`,
            `interpreter`, `sandbox`, `no_mcp`, `mcp_config`, and
            `trust_project_mcp`.

    Returns:
        `0` on success (including best-effort MCP degradation); `1` when an
        explicit `--mcp-config` was given but MCP discovery failed.
    """
    from deepagents_code._constants import DEFAULT_AGENT_NAME
    from deepagents_code._server_config import _resolve_enable_interpreter
    from deepagents_code.main import _resolve_agent_arg
    from deepagents_code.tool_catalog import collect_catalog

    output_format: OutputFormat = getattr(args, "output_format", "text")
    mcp_config_path: str | None = getattr(args, "mcp_config", None)
    assistant_id = (
        _resolve_agent_arg(args) if hasattr(args, "agent") else DEFAULT_AGENT_NAME
    )
    enable_interpreter = _resolve_enable_interpreter(
        getattr(args, "interpreter", None), getattr(args, "sandbox", None)
    )
    catalog = collect_catalog(
        assistant_id=assistant_id,
        enable_interpreter=enable_interpreter,
        include_mcp=not getattr(args, "no_mcp", False),
        mcp_config_path=mcp_config_path,
        trust_project_mcp=_tools_list_project_mcp_trust(args),
    )

    # A failed *explicit* --mcp-config is a failed user request → non-zero exit;
    # best-effort auto-discovery failures stay exit 0 (built-ins still render).
    exit_code = 1 if catalog.mcp_error and mcp_config_path else 0

    if output_format == "json":
        tools_payload = [
            {
                "name": entry.name,
                "description": entry.description,
                "group": group.label,
                "source": group.source,
            }
            for group in catalog.groups
            for entry in group.tools
        ]
        write_json(
            "tools list",
            {
                "tools": tools_payload,
                "count": len(tools_payload),
                "unavailable": [
                    {
                        "name": server.name,
                        "status": server.status,
                        "detail": server.detail,
                    }
                    for server in catalog.unavailable
                ],
                "mcp_error": catalog.mcp_error,
            },
        )
        return exit_code

    _print_catalog(catalog)
    return exit_code


def _tools_list_project_mcp_trust(args: argparse.Namespace) -> bool | None:
    """Resolve project MCP trust behavior for `dcode tools list`.

    Args:
        args: Parsed CLI namespace.

    Returns:
        `True` when project MCP trust was explicitly requested, otherwise
        `None` so MCP discovery falls back to the user's per-server allow-list.
    """
    if getattr(args, "trust_project_mcp", False):
        return True
    return None


def _print_catalog(catalog: ToolCatalog) -> None:
    """Render a tool catalog to the console.

    Prints the count header, the tool groups, then any unavailable MCP servers
    and a discovery-failure notice.

    Args:
        catalog: Collected groups, unavailable servers, and discovery status.
    """
    from deepagents_code.config import console, get_glyphs

    ellipsis = get_glyphs().ellipsis
    total = sum(len(group.tools) for group in catalog.groups)
    noun = "tool" if total == 1 else "tools"

    console.print()
    console.print(f"{total} {noun} available", highlight=False)

    for group in catalog.groups:
        if not group.tools:
            continue
        name_width = max(len(entry.name) for entry in group.tools)
        # Indent (2) + name column + gap (2) precede the description; keep each
        # row on one line by truncating the description to the terminal width.
        desc_width = console.width - 2 - name_width - 2
        console.print()
        console.print(group.label, style="bold", markup=False, highlight=False)
        for entry in group.tools:
            padded = entry.name.ljust(name_width)
            description = _truncate(entry.description, desc_width, ellipsis)
            # `markup=False`/`highlight=False`: tool names and descriptions are
            # sourced from tool objects and may contain brackets or numbers.
            console.print(
                f"  {padded}  {description}".rstrip(),
                markup=False,
                highlight=False,
                no_wrap=True,
                crop=True,
            )

    _print_unavailable_servers(catalog.unavailable)

    if catalog.mcp_error:
        console.print()
        console.print(f"Note: {catalog.mcp_error}", style="yellow", highlight=False)

    console.print()


def _print_unavailable_servers(servers: tuple[UnavailableServer, ...]) -> None:
    """Render MCP servers that were discovered but expose no tools.

    Args:
        servers: Unavailable servers (errored, needing login, or disabled).
    """
    if not servers:
        return
    from deepagents_code.config import console

    name_width = max(len(server.name) for server in servers)
    console.print()
    console.print(
        "Unavailable MCP servers", style="bold", markup=False, highlight=False
    )
    for server in servers:
        padded = server.name.ljust(name_width)
        # `status: detail` (ASCII-only, no em-dash) so legacy consoles don't hit
        # an encoding error; detail is discovery's own curated reason string.
        detail = f": {server.detail}" if server.detail else ""
        console.print(
            f"  {padded}  {server.status}{detail}".rstrip(),
            style="dim",
            markup=False,
            highlight=False,
            no_wrap=True,
            crop=True,
        )


def _truncate(text: str, width: int, ellipsis: str) -> str:
    """Truncate `text` to `width` columns, appending `ellipsis` when clipped.

    Args:
        text: Description text to truncate.
        width: Maximum column width for the description.
        ellipsis: Marker appended when `text` is clipped.

    Returns:
        `text` unchanged when it fits, otherwise a clipped string ending in
        `ellipsis`.
    """
    if width <= 0 or len(text) <= width:
        return text
    if width <= len(ellipsis):
        return text[:width]
    return text[: width - len(ellipsis)].rstrip() + ellipsis


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
        ManagedToolUnavailableError,
        ensure_ripgrep,
        is_offline,
        managed_rg_path,
        prefers_system_ripgrep,
        prepend_managed_bin_to_path,
    )

    output_format: OutputFormat = getattr(args, "output_format", "text")
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
            message=(
                "ripgrep install aborted: the downloaded archive failed SHA-256 "
                "verification. Refusing to install."
            ),
        )
    except ManagedToolUnavailableError as exc:
        logger.info("ripgrep install unavailable: %s", exc.reason)
        return _emit_install_result(
            output_format,
            status="error",
            message=exc.message,
        )
    except Exception:
        # Backstop for a clean exit instead of a raw traceback.
        # `ensure_ripgrep` is defensive internally, but this is the only
        # `ensure_ripgrep` caller wired into `scripts/install.sh`, so an
        # unexpected escape must degrade to a structured error + exit 1
        # (matching the broad backstops in `app.py` / `main.py`) rather than
        # dumping a traceback and breaking the `--json` envelope.
        logger.warning("ripgrep install failed unexpectedly", exc_info=True)
        return _emit_install_result(
            output_format,
            status="error",
            message="ripgrep install failed unexpectedly. See logs for details.",
        )

    if installed is not None:
        if installed == managed_target:
            prepend_managed_bin_to_path()
            message = f"Managed ripgrep {RIPGREP_VERSION} ready at {installed}"
        else:
            message = f"Using ripgrep already on PATH at {installed}"
        return _emit_install_result(
            output_format,
            status="ok",
            message=message,
            path=str(installed),
        )

    # `ensure_ripgrep` returned `None`: an intentional opt-out is a success
    # (nothing to do), while an unexpected failure is reported as an error.
    if prefers_system_ripgrep():
        return _emit_install_result(
            output_format,
            status="skipped",
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
            message=(
                "Skipped managed ripgrep install: DEEPAGENTS_CODE_OFFLINE is set. "
                "Unset it to download the managed binary."
            ),
        )
    return _emit_install_result(
        output_format,
        status="error",
        message=(
            "Could not install ripgrep (download failure). See logs, or install "
            "ripgrep manually."
        ),
    )


def _emit_install_result(
    output_format: OutputFormat,
    *,
    status: InstallStatus,
    message: str,
    path: str | None = None,
) -> int:
    """Print the install outcome as text or JSON and return its exit code.

    `ok` is derived from `status` (only `"error"` is unhealthy) so the JSON
    envelope and exit code cannot disagree and no illegal `(status, ok)` pair
    is representable.

    Args:
        output_format: `"json"` for machine-readable output, else text.
        status: Stable machine token (`"ok"`, `"skipped"`, or `"error"`).
        message: Human-readable summary line.
        path: Resolved `rg` path when one is available.

    Returns:
        `0` for `"ok"`/`"skipped"`, `1` for `"error"`.
    """
    ok = status != "error"
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
