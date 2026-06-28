"""The `dcode doctor` command: report install health and diagnostics.

Inspired by `claude doctor`, this prints a grouped, tree-style summary of the
running install, update status, and configuration locations so the output is
safe to paste into a bug report. It stays offline: the update section reads
only the local cache and never contacts PyPI.

Help rendering for `dcode doctor -h` is served by `ui.show_doctor_help`, which
does not import this module, so the help path stays light.
"""

from __future__ import annotations

import logging
import platform
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from deepagents_code.output import write_json

if TYPE_CHECKING:
    import argparse

    from deepagents_code.config import TracingStatus

logger = logging.getLogger(__name__)


@dataclass
class DiagnosticItem:
    """A single labeled diagnostic fact.

    `ok` is `False` only for genuine problems (e.g. a missing dependency), not
    for informational states such as an available update.
    """

    label: str
    value: str
    ok: bool = True


@dataclass
class DiagnosticSection:
    """A named group of related diagnostic items."""

    title: str
    items: list[DiagnosticItem] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Whether every item in the section is healthy."""
        return all(item.ok for item in self.items)


def _platform_tag() -> str:
    """Return a compact `<os>-<arch>` platform tag (e.g. `darwin-arm64`)."""
    return f"{platform.system()}-{platform.machine()}".lower()


def _sdk_version() -> tuple[str, bool]:
    """Return the installed `deepagents` SDK version and whether it resolved.

    Maps the shared `resolve_sdk_version` outcome onto doctor's display: a
    genuinely missing package reads as `not installed`, an unexpected lookup
    failure as `unknown`, and only a resolved version is marked healthy.
    """
    from deepagents_code.extras_info import resolve_sdk_version

    sdk_version, status = resolve_sdk_version()
    if status == "resolved":
        # `sdk_version` is a real string when the status is "resolved".
        return sdk_version or "unknown", True
    if status == "not_installed":
        return "not installed", False
    return "unknown", False


def _build_commit() -> str | None:
    """Return the commit stamped into the package at build time, if present.

    Released wheels carry a generated `_build_info.py` (see `hatch_build.py`);
    editable and local installs do not, so this returns `None` for them.
    """
    try:
        from deepagents_code._build_info import (  # ty: ignore[unresolved-import]  # generated at build time
            BUILD_COMMIT,
        )
    except ImportError:
        return None
    except Exception:  # a corrupt stamp must never crash `doctor`
        logger.debug("Build-info module present but failed to import", exc_info=True)
        return None
    commit = (BUILD_COMMIT or "").strip()
    return commit or None


def _commit_hash(path: str) -> str:
    """Return the short git commit hash for the install, if available.

    Prefers the commit stamped into a released wheel at build time, but only for
    non-editable installs: an editable install may carry a stale stamp from a
    prior local build (the generated file is gitignored and survives a failed
    build), so it always probes the live git working tree, which reflects local
    changes.

    Args:
        path: Directory used as the git command working directory.

    Returns:
        The short commit hash, or `unknown` when no commit can be determined.
    """
    baked = _build_commit()
    if baked:
        from deepagents_code.config import _is_editable_install

        # A baked commit only describes a built wheel; ignore it for editable
        # installs so a stale stamp can't mask the live working-tree commit.
        if not _is_editable_install():
            return baked

    import shutil
    import subprocess  # noqa: S404  # fixed-argv git metadata probe
    from pathlib import Path

    cwd = Path(path).expanduser()
    git = shutil.which("git")
    if git is None:
        return "unknown"
    try:
        git_path = Path(git).expanduser().resolve(strict=True)
        result = subprocess.run(  # noqa: S603  # fixed argv with absolute Git path
            [str(git_path), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
            cwd=cwd,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        logger.debug("Git commit hash detection failed", exc_info=True)
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip() or "unknown"


def _collect_diagnostics() -> DiagnosticSection:
    """Collect core version, platform, and install-location facts.

    Returns:
        The `Diagnostics` section.
    """
    from deepagents_code._version import __version__
    from deepagents_code.config import (
        _get_editable_install_path,
        _is_editable_install,
    )
    from deepagents_code.update_check import detect_install_method

    sdk_version, sdk_ok = _sdk_version()
    editable = _is_editable_install()
    if editable:
        method = "editable"
        path = _get_editable_install_path() or sys.prefix
    else:
        method = detect_install_method()
        path = sys.prefix

    return DiagnosticSection(
        title="Diagnostics",
        items=[
            DiagnosticItem("deepagents-code", __version__),
            DiagnosticItem("deepagents (SDK)", sdk_version, ok=sdk_ok),
            DiagnosticItem("Commit hash", _commit_hash(path)),
            DiagnosticItem("Python", platform.python_version()),
            DiagnosticItem("Platform", _platform_tag()),
            DiagnosticItem("Install method", method),
            DiagnosticItem("Path", path),
        ],
    )


def _collect_updates() -> DiagnosticSection:
    """Collect update-channel status from local config and the offline cache.

    Returns:
        The `Updates` section.
    """
    from deepagents_code.config import _is_editable_install
    from deepagents_code.update_check import (
        get_cached_update_available,
        is_auto_update_enabled,
        is_update_check_enabled,
    )

    items = [
        DiagnosticItem(
            "Update checks",
            "enabled" if is_update_check_enabled() else "disabled",
        ),
    ]
    if _is_editable_install():
        items.append(DiagnosticItem("Auto-updates", "disabled (editable install)"))
    else:
        items.append(
            DiagnosticItem(
                "Auto-updates",
                "enabled" if is_auto_update_enabled() else "disabled",
            )
        )

    available, latest = get_cached_update_available()
    if latest is None:
        update_status = "unknown (no recent check)"
    elif available:
        update_status = f"v{latest} available"
    else:
        update_status = "up to date"
    items.extend(
        (
            DiagnosticItem("Latest version", update_status),
            DiagnosticItem("Last checked", _format_last_checked()),
        )
    )

    return DiagnosticSection(title="Updates", items=items)


def _format_last_checked() -> str:
    """Return a relative description of the last update check, or `never`.

    `never` covers both the no-check-recorded case and, defensively, a stamp
    that cannot be formatted. `get_last_update_check_time` only returns finite,
    in-range epochs, so the formatting path does not raise here.
    """
    from datetime import UTC, datetime

    from deepagents_code.sessions import format_relative_timestamp
    from deepagents_code.update_check import get_last_update_check_time

    checked_at = get_last_update_check_time()
    if checked_at is None:
        return "never"
    iso = datetime.fromtimestamp(checked_at, tz=UTC).isoformat()
    return format_relative_timestamp(iso) or "never"


def _sanitize_endpoint(endpoint: str) -> str:
    """Return a paste-safe custom endpoint identifier.

    Args:
        endpoint: Configured endpoint URL.

    Returns:
        The endpoint origin when parseable, otherwise a generic configured
        marker that does not include user-controlled URL contents.
    """
    parsed = urlsplit(endpoint.strip())
    if not parsed.scheme or not parsed.hostname:
        return "(custom endpoint configured)"

    host = parsed.hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"

    try:
        port = parsed.port
    except ValueError:
        port = None

    netloc = f"{host}:{port}" if port is not None else host
    return f"{parsed.scheme}://{netloc}"


def _format_tracing_project(status: TracingStatus) -> str:
    """Render the tracing project, marking the unconfigured default.

    Returns:
        The project name with a `(default)` suffix when it is the built-in
            fallback rather than an explicit setting, or `(unset)` when absent.
    """
    if not status.project:
        return "(unset)"
    if status.project_is_default:
        return f"{status.project} (default)"
    return status.project


def _collect_tracing() -> DiagnosticSection:
    """Collect LangSmith tracing status from env and profile (offline).

    Tracing reads `enabled` when a flag is truthy, `disabled` only when a flag
    is explicitly set to a falsy value, and `not configured` when no flag is set.
    Credentials are reported as configured/not set only — the API key
    value is never read or printed. The `Credentials` item is flagged as a
    problem only when tracing is enabled without a key and without a custom
    endpoint, mirroring the runtime's orphaned-tracing guard (a keyless
    self-hosted endpoint is a valid, healthy setup).

    Returns:
        The `Tracing` section.
    """
    from deepagents_code.config import get_tracing_status

    status = get_tracing_status()
    creds_required = status.enabled and status.endpoint is None
    if status.enabled:
        tracing_value = "enabled"
    elif status.explicitly_disabled:
        tracing_value = "disabled"
    else:
        tracing_value = "not configured"
    items = [
        DiagnosticItem("Tracing", tracing_value),
        DiagnosticItem(
            "Credentials",
            "configured" if status.has_credentials else "not set",
            ok=status.has_credentials or not creds_required,
        ),
        DiagnosticItem("Project", _format_tracing_project(status)),
    ]
    if status.endpoint:
        items.append(DiagnosticItem("Endpoint", _sanitize_endpoint(status.endpoint)))
    if status.replica_project:
        items.append(DiagnosticItem("Replica project", status.replica_project))
    return DiagnosticSection(title="Tracing", items=items)


def _path_status(label: str, path: object) -> DiagnosticItem:
    """Build an item reporting a path and whether it exists on disk.

    An unreadable path (e.g. a parent directory that denies traversal) is
    flagged as a genuine problem (`ok=False`) so it surfaces in the section
    health and exit code, rather than being mistaken for a not-yet-created one.

    Args:
        label: Human-readable name for the path.
        path: Filesystem path to probe.

    Returns:
        A diagnostic item describing the path and its existence.
    """
    from pathlib import Path

    from deepagents_code._paths import PathState, classify_path

    resolved = Path(str(path))
    state = classify_path(resolved)
    suffix = {
        PathState.EXISTS: "exists",
        PathState.MISSING: "not created",
        PathState.UNREADABLE: "unreadable",
    }[state]
    return DiagnosticItem(
        label, f"{resolved} ({suffix})", ok=state is not PathState.UNREADABLE
    )


def _collect_configuration() -> DiagnosticSection:
    """Collect on-disk configuration and data locations.

    Returns:
        The `Configuration` section.
    """
    from deepagents_code.model_config import (
        DEFAULT_CONFIG_DIR,
        DEFAULT_CONFIG_PATH,
    )

    return DiagnosticSection(
        title="Configuration",
        items=[
            _path_status("Data directory", DEFAULT_CONFIG_DIR),
            _path_status("Config file", DEFAULT_CONFIG_PATH),
        ],
    )


def collect_sections() -> list[DiagnosticSection]:
    """Gather every diagnostic section in display order.

    Returns:
        The diagnostic sections, in render order.
    """
    return [
        _collect_diagnostics(),
        _collect_updates(),
        _collect_tracing(),
        _collect_configuration(),
    ]


def _tree_connectors() -> tuple[str, str]:
    """Return the `(tee, corner)` tree connectors for the active charset."""
    from deepagents_code.config import is_ascii_mode

    if is_ascii_mode():
        return "|-", "`-"
    return "\u251c", "\u2514"  # ├ └


def _render_text(sections: list[DiagnosticSection]) -> None:
    """Print the diagnostic sections as a styled tree to the console."""
    from rich.markup import escape

    from deepagents_code import theme
    from deepagents_code.config import console, get_glyphs

    glyphs = get_glyphs()
    tee, corner = _tree_connectors()

    console.print()
    for section in sections:
        status_glyph = glyphs.checkmark if section.ok else glyphs.warning
        status_color = theme.SUCCESS if section.ok else theme.WARNING
        console.print(
            f"  [bold]{escape(section.title)}[/bold] "
            f"[{status_color}]{status_glyph}[/{status_color}]"
        )
        for index, item in enumerate(section.items):
            connector = corner if index == len(section.items) - 1 else tee
            value_color = theme.MUTED if item.ok else "red"
            console.print(
                f"  {connector} {escape(item.label)}: "
                f"[{value_color}]{escape(item.value)}[/{value_color}]",
                highlight=False,
            )
        console.print()

    console.print(
        "  Tip: Run `dcode config show` or `dcode config get <key>` "
        "to drill into config details.",
        style=theme.MUTED,
        highlight=False,
    )
    console.print(
        "       Run `dcode --version` (or `dcode -v`) for dependency versions.",
        style=theme.MUTED,
        highlight=False,
    )
    console.print()


def run_doctor_command(args: argparse.Namespace) -> int:
    """Run `dcode doctor`, printing diagnostics as text or JSON.

    Args:
        args: Parsed CLI namespace. Only `output_format` is read.

    Returns:
        Process exit code: `0` when all sections are healthy, `1` otherwise.
    """
    sections = collect_sections()
    healthy = all(section.ok for section in sections)
    output_format = getattr(args, "output_format", "text")

    if output_format == "json":
        write_json(
            "doctor",
            {
                "healthy": healthy,
                "sections": [
                    {
                        "title": section.title,
                        "ok": section.ok,
                        "items": [
                            {
                                "label": item.label,
                                "value": item.value,
                                "ok": item.ok,
                            }
                            for item in section.items
                        ],
                    }
                    for section in sections
                ],
            },
        )
    else:
        _render_text(sections)

    return 0 if healthy else 1
