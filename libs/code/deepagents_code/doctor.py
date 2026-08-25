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
    from deepagents_code.extras_info import VersionReport

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


def _sdk_diagnostic(report: VersionReport) -> tuple[str, bool]:
    """Return the doctor SDK version value and whether it is healthy.

    A genuinely missing package reads as `not installed`, an unexpected lookup
    failure as `unknown`, and a resolved version carries its editable/drift/
    mismatch annotation. A dependency-requirement mismatch or invalid editable
    source marker is unhealthy; drift between valid source and installed
    metadata alone is informational, since it is normal while editing SDK source.
    """
    from deepagents_code.extras_info import format_sdk_version_annotation

    sdk = report.sdk
    if sdk.status == "not_installed":
        return "not installed", False
    if sdk.status != "resolved":
        return "unknown", False
    value = f"{report.display_sdk_version}{format_sdk_version_annotation(report)}"
    return value, not (
        report.sdk_requirement_mismatch or report.sdk_source_version_invalid
    )


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
    from deepagents_code.extras_info import (
        collect_version_report,
        format_cli_version_annotation,
    )
    from deepagents_code.update_check import detect_install_method

    report = collect_version_report()
    sdk_version, sdk_ok = _sdk_diagnostic(report)

    # Source/metadata drift is informational (normal while editing source), so
    # it annotates the value without flagging the CLI item unhealthy. Reuse the
    # shared annotation helper so the phrasing stays in lockstep with
    # `--version`/`/version` (editable status is shown via `Install method`).
    cli_value = f"{__version__}{format_cli_version_annotation(report.cli)}"

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
            DiagnosticItem("deepagents-code", cli_value),
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

    The same four rows always render, so two `doctor` outputs stay
    line-comparable and the `--json` item labels are stable; `Update checks` and
    `Auto-updates` therefore report configuration as configured, even when this
    install never acts on it. The `Latest version` row carries the reason no
    cached answer is available, which is what makes it consistent with the
    `Last checked` stamp below it.

    Returns:
        The `Updates` section.
    """
    from deepagents_code.config import _is_editable_install
    from deepagents_code.update_check import (
        get_cached_update_available,
        get_last_update_check_time,
        is_auto_update_enabled,
        is_update_check_enabled,
    )

    editable = _is_editable_install()
    checks_enabled = is_update_check_enabled()
    # Read once and share with both rows below so they cannot straddle a
    # concurrent cache refresh and disagree.
    checked_at = get_last_update_check_time()
    if editable:
        auto_updates = "disabled (editable install)"
    else:
        auto_updates = "enabled" if is_auto_update_enabled() else "disabled"

    available, latest = get_cached_update_available()

    return DiagnosticSection(
        title="Updates",
        items=[
            DiagnosticItem(
                "Update checks",
                "enabled" if checks_enabled else "disabled",
            ),
            DiagnosticItem("Auto-updates", auto_updates),
            DiagnosticItem(
                "Latest version",
                _format_latest_version(
                    available,
                    latest,
                    editable=editable,
                    checks_enabled=checks_enabled,
                    checked_at=checked_at,
                ),
            ),
            DiagnosticItem("Last checked", _format_last_checked(checked_at)),
        ],
    )


def _format_latest_version(
    available: bool,
    latest: str | None,
    *,
    editable: bool,
    checks_enabled: bool,
    checked_at: float | None,
) -> str:
    """Describe the cached latest version, or why no answer is available.

    A cached answer is reported whenever one exists, since it is a true
    statement about the installed version. Otherwise the cause is named:
    editable installs and disabled checks never contact PyPI at all, so any
    stamp on disk was written by another install sharing the state directory.
    Failing that, the stamp separates an expired cache (repeated fetch failures
    leave it untouched) from one never written at all, and a current cache that
    still yields no answer is reported as incomplete rather than stale: it holds
    no entry this install can use, as when only pre-release pins were recorded
    or a pre-release install meets a stable-only payload.

    Args:
        available: Whether the cached answer is newer than the running version.
        latest: Cached latest version, or `None` when the cache holds no usable
            answer.
        editable: Whether this is an editable install.
        checks_enabled: Whether update checks are enabled by config and env.
        checked_at: Epoch time of the last recorded check, or `None`.

    Returns:
        The `Latest version` value.
    """
    from deepagents_code.update_check import is_update_cache_fresh

    if latest is not None:
        return f"v{latest} available" if available else "up to date"
    if editable:
        return "not checked (editable install)"
    if not checks_enabled:
        return "not checked (checks disabled)"
    if checked_at is None:
        return "unknown (never checked)"
    if not is_update_cache_fresh(checked_at):
        return "unknown (cache stale)"
    return "unknown (cache incomplete)"


def _format_last_checked(checked_at: float | None) -> str:
    """Return a relative description of the last update check, or `never`.

    `never` covers both the no-check-recorded case and, defensively, a stamp
    that cannot be formatted. `get_last_update_check_time` only returns finite,
    in-range epochs, so the formatting path does not raise here.

    Args:
        checked_at: Epoch time of the last recorded check, or `None`.

    Returns:
        The `Last checked` value.
    """
    from datetime import UTC, datetime

    from deepagents_code.sessions import format_relative_timestamp

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


def _endpoint_gateway_state(endpoint: str) -> str:
    """Classify a single tracing endpoint as gateway, non-gateway, or unknown.

    Traces sent to the managed gateway host (or a subdomain of it) route
    through LangSmith SaaS; any other host is a self-hosted or dev/staging
    target. The exact/subdomain comparison, and the case and root-dot
    normalization it needs, both live in
    `model_config.is_langsmith_gateway_host`.

    Args:
        endpoint: A configured tracing endpoint URL.

    Returns:
        `"yes"` when the endpoint's host is the LangSmith managed gateway (an
            exact host or a subdomain), `"no"` for any other resolvable host,
            and `"unknown"` when the endpoint cannot be parsed into a host — so
            a typo'd or malformed URL is never silently reported as `"no"`.
    """
    from deepagents_code.model_config import is_langsmith_gateway_host

    try:
        host = urlsplit(endpoint.strip()).hostname or ""
    except ValueError:
        # urlsplit raises on bracket-malformed IPv6 (e.g. `http://[::1`); a
        # diagnostic must degrade to "unknown" rather than crash `dcode doctor`.
        return "unknown"
    # A host of only a root dot carries no name, so it stays `"unknown"` rather
    # than being reported as a definite non-gateway.
    if not host.removesuffix("."):
        return "unknown"
    if is_langsmith_gateway_host(host):
        return "yes"
    return "no"


def _tracing_gateway_state(status: TracingStatus) -> str:
    """Report whether all trace ingestion targets are the managed gateway.

    Considers both the primary endpoint and any replica ingestion URLs
    (`LANGSMITH_RUNS_ENDPOINTS`), since a self-hosted replica means traces leave
    for a custom target even when the primary endpoint is unset. With no target
    configured, tracing falls back to the LangSmith SDK default
    (`https://api.smith.langchain.com`), which is the managed gateway.

    Args:
        status: The resolved tracing status.

    Returns:
        `"yes"` when every configured target is the managed gateway (or none is
            configured, i.e. the SDK default), `"no"` when any target is a
            self-hosted or dev/staging host, and `"unknown"` when a target
            cannot be parsed and none is a definite non-gateway host.
    """
    states = [
        _endpoint_gateway_state(target)
        for target in (status.endpoint, *status.runs_endpoints)
        if target
    ]
    if not states:
        return "yes"
    if "no" in states:
        return "no"
    if "unknown" in states:
        return "unknown"
    return "yes"


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
    self-hosted endpoint is a valid, healthy setup). When tracing is enabled, a
    `Gateway` item reports whether traces route through LangSmith's managed
    (SaaS) gateway (`yes`), a custom self-hosted/dev/staging endpoint (`no`), or
    an endpoint that could not be parsed (`unknown`), accounting for both the
    primary endpoint and any replica ingestion targets.

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
    if status.enabled:
        items.append(DiagnosticItem("Gateway", _tracing_gateway_state(status)))
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


def _writable_path_status(label: str, path: object) -> DiagnosticItem:
    """Build a path item that also reports whether the directory is writable.

    `classify_path` answers "is it there", which reports a present but
    root-owned directory as healthy. That is the single condition the profile
    fallbacks exist for, so the row that would send a user looking must not be
    the one that says `exists`.

    Uses `os.access` rather than `probe_writable`: a diagnostic must not create
    directories as a side effect of reporting on them.

    Args:
        label: Human-readable name for the path.
        path: Filesystem directory to probe.

    Returns:
        A diagnostic item describing the directory and its usability.
    """
    import os
    from pathlib import Path

    from deepagents_code._paths import PathState, classify_path

    resolved = Path(str(path))
    state = classify_path(resolved)
    if state is not PathState.EXISTS:
        return _path_status(label, path)
    if os.access(resolved, os.W_OK | os.X_OK):
        return DiagnosticItem(label, f"{resolved} (exists)", ok=True)
    return DiagnosticItem(
        label,
        f"{resolved} (exists but is not writable - the profile fallback is used)",
        ok=False,
    )


def _fallback_location_items() -> list[DiagnosticItem]:
    """Build the managed-bin and update-lock rows, naming the location in use.

    Returns:
        Rows for both preferred locations, each profile fallback that exists,
        and the managed binary actually on `PATH`.
    """
    from deepagents_code._paths import PathState, classify_path
    from deepagents_code.managed_tools import (
        BIN_DIR,
        FALLBACK_BIN_DIR,
        managed_rg_path,
    )
    from deepagents_code.update_check import FALLBACK_UPDATE_LOCK_FILE, UPDATE_LOCK_FILE

    items = [
        _writable_path_status("Managed binaries", BIN_DIR),
        _writable_path_status("Update locks", UPDATE_LOCK_FILE.parent),
    ]
    # Surface a profile-scoped fallback only once it exists, i.e. once it is
    # actually the location in use. The installation-scoped directories are
    # created lazily, so their absence is not evidence of a permission problem.
    for label, fallback in (
        ("Managed binaries (profile)", FALLBACK_BIN_DIR),
        ("Update locks (profile)", FALLBACK_UPDATE_LOCK_FILE.parent),
    ):
        if classify_path(fallback) is PathState.EXISTS:
            items.append(_path_status(label, fallback))
    # Two rows and no statement of which one wins is the gap this closes.
    rg_path = managed_rg_path()
    if classify_path(rg_path) is PathState.EXISTS:
        items.append(DiagnosticItem(label="Managed ripgrep", value=str(rg_path)))
    return items


def _managed_config_diagnostic() -> DiagnosticItem:
    """Report managed TOML location, parse health, and policy enforceability.

    Returns:
        Managed config diagnostic row.
    """
    from deepagents_code.configuration.resolver import (
        MANAGED_RANK,
        resolver_from_snapshots,
    )
    from deepagents_code.configuration.service import (
        get_managed_snapshot,
        managed_snapshot_health,
    )
    from deepagents_code.configuration.types import (
        TomlSnapshot,
    )

    snapshot = get_managed_snapshot(refresh=True)
    user = TomlSnapshot.absent("config.toml")
    status = resolver_from_snapshots(managed=snapshot, user=user).provider_statuses()[
        MANAGED_RANK
    ]
    # `status` round-trips from a provider seeded with this same snapshot, so
    # re-wrapping it was a no-op that read as meaningful -- and the one place
    # `TomlSnapshot.__post_init__` could raise inside the command whose job is
    # to survive a broken config.
    health = managed_snapshot_health(snapshot)
    path = status.path or "(unknown)"
    suffix = status.health.value.lower()
    detail = f" - {status.detail}" if status.detail else ""
    # Doctor exists to explain a failure, so it must carry the parse detail and
    # say who can fix it. Without this a user who just saw exit 78 learns
    # nothing new here.
    hint = "" if status.usable else "; ask your administrator to repair or remove it"
    # A file that parses is not necessarily enforceable, and both halves of
    # exit 78 have to show up here: reporting only `usable` gives a green row
    # to the `ManagedPolicyError` half. Status and violations are both derived
    # from the single `snapshot` read above, so a refreshed status can never be
    # paired with stale violations.
    violations = health.violations
    if violations:
        detail += f" - rejects {', '.join(violations)}"
        hint = "; ask your administrator to correct the value"
    if health.rejections:
        # Declared but ignored, which is not a launch failure and so not part of
        # `ok`. It still has to appear somewhere: the only other announcement is
        # a `logger.warning` that cannot reach stderr.
        detail += f" - ignores {', '.join(health.rejections)}"
    return DiagnosticItem(
        "Managed config",
        f"{path} ({suffix}){detail}{hint}",
        ok=health.ok,
    )


def _user_config_diagnostic() -> DiagnosticItem:
    """Report user TOML location and parse health.

    `_path_status` answers only "is there a file there", so a `config.toml`
    that exists and does not parse produced a green `exists` row - in the one
    command a user runs when their settings are not taking effect. Resolution
    treats an unparseable file as declaring nothing, which is exactly the state
    the row has to distinguish.

    Returns:
        User config diagnostic row.
    """
    from deepagents_code.configuration.providers import TomlFileProvider
    from deepagents_code.configuration.types import ProviderHealth
    from deepagents_code.model_config import DEFAULT_CONFIG_PATH

    status = TomlFileProvider("config.toml", DEFAULT_CONFIG_PATH).load().status
    suffix = {
        ProviderHealth.OK: "exists",
        ProviderHealth.MISSING: "not created",
    }.get(status.health, status.health.value.lower())
    detail = f" - {status.detail}" if status.detail else ""
    hint = "" if status.usable else "; every option in it falls back to its default"
    return DiagnosticItem(
        "Config file",
        f"{DEFAULT_CONFIG_PATH} ({suffix}){detail}{hint}",
        ok=status.usable,
    )


def _collect_configuration() -> DiagnosticSection:
    """Collect on-disk configuration and data locations.

    Returns:
        The `Configuration` section.
    """
    from deepagents_code._paths import PATHS
    from deepagents_code.model_config import DEFAULT_CONFIG_DIR

    items = [
        _path_status("Data directory", DEFAULT_CONFIG_DIR),
        _managed_config_diagnostic(),
        _user_config_diagnostic(),
        *_fallback_location_items(),
    ]
    if PATHS.home_check_skipped:
        items.append(
            DiagnosticItem(
                label="Profile safety check",
                value=(
                    "skipped - the home directory could not be resolved, so "
                    "DEEPAGENTS_HOME was not checked against it. Set $HOME."
                ),
                ok=False,
            )
        )
    if not PATHS.uses_default_profile:
        items.append(
            DiagnosticItem(label="Profile", value=f"{PATHS.profile.root} (configured)")
        )
    return DiagnosticSection(title="Configuration", items=items)


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
        "  Tip: Run `dcode config` or `dcode config get <key>` "
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
