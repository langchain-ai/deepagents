"""Best-effort runtime dependency floor check for editable dev installs.

Editable installs resolve dependencies once at install time and nothing
re-checks them afterwards, so after `pyproject.toml` floors are bumped on
`main` a stale editable venv silently runs new source against old deps.
This module detects editable installs via PEP 610 metadata and warns at
startup when an installed runtime dependency is older than the floor the
checkout declares. Released installs (uv tool / PyPI wheel) carry no
editable `direct_url.json` record and skip the check entirely, so end
users pay no startup cost here.
"""

from __future__ import annotations

import importlib.metadata
import logging
import tomllib
from dataclasses import dataclass
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement
from packaging.version import InvalidVersion, Version

from deepagents_code.config import _is_editable_install

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _FloorViolation:
    """A runtime dependency whose installed version is below the declared floor.

    Attributes:
        dist_name: Distribution name as passed to `importlib.metadata.version`.
        installed: The installed version string.
        required: The requirement spec as declared in `pyproject.toml`.
    """

    dist_name: str
    installed: str
    required: str


def _load_cli_requirements() -> list[str] | None:
    """Read the hard dependency list from the editable source checkout.

    The installed `Requires-Dist` metadata is frozen at install time, so a
    floor bump on `main` would not reach a stale editable venv through it.
    The checkout's own `pyproject.toml` is live, and editable installs keep
    the console entry point importing from the checkout, so the package
    directory's parent holds it.

    Returns:
        The `project.dependencies` entries, or `None` when the source
        checkout or its dependency list cannot be read.
    """
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        entries = data["project"]["dependencies"]
    except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError):
        logger.debug(
            "Could not read dependency floors from the source checkout",
            exc_info=True,
        )
        return None
    if not isinstance(entries, list) or not all(
        isinstance(entry, str) for entry in entries
    ):
        return None
    return entries


def _find_floor_violations(entries: list[str]) -> list[_FloorViolation]:
    """Compare each applicable requirement's floor against installed versions.

    Args:
        entries: Requirement strings from `project.dependencies`.

    Returns:
        One `_FloorViolation` per requirement whose installed version parses
        below its declared `>=`/`~=`/`==` floor. Requirements that fail to
        parse, whose marker does not apply to this environment, whose only
        equality specifier is a `==X.*` wildcard, or whose distribution is
        not installed are skipped.
    """
    violations: list[_FloorViolation] = []
    for entry in entries:
        try:
            req = Requirement(entry)
        except InvalidRequirement:
            logger.debug("Unparseable dependency entry %r; skipping", entry)
            continue
        except Exception:  # noqa: BLE001  # any unexpected parser failure skips the entry
            logger.debug("Unparseable dependency entry %r; skipping", entry)
            continue
        if req.marker is not None:
            try:
                if not req.marker.evaluate():
                    continue
            except Exception:  # an unevaluatable marker must not abort the whole check
                logger.debug(
                    "Could not evaluate marker for %r; skipping", entry, exc_info=True
                )
                continue
        # `==` pins act as floors here too: a hard pin (e.g. the SDK's
        # `deepagents==X.Y.Z`) still means "this version or drift" in an
        # editable venv, and an older-than-pinned install is exactly the
        # staleness this check exists to surface. `==X.*` wildcard equality
        # has no parseable single version, so it stays out.
        floors = [
            spec.version
            for spec in req.specifier
            if spec.operator in {">=", "~="}
            or (spec.operator == "==" and not spec.version.endswith(".*"))
        ]
        if not floors:
            continue
        try:
            installed = importlib.metadata.version(req.name)
        except importlib.metadata.PackageNotFoundError:
            # Workspace `[tool.uv.sources]` path deps and not-yet-installed
            # dists both land here; a missing dist is not a stale floor.
            logger.debug("Distribution %r is not installed; skipping", req.name)
            continue
        try:
            installed_version = Version(installed)
        except InvalidVersion:
            logger.debug("Unparseable installed version %r for %r", installed, req.name)
            continue
        floor = max(floors, key=_version_key)
        if installed_version < Version(floor):
            violations.append(
                _FloorViolation(
                    dist_name=req.name, installed=installed, required=str(req)
                )
            )
    return violations


def _version_key(version: str) -> tuple[int, ...]:
    """Return a sortable key for a floor version string.

    Args:
        version: A PEP 440 version string from a `>=`/`~=` specifier.

    Returns:
        The release tuple when parseable, else an empty tuple so malformed
        floors sort lowest and a parseable floor always wins the `max`.
    """
    try:
        return Version(version).release
    except InvalidVersion:
        return ()


def warn_if_editable_deps_stale() -> None:
    """Warn when an editable install runs against below-floor dependencies.

    Runs only for editable installs (PEP 610 `dir_info.editable: true`);
    released installs return immediately. Prints a warning naming each
    offending distribution with its installed and required versions plus
    the refresh command, then returns so startup continues. This is
    strictly best-effort: any unexpected failure degrades to a debug log
    and never raises.
    """
    try:
        if not _is_editable_install():
            return
        entries = _load_cli_requirements()
        if entries is None:
            return
        violations = _find_floor_violations(entries)
        if not violations:
            return
        from deepagents_code.config import _get_console

        lines = [
            (
                "[bold yellow]Warning:[/bold yellow] this editable dcode install "
                "is running against dependencies older than the floors declared "
                "in the checkout's pyproject.toml:"
            )
        ]
        lines.extend(
            f"  - {v.dist_name} {v.installed} is installed, but {v.required} "
            "is required"
            for v in violations
        )
        lines.append(
            "Refresh the dev venv:\n"
            "  [cyan]uv pip install --python "
            "~/.local/share/dcode-dev/bin/python -e <repo>/libs/code "
            "--upgrade[/cyan]\n"
            "Continuing anyway; behavior may be broken."
        )
        _get_console().print("\n".join(lines), highlight=False)
    except Exception:  # strictly best-effort: a check failure must never break startup
        logger.debug("Dependency floor check failed", exc_info=True)
