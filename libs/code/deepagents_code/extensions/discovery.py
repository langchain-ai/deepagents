"""Resolve configured extension sources into an ordered file list."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from deepagents_code.extensions.models import (
    ExtensionFile,
    SourceInfo,
    UnitOrigin,
    UnitScope,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

logger = logging.getLogger(__name__)

_ENTRY_FILENAMES = ("__init__.py", "extension.py")
EXTENSIONS_DIRNAME = "extensions"


def user_extensions_dir() -> Path:
    """Return the global user extensions directory."""
    return Path.home() / ".deepagents" / EXTENSIONS_DIRNAME


def project_extensions_dir(project_root: Path) -> Path:
    """Return a project's extensions directory.

    Args:
        project_root: Root of the project being worked in.

    Returns:
        `<project_root>/.deepagents/extensions/`.
    """
    return project_root / ".deepagents" / EXTENSIONS_DIRNAME


def scan_extension_dir(directory: Path) -> list[tuple[Path, UnitOrigin]]:
    """Resolve one directory into extension entry files.

    Discovery is deliberately shallow: direct Python files are extensions, and
    a direct subdirectory is a package extension when it contains
    `__init__.py` or `extension.py`.

    Args:
        directory: Directory to scan.

    Returns:
        Entry files and origins sorted by path.
    """
    try:
        entries = sorted(directory.iterdir())
    except OSError:
        logger.debug("Could not scan extensions directory %s", directory, exc_info=True)
        return []

    resolved: list[tuple[Path, UnitOrigin]] = []
    for entry in entries:
        try:
            if entry.is_file() and entry.suffix == ".py":
                resolved.append((entry, UnitOrigin.TOP_LEVEL))
                continue
            if not entry.is_dir():
                continue
            for filename in _ENTRY_FILENAMES:
                candidate = entry / filename
                if candidate.is_file():
                    resolved.append((candidate, UnitOrigin.PACKAGE))
                    break
        except OSError:
            logger.debug("Skipping unreadable extension entry %s", entry, exc_info=True)
    return resolved


def _files_for_source(
    paths: Iterable[Path],
    scope: UnitScope,
) -> list[ExtensionFile]:
    """Resolve files directly and directories through the scan recipe.

    Args:
        paths: Configured extension files or directories.
        scope: Provenance scope for resolved entries.

    Returns:
        Discovered extension files in source order.
    """
    discovered: list[ExtensionFile] = []
    for path in paths:
        expanded = path.expanduser()
        try:
            is_file = expanded.is_file()
            is_dir = expanded.is_dir()
        except OSError:
            logger.debug("Could not stat extension path %s", expanded, exc_info=True)
            continue
        if is_file:
            discovered.append(
                ExtensionFile(
                    path=expanded,
                    source=SourceInfo(
                        path=expanded,
                        scope=scope,
                        origin=UnitOrigin.TOP_LEVEL,
                    ),
                )
            )
        elif is_dir:
            discovered.extend(
                ExtensionFile(
                    path=entry,
                    source=SourceInfo(path=entry, scope=scope, origin=origin),
                )
                for entry, origin in scan_extension_dir(expanded)
            )
    return discovered


def discover_extension_files(
    *,
    user_dir: Path | None = None,
    extra_paths: Sequence[Path] = (),
    project_dir: Path | None = None,
) -> list[ExtensionFile]:
    """Resolve every enabled source into one ordered file list.

    Callers must leave `project_dir` unset until project trust is resolved; this
    module intentionally performs no trust checks or implicit project lookup.

    Args:
        user_dir: Global directory; defaults to `~/.deepagents/extensions/`.
        extra_paths: Explicit user-configured files and directories.
        project_dir: Trusted project extension directory, when allowed.

    Returns:
        Files in load order, deduplicated by canonical path.
    """
    resolved_user_dir = user_extensions_dir() if user_dir is None else user_dir
    discovered = [
        *_files_for_source([resolved_user_dir], UnitScope.USER),
        *_files_for_source(extra_paths, UnitScope.USER),
    ]
    if project_dir is not None:
        discovered.extend(_files_for_source([project_dir], UnitScope.PROJECT))

    unique: list[ExtensionFile] = []
    seen: set[Path] = set()
    for candidate in discovered:
        try:
            key = candidate.path.resolve()
        except OSError:
            key = candidate.path
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique
