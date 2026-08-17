"""Resolution of extension sources into an ordered flat list of files.

Discovery is pure resolution: every source type is a recipe that turns a
directory (or an explicit path) into file paths. The loader consumes that list
and knows nothing about directories, manifests, or provenance, so adding a
source means adding a recipe here — never changing the loader.

Sources resolve in a fixed order (global user directory, configured extra
paths, then the trust-gated project directory), and entries within a directory
resolve alphabetically. That order is the only ordering in the system;
collision rules downstream simply follow it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from deepagents_code.extensions.models import (
    ExtensionFile,
    SourceInfo,
    UnitOrigin,
    UnitScope,
    UnitSource,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

logger = logging.getLogger(__name__)

_ENTRY_FILENAMES = ("__init__.py", "extension.py")
"""Entry files a directory extension may use, in preference order."""

EXTENSIONS_DIRNAME = "extensions"
"""Directory name holding extensions under `~/.deepagents/` and `.deepagents/`."""


def user_extensions_dir() -> Path:
    """Return the global extensions directory.

    Returns:
        `~/.deepagents/extensions/`.
    """
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

    The recipe is deliberately shallow: direct `*.py` files are extensions, and
    a subdirectory is an extension when it has an `__init__.py` or
    `extension.py` entry file. There is no deeper recursion, so a subdirectory
    of helper modules cannot be mistaken for several extensions.

    Args:
        directory: Directory to scan.

    Returns:
        Entry files paired with their origin, sorted by path.
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
    """Resolve configured paths (files as-is, directories by scan recipe).

    Args:
        paths: Files or directories to resolve.
        scope: Scope recorded on the resulting provenance.

    Returns:
        Discovered extension files in the order their sources were listed.
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
                        source=UnitSource.EXTENSION,
                        scope=scope,
                        origin=UnitOrigin.TOP_LEVEL,
                    ),
                )
            )
        elif is_dir:
            discovered.extend(
                ExtensionFile(
                    path=entry,
                    source=SourceInfo(
                        path=entry,
                        source=UnitSource.EXTENSION,
                        scope=scope,
                        origin=origin,
                    ),
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
    """Resolve every enabled source into one ordered list of entry files.

    Args:
        user_dir: Global extensions directory; defaults to
            `~/.deepagents/extensions/`.
        extra_paths: Files or directories from user configuration.
        project_dir: Project extensions directory. Callers must pass `None`
            until project trust has been resolved — this function performs no
            trust checks of its own.

    Returns:
        Entry files in load order, deduplicated by resolved path.
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
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique
