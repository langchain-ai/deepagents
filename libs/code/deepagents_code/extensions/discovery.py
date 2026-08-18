"""Discover configured Python extension files."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from deepagents_code.extensions.models import SourceInfo

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

logger = logging.getLogger(__name__)
EXTENSIONS_DIRNAME = "extensions"


def user_extensions_dir() -> Path:
    """Return the global user extensions directory."""
    return Path.home() / ".deepagents" / EXTENSIONS_DIRNAME


def project_extensions_dir(project_root: Path) -> Path:
    """Return the extension directory beneath `project_root`."""
    return project_root / ".deepagents" / EXTENSIONS_DIRNAME


def _scan(directory: Path) -> list[SourceInfo]:
    try:
        entries = sorted(directory.iterdir())
    except OSError:
        logger.debug("Could not scan extensions directory %s", directory, exc_info=True)
        return []

    sources: list[SourceInfo] = []
    for entry in entries:
        try:
            if entry.is_file() and entry.suffix == ".py":
                sources.append(SourceInfo(entry))
            elif entry.is_dir():
                for filename in ("__init__.py", "extension.py"):
                    candidate = entry / filename
                    if candidate.is_file():
                        sources.append(SourceInfo(candidate, is_package=True))
                        break
        except OSError:
            logger.debug("Skipping unreadable extension %s", entry, exc_info=True)
    return sources


def _resolve(paths: Iterable[Path]) -> list[SourceInfo]:
    sources: list[SourceInfo] = []
    for path in paths:
        expanded = path.expanduser()
        try:
            if expanded.is_file():
                sources.append(SourceInfo(expanded))
            elif expanded.is_dir():
                sources.extend(_scan(expanded))
        except OSError:
            logger.debug("Could not inspect extension path %s", expanded, exc_info=True)
    return sources


def discover_extension_files(
    *,
    user_dir: Path | None = None,
    extra_paths: Sequence[Path] = (),
    project_dir: Path | None = None,
) -> list[SourceInfo]:
    """Return ordered, canonically deduplicated extension sources.

    Args:
        user_dir: Global extension directory.
        extra_paths: Explicit user-configured files or directories.
        project_dir: Trusted project extension directory, when allowed.

    Returns:
        Extension sources in deterministic load order.
    """
    paths = [user_extensions_dir() if user_dir is None else user_dir, *extra_paths]
    sources = _resolve(paths)
    if project_dir is not None:
        sources.extend(_resolve([project_dir]))

    unique: list[SourceInfo] = []
    seen: set[Path] = set()
    for source in sources:
        try:
            key = source.path.resolve()
        except OSError:
            key = source.path
        if key not in seen:
            seen.add(key)
            unique.append(source)
    return unique
