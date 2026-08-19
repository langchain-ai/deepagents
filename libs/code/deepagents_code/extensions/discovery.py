"""Discover configured Python extension files."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from deepagents_code._env_vars import EXPERIMENTAL, is_env_truthy
from deepagents_code.extensions.registry import SourceInfo

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from deepagents_code.plugins.models import PluginInstance

logger = logging.getLogger(__name__)
EXTENSIONS_DIRNAME = "extensions"


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


def discover_extension_files(
    *,
    plugins: Sequence[PluginInstance] = (),
    project_dir: Path | None = None,
) -> list[SourceInfo]:
    """Return ordered, canonically deduplicated extension sources.

    Args:
        plugins: Enabled, installed plugins whose manifests may declare Python
            entry files.
        project_dir: Trusted project extension directory, when allowed.

    Returns:
        Extension sources in deterministic load order.
    """
    if not is_env_truthy(EXPERIMENTAL):
        return []
    sources = [
        SourceInfo(
            path,
            is_package=path.name == "__init__.py",
            plugin_id=plugin.plugin_id,
        )
        for plugin in plugins
        if plugin.manifest is not None
        for path in plugin.manifest.python_extensions
    ]
    if project_dir is not None:
        expanded = project_dir.expanduser()
        try:
            if expanded.is_file():
                sources.append(SourceInfo(expanded))
            elif expanded.is_dir():
                sources.extend(_scan(expanded))
        except OSError:
            logger.debug("Could not inspect extension path %s", expanded, exc_info=True)

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
