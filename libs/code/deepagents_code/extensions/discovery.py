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


def project_extensions_dir(project_root: Path) -> Path:
    """Return the extension directory beneath `project_root`."""
    return project_root / ".deepagents" / "extensions"


def _scan(directory: Path) -> list[SourceInfo]:
    try:
        entries = sorted(directory.iterdir())
    except OSError:
        logger.debug("Could not scan extensions directory %s", directory, exc_info=True)
        return []

    sources: list[SourceInfo] = []
    for entry in entries:
        if entry.is_file() and entry.suffix == ".py":
            sources.append(SourceInfo(entry))
            continue
        for filename in ("__init__.py", "extension.py"):
            candidate = entry / filename
            if candidate.is_file():
                sources.append(SourceInfo(candidate, is_package=True))
                break
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
        if expanded.is_file():
            sources.append(SourceInfo(expanded))
        else:
            sources.extend(_scan(expanded))

    unique: dict[Path, SourceInfo] = {}
    for source in sources:
        unique.setdefault(source.path, source)
    return list(unique.values())
