"""Discover configured Python extension files."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from deepagents_code._env_vars import EXPERIMENTAL, is_env_truthy
from deepagents_code.extensions.models import SourceInfo

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
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
                sources.append(
                    SourceInfo(
                        entry,
                        scope="project",
                        package_root=directory,
                    )
                )
            elif entry.is_dir():
                for filename in ("__init__.py", "extension.py"):
                    candidate = entry / filename
                    if candidate.is_file():
                        sources.append(
                            SourceInfo(
                                candidate,
                                is_package=True,
                                scope="project",
                                package_root=directory,
                            )
                        )
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
                sources.append(
                    SourceInfo(
                        expanded,
                        scope="project",
                        package_root=expanded.parent,
                    )
                )
            elif expanded.is_dir():
                sources.extend(_scan(expanded))
        except OSError:
            logger.debug("Could not inspect extension path %s", expanded, exc_info=True)
    return sources


def _plugin_sources(plugins: Sequence[PluginInstance]) -> list[SourceInfo]:
    """Return manifest-declared Python entries from enabled plugins."""
    sources: list[SourceInfo] = []
    for plugin in plugins:
        manifest = plugin.manifest
        if manifest is None:
            continue
        sources.extend(
            SourceInfo(
                path,
                is_package=path.name == "__init__.py",
                scope="plugin",
                plugin_id=plugin.plugin_id,
                plugin_version=plugin.version,
                package_root=plugin.root,
                data_dir=plugin.data_dir,
            )
            for path in manifest.python_extensions
        )
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
    sources = _plugin_sources(plugins)
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
