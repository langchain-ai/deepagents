"""Adapter from discovered plugins to `SkillsMiddleware` sources."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias

from deepagents_code._env_vars import EXPERIMENTAL, is_env_truthy

if TYPE_CHECKING:
    from deepagents_code.plugins.models import PluginInstance

logger = logging.getLogger(__name__)

SkillPath: TypeAlias = str
SkillLabel: TypeAlias = str
SkillNamespace: TypeAlias = str
DirectorySkillSource: TypeAlias = tuple[SkillPath, SkillLabel]
PluginSkillSource: TypeAlias = tuple[SkillPath, SkillLabel, SkillNamespace]
CodeSkillSource: TypeAlias = DirectorySkillSource | PluginSkillSource


def namespaced_skill_name(
    namespace: SkillNamespace,
    name: str,
    subfolders: tuple[str, ...] = (),
) -> str:
    """Qualify a skill name under its plugin namespace.

    Nested skill directories contribute intermediate `:`-joined segments
    between the plugin namespace and the skill name, matching the plugin skill
    naming convention (e.g. `plugin:sub:review`).

    Args:
        namespace: Plugin namespace (its `plugin_id`).
        name: Skill name from the skill's frontmatter.
        subfolders: Directory names between the plugin skills root and the
            skill directory, in path order.

    Returns:
        The qualified skill name.
    """
    return ":".join((namespace, *subfolders, name))


def plugin_skill_sources(
    plugins: tuple[PluginInstance, ...],
) -> list[PluginSkillSource]:
    """Return skill source tuples for plugin skills.

    Args:
        plugins: Plugin instances.

    Returns:
        Source tuples containing path, label, and plugin namespace.
    """
    sources: list[PluginSkillSource] = []
    for plugin in plugins:
        for path in plugin.inventory.skills:
            source_path = path.parent if path.name == "SKILL.md" else path
            try:
                if not source_path.exists():
                    continue
            except OSError:
                logger.warning("Could not inspect plugin skill path %s", source_path)
                continue
            sources.append(
                (
                    str(source_path),
                    f"Plugin: {plugin.plugin_id}",
                    plugin.plugin_id,
                )
            )
    return sources


def plugin_skill_roots(plugins: tuple[PluginInstance, ...]) -> list[Path]:
    """Return plugin skill roots for skill-content containment checks.

    Args:
        plugins: Discovered plugin instances.

    Returns:
        Skill root directories.
    """
    roots: list[Path] = []
    for plugin in plugins:
        roots.extend(
            path.parent if path.name == "SKILL.md" else path
            for path in plugin.inventory.skills
        )
    return roots


def discover_plugin_skill_sources_and_roots() -> tuple[
    tuple[tuple[Path, str], ...], tuple[Path, ...]
]:
    """Discover plugin skill sources and containment roots.

    Returns:
        Plugin skill sources and roots, or empty tuples when plugins are disabled
        or discovery fails.
    """
    plugin_sources: tuple[tuple[Path, str], ...] = ()
    plugin_roots: tuple[Path, ...] = ()
    try:
        if is_env_truthy(EXPERIMENTAL):
            from deepagents_code.plugins import discover_plugins

            plugins = discover_plugins().plugins
            plugin_sources = tuple(
                (Path(path), namespace)
                for path, _label, namespace in plugin_skill_sources(plugins)
            )
            plugin_roots = tuple(plugin_skill_roots(plugins))
    except (OSError, RuntimeError, TypeError, ValueError):
        logger.warning("Could not discover plugin skills", exc_info=True)
        return (), ()

    return plugin_sources, plugin_roots
