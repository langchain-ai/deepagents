"""Adapter from discovered plugins to `SkillsMiddleware` sources."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from deepagents_code.plugins.models import PluginInstance

logger = logging.getLogger(__name__)

SkillSourceTuple = tuple[str, str, str]


def plugin_skill_sources(plugins: tuple[PluginInstance, ...]) -> list[SkillSourceTuple]:
    """Return skill source tuples for plugin skills.

    Args:
        plugins: Discovered plugin instances.

    Returns:
        Source tuples containing path, label, and skill-name prefix.
    """
    sources: list[SkillSourceTuple] = []
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
                    f"{plugin.plugin_id}:",
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
