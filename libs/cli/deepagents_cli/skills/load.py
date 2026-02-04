"""Skill loader for CLI commands.

This module provides filesystem-based skill loading for CLI operations
(list, create, info). It wraps the prebuilt middleware functionality from
deepagents.middleware.skills and adapts it for direct filesystem access
needed by CLI commands.

For middleware usage within agents, use
deepagents.middleware.skills.SkillsMiddleware directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from deepagents.backends.filesystem import FilesystemBackend

if TYPE_CHECKING:
    from pathlib import Path
from deepagents.middleware.skills import (
    SkillMetadata,
    _list_skills as list_skills_from_backend,
)


class ExtendedSkillMetadata(SkillMetadata):
    """Extended skill metadata for CLI display, adds source tracking."""

    source: str


# Re-export for CLI commands
__all__ = ["SkillMetadata", "list_skills"]


def list_skills(
    *,
    user_skills_dir: Path | None = None,
    project_skills_dir: Path | None = None,
    user_agent_skills_dir: Path | None = None,
    project_agent_skills_dir: Path | None = None,
) -> list[ExtendedSkillMetadata]:
    """List skills from user and/or project directories.

    This is a CLI-specific wrapper around the prebuilt middleware's skill loading
    functionality. It uses FilesystemBackend to load skills from local directories.

    Precedence order (lowest to highest):
    1. `user_skills_dir` (`~/.deepagents/{agent}/skills/`)
    2. `user_agent_skills_dir` (`~/.agents/skills/`)
    3. `project_skills_dir` (`.deepagents/skills/`)
    4. `project_agent_skills_dir` (`.agents/skills/`)

    Skills from higher-precedence directories override those with the same name.

    Args:
        user_skills_dir: Path to `~/.deepagents/{agent}/skills/`.
        project_skills_dir: Path to `.deepagents/skills/`.
        user_agent_skills_dir: Path to `~/.agents/skills/` (alias).
        project_agent_skills_dir: Path to `.agents/skills/` (alias).

    Returns:
        Merged list of skill metadata from all sources, with higher-precedence
            directories taking priority when names conflict.
    """
    all_skills: dict[str, ExtendedSkillMetadata] = {}

    # Load in precedence order (lowest to highest)
    # 1. User deepagents skills (~/.deepagents/{agent}/skills/) - lowest priority
    if user_skills_dir and user_skills_dir.exists():
        user_backend = FilesystemBackend(root_dir=str(user_skills_dir))
        user_skills = list_skills_from_backend(backend=user_backend, source_path=".")
        for skill in user_skills:
            extended_skill: ExtendedSkillMetadata = {**skill, "source": "user"}
            all_skills[skill["name"]] = extended_skill

    # 2. User agent skills (~/.agents/skills/) - overrides user deepagents
    if user_agent_skills_dir and user_agent_skills_dir.exists():
        user_agent_backend = FilesystemBackend(root_dir=str(user_agent_skills_dir))
        user_agent_skills = list_skills_from_backend(
            backend=user_agent_backend, source_path="."
        )
        for skill in user_agent_skills:
            extended_skill: ExtendedSkillMetadata = {**skill, "source": "user"}
            all_skills[skill["name"]] = extended_skill

    # 3. Project deepagents skills (.deepagents/skills/)
    if project_skills_dir and project_skills_dir.exists():
        project_backend = FilesystemBackend(root_dir=str(project_skills_dir))
        project_skills = list_skills_from_backend(
            backend=project_backend, source_path="."
        )
        for skill in project_skills:
            extended_skill: ExtendedSkillMetadata = {**skill, "source": "project"}
            all_skills[skill["name"]] = extended_skill

    # 4. Project agent skills (.agents/skills/) - highest priority
    if project_agent_skills_dir and project_agent_skills_dir.exists():
        project_agent_backend = FilesystemBackend(
            root_dir=str(project_agent_skills_dir)
        )
        project_agent_skills = list_skills_from_backend(
            backend=project_agent_backend, source_path="."
        )
        for skill in project_agent_skills:
            extended_skill: ExtendedSkillMetadata = {**skill, "source": "project"}
            all_skills[skill["name"]] = extended_skill

    return list(all_skills.values())
