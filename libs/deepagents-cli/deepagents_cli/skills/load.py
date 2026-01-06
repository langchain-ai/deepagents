"""Skill loader for CLI commands.

This module provides filesystem-based skill loading for CLI operations (list, create, info).
It wraps the prebuilt middleware functionality from deepagents.middleware.skills and adapts
it for direct filesystem access needed by CLI commands.

For middleware usage within agents, use deepagents.middleware.skills.SkillsMiddleware directly.
"""

from __future__ import annotations

from pathlib import Path

from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.skills import (
    SkillMetadata,
)
from deepagents.middleware.skills import (
    list_skills as list_skills_from_backend,
)

# Re-export constants for CLI commands
__all__ = ["SkillMetadata", "list_skills"]


def list_skills(
    *, user_skills_dir: Path | None = None, project_skills_dir: Path | None = None
) -> list[SkillMetadata]:
    """List skills from user and/or project directories.

    This is a CLI-specific wrapper around the prebuilt middleware's skill loading
    functionality. It uses FilesystemBackend to load skills from local directories.

    When both directories are provided, project skills with the same name as
    user skills will override them (project skills take precedence).

    Args:
        user_skills_dir: Path to the user-level skills directory.
        project_skills_dir: Path to the project-level skills directory.

    Returns:
        Merged list of skill metadata from both sources, with project skills
        taking precedence over user skills when names conflict.
    """
    all_skills: dict[str, SkillMetadata] = {}

    # Load user skills first (foundation)
    if user_skills_dir and user_skills_dir.exists():
        user_backend = FilesystemBackend(root_dir=str(user_skills_dir))
        user_source = {"path": str(user_skills_dir), "name": "user"}
        user_skills = list_skills_from_backend(backend=user_backend, source=user_source)
        for skill in user_skills:
            skill_dict = dict(skill)
            virtual_path = skill_dict["path"]
            real_path = user_skills_dir / virtual_path.lstrip("/")
            skill_dict["path"] = str(real_path)
            all_skills[skill_dict["name"]] = skill_dict  # type: ignore

    # Load project skills second (override/augment)
    if project_skills_dir and project_skills_dir.exists():
        project_backend = FilesystemBackend(root_dir=str(project_skills_dir))
        project_source = {"path": str(project_skills_dir), "name": "project"}
        project_skills = list_skills_from_backend(backend=project_backend, source=project_source)
        for skill in project_skills:
            # Convert virtual backend path to real filesystem path
            skill_dict = dict(skill)  # Make a mutable copy
            virtual_path = skill_dict["path"]
            real_path = project_skills_dir / virtual_path.lstrip("/")
            skill_dict["path"] = str(real_path)
            # Project skills override user skills with the same name
            all_skills[skill_dict["name"]] = skill_dict  # type: ignore

    return list(all_skills.values())  # type: ignore
