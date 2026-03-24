"""Skill loader for CLI commands.

This module provides filesystem-based skill discovery for CLI operations
(list, create, info, delete). It wraps the prebuilt middleware functionality from
deepagents.middleware.skills and adapts it for direct filesystem access
needed by CLI commands.

For middleware usage within agents, use
deepagents.middleware.skills.SkillsMiddleware directly.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal, cast

from deepagents.backends.filesystem import FilesystemBackend

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path
from deepagents.middleware.skills import (
    SkillMetadata,
    _list_skills as list_skills_from_backend,  # noqa: PLC2701  # Intentional access to internal skill listing
)

from deepagents_cli._version import __version__ as _cli_version

logger = logging.getLogger(__name__)


class ExtendedSkillMetadata(SkillMetadata):
    """Extended skill metadata for CLI display, adds source tracking.

    Attributes:
        source: Origin of the skill. One of `'built-in'`, `'user'`, `'project'`,
            or `'claude (experimental)'`.
    """

    source: Literal["built-in", "user", "project", "claude (experimental)"]


# Re-export for CLI commands
__all__ = ["SkillMetadata", "list_skills", "load_skill_content"]


def list_skills(
    *,
    built_in_skills_dir: Path | None = None,
    user_skills_dir: Path | None = None,
    project_skills_dir: Path | None = None,
    user_agent_skills_dir: Path | None = None,
    project_agent_skills_dir: Path | None = None,
    user_claude_skills_dir: Path | None = None,
    project_claude_skills_dir: Path | None = None,
) -> list[ExtendedSkillMetadata]:
    """List skills from built-in, user, and/or project directories.

    This is a CLI-specific wrapper around the prebuilt middleware's skill loading
    functionality. It uses FilesystemBackend to load skills from local directories.

    Precedence order (lowest to highest):
    0. `built_in_skills_dir` (`<package>/built_in_skills/`)
    1. `user_skills_dir` (`~/.deepagents/{agent}/skills/`)
    2. `user_agent_skills_dir` (`~/.agents/skills/`)
    3. `project_skills_dir` (`.deepagents/skills/`)
    4. `project_agent_skills_dir` (`.agents/skills/`)
    5. `user_claude_skills_dir` (`~/.claude/skills/`, experimental)
    6. `project_claude_skills_dir` (`.claude/skills/`, experimental)

    Skills from higher-precedence directories override those with the same name.

    Args:
        built_in_skills_dir: Path to built-in skills shipped with the package.
        user_skills_dir: Path to `~/.deepagents/{agent}/skills/`.
        project_skills_dir: Path to `.deepagents/skills/`.
        user_agent_skills_dir: Path to `~/.agents/skills/` (alias).
        project_agent_skills_dir: Path to `.agents/skills/` (alias).
        user_claude_skills_dir: Path to `~/.claude/skills/` (experimental).
        project_claude_skills_dir: Path to `.claude/skills/` (experimental).

    Returns:
        Merged list of skill metadata from all sources, with higher-precedence
            directories taking priority when names conflict.
    """
    all_skills: dict[str, ExtendedSkillMetadata] = {}

    sources: list[tuple[Path | None, str, bool]] = [
        (built_in_skills_dir, "built-in", False),
        (user_skills_dir, "user", False),
        (user_agent_skills_dir, "user", False),
        (project_skills_dir, "project", False),
        (project_agent_skills_dir, "project", False),
        (user_claude_skills_dir, "claude (experimental)", True),
        (project_claude_skills_dir, "claude (experimental)", True),
    ]
    """Sources in precedence order (lowest to highest).

    Each tuple: `(directory, source label, is_experimental)`.

    Each source is individually try/except-guarded so a single inaccessible
    directory doesn't block the rest.
    """

    for skill_dir, source_label, experimental in sources:
        if not skill_dir or not skill_dir.exists():
            continue
        try:
            backend = FilesystemBackend(root_dir=str(skill_dir))
            skills = list_skills_from_backend(backend=backend, source_path=".")
            if experimental and skills:
                logger.info(
                    "Discovered %d skill(s) from experimental Claude path: %s",
                    len(skills),
                    skill_dir,
                )
            for skill in skills:
                extra: dict[str, object] = {"source": source_label}
                if source_label == "built-in":
                    extra["metadata"] = {
                        **skill["metadata"],
                        "deepagents-cli-version": _cli_version,
                    }
                extended = cast("ExtendedSkillMetadata", {**skill, **extra})
                all_skills[skill["name"]] = extended
        except (OSError, KeyError, TypeError):
            logger.warning(
                "Could not load skills from %s",
                skill_dir,
                exc_info=True,
            )

    return list(all_skills.values())


def load_skill_content(
    skill_path: str,
    *,
    allowed_roots: Sequence[Path] = (),
) -> str | None:
    """Read the full raw SKILL.md content for a skill.

    Returns the complete file content including any YAML frontmatter.
    Callers are responsible for parsing or stripping frontmatter if needed.

    When `allowed_roots` is provided, the resolved path must fall within at
    least one root directory. This prevents symlink traversal from reading files
    outside known skill directories.

    Args:
        skill_path: Path to the SKILL.md file (from `SkillMetadata['path']`).
        allowed_roots: Skill root directories the resolved path must be
            contained within.

            Callers must pre-resolve these via `Path.resolve()` — the resolved
            skill path is compared directly, so un-resolved roots cause false
            containment failures.

            If empty, containment is not checked.

    Returns:
        Full text content of the SKILL.md file, or `None` on read failure.

    Raises:
        PermissionError: If the resolved path is outside all `allowed_roots`.
    """
    from pathlib import Path

    path = Path(skill_path).resolve()

    if allowed_roots and not any(path.is_relative_to(root) for root in allowed_roots):
        logger.warning(
            "Skill path %s is outside all allowed roots, refusing to read",
            skill_path,
        )
        msg = (
            f"Skill path {skill_path} resolves outside all allowed skill "
            "directories. If this is a symlink, add the target directory to "
            "DEEPAGENTS_EXTRA_SKILLS_DIRS or [skills].extra_allowed_dirs "
            "in ~/.deepagents/config.toml."
        )
        raise PermissionError(msg)

    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        logger.warning(
            "Could not read skill content from %s", skill_path, exc_info=True
        )
        return None
