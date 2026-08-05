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
from typing import TYPE_CHECKING, Literal, TypedDict, cast

from deepagents.backends.filesystem import FilesystemBackend

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path
from deepagents.middleware.skills import (
    SkillMetadata,
    _list_skills_with_failures as list_skills_with_failures_from_backend,  # noqa: PLC2701  # Intentional access to internal skill listing
)

from deepagents_code._version import __version__ as _cli_version
from deepagents_code.skills.merge import merge_skill

logger = logging.getLogger(__name__)


class ExtendedSkillMetadata(SkillMetadata):
    """Extended skill metadata for CLI display, adds source tracking.

    Attributes:
        source: Origin of the skill. One of `'built-in'`, `'user'`, `'project'`,
            or `'claude (experimental)'`.
    """

    source: Literal["built-in", "plugin", "user", "project", "claude (experimental)"]


class SkillLoadFailure(TypedDict):
    """A skill that failed to load, for CLI display.

    Attributes:
        path: Path to the `SKILL.md` file (or skill source root) that failed.
        error: Human-readable reason the skill could not be loaded.
        source: Origin label for the skill's directory. One of `'built-in'`,
            `'plugin'`, `'user'`, `'project'`, or `'claude (experimental)'`.
    """

    path: str
    error: str
    source: str


# Re-export for CLI commands
__all__ = [
    "SkillLoadFailure",
    "SkillMetadata",
    "list_skills",
    "list_skills_with_failures",
    "load_skill_content",
]


def list_skills_with_failures(
    *,
    built_in_skills_dir: Path | None = None,
    plugin_skill_sources: Sequence[tuple[Path, str]] = (),
    user_skills_dir: Path | None = None,
    project_skills_dir: Path | None = None,
    user_agent_skills_dir: Path | None = None,
    project_agent_skills_dir: Path | None = None,
    user_claude_skills_dir: Path | None = None,
    project_claude_skills_dir: Path | None = None,
) -> tuple[list[ExtendedSkillMetadata], list[SkillLoadFailure]]:
    """List skills from built-in, user, and/or project directories.

    This is a dcode-specific wrapper around the prebuilt middleware's skill loading
    functionality. It uses `FilesystemBackend` to load skills from local directories.

    Precedence order (lowest to highest):
    0. `built_in_skills_dir` (`<package>/built_in_skills/`)
    1. `plugin_skill_sources`
    2. `user_skills_dir` (`~/.deepagents/{agent}/skills/`)
    3. `user_agent_skills_dir` (`~/.agents/skills/`)
    4. `project_skills_dir` (`.deepagents/skills/`)
    5. `project_agent_skills_dir` (`.agents/skills/`)
    6. `user_claude_skills_dir` (`~/.claude/skills/`, experimental)
    7. `project_claude_skills_dir` (`.claude/skills/`, experimental)

    Skills from higher-precedence directories override those with the same name.

    Args:
        built_in_skills_dir: Path to built-in skills shipped with the package.
        plugin_skill_sources: Plugin skill source directories with namespaces.
        user_skills_dir: Path to `~/.deepagents/{agent}/skills/`.
        project_skills_dir: Path to `.deepagents/skills/`.
        user_agent_skills_dir: Path to `~/.agents/skills/` (alias).
        project_agent_skills_dir: Path to `.agents/skills/` (alias).
        user_claude_skills_dir: Path to `~/.claude/skills/` (experimental).
        project_claude_skills_dir: Path to `.claude/skills/` (experimental).

    Returns:
        Tuple of `(merged skills, load failures)`. The skills list is merged
            with higher-precedence directories taking priority when names
            conflict. Failures carry the `SKILL.md` path and reason for each
            skill that could not be loaded, so callers can surface them.
    """
    all_skills: dict[str, ExtendedSkillMetadata] = {}
    merged_source_labels: dict[str, str | None] = {}
    failures: list[SkillLoadFailure] = []

    sources: list[tuple[Path | None, str, bool, str]] = [
        (built_in_skills_dir, "built-in", False, ""),
        *[
            (path, "plugin", False, namespace)
            for path, namespace in plugin_skill_sources
        ],
        (user_skills_dir, "user", False, ""),
        (user_agent_skills_dir, "user", False, ""),
        (project_skills_dir, "project", False, ""),
        (project_agent_skills_dir, "project", False, ""),
        (user_claude_skills_dir, "claude (experimental)", True, ""),
        (project_claude_skills_dir, "claude (experimental)", True, ""),
    ]
    """Sources in precedence order (lowest to highest).

    Each tuple: `(directory, source label, is_experimental, namespace)`.

    Each source is individually try/except-guarded so a single inaccessible
    directory doesn't block the rest.
    """

    for skill_dir, source_label, experimental, namespace in sources:
        if not skill_dir or not skill_dir.exists():
            continue
        try:
            backend = FilesystemBackend(root_dir=str(skill_dir), virtual_mode=False)
            if namespace:
                # Plugin sources are walked recursively so nested skill
                # directories are namespaced as `plugin:sub:skill`, matching
                # both the runtime middleware and plugin conventions.
                from deepagents_code.plugins.adapters.skills_middleware import (
                    load_namespaced_skills,
                )

                skills = load_namespaced_skills(
                    backend, str(skill_dir.resolve()), namespace
                )
                # Plugin namespaces load through their own walker, which only
                # returns successfully parsed skills; per-file failures are
                # logged by the SDK parser but not collected here.
            else:
                skills, source_failures, _source_error = (
                    list_skills_with_failures_from_backend(
                        backend=backend, source_path="."
                    )
                )
                failures.extend(
                    SkillLoadFailure(path=path, error=error, source=source_label)
                    for path, error in source_failures
                )
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
                        "deepagents-code-version": _cli_version,
                    }
                extended = cast("ExtendedSkillMetadata", {**skill, **extra})
                merge_skill(
                    all_skills,
                    merged_source_labels,
                    extended,
                    source_label=source_label,
                )
        except Exception:
            # Degrade gracefully — one malformed/inaccessible source must not
            # block discovery of others, so catch broadly and log instead.
            # WARNING (not ERROR) because a half-written SKILL.md from a user is
            # an expected condition, not a code defect.
            logger.warning(
                "Could not load skills from %s",
                skill_dir,
                exc_info=True,
            )

    return list(all_skills.values()), failures


def list_skills(
    *,
    built_in_skills_dir: Path | None = None,
    plugin_skill_sources: Sequence[tuple[Path, str]] = (),
    user_skills_dir: Path | None = None,
    project_skills_dir: Path | None = None,
    user_agent_skills_dir: Path | None = None,
    project_agent_skills_dir: Path | None = None,
    user_claude_skills_dir: Path | None = None,
    project_claude_skills_dir: Path | None = None,
) -> list[ExtendedSkillMetadata]:
    """List skills from built-in, user, and/or project directories.

    See `list_skills_with_failures` for the full documentation; this wrapper
    returns only the merged skill metadata and discards load failures.

    Returns:
        Merged list of skill metadata from all sources, with higher-precedence
            directories taking priority when names conflict.
    """
    skills, _failures = list_skills_with_failures(
        built_in_skills_dir=built_in_skills_dir,
        plugin_skill_sources=plugin_skill_sources,
        user_skills_dir=user_skills_dir,
        project_skills_dir=project_skills_dir,
        user_agent_skills_dir=user_agent_skills_dir,
        project_agent_skills_dir=project_agent_skills_dir,
        user_claude_skills_dir=user_claude_skills_dir,
        project_claude_skills_dir=project_claude_skills_dir,
    )
    return skills


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
        from deepagents_code._env_vars import EXTRA_SKILLS_DIRS

        msg = (
            f"Skill path {skill_path} resolves outside all allowed skill "
            "directories. If this is a symlink, add the target directory to "
            f"{EXTRA_SKILLS_DIRS} or [skills].extra_allowed_dirs "
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
