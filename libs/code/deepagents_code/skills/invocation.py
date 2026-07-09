"""Helpers for loading and formatting skill invocations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from deepagents_code.skills.load import ExtendedSkillMetadata


@dataclass(frozen=True)
class SkillInvocationEnvelope:
    """Structured prompt and checkpoint metadata for a skill invocation.

    Attributes:
        prompt: Composed prompt that wraps `SKILL.md` content with
            invocation instructions.
        message_kwargs: Extra fields merged into the initial HumanMessage.
    """

    prompt: str
    message_kwargs: dict[str, Any]


def discover_skills_and_roots(
    assistant_id: str,
) -> tuple[list[ExtendedSkillMetadata], list[Path]]:
    """Discover skills and build pre-resolved containment roots.

    Args:
        assistant_id: Agent identifier used to resolve user skill directories.

    Returns:
        Tuple of `(skill metadata list, pre-resolved containment roots)`.
    """
    from deepagents_code.config import settings
    from deepagents_code.skills.load import list_skills
    from deepagents_code.skills.trust import load_trusted_skill_dirs

    plugin_sources: list[tuple[Path, str]] = []
    plugin_roots: list[Path] = []
    try:
        from deepagents_code._env_vars import experimental_enabled

        if experimental_enabled():
            from deepagents_code.plugins import discover_plugins
            from deepagents_code.plugins.adapters.skills import (
                plugin_skill_roots,
                plugin_skill_sources,
            )

            plugin_result = discover_plugins()
            plugin_sources = [
                (Path(path), prefix)
                for path, _label, prefix in plugin_skill_sources(plugin_result.plugins)
            ]
            plugin_roots = plugin_skill_roots(plugin_result.plugins)
    except (OSError, RuntimeError, TypeError, ValueError):
        plugin_sources = []
        plugin_roots = []
    skills = list_skills(
        built_in_skills_dir=settings.get_built_in_skills_dir(),
        plugin_skill_sources=plugin_sources,
        user_skills_dir=settings.get_user_skills_dir(assistant_id),
        project_skills_dir=settings.get_project_skills_dir(),
        user_agent_skills_dir=settings.get_user_agent_skills_dir(),
        project_agent_skills_dir=settings.get_project_agent_skills_dir(),
        user_claude_skills_dir=settings.get_user_claude_skills_dir(),
        project_claude_skills_dir=settings.get_project_claude_skills_dir(),
    )
    roots = [
        path.resolve()
        for path in (
            settings.get_built_in_skills_dir(),
            *plugin_roots,
            settings.get_user_skills_dir(assistant_id),
            settings.get_project_skills_dir(),
            settings.get_user_agent_skills_dir(),
            settings.get_project_agent_skills_dir(),
            settings.get_user_claude_skills_dir(),
            settings.get_project_claude_skills_dir(),
        )
        if path is not None
    ]
    roots.extend(path.resolve() for path in settings.get_extra_skills_dirs())
    # Persisted in-the-moment approvals extend the containment allowlist just
    # like the declarative `extra_allowed_dirs`, but are managed by the trust
    # store rather than hand-edited config. These entries are already the
    # canonical approved directories and are verified against post-approval
    # symlink swaps by `load_trusted_skill_dirs`, so they are added as-is
    # rather than re-resolved (re-resolving would follow an injected symlink to
    # a directory the user never approved).
    roots.extend(load_trusted_skill_dirs())
    return skills, roots


def build_skill_invocation_envelope(
    skill: ExtendedSkillMetadata,
    content: str,
    args: str = "",
) -> SkillInvocationEnvelope:
    """Build the wrapped prompt and persisted metadata for a skill.

    Args:
        skill: Loaded skill metadata.
        content: Raw `SKILL.md` content.
        args: Optional user request appended after the skill body.

    Returns:
        A `SkillInvocationEnvelope` with the composed prompt and
            `message_kwargs` containing persisted skill metadata.
    """
    prompt = (
        f"I'm invoking the skill `{skill['name']}`. "
        "Below are the full instructions from the skill's SKILL.md file. "
        "Follow these instructions to complete the task.\n\n"
        f"---\n{content}\n---"
    )
    if args:
        prompt += f"\n\n**User request:** {args}"

    message_kwargs = {
        "additional_kwargs": {
            "__skill": {
                "name": skill["name"],
                "description": str(skill.get("description", "")),
                "source": str(skill.get("source", "")),
                "args": args,
            },
        },
    }
    return SkillInvocationEnvelope(prompt=prompt, message_kwargs=message_kwargs)
