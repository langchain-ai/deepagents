"""Helpers for loading and formatting skill invocations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from deepagents_code.skills.load import ExtendedSkillMetadata, SkillLoadFailure


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


@dataclass
class _SkillLoadFailureSnapshot:
    """Module-level snapshot of the most recent discovery's load failures.

    `discover_skills_and_roots` keeps its 2-tuple return shape for existing
    callers, so per-skill failures are exposed through this snapshot instead.
    Discovery runs serially behind the import lock, so a plain replacement on
    each run is sufficient.
    """

    failures: list[SkillLoadFailure] = field(default_factory=list)


_failure_snapshot = _SkillLoadFailureSnapshot()


def get_skill_load_failures() -> list[SkillLoadFailure]:
    """Return the failures recorded by the most recent `discover_skills_and_roots` call.

    Returns:
        A copy of the `(path, error, source)` failure records from the last
            discovery run (empty if none or discovery has not run yet).
    """
    return list(_failure_snapshot.failures)


def discover_skills_and_roots(
    assistant_id: str,
    *,
    plugin_skill_sources: tuple[tuple[Path, str], ...] = (),
    plugin_skill_roots: tuple[Path, ...] = (),
) -> tuple[list[ExtendedSkillMetadata], list[Path]]:
    """Discover skills and build pre-resolved containment roots.

    Args:
        assistant_id: Agent identifier used to resolve user skill directories.
        plugin_skill_sources: Plugin-owned skill directories and namespaces,
            supplied by the plugin composition layer.
        plugin_skill_roots: Plugin-owned roots allowed for content loading.

    Returns:
        Tuple of `(skill metadata list, pre-resolved containment roots)`.
            Per-skill load failures are recorded on the module snapshot and
            retrievable via `get_skill_load_failures()`.
    """
    from deepagents_code.config import settings
    from deepagents_code.skills.load import list_skills_with_failures
    from deepagents_code.skills.trust import load_trusted_skill_dirs

    skills, failures = list_skills_with_failures(
        built_in_skills_dir=settings.get_built_in_skills_dir(),
        plugin_skill_sources=plugin_skill_sources,
        user_skills_dir=settings.get_user_skills_dir(assistant_id),
        project_skills_dir=settings.get_project_skills_dir(),
        user_agent_skills_dir=settings.get_user_agent_skills_dir(),
        project_agent_skills_dir=settings.get_project_agent_skills_dir(),
        user_claude_skills_dir=settings.get_user_claude_skills_dir(),
        project_claude_skills_dir=settings.get_project_claude_skills_dir(),
    )
    _failure_snapshot.failures = failures
    roots = [
        path.resolve()
        for path in (
            settings.get_built_in_skills_dir(),
            *plugin_skill_roots,
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
