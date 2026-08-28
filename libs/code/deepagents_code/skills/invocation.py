"""Helpers for loading and formatting skill invocations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from deepagents_code._paths import (
    get_built_in_skills_dir,
    get_project_agent_skills_dir,
    get_project_claude_skills_dir,
    get_project_skills_dir,
    get_user_agent_skills_dir,
    get_user_claude_skills_dir,
    get_user_skills_dir,
)

if TYPE_CHECKING:
    from pathlib import Path

    from deepagents_code.skills.load import ExtendedSkillMetadata


@dataclass(frozen=True)
class SkillInvocationEnvelope:
    """Structured prompt and checkpoint metadata for a skill invocation.

    Attributes:
        prompt: Composed prompt that wraps `SKILL.md` content with
            invocation instructions.
        message_kwargs: Extra fields merged into the initial HumanMessage.
        skill_name: Invoked skill name for trace attribution.
    """

    prompt: str
    message_kwargs: dict[str, Any]
    skill_name: str


def discover_skills_and_roots(
    assistant_id: str,
    *,
    plugin_skill_sources: tuple[tuple[Path, str], ...] = (),
    plugin_skill_roots: tuple[Path, ...] = (),
    path_base: Path | None = None,
) -> tuple[list[ExtendedSkillMetadata], list[Path]]:
    """Discover skills and build pre-resolved containment roots.

    Args:
        assistant_id: Agent identifier used to resolve user skill directories.
        plugin_skill_sources: Plugin-owned skill directories and namespaces,
            supplied by the plugin composition layer.
        plugin_skill_roots: Plugin-owned roots allowed for content loading.
        path_base: User working directory for resolving relative skill roots.
            Defaults to the process working directory.

    Returns:
        Tuple of `(skill metadata list, pre-resolved containment roots)`.

    Raises:
        RuntimeError: If the extra skill-directory option is absent from the
            manifest.
    """
    from pathlib import Path

    from deepagents_code.config import _use_extra_skills_path_base, credentials
    from deepagents_code.config_manifest import _emit_ranked_diagnostics, get_option
    from deepagents_code.configuration.resolver import get_config_resolver
    from deepagents_code.skills.load import list_skills
    from deepagents_code.skills.trust import load_trusted_skill_dirs

    skills = list_skills(
        built_in_skills_dir=get_built_in_skills_dir(),
        plugin_skill_sources=plugin_skill_sources,
        user_skills_dir=get_user_skills_dir(assistant_id),
        project_skills_dir=get_project_skills_dir(credentials.project_root),
        user_agent_skills_dir=get_user_agent_skills_dir(),
        project_agent_skills_dir=get_project_agent_skills_dir(credentials.project_root),
        user_claude_skills_dir=get_user_claude_skills_dir(),
        project_claude_skills_dir=get_project_claude_skills_dir(
            credentials.project_root
        ),
    )
    roots = [
        path.resolve()
        for path in (
            get_built_in_skills_dir(),
            *plugin_skill_roots,
            get_user_skills_dir(assistant_id),
            get_project_skills_dir(credentials.project_root),
            get_user_agent_skills_dir(),
            get_project_agent_skills_dir(credentials.project_root),
            get_user_claude_skills_dir(),
            get_project_claude_skills_dir(credentials.project_root),
        )
        if path is not None
    ]
    option = get_option("skills.extra_allowed_dirs")
    if option is None:
        msg = "skills.extra_allowed_dirs is missing from the configuration manifest"
        raise RuntimeError(msg)
    with _use_extra_skills_path_base(path_base or Path.cwd()):
        resolved = get_config_resolver().get(option)
    _emit_ranked_diagnostics(option, resolved)
    extra_skills_dirs = cast("list[Path] | None", resolved.value)
    roots.extend(path.resolve() for path in extra_skills_dirs or ())
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
    return SkillInvocationEnvelope(
        prompt=prompt,
        message_kwargs=message_kwargs,
        skill_name=skill["name"],
    )
