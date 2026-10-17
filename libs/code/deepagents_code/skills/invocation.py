"""Helpers for loading and formatting skill invocations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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
    from deepagents.middleware.skills import SkillMetadata

    from deepagents_code.skills.load import ExtendedSkillMetadata


@dataclass(frozen=True)
class SkillInvocationEnvelope:
    """Structured prompt and checkpoint metadata for a skill invocation.

    Attributes:
        prompt: User request identifying the skill for the SDK to pin.
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
    """
    from deepagents_code.config import credentials
    from deepagents_code.skills.load import list_skills

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
    roots.extend(extra_skill_roots(path_base or Path.cwd()))
    return skills, roots


def extra_skill_roots(path_base: Path | None) -> list[Path]:
    """Return configured extra and user-approved skill roots.

    Args:
        path_base: User working directory for resolving relative skill roots.

    Returns:
        Pre-resolved `skills.extra_allowed_dirs` entries, then persisted
            trust-store approvals.

    Raises:
        RuntimeError: If the extra skill-directory option is absent from the
            manifest.
    """
    from deepagents_code.config import _use_extra_skills_path_base
    from deepagents_code.config_manifest import _emit_ranked_diagnostics, get_option
    from deepagents_code.configuration.resolver import get_config_resolver
    from deepagents_code.skills.trust import load_trusted_skill_dirs

    option = get_option("skills.extra_allowed_dirs")
    if option is None:
        msg = "skills.extra_allowed_dirs is missing from the configuration manifest"
        raise RuntimeError(msg)
    with _use_extra_skills_path_base(path_base):
        resolved = get_config_resolver().get(option)
    _emit_ranked_diagnostics(option, resolved)
    extra_skills_dirs = cast("list[Path] | None", resolved.value)
    roots = [path.resolve() for path in extra_skills_dirs or ()]
    # Persisted in-the-moment approvals extend the containment allowlist just
    # like the declarative `extra_allowed_dirs`, but are managed by the trust
    # store rather than hand-edited config. These entries are already the
    # canonical approved directories and are verified against post-approval
    # symlink swaps by `load_trusted_skill_dirs`, so they are added as-is
    # rather than re-resolved (re-resolving would follow an injected symlink to
    # a directory the user never approved).
    roots.extend(load_trusted_skill_dirs())
    return roots


def pinned_skill_body(skill: SkillMetadata, content: str | None) -> str:
    """Return the instructions the SDK pins for `skill`.

    The SDK skips a skill it cannot pin. The agent server raises instead, so a
    skill the user invoked never runs silently without its instructions.

    Args:
        skill: Metadata of the skill to pin.
        content: Raw `SKILL.md` text, or `None` when it could not be read.

    Returns:
        The `SKILL.md` body without its frontmatter.

    Raises:
        ValueError: The file is unreadable or too large, or has no
            instructions after its frontmatter.
    """
    from deepagents.backends.protocol import FileDownloadResponse
    from deepagents.middleware import skills as sdk_skills

    name = skill["name"]
    response = FileDownloadResponse(
        path=skill["path"], content=content.encode("utf-8") if content else None
    )
    body = sdk_skills._pinned_skill_body(skill, response)
    if body is None:
        msg = (
            f"Could not read instructions for skill '{name}'. Check that "
            "its SKILL.md exists, is UTF-8, and is not too large."
        )
        raise ValueError(msg)
    if not body:
        msg = f"Skill '{name}' has no instructions after its frontmatter."
        raise ValueError(msg)
    return body


def build_skill_invocation_envelope(
    name: str,
    args: str = "",
) -> SkillInvocationEnvelope:
    """Build the user request and persisted metadata for SDK skill pinning.

    The agent server resolves, trust-checks, and reads the skill, so only the
    name the user typed is sent.

    Args:
        name: Normalized skill name the user invoked.
        args: Optional user request accompanying the skill invocation.

    Returns:
        A `SkillInvocationEnvelope` with the composed prompt and
            `message_kwargs` containing persisted skill metadata.
    """
    prompt = f"I'm invoking the skill `{name}`."
    if args:
        prompt += f"\n\n**User request:** {args}"

    message_kwargs = {
        "additional_kwargs": {
            # `pinned` marks an invocation whose instructions arrive in a
            # separate SDK message, unlike older ones that embedded them.
            "__skill": {"name": name, "args": args, "pinned": True},
        },
    }
    return SkillInvocationEnvelope(
        prompt=prompt,
        message_kwargs=message_kwargs,
        skill_name=name,
    )


def pinned_skill_parts(message: object) -> tuple[str, str, str] | None:
    """Return the skill a pinned-skill message carries.

    Args:
        message: Any message from the stream or a checkpoint.

    Returns:
        `(name, description, body)` with the SDK's `<skill>` wrapper removed,
            or `None` when `message` does not pin a skill.
    """
    from deepagents_code.goal_state_notice import is_pinned_skill_message

    if not is_pinned_skill_message(message):
        return None
    marker = getattr(message, "additional_kwargs", {}).get("skill")
    if not isinstance(marker, dict) or not marker.get("name"):
        return None
    body = str(getattr(message, "content", ""))
    if body.startswith("<skill ") and body.endswith("\n</skill>"):
        body = body.partition("\n")[2].removesuffix("\n</skill>")
    return str(marker["name"]), str(marker.get("description", "")), body
