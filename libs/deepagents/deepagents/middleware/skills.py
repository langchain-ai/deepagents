"""Middleware for loading and exposing agent skills to the system prompt.

This middleware implements Anthropic's "Agent Skills" pattern with progressive disclosure:
1. Parse YAML frontmatter from SKILL.md files at session start
2. Inject skills metadata (name + description) into system prompt
3. Agent reads full SKILL.md content when relevant to a task

Skills directory structure:
User-level: {skills_dir}/
Project-level: {project_skills_dir}/

Example structure:
skills/
├── web-research/
│   ├── SKILL.md        # Required: YAML frontmatter + instructions
│   └── helper.py       # Optional: supporting files
├── code-review/
│   ├── SKILL.md
│   └── checklist.md
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, NotRequired, TypedDict, cast

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langgraph.runtime import Runtime

# Maximum size for SKILL.md files (10MB)
MAX_SKILL_FILE_SIZE = 10 * 1024 * 1024


class SkillMetadata(TypedDict):
    """Metadata for a skill."""

    name: str
    """Name of the skill."""

    description: str
    """Description of what the skill does."""

    path: str
    """Path to the SKILL.md file."""

    source: str
    """Source of the skill ('user' or 'project')."""


def _is_safe_path(path: Path, base_dir: Path) -> bool:
    """Check if a path is safely contained within base_dir.

    This prevents directory traversal attacks via symlinks or path manipulation.
    The function resolves both paths to their canonical form (following symlinks)
    and verifies that the target path is within the base directory.

    Args:
        path: The path to validate
        base_dir: The base directory that should contain the path

    Returns:
        True if the path is safely within base_dir, False otherwise
    """
    try:
        resolved_path = path.resolve()
        resolved_base = base_dir.resolve()
        resolved_path.relative_to(resolved_base)
    except ValueError:
        return False
    except (OSError, RuntimeError):
        return False
    else:
        return True


def _parse_skill_metadata(skill_md_path: Path, source: str) -> SkillMetadata | None:
    """Parse YAML frontmatter from a SKILL.md file.

    Args:
        skill_md_path: Path to the SKILL.md file.
        source: Source of the skill ('user' or 'project').

    Returns:
        SkillMetadata with name, description, path, and source, or None if parsing fails.
    """
    try:
        file_size = skill_md_path.stat().st_size
        if file_size > MAX_SKILL_FILE_SIZE:
            return None

        content = skill_md_path.read_text(encoding="utf-8")

        frontmatter_pattern = r"^---\s*\n(.*?)\n---\s*\n"
        match = re.match(frontmatter_pattern, content, re.DOTALL)

        if not match:
            return None

        frontmatter = match.group(1)

        metadata: dict[str, str] = {}
        for line in frontmatter.split("\n"):
            kv_match = re.match(r"^(\w+):\s*(.+)$", line.strip())
            if kv_match:
                key, value = kv_match.groups()
                metadata[key] = value.strip()

        if "name" not in metadata or "description" not in metadata:
            return None

        return SkillMetadata(
            name=metadata["name"],
            description=metadata["description"],
            path=str(skill_md_path),
            source=source,
        )

    except (OSError, UnicodeDecodeError):
        return None


def _list_skills_from_dir(skills_dir: Path, source: str) -> list[SkillMetadata]:
    """List all skills from a single skills directory (internal helper).

    Scans the skills directory for subdirectories containing SKILL.md files,
    parses YAML frontmatter, and returns skill metadata.

    Args:
        skills_dir: Path to the skills directory.
        source: Source of the skills ('user' or 'project').

    Returns:
        List of skill metadata dictionaries with name, description, path, and source.
    """
    skills_dir = skills_dir.expanduser()
    if not skills_dir.exists():
        return []

    try:
        resolved_base = skills_dir.resolve()
    except (OSError, RuntimeError):
        return []

    skills: list[SkillMetadata] = []

    for skill_dir in skills_dir.iterdir():
        if not _is_safe_path(skill_dir, resolved_base):
            continue

        if not skill_dir.is_dir():
            continue

        skill_md_path = skill_dir / "SKILL.md"
        if not skill_md_path.exists():
            continue

        if not _is_safe_path(skill_md_path, resolved_base):
            continue

        metadata = _parse_skill_metadata(skill_md_path, source=source)
        if metadata:
            skills.append(metadata)

    return skills


def list_skills(*, user_skills_dir: Path | None = None, project_skills_dir: Path | None = None) -> list[SkillMetadata]:
    """List skills from user and/or project directories.

    When both directories are provided, project skills with the same name as
    user skills will override them.

    Args:
        user_skills_dir: Path to the user-level skills directory.
        project_skills_dir: Path to the project-level skills directory.

    Returns:
        Merged list of skill metadata from both sources, with project skills
        taking precedence over user skills when names conflict.
    """
    all_skills: dict[str, SkillMetadata] = {}

    if user_skills_dir:
        user_skills = _list_skills_from_dir(user_skills_dir, source="user")
        for skill in user_skills:
            all_skills[skill["name"]] = skill

    if project_skills_dir:
        project_skills = _list_skills_from_dir(project_skills_dir, source="project")
        for skill in project_skills:
            all_skills[skill["name"]] = skill

    return list(all_skills.values())


class SkillsState(AgentState):
    """State for the skills middleware."""

    skills_metadata: NotRequired[list[SkillMetadata]]
    """List of loaded skill metadata (name, description, path)."""


class SkillsStateUpdate(TypedDict):
    """State update for the skills middleware."""

    skills_metadata: list[SkillMetadata]
    """List of loaded skill metadata (name, description, path)."""


# Skills System Documentation
SKILLS_SYSTEM_PROMPT = """

## Skills System

You have access to a skills library that provides specialized capabilities and domain knowledge.

{skills_locations}

**Available Skills:**

{skills_list}

**How to Use Skills (Progressive Disclosure):**

Skills follow a **progressive disclosure** pattern - you know they exist (name + description above),
but you only read the full instructions when needed:

1. **Recognize when a skill applies**: Check if the user's task matches any skill's description
2. **Read the skill's full instructions**: The skill list above shows the exact path to use with read_file
3. **Follow the skill's instructions**: SKILL.md contains step-by-step workflows, best practices, and examples
4. **Access supporting files**: Skills may include Python scripts, configs, or reference docs - use absolute paths

**When to Use Skills:**
- When the user's request matches a skill's domain (e.g., "research X" → web-research skill)
- When you need specialized knowledge or structured workflows
- When a skill provides proven patterns for complex tasks

**Skills are Self-Documenting:**
- Each SKILL.md tells you exactly what the skill does and how to use it
- The skill list above shows the full path for each skill's SKILL.md file

**Executing Skill Scripts:**
Skills may contain Python scripts or other executable files. Always use absolute paths from the skill list.

**Example Workflow:**

User: "Can you research the latest developments in quantum computing?"

1. Check available skills above → See "web-research" skill with its full path
2. Read the skill using the path shown in the list
3. Follow the skill's research workflow (search → organize → synthesize)
4. Use any helper scripts with absolute paths

Remember: Skills are tools to make you more capable and consistent. When in doubt, check if a skill exists for the task!
"""


class SkillsMiddleware(AgentMiddleware):
    """Middleware for loading and exposing agent skills.

    This middleware implements Anthropic's agent skills pattern:
    - Loads skills metadata (name, description) from YAML frontmatter at session start
    - Injects skills list into system prompt for discoverability
    - Agent reads full SKILL.md content when a skill is relevant (progressive disclosure)

    Supports both user-level and project-level skills:
    - User skills: {skills_dir}/
    - Project skills: {project_skills_dir}/
    - Project skills override user skills with the same name

    Args:
        skills_dir: Path to the user-level skills directory.
        assistant_id: The agent identifier for path references in prompts.
        project_skills_dir: Optional path to project-level skills directory.

    Example:
        ```python
        from deepagents import create_deep_agent, SkillsMiddleware

        agent = create_deep_agent(
            model="anthropic:claude-sonnet-4-20250514",
            middleware=[
                SkillsMiddleware(
                    skills_dir="~/.myapp/skills",
                    assistant_id="my-agent",
                    project_skills_dir="./.skills",  # optional
                )
            ],
        )
        ```
    """

    state_schema = SkillsState

    def __init__(
        self,
        *,
        skills_dir: str | Path,
        assistant_id: str,
        project_skills_dir: str | Path | None = None,
    ) -> None:
        """Initialize the skills middleware.

        Args:
            skills_dir: Path to the user-level skills directory.
            assistant_id: The agent identifier.
            project_skills_dir: Optional path to the project-level skills directory.
        """
        self.skills_dir = Path(skills_dir).expanduser()
        self.assistant_id = assistant_id
        self.project_skills_dir = Path(project_skills_dir).expanduser() if project_skills_dir else None
        self.user_skills_display = str(self.skills_dir)
        self.system_prompt_template = SKILLS_SYSTEM_PROMPT

    def _format_skills_locations(self) -> str:
        """Format skills locations for display in system prompt."""
        locations = [f"**User Skills**: `{self.user_skills_display}`"]
        if self.project_skills_dir:
            locations.append(f"**Project Skills**: `{self.project_skills_dir}` (overrides user skills)")
        return "\n".join(locations)

    def _format_skills_list(self, skills: list[SkillMetadata]) -> str:
        """Format skills metadata for display in system prompt."""
        if not skills:
            locations = [f"{self.user_skills_display}/"]
            if self.project_skills_dir:
                locations.append(f"{self.project_skills_dir}/")
            return f"(No skills available yet. You can create skills in {' or '.join(locations)})"

        user_skills = [s for s in skills if s["source"] == "user"]
        project_skills = [s for s in skills if s["source"] == "project"]

        lines = []

        if user_skills:
            lines.append("**User Skills:**")
            for skill in user_skills:
                lines.append(f"- **{skill['name']}**: {skill['description']}")
                lines.append(f"  → Read `{skill['path']}` for full instructions")
            lines.append("")

        if project_skills:
            lines.append("**Project Skills:**")
            for skill in project_skills:
                lines.append(f"- **{skill['name']}**: {skill['description']}")
                lines.append(f"  → Read `{skill['path']}` for full instructions")

        return "\n".join(lines)

    def before_agent(self, state: SkillsState, runtime: Runtime) -> SkillsStateUpdate | None:  # noqa: ARG002
        """Load skills metadata before agent execution.

        This runs once at session start to discover available skills from both
        user-level and project-level directories.

        Args:
            state: Current agent state (unused, required by interface).
            runtime: Runtime context (unused, required by interface).

        Returns:
            Updated state with skills_metadata populated.
        """
        skills = list_skills(
            user_skills_dir=self.skills_dir,
            project_skills_dir=self.project_skills_dir,
        )
        return SkillsStateUpdate(skills_metadata=skills)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Inject skills documentation into the system prompt.

        This runs on every model call to ensure skills info is always available.

        Args:
            request: The model request being processed.
            handler: The handler function to call with the modified request.

        Returns:
            The model response from the handler.
        """
        skills_metadata = request.state.get("skills_metadata", [])

        skills_locations = self._format_skills_locations()
        skills_list = self._format_skills_list(skills_metadata)

        skills_section = self.system_prompt_template.format(
            skills_locations=skills_locations,
            skills_list=skills_list,
        )

        system_prompt = request.system_prompt + "\n\n" + skills_section if request.system_prompt else skills_section

        return handler(request.override(system_prompt=system_prompt))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """(async) Inject skills documentation into the system prompt.

        Args:
            request: The model request being processed.
            handler: The handler function to call with the modified request.

        Returns:
            The model response from the handler.
        """
        state = cast("SkillsState", request.state)
        skills_metadata = state.get("skills_metadata", [])

        skills_locations = self._format_skills_locations()
        skills_list = self._format_skills_list(skills_metadata)

        skills_section = self.system_prompt_template.format(
            skills_locations=skills_locations,
            skills_list=skills_list,
        )

        system_prompt = request.system_prompt + "\n\n" + skills_section if request.system_prompt else skills_section

        return await handler(request.override(system_prompt=system_prompt))


__all__ = [
    "MAX_SKILL_FILE_SIZE",
    "SKILLS_SYSTEM_PROMPT",
    "SkillMetadata",
    "SkillsMiddleware",
    "SkillsState",
    "SkillsStateUpdate",
    "list_skills",
]
