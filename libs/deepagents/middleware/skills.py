"""Middleware for providing skills to an agent."""

import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

import yaml
from langchain.agents.middleware.types import AgentMiddleware
from langchain.tools import ToolRuntime
from langchain_core.tools import tool
from typing_extensions import TypedDict


@dataclass
class Skill:
    """Specification for a skill.

    Skills are folders of instructions, scripts, and resources that Claude loads
    dynamically to improve performance on specialized tasks.
    """

    name: str
    """The name of the skill (unique identifier)."""

    description: str
    """A complete description of what the skill does and when to use it."""

    instructions: str
    """The markdown content below the frontmatter containing instructions, examples, and guidelines."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata from the YAML frontmatter."""

    source_path: str | None = None
    """Path to the skill's directory."""


class SkillsState(TypedDict):
    """State schema for skills middleware."""

    active_skills: Annotated[list[str], lambda left, right: list(set((left or []) + right))]
    """Names of currently active skills. Merged using set union to avoid duplicates."""


def _parse_skill_md(skill_path: Path) -> Skill:
    """Parse a SKILL.md file and return a Skill object.

    Args:
        skill_path: Path to the SKILL.md file.

    Returns:
        Skill object with parsed metadata and instructions.

    Raises:
        ValueError: If SKILL.md is missing required fields or is malformed.
    """
    if not skill_path.exists():
        raise ValueError(f"SKILL.md not found at {skill_path}")

    content = skill_path.read_text(encoding="utf-8")

    # Parse YAML frontmatter
    frontmatter_match = re.match(r"^---\n(.*?)\n---\n(.*)$", content, re.DOTALL)
    if not frontmatter_match:
        raise ValueError(f"SKILL.md at {skill_path} is missing YAML frontmatter")

    frontmatter_text = frontmatter_match.group(1)
    instructions = frontmatter_match.group(2).strip()

    try:
        frontmatter = yaml.safe_load(frontmatter_text)
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse YAML frontmatter in {skill_path}: {e}") from e

    # Validate required fields
    if "name" not in frontmatter:
        raise ValueError(f"SKILL.md at {skill_path} is missing required field 'name'")
    if "description" not in frontmatter:
        raise ValueError(f"SKILL.md at {skill_path} is missing required field 'description'")

    # Extract name and description
    name = frontmatter.pop("name")
    description = frontmatter.pop("description")

    return Skill(
        name=name,
        description=description,
        instructions=instructions,
        metadata=frontmatter,
        source_path=str(skill_path.parent),
    )


def _discover_skills(skills_dir: Path) -> dict[str, Skill]:
    """Discover all skills in a directory.

    Args:
        skills_dir: Path to the directory containing skill folders.

    Returns:
        Dictionary mapping skill names to Skill objects.

    Raises:
        ValueError: If multiple skills have the same name.
    """
    if not skills_dir.exists():
        return {}

    skills = {}
    for skill_folder in skills_dir.iterdir():
        if not skill_folder.is_dir():
            continue

        skill_md = skill_folder / "SKILL.md"
        if not skill_md.exists():
            continue

        try:
            skill = _parse_skill_md(skill_md)
            if skill.name in skills:
                raise ValueError(
                    f"Duplicate skill name '{skill.name}' found in {skill_folder} "
                    f"and {skills[skill.name].source_path}"
                )
            skills[skill.name] = skill
        except ValueError as e:
            # Log warning but continue discovering other skills
            print(f"Warning: Failed to load skill from {skill_folder}: {e}")

    return skills


DEFAULT_SKILLS_SYSTEM_PROMPT = """# Skills

You have access to a collection of skills that provide specialized capabilities for specific tasks.
Skills are loaded dynamically using the `use_skill` tool.

When a skill is active, you will receive additional instructions in your system prompt to guide you
in completing tasks using that skill's capabilities.

Use the `use_skill` tool to activate a skill when you need its specialized capabilities."""


class SkillsMiddleware(AgentMiddleware):
    """Middleware that provides skills to an agent.

    Skills are folders containing a SKILL.md file with YAML frontmatter and markdown instructions.
    This middleware provides a `use_skill` tool that activates skills dynamically.

    Example:
        ```python
        from deepagents import create_deep_agent
        from deepagents.middleware import SkillsMiddleware

        agent = create_deep_agent(
            model="claude-sonnet-4-5-20250929",
            middleware=[
                SkillsMiddleware(
                    skills_dir="./skills",
                    auto_activate=["template-skill"],
                )
            ]
        )
        ```

    Args:
        skills_dir: Path to directory containing skill folders. Each skill should be in its own
            subdirectory with a SKILL.md file. If not provided, defaults to "./skills".
        auto_activate: Optional list of skill names to activate automatically on initialization.
        system_prompt: Optional custom system prompt. If not provided, uses DEFAULT_SKILLS_SYSTEM_PROMPT.
        skills: Optional pre-loaded skills dictionary. If provided, skills_dir is ignored.
    """

    state_schema = SkillsState

    def __init__(
        self,
        skills_dir: str | Path | None = None,
        auto_activate: Sequence[str] | None = None,
        system_prompt: str | None = None,
        skills: dict[str, Skill] | None = None,
    ):
        """Initialize the skills middleware."""
        # Load skills
        if skills is not None:
            self._skills = skills
        else:
            skills_path = Path(skills_dir) if skills_dir else Path("./skills")
            self._skills = _discover_skills(skills_path)

        self._auto_activate = list(auto_activate) if auto_activate else []

        # Validate auto_activate skills exist
        for skill_name in self._auto_activate:
            if skill_name not in self._skills:
                raise ValueError(
                    f"Auto-activate skill '{skill_name}' not found in skills directory. "
                    f"Available skills: {list(self._skills.keys())}"
                )

        # Create use_skill tool
        available_skills = self._skills

        @tool
        def use_skill(
            runtime: ToolRuntime,
            name: str,
        ) -> str:
            """Activate a skill to gain access to its specialized capabilities.

            When you activate a skill, you will receive additional instructions and capabilities
            specific to that skill. Use this tool when you need specialized functionality for a task.

            Args:
                name: The name of the skill to activate. Must be one of the available skills.

            Returns:
                A confirmation message with the skill's description and instructions.

            Available skills:
{available_skills_list}
            """
            if name not in available_skills:
                return (
                    f"Error: Skill '{name}' not found.\n\n"
                    f"Available skills: {', '.join(available_skills.keys())}"
                )

            skill = available_skills[name]

            # Add skill to active skills in state
            state = runtime.state
            active_skills = state.get("active_skills", [])

            if name in active_skills:
                return f"Skill '{name}' is already active."

            # Update state to add this skill to active skills
            runtime.stream_writer({"active_skills": [name]})

            return (
                f"Activated skill: {name}\n\n"
                f"Description: {skill.description}\n\n"
                f"Instructions:\n{skill.instructions}"
            )

        # Format available skills list for tool description
        skills_list = "\n".join(
            f"- **{name}**: {skill.description}" for name, skill in self._skills.items()
        )
        use_skill.__doc__ = use_skill.__doc__.format(available_skills_list=skills_list)  # type: ignore

        self.tools = [use_skill]
        self.system_prompt = system_prompt or DEFAULT_SKILLS_SYSTEM_PROMPT

    def wrap_model_call(self, request, handler):
        """Inject active skill instructions into system prompt before model call."""
        # Get active skills from state
        active_skills = request.state.get("active_skills", [])

        # Add auto-activated skills if this is the first call
        messages = request.state.get("messages", [])
        if not messages:
            active_skills = list(set(active_skills + self._auto_activate))

        if not active_skills:
            return handler(request)

        # Build additional system prompt with active skill instructions
        skill_prompts = []
        for skill_name in active_skills:
            if skill_name in self._skills:
                skill = self._skills[skill_name]
                skill_prompts.append(
                    f"# Active Skill: {skill.name}\n\n"
                    f"{skill.description}\n\n"
                    f"{skill.instructions}"
                )

        if skill_prompts:
            additional_prompt = "\n\n---\n\n".join(skill_prompts)
            # Add to existing system prompt
            if request.system_prompt:
                request.system_prompt = f"{request.system_prompt}\n\n{additional_prompt}"
            else:
                request.system_prompt = additional_prompt

        return handler(request)
