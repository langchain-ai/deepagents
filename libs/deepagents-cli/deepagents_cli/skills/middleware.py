"""Middleware for loading and exposing agent skills to the system prompt.

This middleware implements Anthropic's "Agent Skills" pattern with progressive disclosure:
1. Parse YAML frontmatter from SKILL.md files at session start
2. Inject skills metadata (name + description) into system prompt
3. Agent reads full SKILL.md content when relevant to a task

Skills directory structure (per-agent + project):
User-level: ~/.deepagents/{AGENT_NAME}/skills/
Project-level: {PROJECT_ROOT}/.deepagents/skills/

Example structure:
~/.deepagents/{AGENT_NAME}/skills/
├── web-research/
│   ├── SKILL.md        # Required: YAML frontmatter + instructions
│   └── helper.py       # Optional: supporting files
├── code-review/
│   ├── SKILL.md
│   └── checklist.md

.deepagents/skills/
├── project-specific/
│   └── SKILL.md        # Project-specific skills
"""

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import NotRequired, TypedDict, cast

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from langgraph.runtime import Runtime

from deepagents_cli.skills.load import SkillMetadata, list_skills


class SkillsState(AgentState):
    """State for the skills middleware."""

    skills_metadata: NotRequired[list[SkillMetadata]]
    """List of loaded skill metadata (name, description, path)."""


class SkillsStateUpdate(TypedDict):
    """State update for the skills middleware."""

    skills_metadata: list[SkillMetadata]
    """List of loaded skill metadata (name, description, path)."""


# Skills System Documentation (per Agent Skills spec: https://agentskills.io/specification)
SKILLS_SYSTEM_PROMPT = """

## Skills System

You have access to a skills library that provides specialized capabilities and domain knowledge.

{skills_list}

**How to Use Skills (Progressive Disclosure):**

Skills follow a progressive disclosure pattern - you see their name and description above,
but read full instructions only when needed:

1. **Recognize when a skill applies**: Check if the user's task matches any skill description
2. **Read full instructions**: Use `read_file` with the `<location>` path from the skill list
3. **Follow the instructions**: SKILL.md contains workflows, best practices, and examples
4. **Access supporting files**: Skills may include scripts, configs, or docs in the same dir

**When to Use Skills:**
- When the user's request matches a skill's domain (e.g., "research X" → web-research skill)
- When you need specialized knowledge or structured workflows
- When a skill provides proven patterns for complex tasks

**Example Workflow:**

User: "Can you research the latest developments in quantum computing?"

1. Check `<available_skills>` above → Find skill with matching description
2. Read the skill's SKILL.md using the `<location>` path
3. Follow the skill's workflow instructions
4. Access any helper scripts in the skill directory

Remember: Skills make you more capable. When in doubt, check if a skill exists for the task!
"""


class SkillsMiddleware(AgentMiddleware):
    """Middleware for loading and exposing agent skills.

    This middleware implements Anthropic's agent skills pattern:
    - Loads skills metadata (name, description) from YAML frontmatter at session start
    - Injects skills list into system prompt for discoverability
    - Agent reads full SKILL.md content when a skill is relevant (progressive disclosure)

    Supports both user-level and project-level skills:
    - User skills: ~/.deepagents/{AGENT_NAME}/skills/
    - Project skills: {PROJECT_ROOT}/.deepagents/skills/
    - Project skills override user skills with the same name

    Args:
        skills_dir: Path to the user-level skills directory (per-agent).
        assistant_id: The agent identifier for path references in prompts.
        project_skills_dir: Optional path to project-level skills directory.
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
        self.project_skills_dir = (
            Path(project_skills_dir).expanduser() if project_skills_dir else None
        )
        # Store display paths for prompts
        self.user_skills_display = f"~/.deepagents/{assistant_id}/skills"
        self.system_prompt_template = SKILLS_SYSTEM_PROMPT

    def _escape_xml(self, text: str) -> str:
        """Escape special XML characters in text.

        Args:
            text: The text to escape.

        Returns:
            Text with &, <, > escaped for safe XML inclusion.
        """
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def _format_skills_list(self, skills: list[SkillMetadata]) -> str:
        """Format skills metadata as XML per Agent Skills spec.

        Generates XML format recommended for Claude models:
        https://agentskills.io/docs/integrate-skills

        Args:
            skills: List of skill metadata to format.

        Returns:
            XML-formatted string with available skills, or a message if no skills.
        """
        if not skills:
            locations = [f"{self.user_skills_display}/"]
            if self.project_skills_dir:
                locations.append(f"{self.project_skills_dir}/")
            return f"(No skills available yet. You can create skills in {' or '.join(locations)})"

        lines = ["<available_skills>"]
        for skill in skills:
            lines.append("  <skill>")
            lines.append(f"    <name>{self._escape_xml(skill['name'])}</name>")
            lines.append(f"    <description>{self._escape_xml(skill['description'])}</description>")
            lines.append(f"    <location>{skill['path']}</location>")
            lines.append("  </skill>")
        lines.append("</available_skills>")

        return "\n".join(lines)

    def before_agent(self, state: SkillsState, runtime: Runtime) -> SkillsStateUpdate | None:
        """Load skills metadata before agent execution.

        This runs once at session start to discover available skills from both
        user-level and project-level directories.

        Args:
            state: Current agent state.
            runtime: Runtime context.

        Returns:
            Updated state with skills_metadata populated.
        """
        # We re-load skills on every new interaction with the agent to capture
        # any changes in the skills directories.
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
        # Get skills metadata from state
        skills_metadata = request.state.get("skills_metadata", [])

        # Format skills list as XML
        skills_list = self._format_skills_list(skills_metadata)

        # Format the skills documentation
        skills_section = self.system_prompt_template.format(
            skills_list=skills_list,
        )

        if request.system_prompt:
            system_prompt = request.system_prompt + "\n\n" + skills_section
        else:
            system_prompt = skills_section

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
        # The state is guaranteed to be SkillsState due to state_schema
        state = cast("SkillsState", request.state)
        skills_metadata = state.get("skills_metadata", [])

        # Format skills list as XML
        skills_list = self._format_skills_list(skills_metadata)

        # Format the skills documentation
        skills_section = self.system_prompt_template.format(
            skills_list=skills_list,
        )

        # Inject into system prompt
        if request.system_prompt:
            system_prompt = request.system_prompt + "\n\n" + skills_section
        else:
            system_prompt = skills_section

        return await handler(request.override(system_prompt=system_prompt))
