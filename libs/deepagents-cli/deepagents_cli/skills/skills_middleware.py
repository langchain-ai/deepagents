"""Middleware for loading and exposing agent skills to the system prompt.

This middleware implements Anthropic's "Agent Skills" pattern with progressive disclosure:
1. Parse YAML frontmatter from SKILL.md files at session start
2. Inject skills metadata (name + description) into system prompt
3. Agent reads full SKILL.md content when relevant to a task

Skills directory structure:
~/.deepagents/skills/
├── web-research/
│   ├── SKILL.md        # Required: YAML frontmatter + instructions
│   └── helper.py       # Optional: supporting files
├── code-review/
│   ├── SKILL.md
│   └── checklist.md
"""

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import NotRequired

from deepagents.backends.protocol import BackendProtocol
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)

from .skill_loader import SkillLoader, SkillMetadata


class SkillsState(AgentState):
    """State for the skills middleware."""

    skills_metadata: NotRequired[list[SkillMetadata] | None]
    """List of loaded skill metadata (name, description, path)."""


# Skills System Documentation
SKILLS_SYSTEM_PROMPT = """

## Skills System

You have access to a skills library that provides specialized capabilities and domain knowledge.

**Available Skills:**

{skills_list}

**How to Use Skills (Progressive Disclosure):**

Skills follow a **progressive disclosure** pattern - you know they exist (name + description above), but you only read the full instructions when needed:

1. **Recognize when a skill applies**: Check if the user's task matches any skill's description
2. **Read the skill's full instructions**: Use `read_file '{skills_path}[skill-name]/SKILL.md'` to see detailed guidance
3. **Follow the skill's instructions**: SKILL.md contains step-by-step workflows, best practices, and examples
4. **Access supporting files**: Skills may include Python scripts, configs, or reference docs - paths are in SKILL.md

**When to Use Skills:**
- When the user's request matches a skill's domain (e.g., "research X" → web-research skill)
- When you need specialized knowledge or structured workflows
- When a skill provides proven patterns for complex tasks

**Skills are Self-Documenting:**
- Each SKILL.md tells you exactly what the skill does and how to use it
- You can explore available skills with `ls {skills_path}`
- You can read any skill's directory with `ls {skills_path}[skill-name]/`

**Example Workflow:**

User: "Can you research the latest developments in quantum computing?"

1. Check available skills above → See "web-research" skill
2. Read the skill: `read_file '{skills_path}web-research/SKILL.md'`
3. Follow the skill's research workflow (search → organize → synthesize)
4. Use any helper scripts referenced in SKILL.md

Remember: Skills are tools to make you more capable and consistent. When in doubt, check if a skill exists for the task!
"""


class SkillsMiddleware(AgentMiddleware):
    """Middleware for loading and exposing agent skills.

    This middleware implements Anthropic's agent skills pattern:
    - Loads skills metadata (name, description) from YAML frontmatter at session start
    - Injects skills list into system prompt for discoverability
    - Agent reads full SKILL.md content when a skill is relevant (progressive disclosure)

    Args:
        skills_dir: Path to the skills directory. Defaults to ~/.deepagents/skills
        skills_path: Virtual path prefix for skills in the filesystem (e.g., "/skills/")
        system_prompt_template: Optional custom template for skills documentation.
            Use {skills_list} for the formatted skills list and {skills_path} for the path.

    Example:
        ```python
        from pathlib import Path
        from deepagents.backends.filesystem import FilesystemBackend
        from deepagents.backends.composite import CompositeBackend
        from deepagents_cli.skills import SkillsMiddleware

        # Set up skills backend
        skills_dir = Path.home() / ".deepagents" / "skills"
        skills_backend = FilesystemBackend(root_dir=skills_dir, virtual_mode=True)

        # Create composite backend with skills routing
        backend = CompositeBackend(default=FilesystemBackend(), routes={"/skills/": skills_backend})

        # Create middleware
        middleware = SkillsMiddleware(skills_dir=skills_dir, skills_path="/skills/")
        ```
    """

    state_schema = SkillsState

    def __init__(
        self,
        *,
        skills_dir: str | Path = "~/.deepagents/skills",
        skills_path: str = "/skills/",
        system_prompt_template: str | None = None,
    ) -> None:
        """Initialize the skills middleware.

        Args:
            skills_dir: Path to the skills directory.
            skills_path: Virtual path prefix for skills in the filesystem.
            system_prompt_template: Optional custom template for skills docs.
        """
        self.skills_dir = Path(skills_dir).expanduser()
        self.skills_path = skills_path
        self.system_prompt_template = system_prompt_template or SKILLS_SYSTEM_PROMPT
        self.loader = SkillLoader(skills_dir=self.skills_dir)

    def _format_skills_list(self, skills: list[SkillMetadata]) -> str:
        """Format skills metadata for display in system prompt.

        Args:
            skills: List of skill metadata.

        Returns:
            Formatted string with skills list.
        """
        if not skills:
            return "(No skills available yet. You can create skills in ~/.deepagents/skills/)"

        lines = []
        for skill in skills:
            skill_dir = Path(skill["path"]).parent.name
            lines.append(f"- **{skill['name']}**: {skill['description']}")
            lines.append(f"  → Read `{self.skills_path}{skill_dir}/SKILL.md` for full instructions")

        return "\n".join(lines)

    def before_agent(
        self,
        state: SkillsState,
        runtime,
    ) -> SkillsState:
        """Load skills metadata before agent execution.

        This runs once at session start to discover available skills.

        Args:
            state: Current agent state.
            runtime: Runtime context.

        Returns:
            Updated state with skills_metadata populated.
        """
        # Only load skills if not already loaded
        if "skills_metadata" not in state or state.get("skills_metadata") is None:
            try:
                # Load skills from directory
                skills = self.loader.list()
                return {"skills_metadata": skills}
            except Exception:
                # Silently handle errors, return empty list
                return {"skills_metadata": []}

    async def abefore_agent(
        self,
        state: SkillsState,
        runtime,
    ) -> SkillsState:
        """(async) Load skills metadata before agent execution.

        Args:
            state: Current agent state.
            runtime: Runtime context.

        Returns:
            Updated state with skills_metadata populated.
        """
        # Sync version is fine since file operations are fast
        return self.before_agent(state, runtime)

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

        # Format skills list
        skills_list = self._format_skills_list(skills_metadata)

        # Format the skills documentation
        skills_section = self.system_prompt_template.format(
            skills_list=skills_list, skills_path=self.skills_path
        )

        # Inject into system prompt
        if request.system_prompt:
            request.system_prompt = request.system_prompt + "\n\n" + skills_section
        else:
            request.system_prompt = skills_section

        return handler(request)

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
        # Get skills metadata from state
        skills_metadata = request.state.get("skills_metadata", [])

        # Format skills list
        skills_list = self._format_skills_list(skills_metadata)

        # Format the skills documentation
        skills_section = self.system_prompt_template.format(
            skills_list=skills_list, skills_path=self.skills_path
        )

        # Inject into system prompt
        if request.system_prompt:
            request.system_prompt = request.system_prompt + "\n\n" + skills_section
        else:
            request.system_prompt = skills_section

        return await handler(request)
