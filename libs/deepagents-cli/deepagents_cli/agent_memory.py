"""Middleware for loading agent-specific long-term memory into the system prompt."""

import contextlib
from collections.abc import Awaitable, Callable
from typing import NotRequired, TypedDict, cast

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from langgraph.runtime import Runtime

from deepagents_cli.config import Settings


class AgentMemoryState(AgentState):
    """State for the agent memory middleware."""

    user_memory: NotRequired[str]
    """Personal preferences from ~/.deepagents/{agent}/ (applies everywhere)."""

    project_memory: NotRequired[str]
    """Project-specific context (loaded from project root)."""


class AgentMemoryStateUpdate(TypedDict):
    """A state update for the agent memory middleware."""

    user_memory: NotRequired[str]
    """Personal preferences from ~/.deepagents/{agent}/ (applies everywhere)."""

    project_memory: NotRequired[str]
    """Project-specific context (loaded from project root)."""


# Long-term Memory Documentation
# Note: Claude Code loads CLAUDE.md files hierarchically and combines them (not precedence-based):
# - Loads recursively from cwd up to (but not including) root directory
# - Multiple files are combined hierarchically: enterprise → project → user
# - Both [project-root]/CLAUDE.md and [project-root]/.claude/CLAUDE.md are loaded if both exist
# - Files higher in hierarchy load first, providing foundation for more specific memories
# We will follow that pattern for deepagents-cli
LONGTERM_MEMORY_SYSTEM_PROMPT = """

## Long-term Memory

Your long-term memory is stored in files on the filesystem and persists across sessions.

**User Memory Location**: `{agent_dir_absolute}` (displays as `{agent_dir_display}`)
**Project Memory Location**: {project_memory_info}

Your system prompt is loaded from TWO sources at startup:
1. **User agent.md**: `{agent_dir_absolute}/agent.md` - Your personal preferences across all projects
2. **Project agent.md**: Loaded from project root if available - Project-specific instructions

Project-specific agent.md is loaded from these locations (both combined if both exist):
- `[project-root]/.deepagents/agent.md` (preferred)
- `[project-root]/agent.md` (fallback, but also included if both exist)

**When to CHECK/READ memories (CRITICAL - do this FIRST):**
- **At the start of ANY new session**: Check both user and project memories
  - User: `ls {agent_dir_absolute}`
  - Project: `ls {project_deepagents_dir}` (if in a project)
- **BEFORE answering questions**: If asked "what do you know about X?" or "how do I do Y?", check project memories FIRST, then user
- **When user asks you to do something**: Check if you have project-specific guides or examples
- **When user references past work**: Search project memory files for related context

**Memory-first response pattern:**
1. User asks a question → Check project directory first: `ls {project_deepagents_dir}`
2. If relevant files exist → Read them with `read_file '{project_deepagents_dir}/[filename]'`
3. Check user memory if needed → `ls {agent_dir_absolute}`
4. Base your answer on saved knowledge supplemented by general knowledge

**When to update memories:**
- **IMMEDIATELY when the user describes your role or how you should behave**
- **IMMEDIATELY when the user gives feedback on your work** - Update memories to capture what was wrong and how to do it better
- When the user explicitly asks you to remember something
- When patterns or preferences emerge (coding styles, conventions, workflows)
- After significant work where context would help in future sessions

**Learning from feedback:**
- When user says something is better/worse, capture WHY and encode it as a pattern
- Each correction is a chance to improve permanently - don't just fix the immediate issue, update your instructions
- When user says "you should remember X" or "be careful about Y", treat this as HIGH PRIORITY - update memories IMMEDIATELY
- Look for the underlying principle behind corrections, not just the specific mistake

## Deciding Where to Store Memory

When writing or updating agent memory, decide whether each fact, configuration, or behavior belongs in:

### User Agent File: `{agent_dir_absolute}/agent.md`
→ Describes the agent's **personality, style, and universal behavior** across all projects.

**Store here:**
- Your general tone and communication style
- Universal coding preferences (formatting, comment style, etc.)
- General workflows and methodologies you follow
- Tool usage patterns that apply everywhere
- Personal preferences that don't change per-project

**Examples:**
- "Be concise and direct in responses"
- "Always use type hints in Python"
- "Prefer functional programming patterns"

### Project Agent File: `{project_deepagents_dir}/agent.md`
→ Describes **how this specific project works** and **how the agent should behave here only.**

**Store here:**
- Project-specific architecture and design patterns
- Coding conventions specific to this codebase
- Project structure and organization
- Testing strategies for this project
- Deployment processes and workflows
- Team conventions and guidelines

**Examples:**
- "This project uses FastAPI with SQLAlchemy"
- "Tests go in tests/ directory mirroring src/ structure"
- "All API changes require updating OpenAPI spec"

### Project Memory Files: `{project_deepagents_dir}/*.md`
→ Use for **project-specific reference information** and structured notes.

**Store here:**
- API design documentation
- Architecture decisions and rationale
- Deployment procedures
- Common debugging patterns
- Onboarding information

**Examples:**
- `{project_deepagents_dir}/api-design.md` - REST API patterns used
- `{project_deepagents_dir}/architecture.md` - System architecture overview
- `{project_deepagents_dir}/deployment.md` - How to deploy this project

### File Operations:

**User memory:**
```
ls {agent_dir_absolute}                              # List user memory files
read_file '{agent_dir_absolute}/agent.md'            # Read user preferences
edit_file '{agent_dir_absolute}/agent.md' ...        # Update user preferences
```

**Project memory (preferred for project-specific information):**
```
ls {project_deepagents_dir}                          # List project memory files
read_file '{project_deepagents_dir}/agent.md'        # Read project instructions
edit_file '{project_deepagents_dir}/agent.md' ...    # Update project instructions
write_file '{project_deepagents_dir}/agent.md' ...  # Create project memory file
```

**Important**:
- Project memory files are stored in `.deepagents/` inside the project root
- Always use absolute paths for file operations
- Check project memories BEFORE user when answering project-specific questions"""


DEFAULT_MEMORY_SNIPPET = """<user_memory>
{user_memory}
</user_memory>

<project_memory>
{project_memory}
</project_memory>"""


class AgentMemoryMiddleware(AgentMiddleware):
    """Middleware for loading agent-specific long-term memory.

    This middleware loads the agent's long-term memory from a file (agent.md)
    and injects it into the system prompt. The memory is loaded once at the
    start of the conversation and stored in state.
    """

    state_schema = AgentMemoryState

    def __init__(
        self,
        *,
        settings: Settings,
        assistant_id: str,
        system_prompt_template: str | None = None,
    ) -> None:
        """Initialize the agent memory middleware.

        Args:
            settings: Global settings instance with project detection and paths.
            assistant_id: The agent identifier.
            system_prompt_template: Optional custom template for injecting
                agent memory into system prompt.
        """
        self.settings = settings
        self.assistant_id = assistant_id

        # User paths
        self.agent_dir = settings.get_agent_dir(assistant_id)
        # Store both display path (with ~) and absolute path for file operations
        self.agent_dir_display = f"~/.deepagents/{assistant_id}"
        self.agent_dir_absolute = str(self.agent_dir)

        # Project paths (from settings)
        self.project_root = settings.project_root

        self.system_prompt_template = system_prompt_template or DEFAULT_MEMORY_SNIPPET

    def before_agent(
        self,
        state: AgentMemoryState,
        runtime: Runtime,
    ) -> AgentMemoryStateUpdate:
        """Load agent memory from file before agent execution.

        Loads both user agent.md and project-specific agent.md if available.
        Only loads if not already present in state.

        Dynamically checks for file existence on every call to catch user updates.

        Args:
            state: Current agent state.
            runtime: Runtime context.

        Returns:
            Updated state with user_memory and project_memory populated.
        """
        result: AgentMemoryStateUpdate = {}

        # Load user memory if not already in state
        if "user_memory" not in state:
            user_path = self.settings.get_user_agent_md_path(self.assistant_id)
            if user_path.exists():
                with contextlib.suppress(OSError, UnicodeDecodeError):
                    result["user_memory"] = user_path.read_text()

        # Load project memory if not already in state
        if "project_memory" not in state:
            project_path = self.settings.get_project_agent_md_path()
            if project_path and project_path.exists():
                with contextlib.suppress(OSError, UnicodeDecodeError):
                    result["project_memory"] = project_path.read_text()

        return result

    def _build_system_prompt(self, request: ModelRequest) -> str:
        """Build the complete system prompt with memory sections.

        Args:
            request: The model request containing state and base system prompt.

        Returns:
            Complete system prompt with memory sections injected.
        """
        # Extract memory from state
        state = cast("AgentMemoryState", request.state)
        user_memory = state.get("user_memory")
        project_memory = state.get("project_memory")
        base_system_prompt = request.system_prompt

        # Build project memory info for documentation
        if self.project_root and project_memory:
            project_memory_info = f"`{self.project_root}` (detected)"
        elif self.project_root:
            project_memory_info = f"`{self.project_root}` (no agent.md found)"
        else:
            project_memory_info = "None (not in a git project)"

        # Build project deepagents directory path
        if self.project_root:
            project_deepagents_dir = str(self.project_root / ".deepagents")
        else:
            project_deepagents_dir = "[project-root]/.deepagents (not in a project)"

        # Format memory section with both memories
        memory_section = self.system_prompt_template.format(
            user_memory=user_memory if user_memory else "(No user agent.md)",
            project_memory=project_memory if project_memory else "(No project agent.md)",
        )

        system_prompt = memory_section

        if base_system_prompt:
            system_prompt += "\n\n" + base_system_prompt

        system_prompt += "\n\n" + LONGTERM_MEMORY_SYSTEM_PROMPT.format(
            agent_dir_absolute=self.agent_dir_absolute,
            agent_dir_display=self.agent_dir_display,
            project_memory_info=project_memory_info,
            project_deepagents_dir=project_deepagents_dir,
        )

        return system_prompt

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Inject agent memory into the system prompt.

        Args:
            request: The model request being processed.
            handler: The handler function to call with the modified request.

        Returns:
            The model response from the handler.
        """
        system_prompt = self._build_system_prompt(request)
        return handler(request.override(system_prompt=system_prompt))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """(async) Inject agent memory into the system prompt.

        Args:
            request: The model request being processed.
            handler: The handler function to call with the modified request.

        Returns:
            The model response from the handler.
        """
        system_prompt = self._build_system_prompt(request)
        return await handler(request.override(system_prompt=system_prompt))
