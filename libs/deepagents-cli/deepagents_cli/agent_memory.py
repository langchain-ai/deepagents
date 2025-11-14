"""Middleware for loading agent-specific long-term memory into the system prompt."""

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import NotRequired, TypedDict

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from langgraph.runtime import Runtime

from deepagents_cli.project_utils import find_project_agent_md, find_project_root


class AgentMemoryState(AgentState):
    """State for the agent memory middleware."""

    agent_memory: NotRequired[str]
    """Long-term memory content for the agent (global)."""

    project_memory: NotRequired[str]
    """Project-specific memory content (from project root)."""


class AgentMemoryStateUpdate(TypedDict):
    """A state update for the agent memory middleware."""

    agent_memory: NotRequired[str]
    """Long-term memory content for the agent (global)."""

    project_memory: NotRequired[str]
    """Project-specific memory content (from project root)."""


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

**Global Memory Location**: `{agent_dir_absolute}` (displays as `{agent_dir_display}`)
**Project Memory Location**: {project_memory_info}

Your system prompt is loaded from TWO sources at startup:
1. **Global agent.md**: `{agent_dir_absolute}/agent.md` - Your general instructions across all projects
2. **Project agent.md**: Loaded from project root if available - Project-specific instructions

Project-specific agent.md is loaded from these locations (both combined if both exist):
- `[project-root]/.deepagents/agent.md` (preferred)
- `[project-root]/agent.md` (fallback, but also included if both exist)

**When to CHECK/READ memories (CRITICAL - do this FIRST):**
- **At the start of ANY new session**: Check both global and project memories
  - Global: `ls {agent_dir_absolute}`
  - Project: `ls {project_deepagents_dir}` (if in a project)
- **BEFORE answering questions**: If asked "what do you know about X?" or "how do I do Y?", check project memories FIRST, then global
- **When user asks you to do something**: Check if you have project-specific guides or examples
- **When user references past work**: Search project memory files for related context

**Memory-first response pattern:**
1. User asks a question → Check project directory first: `ls {project_deepagents_dir}`
2. If relevant files exist → Read them with `read_file '{project_deepagents_dir}/[filename]'`
3. Check global if needed → `ls {agent_dir_absolute}`
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

### Global Agent File: `{agent_dir_absolute}/agent.md`
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

**Global memory:**
```
ls {agent_dir_absolute}                              # List global memory files
read_file '{agent_dir_absolute}/agent.md'            # Read global instructions
edit_file '{agent_dir_absolute}/agent.md' ...        # Update global instructions
```

**Project memory (preferred for project-specific information):**
```
ls {project_deepagents_dir}                          # List project memory files
read_file '{project_deepagents_dir}/agent.md'        # Read project instructions
edit_file '{project_deepagents_dir}/agent.md' ...    # Update project instructions
write_file '{project_deepagents_dir}/api-design.md' ...  # Create project memory file
```

**Important**:
- Project memory files are stored in `.deepagents/` inside the project root
- Always use absolute paths for file operations
- Check project memories BEFORE global when answering project-specific questions"""


DEFAULT_MEMORY_SNIPPET = """<global_agent_memory>
{agent_memory}
</global_agent_memory>

<project_agent_memory>
{project_memory}
</project_agent_memory>"""


class AgentMemoryMiddleware(AgentMiddleware):
    """Middleware for loading agent-specific long-term memory.

    This middleware loads the agent's long-term memory from a file (agent.md)
    and injects it into the system prompt. The memory is loaded once at the
    start of the conversation and stored in state.

    Args:
        agent_dir: Path to the agent directory containing agent.md.
        assistant_id: The agent identifier for path references in prompts.
        system_prompt_template: Optional custom template for how to inject
            the agent memory into the system prompt. Use {agent_memory} as
            a placeholder. Defaults to a simple section header.

    Example:
        ```python
        from deepagents_cli.agent_memory import AgentMemoryMiddleware
        from pathlib import Path

        # Set up with agent directory path
        agent_dir = Path.home() / ".deepagents" / "my-agent"

        # Create middleware
        middleware = AgentMemoryMiddleware(agent_dir=agent_dir, assistant_id="my-agent")
        ```
    """

    state_schema = AgentMemoryState

    def __init__(
        self,
        *,
        agent_dir: Path,
        assistant_id: str,
        system_prompt_template: str | None = None,
    ) -> None:
        """Initialize the agent memory middleware.

        Args:
            agent_dir: Path to the agent directory.
            assistant_id: The agent identifier.
            system_prompt_template: Optional custom template for injecting
                agent memory into system prompt.
        """
        self.agent_dir = Path(agent_dir).expanduser()
        self.assistant_id = assistant_id
        # Store both display path (with ~) and absolute path for file operations
        self.agent_dir_display = f"~/.deepagents/{assistant_id}"
        self.agent_dir_absolute = str(self.agent_dir)
        self.system_prompt_template = system_prompt_template or DEFAULT_MEMORY_SNIPPET

    def before_agent(
        self,
        state: AgentMemoryState,
        runtime: Runtime,
    ) -> AgentMemoryStateUpdate:
        """Load agent memory from file before agent execution.

        Loads both global agent.md and project-specific agent.md if available.

        Args:
            state: Current agent state.
            runtime: Runtime context.

        Returns:
            Updated state with agent_memory and project_memory populated.
        """
        result = {}

        # Load global agent memory if not already loaded
        if "agent_memory" not in state or state.get("agent_memory") is None:
            agent_md_path = self.agent_dir / "agent.md"
            try:
                if agent_md_path.exists():
                    result["agent_memory"] = agent_md_path.read_text()
                else:
                    result["agent_memory"] = ""
            except Exception:
                result["agent_memory"] = ""

        # Detect project root and load project memory if not already loaded
        if "project_memory" not in state or state.get("project_memory") is None:
            project_root = find_project_root()
            result["project_root"] = project_root

            if project_root:
                project_md_paths = find_project_agent_md(project_root)
                if project_md_paths:
                    try:
                        # Combine all project agent.md files (if multiple exist)
                        contents = []
                        for path in project_md_paths:
                            contents.append(path.read_text())
                        result["project_memory"] = "\n\n".join(contents)
                    except Exception:
                        result["project_memory"] = ""
                else:
                    result["project_memory"] = ""
            else:
                result["project_memory"] = ""

        return result

    async def abefore_agent(
        self,
        state: AgentMemoryState,
        runtime: Runtime,
    ) -> AgentMemoryStateUpdate:
        """(async) Load agent memory from file before agent execution.

        Loads both global agent.md and project-specific agent.md if available.

        Args:
            state: Current agent state.
            runtime: Runtime context.

        Returns:
            Updated state with agent_memory and project_memory populated.
        """
        # Sync version is fine since file operations are fast
        return self.before_agent(state, runtime)

    def _build_system_prompt(
        self,
        agent_memory: str,
        project_memory: str,
        project_root: Path | None,
        base_system_prompt: str | None,
    ) -> str:
        """Build the complete system prompt with memory sections.

        Args:
            agent_memory: Global agent memory content.
            project_memory: Project-specific memory content.
            project_root: Detected project root path.
            base_system_prompt: Base system prompt to append to.

        Returns:
            Complete system prompt with memory sections injected.
        """
        # Build project memory info for documentation
        if project_root and project_memory:
            project_memory_info = f"`{project_root}` (detected)"
        elif project_root:
            project_memory_info = f"`{project_root}` (no agent.md found)"
        else:
            project_memory_info = "None (not in a git project)"

        # Build project deepagents directory path
        if project_root:
            project_deepagents_dir = f"{project_root}/.deepagents"
        else:
            project_deepagents_dir = "[project-root]/.deepagents (not in a project)"

        # Format memory section with both memories
        memory_section = self.system_prompt_template.format(
            agent_memory=agent_memory or "(No global agent.md)",
            project_memory=project_memory or "(No project agent.md)",
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
        system_prompt = self._build_system_prompt(
            request.state.get("agent_memory", ""),
            request.state.get("project_memory", ""),
            request.state.get("project_root"),
            request.system_prompt,
        )
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
        system_prompt = self._build_system_prompt(
            request.state.get("agent_memory", ""),
            request.state.get("project_memory", ""),
            request.state.get("project_root"),
            request.system_prompt,
        )
        return await handler(request.override(system_prompt=system_prompt))
