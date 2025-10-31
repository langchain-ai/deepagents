"""Middleware for loading agent-specific long-term memory into the system prompt."""

from collections.abc import Awaitable, Callable
from typing import NotRequired

from deepagents.backends.protocol import BackendProtocol
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)


class AgentMemoryState(AgentState):
    """State for the agent memory middleware."""

    agent_memory: NotRequired[str | None]
    """Long-term memory content for the agent."""


AGENT_MEMORY_FILE_PATH = "/agent.md"

# Long-term Memory Documentation
LONGTERM_MEMORY_SYSTEM_PROMPT = """

## Long-term Memory

You have access to a long-term memory system using the {memory_path} path prefix.
Files stored in {memory_path} persist across sessions and conversations.

Your system prompt is loaded from {memory_path}agent.md at startup. You can update your own instructions by editing this file.

**When to CHECK/READ memories (CRITICAL - do this FIRST):**
- **At the start of ANY new session**: Run `ls {memory_path}` to see what you know
- **BEFORE answering questions**: If asked "what do you know about X?" or "how do I do Y?", check `ls {memory_path}` for relevant files FIRST
- **When user asks you to do something**: Check if you have guides, examples, or patterns in {memory_path} before proceeding
- **When user references past work or conversations**: Search {memory_path} for related content
- **If you're unsure**: Check your memories rather than guessing or using only general knowledge

**Memory-first response pattern:**
1. User asks a question → Run `ls {memory_path}` to check for relevant files
2. If relevant files exist → Read them with `read_file {memory_path}[filename]`
3. Base your answer on saved knowledge (from memories) supplemented by general knowledge
4. If no relevant memories exist → Use general knowledge, then consider if this is worth saving

**When to update memories:**
- **IMMEDIATELY when the user describes your role or how you should behave** (e.g., "you are a web researcher", "you are an expert in X")
- **IMMEDIATELY when the user gives feedback on your work** - Before continuing, update memories to capture what was wrong and how to do it better
- When the user explicitly asks you to remember something
- When patterns or preferences emerge (coding styles, conventions, workflows)
- After significant work where context would help in future sessions

**Learning from feedback:**
- When user says something is better/worse, capture WHY and encode it as a pattern
- Each correction is a chance to improve permanently - don't just fix the immediate issue, update your instructions
- When user says "you should remember X" or "be careful about Y", treat this as HIGH PRIORITY - update memories IMMEDIATELY
- Look for the underlying principle behind corrections, not just the specific mistake
- If it's something you "should have remembered", identify where that instruction should live permanently

**What to store where:**
- **{memory_path}agent.md**: Update this to modify your core instructions and behavioral patterns
- **Other {memory_path} files**: Use for project-specific context, reference information, or structured notes
  - If you create additional memory files, add references to them in {memory_path}agent.md so you remember to consult them

The portion of your system prompt that comes from {memory_path}agent.md is marked with `<agent_memory>` tags so you can identify what instructions come from your persistent memory.

Example: `ls {memory_path}` to see what memories you have
Example: `read_file '{memory_path}deep-agents-guide.md'` to recall saved knowledge
Example: `edit_file('{memory_path}agent.md', ...)` to update your instructions
Example: `write_file('{memory_path}project_context.md', ...)` for project-specific notes, then reference it in agent.md

Remember: To interact with the longterm filesystem, you must prefix the filename with the {memory_path} path."""


DEFAULT_MEMORY_SNIPPET = """<agent_memory>
{agent_memory}
</agent_memory>
"""


class AgentMemoryMiddleware(AgentMiddleware):
    """Middleware for loading agent-specific long-term memory.

    This middleware loads the agent's long-term memory from a file (agent.md)
    and injects it into the system prompt. The memory is loaded once at the
    start of the conversation and stored in state.

    Args:
        backend: Backend to use for loading the agent memory file.
        system_prompt_template: Optional custom template for how to inject
            the agent memory into the system prompt. Use {agent_memory} as
            a placeholder. Defaults to a simple section header.

    Example:
        ```python
        from deepagents.middleware.agent_memory import AgentMemoryMiddleware
        from deepagents.memory.backends import FilesystemBackend
        from pathlib import Path

        # Set up backend pointing to agent's directory
        agent_dir = Path.home() / ".deepagents" / "my-agent"
        backend = FilesystemBackend(root_dir=agent_dir)

        # Create middleware
        middleware = AgentMemoryMiddleware(backend=backend)
        ```
    """

    state_schema = AgentMemoryState

    def __init__(
        self,
        *,
        backend: BackendProtocol,
        memory_path: str,
        system_prompt_template: str | None = None,
    ) -> None:
        """Initialize the agent memory middleware.

        Args:
            backend: Backend to use for loading the agent memory file.
            system_prompt_template: Optional custom template for injecting
                agent memory into system prompt.
        """
        self.backend = backend
        self.memory_path = memory_path
        self.system_prompt_template = system_prompt_template or DEFAULT_MEMORY_SNIPPET

    def before_agent(
        self,
        state: AgentMemoryState,
        runtime,
    ) -> AgentMemoryState:
        """Load agent memory from file before agent execution.

        Args:
            state: Current agent state.
            handler: Handler function to call after loading memory.

        Returns:
            Updated state with agent_memory populated.
        """
        # Only load memory if it hasn't been loaded yet
        if "agent_memory" not in state or state.get("agent_memory") is None:
            file_data = self.backend.read(AGENT_MEMORY_FILE_PATH)
            return {"agent_memory": file_data}

    async def abefore_agent(
        self,
        state: AgentMemoryState,
        runtime,
    ) -> AgentMemoryState:
        """(async) Load agent memory from file before agent execution.

        Args:
            state: Current agent state.
            handler: Handler function to call after loading memory.

        Returns:
            Updated state with agent_memory populated.
        """
        # Only load memory if it hasn't been loaded yet
        if "agent_memory" not in state or state.get("agent_memory") is None:
            file_data = self.backend.read(AGENT_MEMORY_FILE_PATH)
            return {"agent_memory": file_data}

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
        # Get agent memory from state
        agent_memory = request.state.get("agent_memory", "")

        memory_section = self.system_prompt_template.format(agent_memory=agent_memory)
        if request.system_prompt:
            request.system_prompt = memory_section + "\n\n" + request.system_prompt
        else:
            request.system_prompt = memory_section
        request.system_prompt = (
            request.system_prompt
            + "\n\n"
            + LONGTERM_MEMORY_SYSTEM_PROMPT.format(memory_path=self.memory_path)
        )

        return handler(request)

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
        # Get agent memory from state
        agent_memory = request.state.get("agent_memory", "")

        memory_section = self.system_prompt_template.format(agent_memory=agent_memory)
        if request.system_prompt:
            request.system_prompt = memory_section + "\n\n" + request.system_prompt
        else:
            request.system_prompt = memory_section
        request.system_prompt = (
            request.system_prompt
            + "\n\n"
            + LONGTERM_MEMORY_SYSTEM_PROMPT.format(memory_path=self.memory_path)
        )

        return await handler(request)
