"""Agent management and creation for the CLI."""

import os
import shutil
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.sandbox import SandboxBackendProtocol
from langchain.agents.middleware import (
    HostExecutionPolicy,
    InterruptOnConfig,
)
from langchain.agents.middleware.types import AgentState
from langchain.messages import ToolCall
from langchain.tools import BaseTool
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.pregel import Pregel
from langgraph.runtime import Runtime

from deepagents_cli._internal import ResumableShellToolMiddleware
from deepagents_cli.agent_memory import AgentMemoryMiddleware
from deepagents_cli.config import COLORS, config, console, get_default_coding_instructions, settings
from deepagents_cli.integrations.sandbox_factory import get_default_working_dir
from deepagents_cli.skills import SkillsMiddleware
from deepagents_cli.skills.load import get_skills_directory


def get_agent_working_directory() -> Path:
    """Get the agent working directory from environment variable or default to current directory.
    
    Returns:
        Path to the working directory for the FilesystemBackend.
        If AGENT_WORKING_DIRECTORY is set, uses that (relative to project root or absolute).
        Otherwise, uses the current working directory.
    """
    working_dir = os.environ.get("AGENT_WORKING_DIRECTORY")
    if working_dir:
        # If it's an absolute path, use it directly
        if os.path.isabs(working_dir):
            return Path(working_dir).resolve()
        # Otherwise, treat it as relative to the project root or current directory
        # Try project root first if available
        if settings.project_root:
            return (settings.project_root / working_dir).resolve()
        # Fallback to current directory
        return (Path.cwd() / working_dir).resolve()
    # Default to current working directory
    return Path.cwd()


def list_agents() -> None:
    """List all available agents."""
    agents_dir = Path.home() / ".deepagents"

    if not agents_dir.exists() or not any(agents_dir.iterdir()):
        console.print("[yellow]No agents found.[/yellow]")
        console.print(
            "[dim]Agents will be created in ~/.deepagents/ when you first use them.[/dim]",
            style=COLORS["dim"],
        )
        return

    console.print("\n[bold]Available Agents:[/bold]\n", style=COLORS["primary"])

    for agent_path in sorted(agents_dir.iterdir()):
        if agent_path.is_dir():
            agent_name = agent_path.name
            agent_md = agent_path / "agent.md"

            if agent_md.exists():
                console.print(f"  • [bold]{agent_name}[/bold]", style=COLORS["primary"])
                console.print(f"    {agent_path}", style=COLORS["dim"])
            else:
                console.print(
                    f"  • [bold]{agent_name}[/bold] [dim](incomplete)[/dim]", style=COLORS["tool"]
                )
                console.print(f"    {agent_path}", style=COLORS["dim"])

    console.print()


def reset_agent(agent_name: str, source_agent: str | None = None) -> None:
    """Reset an agent to default or copy from another agent."""
    agents_dir = Path.home() / ".deepagents"
    agent_dir = agents_dir / agent_name

    if source_agent:
        source_dir = agents_dir / source_agent
        source_md = source_dir / "agent.md"

        if not source_md.exists():
            console.print(
                f"[bold red]Error:[/bold red] Source agent '{source_agent}' not found or has no agent.md"
            )
            return

        source_content = source_md.read_text()
        action_desc = f"contents of agent '{source_agent}'"
    else:
        source_content = get_default_coding_instructions()
        action_desc = "default"

    if agent_dir.exists():
        shutil.rmtree(agent_dir)
        console.print(f"Removed existing agent directory: {agent_dir}", style=COLORS["tool"])

    agent_dir.mkdir(parents=True, exist_ok=True)
    agent_md = agent_dir / "agent.md"
    agent_md.write_text(source_content)

    console.print(f"✓ Agent '{agent_name}' reset to {action_desc}", style=COLORS["primary"])
    console.print(f"Location: {agent_dir}\n", style=COLORS["dim"])


def get_system_prompt(assistant_id: str, sandbox_type: str | None = None) -> str:
    """Get the base system prompt for the agent.

    Args:
        assistant_id: The agent identifier for path references
        sandbox_type: Type of sandbox provider ("modal", "runloop", "daytona").
                     If None, agent is operating in local mode.

    Returns:
        The system prompt string (without agent.md content)
    """
    agent_dir_path = f"~/.deepagents/{assistant_id}"

    if sandbox_type:
        # Get provider-specific working directory

        working_dir = get_default_working_dir(sandbox_type)

        working_dir_section = f"""### Current Working Directory

You are operating in a **remote Linux sandbox** at `{working_dir}`.

All code execution and file operations happen in this sandbox environment.

**Important:**
- The CLI is running locally on the user's machine, but you execute code remotely
- Use `{working_dir}` as your working directory for all operations

"""
    else:
        # Get the actual working directory (may be from environment variable)
        agent_working_dir = get_agent_working_directory()
        working_dir_section = f"""<env>
Working directory: {agent_working_dir}
</env>

### Current Working Directory

The filesystem backend is currently operating in: `{agent_working_dir}`

### File System and Paths

**IMPORTANT - Path Handling:**
- All file paths must be absolute paths (e.g., `{agent_working_dir}/file.txt`)
- Use the working directory from <env> to construct absolute paths
- Example: To create a file in your working directory, use `{agent_working_dir}/research_project/file.md`
- Never use relative paths - always construct full absolute paths

"""

    return (
        working_dir_section
        + f"""### Skills Directory

Your skills are stored at: `{agent_dir_path}/skills/`
Skills may contain scripts or supporting files. When executing skill scripts with bash, use the real filesystem path:
Example: `bash python {agent_dir_path}/skills/web-research/script.py`

### Human-in-the-Loop Tool Approval

Some tool calls require user approval before execution. When a tool call is rejected by the user:
1. Accept their decision immediately - do NOT retry the same command
2. Explain that you understand they rejected the action
3. Suggest an alternative approach or ask for clarification
4. Never attempt the exact same rejected command again

Respect the user's decisions and work with them collaboratively.

### Todo List Management

When using the write_todos tool:
1. Keep the todo list MINIMAL - aim for 3-6 items maximum
2. Only create todos for complex, multi-step tasks that truly need tracking
3. Break down work into clear, actionable items without over-fragmenting
4. For simple tasks (1-2 steps), just do them directly without creating todos
5. When first creating a todo list for a task, ALWAYS ask the user if the plan looks good before starting work
   - Create the todos, let them render, then ask: "Does this plan look good?" or similar
   - Wait for the user's response before marking the first todo as in_progress
   - If they want changes, adjust the plan accordingly
6. Update todo status promptly as you complete each item

The todo list is a planning tool - use it judiciously to avoid overwhelming the user with excessive task tracking."""
    )


def _format_write_file_description(tool_call: ToolCall, state: AgentState, runtime: Runtime) -> str:
    """Format write_file tool call for approval prompt."""
    args = tool_call["args"]
    file_path = args.get("file_path", "unknown")
    content = args.get("content", "")

    action = "Overwrite" if os.path.exists(file_path) else "Create"
    line_count = len(content.splitlines())

    return f"File: {file_path}\nAction: {action} file\nLines: {line_count}"


def _format_edit_file_description(tool_call: ToolCall, state: AgentState, runtime: Runtime) -> str:
    """Format edit_file tool call for approval prompt."""
    args = tool_call["args"]
    file_path = args.get("file_path", "unknown")
    replace_all = bool(args.get("replace_all", False))

    return (
        f"File: {file_path}\n"
        f"Action: Replace text ({'all occurrences' if replace_all else 'single occurrence'})"
    )


def _format_web_search_description(tool_call: ToolCall, state: AgentState, runtime: Runtime) -> str:
    """Format web_search tool call for approval prompt."""
    args = tool_call["args"]
    query = args.get("query", "unknown")
    max_results = args.get("max_results", 5)

    return f"Query: {query}\nMax results: {max_results}\n\n⚠️  This will use Tavily API credits"


def _format_fetch_url_description(tool_call: ToolCall, state: AgentState, runtime: Runtime) -> str:
    """Format fetch_url tool call for approval prompt."""
    args = tool_call["args"]
    url = args.get("url", "unknown")
    timeout = args.get("timeout", 30)

    return f"URL: {url}\nTimeout: {timeout}s\n\n⚠️  Will fetch and convert web content to markdown"


def _format_task_description(tool_call: ToolCall, state: AgentState, runtime: Runtime) -> str:
    """Format task (subagent) tool call for approval prompt."""
    args = tool_call["args"]
    description = args.get("description", "unknown")
    prompt = args.get("prompt", "")

    # Truncate prompt if too long
    prompt_preview = prompt[:300]
    if len(prompt) > 300:
        prompt_preview += "..."

    return (
        f"Task: {description}\n\n"
        f"Instructions to subagent:\n"
        f"{'─' * 40}\n"
        f"{prompt_preview}\n"
        f"{'─' * 40}\n\n"
        f"⚠️  Subagent will have access to file operations and shell commands"
    )


def _format_shell_description(tool_call: ToolCall, state: AgentState, runtime: Runtime) -> str:
    """Format shell tool call for approval prompt."""
    args = tool_call["args"]
    command = args.get("command", "N/A")
    return f"Shell Command: {command}\nWorking Directory: {os.getcwd()}"


def _format_execute_description(tool_call: ToolCall, state: AgentState, runtime: Runtime) -> str:
    """Format execute tool call for approval prompt."""
    args = tool_call["args"]
    command = args.get("command", "N/A")
    return f"Execute Command: {command}\nLocation: Remote Sandbox"


def create_agent_with_config(
    model: str | BaseChatModel,
    assistant_id: str,
    tools: list[BaseTool],
    *,
    sandbox: SandboxBackendProtocol | None = None,
    sandbox_type: str | None = None,
) -> tuple[Pregel, CompositeBackend]:
    """Create and configure an agent with the specified model and tools.

    Args:
        model: LLM model to use
        assistant_id: Agent identifier for memory storage
        tools: Additional tools to provide to agent
        sandbox: Optional sandbox backend for remote execution (e.g., ModalBackend).
                 If None, uses local filesystem + shell.
        sandbox_type: Type of sandbox provider ("modal", "runloop", "daytona")

    Returns:
        2-tuple of graph and backend
    """
    # Setup agent directory for persistent memory (same for both local and remote modes)
    agent_dir = Path.home() / ".deepagents" / assistant_id
    agent_dir.mkdir(parents=True, exist_ok=True)
    agent_md = agent_dir / "agent.md"
    if not agent_md.exists():
        source_content = get_default_coding_instructions()
        agent_md.write_text(source_content)

    # Skills directory - from env var or default per-agent location
    skills_dir = get_skills_directory(assistant_id)
    skills_dir.mkdir(parents=True, exist_ok=True)

    # CONDITIONAL SETUP: Local vs Remote Sandbox
    if sandbox is None:
        # ========== LOCAL MODE ==========
        # Get working directory from environment variable or use current directory
        agent_working_dir = get_agent_working_directory()
        
        # Backend: Local filesystem rooted at the working directory
        composite_backend = CompositeBackend(
            # Disable virtual mode to let absolute paths under the working directory work without being
            # re-prefixed (temporary workaround for path resolution issues with uploaded files)
            default=FilesystemBackend(root_dir=str(agent_working_dir), virtual_mode=False),
            routes={
            },
        )

        # Middleware: AgentMemoryMiddleware, SkillsMiddleware, ResumableShellToolMiddleware
        agent_middleware = [
            AgentMemoryMiddleware(settings=settings, assistant_id=assistant_id),
            SkillsMiddleware(
                skills_dir=skills_dir,
                assistant_id=assistant_id,
                working_dir=agent_working_dir,
            ),
            ResumableShellToolMiddleware(
                workspace_root=str(agent_working_dir),
                execution_policy=HostExecutionPolicy(),
            ),
        ]
    else:
        # ========== REMOTE SANDBOX MODE ==========
        # Backend: Remote sandbox for code with /memories/ route to permanent filesystem storage
        # Even in sandbox mode, memories are stored locally on the filesystem
        composite_backend = CompositeBackend(
            default=sandbox,  # Remote sandbox (ModalBackend, etc.)
            routes={},  # No virtualization
        )

        # Middleware: AgentMemoryMiddleware and SkillsMiddleware
        # NOTE: File operations (ls, read, write, edit, glob, grep) and execute tool
        # are automatically provided by create_deep_agent when backend is a SandboxBackend.
        agent_middleware = [
            AgentMemoryMiddleware(settings=settings, assistant_id=assistant_id),
            SkillsMiddleware(skills_dir=skills_dir, assistant_id=assistant_id),
        ]

    # Get the system prompt (sandbox-aware and with skills)
    system_prompt = get_system_prompt(assistant_id=assistant_id, sandbox_type=sandbox_type)

    # Configure human-in-the-loop for potentially destructive tools
    shell_interrupt_config: InterruptOnConfig = {
        "allowed_decisions": ["approve", "reject"],
        "description": _format_shell_description,
    }

    execute_interrupt_config: InterruptOnConfig = {
        "allowed_decisions": ["approve", "reject"],
        "description": _format_execute_description,
    }

    write_file_interrupt_config: InterruptOnConfig = {
        "allowed_decisions": ["approve", "reject"],
        "description": _format_write_file_description,
    }

    edit_file_interrupt_config: InterruptOnConfig = {
        "allowed_decisions": ["approve", "reject"],
        "description": _format_edit_file_description,
    }

    web_search_interrupt_config: InterruptOnConfig = {
        "allowed_decisions": ["approve", "reject"],
        "description": _format_web_search_description,
    }

    fetch_url_interrupt_config: InterruptOnConfig = {
        "allowed_decisions": ["approve", "reject"],
        "description": _format_fetch_url_description,
    }

    task_interrupt_config: InterruptOnConfig = {
        "allowed_decisions": ["approve", "reject"],
        "description": _format_task_description,
    }

    agent = create_deep_agent(
        model=model,
        system_prompt=system_prompt,
        tools=tools,
        backend=composite_backend,
        middleware=agent_middleware,
        interrupt_on={
            "shell": shell_interrupt_config,
            "execute": execute_interrupt_config,
            "write_file": write_file_interrupt_config,
            "edit_file": edit_file_interrupt_config,
            "web_search": web_search_interrupt_config,
            "fetch_url": fetch_url_interrupt_config,
            "task": task_interrupt_config,
        },
    ).with_config(config)

    agent.checkpointer = InMemorySaver()

    return agent, composite_backend
