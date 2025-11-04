"""Agent management and creation for the CLI."""

import os
import shutil
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.resumable_shell import ResumableShellToolMiddleware
from langchain.agents.middleware import HostExecutionPolicy
from langgraph.checkpoint.memory import InMemorySaver

from .agent_memory import AgentMemoryMiddleware
from .config import COLORS, config, console, get_default_coding_instructions


def list_agents():
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


def reset_agent(agent_name: str, source_agent: str = None):
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


def get_system_prompt() -> str:
    """Get the base system prompt for the agent.

    Returns:
        The system prompt string (without agent.md content)
    """
    return f"""### Current Working Directory

The filesystem backend is currently operating in: `{Path.cwd()}`

### Memory System Reminder

Your long-term memory is stored in /memories/ and persists across sessions.

**IMPORTANT - Check memories before answering:**
- When asked "what do you know about X?" → Run `ls /memories/` FIRST, then read relevant files
- When starting a task → Check if you have guides or examples in /memories/
- At the beginning of new sessions → Consider checking `ls /memories/` to see what context you have

Base your answers on saved knowledge (from /memories/) when available, supplemented by general knowledge.

### Human-in-the-Loop Tool Approval

Some tool calls require user approval before execution. When a tool call is rejected by the user:
1. Accept their decision immediately - do NOT retry the same command
2. Explain that you understand they rejected the action
3. Suggest an alternative approach or ask for clarification
4. Never attempt the exact same rejected command again

Respect the user's decisions and work with them collaboratively.

### Web Search Tool Usage

When you use the web_search tool:
1. The tool will return search results with titles, URLs, and content excerpts
2. You MUST read and process these results, then respond naturally to the user
3. NEVER show raw JSON or tool results directly to the user
4. Synthesize the information from multiple sources into a coherent answer
5. Cite your sources by mentioning page titles or URLs when relevant
6. If the search doesn't find what you need, explain what you found and ask clarifying questions

The user only sees your text responses - not tool results. Always provide a complete, natural language answer after using web_search.

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


def create_agent_with_config(model, assistant_id: str, tools: list):
    """Create and configure an agent with the specified model and tools."""
    shell_middleware = ResumableShellToolMiddleware(
        workspace_root=os.getcwd(), execution_policy=HostExecutionPolicy()
    )

    # For long-term memory, point to ~/.deepagents/AGENT_NAME/ with /memories/ prefix
    agent_dir = Path.home() / ".deepagents" / assistant_id
    agent_dir.mkdir(parents=True, exist_ok=True)
    agent_md = agent_dir / "agent.md"
    if not agent_md.exists():
        source_content = get_default_coding_instructions()
        agent_md.write_text(source_content)

    # Long-term backend - rooted at agent directory
    # This handles both /memories/ files and /agent.md
    long_term_backend = FilesystemBackend(root_dir=agent_dir, virtual_mode=True)

    # Composite backend: current working directory for default, agent directory for /memories/
    backend = CompositeBackend(
        default=FilesystemBackend(), routes={"/memories/": long_term_backend}
    )

    # Use the same backend for agent memory middleware
    agent_middleware = [
        AgentMemoryMiddleware(backend=long_term_backend, memory_path="/memories/"),
        shell_middleware,
    ]

    # Get the system prompt
    system_prompt = get_system_prompt()

    # Helper functions for formatting tool descriptions in HITL prompts
    def format_write_file_description(tool_call: dict) -> str:
        """Format write_file tool call for approval prompt."""
        args = tool_call.get("args", {})
        file_path = args.get("file_path", "unknown")
        content = args.get("content", "")

        action = "Overwrite" if os.path.exists(file_path) else "Create"
        line_count = len(content.splitlines())
        size = len(content.encode("utf-8"))

        return f"File: {file_path}\nAction: {action} file\nLines: {line_count} · Bytes: {size}"

    def format_edit_file_description(tool_call: dict) -> str:
        """Format edit_file tool call for approval prompt."""
        args = tool_call.get("args", {})
        file_path = args.get("file_path", "unknown")
        old_string = args.get("old_string", "")
        new_string = args.get("new_string", "")
        replace_all = bool(args.get("replace_all", False))

        delta = len(new_string) - len(old_string)

        return (
            f"File: {file_path}\n"
            f"Action: Replace text ({'all occurrences' if replace_all else 'single occurrence'})\n"
            f"Snippet delta: {delta:+} characters"
        )

    def format_web_search_description(tool_call: dict) -> str:
        """Format web_search tool call for approval prompt."""
        args = tool_call.get("args", {})
        query = args.get("query", "unknown")
        max_results = args.get("max_results", 5)

        return f"Query: {query}\nMax results: {max_results}\n\n⚠️  This will use Tavily API credits"

    def format_task_description(tool_call: dict) -> str:
        """Format task (subagent) tool call for approval prompt."""
        args = tool_call.get("args", {})
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

    # Configure human-in-the-loop for potentially destructive tools
    from langchain.agents.middleware import InterruptOnConfig

    shell_interrupt_config: InterruptOnConfig = {
        "allowed_decisions": ["approve", "reject"],
        "description": lambda tool_call, state, runtime: (
            f"Shell Command: {tool_call['args'].get('command', 'N/A')}\n"
            f"Working Directory: {os.getcwd()}"
        ),
    }

    write_file_interrupt_config: InterruptOnConfig = {
        "allowed_decisions": ["approve", "reject"],
        "description": lambda tool_call, state, runtime: format_write_file_description(tool_call),
    }

    edit_file_interrupt_config: InterruptOnConfig = {
        "allowed_decisions": ["approve", "reject"],
        "description": lambda tool_call, state, runtime: format_edit_file_description(tool_call),
    }

    web_search_interrupt_config: InterruptOnConfig = {
        "allowed_decisions": ["approve", "reject"],
        "description": lambda tool_call, state, runtime: format_web_search_description(tool_call),
    }

    task_interrupt_config: InterruptOnConfig = {
        "allowed_decisions": ["approve", "reject"],
        "description": lambda tool_call, state, runtime: format_task_description(tool_call),
    }

    agent = create_deep_agent(
        model=model,
        system_prompt=system_prompt,
        tools=tools,
        backend=backend,
        middleware=agent_middleware,
        interrupt_on={
            "shell": shell_interrupt_config,
            "write_file": write_file_interrupt_config,
            "edit_file": edit_file_interrupt_config,
            "web_search": web_search_interrupt_config,
            "task": task_interrupt_config,
        },
    ).with_config(config)

    agent.checkpointer = InMemorySaver()

    return agent
