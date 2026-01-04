"""Command handlers for slash commands and bash execution."""

import subprocess
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

from .config import COLORS, DEEP_AGENTS_ASCII, console, settings
from .slash_commands import discover_commands, execute_command
from .ui import TokenTracker, show_interactive_help


def handle_command(
    command: str,
    agent,
    token_tracker: TokenTracker,
    agent_name: str = "agent",
) -> str | bool | tuple[str, Any]:
    """Handle slash commands.

    Args:
        command: The command string (e.g., "/help" or "/deploy production")
        agent: The agent instance
        token_tracker: Token tracking instance
        agent_name: Current agent name for loading custom commands

    Returns:
        - 'exit': Exit the CLI
        - True: Command handled, continue
        - False: Pass to agent
        - tuple[str, str]: ('switch_agent', agent_name) to switch agent profiles
        - tuple[str, str]: ('prefill_prompt', message) to send message to agent
        - tuple[str, dict]: ('custom_command', {prompt, model, allowed_tools}) for custom commands
    """
    cmd_full = command.strip().lstrip("/")
    cmd_parts = cmd_full.split(None, 1)  # Split on whitespace, max 2 parts
    cmd = cmd_parts[0].lower() if cmd_parts else ""

    if cmd in ["quit", "exit", "q"]:
        return "exit"
    
    if cmd == "agent":
        # Parse agent name
        if len(cmd_parts) < 2:
            console.print()
            console.print("[yellow]Usage: /agent <agent_name>[/yellow]")
            console.print("[dim]Example: /agent foo[/dim]")
            console.print()
            return True
        
        agent_name = cmd_parts[1].strip()
        if not agent_name:
            console.print()
            console.print("[yellow]Please specify an agent name[/yellow]")
            console.print()
            return True
        
        # Return signal to switch agent
        return ("switch_agent", agent_name)
    
    if cmd == "remember":
        # Get optional additional text after /remember
        additional_context = ""
        if len(cmd_parts) > 1:
            # Rejoin everything after "remember" (preserve original spacing)
            additional_context = command.strip()[len("/remember"):].strip()
        
        base_message = """Please review our conversation and update your memory files accordingly:

1. **Review what we worked on**: Look at the key decisions, patterns, preferences, and learnings from this conversation.

2. **Update your agent.md**: If there are universal behaviors, preferences, or patterns that should apply across all my projects, update `~/.deepagents/agent/agent.md`.

3. **Update project memory**: If there are project-specific conventions, architecture decisions, or workflows we established, update the project's `.deepagents/agent.md` or create relevant documentation files in `.deepagents/`.

4. **Update or create skills**: If we developed a reusable workflow or process that could be captured as a skill, create or update a skill in `~/.deepagents/agent/skills/`.

Focus on:
- Coding conventions and patterns I prefer
- Project architecture and design decisions
- Common workflows or debugging approaches
- Tools, libraries, or techniques we used
- Mistakes to avoid or gotchas we encountered
- Any feedback I gave about your behavior

Be specific and actionable in your updates. Use `edit_file` to update existing files or `write_file` to create new ones."""
        
        # Append user's additional context if provided
        if additional_context:
            final_message = f"{base_message}\n\nAdditional context: {additional_context}"
        else:
            final_message = base_message
        
        # Return signal to prefill prompt with memory reflection message
        return ("prefill_prompt", final_message)

    if cmd == "clear":
        # Reset agent conversation state
        agent.checkpointer = InMemorySaver()

        # Reset token tracking to baseline
        token_tracker.reset()

        # Clear screen and show fresh UI
        console.clear()
        console.print(DEEP_AGENTS_ASCII, style=f"bold {COLORS['primary']}")
        console.print()
        console.print(
            "... Fresh start! Screen cleared and conversation reset.", style=COLORS["agent"]
        )
        console.print()
        return True

    if cmd == "help":
        show_interactive_help(agent_name=agent_name, project_root=settings.project_root)
        return True

    if cmd == "tokens":
        token_tracker.display_session()
        return True

    # Check for custom slash commands
    custom_commands = discover_commands(agent_name, settings.project_root)
    if cmd in custom_commands:
        slash_cmd = custom_commands[cmd]
        args = cmd_parts[1] if len(cmd_parts) > 1 else ""

        # Execute the command (processes template, shell injections, file inclusions)
        result = execute_command(
            slash_cmd,
            args,
            cwd=Path.cwd(),
            backend=None,  # Could pass backend if available
        )

        # Return as custom_command tuple with metadata
        return (
            "custom_command",
            {
                "prompt": result.prompt,
                "model": result.model,
                "allowed_tools": result.allowed_tools,
            },
        )

    console.print()
    console.print(f"[yellow]Unknown command: /{cmd}[/yellow]")
    console.print("[dim]Type /help for available commands.[/dim]")
    console.print()
    return True


def execute_bash_command(command: str) -> bool:
    """Execute a bash command and display output. Returns True if handled."""
    cmd = command.strip().lstrip("!")

    if not cmd:
        return True

    try:
        console.print()
        console.print(f"[dim]$ {cmd}[/dim]")

        # Execute the command
        result = subprocess.run(
            cmd, check=False, shell=True, capture_output=True, text=True, timeout=30, cwd=Path.cwd()
        )

        # Display output
        if result.stdout:
            console.print(result.stdout, style=COLORS["dim"], markup=False)
        if result.stderr:
            console.print(result.stderr, style="red", markup=False)

        # Show return code if non-zero
        if result.returncode != 0:
            console.print(f"[dim]Exit code: {result.returncode}[/dim]")

        console.print()
        return True

    except subprocess.TimeoutExpired:
        console.print("[red]Command timed out after 30 seconds[/red]")
        console.print()
        return True
    except Exception as e:
        console.print(f"[red]Error executing command: {e}[/red]")
        console.print()
        return True
