"""Command handlers for slash commands and bash execution."""

import subprocess
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver

from .config import COLORS, DEEP_AGENTS_ASCII, console
from .ui import TokenTracker, show_interactive_help


def handle_command(command: str, agent, token_tracker: TokenTracker) -> str | bool | tuple[str, str]:
    """Handle slash commands.
    
    Returns:
        - 'exit': Exit the CLI
        - True: Command handled, continue
        - False: Pass to agent
        - tuple[str, str]: ('switch_agent', agent_name) to switch agent profiles
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
        show_interactive_help()
        return True

    if cmd == "tokens":
        token_tracker.display_session()
        return True

    console.print()
    console.print(f"[yellow]Unknown command: /{cmd}[/yellow]")
    console.print("[dim]Type /help for available commands.[/dim]")
    console.print()
    return True

    return False


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
