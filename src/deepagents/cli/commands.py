"""Command handlers for slash commands and bash execution."""
import subprocess
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver

from .config import console, COLORS, DEEP_AGENTS_ASCII
from .ui import show_interactive_help, TokenTracker


def handle_command(command: str, agent, token_tracker: TokenTracker) -> str | bool:
    """Handle slash commands. Returns 'exit' to exit, True if handled, False to pass to agent."""
    cmd = command.lower().strip().lstrip('/')

    if cmd in ['quit', 'exit', 'q']:
        return 'exit'

    elif cmd == 'clear':
        # Reset agent conversation state
        agent.checkpointer = InMemorySaver()

        # Clear screen and show fresh UI
        console.clear()
        console.print(DEEP_AGENTS_ASCII, style=f"bold {COLORS['primary']}")
        console.print()
        console.print("... Fresh start! Screen cleared and conversation reset.", style=COLORS["agent"])
        console.print()
        return True

    elif cmd == 'help':
        show_interactive_help()
        return True

    elif cmd == 'tokens':
        token_tracker.display_session()
        return True

    else:
        console.print()
        console.print(f"[yellow]Unknown command: /{cmd}[/yellow]")
        console.print(f"[dim]Type /help for available commands.[/dim]")
        console.print()
        return True

    return False


def execute_bash_command(command: str) -> bool:
    """Execute a bash command and display output. Returns True if handled."""
    cmd = command.strip().lstrip('!')

    if not cmd:
        return True

    try:
        console.print()
        console.print(f"[dim]$ {cmd}[/dim]")

        # Execute the command
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=Path.cwd()
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
