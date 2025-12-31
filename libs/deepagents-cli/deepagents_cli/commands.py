"""Command handlers for slash commands and bash execution."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from .config import COLORS, DEEP_AGENTS_ASCII, console
from .sessions import generate_thread_id
from .ui import TokenTracker, show_interactive_help

if TYPE_CHECKING:
    from .config import SessionState


async def handle_command(
    command: str,
    token_tracker: TokenTracker,
    session_state: SessionState | None = None,
) -> str | bool:
    """Handle slash commands.

    Args:
        command: The slash command to handle
        token_tracker: Token usage tracker
        session_state: Optional session state for persistent threads

    Returns:
        'exit' to exit, True if handled, False to pass to agent
    """
    cmd = command.lower().strip().lstrip("/")

    if cmd in ["quit", "exit", "q"]:
        return "exit"

    if cmd == "clear":
        new_thread_id = None
        if session_state is not None:
            new_thread_id = generate_thread_id()
            session_state.thread_id = new_thread_id

        token_tracker.reset()
        console.clear()
        console.print(DEEP_AGENTS_ASCII, style=f"bold {COLORS['primary']}")
        console.print()
        console.print("Conversation cleared. Starting new thread.", style=COLORS["agent"])
        if new_thread_id:
            console.print(f"[dim]Thread: {new_thread_id}[/dim]")
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
