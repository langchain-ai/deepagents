"""Command handlers for slash commands and bash execution."""

import os
import signal
import subprocess
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver

from .config import COLORS, DEEP_AGENTS_ASCII, console
from .ui import TokenTracker, show_interactive_help


def handle_command(command: str, agent, token_tracker: TokenTracker) -> str | bool:
    """Handle slash commands. Returns 'exit' to exit, True if handled, False to pass to agent."""
    cmd = command.lower().strip().lstrip("/")

    if cmd in ["quit", "exit", "q"]:
        return "exit"

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

    process = None
    interrupted = False
    original_handler = None

    def sigint_handler(signum, frame):
        """Custom SIGINT handler - kills subprocess immediately and sets flag."""
        nonlocal interrupted, process
        interrupted = True

        # Kill subprocess immediately when Ctrl+C is pressed
        if process and process.poll() is None:
            try:
                if hasattr(os, "killpg"):
                    os.killpg(os.getpgid(process.pid), signal.SIGINT)
                else:
                    process.send_signal(signal.SIGINT)
            except (ProcessLookupError, OSError):
                pass

    try:
        # Install our signal handler (temporarily overrides asyncio's handler)
        original_handler = signal.signal(signal.SIGINT, sigint_handler)

        console.print()
        console.print(f"[dim]$ {cmd}[/dim]")

        # Create subprocess in new session (process group) to isolate from parent's signals
        process = subprocess.Popen(
            cmd,
            shell=True,
            text=True,
            stdin=subprocess.DEVNULL,  # Close stdin to prevent interactive commands from hanging
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,  # Isolate subprocess in its own process group
            cwd=Path.cwd(),
        )

        try:
            # Wait for command to complete with timeout
            stdout, stderr = process.communicate(timeout=30)

            # Display output
            if stdout:
                console.print(stdout, style=COLORS["dim"], markup=False)
            if stderr:
                console.print(stderr, style="red", markup=False)

            # Check if interrupted via our flag
            if interrupted:
                console.print("\n[yellow]Command interrupted by user[/yellow]\n")
            elif process.returncode != 0:
                # Exit code 130 = 128 + SIGINT (2) - command was interrupted
                # Exit code -2 also indicates interrupt in some shells
                if process.returncode == 130 or process.returncode == -2:
                    console.print("[yellow]Command interrupted[/yellow]")
                else:
                    console.print(f"[dim]Exit code: {process.returncode}[/dim]")

        except subprocess.TimeoutExpired:
            # Timeout - kill the process group
            if hasattr(os, "killpg"):
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    pass
            else:
                process.kill()

            # Clean up zombie process
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass

            console.print("[red]Command timed out after 30 seconds[/red]")

        console.print()
        return True

    except Exception as e:
        console.print(f"[red]Error executing command: {e}[/red]")
        console.print()
        return True

    finally:
        # CRITICAL: Always restore original signal handler so asyncio can handle Ctrl+C at prompt
        if original_handler is not None:
            signal.signal(signal.SIGINT, original_handler)
