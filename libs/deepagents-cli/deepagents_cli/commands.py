"""Command handlers for slash commands and bash execution."""

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver

from .config import COLORS, DEEP_AGENTS_ASCII, SessionState, console
from .ui import TokenTracker, show_interactive_help


@dataclass
class CommandResult:
    """Result of a command handler."""

    handled: bool = True
    exit: bool = False
    return_to_parent: bool = False  # Signals main loop to inject summary and switch context


def handle_command(
    command: str, agent, token_tracker: TokenTracker, session_state: SessionState
) -> CommandResult | str | bool:
    """Handle slash commands. Returns 'exit' to exit, True if handled, False to pass to agent."""
    cmd = command.lower().strip().lstrip("/")

    if cmd in ["quit", "exit", "q"]:
        return "exit"

    if cmd == "clear":
        # Reset agent conversation state
        agent.checkpointer = InMemorySaver()

        # Reset context stack to fresh root
        session_state.reset_to_root()

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

    if cmd == "return":
        return _handle_return_command(session_state)

    if cmd == "summary":
        return _handle_summary_command(session_state)

    if cmd == "context":
        return _handle_context_command(session_state)

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


def _handle_return_command(session_state: SessionState) -> CommandResult | bool:
    """Handle /return command to exit from a stepped-into subagent.

    Returns CommandResult with return_to_parent=True to signal main loop to inject summary.
    """
    if session_state.depth == 0:
        console.print()
        console.print("[yellow]Already at root context - nothing to return from.[/yellow]")
        console.print()
        return True

    ctx = session_state.current_context
    console.print()
    console.print("━" * 60)
    console.print(f"[bold cyan]Returning from: {ctx.subagent_type}[/bold cyan]")

    # Read summary file if it exists
    summary_content = ""
    if ctx.summary_path and ctx.summary_path.exists():
        summary_content = ctx.summary_path.read_text()
        console.print(f"[dim]Summary loaded from: {ctx.summary_path}[/dim]")
    else:
        console.print("[yellow]No summary file found - returning with empty summary.[/yellow]")

    console.print("━" * 60)
    console.print()

    # Store summary content in context before returning
    # The main loop will handle popping and injecting
    ctx.summary_content = summary_content  # type: ignore[attr-defined]

    return CommandResult(handled=True, return_to_parent=True)


def _handle_summary_command(session_state: SessionState) -> bool:
    """Handle /summary command to view/edit the summary file."""
    if session_state.depth == 0:
        console.print()
        console.print("[yellow]Not in a subagent context - no summary file.[/yellow]")
        console.print()
        return True

    ctx = session_state.current_context
    if not ctx.summary_path:
        console.print()
        console.print("[yellow]No summary file for this context.[/yellow]")
        console.print()
        return True

    console.print()
    console.print(f"[bold]Summary file:[/bold] {ctx.summary_path}")
    console.print()

    # Check if $EDITOR is set
    editor = os.environ.get("EDITOR")
    if editor:
        console.print(f"[dim]Opening in {editor}...[/dim]")
        console.print()
        try:
            subprocess.run([editor, str(ctx.summary_path)], check=False)
        except Exception as e:
            console.print(f"[red]Error opening editor: {e}[/red]")
            console.print("[dim]Displaying file contents instead:[/dim]")
            console.print()
            _display_summary_file(ctx.summary_path)
    else:
        console.print("[dim]Set $EDITOR to open in your preferred editor.[/dim]")
        console.print("[dim]Displaying current contents:[/dim]")
        console.print()
        _display_summary_file(ctx.summary_path)

    console.print()
    return True


def _display_summary_file(path: Path) -> None:
    """Display the contents of a summary file."""
    if path.exists():
        content = path.read_text()
        console.print("─" * 40)
        console.print(content, markup=False)
        console.print("─" * 40)
    else:
        console.print("[yellow]File does not exist yet.[/yellow]")


def _handle_context_command(session_state: SessionState) -> bool:
    """Handle /context command to show the conversation stack."""
    console.print()
    console.print("━" * 40)
    console.print("[bold]Context Stack:[/bold]")
    console.print()

    for i, ctx in enumerate(session_state.context_stack):
        is_current = i == len(session_state.context_stack) - 1
        marker = " ← current" if is_current else ""

        if ctx.subagent_type == "root":
            console.print(f"  [{i}] root (main conversation){marker}")
        else:
            task_preview = ctx.task_description[:40] + "..." if len(ctx.task_description) > 40 else ctx.task_description
            console.print(f"  [{i}] {ctx.subagent_type}{marker}")
            if task_preview:
                console.print(f"      [dim]task: \"{task_preview}\"[/dim]")

    console.print()
    console.print("━" * 40)

    # Show summary path if in subagent
    if session_state.depth > 0:
        ctx = session_state.current_context
        if ctx.summary_path:
            console.print(f"[dim]Summary: {ctx.summary_path}[/dim]")

    console.print()
    return True
