"""UI rendering and display utilities for the CLI."""
import json
from typing import Any
from pathlib import Path

from rich.panel import Panel
from rich import box

from .config import console, COLORS, COMMANDS, MAX_ARG_LENGTH, DEEP_AGENTS_ASCII


def truncate_value(value: str, max_length: int = MAX_ARG_LENGTH) -> str:
    """Truncate a string value if it exceeds max_length."""
    if len(value) > max_length:
        return value[:max_length] + "..."
    return value


def format_tool_message_content(content: Any) -> str:
    """Convert ToolMessage content into a printable string."""
    if content is None:
        return ""
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            else:
                try:
                    parts.append(json.dumps(item))
                except Exception:
                    parts.append(str(item))
        return "\n".join(parts)
    return str(content)


class TokenTracker:
    """Track token usage across the conversation."""

    def __init__(self):
        self.session_input = 0
        self.session_output = 0
        self.last_input = 0
        self.last_output = 0

    def add(self, input_tokens: int, output_tokens: int):
        """Add tokens from a response."""
        self.session_input += input_tokens
        self.session_output += output_tokens
        self.last_input = input_tokens
        self.last_output = output_tokens

    def display_last(self):
        """Display tokens for the last response."""
        # Only show output tokens generated in this turn
        if self.last_output:
            if self.last_output >= 1000:
                console.print(f"  {self.last_output:,} tokens", style="dim")

    def display_session(self):
        """Display cumulative session tokens."""
        total = self.session_input + self.session_output
        console.print(f"\n[bold]Session Token Usage:[/bold]", style=COLORS["primary"])
        console.print(f"  Input:  {self.session_input:,} tokens", style=COLORS["dim"])
        console.print(f"  Output: {self.session_output:,} tokens", style=COLORS["dim"])
        console.print(f"  Total:  {total:,} tokens\n", style=COLORS["dim"])


def render_todo_list(todos: list[dict]) -> None:
    """Render todo list as a rich Panel with checkboxes."""
    if not todos:
        return

    lines = []
    for todo in todos:
        status = todo.get("status", "pending")
        content = todo.get("content", "")

        if status == "completed":
            icon = "☑"
            style = "green"
        elif status == "in_progress":
            icon = "⏳"
            style = "yellow"
        else:  # pending
            icon = "☐"
            style = "dim"

        lines.append(f"[{style}]{icon} {content}[/{style}]")

    panel = Panel(
        "\n".join(lines),
        title="[bold]Task List[/bold]",
        border_style="cyan",
        box=box.ROUNDED,
        padding=(0, 1)
    )
    console.print(panel)


def show_interactive_help():
    """Show available commands during interactive session."""
    console.print()
    console.print("[bold]Interactive Commands:[/bold]", style=COLORS["primary"])
    console.print()

    for cmd, desc in COMMANDS.items():
        console.print(f"  /{cmd:<12} {desc}", style=COLORS["dim"])

    console.print()
    console.print("[bold]Editing Features:[/bold]", style=COLORS["primary"])
    console.print("  Enter           Submit your message", style=COLORS["dim"])
    console.print("  Alt+Enter       Insert newline (Option+Enter on Mac, or ESC then Enter)", style=COLORS["dim"])
    console.print("  Ctrl+E          Open in external editor (nano by default)", style=COLORS["dim"])
    console.print("  Ctrl+T          Toggle auto-approve mode", style=COLORS["dim"])
    console.print("  Arrow keys      Navigate input and history", style=COLORS["dim"])
    console.print("  Ctrl+C          Cancel input or interrupt agent mid-work", style=COLORS["dim"])
    console.print()
    console.print("[bold]Special Features:[/bold]", style=COLORS["primary"])
    console.print("  @filename       Type @ to auto-complete files and inject content", style=COLORS["dim"])
    console.print("  /command        Type / to see available commands", style=COLORS["dim"])
    console.print("  !command        Type ! to run bash commands (e.g., !ls, !git status)", style=COLORS["dim"])
    console.print("                  Completions appear automatically as you type", style=COLORS["dim"])
    console.print()
    console.print("[bold]Auto-Approve Mode:[/bold]", style=COLORS["primary"])
    console.print("  Ctrl+T          Toggle auto-approve mode", style=COLORS["dim"])
    console.print("  --auto-approve  Start CLI with auto-approve enabled (via command line)", style=COLORS["dim"])
    console.print("  When enabled, tool actions execute without confirmation prompts", style=COLORS["dim"])
    console.print()


def show_help():
    """Show help information."""
    console.print()
    console.print(DEEP_AGENTS_ASCII, style=f"bold {COLORS['primary']}")
    console.print()

    console.print("[bold]Usage:[/bold]", style=COLORS["primary"])
    console.print("  deepagents [--agent NAME] [--auto-approve]     Start interactive session")
    console.print("  deepagents list                                List all available agents")
    console.print("  deepagents reset --agent AGENT                 Reset agent to default prompt")
    console.print("  deepagents reset --agent AGENT --target SOURCE Reset agent to copy of another agent")
    console.print("  deepagents help                                Show this help message")
    console.print()

    console.print("[bold]Examples:[/bold]", style=COLORS["primary"])
    console.print("  deepagents                              # Start with default agent", style=COLORS["dim"])
    console.print("  deepagents --agent mybot                # Start with agent named 'mybot'", style=COLORS["dim"])
    console.print("  deepagents --auto-approve               # Start with auto-approve enabled", style=COLORS["dim"])
    console.print("  deepagents list                         # List all agents", style=COLORS["dim"])
    console.print("  deepagents reset --agent mybot          # Reset mybot to default", style=COLORS["dim"])
    console.print("  deepagents reset --agent mybot --target other # Reset mybot to copy of 'other' agent", style=COLORS["dim"])
    console.print()

    console.print("[bold]Long-term Memory:[/bold]", style=COLORS["primary"])
    console.print("  By default, long-term memory is ENABLED using agent name 'agent'.", style=COLORS["dim"])
    console.print("  Memory includes:", style=COLORS["dim"])
    console.print("  - Persistent agent.md file with your instructions", style=COLORS["dim"])
    console.print("  - /memories/ folder for storing context across sessions", style=COLORS["dim"])
    console.print()

    console.print("[bold]Agent Storage:[/bold]", style=COLORS["primary"])
    console.print("  Agents are stored in: ~/.deepagents/AGENT_NAME/", style=COLORS["dim"])
    console.print("  Each agent has an agent.md file containing its prompt", style=COLORS["dim"])
    console.print()

    console.print("[bold]Interactive Features:[/bold]", style=COLORS["primary"])
    console.print("  Enter           Submit your message", style=COLORS["dim"])
    console.print("  Alt+Enter       Insert newline for multi-line (Option+Enter or ESC then Enter)", style=COLORS["dim"])
    console.print("  Ctrl+J          Insert newline (alternative)", style=COLORS["dim"])
    console.print("  Ctrl+T          Toggle auto-approve mode", style=COLORS["dim"])
    console.print("  Arrow keys      Navigate input and command history", style=COLORS["dim"])
    console.print("  @filename       Type @ to auto-complete files and inject content", style=COLORS["dim"])
    console.print("  /command        Type / to see available commands (auto-completes)", style=COLORS["dim"])
    console.print()

    console.print("[bold]Interactive Commands:[/bold]", style=COLORS["primary"])
    console.print("  /help           Show available commands and features", style=COLORS["dim"])
    console.print("  /clear          Clear screen and reset conversation", style=COLORS["dim"])
    console.print("  /tokens         Show token usage for current session", style=COLORS["dim"])
    console.print("  /quit, /exit    Exit the session", style=COLORS["dim"])
    console.print("  quit, exit, q   Exit the session (just type and press Enter)", style=COLORS["dim"])
    console.print()
