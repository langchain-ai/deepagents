"""UI rendering and display utilities for the CLI."""

import argparse
import json
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any

from deepagents_cli._version import __version__
from deepagents_cli.backends import DEFAULT_EXECUTE_TIMEOUT
from deepagents_cli.config import (
    COLORS,
    MAX_ARG_LENGTH,
    _is_editable_install,
    console,
    get_glyphs,
)


def build_help_parent(
    help_fn: Callable[[], None],
    make_help_action: Callable[[Callable[[], None]], type[argparse.Action]],
) -> list[argparse.ArgumentParser]:
    """Build a parent parser whose `-h` invokes *help_fn*.

    This eliminates boilerplate: without the helper every `add_parser`
    call would need its own three-line parent-parser setup.  Used by both
    `main.parse_args` and `skills.commands.setup_skills_parser`.

    Args:
        help_fn: Zero-argument callable that renders a Rich help screen.
        make_help_action: Factory that turns *help_fn* into an argparse
            Action class (see `main._make_help_action`).

    Returns:
        Single-element list suitable for the `parents` kwarg of
        `add_parser`.
    """
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("-h", "--help", action=make_help_action(help_fn))
    return [parent]


def _format_timeout(seconds: int) -> str:
    """Format timeout in human-readable units (e.g., 300 -> '5m', 3600 -> '1h').

    Args:
        seconds: The timeout value in seconds to format.

    Returns:
        Human-readable timeout string (e.g., '5m', '1h', '300s').
    """
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600 and seconds % 60 == 0:
        return f"{seconds // 60}m"
    if seconds % 3600 == 0:
        return f"{seconds // 3600}h"
    # For odd values, just show seconds
    return f"{seconds}s"


def truncate_value(value: str, max_length: int = MAX_ARG_LENGTH) -> str:
    """Truncate a string value if it exceeds max_length.

    Returns:
        Truncated string with ellipsis suffix if exceeded, otherwise original.
    """
    if len(value) > max_length:
        return value[:max_length] + get_glyphs().ellipsis
    return value


def format_tool_display(tool_name: str, tool_args: dict) -> str:
    """Format tool calls for display with tool-specific smart formatting.

    Shows the most relevant information for each tool type rather than all arguments.

    Args:
        tool_name: Name of the tool being called
        tool_args: Dictionary of tool arguments

    Returns:
        Formatted string for display (e.g., "(*) read_file(config.py)" in ASCII mode)

    Examples:
        read_file(path="/long/path/file.py") → "<prefix> read_file(file.py)"
        web_search(query="how to code") → '<prefix> web_search("how to code")'
        execute(command="pip install foo") → '<prefix> execute("pip install foo")'
    """
    prefix = get_glyphs().tool_prefix

    def abbreviate_path(path_str: str, max_length: int = 60) -> str:
        """Abbreviate a file path intelligently - show basename or relative path.

        Returns:
            Shortened path string suitable for display.
        """
        try:
            path = Path(path_str)

            # If it's just a filename (no directory parts), return as-is
            if len(path.parts) == 1:
                return path_str

            # Try to get relative path from current working directory
            with suppress(
                ValueError,  # ValueError: path is not relative to cwd
                OSError,  # OSError: filesystem errors when resolving paths
            ):
                rel_path = path.relative_to(Path.cwd())
                rel_str = str(rel_path)
                # Use relative if it's shorter and not too long
                if len(rel_str) < len(path_str) and len(rel_str) <= max_length:
                    return rel_str

            # If absolute path is reasonable length, use it
            if len(path_str) <= max_length:
                return path_str
        except Exception:
            # Fallback to original string if any error
            return truncate_value(path_str, max_length)
        else:
            # Otherwise, just show basename (filename only)
            return path.name

    # Tool-specific formatting - show the most important argument(s)
    if tool_name in {"read_file", "write_file", "edit_file"}:
        # File operations: show the primary file path argument (file_path or path)
        path_value = tool_args.get("file_path")
        if path_value is None:
            path_value = tool_args.get("path")
        if path_value is not None:
            path = abbreviate_path(str(path_value))
            return f"{prefix} {tool_name}({path})"

    elif tool_name == "web_search":
        # Web search: show the query string
        if "query" in tool_args:
            query = str(tool_args["query"])
            query = truncate_value(query, 100)
            return f'{prefix} {tool_name}("{query}")'

    elif tool_name == "grep":
        # Grep: show the search pattern
        if "pattern" in tool_args:
            pattern = str(tool_args["pattern"])
            pattern = truncate_value(pattern, 70)
            return f'{prefix} {tool_name}("{pattern}")'

    elif tool_name == "execute":
        # Execute: show the command, and timeout only if non-default
        if "command" in tool_args:
            command = str(tool_args["command"])
            command = truncate_value(command, 120)
            timeout = tool_args.get("timeout")
            if timeout is not None and timeout != DEFAULT_EXECUTE_TIMEOUT:
                timeout_str = _format_timeout(timeout)
                return f'{prefix} {tool_name}("{command}", timeout={timeout_str})'
            return f'{prefix} {tool_name}("{command}")'

    elif tool_name == "ls":
        # ls: show directory, or empty if current directory
        if tool_args.get("path"):
            path = abbreviate_path(str(tool_args["path"]))
            return f"{prefix} {tool_name}({path})"
        return f"{prefix} {tool_name}()"

    elif tool_name == "glob":
        # Glob: show the pattern
        if "pattern" in tool_args:
            pattern = str(tool_args["pattern"])
            pattern = truncate_value(pattern, 80)
            return f'{prefix} {tool_name}("{pattern}")'

    elif tool_name == "http_request":
        # HTTP: show method and URL
        parts = []
        if "method" in tool_args:
            parts.append(str(tool_args["method"]).upper())
        if "url" in tool_args:
            url = str(tool_args["url"])
            url = truncate_value(url, 80)
            parts.append(url)
        if parts:
            return f"{prefix} {tool_name}({' '.join(parts)})"

    elif tool_name == "fetch_url":
        # Fetch URL: show the URL being fetched
        if "url" in tool_args:
            url = str(tool_args["url"])
            url = truncate_value(url, 80)
            return f'{prefix} {tool_name}("{url}")'

    elif tool_name == "task":
        # Task: show the task description
        if "description" in tool_args:
            desc = str(tool_args["description"])
            desc = truncate_value(desc, 100)
            return f'{prefix} {tool_name}("{desc}")'

    elif tool_name == "write_todos":
        # Todos: show count of items
        if "todos" in tool_args and isinstance(tool_args["todos"], list):
            count = len(tool_args["todos"])
            return f"{prefix} {tool_name}({count} items)"

    # Fallback: generic formatting for unknown tools
    # Show all arguments in key=value format
    args_str = ", ".join(
        f"{k}={truncate_value(str(v), 50)}" for k, v in tool_args.items()
    )
    return f"{prefix} {tool_name}({args_str})"


def format_tool_message_content(content: Any) -> str:
    """Convert ToolMessage content into a printable string.

    Returns:
        Formatted string representation of the tool message content.
    """
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


def show_help() -> None:
    """Show top-level help information for the deepagents CLI."""
    install_type = " (local)" if _is_editable_install() else ""
    banner_color = (
        COLORS["primary_dev"] if _is_editable_install() else COLORS["primary"]
    )
    console.print()
    console.print(
        f"[bold {banner_color}]deepagents-cli[/bold {banner_color}]"
        f" v{__version__}{install_type}"
    )
    console.print()
    console.print(
        "Docs: https://docs.langchain.com/oss/python/deepagents/cli",
        style=COLORS["dim"],
    )
    console.print()
    console.print("[bold]Usage:[/bold]", style=COLORS["primary"])
    console.print(
        "  deepagents [OPTIONS]                           Start interactive thread"
    )
    console.print(
        "  deepagents list                                List all available agents"
    )
    console.print(
        "  deepagents reset --agent AGENT [--target SRC]  Reset an agent's prompt"
    )
    console.print(
        "  deepagents skills <list|create|info>           Manage agent skills"
    )
    console.print(
        "  deepagents threads <list|delete>               Manage conversation threads"
    )
    console.print()

    console.print("[bold]Options:[/bold]", style=COLORS["primary"])
    console.print(
        "  -r, --resume [ID]          Resume thread: -r for most recent, -r ID for specific"  # noqa: E501
    )
    console.print("  -a, --agent NAME           Agent to use (e.g., coder, researcher)")
    console.print("  -M, --model MODEL          Model to use (e.g., gpt-4o)")
    console.print("  -m, --message TEXT         Initial prompt to auto-submit on start")
    console.print(
        "  --auto-approve             Auto-approve all tool calls (toggle: Shift+Tab)"
    )
    console.print("  --sandbox TYPE             Remote sandbox for execution")
    console.print(
        "  --sandbox-id ID            Reuse existing sandbox (skips creation/cleanup)"
    )
    console.print(
        "  --sandbox-setup PATH       Setup script to run in sandbox after creation"
    )
    console.print("  -n, --non-interactive MSG  Run a single task and exit")
    console.print(
        "  --shell-allow-list CMDS    Comma-separated local shell commands to allow"
    )
    console.print("  --default-model [MODEL]    Set, show, or manage the default model")
    console.print("  --clear-default-model      Clear the default model")
    console.print("  -v, --version              Show deepagents CLI version")
    console.print("  -h, --help                 Show this help message and exit")
    console.print()

    console.print("[bold]Non-Interactive Mode:[/bold]", style=COLORS["primary"])
    console.print(
        "  deepagents -n 'Summarize README.md'     # Run task (no local shell access)",
        style=COLORS["dim"],
    )
    console.print(
        "  deepagents -n 'List files' --shell-allow-list recommended  # Use safe commands",  # noqa: E501
        style=COLORS["dim"],
    )
    console.print(
        "  deepagents -n 'Search logs' --shell-allow-list ls,cat,grep # Specify list",
        style=COLORS["dim"],
    )
    console.print()


def show_list_help() -> None:
    """Show help information for the `list` subcommand.

    Invoked via the `-h` argparse action or directly from `cli_main`.
    """
    console.print()
    console.print("[bold]Usage:[/bold]", style=COLORS["primary"])
    console.print("  deepagents list")
    console.print()
    console.print(
        "List all agents found in ~/.deepagents/. Each agent has its own",
    )
    console.print(
        "AGENTS.md system prompt and separate thread history.",
    )
    console.print()
    console.print("[bold]Options:[/bold]", style=COLORS["primary"])
    console.print("  -h, --help        Show this help message")
    console.print()


def show_reset_help() -> None:
    """Show help information for the `reset` subcommand."""
    console.print()
    console.print("[bold]Usage:[/bold]", style=COLORS["primary"])
    console.print("  deepagents reset --agent NAME [--target SRC]")
    console.print()
    console.print(
        "Restore an agent's AGENTS.md to the built-in default, or copy",
    )
    console.print(
        "another agent's AGENTS.md. This deletes the agent's directory",
    )
    console.print(
        "and recreates it with the new prompt.",
    )
    console.print()
    console.print("[bold]Options:[/bold]", style=COLORS["primary"])
    console.print("  --agent NAME      Agent to reset (required)")
    console.print("  --target SRC      Copy AGENTS.md from another agent instead")
    console.print("  -h, --help        Show this help message")
    console.print()
    console.print("[bold]Examples:[/bold]", style=COLORS["primary"])
    console.print("  deepagents reset --agent coder")
    console.print("  deepagents reset --agent coder --target researcher")
    console.print()


def show_skills_help() -> None:
    """Show help information for the `skills` subcommand.

    Invoked via the `-h` argparse action or directly from
    `execute_skills_command` when no subcommand is given.
    """
    console.print()
    console.print("[bold]Usage:[/bold]", style=COLORS["primary"])
    console.print("  deepagents skills <command> [options]")
    console.print()
    console.print("[bold]Commands:[/bold]", style=COLORS["primary"])
    console.print("  list|ls           List all available skills")
    console.print("  create <name>     Create a new skill")
    console.print("  info <name>       Show detailed information about a skill")
    console.print()
    console.print("[bold]Options:[/bold]", style=COLORS["primary"])
    console.print("  -h, --help        Show this help message")
    console.print()
    console.print(
        "[dim]Skills are loaded from these directories "
        "(highest precedence first):\n"
        "  1. .agents/skills/                 project skills\n"
        "  2. .deepagents/skills/             project skills (alias)\n"
        "  3. ~/.agents/skills/               user skills\n"
        "  4. ~/.deepagents/<agent>/skills/   user skills (alias)\n"
        "  5. <package>/built_in_skills/      built-in skills[/dim]",
        style=COLORS["dim"],
    )
    console.print(
        "\n[dim]Create your first skill:\n  deepagents skills create my-skill[/dim]",
        style=COLORS["dim"],
    )
    console.print()


def show_skills_list_help() -> None:
    """Show help information for the `skills list` subcommand."""
    console.print()
    console.print("[bold]Usage:[/bold]", style=COLORS["primary"])
    console.print("  deepagents skills list [options]")
    console.print()
    console.print("[bold]Options:[/bold]", style=COLORS["primary"])
    console.print("  --agent NAME      Agent identifier (default: agent)")
    console.print("  --project         Show only project-level skills")
    console.print("  -h, --help        Show this help message")
    console.print()


def show_skills_create_help() -> None:
    """Show help information for the `skills create` subcommand."""
    console.print()
    console.print("[bold]Usage:[/bold]", style=COLORS["primary"])
    console.print("  deepagents skills create <name> [options]")
    console.print()
    console.print("[bold]Options:[/bold]", style=COLORS["primary"])
    console.print("  --agent NAME      Agent identifier (default: agent)")
    console.print(
        "  --project         Create in project directory instead of user directory"
    )
    console.print("  -h, --help        Show this help message")
    console.print()
    console.print("[bold]Examples:[/bold]", style=COLORS["primary"])
    console.print("  deepagents skills create web-research")
    console.print("  deepagents skills create my-skill --project")
    console.print()


def show_skills_info_help() -> None:
    """Show help information for the `skills info` subcommand."""
    console.print()
    console.print("[bold]Usage:[/bold]", style=COLORS["primary"])
    console.print("  deepagents skills info <name> [options]")
    console.print()
    console.print("[bold]Options:[/bold]", style=COLORS["primary"])
    console.print("  --agent NAME      Agent identifier (default: agent)")
    console.print("  --project         Search only in project skills")
    console.print("  -h, --help        Show this help message")
    console.print()


def show_threads_help() -> None:
    """Show help information for the `threads` subcommand.

    Invoked via the `-h` argparse action or directly from `cli_main`
    when no threads subcommand is given.
    """
    console.print()
    console.print("[bold]Usage:[/bold]", style=COLORS["primary"])
    console.print("  deepagents threads <command> [options]")
    console.print()
    console.print("[bold]Commands:[/bold]", style=COLORS["primary"])
    console.print("  list|ls           List all threads")
    console.print("  delete <ID>       Delete a thread")
    console.print()
    console.print("[bold]Options:[/bold]", style=COLORS["primary"])
    console.print("  -h, --help        Show this help message")
    console.print()
    console.print("[bold]Examples:[/bold]", style=COLORS["primary"])
    console.print("  deepagents threads list")
    console.print("  deepagents threads delete abc123")
    console.print()


def show_threads_delete_help() -> None:
    """Show help information for the `threads delete` subcommand."""
    console.print()
    console.print("[bold]Usage:[/bold]", style=COLORS["primary"])
    console.print("  deepagents threads delete <ID>")
    console.print()
    console.print("[bold]Options:[/bold]", style=COLORS["primary"])
    console.print("  -h, --help        Show this help message")
    console.print()
    console.print("[bold]Examples:[/bold]", style=COLORS["primary"])
    console.print("  deepagents threads delete abc123")
    console.print()


def show_threads_list_help() -> None:
    """Show help information for the `threads list` subcommand."""
    console.print()
    console.print("[bold]Usage:[/bold]", style=COLORS["primary"])
    console.print("  deepagents threads list [options]")
    console.print()
    console.print("[bold]Options:[/bold]", style=COLORS["primary"])
    console.print("  --agent NAME      Filter by agent name")
    console.print("  --limit N         Maximum threads to display (default: 20)")
    console.print("  -h, --help        Show this help message")
    console.print()
    console.print("[bold]Examples:[/bold]", style=COLORS["primary"])
    console.print("  deepagents threads list")
    console.print("  deepagents threads list --agent mybot")
    console.print("  deepagents threads list --limit 50")
    console.print()
