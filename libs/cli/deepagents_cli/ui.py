"""Help screens and argparse utilities for the CLI.

This module is imported at CLI startup to wire `-h` actions into the
argparse tree.  It must stay lightweight — no SDK or langchain imports.
"""

from rich.markup import escape

from deepagents_cli import theme
from deepagents_cli._version import DOCS_URL, __version__
from deepagents_cli.config import (
    _get_editable_install_path,
    _is_editable_install,
    console,
)

_JSON_OPTION_LINE = "  --json                  Emit machine-readable JSON"
_HELP_OPTION_LINE = "  -h, --help              Show this help message"


def _print_option_section(*lines: str, title: str = "Options") -> None:
    """Print a help-screen options section with shared JSON/help flags.

    Args:
        *lines: Command-specific option lines to print before the shared flags.
        title: Section title to display.
    """
    console.print(f"[bold]{title}:[/bold]", style=theme.PRIMARY)
    for line in lines:
        console.print(line)
    console.print(_JSON_OPTION_LINE)
    console.print(_HELP_OPTION_LINE)


def show_help() -> None:
    """Show top-level help information for the deepagents CLI."""
    editable_path = _get_editable_install_path()
    install_type = f" (local: {escape(editable_path)})" if editable_path else ""
    banner_color = theme.PRIMARY_DEV if _is_editable_install() else theme.PRIMARY
    console.print()
    console.print(
        f"[bold {banner_color}]deepagents-cli[/bold {banner_color}]"
        f" v{__version__}{install_type}"
    )
    console.print()
    console.print(
        f"Docs: [link={DOCS_URL}]{DOCS_URL}[/link]",
        style=theme.MUTED,
    )
    console.print()
    console.print("[bold]Usage:[/bold]", style=theme.PRIMARY)
    console.print(
        "  deepagents [OPTIONS]                           Start interactive thread"
    )
    console.print("  deepagents agents <list|reset>                 Manage agents")
    console.print(
        "  deepagents skills <list|create|info|delete>    Manage agent skills"
    )
    console.print(
        "  deepagents threads <list|delete>               Manage conversation threads"
    )
    console.print(
        "  deepagents mcp <login>                         Manage MCP servers"
    )
    console.print(
        "  deepagents update                              Check for and install updates"
    )
    console.print()
    console.print("[bold]Deploy (beta):[/bold]", style=theme.PRIMARY)
    console.print(
        "  deepagents init [NAME]                  Scaffold a new deploy project"
    )
    console.print(
        "  deepagents dev    --config deepagents.toml  Run a local dev server"
    )
    console.print("  deepagents deploy --config deepagents.toml  Bundle and deploy")
    console.print()

    console.print("[bold]Options:[/bold]", style=theme.PRIMARY)
    console.print(
        "  -r, --resume [ID]          Resume thread: -r for most recent, -r ID for specific"  # noqa: E501
    )
    console.print("  -a, --agent NAME           Agent to use (e.g., coder, researcher)")
    console.print("  -M, --model MODEL          Model to use (e.g., gpt-4o)")
    console.print(
        "  --model-params JSON        Extra model kwargs (e.g., '{\"temperature\": 0.7}')"  # noqa: E501
    )
    console.print("  --profile-override JSON    Override model profile fields as JSON")
    console.print("  -m, --message TEXT         Initial prompt to auto-submit on start")
    console.print("  --skill NAME              Invoke a skill when the session starts")
    console.print(
        "  -y, --auto-approve         Auto-approve all tool calls (toggle: Shift+Tab)"
    )
    console.print("  --sandbox TYPE             Remote sandbox for execution")
    console.print(
        "                             LangSmith is included;"
        " Agentcore/Modal/Daytona/Runloop"
        " require downloading extras"
    )
    console.print(
        "  --sandbox-id ID            Reuse existing sandbox (skips creation/cleanup)"
    )
    console.print(
        "  --sandbox-setup PATH       Setup script to run in sandbox after creation"
    )
    console.print(
        "  --mcp-config PATH          Load MCP tools from config file"
        " (merged on top of auto-discovered configs)"
    )
    console.print("  --no-mcp                   Disable all MCP tool loading")
    console.print(
        "  --trust-project-mcp        Trust project MCP configs (skip approval prompt)"
    )
    console.print("  -n, --non-interactive MSG  Run a single task and exit")
    console.print("  -q, --quiet                Clean output for piping (needs -n)")
    console.print(
        "  --no-stream                Buffer full response instead of streaming"
    )
    console.print("  --stdin                    Read input from stdin explicitly")
    console.print(
        "  --json                     Emit machine-readable JSON for commands"
    )
    console.print(
        "  -S, --shell-allow-list CMDS  Comma-separated cmds, 'recommended', or 'all'"
    )
    console.print("  --default-model [MODEL]    Set, show, or manage the default model")
    console.print("  --clear-default-model      Clear the default model")
    console.print(
        "  --update                   Check for and install updates, then exit"
    )
    console.print(
        "  --auto-update              Toggle automatic updates on or off, then exit"
    )
    console.print("  --acp                      Run as an ACP server over stdio")
    console.print("  -v, --version              Show deepagents CLI and SDK versions")
    console.print("  -h, --help                 Show this help message and exit")
    console.print()

    console.print("[bold]Non-Interactive Mode:[/bold]", style=theme.PRIMARY)
    console.print(
        "  deepagents -n 'Summarize README.md'     # Run task (no local shell access)",
        style=theme.MUTED,
    )
    console.print(
        "  deepagents -n 'List files' -S recommended  # Use safe commands",
        style=theme.MUTED,
    )
    console.print(
        "  deepagents -n 'Search logs' -S ls,cat,grep # Specify list",
        style=theme.MUTED,
    )
    console.print(
        "  deepagents -n 'Fix tests' -S all           # Any command",
        style=theme.MUTED,
    )
    console.print(
        "  cat prompt.txt | deepagents --stdin -q      # Explicit stdin",
        style=theme.MUTED,
    )
    console.print(
        "  deepagents --skill code-review -m 'review this patch'",
        style=theme.MUTED,
    )
    console.print()


def show_list_help() -> None:
    """Show help information for the `list` subcommand.

    Invoked via the `-h` argparse action or directly from `cli_main`.
    """
    console.print()
    console.print("[bold]Usage:[/bold]", style=theme.PRIMARY)
    console.print("  deepagents list [options]")
    console.print()
    console.print(
        "List all agents found in ~/.deepagents/. Each agent has its own",
    )
    console.print(
        "AGENTS.md system prompt and separate thread history.",
    )
    console.print()
    _print_option_section()
    console.print()
    console.print("[bold]Examples:[/bold]", style=theme.PRIMARY)
    console.print("  deepagents list")
    console.print("  deepagents list --json")
    console.print()


def show_agents_help() -> None:
    """Show help information for the `agents` subcommand."""
    console.print()
    console.print("[bold]Usage:[/bold]", style=theme.PRIMARY)
    console.print("  deepagents agents <command> [options]")
    console.print()
    console.print("[bold]Commands:[/bold]", style=theme.PRIMARY)
    console.print("  list|ls           List all agents")
    console.print("  reset             Reset an agent's prompt to default")
    console.print()
    _print_option_section()
    console.print()
    console.print("[bold]Examples:[/bold]", style=theme.PRIMARY)
    console.print("  deepagents agents list")
    console.print("  deepagents agents reset --agent coder")
    console.print("  deepagents agents reset --agent coder --target researcher")
    console.print()


def show_reset_help() -> None:
    """Show help information for the `reset` subcommand."""
    console.print()
    console.print("[bold]Usage:[/bold]", style=theme.PRIMARY)
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
    _print_option_section(
        "  --agent NAME            Agent to reset (required)",
        "  --target SRC            Copy AGENTS.md from another agent instead",
        "  --dry-run               Show what would happen without making changes",
    )
    console.print()
    console.print("[bold]Examples:[/bold]", style=theme.PRIMARY)
    console.print("  deepagents reset --agent coder")
    console.print("  deepagents reset --agent coder --target researcher")
    console.print("  deepagents reset --agent coder --dry-run")
    console.print()


def show_skills_help() -> None:
    """Show help information for the `skills` subcommand.

    Invoked via the `-h` argparse action or directly from
    `execute_skills_command` when no subcommand is given.
    """
    console.print()
    console.print("[bold]Usage:[/bold]", style=theme.PRIMARY)
    console.print("  deepagents skills <command> [options]")
    console.print()
    console.print("[bold]Commands:[/bold]", style=theme.PRIMARY)
    console.print("  list|ls           List all available skills")
    console.print("  create <name>     Create a new skill")
    console.print("  info <name>       Show detailed information about a skill")
    console.print("  delete <name>     Delete a skill")
    console.print()
    _print_option_section(
        "  --agent <name>    Specify agent identifier (default: agent)",
        "  --project         Use project-level skills instead of user-level",
        title="Common options",
    )
    console.print()
    console.print("[bold]Examples:[/bold]", style=theme.PRIMARY)
    console.print("  deepagents skills list")
    console.print("  deepagents skills list --project")
    console.print("  deepagents skills create my-skill")
    console.print("  deepagents skills create my-skill --agent myagent")
    console.print("  deepagents skills info my-skill")
    console.print("  deepagents skills delete my-skill")
    console.print("  deepagents skills delete my-skill --force --project")
    console.print("  deepagents skills delete -h")
    console.print()
    console.print(
        "[bold]Skill directories (highest precedence first):[/bold]",
        style=theme.PRIMARY,
    )
    console.print(
        "  1. .agents/skills/                 project skills\n"
        "  2. .deepagents/skills/             project skills (alias)\n"
        "  3. ~/.agents/skills/               user skills\n"
        "  4. ~/.deepagents/<agent>/skills/   user skills (alias)\n"
        "  5. <package>/built_in_skills/      built-in skills",
    )
    console.print()


def show_skills_list_help() -> None:
    """Show help information for the `skills list` subcommand."""
    console.print()
    console.print("[bold]Usage:[/bold]", style=theme.PRIMARY)
    console.print("  deepagents skills list [options]")
    console.print()
    _print_option_section(
        "  --agent NAME            Agent identifier (default: agent)",
        "  --project               Show only project-level skills",
    )
    console.print()
    console.print("[bold]Examples:[/bold]", style=theme.PRIMARY)
    console.print("  deepagents skills list")
    console.print("  deepagents skills list --project")
    console.print("  deepagents skills list --json")
    console.print()


def show_skills_create_help() -> None:
    """Show help information for the `skills create` subcommand."""
    console.print()
    console.print("[bold]Usage:[/bold]", style=theme.PRIMARY)
    console.print("  deepagents skills create <name> [options]")
    console.print()
    _print_option_section(
        "  --agent NAME            Agent identifier (default: agent)",
        "  --project               Create in project directory "
        "instead of user directory",
    )
    console.print()
    console.print("[bold]Examples:[/bold]", style=theme.PRIMARY)
    console.print("  deepagents skills create web-research")
    console.print("  deepagents skills create my-skill --project")
    console.print()


def show_skills_info_help() -> None:
    """Show help information for the `skills info` subcommand."""
    console.print()
    console.print("[bold]Usage:[/bold]", style=theme.PRIMARY)
    console.print("  deepagents skills info <name> [options]")
    console.print()
    _print_option_section(
        "  --agent NAME            Agent identifier (default: agent)",
        "  --project               Search only in project skills",
    )
    console.print()
    console.print("[bold]Examples:[/bold]", style=theme.PRIMARY)
    console.print("  deepagents skills info web-research")
    console.print("  deepagents skills info my-skill --project")
    console.print()


def show_skills_delete_help() -> None:
    """Show help information for the `skills delete` subcommand."""
    console.print()
    console.print("[bold]Usage:[/bold]", style=theme.PRIMARY)
    console.print("  deepagents skills delete <name> [options]")
    console.print()
    _print_option_section(
        "  --agent NAME            Agent identifier (default: agent)",
        "  --project               Search only in project skills",
        "  -f, --force             Skip confirmation prompt",
        "  --dry-run               Show what would happen without making changes",
    )
    console.print()
    console.print("[bold]Examples:[/bold]", style=theme.PRIMARY)
    console.print("  deepagents skills delete old-skill")
    console.print("  deepagents skills delete old-skill --force")
    console.print("  deepagents skills delete old-skill --project")
    console.print("  deepagents skills delete old-skill --dry-run")
    console.print()


def show_update_help() -> None:
    """Show help information for the `update` subcommand."""
    console.print()
    console.print("[bold]Usage:[/bold]", style=theme.PRIMARY)
    console.print("  deepagents update [options]")
    console.print()
    console.print(
        "Check for and install CLI updates from PyPI.",
    )
    console.print()
    _print_option_section()
    console.print()
    console.print("[bold]Examples:[/bold]", style=theme.PRIMARY)
    console.print("  deepagents update")
    console.print("  deepagents update --json")
    console.print()


def show_mcp_help() -> None:
    """Show help information for the `mcp` subcommand.

    Invoked via the `-h` argparse action or directly from `cli_main`
    when no mcp subcommand is given.
    """
    console.print()
    console.print("[bold]Usage:[/bold]", style=theme.PRIMARY)
    console.print("  deepagents mcp <command> [options]")
    console.print()
    console.print("[bold]Commands:[/bold]", style=theme.PRIMARY)
    console.print("  login <server>    Run the OAuth login flow for an MCP server")
    console.print()
    _print_option_section()
    console.print()
    console.print("[bold]Examples:[/bold]", style=theme.PRIMARY)
    console.print("  deepagents mcp login notion")
    console.print("  deepagents mcp login linear --config ./mcp-config.json")
    console.print()


def show_mcp_login_help() -> None:
    """Show help information for the `mcp login` subcommand."""
    console.print()
    console.print("[bold]Usage:[/bold]", style=theme.PRIMARY)
    console.print("  deepagents mcp login <server> [--config PATH]")
    console.print()
    _print_option_section(
        "  --config PATH           Path to an MCP config JSON file"
        " (default: auto-discovered)",
    )
    console.print()
    console.print("[bold]Examples:[/bold]", style=theme.PRIMARY)
    console.print("  deepagents mcp login notion")
    console.print("  deepagents mcp login linear --config ./mcp-config.json")
    console.print()


def show_threads_help() -> None:
    """Show help information for the `threads` subcommand.

    Invoked via the `-h` argparse action or directly from `cli_main`
    when no threads subcommand is given.
    """
    console.print()
    console.print("[bold]Usage:[/bold]", style=theme.PRIMARY)
    console.print("  deepagents threads <command> [options]")
    console.print()
    console.print("[bold]Commands:[/bold]", style=theme.PRIMARY)
    console.print("  list|ls           List all threads")
    console.print("  delete <ID>       Delete a thread")
    console.print()
    _print_option_section()
    console.print()
    console.print("[bold]Examples:[/bold]", style=theme.PRIMARY)
    console.print("  deepagents threads list")
    console.print("  deepagents threads list -n 10")
    console.print("  deepagents threads list --agent mybot")
    console.print("  deepagents threads delete abc123")
    console.print()


def show_threads_delete_help() -> None:
    """Show help information for the `threads delete` subcommand."""
    console.print()
    console.print("[bold]Usage:[/bold]", style=theme.PRIMARY)
    console.print("  deepagents threads delete <ID> [options]")
    console.print()
    _print_option_section(
        "  --dry-run               Show what would happen without making changes",
    )
    console.print()
    console.print("[bold]Examples:[/bold]", style=theme.PRIMARY)
    console.print("  deepagents threads delete abc123")
    console.print("  deepagents threads delete abc123 --dry-run")
    console.print()


def show_threads_list_help() -> None:
    """Show help information for the `threads list` subcommand."""
    console.print()
    console.print("[bold]Usage:[/bold]", style=theme.PRIMARY)
    console.print("  deepagents threads list [options]")
    console.print()
    _print_option_section(
        "  --agent NAME              Filter by agent name",
        "  --branch TEXT             Filter by git branch name",
        "  --sort {created,updated}  Sort order (default: from config, or updated)",
        "  -n, --limit N             Maximum threads to display (default: 20)",
        "  -v, --verbose             Show all columns (branch, created, prompt)",
        "  -r, --relative/--no-relative"
        "  Show relative timestamps (default: from config)",
    )
    console.print()
    console.print("[bold]Examples:[/bold]", style=theme.PRIMARY)
    console.print("  deepagents threads list")
    console.print("  deepagents threads list -n 10")
    console.print("  deepagents threads list --agent mybot")
    console.print("  deepagents threads list --branch main -v")
    console.print("  deepagents threads list --sort created --limit 50")
    console.print("  deepagents threads list -r")
    console.print()
