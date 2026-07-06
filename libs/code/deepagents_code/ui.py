"""Help screens and argparse utilities for the app.

This module is imported at app startup to wire `-h` actions into the
argparse tree.  It must stay lightweight — no SDK or langchain imports.
"""

import argparse

from rich.markup import escape

from deepagents_code import theme
from deepagents_code._version import DOCS_URL, __version__
from deepagents_code.config import (
    _get_editable_install_path,
    _is_editable_install,
    console,
)

_JSON_OPTION_LINE = "  --json                  Emit machine-readable JSON"
_HELP_OPTION_LINE = "  -h, --help              Show this help message"


def positive_int(value: str) -> int:
    """Argparse type for integer arguments that must be >= 1.

    Args:
        value: Raw argument string to parse.

    Returns:
        Parsed positive integer.

    Raises:
        argparse.ArgumentTypeError: If `value` is not an integer or is < 1.
    """
    try:
        parsed = int(value)
    except ValueError as exc:
        msg = f"invalid int value: {value!r}"
        raise argparse.ArgumentTypeError(msg) from exc
    if parsed < 1:
        msg = f"must be a positive integer (>= 1), got {parsed}"
        raise argparse.ArgumentTypeError(msg)
    return parsed


def non_negative_int(value: str) -> int:
    """Argparse type for integer arguments that must be >= 0.

    Args:
        value: Raw argument string to parse.

    Returns:
        Parsed non-negative integer.

    Raises:
        argparse.ArgumentTypeError: If `value` is not an integer or is < 0.
    """
    try:
        parsed = int(value)
    except ValueError as exc:
        msg = f"invalid int value: {value!r}"
        raise argparse.ArgumentTypeError(msg) from exc
    if parsed < 0:
        msg = f"must be a non-negative integer (>= 0), got {parsed}"
        raise argparse.ArgumentTypeError(msg)
    return parsed


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
    """Show top-level help information."""
    editable_path = _get_editable_install_path()
    install_type = f" (local: {escape(editable_path)})" if editable_path else ""
    banner_color = theme.PRIMARY_DEV if _is_editable_install() else theme.PRIMARY
    console.print()
    console.print(
        f"[bold {banner_color}]deepagents-code[/bold {banner_color}]"
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
        "  dcode [OPTIONS]                           Start interactive thread"
    )
    console.print("  dcode agents <list|reset>                 Manage agents")
    console.print("  dcode skills <list|create|info|delete>    Manage agent skills")
    console.print(
        "  dcode threads <list|delete>               Manage conversation threads"
    )
    console.print("  dcode mcp <login>                         Manage MCP servers")
    console.print("  dcode config <show|list|get|path>         Inspect configuration")
    console.print(
        "  dcode auth <list|set|remove|status|path>  Manage provider credentials"
    )
    console.print(
        "  dcode update                              Check for and install updates"
    )
    console.print(
        "  dcode doctor                              Print install diagnostics"
    )
    console.print(
        "  dcode tools <install>                     Manage managed tools (ripgrep)"
    )
    console.print()

    console.print("[bold]Options:[/bold]", style=theme.PRIMARY)
    console.print(
        "  -r, --resume [ID]          Resume thread: -r for most recent, -r ID for specific"  # noqa: E501
    )
    console.print("  -a, --agent NAME           Agent to use (e.g., coder, researcher)")
    console.print("  -M, --model MODEL          Model to use (e.g., gpt-5.5)")
    console.print(
        "  --model-params JSON        Extra model kwargs (e.g., '{\"temperature\": 0.7}')"  # noqa: E501
    )
    console.print(
        "  --max-retries N            Override max retries for transient model errors"
    )
    console.print("  --profile-override JSON    Override model profile fields as JSON")
    console.print("  -m, --message TEXT         Initial prompt to auto-submit on start")
    console.print("  --skill NAME               Invoke a skill when the session starts")
    console.print(
        "  --startup-cmd CMD          Shell command to run at startup, before first prompt"  # noqa: E501
    )
    console.print(
        "  -y, --auto-approve         Auto-approve all tool calls (toggle: Shift+Tab)"
    )
    console.print("  --sandbox TYPE             Remote sandbox for execution")
    console.print(
        "                             LangSmith is included;"
        " Agentcore/Modal/Daytona/Runloop/Vercel"
        " require downloading extras"
    )
    console.print("  --sandbox-id ID            Attach to existing sandbox")
    console.print("  --sandbox-snapshot-name NAME")
    console.print(
        "                             Snapshot (langsmith) or blueprint (runloop)"
        " name to use or create"
    )
    console.print(
        "  --sandbox-setup PATH       Setup script to run in sandbox after creation"
    )
    console.print(
        "  --mcp-config PATH          Load MCP tools from config file"
        " (merged on top of auto-discovered configs;"
        " run `dcode mcp config` to list discovery paths)"
    )
    console.print("  --no-mcp                   Disable all MCP tool loading")
    console.print(
        "  --trust-project-mcp        Trust project MCP configs (skip approval prompt)"
    )
    console.print(
        "  --interpreter, --no-interpreter"
        "  Enable or disable JS interpreter (`js_eval`) middleware"
    )
    console.print(
        "                             Enabled by default when not using a sandbox"
    )
    console.print(
        "  --interpreter-tools VALUE  PTC allowlist: 'safe', 'all', or comma-separated "
        "tool names (may include 'safe')"
    )
    console.print("  -n, --non-interactive MSG  Run a single task and exit")
    console.print("  -q, --quiet                Clean output for piping (needs -n)")
    console.print(
        "  --no-stream                Buffer full response instead of streaming"
    )
    console.print(
        "  --max-turns N              Max agentic turns before stopping (needs -n)"
    )
    console.print(
        "  --goal TEXT                Draft goal criteria; review, then run "
        "accepted goal"
    )
    console.print(
        "  --rubric TEXT|@PATH        Acceptance criteria to grade against; "
        "'@path' reads a file relative to cwd, '~' ok (needs -n)"
    )
    console.print(
        "  --rubric-model MODEL       Model the rubric grader uses "
        "(defaults to main model)"
    )
    console.print(
        "  --rubric-max-iterations N  Override grader iterations per rubric attempt"
    )
    console.print(
        "  --timeout SECONDS          Hard wall-clock limit; exits 124 on expiry"
        " (needs -n/stdin)"
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
        "  --prerelease               With --update, include alpha/beta/rc releases"
    )
    console.print(
        "  --auto-update              Toggle automatic updates on or off, then exit"
    )
    console.print(
        "  --install NAME             Install an optional extra (e.g. daytona)"
    )
    console.print(
        "  --package                  With --install, treat NAME as a package "
        "(uv --with), not an extra"
    )
    console.print("  --yes                      Skip --install confirmation prompts")
    console.print("  --acp                      Run as an ACP server over stdio")
    console.print("  -v, --version              Show dcode and SDK versions")
    console.print("  -h, --help                 Show this help message and exit")
    console.print()

    console.print("[bold]Non-Interactive Mode:[/bold]", style=theme.PRIMARY)
    console.print(
        "  dcode -n 'Summarize README.md'     # Run task (no local shell access)",
        style=theme.MUTED,
    )
    console.print(
        "  dcode -n 'List files' -S recommended  # Use safe commands",
        style=theme.MUTED,
    )
    console.print(
        "  dcode -n 'Search logs' -S ls,cat,grep # Specify list",
        style=theme.MUTED,
    )
    console.print(
        "  dcode -n 'Fix tests' -S all           # Any command",
        style=theme.MUTED,
    )
    console.print(
        "  cat prompt.txt | dcode --stdin -q           # Explicit stdin",
        style=theme.MUTED,
    )
    console.print(
        "  dcode --skill code-review -m 'review this patch'",
        style=theme.MUTED,
    )
    console.print()


def show_list_help() -> None:
    """Show help information for the `list` subcommand.

    Invoked via the `-h` argparse action or directly from `cli_main`.
    """
    console.print()
    console.print("[bold]Usage:[/bold]", style=theme.PRIMARY)
    console.print("  dcode list [options]")
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
    console.print("  dcode list")
    console.print("  dcode list --json")
    console.print()


def show_agents_help() -> None:
    """Show help information for the `agents` subcommand."""
    console.print()
    console.print("[bold]Usage:[/bold]", style=theme.PRIMARY)
    console.print("  dcode agents <command> [options]")
    console.print()
    console.print("[bold]Commands:[/bold]", style=theme.PRIMARY)
    console.print("  list|ls           List all agents")
    console.print("  reset             Reset an agent's prompt to default")
    console.print()
    _print_option_section()
    console.print()
    console.print("[bold]Examples:[/bold]", style=theme.PRIMARY)
    console.print("  dcode agents list")
    console.print("  dcode agents reset --agent coder")
    console.print("  dcode agents reset --agent coder --target researcher")
    console.print()


def show_reset_help() -> None:
    """Show help information for the `reset` subcommand."""
    console.print()
    console.print("[bold]Usage:[/bold]", style=theme.PRIMARY)
    console.print("  dcode reset --agent NAME [--target SRC]")
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
    console.print("  dcode reset --agent coder")
    console.print("  dcode reset --agent coder --target researcher")
    console.print("  dcode reset --agent coder --dry-run")
    console.print()


def show_skills_help() -> None:
    """Show help information for the `skills` subcommand.

    Invoked via the `-h` argparse action or directly from
    `execute_skills_command` when no subcommand is given.
    """
    console.print()
    console.print("[bold]Usage:[/bold]", style=theme.PRIMARY)
    console.print("  dcode skills <command> [options]")
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
    console.print("  dcode skills list")
    console.print("  dcode skills list --project")
    console.print("  dcode skills create my-skill")
    console.print("  dcode skills create my-skill --agent myagent")
    console.print("  dcode skills info my-skill")
    console.print("  dcode skills delete my-skill")
    console.print("  dcode skills delete my-skill --force --project")
    console.print("  dcode skills delete -h")
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
    console.print("  dcode skills list [options]")
    console.print()
    _print_option_section(
        "  --agent NAME            Agent identifier (default: agent)",
        "  --project               Show only project-level skills",
    )
    console.print()
    console.print("[bold]Examples:[/bold]", style=theme.PRIMARY)
    console.print("  dcode skills list")
    console.print("  dcode skills list --project")
    console.print("  dcode skills list --json")
    console.print()


def show_skills_create_help() -> None:
    """Show help information for the `skills create` subcommand."""
    console.print()
    console.print("[bold]Usage:[/bold]", style=theme.PRIMARY)
    console.print("  dcode skills create <name> [options]")
    console.print()
    _print_option_section(
        "  --agent NAME            Agent identifier (default: agent)",
        "  --project               Create in project directory "
        "instead of user directory",
    )
    console.print()
    console.print("[bold]Examples:[/bold]", style=theme.PRIMARY)
    console.print("  dcode skills create web-research")
    console.print("  dcode skills create my-skill --project")
    console.print()


def show_skills_info_help() -> None:
    """Show help information for the `skills info` subcommand."""
    console.print()
    console.print("[bold]Usage:[/bold]", style=theme.PRIMARY)
    console.print("  dcode skills info <name> [options]")
    console.print()
    _print_option_section(
        "  --agent NAME            Agent identifier (default: agent)",
        "  --project               Search only in project skills",
    )
    console.print()
    console.print("[bold]Examples:[/bold]", style=theme.PRIMARY)
    console.print("  dcode skills info web-research")
    console.print("  dcode skills info my-skill --project")
    console.print()


def show_skills_delete_help() -> None:
    """Show help information for the `skills delete` subcommand."""
    console.print()
    console.print("[bold]Usage:[/bold]", style=theme.PRIMARY)
    console.print("  dcode skills delete <name> [options]")
    console.print()
    _print_option_section(
        "  --agent NAME            Agent identifier (default: agent)",
        "  --project               Search only in project skills",
        "  -f, --force             Skip confirmation prompt",
        "  --dry-run               Show what would happen without making changes",
    )
    console.print()
    console.print("[bold]Examples:[/bold]", style=theme.PRIMARY)
    console.print("  dcode skills delete old-skill")
    console.print("  dcode skills delete old-skill --force")
    console.print("  dcode skills delete old-skill --project")
    console.print("  dcode skills delete old-skill --dry-run")
    console.print()


def show_update_help() -> None:
    """Show help information for the `update` subcommand."""
    console.print()
    console.print("[bold]Usage:[/bold]", style=theme.PRIMARY)
    console.print("  dcode update [options]")
    console.print()
    console.print(
        "Check for and install updates from PyPI.",
    )
    console.print()
    _print_option_section(
        "  --prerelease            Include alpha/beta/rc releases",
    )
    console.print()
    console.print("[bold]Examples:[/bold]", style=theme.PRIMARY)
    console.print("  dcode update")
    console.print("  dcode update --prerelease")
    console.print("  dcode update --json")
    console.print()


def show_doctor_help() -> None:
    """Show help information for the `doctor` subcommand."""
    console.print()
    console.print("[bold]Usage:[/bold]", style=theme.PRIMARY)
    console.print("  dcode doctor [options]", markup=False)
    console.print()
    console.print(
        "Print install health and diagnostics (versions, platform, install",
    )
    console.print(
        "method, update status, and config locations). Runs offline and is",
    )
    console.print(
        "safe to paste into a bug report.",
    )
    console.print()
    _print_option_section()
    console.print()
    console.print("[bold]Examples:[/bold]", style=theme.PRIMARY)
    console.print("  dcode doctor")
    console.print("  dcode doctor --json")
    console.print()
    console.print(
        "Tip: Run `dcode config show` or `dcode config get <key>` "
        "to drill into config details.",
        style=theme.MUTED,
        highlight=False,
    )
    console.print(
        "     Run `dcode --version` (or `dcode -v`) for dependency versions.",
        style=theme.MUTED,
        highlight=False,
    )
    console.print()


def show_tools_help() -> None:
    """Show help information for the `tools` subcommand."""
    console.print()
    console.print("[bold]Usage:[/bold]", style=theme.PRIMARY)
    console.print("  dcode tools <command> [options]")
    console.print()
    console.print("[bold]Commands:[/bold]", style=theme.PRIMARY)
    console.print("  install           Install or repair the managed ripgrep binary")
    console.print()
    _print_option_section()
    console.print()
    console.print("[bold]Examples:[/bold]", style=theme.PRIMARY)
    console.print("  dcode tools install")
    console.print("  dcode tools install --json")
    console.print()


def show_tools_install_help() -> None:
    """Show help information for the `tools install` subcommand."""
    console.print()
    console.print("[bold]Usage:[/bold]", style=theme.PRIMARY)
    console.print("  dcode tools install [options]")
    console.print()
    console.print(
        "Download the pinned, SHA-256-verified ripgrep binary into",
    )
    console.print(
        "~/.deepagents/bin (no sudo). Reuses a system `rg` already on PATH and",
    )
    console.print(
        "is also handy for repairing a missing or stale managed binary.",
    )
    console.print()
    _print_option_section()
    console.print()
    console.print("[bold]Examples:[/bold]", style=theme.PRIMARY)
    console.print("  dcode tools install")
    console.print("  dcode tools install --json")
    console.print()
    console.print(
        "Opt out with DEEPAGENTS_CODE_OFFLINE=1 or set",
        style=theme.MUTED,
        highlight=False,
    )
    console.print(
        "DEEPAGENTS_CODE_RIPGREP_INSTALLER=system to use your package manager.",
        style=theme.MUTED,
        highlight=False,
    )
    console.print()


def _print_mcp_discovery_paths() -> None:
    """Print the auto-discovered MCP config paths in precedence order."""
    from deepagents_code.mcp_tools import MCP_CONFIG_DISCOVERY_PATHS

    console.print(
        "[bold]Auto-discovered config paths (precedence order):[/bold]",
        style=theme.PRIMARY,
    )
    width = max(len(path) for path, _ in MCP_CONFIG_DISCOVERY_PATHS)
    for path, label in MCP_CONFIG_DISCOVERY_PATHS:
        console.print(f"  {path:<{width}}  ({label})")
    console.print(
        "  <project-root> = nearest ancestor with a `.git` entry, else CWD.",
        style=theme.MUTED,
    )


_MCP_CONFIG_FORMAT_EXAMPLE = """\
  {
    "mcpServers": {
      "notion": {
        "transport": "http",
        "url": "https://mcp.notion.com/mcp",
        "auth": "oauth"
      }
    }
  }"""


def show_mcp_help() -> None:
    """Show help information for the `mcp` subcommand."""
    console.print()
    console.print("[bold]Usage:[/bold]", style=theme.PRIMARY)
    console.print("  dcode mcp <command> [options]")
    console.print()
    console.print("[bold]Commands:[/bold]", style=theme.PRIMARY)
    console.print("  login <server>    Run the OAuth login flow for an MCP server")
    console.print("  config            Show MCP config discovery paths")
    console.print()
    _print_option_section()
    console.print()
    _print_mcp_discovery_paths()
    console.print()
    console.print(
        "  Pass --mcp-config <path> to any subcommand to bypass discovery.",
        style=theme.MUTED,
    )
    console.print()
    console.print("[bold]Examples:[/bold]", style=theme.PRIMARY)
    console.print("  dcode mcp config")
    console.print("  dcode mcp login notion")
    console.print("  dcode mcp login linear --mcp-config ./mcp-config.json")
    console.print()


def show_mcp_login_help() -> None:
    """Show help information for the `mcp login` subcommand."""
    console.print()
    console.print("[bold]Usage:[/bold]", style=theme.PRIMARY)
    console.print("  dcode mcp login <server> [--mcp-config PATH]")
    console.print()
    _print_option_section(
        "  --mcp-config PATH       Path to an MCP config JSON file "
        "(default: auto-discovered)",
    )
    console.print()
    _print_mcp_discovery_paths()
    console.print()
    console.print("[bold]Config format:[/bold]", style=theme.PRIMARY)
    console.print(_MCP_CONFIG_FORMAT_EXAMPLE, style=theme.MUTED)
    console.print()
    console.print("[bold]Examples:[/bold]", style=theme.PRIMARY)
    console.print("  dcode mcp login notion")
    console.print("  dcode mcp login linear --mcp-config ./mcp-config.json")
    console.print()


def show_mcp_config_help() -> None:
    """Show help information for the `mcp config` subcommand."""
    console.print()
    console.print("[bold]Usage:[/bold]", style=theme.PRIMARY)
    console.print("  dcode mcp config")
    console.print()
    console.print(
        "Print the MCP config discovery paths in precedence order, marking"
        " which files exist on disk.",
    )
    console.print()
    _print_mcp_discovery_paths()
    console.print()


def show_config_help() -> None:
    """Show help information for the `config` subcommand.

    Invoked via the `-h` argparse action, the startup fast-path, or
    `run_config_command` when no config subcommand is given. Kept import-light
    so it stays on the startup fast path.
    """
    console.print()
    console.print("[bold]Usage:[/bold]", style=theme.PRIMARY)
    console.print("  dcode config <command> [options]")
    console.print()
    console.print("[bold]Commands:[/bold]", style=theme.PRIMARY)
    console.print("  show|list|ls      Show effective values and their source")
    console.print("  get <key>         Show one option's value and source")
    console.print("  path              Show config file locations")
    console.print()
    _print_option_section(
        "  -v, --verbose, --all  Also show each option's description and how to set it",
    )
    console.print()
    console.print(
        "  Credentials are reported as set/not set only; values are never printed.",
        style=theme.MUTED,
    )
    console.print()
    console.print("[bold]Examples:[/bold]", style=theme.PRIMARY)
    console.print("  dcode config show")
    console.print("  dcode config show --verbose")
    console.print("  dcode config get interpreter.memory_limit_mb")
    console.print("  dcode config path")
    console.print()


def show_auth_help() -> None:
    """Show help information for the `auth` subcommand.

    Invoked via the `-h` argparse action, the startup fast-path, or
    `run_auth_command` when no auth subcommand is given. Kept import-light so
    it stays on the startup fast path.
    """
    console.print()
    console.print("[bold]Usage:[/bold]", style=theme.PRIMARY)
    console.print("  dcode auth <command> [options]")
    console.print()
    console.print("[bold]Commands:[/bold]", style=theme.PRIMARY)
    console.print("  list|ls               List providers and their status")
    console.print("  set <provider>        Store an API key (read from stdin)")
    console.print("  remove <provider>     Remove a stored credential (rm|delete)")
    console.print("  status <provider>     Show resolution source for one provider")
    console.print("  path                  Print the resolved auth.json path")
    console.print()
    console.print("[bold]Options:[/bold]", style=theme.PRIMARY)
    console.print("  --from-env VAR        With `set`, copy the key from env var VAR")
    console.print("  --project NAME        With `set langsmith`, set the trace project")
    console.print(
        "  --base-url URL        With `set`, pair an endpoint with the key "
        "(langsmith accepts us|eu)"
    )
    console.print("  -h, --help            Show this help message")
    console.print()
    console.print(
        "  Keys are read from stdin by default so they never land in shell"
        " history or argv. An interactive terminal is rejected; pipe the key"
        " or use --from-env.",
        style=theme.MUTED,
    )
    console.print()
    console.print("[bold]Examples:[/bold]", style=theme.PRIMARY)
    console.print("  dcode auth list")
    console.print("  echo $ANTHROPIC_API_KEY | dcode auth set anthropic")
    console.print("  dcode auth set openai --from-env OPENAI_API_KEY")
    console.print(
        "  echo $LANGSMITH_API_KEY | dcode auth set langsmith --project my-app"
    )
    console.print("  echo $LANGSMITH_API_KEY | dcode auth set langsmith --base-url eu")
    console.print("  dcode auth status anthropic")
    console.print("  dcode auth remove anthropic")
    console.print("  dcode auth path")
    console.print()


def show_threads_help() -> None:
    """Show help information for the `threads` subcommand.

    Invoked via the `-h` argparse action or directly from `cli_main`
    when no threads subcommand is given.
    """
    console.print()
    console.print("[bold]Usage:[/bold]", style=theme.PRIMARY)
    console.print("  dcode threads <command> [options]")
    console.print()
    console.print("[bold]Commands:[/bold]", style=theme.PRIMARY)
    console.print("  list|ls           List all threads")
    console.print("  delete <ID>       Delete a thread")
    console.print()
    _print_option_section()
    console.print()
    console.print("[bold]Examples:[/bold]", style=theme.PRIMARY)
    console.print("  dcode threads list")
    console.print("  dcode threads list -n 10")
    console.print("  dcode threads list --agent mybot")
    console.print("  dcode threads delete abc123")
    console.print()


def show_threads_delete_help() -> None:
    """Show help information for the `threads delete` subcommand."""
    console.print()
    console.print("[bold]Usage:[/bold]", style=theme.PRIMARY)
    console.print("  dcode threads delete <ID> [options]")
    console.print()
    _print_option_section(
        "  --dry-run               Show what would happen without making changes",
    )
    console.print()
    console.print("[bold]Examples:[/bold]", style=theme.PRIMARY)
    console.print("  dcode threads delete abc123")
    console.print("  dcode threads delete abc123 --dry-run")
    console.print()


def show_threads_list_help() -> None:
    """Show help information for the `threads list` subcommand."""
    console.print()
    console.print("[bold]Usage:[/bold]", style=theme.PRIMARY)
    console.print("  dcode threads list [options]")
    console.print()
    _print_option_section(
        "  --agent NAME              Filter by agent name",
        "  --branch TEXT             Filter by git branch name",
        "  --cwd [PATH]              Filter by working directory (no value = current)",
        "  --sort {created,updated}  Sort order (default: from config, or updated)",
        "  -n, --limit N             Maximum threads to display (default: 20)",
        "  -v, --verbose             Show all columns (branch, created, prompt)",
        "  -r, --relative/--no-relative"
        "  Show relative timestamps (default: from config)",
    )
    console.print()
    console.print("[bold]Examples:[/bold]", style=theme.PRIMARY)
    console.print("  dcode threads list")
    console.print("  dcode threads list -n 10")
    console.print("  dcode threads list --agent mybot")
    console.print("  dcode threads list --branch main -v")
    console.print("  dcode threads list --cwd")
    console.print("  dcode threads list --sort created --limit 50")
    console.print("  dcode threads list -r")
    console.print()
