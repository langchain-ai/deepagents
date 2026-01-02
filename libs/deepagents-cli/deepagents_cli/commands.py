"""Command handlers for slash commands and bash execution."""

import subprocess
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver

from .config import COLORS, DEEP_AGENTS_ASCII, console
from .ui import TokenTracker, show_interactive_help


def handle_command(command: str, agent, token_tracker: TokenTracker, session_state) -> str | bool:
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

    # Handle /mcp commands
    if cmd.startswith("mcp"):
        return _handle_mcp_command(cmd, session_state)

    console.print()
    console.print(f"[yellow]Unknown command: /{cmd}[/yellow]")
    console.print("[dim]Type /help for available commands.[/dim]")
    console.print()
    return True


def _interactive_mcp_menu(session_state) -> bool:
    """Display interactive menu for MCP server selection and actions."""
    from .config import Settings
    from .mcp.commands import _list_mcp_servers

    settings = Settings.from_environment()

    if not settings.has_mcp_servers:
        console.print("[yellow]No MCP servers configured.[/yellow]")
        console.print("[dim]Add MCP servers to ~/.deepagents.json to use them.[/dim]")
        return True

    # Display server list
    console.print()
    console.print("[bold]Select an MCP server:[/bold]", style="cyan")
    console.print()

    server_names = list(settings.mcp_servers.keys())
    for i, name in enumerate(server_names, 1):
        console.print(f"  {i}. {name}")

    console.print()
    console.print("  L. List all servers (detailed)")
    console.print("  Q. Back to main prompt")
    console.print()

    try:
        choice = input("Enter choice (number, L, or Q): ").strip().upper()
    except EOFError:
        return True

    if choice == "Q":
        return True
    elif choice == "L":
        _list_mcp_servers(settings)
        return True

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(server_names):
            selected_server = server_names[idx]
            # Show action menu for selected server
            return _mcp_server_action_menu(selected_server, settings, session_state)
        else:
            console.print("[red]Invalid selection.[/red]")
            return True
    except ValueError:
        console.print("[red]Invalid input.[/red]")
        return True


def _mcp_server_action_menu(server_name: str, settings, session_state) -> bool:
    """Display action menu for a specific MCP server."""
    from .mcp.commands import _connect_mcp_server, _reconnect_mcp_server, _list_mcp_tools, _stop_mcp_server

    console.print()
    console.print(f"[bold]MCP server: {server_name}[/bold]", style="cyan")
    console.print()


    console.print("Select action:")
    console.print()
    console.print("  1. Connect")
    console.print("  2. Reconnect")
    console.print("  3. Show tools")
    console.print("  4. Stop")
    console.print("  5. Back to server selection")
    console.print("  6. Back to main prompt")
    console.print()

    try:
        choice = input(f"Enter choice (1-{'6'}): ").strip()
    except EOFError:
        return True

    if choice == "1":
        _connect_mcp_server(server_name, settings)
        return True
    elif choice == "2":
        _reconnect_mcp_server(server_name, settings)
        return True
    elif choice == "3":
        _list_mcp_tools(server_name, settings)
        return True
    elif choice == "4":
        _stop_mcp_server(server_name, settings)
        return True
    elif choice == "5":
        return _interactive_mcp_menu(session_state)  # Recursive, but depth is limited
    elif choice == "6":
        return True
    else:
        console.print("[red]Invalid selection.[/red]")
        return True


def _handle_mcp_command(cmd: str, session_state) -> bool:
    """Handle /mcp commands. Returns True if handled."""
    from .config import Settings
    from .mcp.commands import _list_mcp_servers, _connect_mcp_server, _reconnect_mcp_server, _list_mcp_tools, _stop_mcp_server

    # Remove leading "mcp" and split into parts
    parts = cmd.split()
    if len(parts) < 2:
        # No subcommand provided, show interactive menu
        return _interactive_mcp_menu(session_state)

    subcommand = parts[1].lower()
    settings = Settings.from_environment()

    if subcommand == "list":
        _list_mcp_servers(settings)
        return True
    elif subcommand == "connect":
        if len(parts) < 3:
            console.print("[red]Error: Missing server name for connect command[/red]")
            console.print("[dim]Usage: /mcp connect <server_name>[/dim]")
            return True
        server_name = parts[2]
        _connect_mcp_server(server_name, settings)
        return True
    elif subcommand == "reconnect":
        if len(parts) < 3:
            console.print("[red]Error: Missing server name for reconnect command[/red]")
            console.print("[dim]Usage: /mcp reconnect <server_name>[/dim]")
            return True
        server_name = parts[2]
        _reconnect_mcp_server(server_name, settings)
        return True
    elif subcommand == "tools":
        if len(parts) < 3:
            console.print("[red]Error: Missing server name for tools command[/red]")
            console.print("[dim]Usage: /mcp tools <server_name>[/dim]")
            return True
        server_name = parts[2]
        _list_mcp_tools(server_name, settings)
        return True
    elif subcommand == "stop":
        if len(parts) < 3:
            console.print("[red]Error: Missing server name for stop command[/red]")
            console.print("[dim]Usage: /mcp stop <server_name>[/dim]")
            return True
        server_name = parts[2]
        _stop_mcp_server(server_name, settings)
        return True
    else:
        console.print(f"[red]Error: Unknown MCP subcommand '{subcommand}'[/red]")
        console.print("[dim]Available subcommands: list, connect, reconnect, tools, stop[/dim]")
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
