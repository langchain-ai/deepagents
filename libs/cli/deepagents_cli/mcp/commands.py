"""MCP server management commands.

Provides CLI commands for managing MCP server configurations,
similar to Claude Code's `claude mcp add/list/remove/get` commands.
"""

import json
import os
import re
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax

from deepagents_cli.config import settings

console = Console()


def _expand_env_vars(value: str) -> str:
    """Expand ${VAR} and ${VAR:-default} patterns in a string.
    
    Supports:
    - ${VAR} - Expands to the value of environment variable VAR
    - ${VAR:-default} - Expands to VAR if set, otherwise uses default
    
    Args:
        value: String potentially containing environment variable references
        
    Returns:
        String with environment variables expanded
    """
    pattern = r'\$\{([^}:]+)(?::-([^}]*))?\}'
    
    def replacer(match: re.Match) -> str:
        var_name = match.group(1)
        default = match.group(2)
        return os.environ.get(var_name, default or "")
    
    return re.sub(pattern, replacer, value)


def _expand_env_vars_in_config(config: dict[str, Any]) -> dict[str, Any]:
    """Recursively expand environment variables in a config dict.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        New dictionary with environment variables expanded
    """
    result = {}
    for key, value in config.items():
        if isinstance(value, str):
            result[key] = _expand_env_vars(value)
        elif isinstance(value, dict):
            result[key] = _expand_env_vars_in_config(value)
        elif isinstance(value, list):
            result[key] = [
                _expand_env_vars(item) if isinstance(item, str) else item
                for item in value
            ]
        else:
            result[key] = value
    return result


def _get_config_path(scope: str) -> Path:
    """Get the MCP configuration file path for a given scope.
    
    Args:
        scope: Configuration scope ("user" or "project")
        
    Returns:
        Path to the configuration file
        
    Raises:
        ValueError: If scope is invalid or project scope used outside a project
    """
    if scope == "user":
        return settings.get_user_mcp_config_path()
    elif scope == "project":
        path = settings.get_project_mcp_config_path()
        if path is None:
            raise ValueError(
                "Not in a project directory. Cannot use project scope.\n"
                "Use --scope user for user-level configuration."
            )
        return path
    else:
        raise ValueError(f"Invalid scope: {scope}. Must be 'user' or 'project'.")


def _load_config(path: Path) -> dict[str, Any]:
    """Load MCP configuration from a JSON file.
    
    Args:
        path: Path to the configuration file
        
    Returns:
        Configuration dictionary with mcpServers key
    """
    if not path.exists():
        return {"mcpServers": {}}
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)
            if "mcpServers" not in config:
                config["mcpServers"] = {}
            return config
    except json.JSONDecodeError as e:
        console.print(f"[red]Error:[/red] Invalid JSON in {path}: {e}")
        return {"mcpServers": {}}
    except Exception as e:
        console.print(f"[red]Error:[/red] Failed to read {path}: {e}")
        return {"mcpServers": {}}


def _save_config(path: Path, config: dict[str, Any]) -> bool:
    """Save MCP configuration to a JSON file.
    
    Args:
        path: Path to the configuration file
        config: Configuration dictionary to save
        
    Returns:
        True if save was successful, False otherwise
    """
    try:
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        console.print(f"[red]Error:[/red] Failed to save {path}: {e}")
        return False


def add_mcp_server(
    name: str,
    *,
    transport: str = "stdio",
    url: str | None = None,
    command: str | None = None,
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
    scope: str = "user",
) -> None:
    """Add an MCP server configuration.
    
    Args:
        name: Server name (identifier)
        transport: Transport type (stdio, sse, http, websocket)
        url: URL for http/sse/websocket transports
        command: Command for stdio transport
        args: Command arguments for stdio transport
        env: Environment variables
        headers: HTTP headers for http/sse transports
        scope: Configuration scope ("user" or "project")
    """
    try:
        config_path = _get_config_path(scope)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        return
    
    # Build server configuration
    server_config: dict[str, Any] = {"transport": transport}
    
    if transport == "stdio":
        if not command:
            console.print("[red]Error:[/red] --command is required for stdio transport")
            return
        server_config["command"] = command
        server_config["args"] = args or []
        if env:
            server_config["env"] = env
    elif transport in ("http", "sse", "streamable_http", "websocket"):
        if not url:
            console.print(f"[red]Error:[/red] --url is required for {transport} transport")
            return
        server_config["url"] = url
        if headers:
            server_config["headers"] = headers
    else:
        console.print(f"[red]Error:[/red] Unknown transport: {transport}")
        console.print("Supported transports: stdio, sse, http, websocket")
        return
    
    # Load existing config
    config = _load_config(config_path)
    
    # Check for existing server
    if name in config["mcpServers"]:
        console.print(f"[yellow]Warning:[/yellow] Server '{name}' already exists. Overwriting.")
    
    # Add/update server
    config["mcpServers"][name] = server_config
    
    # Save config
    if _save_config(config_path, config):
        console.print(f"[green]✓[/green] Added MCP server '{name}' to {scope} configuration")
        console.print(f"[dim]Config file: {config_path}[/dim]")


def add_mcp_server_json(name: str, json_config: str, *, scope: str = "user") -> None:
    """Add an MCP server from JSON configuration.
    
    Args:
        name: Server name (identifier)
        json_config: JSON string with server configuration
        scope: Configuration scope ("user" or "project")
    """
    try:
        config_path = _get_config_path(scope)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        return
    
    # Parse JSON
    try:
        server_config = json.loads(json_config)
    except json.JSONDecodeError as e:
        console.print(f"[red]Error:[/red] Invalid JSON: {e}")
        return
    
    if not isinstance(server_config, dict):
        console.print("[red]Error:[/red] JSON must be an object")
        return
    
    # Normalize type -> transport
    if "type" in server_config and "transport" not in server_config:
        type_val = server_config.pop("type")
        type_to_transport = {
            "stdio": "stdio",
            "sse": "sse",
            "http": "streamable_http",
            "streamable_http": "streamable_http",
            "streamable-http": "streamable_http",
            "websocket": "websocket",
        }
        server_config["transport"] = type_to_transport.get(type_val, type_val)
    
    # Load existing config
    config = _load_config(config_path)
    
    # Check for existing server
    if name in config["mcpServers"]:
        console.print(f"[yellow]Warning:[/yellow] Server '{name}' already exists. Overwriting.")
    
    # Add/update server
    config["mcpServers"][name] = server_config
    
    # Save config
    if _save_config(config_path, config):
        console.print(f"[green]✓[/green] Added MCP server '{name}' to {scope} configuration")
        console.print(f"[dim]Config file: {config_path}[/dim]")


def list_mcp_servers(*, scope: str | None = None, show_details: bool = False) -> None:
    """List all configured MCP servers.
    
    Args:
        scope: Optional scope filter ("user" or "project"). If None, shows all.
        show_details: If True, show full configuration details
    """
    servers_found = False
    
    # Collect servers from both scopes
    all_servers: dict[str, tuple[str, dict[str, Any]]] = {}
    
    # User config
    if scope is None or scope == "user":
        user_path = settings.get_user_mcp_config_path()
        if user_path.exists():
            user_config = _load_config(user_path)
            for name, server in user_config.get("mcpServers", {}).items():
                all_servers[name] = ("user", server)
    
    # Project config
    if scope is None or scope == "project":
        project_path = settings.get_project_mcp_config_path()
        if project_path and project_path.exists():
            project_config = _load_config(project_path)
            for name, server in project_config.get("mcpServers", {}).items():
                # Project overrides user
                all_servers[name] = ("project", server)
    
    if not all_servers:
        console.print("[dim]No MCP servers configured.[/dim]")
        console.print("\nAdd a server with:")
        console.print("  deepagents mcp add --transport http <name> <url>")
        console.print("  deepagents mcp add --transport stdio <name> --command <cmd> --args <args>")
        return
    
    # Create table
    table = Table(title="MCP Servers", show_header=True, header_style="bold cyan")
    table.add_column("Name", style="green")
    table.add_column("Transport", style="yellow")
    table.add_column("Target", style="blue")
    table.add_column("Scope", style="dim")
    
    for name, (server_scope, server) in sorted(all_servers.items()):
        transport = server.get("transport", "unknown")
        
        # Determine target based on transport
        if transport == "stdio":
            cmd = server.get("command", "")
            args = server.get("args", [])
            target = f"{cmd} {' '.join(args)}".strip()
        else:
            target = server.get("url", "")
        
        # Truncate long targets
        if len(target) > 50:
            target = target[:47] + "..."
        
        table.add_row(name, transport, target, server_scope)
    
    console.print(table)
    console.print(f"\n[dim]Total: {len(all_servers)} server(s)[/dim]")
    
    if show_details:
        console.print("\n[bold]Detailed Configuration:[/bold]")
        for name, (server_scope, server) in sorted(all_servers.items()):
            json_str = json.dumps(server, indent=2)
            syntax = Syntax(json_str, "json", theme="monokai")
            console.print(Panel(syntax, title=f"{name} ({server_scope})", border_style="dim"))


def get_mcp_server(name: str) -> None:
    """Get details for a specific MCP server.
    
    Args:
        name: Server name to look up
    """
    # Check both scopes
    server = None
    server_scope = None
    config_path = None
    
    # Check project first (higher priority)
    project_path = settings.get_project_mcp_config_path()
    if project_path and project_path.exists():
        project_config = _load_config(project_path)
        if name in project_config.get("mcpServers", {}):
            server = project_config["mcpServers"][name]
            server_scope = "project"
            config_path = project_path
    
    # Fall back to user config
    if server is None:
        user_path = settings.get_user_mcp_config_path()
        if user_path.exists():
            user_config = _load_config(user_path)
            if name in user_config.get("mcpServers", {}):
                server = user_config["mcpServers"][name]
                server_scope = "user"
                config_path = user_path
    
    if server is None:
        console.print(f"[red]Error:[/red] Server '{name}' not found")
        console.print("\nUse 'deepagents mcp list' to see available servers.")
        return
    
    # Display server details
    console.print(f"[bold green]{name}[/bold green] ({server_scope} scope)")
    console.print(f"[dim]Config file: {config_path}[/dim]\n")
    
    json_str = json.dumps(server, indent=2)
    syntax = Syntax(json_str, "json", theme="monokai")
    console.print(Panel(syntax, border_style="cyan"))


def remove_mcp_server(name: str, *, scope: str | None = None) -> None:
    """Remove an MCP server configuration.
    
    Args:
        name: Server name to remove
        scope: Optional scope to remove from. If None, removes from all scopes.
    """
    removed_from = []
    
    # Remove from user config
    if scope is None or scope == "user":
        user_path = settings.get_user_mcp_config_path()
        if user_path.exists():
            user_config = _load_config(user_path)
            if name in user_config.get("mcpServers", {}):
                del user_config["mcpServers"][name]
                if _save_config(user_path, user_config):
                    removed_from.append("user")
    
    # Remove from project config
    if scope is None or scope == "project":
        project_path = settings.get_project_mcp_config_path()
        if project_path and project_path.exists():
            project_config = _load_config(project_path)
            if name in project_config.get("mcpServers", {}):
                del project_config["mcpServers"][name]
                if _save_config(project_path, project_config):
                    removed_from.append("project")
    
    if removed_from:
        console.print(f"[green]✓[/green] Removed server '{name}' from: {', '.join(removed_from)}")
    else:
        console.print(f"[yellow]Warning:[/yellow] Server '{name}' not found")
        if scope:
            console.print(f"[dim]Searched in {scope} scope only.[/dim]")


def setup_mcp_parser(subparsers) -> None:
    """Set up the MCP subcommand parser.
    
    Args:
        subparsers: argparse subparsers object from main parser
    """
    mcp_parser = subparsers.add_parser(
        "mcp",
        help="Manage MCP (Model Context Protocol) servers"
    )
    mcp_sub = mcp_parser.add_subparsers(dest="mcp_command")
    
    # mcp add
    add_parser = mcp_sub.add_parser("add", help="Add an MCP server")
    add_parser.add_argument("name", help="Server name (identifier)")
    add_parser.add_argument("url", nargs="?", help="Server URL (for http/sse/websocket)")
    add_parser.add_argument(
        "--transport", "-t",
        choices=["stdio", "sse", "http", "websocket"],
        default="http",
        help="Transport type (default: http)"
    )
    add_parser.add_argument("--command", "-c", help="Command for stdio transport")
    add_parser.add_argument(
        "--args", "-a", 
        action="append",
        default=[],
        help="Command argument for stdio transport (can be repeated, e.g., -a arg1 -a arg2)"
    )
    add_parser.add_argument(
        "--env", "-e",
        action="append",
        help="Environment variable (KEY=VALUE format, can be repeated)"
    )
    add_parser.add_argument(
        "--header", "-H",
        action="append",
        help="HTTP header (Key: Value format, can be repeated)"
    )
    add_parser.add_argument(
        "--scope", "-s",
        choices=["user", "project"],
        default="user",
        help="Configuration scope (default: user)"
    )
    
    # mcp add-json
    add_json_parser = mcp_sub.add_parser("add-json", help="Add an MCP server from JSON")
    add_json_parser.add_argument("name", help="Server name (identifier)")
    add_json_parser.add_argument("json", help="JSON configuration string")
    add_json_parser.add_argument(
        "--scope", "-s",
        choices=["user", "project"],
        default="user",
        help="Configuration scope (default: user)"
    )
    
    # mcp list
    list_parser = mcp_sub.add_parser("list", help="List MCP servers")
    list_parser.add_argument(
        "--scope", "-s",
        choices=["user", "project"],
        help="Filter by scope"
    )
    list_parser.add_argument(
        "--details", "-d",
        action="store_true",
        help="Show full configuration details"
    )
    
    # mcp get
    get_parser = mcp_sub.add_parser("get", help="Get details for an MCP server")
    get_parser.add_argument("name", help="Server name")
    
    # mcp remove
    remove_parser = mcp_sub.add_parser("remove", help="Remove an MCP server")
    remove_parser.add_argument("name", help="Server name to remove")
    remove_parser.add_argument(
        "--scope", "-s",
        choices=["user", "project"],
        help="Remove from specific scope only"
    )


def execute_mcp_command(args) -> None:
    """Execute an MCP subcommand.
    
    Args:
        args: Parsed command line arguments
    """
    if args.mcp_command == "add":
        # Parse env and header arguments
        env_dict = None
        if args.env:
            env_dict = {}
            for item in args.env:
                if "=" in item:
                    key, value = item.split("=", 1)
                    env_dict[key] = value
                else:
                    console.print(f"[yellow]Warning:[/yellow] Invalid env format: {item}")
        
        headers_dict = None
        if args.header:
            headers_dict = {}
            for item in args.header:
                if ":" in item:
                    key, value = item.split(":", 1)
                    headers_dict[key.strip()] = value.strip()
                else:
                    console.print(f"[yellow]Warning:[/yellow] Invalid header format: {item}")
        
        add_mcp_server(
            args.name,
            transport=args.transport,
            url=args.url,
            command=args.command,
            args=args.args,
            env=env_dict,
            headers=headers_dict,
            scope=args.scope,
        )
    
    elif args.mcp_command == "add-json":
        add_mcp_server_json(args.name, args.json, scope=args.scope)
    
    elif args.mcp_command == "list":
        list_mcp_servers(
            scope=getattr(args, "scope", None),
            show_details=getattr(args, "details", False)
        )
    
    elif args.mcp_command == "get":
        get_mcp_server(args.name)
    
    elif args.mcp_command == "remove":
        remove_mcp_server(args.name, scope=getattr(args, "scope", None))
    
    else:
        console.print("[yellow]Usage:[/yellow] deepagents mcp <add|add-json|list|get|remove>")
        console.print("\nExamples:")
        console.print("  deepagents mcp add --transport http notion https://mcp.notion.com/mcp")
        console.print("  deepagents mcp add --transport stdio myserver -c python -a server.py")
        console.print("  deepagents mcp list")
        console.print("  deepagents mcp get notion")
        console.print("  deepagents mcp remove notion")
