"""CLI commands for MCP server management.

These commands are registered with the CLI via main.py:
- deepagents mcp list
- deepagents mcp connect <server_name>
- deepagents mcp reconnect <server_name>
- deepagents mcp tools <server_name>
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from deepagents_cli.config import COLORS, Settings, console
from deepagents_cli.mcp.manager import get_mcp_manager


def _list_mcp_servers(settings: Settings) -> None:
    """List all configured MCP servers from ~/.deepagents.json.

    Args:
        settings: Settings instance containing MCP server configurations.
    """
    if not settings.has_mcp_servers:
        console.print("[yellow]No MCP servers configured.[/yellow]")
        console.print(
            "[dim]Add MCP servers to ~/.deepagents.json to use them.[/dim]",
            style=COLORS["dim"],
        )
        console.print(
            "\n[dim]Example configuration:\n"
            '{\n  "mcpServers": {\n'
            '    "context7": {\n'
            '      "type": "stdio",\n'
            '      "command": "npx",\n'
            '      "args": [\n'
            '        "-y",\n'
            '        "@upstash/context7-mcp",\n'
            '        "--api-key",\n'
            '        "your-api-key"\n'
            '      ],\n'
            '      "env": {}\n'
            '    }\n'
            '  }\n'
            '}\n',
            style=COLORS["dim"],
        )
        return

    # Register configurations with manager
    manager = get_mcp_manager()
    for server_name, server_config in settings.mcp_servers.items():
        manager.register_config(server_name, server_config)

    # Get server status
    server_status = manager.list_servers()

    console.print("\n[bold]Configured MCP Servers:[/bold]\n", style=COLORS["primary"])
    for server_name, server_config in settings.mcp_servers.items():
        server_type = server_config.get("type", "unknown")
        command = server_config.get("command", "")
        args = server_config.get("args", [])
        status = server_status.get(server_name, {})
        running = status.get("running", False)

        status_icon = "[green]●[/green]" if running else "[dim]○[/dim]"
        status_text = "[green]running[/green]" if running else "[dim]stopped[/dim]"

        console.print(f"  {status_icon} [bold]{server_name}[/bold] ([dim]{server_type}[/dim]) - {status_text}", style=COLORS["primary"])
        console.print(f"    Command: {command} {' '.join(args)}", style=COLORS["dim"])
        if running and status.get("pid"):
            console.print(f"    PID: {status['pid']}", style=COLORS["dim"])
        if "env" in server_config and server_config["env"]:
            console.print(f"    Environment variables: {len(server_config['env'])} set", style=COLORS["dim"])
        console.print()


def _connect_mcp_server(server_name: str, settings: Settings) -> None:
    """Connect to a specific MCP server.

    Args:
        server_name: Name of the MCP server to connect to.
        settings: Settings instance containing MCP server configurations.
    """
    if server_name not in settings.mcp_servers:
        console.print(f"[bold red]Error:[/bold red] MCP server '{server_name}' not found.")
        console.print("\n[dim]Available MCP servers:[/dim]", style=COLORS["dim"])
        for name in settings.mcp_servers.keys():
            console.print(f"  - {name}", style=COLORS["dim"])
        return

    # Register configuration with manager
    manager = get_mcp_manager()
    manager.register_config(server_name, settings.mcp_servers[server_name])

    # Check if server is already running
    if manager.is_running(server_name):
        process_info = manager.get_process_info(server_name)
        console.print(f"[yellow]MCP server '{server_name}' is already running.[/yellow]")
        console.print(f"[dim]Process ID: {process_info['pid'] if process_info else 'unknown'}[/dim]")
        console.print("[dim]Use '/mcp reconnect {server_name}' to restart the server.[/dim]")
        return

    console.print(f"[green]Connecting to MCP server '{server_name}'...[/green]")
    try:
        process = manager.start_server(server_name)
        console.print(f"[green]✓ MCP server '{server_name}' started successfully.[/green]")
        console.print(f"[dim]Process ID: {process.pid}[/dim]")
        console.print("[dim]Note: Server is running in background. Use 'reconnect' to restart if needed.[/dim]")
    except ValueError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
    except RuntimeError as e:
        console.print(f"[bold red]Error:[/bold red] MCP server '{server_name}' failed to start: {e}")
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] Failed to start MCP server '{server_name}': {e}")


def _reconnect_mcp_server(server_name: str, settings: Settings) -> None:
    """Reconnect to a specific MCP server (restart if already running).

    Args:
        server_name: Name of the MCP server to reconnect to.
        settings: Settings instance containing MCP server configurations.
    """
    if server_name not in settings.mcp_servers:
        console.print(f"[bold red]Error:[/bold red] MCP server '{server_name}' not found.")
        console.print("\n[dim]Available MCP servers:[/dim]", style=COLORS["dim"])
        for name in settings.mcp_servers.keys():
            console.print(f"  - {name}", style=COLORS["dim"])
        return

    # Register configuration with manager
    manager = get_mcp_manager()
    manager.register_config(server_name, settings.mcp_servers[server_name])

    console.print(f"[green]Reconnecting to MCP server '{server_name}'...[/green]")
    try:
        process = manager.restart_server(server_name)
        console.print(f"[green]✓ MCP server '{server_name}' restarted successfully.[/green]")
        console.print(f"[dim]Process ID: {process.pid}[/dim]")
    except ValueError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
    except RuntimeError as e:
        console.print(f"[bold red]Error:[/bold red] MCP server '{server_name}' failed to restart: {e}")
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] Failed to restart MCP server '{server_name}': {e}")


def _list_mcp_tools(server_name: str, settings: Settings) -> None:
    """List tools available from a specific MCP server.

    Args:
        server_name: Name of the MCP server to list tools for.
        settings: Settings instance containing MCP server configurations.
    """
    if server_name not in settings.mcp_servers:
        console.print(f"[bold red]Error:[/bold red] MCP server '{server_name}' not found.")
        console.print("\n[dim]Available MCP servers:[/dim]", style=COLORS["dim"])
        for name in settings.mcp_servers.keys():
            console.print(f"  - {name}", style=COLORS["dim"])
        return

    # Register configuration with manager (for consistency)
    manager = get_mcp_manager()
    manager.register_config(server_name, settings.mcp_servers[server_name])

    # Check if server is running via our manager (informational only)
    if not manager.is_running(server_name):
        console.print(f"[yellow]Note:[/yellow] MCP server '{server_name}' is not running via deepagents manager.")
        console.print("[dim]Attempting to start server for tool listing...[/dim]")

    console.print(f"[green]Listing tools for MCP server '{server_name}'...[/green]")

    # First try langchain_mcp_adapters (preferred method)
    try:
        from langchain_mcp_adapters.client import StdioConnection, create_session

        console.print("[dim]Using langchain_mcp_adapters to connect to server...[/dim]")

        # Get server configuration
        server_config = settings.mcp_servers[server_name]
        command = server_config.get("command")
        args = server_config.get("args", [])
        env = server_config.get("env", {})

        if not command:
            console.print(f"[bold red]Error:[/bold red] No command specified for MCP server '{server_name}'.")
            return

        import asyncio
        import concurrent.futures

        async def list_tools_with_langchain():
            # Create stdio connection
            connection = StdioConnection(
                command=command,
                args=args,
                env=env,
            )

            # Create session and get tools
            async with create_session(connection) as session:
                # Get tools from the session
                tools = await session.get_tools()
                return tools

        # Handle async execution properly
        try:
            # Check if we're already in an event loop
            asyncio.get_running_loop()
            # We're in an async context, need to run in a thread to avoid nesting loops
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(lambda: asyncio.new_event_loop().run_until_complete(list_tools_with_langchain()))
                tools_result = future.result()
        except RuntimeError:
            # No running event loop, safe to create one
            tools_result = asyncio.new_event_loop().run_until_complete(list_tools_with_langchain())

        if not tools_result:
            console.print(f"[yellow]No tools found for MCP server '{server_name}'.[/yellow]")
            return

        console.print(f"\n[bold]Available tools ({len(tools_result)}):[/bold]\n", style=COLORS["primary"])
        for tool in tools_result:
            console.print(f"  • [bold]{tool.name}[/bold]", style=COLORS["primary"])
            if tool.description:
                console.print(f"    {tool.description}", style=COLORS["dim"])
            console.print()

        return  # Successfully listed tools, exit function

    except ImportError as e:
        # langchain_mcp_adapters not available, fall back to mcp library
        console.print(f"[dim]langchain_mcp_adapters import failed: {e}[/dim]")
        console.print("[dim]Falling back to mcp library...[/dim]")
    except Exception as e:
        console.print(f"[dim]langchain_mcp_adapters failed: {e}[/dim]")
        console.print("[dim]Falling back to mcp library...[/dim]")

    # Fall back to mcp library
    try:
        import mcp.client as mcp_client
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        console.print("[dim]MCP client library found. Attempting to connect to server...[/dim]")

        # Get server configuration
        server_config = settings.mcp_servers[server_name]
        command = server_config.get("command")
        args = server_config.get("args", [])
        env = server_config.get("env", {})

        if not command:
            console.print(f"[bold red]Error:[/bold red] No command specified for MCP server '{server_name}'.")
            return

        # Prepare server parameters
        server_params = StdioServerParameters(command=command, args=args, env=env)

        # Connect to server and list tools
        import asyncio
        import concurrent.futures

        async def list_tools():
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    # Initialize the session
                    await session.initialize()

                    # List available tools
                    tools = await session.list_tools()
                    return tools

        try:
            # Check if we're already in an event loop
            asyncio.get_running_loop()
            # We're in an async context, need to run in a thread to avoid nesting loops
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(lambda: asyncio.new_event_loop().run_until_complete(list_tools()))
                tools_result = future.result()
        except RuntimeError:
            # No running event loop, safe to create one
            tools_result = asyncio.new_event_loop().run_until_complete(list_tools())

        if not tools_result.tools:
            console.print(f"[yellow]No tools found for MCP server '{server_name}'.[/yellow]")
            return

        console.print(f"\n[bold]Available tools ({len(tools_result.tools)}):[/bold]\n", style=COLORS["primary"])
        for tool in tools_result.tools:
            console.print(f"  • [bold]{tool.name}[/bold]", style=COLORS["primary"])
            if tool.description:
                console.print(f"    {tool.description}", style=COLORS["dim"])
            if tool.inputSchema:
                console.print(f"    Input schema: {tool.inputSchema}", style=COLORS["dim"])
            console.print()

    except ImportError:
        console.print("[yellow]MCP client library not installed.[/yellow]")
        console.print(
            "[dim]To list MCP tools, install the mcp library:\n"
            "  pip install mcp\n"
            "Or use uv:\n"
            "  uv add mcp[/dim]"
        )
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] Failed to list tools: {e}")
        console.print("[dim]Make sure the MCP server is running and responding.[/dim]")


def _stop_mcp_server(server_name: str, settings: Settings) -> None:
    """Stop a running MCP server.

    Args:
        server_name: Name of the MCP server to stop.
        settings: Settings instance containing MCP server configurations.
    """
    if server_name not in settings.mcp_servers:
        console.print(f"[bold red]Error:[/bold red] MCP server '{server_name}' not found.")
        console.print("\n[dim]Available MCP servers:[/dim]", style=COLORS["dim"])
        for name in settings.mcp_servers.keys():
            console.print(f"  - {name}", style=COLORS["dim"])
        return

    # Register configuration with manager
    manager = get_mcp_manager()
    manager.register_config(server_name, settings.mcp_servers[server_name])

    # Check if server is running
    if not manager.is_running(server_name):
        console.print(f"[yellow]MCP server '{server_name}' is not running.[/yellow]")
        return

    console.print(f"[green]Stopping MCP server '{server_name}'...[/green]")
    try:
        stopped = manager.stop_server(server_name)
        if stopped:
            console.print(f"[green]✓ MCP server '{server_name}' stopped successfully.[/green]")
        else:
            console.print(f"[yellow]MCP server '{server_name}' was not running.[/yellow]")
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] Failed to stop MCP server '{server_name}': {e}")


def setup_mcp_parser(
    subparsers: Any,
) -> argparse.ArgumentParser:
    """Setup the MCP subcommand parser with all its subcommands."""
    mcp_parser = subparsers.add_parser(
        "mcp",
        help="Manage MCP servers",
        description="Manage Model Context Protocol (MCP) servers - list, connect, and view tools",
    )
    mcp_subparsers = mcp_parser.add_subparsers(dest="mcp_command", help="MCP command")

    # MCP list
    list_parser = mcp_subparsers.add_parser(
        "list", help="List all configured MCP servers", description="List all configured MCP servers from ~/.deepagents.json"
    )

    # MCP connect
    connect_parser = mcp_subparsers.add_parser(
        "connect",
        help="Connect to an MCP server",
        description="Connect to a specific MCP server",
    )
    connect_parser.add_argument("server_name", help="Name of the MCP server to connect to")

    # MCP reconnect
    reconnect_parser = mcp_subparsers.add_parser(
        "reconnect",
        help="Reconnect to an MCP server",
        description="Reconnect to a specific MCP server (restart if already running)",
    )
    reconnect_parser.add_argument("server_name", help="Name of the MCP server to reconnect to")

    # MCP tools
    tools_parser = mcp_subparsers.add_parser(
        "tools",
        help="List tools from an MCP server",
        description="List tools available from a specific MCP server",
    )
    tools_parser.add_argument("server_name", help="Name of the MCP server to list tools for")

    # MCP stop
    stop_parser = mcp_subparsers.add_parser(
        "stop",
        help="Stop a running MCP server",
        description="Stop a specific MCP server",
    )
    stop_parser.add_argument("server_name", help="Name of the MCP server to stop")

    return mcp_parser


def execute_mcp_command(args: argparse.Namespace) -> None:
    """Execute MCP subcommands based on parsed arguments.

    Args:
        args: Parsed command line arguments with mcp_command attribute
    """
    settings = Settings.from_environment()

    if args.mcp_command == "list":
        _list_mcp_servers(settings)
    elif args.mcp_command == "connect":
        _connect_mcp_server(args.server_name, settings)
    elif args.mcp_command == "reconnect":
        _reconnect_mcp_server(args.server_name, settings)
    elif args.mcp_command == "tools":
        _list_mcp_tools(args.server_name, settings)
    elif args.mcp_command == "stop":
        _stop_mcp_server(args.server_name, settings)
    else:
        # No subcommand provided, show help
        console.print("[yellow]Please specify an MCP subcommand: list, connect, reconnect, tools, or stop[/yellow]")
        console.print("\n[bold]Usage:[/bold]", style=COLORS["primary"])
        console.print("  deepagents mcp <command> [options]\n")
        console.print("[bold]Available commands:[/bold]", style=COLORS["primary"])
        console.print("  list                     List all configured MCP servers")
        console.print("  connect <server_name>    Connect to a specific MCP server")
        console.print("  reconnect <server_name>  Reconnect to a specific MCP server")
        console.print("  tools <server_name>      List tools from a specific MCP server")
        console.print("  stop <server_name>       Stop a specific MCP server")
        console.print("\n[bold]Examples:[/bold]", style=COLORS["primary"])
        console.print("  deepagents mcp list")
        console.print("  deepagents mcp connect context7")
        console.print("  deepagents mcp tools context7")
        console.print("  deepagents mcp stop context7")
        console.print("\n[dim]For more help on a specific command:[/dim]", style=COLORS["dim"])
        console.print("  deepagents mcp <command> --help", style=COLORS["dim"])


__all__ = [
    "execute_mcp_command",
    "setup_mcp_parser",
]