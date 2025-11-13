"""Dev server launcher for deepagents CLI."""

import argparse
import os
import sys
import tempfile
from pathlib import Path

from deepagents_cli.config import console


def add_dev_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add dev command parser to subparsers.

    Args:
        subparsers: The subparsers object from argparse
    """
    # Dev command - launches langgraph dev server
    dev_parser = subparsers.add_parser("dev", help="Launch a LangGraph dev server for the agent")
    # Existing deepagents options
    dev_parser.add_argument(
        "--agent",
        default="agent",
        help="Agent identifier for separate memory stores (default: agent).",
    )
    dev_parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Auto-approve tool usage without prompting (disables human-in-the-loop)",
    )
    dev_parser.add_argument(
        "--sandbox",
        choices=["none", "modal", "daytona", "runloop"],
        default="none",
        help="Remote sandbox for code execution (default: none - local only)",
    )
    dev_parser.add_argument(
        "--sandbox-id",
        help="Existing sandbox ID to reuse (skips creation and cleanup)",
    )
    dev_parser.add_argument(
        "--sandbox-setup",
        help="Path to setup script to run in sandbox after creation",
    )
    # LangGraph dev server options
    dev_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Network interface to bind the development server to (default: 127.0.0.1)",
    )
    dev_parser.add_argument(
        "--port",
        default=2024,
        type=int,
        help="Port number to bind the development server to (default: 2024)",
    )
    dev_parser.add_argument(
        "--no-reload",
        action="store_true",
        help="Disable automatic reloading when code changes are detected",
    )
    dev_parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Skip automatically opening the browser when the server starts",
    )
    dev_parser.add_argument(
        "--tunnel",
        action="store_true",
        help="Expose the local server via a public tunnel for remote access",
    )
    dev_parser.add_argument(
        "--debug-port",
        default=None,
        type=int,
        help="Enable remote debugging by listening on specified port",
    )
    dev_parser.add_argument(
        "--wait-for-client",
        action="store_true",
        default=False,
        help="Wait for a debugger client to connect before starting the server",
    )
    dev_parser.add_argument(
        "--studio-url",
        type=str,
        default=None,
        help="URL of the LangGraph Studio instance to connect to",
    )


def create_server_agent(
    agent_name: str,
    auto_approve: bool = False,
    sandbox_type: str = "none",
    sandbox_id: str | None = None,
    sandbox_setup: str | None = None,
):
    """Create and configure an agent for the dev server.

    This function is called by the generated module at import time.

    Args:
        agent_name: Agent identifier for memory storage
        auto_approve: Whether to auto-approve tool usage (disable interrupts)
        sandbox_type: Type of sandbox ("none", "modal", "runloop", "daytona")
        sandbox_id: Optional existing sandbox ID to reuse
        sandbox_setup: Optional path to setup script for sandbox

    Returns:
        Configured agent (Pregel graph)
    """
    # Setup tools
    from deepagents_cli.tools import fetch_url, http_request, tavily_client, web_search

    tools = [http_request, fetch_url]
    if tavily_client is not None:
        tools.append(web_search)

    # Setup sandbox if needed
    sandbox_backend = None
    sandbox_type_str = None
    if sandbox_type != "none":
        from deepagents_cli.integrations.sandbox_factory import create_sandbox

        try:
            sandbox_context = create_sandbox(
                sandbox_type,
                sandbox_id=sandbox_id,
                setup_script_path=sandbox_setup,
            )
            sandbox_backend = sandbox_context.__enter__()
            sandbox_type_str = sandbox_type
        except Exception as e:
            raise RuntimeError(f"Failed to create sandbox: {e}") from e

    # Create the agent
    from deepagents_cli.agent import create_agent_with_config
    from deepagents_cli.config import create_model

    model = create_model()

    agent, backend = create_agent_with_config(
        model=model,
        assistant_id=agent_name,
        tools=tools,
        sandbox=sandbox_backend,
        sandbox_type=sandbox_type_str,
    )

    # Handle auto-approve by removing interrupts
    if auto_approve:
        # Recreate agent without interrupts
        from deepagents import create_deep_agent
        from deepagents.backends.filesystem import FilesystemBackend

        from deepagents_cli.agent import get_system_prompt
        from deepagents_cli.agent_memory import AgentMemoryMiddleware
        from deepagents_cli.config import config

        system_prompt = get_system_prompt(sandbox_type=sandbox_type_str)

        # Build middleware based on sandbox type
        agent_dir = Path.home() / ".deepagents" / agent_name
        long_term_backend = FilesystemBackend(root_dir=agent_dir, virtual_mode=True)

        if sandbox_backend is None:
            from deepagents.middleware.resumable_shell import ResumableShellToolMiddleware
            from langchain.agents.middleware import HostExecutionPolicy

            agent_middleware = [
                AgentMemoryMiddleware(backend=long_term_backend, memory_path="/memories/"),
                ResumableShellToolMiddleware(
                    workspace_root=os.getcwd(), execution_policy=HostExecutionPolicy()
                ),
            ]
        else:
            agent_middleware = [
                AgentMemoryMiddleware(backend=long_term_backend, memory_path="/memories/"),
            ]

        agent = create_deep_agent(
            model=model,
            system_prompt=system_prompt,
            tools=tools,
            backend=backend,
            middleware=agent_middleware,
            interrupt_on={},  # No interrupts
        ).with_config(config)

        from langgraph.checkpoint.memory import InMemorySaver

        agent.checkpointer = InMemorySaver()

    return agent


def run_dev_server(args) -> None:
    """Run the LangGraph dev server with deepagents agent.

    Args:
        args: Parsed command line arguments
    """
    console.print("\n[bold cyan]🚀 Starting DeepAgents Dev Server[/bold cyan]\n")

    # Validate dependencies
    try:
        from langgraph_api.cli import run_server
    except ImportError:
        console.print("[red]❌ langgraph-api not installed[/red]")
        console.print("\nThe dev server requires langgraph-api to be installed.")
        console.print("Install with:")
        console.print("  pip install 'langgraph-cli[inmem]'")
        console.print("\nOr install all deepagents extras:")
        console.print("  pip install 'deepagents[cli,server]'")
        sys.exit(1)

    # Create a temporary directory for the generated module
    temp_dir = Path(tempfile.mkdtemp(prefix="deepagents_dev_"))
    console.print(f"[dim]Working directory: {temp_dir}[/dim]")

    try:
        # Generate minimal module that calls create_server_agent
        console.print(f"[cyan]Generating agent module for '{args.agent}'...[/cyan]")

        module_code = f'''"""Auto-generated agent graph for LangGraph dev server.

This module is generated by deepagents CLI dev command.
DO NOT EDIT MANUALLY - changes will be overwritten.
"""

from deepagents_cli.dev_server import create_server_agent

graph = create_server_agent(
    agent_name={args.agent!r},
    auto_approve={args.auto_approve!r},
    sandbox_type={args.sandbox!r},
    sandbox_id={args.sandbox_id!r},
    sandbox_setup={args.sandbox_setup!r},
)
'''

        # Write the module to a file
        module_path = temp_dir / "agent_graph.py"
        module_path.write_text(module_code)
        console.print(f"[green]✓[/green] Generated: {module_path}")

        # Point langgraph to the generated module
        import json

        langserve_graphs = {args.agent: str(module_path) + ":graph"}

        # Set LANGSERVE_GRAPHS environment variable
        os.environ["LANGSERVE_GRAPHS"] = json.dumps(langserve_graphs)

        console.print("\n[bold green]✓ Agent configured successfully[/bold green]")
        console.print(f"[dim]Agent: {args.agent}[/dim]")
        console.print(f"[dim]Auto-approve: {args.auto_approve}[/dim]")
        console.print(f"[dim]Sandbox: {args.sandbox}[/dim]")
        if args.sandbox_id:
            console.print(f"[dim]Sandbox ID: {args.sandbox_id}[/dim]")

        console.print("\n[bold cyan]Starting LangGraph server...[/bold cyan]")
        console.print(f"[dim]Host: {args.host}[/dim]")
        console.print(f"[dim]Port: {args.port}[/dim]")
        if args.tunnel:
            console.print("[yellow]Tunnel: enabled[/yellow]")
        console.print()

        # Call langgraph dev server
        run_server(
            host=args.host,
            port=args.port,
            reload=not args.no_reload,
            graphs=langserve_graphs,
            n_jobs_per_worker=None,
            open_browser=not args.no_browser,
            debug_port=args.debug_port,
            env=None,
            store=None,
            wait_for_client=args.wait_for_client,
            auth=None,
            http=None,
            ui=None,
            ui_config=None,
            studio_url=args.studio_url,
            allow_blocking=False,
            tunnel=args.tunnel,
            server_level="WARNING",
        )

    except Exception as e:
        console.print("\n[bold red]❌ Error starting dev server:[/bold red]")
        console.print(f"[red]{e}[/red]")
        console.print_exception()
        sys.exit(1)
    finally:
        # Cleanup temp directory
        import shutil

        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass  # Best effort cleanup
