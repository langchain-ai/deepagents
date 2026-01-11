"""Main entry point and CLI loop for deepagents."""
# ruff: noqa: T201

import argparse
import asyncio
import contextlib
import os
import sys
from pathlib import Path

# Now safe to import agent (which imports LangChain modules)
from deepagents_cli.agent import create_cli_agent, list_agents, reset_agent
from deepagents_cli.cli_config import CLIConfig, create_example_config

# CRITICAL: Import config FIRST to set LANGSMITH_PROJECT before LangChain loads
from deepagents_cli.config import (
    console,
    create_model,
    settings,
)
from deepagents_cli.integrations.sandbox_factory import create_sandbox
from deepagents_cli.sessions import (
    delete_thread_command,
    generate_thread_id,
    get_checkpointer,
    get_most_recent,
    get_thread_agent,
    list_threads_command,
    thread_exists,
)
from deepagents_cli.skills import execute_skills_command, setup_skills_parser
from deepagents_cli.tools import fetch_url, http_request, web_search
from deepagents_cli.ui import show_help


def check_cli_dependencies() -> None:
    """Check if CLI optional dependencies are installed."""
    missing = []

    try:
        import requests  # noqa: F401
    except ImportError:
        missing.append("requests")

    try:
        import dotenv  # noqa: F401
    except ImportError:
        missing.append("python-dotenv")

    try:
        import tavily  # noqa: F401
    except ImportError:
        missing.append("tavily-python")

    try:
        import textual  # noqa: F401
    except ImportError:
        missing.append("textual")

    if missing:
        print("\n❌ Missing required CLI dependencies!")
        print("\nThe following packages are required to use the deepagents CLI:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nPlease install them with:")
        print("  pip install deepagents[cli]")
        print("\nOr install all dependencies:")
        print("  pip install 'deepagents[cli]'")
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="DeepAgents - AI Coding Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # List command
    subparsers.add_parser("list", help="List all available agents")

    # Help command
    subparsers.add_parser("help", help="Show help information")

    # Reset command
    reset_parser = subparsers.add_parser("reset", help="Reset an agent")
    reset_parser.add_argument("--agent", required=True, help="Name of agent to reset")
    reset_parser.add_argument(
        "--target", dest="source_agent", help="Copy prompt from another agent"
    )

    # Skills command - setup delegated to skills module
    setup_skills_parser(subparsers)
    
    # Config command
    config_parser = subparsers.add_parser("config", help="Manage CLI configuration")
    config_subparsers = config_parser.add_subparsers(dest="config_command")
    
    # config init
    init_parser = config_subparsers.add_parser("init", help="Create example config file")
    init_parser.add_argument(
        "--global",
        dest="is_global",
        action="store_true",
        help="Create global config (~/.deepagents/deepagentscli.json)",
    )
    init_parser.add_argument(
        "--mode",
        choices=["safe", "permissive", "strict", "custom"],
        default="safe",
        help="Shell approval mode (default: safe)",
    )
    
    # config show
    show_parser = config_subparsers.add_parser("show", help="Show current configuration")
    show_parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )

    # Threads command
    threads_parser = subparsers.add_parser("threads", help="Manage conversation threads")
    threads_sub = threads_parser.add_subparsers(dest="threads_command")

    # threads list
    threads_list = threads_sub.add_parser("list", help="List threads")
    threads_list.add_argument(
        "--agent", default=None, help="Filter by agent name (default: show all)"
    )
    threads_list.add_argument("--limit", type=int, default=20, help="Max threads (default: 20)")

    # threads delete
    threads_delete = threads_sub.add_parser("delete", help="Delete a thread")
    threads_delete.add_argument("thread_id", help="Thread ID to delete")

    # Default interactive mode
    parser.add_argument(
        "--agent",
        default="agent",
        help="Agent identifier for separate memory stores (default: agent).",
    )

    # Thread resume argument - matches PR #638: -r for most recent, -r <ID> for specific
    parser.add_argument(
        "-r",
        "--resume",
        dest="resume_thread",
        nargs="?",
        const="__MOST_RECENT__",
        default=None,
        help="Resume thread: -r for most recent, -r <ID> for specific thread",
    )

    parser.add_argument(
        "--model",
        help="Model to use (e.g., claude-sonnet-4-5-20250929, gpt-5-mini). "
        "Provider is auto-detected from model name.",
    )
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Auto-approve tool usage without prompting (disables human-in-the-loop)",
    )
    parser.add_argument(
        "--sandbox",
        choices=["none", "modal", "daytona", "runloop"],
        default="none",
        help="Remote sandbox for code execution (default: none - local only)",
    )
    parser.add_argument(
        "--sandbox-id",
        help="Existing sandbox ID to reuse (skips creation and cleanup)",
    )
    parser.add_argument(
        "--sandbox-setup",
        help="Path to setup script to run in sandbox after creation",
    )
    return parser.parse_args()


async def run_textual_cli_async(
    assistant_id: str,
    *,
    auto_approve: bool = False,
    sandbox_type: str = "none",
    sandbox_id: str | None = None,
    model_name: str | None = None,
    thread_id: str | None = None,
    is_resumed: bool = False,
) -> None:
    """Run the Textual CLI interface (async version).

    Args:
        assistant_id: Agent identifier for memory storage
        auto_approve: Whether to auto-approve tool usage
        sandbox_type: Type of sandbox ("none", "modal", "runloop", "daytona")
        sandbox_id: Optional existing sandbox ID to reuse
        model_name: Optional model name to use
        thread_id: Thread ID to use (new or resumed)
        is_resumed: Whether this is a resumed session
    """
    from deepagents_cli.app import run_textual_app

    model = create_model(model_name)

    # Show thread info
    if is_resumed:
        console.print(f"[green]Resuming thread:[/green] {thread_id}")
    else:
        console.print(f"[dim]Thread: {thread_id}[/dim]")

    # Use async context manager for checkpointer
    async with get_checkpointer() as checkpointer:
        # Create agent with conditional tools
        tools = [http_request, fetch_url]
        if settings.has_tavily:
            tools.append(web_search)

        # Handle sandbox mode
        sandbox_backend = None
        sandbox_cm = None

        if sandbox_type != "none":
            try:
                # Create sandbox context manager but keep it open
                sandbox_cm = create_sandbox(sandbox_type, sandbox_id=sandbox_id)
                sandbox_backend = sandbox_cm.__enter__()
            except (ImportError, ValueError, RuntimeError, NotImplementedError) as e:
                console.print()
                console.print("[red]❌ Sandbox creation failed[/red]")
                console.print(f"[dim]{e}[/dim]")
                sys.exit(1)

        try:
            agent, composite_backend = create_cli_agent(
                model=model,
                assistant_id=assistant_id,
                tools=tools,
                sandbox=sandbox_backend,
                sandbox_type=sandbox_type if sandbox_type != "none" else None,
                auto_approve=auto_approve,
                checkpointer=checkpointer,
            )

            # Run Textual app
            await run_textual_app(
                agent=agent,
                assistant_id=assistant_id,
                backend=composite_backend,
                auto_approve=auto_approve,
                cwd=Path.cwd(),
                thread_id=thread_id,
            )
        except Exception as e:
            console.print(f"[red]❌ Failed to create agent: {e}[/red]")
            sys.exit(1)
        finally:
            # Clean up sandbox if we created one
            if sandbox_cm is not None:
                with contextlib.suppress(Exception):
                    sandbox_cm.__exit__(None, None, None)


def _handle_config_init(args: argparse.Namespace) -> None:
    """Handle 'deepagents config init' command."""
    if args.is_global:
        # Create global config
        config_path = Path.home() / ".deepagents" / "deepagentscli.json"
        location_desc = "global"
    else:
        # Create project config
        project_root = settings.project_root
        if project_root is None:
            console.print("[red]Not in a git project. Use --global for global config.[/red]")
            sys.exit(1)
        config_path = project_root / ".deepagentscli.json"
        location_desc = "project"
    
    if config_path.exists():
        console.print(f"[yellow]Config already exists: {config_path}[/yellow]")
        response = input("Overwrite? (y/N): ")
        if response.lower() != "y":
            console.print("Cancelled.")
            return
    
    try:
        create_example_config(config_path, mode=args.mode)
        console.print(f"[green]✓[/green] Created {location_desc} config: {config_path}")
        console.print(f"\nShell approval mode: [bold]{args.mode}[/bold]")
        console.print(f"\nEdit {config_path} to customize settings.")
    except Exception as e:
        console.print(f"[red]Failed to create config: {e}[/red]")
        sys.exit(1)


def _handle_config_show(args: argparse.Namespace) -> None:
    """Handle 'deepagents config show' command."""
    try:
        cli_config = CLIConfig.load(project_root=settings.project_root)
        
        if args.json:
            # Output as JSON
            import json
            output = {}
            if cli_config.shell_approval:
                output["shellApproval"] = {"mode": "custom"}  # Simplified
            print(json.dumps(output, indent=2))
        else:
            # Human-readable output
            console.print("\n[bold]DeepAgents CLI Configuration[/bold]\n")
            
            # Show config file locations
            console.print("[dim]Config file locations (in order of precedence):[/dim]")
            if settings.project_root:
                console.print(f"  1. {settings.project_root}/.deepagentscli.json")
                console.print(f"  2. {settings.project_root}/.deepagents/deepagentscli.json")
            console.print(f"  3. ~/.deepagents/deepagentscli.json")
            console.print()
            
            # Shell approval config
            console.print("[bold]Shell Approval:[/bold]")
            if cli_config.shell_approval:
                console.print("  [green]✓[/green] Configured")
                
                # Test some common commands
                console.print("\n  Example commands:")
                test_commands = [
                    "ls -la",
                    "git status",
                    "rm file.txt",
                    "rm -rf /",
                ]
                for cmd in test_commands:
                    action = cli_config.shell_approval.should_approve(cmd)
                    emoji = {"allow": "✅", "ask": "❓", "deny": "🚫"}[action]
                    console.print(f"    {emoji} {cmd:<20} -> {action}")
            else:
                console.print("  [yellow]Not configured[/yellow] (using default HITL)")
            
            console.print()
            console.print("[dim]Run 'deepagents config init' to create a config file[/dim]")
    except Exception as e:
        console.print(f"[red]Failed to load config: {e}[/red]")
        sys.exit(1)


def cli_main() -> None:
    """Entry point for console script."""
    # Fix for gRPC fork issue on macOS
    # https://github.com/grpc/grpc/issues/37642
    if sys.platform == "darwin":
        os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"

    # Note: LANGSMITH_PROJECT is already overridden in config.py (before LangChain imports)
    # This ensures agent traces → DEEPAGENTS_LANGSMITH_PROJECT
    # Shell commands → user's original LANGSMITH_PROJECT (via ShellMiddleware env)

    # Check dependencies first
    check_cli_dependencies()

    try:
        args = parse_args()

        if args.command == "help":
            show_help()
        elif args.command == "list":
            list_agents()
        elif args.command == "reset":
            reset_agent(args.agent, args.source_agent)
        elif args.command == "skills":
            execute_skills_command(args)
        elif args.command == "config":
            if args.config_command == "init":
                _handle_config_init(args)
            elif args.config_command == "show":
                _handle_config_show(args)
            else:
                console.print("[yellow]Usage: deepagents config <init|show>[/yellow]")
        elif args.command == "threads":
            if args.threads_command == "list":
                asyncio.run(
                    list_threads_command(
                        agent_name=getattr(args, "agent", None),
                        limit=getattr(args, "limit", 20),
                    )
                )
            elif args.threads_command == "delete":
                asyncio.run(delete_thread_command(args.thread_id))
            else:
                console.print("[yellow]Usage: deepagents threads <list|delete>[/yellow]")
        else:
            # Interactive mode - handle thread resume
            thread_id = None
            is_resumed = False

            if args.resume_thread == "__MOST_RECENT__":
                # -r (no ID): Get most recent thread
                # If --agent specified, filter by that agent; otherwise get most recent overall
                agent_filter = args.agent if args.agent != "agent" else None
                thread_id = asyncio.run(get_most_recent(agent_filter))
                if thread_id:
                    is_resumed = True
                    agent_name = asyncio.run(get_thread_agent(thread_id))
                    if agent_name:
                        args.agent = agent_name
                else:
                    msg = (
                        f"No previous thread for '{args.agent}'"
                        if agent_filter
                        else "No previous threads"
                    )
                    console.print(f"[yellow]{msg}, starting new.[/yellow]")

            elif args.resume_thread:
                # -r <ID>: Resume specific thread
                if asyncio.run(thread_exists(args.resume_thread)):
                    thread_id = args.resume_thread
                    is_resumed = True
                    if args.agent == "agent":
                        agent_name = asyncio.run(get_thread_agent(thread_id))
                        if agent_name:
                            args.agent = agent_name
                else:
                    console.print(f"[red]Thread '{args.resume_thread}' not found.[/red]")
                    console.print(
                        "[dim]Use 'deepagents threads list' to see available threads.[/dim]"
                    )
                    sys.exit(1)

            # Generate new thread ID if not resuming
            if thread_id is None:
                thread_id = generate_thread_id()

            # Run Textual CLI
            asyncio.run(
                run_textual_cli_async(
                    assistant_id=args.agent,
                    auto_approve=args.auto_approve,
                    sandbox_type=args.sandbox,
                    sandbox_id=args.sandbox_id,
                    model_name=getattr(args, "model", None),
                    thread_id=thread_id,
                    is_resumed=is_resumed,
                )
            )
    except KeyboardInterrupt:
        # Clean exit on Ctrl+C - suppress ugly traceback
        console.print("\n\n[yellow]Interrupted[/yellow]")
        sys.exit(0)


if __name__ == "__main__":
    cli_main()
