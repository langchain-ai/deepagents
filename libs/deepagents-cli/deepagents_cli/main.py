"""Main entry point and CLI loop for deepagents."""
# ruff: noqa: T201

import argparse
import contextlib
import os
import sys
from pathlib import Path

# Now safe to import agent (which imports LangChain modules)
from deepagents_cli.agent import create_cli_agent, list_agents, reset_agent

# CRITICAL: Import config FIRST to set LANGSMITH_PROJECT before LangChain loads
from deepagents_cli.config import (
    console,
    create_model,
    settings,
)
from deepagents_cli.integrations.sandbox_factory import create_sandbox
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

    # Default interactive mode
    parser.add_argument(
        "--agent",
        default="agent",
        help="Agent identifier for separate memory stores (default: agent).",
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


def run_textual_cli(
    assistant_id: str,
    *,
    auto_approve: bool = False,
    sandbox_type: str = "none",
    sandbox_id: str | None = None,
    model_name: str | None = None,
) -> None:
    """Run the Textual CLI interface.

    Args:
        assistant_id: Agent identifier for memory storage
        auto_approve: Whether to auto-approve tool usage
        sandbox_type: Type of sandbox ("none", "modal", "runloop", "daytona")
        sandbox_id: Optional existing sandbox ID to reuse
        model_name: Optional model name to use
    """
    from deepagents_cli.app import run_textual_app

    model = create_model(model_name)

    # Create agent with conditional tools
    tools = [http_request, fetch_url]
    if settings.has_tavily:
        tools.append(web_search)

    # Handle sandbox mode
    sandbox_backend = None
    composite_backend = None

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
        )
    except Exception as e:
        console.print(f"[red]❌ Failed to create agent: {e}[/red]")
        sys.exit(1)

    # Run Textual app
    try:
        run_textual_app(
            agent=agent,
            assistant_id=assistant_id,
            backend=composite_backend,
            auto_approve=auto_approve,
            cwd=Path.cwd(),
        )
    finally:
        # Clean up sandbox if we created one
        if sandbox_type != "none" and sandbox_backend:
            with contextlib.suppress(Exception):
                sandbox_cm.__exit__(None, None, None)


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
        else:
            # Run Textual CLI
            run_textual_cli(
                assistant_id=args.agent,
                auto_approve=args.auto_approve,
                sandbox_type=args.sandbox,
                sandbox_id=args.sandbox_id,
                model_name=getattr(args, "model", None),
            )
    except KeyboardInterrupt:
        # Clean exit on Ctrl+C - suppress ugly traceback
        console.print("\n\n[yellow]Interrupted[/yellow]")
        sys.exit(0)


if __name__ == "__main__":
    cli_main()
