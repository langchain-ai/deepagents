"""Main entry point and CLI loop for deepagents."""
import sys
import argparse
import asyncio
from pathlib import Path

from .config import console, COLORS, DEEP_AGENTS_ASCII, create_model, SessionState
from .tools import tavily_client, http_request, web_search
from .ui import show_help, TokenTracker
from .input import create_prompt_session
from .execution import execute_task
from .commands import handle_command, execute_bash_command
from .agent import list_agents, reset_agent, create_agent_with_config


def check_cli_dependencies():
    """Check if CLI optional dependencies are installed."""
    missing = []

    try:
        import rich
    except ImportError:
        missing.append("rich")

    try:
        import requests
    except ImportError:
        missing.append("requests")

    try:
        import dotenv
    except ImportError:
        missing.append("python-dotenv")

    try:
        import tavily
    except ImportError:
        missing.append("tavily-python")

    try:
        import prompt_toolkit
    except ImportError:
        missing.append("prompt-toolkit")

    if missing:
        print("\n❌ Missing required CLI dependencies!")
        print(f"\nThe following packages are required to use the deepagents CLI:")
        for pkg in missing:
            print(f"  - {pkg}")
        print(f"\nPlease install them with:")
        print(f"  pip install deepagents[cli]")
        print(f"\nOr install all dependencies:")
        print(f"  pip install 'deepagents[cli]'")
        sys.exit(1)


def parse_args():
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
    reset_parser.add_argument("--target", dest="source_agent", help="Copy prompt from another agent")

    # Default interactive mode
    parser.add_argument(
        "--agent",
        default="agent",
        help="Agent identifier for separate memory stores (default: agent).",
    )
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Auto-approve tool usage without prompting (disables human-in-the-loop)",
    )

    return parser.parse_args()


async def simple_cli(agent, assistant_id: str | None, session_state):
    """Main CLI loop."""
    console.clear()
    console.print(DEEP_AGENTS_ASCII, style=f"bold {COLORS['primary']}")
    console.print()

    if tavily_client is None:
        console.print(f"[yellow]⚠ Web search disabled:[/yellow] TAVILY_API_KEY not found.", style=COLORS["dim"])
        console.print(f"  To enable web search, set your Tavily API key:", style=COLORS["dim"])
        console.print(f"    export TAVILY_API_KEY=your_api_key_here", style=COLORS["dim"])
        console.print(f"  Or add it to your .env file. Get your key at: https://tavily.com", style=COLORS["dim"])
        console.print()

    console.print("... Ready to code! What would you like to build?", style=COLORS["agent"])
    console.print(f"  [dim]Working directory: {Path.cwd()}[/dim]")
    console.print()

    if session_state.auto_approve:
        console.print(
            "  [yellow]⚡ Auto-approve: ON[/yellow] [dim](tools run without confirmation)[/dim]"
        )
        console.print()

    console.print(f"  Tips: Enter to submit, Alt+Enter for newline, Ctrl+E for editor, Ctrl+T to toggle auto-approve, Ctrl+C to interrupt", style=f"dim {COLORS['dim']}")
    console.print()

    # Create prompt session and token tracker
    session = create_prompt_session(assistant_id, session_state)
    token_tracker = TokenTracker()

    while True:
        try:
            user_input = await session.prompt_async()
            user_input = user_input.strip()
        except EOFError:
            break
        except KeyboardInterrupt:
            # Ctrl+C at prompt - exit the program
            console.print(f"\n\nGoodbye!", style=COLORS["primary"])
            break

        if not user_input:
            continue

        # Check for slash commands first
        if user_input.startswith('/'):
            result = handle_command(user_input, agent, token_tracker)
            if result == 'exit':
                console.print(f"\nGoodbye!", style=COLORS["primary"])
                break
            elif result:
                # Command was handled, continue to next input
                continue

        # Check for bash commands (!)
        if user_input.startswith('!'):
            execute_bash_command(user_input)
            continue

        # Handle regular quit keywords
        if user_input.lower() in ["quit", "exit", "q"]:
            console.print(f"\nGoodbye!", style=COLORS["primary"])
            break

        execute_task(user_input, agent, assistant_id, session_state, token_tracker)


async def main(assistant_id: str, session_state):
    """Main entry point."""

    # Create the model (checks API keys)
    model = create_model()

    # Create agent with conditional tools
    tools = [http_request]
    if tavily_client is not None:
        tools.append(web_search)

    agent = create_agent_with_config(model, assistant_id, tools)

    try:
        await simple_cli(agent, assistant_id, session_state)
    except Exception as e:
        console.print(f"\n[bold red]❌ Error:[/bold red] {e}\n")


def cli_main():
    """Entry point for console script."""
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
        else:
            # Create session state from args
            session_state = SessionState(auto_approve=args.auto_approve)

            # API key validation happens in create_model()
            asyncio.run(main(args.agent, session_state))
    except KeyboardInterrupt:
        # Clean exit on Ctrl+C - suppress ugly traceback
        console.print("\n\n[yellow]Interrupted[/yellow]")
        sys.exit(0)


if __name__ == "__main__":
    cli_main()
