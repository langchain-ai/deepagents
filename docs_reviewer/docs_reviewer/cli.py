"""Conversational CLI powered by DeepAgents."""

import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.prompt import Prompt
from dotenv import load_dotenv

from docs_reviewer.cli_agent import DocsReviewerCLIAgent

# Load environment variables from .env file if it exists
load_dotenv()

app = typer.Typer(
    name="docs-reviewer",
    help="AI-powered documentation code reviewer",
    add_completion=False,
)
console = Console()


def check_api_key() -> bool:
    """Check if Anthropic API key is configured."""
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def show_welcome():
    """Display welcome message."""
    console.print("\n[bold #beb4fd]Docs Reviewer[/bold #beb4fd] - AI-powered documentation code reviewer\n")
    console.print("I can review markdown files and fix broken code snippets.")
    console.print("Just provide a file path to get started!\n")


def show_setup_instructions():
    """Show setup instructions if not configured."""
    console.print("\n[yellow]⚠️  API key required![/yellow]\n")
    console.print("Set your Anthropic API key:")
    console.print("   [cyan]export ANTHROPIC_API_KEY='your-api-key'[/cyan]\n")
    console.print("Then run: [cyan]docs-reviewer chat[/cyan]\n")


@app.command()
def chat(
    message: Optional[str] = typer.Option(
        None,
        "--message",
        "-m",
        help="Single message to process (non-interactive mode)",
    ),
    no_mcp: bool = typer.Option(
        False,
        "--no-mcp",
        help="Disable MCP integration for LangChain docs",
    ),
) -> None:
    """Start interactive chat session with the docs reviewer agent."""
    import asyncio

    # Check API key
    if not check_api_key():
        show_setup_instructions()
        raise typer.Exit(1)

    # Show minimal welcome message (only in interactive mode)
    if not message:
        show_welcome()
        console.print()

    # Run async initialization and chat loop
    # Use asyncio.run to create a new event loop
    try:
        asyncio.run(_async_chat(message, enable_mcp=not no_mcp))
    except RuntimeError as e:
        if "asyncio.run() cannot be called from a running event loop" in str(e):
            # Already in an async context (e.g., Jupyter)
            # Get the current loop and run the coroutine
            import sys
            if sys.platform == 'win32':
                asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

            loop = asyncio.get_event_loop()
            loop.run_until_complete(_async_chat(message, enable_mcp=not no_mcp))
        else:
            raise


async def _async_chat(message: Optional[str], enable_mcp: bool = True) -> None:
    """Async chat handler for proper MCP initialization."""
    # Initialize the CLI agent (simple - just needs API key in env)
    try:
        agent = DocsReviewerCLIAgent(enable_mcp=enable_mcp)
        await agent.async_init()
    except Exception as e:
        console.print(f"[red]Error initializing agent: {e}[/red]")
        raise typer.Exit(1)

    # Single message mode
    if message:
        console.print("\n[bold #beb4fd]Docs Reviewer[/bold #beb4fd]")
        response = await agent.process_message(message, console)
        console.print()
        return

    # Interactive chat loop with command history
    conversation_active = True

    # Use prompt_toolkit for better input with persistent history
    # IMPORTANT: Use async version (prompt_async) when in async context
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import FileHistory
        from prompt_toolkit.styles import Style

        # Store history in user's home directory
        history_file = Path.home() / ".docs_reviewer_history"
        prompt_history = FileHistory(str(history_file))
        prompt_style = Style.from_dict({
            'prompt': '#beb4fd bold',
        })
        session = PromptSession(history=prompt_history, style=prompt_style)
        use_prompt_toolkit = True
    except ImportError:
        use_prompt_toolkit = False
        console.print("[dim]Tip: Install prompt_toolkit for command history (pip install prompt_toolkit)[/dim]\n")

    while conversation_active:
        try:
            # Get user input with history support
            if use_prompt_toolkit:
                from prompt_toolkit.formatted_text import HTML
                # Use prompt_async() in async context, not prompt()!
                user_input = await session.prompt_async(HTML('\n<prompt>You:</prompt> '))
            else:
                user_input = Prompt.ask("\n[bold #beb4fd]You[/bold #beb4fd]")

            if user_input.lower() in ["exit", "quit", "bye", "goodbye"]:
                console.print()
                conversation_active = False
                continue

            if not user_input.strip():
                continue

            # Process with agent - show streaming progress
            console.print("\n[bold #beb4fd]Docs Reviewer[/bold #beb4fd]")
            response = await agent.process_message(user_input, console)
            console.print()

        except KeyboardInterrupt:
            console.print("\n")
            conversation_active = False
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", "-v", help="Show version and exit"),
    message: Optional[str] = typer.Option(None, "--message", "-m", help="Single message mode"),
) -> None:
    """
    Docs Reviewer - AI-powered documentation code reviewer.

    Just run 'docs-reviewer' to start chatting!
    """
    if version:
        console.print("docs-reviewer version 0.1.0")
        raise typer.Exit()

    # If no subcommand was invoked, start chat by default
    if ctx.invoked_subcommand is None:
        ctx.invoke(chat, message=message)


if __name__ == "__main__":
    app()
