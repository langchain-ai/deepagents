"""Configuration, constants, and model creation for the CLI."""

import os
import sys
from pathlib import Path

import dotenv
from rich.console import Console

dotenv.load_dotenv()

# Color scheme
COLORS = {
    "primary": "#10b981",
    "dim": "#6b7280",
    "user": "#ffffff",
    "agent": "#10b981",
    "thinking": "#34d399",
    "tool": "#fbbf24",
}

# ASCII art banner
DEEP_AGENTS_ASCII = """
 ██████╗  ███████╗ ███████╗ ██████╗
 ██╔══██╗ ██╔════╝ ██╔════╝ ██╔══██╗
 ██║  ██║ █████╗   █████╗   ██████╔╝
 ██║  ██║ ██╔══╝   ██╔══╝   ██╔═══╝
 ██████╔╝ ███████╗ ███████╗ ██║
 ╚═════╝  ╚══════╝ ╚══════╝ ╚═╝

  █████╗   ██████╗  ███████╗ ███╗   ██╗ ████████╗ ███████╗
 ██╔══██╗ ██╔════╝  ██╔════╝ ████╗  ██║ ╚══██╔══╝ ██╔════╝
 ███████║ ██║  ███╗ █████╗   ██╔██╗ ██║    ██║    ███████╗
 ██╔══██║ ██║   ██║ ██╔══╝   ██║╚██╗██║    ██║    ╚════██║
 ██║  ██║ ╚██████╔╝ ███████╗ ██║ ╚████║    ██║    ███████║
 ╚═╝  ╚═╝  ╚═════╝  ╚══════╝ ╚═╝  ╚═══╝    ╚═╝    ╚══════╝
"""

# Interactive commands
COMMANDS = {
    "clear": "Clear screen and reset conversation",
    "help": "Show help information",
    "tokens": "Show token usage for current session",
    "quit": "Exit the CLI",
    "exit": "Exit the CLI",
}


# Maximum argument length for display
MAX_ARG_LENGTH = 150

# Agent configuration
config = {"recursion_limit": 1000}

# Rich console instance
console = Console(highlight=False)


class SessionState:
    """Holds mutable session state (auto-approve mode, etc)."""

    def __init__(self, auto_approve: bool = False):
        self.auto_approve = auto_approve

    def toggle_auto_approve(self) -> bool:
        """Toggle auto-approve and return new state."""
        self.auto_approve = not self.auto_approve
        return self.auto_approve


def get_default_coding_instructions() -> str:
    """Get the default coding agent instructions.

    These are the immutable base instructions that cannot be modified by the agent.
    Long-term memory (agent.md) is handled separately by the middleware.
    """
    default_prompt_path = Path(__file__).parent / "default_agent_prompt.md"
    return default_prompt_path.read_text()


def create_model():
    """Create the appropriate model based on available API keys.

    Supports OpenAI and Anthropic APIs, as well as compatible APIs from other providers
    (e.g., MiniMax M2, OpenRouter, etc.) via custom base URLs.

    Environment Variables:
        OPENAI_API_KEY: API key for OpenAI or OpenAI-compatible APIs
        OPENAI_MODEL: Model name (default: gpt-5-mini)
        OPENAI_BASE_URL: Custom base URL for OpenAI-compatible APIs (optional)

        ANTHROPIC_API_KEY: API key for Anthropic or Anthropic-compatible APIs
        ANTHROPIC_MODEL: Model name (default: claude-sonnet-4-5-20250929)
        ANTHROPIC_BASE_URL: Custom base URL for Anthropic-compatible APIs (optional)

    Returns:
        ChatModel instance (OpenAI or Anthropic)

    Raises:
        SystemExit if no API key is configured

    Examples:
        # Using OpenAI
        export OPENAI_API_KEY=sk-...
        export OPENAI_MODEL=gpt-4

        # Using MiniMax M2 (Anthropic-compatible)
        export ANTHROPIC_API_KEY=your_minimax_key
        export ANTHROPIC_MODEL=MiniMax-M2
        export ANTHROPIC_BASE_URL=https://api.minimax.io/anthropic

        # Using other OpenAI-compatible providers
        export OPENAI_API_KEY=your_api_key
        export OPENAI_MODEL=your_model_name
        export OPENAI_BASE_URL=https://api.your-provider.com/v1
    """
    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

    if openai_key:
        from langchain_openai import ChatOpenAI

        model_name = os.environ.get("OPENAI_MODEL", "gpt-5-mini")
        base_url = os.environ.get("OPENAI_BASE_URL")

        if base_url:
            console.print(f"[dim]Using OpenAI-compatible model: {model_name} (base_url: {base_url})[/dim]")
        else:
            console.print(f"[dim]Using OpenAI model: {model_name}[/dim]")

        kwargs = {
            "model": model_name,
            "temperature": 0.7,
        }
        if base_url:
            kwargs["base_url"] = base_url

        return ChatOpenAI(**kwargs)

    if anthropic_key:
        from langchain_anthropic import ChatAnthropic

        model_name = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")
        base_url = os.environ.get("ANTHROPIC_BASE_URL")

        if base_url:
            console.print(f"[dim]Using Anthropic-compatible model: {model_name} (base_url: {base_url})[/dim]")
        else:
            console.print(f"[dim]Using Anthropic model: {model_name}[/dim]")

        kwargs = {
            "model_name": model_name,
            "max_tokens": 20000,
        }
        if base_url:
            kwargs["base_url"] = base_url

        return ChatAnthropic(**kwargs)

    console.print("[bold red]Error:[/bold red] No API key configured.")
    console.print("\nPlease set one of the following environment variables:")
    console.print("  - OPENAI_API_KEY     (for OpenAI models like gpt-5-mini)")
    console.print("  - ANTHROPIC_API_KEY  (for Claude models)")
    console.print("\n[bold]For OpenAI-compatible providers:[/bold]")
    console.print("  export OPENAI_API_KEY=your_api_key")
    console.print("  export OPENAI_MODEL=your_model_name")
    console.print("  export OPENAI_BASE_URL=https://api.your-provider.com/v1")
    console.print("\n[bold]For Anthropic-compatible providers (e.g., MiniMax M2):[/bold]")
    console.print("  export ANTHROPIC_API_KEY=your_api_key")
    console.print("  export ANTHROPIC_MODEL=MiniMax-M2")
    console.print("  export ANTHROPIC_BASE_URL=https://api.minimax.io/anthropic")
    console.print("\nOr add these to your .env file.")
    sys.exit(1)
