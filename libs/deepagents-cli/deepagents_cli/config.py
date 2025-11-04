"""Configuration, constants, and model creation for the CLI."""

import os
import sys
from pathlib import Path

import dotenv
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
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
    "provider": "Switch between OpenAI and Anthropic providers",
    "tokens": "Show token usage for current session",
    "quit": "Exit the CLI",
    "exit": "Exit the CLI",
}

# Common bash commands for autocomplete
COMMON_BASH_COMMANDS = {
    "ls": "List directory contents",
    "ls -la": "List all files with details",
    "cd": "Change directory",
    "pwd": "Print working directory",
    "cat": "Display file contents",
    "grep": "Search text patterns",
    "find": "Find files",
    "mkdir": "Make directory",
    "rm": "Remove file",
    "cp": "Copy file",
    "mv": "Move/rename file",
    "echo": "Print text",
    "touch": "Create empty file",
    "head": "Show first lines",
    "tail": "Show last lines",
    "wc": "Count lines/words",
    "chmod": "Change permissions",
}

# Maximum argument length for display
MAX_ARG_LENGTH = 150

# Agent configuration
config = {"recursion_limit": 1000}

# Rich console instance
console = Console(highlight=False)


class SessionState:
    """Holds mutable session state (auto-approve mode, etc)."""

    def __init__(self, auto_approve: bool = False, preferred_provider: str | None = None):
        self.auto_approve = auto_approve
        self.preferred_provider = preferred_provider

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


def _create_openai_model():
    """Create an OpenAI model instance."""
    console.print("[dim]Using OpenAI model: gpt-5-mini[/dim]")
    return ChatOpenAI(
        model="gpt-5-mini",
        temperature=0.7,
    )


def _create_anthropic_model():
    """Create an Anthropic model instance."""
    console.print("[dim]Using Anthropic model: claude-sonnet-4-5-20250929[/dim]")
    return ChatAnthropic(
        model_name="claude-sonnet-4-5-20250929",
        max_tokens=20000,
    )


def _show_api_key_error(provider: str | None = None):
    """Show error message for missing API key and exit."""
    if provider == "openai":
        console.print("[bold red]Error:[/bold red] OPENAI_API_KEY not configured.")
        console.print("\nPlease set your OpenAI API key:")
        console.print("  export OPENAI_API_KEY=your_api_key_here")
    elif provider == "anthropic":
        console.print("[bold red]Error:[/bold red] ANTHROPIC_API_KEY not configured.")
        console.print("\nPlease set your Anthropic API key:")
        console.print("  export ANTHROPIC_API_KEY=your_api_key_here")
    else:
        console.print("[bold red]Error:[/bold red] No API key configured.")
        console.print("\nPlease set one of the following environment variables:")
        console.print("  - OPENAI_API_KEY     (for OpenAI models like gpt-5-mini)")
        console.print("  - ANTHROPIC_API_KEY  (for Claude models)")
        console.print("\nExample:")
        console.print("  export OPENAI_API_KEY=your_api_key_here")

    console.print("\nOr add it to your .env file.")
    sys.exit(1)


def create_model(force_provider: str | None = None):
    """Create the appropriate model based on available API keys.

    Args:
        force_provider: Optional provider to force ("openai" or "anthropic").
                       If specified, only that provider will be used.

    Returns:
        ChatModel instance (OpenAI or Anthropic)

    Raises:
        SystemExit if no API key is configured
    """
    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

    # Determine which provider to use
    if force_provider == "openai":
        if not openai_key:
            _show_api_key_error("openai")
        return _create_openai_model()

    if force_provider == "anthropic":
        if not anthropic_key:
            _show_api_key_error("anthropic")
        return _create_anthropic_model()

    # Default behavior: prefer OpenAI, fallback to Anthropic
    if openai_key:
        return _create_openai_model()
    if anthropic_key:
        return _create_anthropic_model()

    _show_api_key_error()


def load_agent_config(agent_name: str) -> dict:
    """Load agent configuration from config.json.

    Args:
        agent_name: Name of the agent

    Returns:
        Dictionary with config data, empty dict if file doesn't exist
    """
    import json
    from pathlib import Path

    config_path = Path.home() / ".deepagents" / agent_name / "config.json"

    if not config_path.exists():
        return {}

    try:
        with open(config_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        # If file is corrupted or unreadable, return empty config
        return {}


def save_agent_config(agent_name: str, config_data: dict) -> None:
    """Save agent configuration to config.json.

    Args:
        agent_name: Name of the agent
        config_data: Dictionary to save as JSON
    """
    import json
    from datetime import datetime, timezone
    from pathlib import Path

    agent_dir = Path.home() / ".deepagents" / agent_name
    agent_dir.mkdir(parents=True, exist_ok=True)

    config_path = agent_dir / "config.json"

    # Add timestamp
    config_data["last_updated"] = datetime.now(timezone.utc).isoformat()

    try:
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2)
    except IOError as e:
        console.print(f"[yellow]Warning: Could not save config: {e}[/yellow]")
