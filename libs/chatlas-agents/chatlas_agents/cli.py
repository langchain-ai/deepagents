"""Command-line interface for ChATLAS agents."""

import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler

from chatlas_agents import __version__ as CHATLAS_AGENTS_VERSION

app = typer.Typer(
    name="chatlas-agents",
    help="ChATLAS AI agents using DeepAgents and LangChain",
)
console = Console()


def setup_logging(verbose: bool = False):
    """Setup logging configuration.

    Args:
        verbose: Enable verbose logging
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


def _setup_chatlas_env(
    config_file: Optional[Path] = None,
    mcp_url: Optional[str] = None,
    mcp_timeout: Optional[int] = None,
):
    """Setup ChATLAS environment variables for deepagents-cli.

    Args:
        config_file: Path to configuration file
        mcp_url: MCP server URL override
        mcp_timeout: MCP server timeout override
    """
    # Load configuration if provided
    if config_file:
        from chatlas_agents.config import load_config_from_yaml
        config = load_config_from_yaml(str(config_file))
        
        # Set environment variables from config
        if config.mcp.url:
            os.environ["CHATLAS_MCP_URL"] = config.mcp.url
        if config.mcp.timeout:
            os.environ["CHATLAS_MCP_TIMEOUT"] = str(config.mcp.timeout)
        if config.llm.api_key:
            os.environ["OPENAI_API_KEY"] = config.llm.api_key
    else:
        # Load from environment variables with CHATLAS_ prefix
        from chatlas_agents.config import load_config_from_env
        config = load_config_from_env()
        
        # Set MCP environment variables
        os.environ["CHATLAS_MCP_URL"] = config.mcp.url
        os.environ["CHATLAS_MCP_TIMEOUT"] = str(config.mcp.timeout)
    
    # Apply overrides if provided
    if mcp_url:
        os.environ["CHATLAS_MCP_URL"] = mcp_url
    if mcp_timeout:
        os.environ["CHATLAS_MCP_TIMEOUT"] = str(mcp_timeout)


@app.command()
def run(
    config_file: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to agent configuration YAML file",
    ),
    agent: str = typer.Option(
        "chatlas",
        "--agent",
        "-a",
        help="Agent identifier for separate memory stores",
    ),
    interactive: bool = typer.Option(
        False,
        "--interactive",
        "-I",
        help="Run in interactive mode",
    ),
    auto_approve: bool = typer.Option(
        False,
        "--auto-approve",
        help="Auto-approve tool usage without prompting",
    ),
    mcp_url: Optional[str] = typer.Option(
        None,
        "--mcp-url",
        help="ChATLAS MCP server URL",
    ),
    mcp_timeout: Optional[int] = typer.Option(
        None,
        "--mcp-timeout",
        help="ChATLAS MCP server timeout in seconds",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging",
    ),
):
    """Run a ChATLAS agent using DeepAgents.

    This command launches the DeepAgents interactive CLI with ChATLAS MCP tools loaded.
    """
    setup_logging(verbose)
    
    # Setup ChATLAS environment
    _setup_chatlas_env(config_file, mcp_url, mcp_timeout)
    
    # Import and run deepagents CLI
    from deepagents_cli.main import cli_main
    
    # Override sys.argv to pass arguments to deepagents CLI
    sys.argv = ["deepagents"]
    sys.argv.extend(["--agent", agent])
    
    if auto_approve:
        sys.argv.append("--auto-approve")
    
    # Run deepagents CLI (will show ChATLAS splash from submodule)
    try:
        cli_main()
    except SystemExit:
        # Gracefully handle exit
        pass


@app.command()
def init(
    output: Path = typer.Option(
        "chatlas-config.env",
        "--output",
        "-o",
        help="Output configuration file path",
    ),
):
    """Initialize a new ChATLAS configuration file.
    
    Creates a simple environment file with ChATLAS configuration variables.
    """
    config_content = """# ChATLAS Agent Configuration
# Copy this file to .env and fill in your values

# ChATLAS MCP Server Configuration
CHATLAS_MCP_URL=https://chatlas-mcp.app.cern.ch/mcp
CHATLAS_MCP_TIMEOUT=120

# LLM API Key (required - mapped to OPENAI_API_KEY for DeepAgents)
OPENAI_API_KEY=your-openai-api-key-here

# Optional: Alternative LLM providers (used by DeepAgents)
# ANTHROPIC_API_KEY=your-anthropic-api-key-here
# GOOGLE_API_KEY=your-google-api-key-here

# Optional: Web search via Tavily (used by DeepAgents)
# TAVILY_API_KEY=your-tavily-api-key-here
"""
    
    with open(output, "w") as f:
        f.write(config_content)
    
    console.print(f"[green]✓ Created configuration file: {output}[/green]")
    console.print(f"[dim]Copy to .env and fill in your API keys[/dim]")


@app.command()
def version():
    """Show version information."""
    console.print(f"ChATLAS Agents v{CHATLAS_AGENTS_VERSION}")


def main():
    """Main entry point."""
    # Splash screen (if any) is handled by deepagents_cli via the run command, not here
    app()


if __name__ == "__main__":
    main()
