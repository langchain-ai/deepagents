"""Command-line interface for ChATLAS agents."""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler

from chatlas_agents import __version__ as CHATLAS_AGENTS_VERSION

app = typer.Typer(
    name="chatlas",
    help="ChATLAS AI agents with MCP integration and sandbox support",
    invoke_without_command=True,
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


async def _run_interactive_session(
    agent_id: str,
    config_file: Optional[Path] = None,
    mcp_url: Optional[str] = None,
    mcp_timeout: Optional[int] = None,
    model: Optional[str] = None,
    auto_approve: bool = False,
    sandbox: Optional[str] = None,
    sandbox_image: Optional[str] = None,
    verbose: bool = False,
):
    """Run an interactive ChATLAS agent session.

    Args:
        agent_id: Agent identifier for memory storage
        config_file: Optional path to YAML configuration file
        mcp_url: Optional MCP server URL override
        mcp_timeout: Optional MCP server timeout override
        model: Optional LLM model override
        auto_approve: Auto-approve tool usage without prompting
        sandbox: Sandbox type ("docker", "apptainer", or None for local)
        sandbox_image: Container image to use for sandbox
        verbose: Enable verbose logging
    """
    from chatlas_agents.config import (
        AgentConfig,
        MCPServerConfig,
        load_config_from_yaml,
        load_config_from_env,
    )
    from chatlas_agents.middleware import MCPMiddleware
    from chatlas_agents.sandbox import (
        DockerSandboxBackend,
        ApptainerSandboxBackend,
        SandboxBackendType,
    )
    from deepagents_cli.agent import create_cli_agent
    from deepagents_cli.config import SessionState, create_model
    from deepagents_cli.main import simple_cli
    from deepagents_cli.token_utils import calculate_baseline_tokens

    logger = logging.getLogger(__name__)

    # Load configuration
    if config_file:
        config = load_config_from_yaml(str(config_file))
        logger.info(f"Loaded configuration from {config_file}")
    else:
        config = load_config_from_env()

    # Apply overrides
    if mcp_url:
        config.mcp.url = mcp_url
    if mcp_timeout is not None:
        config.mcp.timeout = mcp_timeout
    if model:
        config.llm.model = model

    # Create MCP middleware and load tools
    logger.info(f"Connecting to ChATLAS MCP server at {config.mcp.url}...")
    try:
        # Enforce an explicit timeout so users are not stuck waiting indefinitely
        connect_timeout = getattr(config.mcp, "timeout", None) or 120
        mcp_middleware = await asyncio.wait_for(
            MCPMiddleware.create(config.mcp),
            timeout=connect_timeout,
        )
        mcp_tools = mcp_middleware.tools
        logger.info(f"Successfully loaded {len(mcp_tools)} MCP tools")
    except asyncio.TimeoutError:
        console.print(
            f"[red]Timed out after {connect_timeout} seconds while connecting to MCP server at {config.mcp.url}[/red]"
        )
        raise typer.Exit(1)
    except KeyboardInterrupt:
        console.print("[yellow]MCP connection cancelled by user (Ctrl+C).[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Failed to connect to MCP server: {e}[/red]")
        raise typer.Exit(1)

    # Setup sandbox backend if requested
    sandbox_backend = None
    sandbox_type_str = None
    if sandbox:
        sandbox_type_str = sandbox.lower()
        try:
            sandbox_type = SandboxBackendType(sandbox_type_str)
        except ValueError:
            valid_types = ", ".join(t.value for t in SandboxBackendType)
            console.print(
                f"[red]Invalid sandbox type '{sandbox}'. "
                f"Valid options are: {valid_types}[/red]"
            )
            sys.exit(1)
        image = sandbox_image or "python:3.13-slim"

        try:
            if sandbox_type == SandboxBackendType.DOCKER:
                logger.info(f"Creating Docker sandbox with image: {image}")
                sandbox_backend = DockerSandboxBackend(image=image)
                logger.info(f"Docker sandbox created: {sandbox_backend.id[:12]}")
            elif sandbox_type == SandboxBackendType.APPTAINER:
                logger.info(f"Creating Apptainer sandbox with image: {image}")
                # Ensure proper format for Apptainer
                if not any(image.startswith(p) for p in ["docker://", "oras://", "library://", "/"]):
                    image = f"docker://{image}"
                sandbox_backend = ApptainerSandboxBackend(image=image)
                logger.info(f"Apptainer sandbox created: {sandbox_backend.id}")
        except Exception as e:
            console.print(f"[red]Failed to create {sandbox} sandbox: {e}[/red]")
            sys.exit(1)

    # Create LLM model
    # Note: create_model() uses global settings, we need to set the model first
    import os
    if config.llm.model:
        os.environ["OPENAI_MODEL"] = config.llm.model
    llm_model = create_model()

    # Create CLI agent with MCP tools
    logger.info(f"Creating ChATLAS agent '{agent_id}'...")
    
    # Use try-finally to ensure sandbox cleanup
    try:
        agent, composite_backend = create_cli_agent(
            model=llm_model,
            assistant_id=agent_id,
            tools=mcp_tools,
            sandbox=sandbox_backend,
            sandbox_type=sandbox_type_str,
            auto_approve=auto_approve,
            enable_memory=True,
            enable_skills=True,
            enable_shell=(sandbox is None),  # Shell only in local mode
        )

        # Calculate baseline tokens for tracking
        from deepagents_cli.agent import get_system_prompt
        from deepagents_cli.config import settings

        agent_dir = settings.get_agent_dir(agent_id)
        system_prompt = get_system_prompt(assistant_id=agent_id, sandbox_type=sandbox_type_str)
        baseline_tokens = calculate_baseline_tokens(llm_model, agent_dir, system_prompt, agent_id)

        # Create session state
        session_state = SessionState()
        session_state.auto_approve = auto_approve
        session_state.no_splash = False

        # Run interactive CLI
        logger.info("Starting interactive session...")
        await simple_cli(
            agent,
            agent_id,
            session_state,
            baseline_tokens=baseline_tokens,
            backend=composite_backend,
            sandbox_type=sandbox_type_str,
            setup_script_path=None,
            no_splash=False,
        )
    finally:
        # Cleanup sandbox backend if it was created
        if sandbox_backend is not None:
            try:
                logger.info("Cleaning up sandbox backend...")
                if hasattr(sandbox_backend, 'cleanup'):
                    sandbox_backend.cleanup()
                elif hasattr(sandbox_backend, '__exit__'):
                    sandbox_backend.__exit__(None, None, None)
            except Exception as e:
                logger.warning(f"Failed to cleanup sandbox backend: {e}")


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    agent: str = typer.Option(
        "chatlas",
        "--agent",
        "-a",
        help="Agent identifier for memory storage (default: chatlas)",
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to YAML configuration file",
    ),
    mcp_url: Optional[str] = typer.Option(
        None,
        "--mcp-url",
        help="ChATLAS MCP server URL (default: https://chatlas-mcp.app.cern.ch/mcp)",
    ),
    mcp_timeout: Optional[int] = typer.Option(
        None,
        "--mcp-timeout",
        help="MCP server timeout in seconds (default: 120)",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="LLM model to use (default: from config or gpt-4)",
    ),
    auto_approve: bool = typer.Option(
        False,
        "--auto-approve",
        help="Auto-approve tool usage without prompting",
    ),
    sandbox: Optional[str] = typer.Option(
        None,
        "--sandbox",
        help="Sandbox type: 'docker' or 'apptainer' (default: none - local execution)",
    ),
    sandbox_image: Optional[str] = typer.Option(
        None,
        "--sandbox-image",
        help="Container image for sandbox (default: python:3.13-slim)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging",
    ),
):
    """ChATLAS AI agent with MCP integration.

    Simply running 'chatlas' launches an interactive session with the ChATLAS agent.
    The agent has access to ChATLAS MCP tools for searching ATLAS documentation.

    Examples:
        # Start interactive session with default settings
        chatlas

        # Use custom agent name and MCP server
        chatlas --agent my-agent --mcp-url https://custom-mcp.example.com/mcp

        # Enable Docker sandbox for isolated code execution
        chatlas --sandbox docker

        # Use Apptainer sandbox (for HPC environments like lxplus)
        chatlas --sandbox apptainer --sandbox-image docker://python:3.13-slim

        # Auto-approve all tool calls (non-interactive)
        chatlas --auto-approve

        # Use custom configuration file
        chatlas --config my-config.yaml
    """
    setup_logging(verbose)

    # If a subcommand is invoked, let it handle execution
    if ctx.invoked_subcommand is not None:
        return

    # Default behavior: run interactive session
    try:
        asyncio.run(
            _run_interactive_session(
                agent_id=agent,
                config_file=config,
                mcp_url=mcp_url,
                mcp_timeout=mcp_timeout,
                model=model,
                auto_approve=auto_approve,
                sandbox=sandbox,
                sandbox_image=sandbox_image,
                verbose=verbose,
            )
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        if verbose:
            console.print_exception()
        sys.exit(1)


@app.command()
def init(
    output: Path = typer.Option(
        ".env",
        "--output",
        "-o",
        help="Output configuration file path (default: .env)",
    ),
):
    """Initialize a new ChATLAS configuration file.

    Creates an environment file with ChATLAS configuration variables.
    """
    config_content = """# ChATLAS Agent Configuration
# Set your environment variables below

# ChATLAS MCP Server Configuration
CHATLAS_MCP_URL=https://chatlas-mcp.app.cern.ch/mcp
CHATLAS_MCP_TIMEOUT=120

# LLM Configuration
CHATLAS_LLM_PROVIDER=openai
CHATLAS_LLM_MODEL=gpt-4

# LLM API Keys (at least one required)
OPENAI_API_KEY=your-openai-api-key-here
# ANTHROPIC_API_KEY=your-anthropic-api-key-here

# Optional: Web search via Tavily (for deepagents built-in tools)
# TAVILY_API_KEY=your-tavily-api-key-here
"""

    with open(output, "w") as f:
        f.write(config_content)

    console.print(f"[green]✓ Created configuration file: {output}[/green]")
    console.print("[dim]Edit this file and add your API keys, then run:[/dim]")
    console.print(f"[dim]  export $(cat {output} | xargs)[/dim]")
    console.print("[dim]You can now run the CLI:[/dim]")
    console.print("[dim]  chatlas[/dim]")


@app.command()
def version():
    """Show version information."""
    console.print(f"ChATLAS Agents v{CHATLAS_AGENTS_VERSION}")
    console.print("[dim]Built on DeepAgents framework with MCP integration[/dim]")


if __name__ == "__main__":
    app()
