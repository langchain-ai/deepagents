"""Command-line interface for ChATLAS agents."""

import asyncio
import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler

from chatlas_agents.config import AgentConfig, load_config_from_env, load_config_from_yaml
from chatlas_agents.agents import create_deep_agent
from chatlas_agents.sandbox import SandboxBackendType
from chatlas_agents import __version__ as CHATLAS_AGENTS_VERSION

app = typer.Typer(
    name="chatlas-agents",
    help="ChATLAS AI agents using DeepAgents and LangChain",
)
console = Console()

# ASCII art splash screen
SPLASH_ART = r"""
[deep_sky_blue1]               ++============++               
            +=====++[/deep_sky_blue1]        [deep_sky_blue1]++++++            
          +++=-=[/deep_sky_blue1][bright_cyan]:::..[/bright_cyan]           [deep_sky_blue1]+++++         
       ++++=+     [/deep_sky_blue1][bright_cyan]:::..[/bright_cyan]     [deep_sky_blue1]+++++  +=+++      
      +++ +++       [/deep_sky_blue1][bright_cyan]:::.[/bright_cyan][deep_sky_blue1]+++++      [bright_cyan].::[/bright_cyan]=++[/deep_sky_blue1]      
[magenta]  ____ _ [/magenta][cyan]          _______ _            _____ [/cyan][magenta]
 / ___| |      [/magenta][cyan] /\|__   __| |      /\  /  ___| [/cyan][magenta]
| |   | |__   [/magenta][cyan] /  \  | |  | |     /  \ | (___  [/cyan][magenta]
| |   | '_ \ [/magenta][cyan] / /\ \ | |  | |    / /\ \ \__  \ [/cyan][magenta]
| |___| | | |[/magenta][cyan]/ ____ \| |  | |__ / ____ \___) | [/cyan][magenta]
 \____|_| |_[/magenta][cyan]/_/    \_\_|  |____/_/    \_\____/ 
[/cyan]                                              
[deep_sky_blue1]       ++++[/deep_sky_blue1][bright_cyan].:::[/bright_cyan]   [deep_sky_blue1]++++       [/deep_sky_blue1][bright_cyan]::::.[/bright_cyan] [bright_cyan]::[/bright_cyan][deep_sky_blue1]=+++      
        ++++       +++++      [/deep_sky_blue1][bright_cyan]:::.[/bright_cyan][deep_sky_blue1].++++       
         ++++         +++     [/deep_sky_blue1][bright_cyan].::[/bright_cyan][deep_sky_blue1]+==+         
        +++++++       +++++[/deep_sky_blue1][bright_cyan]:.:[/bright_cyan][deep_sky_blue1]==++           
       +++   ++++============+++              
      ++====++[/deep_sky_blue1]                        [dim]v. {version}[/dim]

[dim]AI Agents for ATLAS • Powered by DeepAgents[/dim]
"""
SPLASH_ART = SPLASH_ART.format(version=CHATLAS_AGENTS_VERSION)

def show_splash():
    """Display the ASCII art splash screen."""
    console.print(SPLASH_ART)
    console.print()


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


@app.command()
def run(
    config_file: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to agent configuration YAML file",
    ),
    input_text: Optional[str] = typer.Option(
        None,
        "--input",
        "-i",
        help="Input text to send to the agent",
    ),
    interactive: bool = typer.Option(
        False,
        "--interactive",
        "-I",
        help="Run in interactive mode",
    ),
    thread_id: str = typer.Option(
        "default",
        "--thread",
        "-t",
        help="Thread ID for conversation persistence",
    ),
    stream: bool = typer.Option(
        False,
        "--stream",
        help="Stream responses",
    ),
    sandbox: bool = typer.Option(
        False,
        "--sandbox",
        help="Enable sandbox for secure code execution",
    ),
    sandbox_image: str = typer.Option(
        "python:3.13-slim",
        "--sandbox-image",
        help="Container image to use for sandbox",
    ),
    sandbox_backend: str = typer.Option(
        "apptainer",
        "--sandbox-backend",
        help="Sandbox backend type: docker or apptainer",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging",
    ),
):
    """Run a ChATLAS agent using DeepAgents."""
    setup_logging(verbose)

    # Load configuration
    if config_file:
        console.print(f"[bright_cyan]Loading configuration from {config_file}[/bright_cyan]")
        config = load_config_from_yaml(str(config_file))
    else:
        console.print("[bright_cyan]Loading configuration from environment[/bright_cyan]")
        config = load_config_from_env()

    if verbose:
        config.verbose = True

    # Validate and convert sandbox_backend
    try:
        backend_type = SandboxBackendType(sandbox_backend.lower())
    except ValueError:
        console.print(f"[red]Error: Invalid sandbox backend '{sandbox_backend}'. Must be 'docker' or 'apptainer'.[/red]")
        raise typer.Exit(1)

    # Run the DeepAgent
    asyncio.run(_run_agent(config, input_text, interactive, thread_id, stream, sandbox, sandbox_image, backend_type))


async def _run_agent(
    config: AgentConfig,
    input_text: Optional[str],
    interactive: bool,
    thread_id: str,
    stream: bool,
    use_sandbox: bool = False,
    sandbox_image: str = "python:3.13-slim",
    sandbox_backend: SandboxBackendType = SandboxBackendType.APPTAINER,
):
    """Run the DeepAgent asynchronously.

    Args:
        config: Agent configuration
        input_text: Optional input text
        interactive: Whether to run in interactive mode
        thread_id: Thread ID for conversation
        stream: Whether to stream responses
        use_sandbox: Whether to use container sandbox
        sandbox_image: Container image to use for sandbox
        sandbox_backend: Type of sandbox backend (docker or apptainer)
    """
    if use_sandbox:
        console.print(f"[green]Initializing DeepAgent '{config.name}' with {sandbox_backend.value.capitalize()} sandbox...[/green]")
        console.print(f"[dim]Using container image: {sandbox_image}[/dim]")
    else:
        console.print(f"[green]Initializing DeepAgent '{config.name}'...[/green]")

    agent = await create_deep_agent(
        config, 
        use_docker_sandbox=use_sandbox, 
        docker_image=sandbox_image,
        sandbox_backend=sandbox_backend,
    )

    try:
        if interactive:
            await _run_interactive(agent, thread_id, stream)
        elif input_text:
            if stream:
                console.print("\n[bold]Agent Response (streaming):[/bold]")
                async for event in agent.stream(input_text, thread_id):
                    console.print(event)
            else:
                result = await agent.run(input_text, thread_id)
                console.print("\n[bold]Agent Response:[/bold]")
                console.print(result.get("output", result))
        else:
            console.print("[red]Error: Either --input or --interactive must be provided[/red]")
            raise typer.Exit(1)
    finally:
        await agent.close()


async def _run_interactive(agent, thread_id: str, stream: bool):
    """Run the agent in interactive mode.

    Args:
        agent: DeepAgent instance
        thread_id: Thread ID for conversation
        stream: Whether to stream responses
    """
    console.print(f"\n[bold green]Interactive mode (thread: {thread_id}) - Type 'exit' or 'quit' to end[/bold green]\n")

    while True:
        try:
            user_input = console.input("[bold cyan]You:[/bold cyan] ")

            if user_input.lower() in ["exit", "quit"]:
                console.print("[yellow]Goodbye![/yellow]")
                break

            if not user_input.strip():
                continue

            if stream:
                console.print(f"\n[bold green]Agent:[/bold green] ", end="")
                async for event in agent.stream(user_input, thread_id):
                    # Extract message content from events
                    if isinstance(event, dict):
                        for node, data in event.items():
                            if "messages" in data:
                                for msg in data["messages"]:
                                    if hasattr(msg, "content"):
                                        console.print(msg.content, end="")
                console.print("\n")
            else:
                result = await agent.run(user_input, thread_id)
                console.print(f"\n[bold green]Agent:[/bold green] {result.get('output', result)}\n")

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Goodbye![/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


@app.command()
def init(
    output: Path = typer.Option(
        "agent-config.yaml",
        "--output",
        "-o",
        help="Output configuration file path",
    ),
):
    """Initialize a new agent configuration file."""
    from chatlas_agents.config import AgentConfig
    import yaml

    config = AgentConfig()
    # Convert to dict with proper serialization
    config_dict = config.model_dump(mode='json')

    with open(output, "w") as f:
        yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)

    console.print(f"[green]Created configuration file: {output}[/green]")


@app.command()
def version():
    """Show version information."""
    from chatlas_agents import __version__

    console.print(f"ChATLAS Agents v{__version__}")


@app.command()
def htcondor_submit(
    job_name: str = typer.Option(
        ...,
        "--job-name",
        "-n",
        help="Name for the HTCondor batch job",
    ),
    prompt: str = typer.Option(
        ...,
        "--prompt",
        "-p",
        help="Input prompt to send to the agent",
    ),
    config_file: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to agent configuration YAML file",
    ),
    docker_image: str = typer.Option(
        "python:3.13-slim",
        "--docker-image",
        help="Docker image to use for sandbox",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Directory for job output files (default: ./htcondor_jobs)",
    ),
    env_file: Optional[Path] = typer.Option(
        None,
        "--env-file",
        "-e",
        help="Path to .env file with environment variables",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Generate submit file without submitting to HTCondor",
    ),
    cpus: int = typer.Option(
        1,
        "--cpus",
        help="Number of CPUs to request",
    ),
    memory: str = typer.Option(
        "2GB",
        "--memory",
        help="Memory to request (e.g., 2GB, 4GB)",
    ),
    disk: str = typer.Option(
        "1GB",
        "--disk",
        help="Disk space to request (e.g., 1GB, 5GB)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging",
    ),
):
    """Submit a ChATLAS agent job to CERN HTCondor batch farm.
    
    This command generates an HTCondor submit file and submits a batch job
    that runs a ChATLAS agent with a Docker sandbox.
    
    Example:
        chatlas-agent htcondor-submit --job-name my-job --prompt "Analyze data"
    """
    setup_logging(verbose)
    
    from chatlas_agents.htcondor import HTCondorJobSubmitter
    
    # Load environment variables if provided
    env_vars = {}
    if env_file:
        console.print(f"[blue]Loading environment from {env_file}[/blue]")
        try:
            from dotenv import dotenv_values
            env_vars = dict(dotenv_values(env_file))
        except ImportError:
            console.print("[red]Error: python-dotenv is required for --env-file option[/red]")
            console.print("[yellow]Install it with: pip install python-dotenv[/yellow]")
            raise typer.Exit(1)
    
    # Create submitter
    submitter = HTCondorJobSubmitter(
        docker_image=docker_image,
        output_dir=output_dir,
    )
    
    # Prepare HTCondor parameters
    htcondor_params = {
        "request_cpus": cpus,
        "request_memory": memory,
        "request_disk": disk,
    }
    
    try:
        # Submit the job
        console.print(f"\n[bold green]Submitting HTCondor job: {job_name}[/bold green]")
        console.print(f"[dim]Prompt: {prompt}[/dim]")
        console.print(f"[dim]Docker image: {docker_image}[/dim]")
        
        if config_file:
            console.print(f"[dim]Config file: {config_file}[/dim]")
        
        cluster_id = submitter.submit_job(
            job_name=job_name,
            prompt=prompt,
            config_file=str(config_file) if config_file else None,
            env_vars=env_vars if env_vars else None,
            dry_run=dry_run,
            **htcondor_params,
        )
        
        if dry_run:
            console.print("\n[yellow]Dry run complete - submit file generated[/yellow]")
            submit_file = submitter.output_dir / job_name / f"{job_name}.sub"
            console.print(f"[blue]Submit file: {submit_file}[/blue]")
            console.print("\n[dim]To submit manually, run:[/dim]")
            console.print(f"[dim]  condor_submit {submit_file}[/dim]")
        else:
            console.print(f"\n[green]✓ Job submitted successfully![/green]")
            if cluster_id:
                console.print(f"[blue]Cluster ID: {cluster_id}[/blue]")
                console.print(f"\n[dim]To check status:[/dim]")
                console.print(f"[dim]  condor_q {cluster_id}[/dim]")
                console.print(f"\n[dim]To view logs:[/dim]")
                log_dir = submitter.output_dir / job_name
                console.print(f"[dim]  tail -f {log_dir}/job.{cluster_id}.*.out[/dim]")
    
    except Exception as e:
        console.print(f"\n[red]✗ Error submitting job: {e}[/red]")
        raise typer.Exit(1)


def main():
    """Main entry point."""
    show_splash()
    app()


if __name__ == "__main__":
    main()
