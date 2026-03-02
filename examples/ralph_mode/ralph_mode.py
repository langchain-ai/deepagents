"""Ralph Mode - Autonomous looping for Deep Agents.

Ralph is an autonomous looping pattern created by Geoff Huntley
(https://ghuntley.com/ralph/). Each loop starts with fresh context.
The filesystem and git serve as the agent's memory across iterations.

Each iteration delegates to `run_non_interactive` from `deepagents-cli`,
which handles model resolution, tool registration, checkpointing, streaming,
and HITL approval. This script only orchestrates the outer loop.

Setup:
    uv venv
    source .venv/bin/activate
    uv pip install deepagents-cli

Usage:
    python ralph_mode.py "Build a Python course. Use git."
    python ralph_mode.py "Build a REST API" --iterations 5
    python ralph_mode.py "Create a CLI tool" --work-dir ./my-project
    python ralph_mode.py "Create a CLI tool" --model claude-sonnet-4-6
    python ralph_mode.py "Build an app" --sandbox modal
    python ralph_mode.py "Build an app" --sandbox modal --sandbox-id my-sandbox
    python ralph_mode.py "Build an app" --shell-allow-list recommended
    python ralph_mode.py "Build an app" --no-stream
    python ralph_mode.py "Build an app" --model-params '{"temperature": 0.5}'
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import warnings
from pathlib import Path
from typing import Any

from deepagents_cli.non_interactive import run_non_interactive
from rich.console import Console

logger = logging.getLogger(__name__)


async def ralph(
    task: str,
    max_iterations: int = 0,
    model_name: str | None = None,
    model_params: dict[str, Any] | None = None,
    sandbox_type: str = "none",
    sandbox_id: str | None = None,
    sandbox_setup: str | None = None,
    *,
    stream: bool = True,
) -> None:
    """Run agent in an autonomous Ralph loop.

    Each iteration invokes the Deep Agents CLI's `run_non_interactive` with a
    fresh thread (the default behavior) while the filesystem persists across
    iterations. This is the core Ralph pattern: fresh context, persistent
    filesystem.

    Uses `Path.cwd()` as the working directory; the caller may optionally
    change the working directory before invoking this coroutine.

    Args:
        task: Declarative description of what to build.
        max_iterations: Maximum number of iterations (0 = unlimited).
        model_name: Model spec in `provider:model` format (e.g.
            `'anthropic:claude-sonnet-4-6'`).

            When `None`, `deepagents-cli` resolves a default via its config
            file (`[models].default`, then `[models].recent`) and falls back
            to auto-detection from environment API keys
            (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`).
        model_params: Additional model parameters (e.g. `{"temperature": 0.5}`).
        sandbox_type: Sandbox provider (`"none"`, `"modal"`, `"daytona"`, etc.).
        sandbox_id: Existing sandbox instance ID to reuse.
        sandbox_setup: Path to a setup script to run inside the sandbox.
        stream: Whether to stream model output.
    """
    work_path = Path.cwd()
    console = Console()

    console.print("\n[bold magenta]Ralph Mode[/bold magenta]")
    console.print(f"[dim]Task: {task}[/dim]")
    iters_label = (
        "unlimited (Ctrl+C to stop)" if max_iterations == 0 else str(max_iterations)
    )
    console.print(f"[dim]Iterations: {iters_label}[/dim]")
    if model_name:
        console.print(f"[dim]Model: {model_name}[/dim]")
    if sandbox_type != "none":
        sandbox_label = sandbox_type
        if sandbox_id:
            sandbox_label += f" (id: {sandbox_id})"
        console.print(f"[dim]Sandbox: {sandbox_label}[/dim]")
    console.print(f"[dim]Working directory: {work_path}[/dim]\n")

    iteration = 1
    try:
        while max_iterations == 0 or iteration <= max_iterations:
            separator = "=" * 60
            console.print(f"\n[bold cyan]{separator}[/bold cyan]")
            console.print(f"[bold cyan]RALPH ITERATION {iteration}[/bold cyan]")
            console.print(f"[bold cyan]{separator}[/bold cyan]\n")

            iter_display = (
                f"{iteration}/{max_iterations}"
                if max_iterations > 0
                else str(iteration)
            )
            prompt = (
                f"## Ralph Iteration {iter_display}\n\n"
                f"Your previous work is in the filesystem. "
                f"Check what exists and keep building.\n\n"
                f"TASK:\n{task}\n\n"
                f"Make progress. You'll be called again."
            )

            exit_code = await run_non_interactive(
                message=prompt,
                assistant_id="ralph",
                model_name=model_name,
                model_params=model_params,
                sandbox_type=sandbox_type,
                sandbox_id=sandbox_id,
                sandbox_setup=sandbox_setup,
                quiet=True,
                stream=stream,
            )

            if exit_code == 130:  # noqa: PLR2004
                break

            if exit_code != 0:
                console.print(
                    f"[bold red]Iteration {iteration} exited with code {exit_code}[/bold red]"
                )

            console.print(f"\n[dim]...continuing to iteration {iteration + 1}[/dim]")
            iteration += 1

    except KeyboardInterrupt:
        console.print(
            f"\n[bold yellow]Stopped after {iteration} iterations[/bold yellow]"
        )

    console.print(f"\n[bold]Files in {work_path}:[/bold]")
    for path in sorted(work_path.rglob("*")):
        if path.is_file() and ".git" not in str(path):
            console.print(f"  {path.relative_to(work_path)}", style="dim")


def main() -> None:
    """Parse CLI arguments and run the Ralph loop."""
    warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality")

    parser = argparse.ArgumentParser(
        description="Ralph Mode - Autonomous looping for Deep Agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ralph_mode.py "Build a Python course. Use git."
  python ralph_mode.py "Build a REST API" --iterations 5
  python ralph_mode.py "Create a CLI tool" --model claude-sonnet-4-6
  python ralph_mode.py "Build a web app" --work-dir ./my-project
  python ralph_mode.py "Build an app" --sandbox modal
  python ralph_mode.py "Build an app" --shell-allow-list recommended
  python ralph_mode.py "Build an app" --model-params '{"temperature": 0.5}'
        """,
    )
    parser.add_argument("task", help="Task to work on (declarative, what you want)")
    parser.add_argument(
        "--iterations",
        type=int,
        default=0,
        help="Max iterations (0 = unlimited, default: unlimited)",
    )
    parser.add_argument("--model", help="Model to use (e.g., claude-sonnet-4-6)")
    parser.add_argument(
        "--work-dir",
        help="Working directory for the agent (default: current directory)",
    )
    parser.add_argument(
        "--model-params",
        help="JSON string of model parameters (e.g., '{\"temperature\": 0.5}')",
    )
    parser.add_argument(
        "--sandbox",
        default="none",
        help="Sandbox provider (e.g., modal, daytona). Default: none",
    )
    parser.add_argument(
        "--sandbox-id",
        help="Existing sandbox instance ID to reuse",
    )
    parser.add_argument(
        "--sandbox-setup",
        help="Path to a setup script to run inside the sandbox",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming output",
    )
    parser.add_argument(
        "--shell-allow-list",
        help=(
            "Comma-separated shell commands to auto-approve, "
            'or "recommended" for safe defaults'
        ),
    )
    args = parser.parse_args()

    if args.work_dir:
        resolved = Path(args.work_dir).resolve()
        resolved.mkdir(parents=True, exist_ok=True)
        os.chdir(resolved)

    if args.shell_allow_list:
        from deepagents_cli.config import parse_shell_allow_list, settings

        settings.shell_allow_list = parse_shell_allow_list(args.shell_allow_list)

    model_params: dict[str, Any] | None = None
    if args.model_params:
        model_params = json.loads(args.model_params)

    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(
            ralph(
                args.task,
                args.iterations,
                args.model,
                model_params=model_params,
                sandbox_type=args.sandbox,
                sandbox_id=args.sandbox_id,
                sandbox_setup=args.sandbox_setup,
                stream=not args.no_stream,
            )
        )


if __name__ == "__main__":
    main()
