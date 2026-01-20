#!/usr/bin/env python3
"""
Ralph Mode - Autonomous looping for DeepAgents

Ralph is an autonomous looping pattern created by Geoff Huntley.
Each loop starts with fresh context. The filesystem and git serve as memory.

Usage:
    uv pip install deepagents-cli
    python ralph_mode.py "Build a Python course. Use git."
    python ralph_mode.py "Build a REST API" --iterations 5
"""
import warnings
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality")

import argparse
import asyncio
import tempfile
from pathlib import Path

from deepagents_cli.agent import create_cli_agent
from deepagents_cli.config import console, COLORS, SessionState, create_model
from deepagents_cli.execution import execute_task
from deepagents_cli.ui import TokenTracker


async def ralph(task: str, max_iterations: int = 0, model_name: str = None):
    """Run agent in Ralph loop with beautiful CLI output."""
    work_dir = tempfile.mkdtemp(prefix="ralph-")
    
    model = create_model(model_name)
    agent, backend = create_cli_agent(
        model=model,
        assistant_id="ralph",
        tools=[],
        auto_approve=True,
    )
    session_state = SessionState(auto_approve=True)
    token_tracker = TokenTracker()
    
    console.print(f"\n[bold {COLORS['primary']}]Ralph Mode[/bold {COLORS['primary']}]")
    console.print(f"[dim]Task: {task}[/dim]")
    console.print(f"[dim]Iterations: {'unlimited (Ctrl+C to stop)' if max_iterations == 0 else max_iterations}[/dim]")
    console.print(f"[dim]Working directory: {work_dir}[/dim]\n")
    
    iteration = 1
    try:
        while max_iterations == 0 or iteration <= max_iterations:
            console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
            console.print(f"[bold cyan]RALPH ITERATION {iteration}[/bold cyan]")
            console.print(f"[bold cyan]{'='*60}[/bold cyan]\n")
            
            iter_display = f"{iteration}/{max_iterations}" if max_iterations > 0 else str(iteration)
            prompt = f"""## Iteration {iter_display}

Your previous work is in the filesystem. Check what exists and keep building.

TASK:
{task}

Make progress. You'll be called again."""

            await execute_task(
                prompt,
                agent,
                "ralph",
                session_state,
                token_tracker,
                backend=backend,
            )
            
            console.print(f"\n[dim]...continuing to iteration {iteration + 1}[/dim]")
            iteration += 1
            
    except KeyboardInterrupt:
        console.print(f"\n[bold yellow]Stopped after {iteration} iterations[/bold yellow]")
    
    # Show created files
    console.print(f"\n[bold]Files created in {work_dir}:[/bold]")
    for f in sorted(Path(work_dir).rglob("*")):
        if f.is_file() and ".git" not in str(f):
            console.print(f"  {f.relative_to(work_dir)}", style="dim")


def main():
    parser = argparse.ArgumentParser(
        description="Ralph Mode - Autonomous looping for DeepAgents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ralph_mode.py "Build a Python course. Use git."
  python ralph_mode.py "Build a REST API" --iterations 5
  python ralph_mode.py "Create a CLI tool" --model claude-haiku-4-5-20251001
        """
    )
    parser.add_argument("task", help="Task to work on (declarative, what you want)")
    parser.add_argument("--iterations", type=int, default=0, help="Max iterations (0 = unlimited, default: unlimited)")
    parser.add_argument("--model", help="Model to use (e.g., claude-haiku-4-5-20251001)")
    args = parser.parse_args()

    try:
        asyncio.run(ralph(args.task, args.iterations, args.model))
    except KeyboardInterrupt:
        pass  # Clean exit on Ctrl+C


if __name__ == "__main__":
    main()
