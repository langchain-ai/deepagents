"""Ralph Mode - Autonomous looping for Deep Agents.

Ralph is an autonomous looping pattern created by Geoff Huntley
(https://ghuntley.com/ralph/). Each loop starts with fresh context.
The filesystem and git serve as the agent's memory across iterations.

This script uses `deepagents-cli` as its runtime: model resolution, tool
registration, checkpointing, and streaming all come from the CLI's public API
(`deepagents_cli.agent`, `deepagents_cli.config`, etc.).

Setup:
    uv venv
    source .venv/bin/activate
    uv pip install deepagents-cli

Usage:
    python ralph_mode.py "Build a Python course. Use git."
    python ralph_mode.py "Build a REST API" --iterations 5
    python ralph_mode.py "Create a CLI tool" --work-dir ./my-project
    python ralph_mode.py "Create a CLI tool" --model claude-sonnet-4-6
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import sys
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from deepagents_cli.agent import create_cli_agent
from deepagents_cli.config import ModelResult, create_model, settings
from deepagents_cli.file_ops import FileOpTracker
from deepagents_cli.sessions import generate_thread_id, get_checkpointer
from deepagents_cli.tools import fetch_url, http_request, web_search
from langchain.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from rich.console import Console

if TYPE_CHECKING:
    from langgraph.pregel import Pregel

_STREAM_TUPLE_LENGTH = 3
_MESSAGE_TUPLE_LENGTH = 2

console = Console()


async def _stream_iteration(
    agent: Pregel,
    prompt: str,
    config: dict[str, Any],
    file_op_tracker: FileOpTracker,
) -> None:
    """Stream a single agent invocation, printing output to the console.

    Processes the agent's streamed response chunks: text blocks are written
    to stdout as they arrive, tool-call names are printed as dim status
    lines, and file operations are tracked via `FileOpTracker`.

    Args:
        agent: The compiled LangGraph agent to stream.
        prompt: The user message to send to the agent.
        config: LangGraph runnable config (thread ID, metadata).
        file_op_tracker: Tracker for file-operation diffs.
    """
    stream_input: dict[str, list[dict[str, str]]] = {
        "messages": [{"role": "user", "content": prompt}]
    }

    async for chunk in agent.astream(
        stream_input,
        stream_mode=["messages", "updates"],
        subgraphs=True,
        config=config,
        durability="exit",
    ):
        if not isinstance(chunk, tuple) or len(chunk) != _STREAM_TUPLE_LENGTH:
            continue
        namespace, stream_mode, data = chunk
        if namespace:
            continue

        if (
            stream_mode != "messages"
            or not isinstance(data, tuple)
            or len(data) != _MESSAGE_TUPLE_LENGTH
        ):
            continue

        message_obj, metadata = data
        if metadata and metadata.get("lc_source") == "summarization":
            continue

        if isinstance(message_obj, AIMessage) and hasattr(
            message_obj, "content_blocks"
        ):
            for block in message_obj.content_blocks:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type == "text" and block.get("text"):
                    sys.stdout.write(block["text"])
                    sys.stdout.flush()
                elif block_type in {"tool_call_chunk", "tool_call"} and block.get(
                    "name"
                ):
                    console.print(f"\n[dim]🔧 {block['name']}[/dim]")
        elif isinstance(message_obj, ToolMessage):
            file_op_tracker.complete_with_message(message_obj)

    sys.stdout.write("\n")
    sys.stdout.flush()


async def ralph(
    task: str,
    max_iterations: int = 0,
    model_name: str | None = None,
) -> None:
    """Run agent in an autonomous Ralph loop.

    Each iteration creates a fresh agent context (new thread ID, new
    checkpointer state) while the filesystem persists across iterations.
    This is the core Ralph pattern: fresh context, persistent filesystem.

    Uses `deepagents_cli.config.create_model` for model resolution and
    `deepagents_cli.agent.create_cli_agent` to build the underlying LangGraph
    agent with tool registration and auto-approval.

    Uses `Path.cwd()` as the working directory; the caller may optionally
    change the working directory before invoking this coroutine.

    Args:
        task: Declarative description of what to build.
        max_iterations: Maximum number of iterations (0 = unlimited).
        model_name: Model spec in `provider:model` format (e.g.
            `'anthropic:claude-sonnet-4-5'`).

            When `None`, `deepagents-cli` resolves a default via its config file
            (`[models].default`, then `[models].recent`) and falls back to
            auto-detection from environment API keys
            (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`).
    """
    work_path = Path.cwd()

    result: ModelResult = create_model(model_name)
    model: BaseChatModel = result.model
    result.apply_to_settings()

    console.print("\n[bold magenta]Ralph Mode[/bold magenta]")
    console.print(f"[dim]Task: {task}[/dim]")
    iters_label = (
        "unlimited (Ctrl+C to stop)" if max_iterations == 0 else str(max_iterations)
    )
    console.print(f"[dim]Iterations: {iters_label}[/dim]")
    if settings.model_name:
        console.print(f"[dim]Model: {settings.model_name}[/dim]")
    console.print(f"[dim]Working directory: {work_path}[/dim]\n")

    iteration = 1
    try:
        while max_iterations == 0 or iteration <= max_iterations:
            separator = "=" * 60
            console.print(f"\n[bold cyan]{separator}[/bold cyan]")
            console.print(f"[bold cyan]RALPH ITERATION {iteration}[/bold cyan]")
            console.print(f"[bold cyan]{separator}[/bold cyan]\n")

            thread_id = generate_thread_id()

            async with get_checkpointer() as checkpointer:
                tools: list[Any] = [http_request, fetch_url]
                if settings.has_tavily:
                    tools.append(web_search)

                agent, backend = create_cli_agent(
                    model=model,
                    assistant_id="ralph",
                    tools=tools,
                    auto_approve=True,
                    checkpointer=checkpointer,
                )

                file_op_tracker = FileOpTracker(assistant_id="ralph", backend=backend)

                config: dict[str, Any] = {
                    "configurable": {"thread_id": thread_id},
                    "metadata": {
                        "assistant_id": "ralph",
                        "agent_name": "ralph",
                        "updated_at": datetime.now(UTC).isoformat(),
                    },
                }

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

                await _stream_iteration(agent, prompt, config, file_op_tracker)

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
    args = parser.parse_args()

    if args.work_dir:
        resolved = Path(args.work_dir).resolve()
        resolved.mkdir(parents=True, exist_ok=True)
        os.chdir(resolved)

    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(ralph(args.task, args.iterations, args.model))


if __name__ == "__main__":
    main()
