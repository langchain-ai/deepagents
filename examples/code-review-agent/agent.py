#!/usr/bin/env python3
import warnings

warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality")

"""
Code Review Agent

An AI code review agent that analyzes code for bugs, security vulnerabilities,
performance issues, and best practice violations.

Configured through:
- AGENTS.md defines review philosophy and output format
- skills/ provides specialized review workflows (general, security, performance)
- subagents.yaml defines the security analyzer subagent

Usage:
    uv run python agent.py ./src/                        # Review a directory
    uv run python agent.py ./src/main.py                 # Review a single file
    uv run python agent.py ./src/ --focus security       # Security-focused review
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner

from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend

# Load environment variables from .env file
load_dotenv()

EXAMPLE_DIR = Path(__file__).parent
console = Console()


# ---------------------------------------------------------------------------
# Subagent loader
# ---------------------------------------------------------------------------


def load_subagents(config_path: Path) -> list:
    """Load subagent definitions from YAML and wire up tools.

    NOTE: This is a custom utility for this example. Unlike `memory` and `skills`,
    deepagents doesn't natively load subagents from files — they're normally
    defined inline in the create_deep_agent() call. We externalize to YAML here
    to keep configuration separate from code.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    subagents = []
    for name, spec in config.items():
        subagent = {
            "name": name,
            "description": spec["description"],
            "system_prompt": spec["system_prompt"],
        }
        if "model" in spec:
            subagent["model"] = spec["model"]
        # Security analyzer uses built-in filesystem tools only — no custom tools needed
        subagents.append(subagent)

    return subagents


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------


def create_code_review_agent(target_dir: str):
    """Create a code review agent rooted at the target directory."""
    model = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.0)

    return create_deep_agent(
        model=model,
        memory=[str(EXAMPLE_DIR / "AGENTS.md")],
        skills=[str(EXAMPLE_DIR / "skills/")],
        tools=[],
        subagents=load_subagents(EXAMPLE_DIR / "subagents.yaml"),
        backend=LocalShellBackend(root_dir=target_dir, inherit_env=True),
    )


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


class ReviewDisplay:
    """Manages the display of agent progress during code review."""

    def __init__(self):
        self.printed_count = 0
        self.current_status = ""
        self.spinner = Spinner("dots", text="Reviewing...")

    def update_status(self, status: str):
        self.current_status = status
        self.spinner = Spinner("dots", text=status)

    def print_message(self, msg):
        """Print a message with formatting."""
        if isinstance(msg, HumanMessage):
            console.print(Panel(str(msg.content), title="Task", border_style="blue"))

        elif isinstance(msg, AIMessage):
            content = msg.content
            if isinstance(content, list):
                text_parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
                content = "\n".join(text_parts)

            if content and content.strip():
                console.print(Panel(Markdown(content), title="Reviewer", border_style="green"))

            if msg.tool_calls:
                for tc in msg.tool_calls:
                    name = tc.get("name", "unknown")
                    args = tc.get("args", {})

                    if name == "task":
                        desc = args.get("description", "analyzing...")
                        console.print(f"  [bold magenta]>> Delegating:[/] {desc[:70]}...")
                        self.update_status(f"Security analysis: {desc[:40]}...")
                    elif name == "ls":
                        path = args.get("path", ".")
                        console.print(f"  [bold cyan]>> Listing:[/] {path}")
                        self.update_status(f"Listing: {path}...")
                    elif name == "read_file":
                        path = args.get("file_path", "file")
                        console.print(f"  [bold yellow]>> Reading:[/] {path}")
                        self.update_status(f"Reading: {Path(path).name}...")
                    elif name == "grep":
                        pattern = args.get("pattern", "")
                        console.print(f"  [bold blue]>> Searching:[/] {pattern[:50]}")
                        self.update_status(f"Searching: {pattern[:30]}...")
                    elif name == "glob":
                        pattern = args.get("pattern", "")
                        console.print(f"  [bold blue]>> Finding files:[/] {pattern}")
                    elif name == "write_file":
                        path = args.get("file_path", "file")
                        console.print(f"  [bold green]>> Writing:[/] {path}")
                    elif name == "write_todos":
                        console.print(f"  [bold white]>> Planning review steps...[/]")
                    elif name == "execute":
                        cmd = args.get("command", "")
                        console.print(f"  [bold white]>> Running:[/] {cmd[:60]}")

        elif isinstance(msg, ToolMessage):
            name = getattr(msg, "name", "")
            if name == "write_file":
                console.print(f"  [green]✓ File written[/]")
            elif name == "task":
                console.print(f"  [green]✓ Security analysis complete[/]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_user_message(args) -> str:
    """Build the user message based on CLI arguments."""
    agent_dir = os.path.dirname(os.path.abspath(__file__))
    report_path = os.path.join(agent_dir, args.output)
    parts = []

    if os.path.isfile(os.path.abspath(args.target)):
        parts.append(f"Review the file at {args.target}.")
        parts.append("Read it and perform a thorough code review.")
    else:
        parts.append("Review the code in the current directory.")
        parts.append("Start by exploring the file structure with ls and glob, then review key files.")

    parts.append(f"\nIMPORTANT: You MUST use the write_file tool to save the full review report to '{report_path}'. Do NOT just print the review — it must be written to disk.")

    if args.focus:
        parts.append(f"\nFocus specifically on {args.focus} review aspects.")

    return "\n".join(parts)


async def main():
    """Run the code review agent with streaming output."""
    parser = argparse.ArgumentParser(
        description="Code Review Agent powered by Deep Agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python agent.py ./src/                        # Review a directory
  python agent.py ./src/main.py                 # Review a single file
  python agent.py ./src/ --focus security       # Security-focused review
        """,
    )
    parser.add_argument(
        "target",
        nargs="?",
        default=".",
        help="File or directory to review (default: current directory)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="review-report.md",
        help="Output file for the review report (default: review-report.md)",
    )
    parser.add_argument(
        "--focus",
        choices=["general", "security", "performance"],
        default=None,
        help="Focus on a specific review type (default: auto-detect)",
    )

    args = parser.parse_args()

    # Determine target directory for the backend
    target_path = os.path.abspath(args.target)
    if os.path.isfile(target_path):
        target_dir = os.path.dirname(target_path)
    elif os.path.isdir(target_path):
        target_dir = target_path
    else:
        console.print(f"[red]Error: {args.target} does not exist[/]")
        sys.exit(1)

    user_msg = build_user_message(args)

    console.print()
    console.print("[bold blue]Code Review Agent[/]")
    console.print(f"[dim]Target: {args.target}[/]")
    if args.focus:
        console.print(f"[dim]Focus: {args.focus}[/]")
    console.print()

    agent = create_code_review_agent(target_dir)
    display = ReviewDisplay()

    final_messages = []
    with Live(display.spinner, console=console, refresh_per_second=10, transient=True) as live:
        async for chunk in agent.astream(
            {"messages": [("user", user_msg)]},
            config={"configurable": {"thread_id": "code-review"}},
            stream_mode="values",
        ):
            if "messages" in chunk:
                final_messages = chunk["messages"]
                if len(final_messages) > display.printed_count:
                    live.stop()
                    for msg in final_messages[display.printed_count :]:
                        display.print_message(msg)
                    display.printed_count = len(final_messages)
                    live.start()
                    live.update(display.spinner)

    console.print()

    # Debug: show what the agent actually produced
    if os.environ.get("DEBUG"):
        console.print(f"\n[dim]DEBUG: Total messages: {len(final_messages)}[/]")
        for i, msg in enumerate(final_messages):
            tool_calls = getattr(msg, "tool_calls", [])
            console.print(
                f"[dim]DEBUG: msg[{i}] type={type(msg).__name__} "
                f"content={repr(msg.content)[:300]} "
                f"tool_calls={tool_calls[:2] if tool_calls else '[]'} "
                f"additional_kwargs={getattr(msg, 'additional_kwargs', {})}"
                f"[/]"
            )

    # Check if the report was actually written — save in the agent's own directory
    agent_dir = os.path.dirname(os.path.abspath(__file__))
    report_path = os.path.join(agent_dir, args.output)
    if os.path.isfile(report_path):
        console.print(f"[bold green]✓ Review complete![/] Report saved to [cyan]{report_path}[/]")
    else:
        console.print("[bold yellow]⚠ Review complete but report was not saved to disk.[/]")
        # Fallback: find the longest AI message content and save it as the report
        best_content = ""
        for msg in reversed(final_messages):
            if isinstance(msg, AIMessage):
                content = msg.content
                if isinstance(content, list):
                    text_parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
                    content = "\n".join(text_parts)
                if content and content.strip() and len(content) > len(best_content):
                    best_content = content
        if best_content:
            with open(report_path, "w") as f:
                f.write(best_content)
            console.print(f"[green]Saved agent output to [cyan]{report_path}[/][/]")
        else:
            console.print("[dim]No AI message content found in output.[/]")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/]")
