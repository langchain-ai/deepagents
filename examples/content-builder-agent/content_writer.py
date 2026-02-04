#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality")

"""
Content Builder Agent

A content writer agent configured entirely through files on disk:
- AGENTS.md defines brand voice and style guide
- skills/ provides specialized workflows (blog posts, social media)
- skills/*/scripts/ provides tools bundled with each skill
- subagents handle research and other delegated tasks

Usage:
    uv run python content_writer.py "Write a blog post about AI agents"
    uv run python content_writer.py "Create a LinkedIn post about prompt engineering"
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Literal

import yaml

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

EXAMPLE_DIR = Path(__file__).parent
console = Console()


# Web search tool for the researcher subagent
@tool
def web_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news"] = "general",
) -> dict:
    """Search the web for current information.

    Args:
        query: The search query (be specific and detailed)
        max_results: Number of results to return (default: 5)
        topic: "general" for most queries, "news" for current events

    Returns:
        Search results with titles, URLs, and content excerpts.
    """
    try:
        from tavily import TavilyClient

        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            return {"error": "TAVILY_API_KEY not set"}

        client = TavilyClient(api_key=api_key)
        return client.search(query, max_results=max_results, topic=topic)
    except Exception as e:
        return {"error": f"Search failed: {e}"}


@tool
def generate_cover(prompt: str, slug: str) -> str:
    """Generate a cover image for a blog post.

    Args:
        prompt: Detailed description of the image to generate.
        slug: Blog post slug. Image saves to blogs/<slug>/hero.png
    """
    try:
        from google import genai

        client = genai.Client()
        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=[prompt],
        )

        for part in response.parts:
            if part.inline_data is not None:
                image = part.as_image()
                output_path = EXAMPLE_DIR / "blogs" / slug / "hero.png"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(str(output_path))
                return f"Image saved to {output_path}"

        return "No image generated"
    except Exception as e:
        return f"Error: {e}"


@tool
def generate_social_image(prompt: str, platform: str, slug: str) -> str:
    """Generate an image for a social media post.

    Args:
        prompt: Detailed description of the image to generate.
        platform: Either "linkedin" or "tweets"
        slug: Post slug. Image saves to <platform>/<slug>/image.png
    """
    try:
        from google import genai

        client = genai.Client()
        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=[prompt],
        )

        for part in response.parts:
            if part.inline_data is not None:
                image = part.as_image()
                output_path = EXAMPLE_DIR / platform / slug / "image.png"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(str(output_path))
                return f"Image saved to {output_path}"

        return "No image generated"
    except Exception as e:
        return f"Error: {e}"


def load_subagents(config_path: Path) -> list:
    """Load subagent definitions from YAML and wire up tools.

    NOTE: This is a custom utility for this example. Unlike `memory` and `skills`,
    deepagents doesn't natively load subagents from files - they're normally
    defined inline in the create_deep_agent() call. We externalize to YAML here
    to keep configuration separate from code.
    """
    # Map tool names to actual tool objects
    available_tools = {
        "web_search": web_search,
    }

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
        if "tools" in spec:
            subagent["tools"] = [available_tools[t] for t in spec["tools"]]
        subagents.append(subagent)

    return subagents


def create_content_writer():
    """Create a content writer agent configured by filesystem files."""
    return create_deep_agent(
        memory=["./AGENTS.md"],           # Loaded by MemoryMiddleware
        skills=["./skills/"],             # Loaded by SkillsMiddleware
        tools=[generate_cover, generate_social_image],  # Image generation
        subagents=load_subagents(EXAMPLE_DIR / "subagents.yaml"),  # Custom helper
        backend=FilesystemBackend(root_dir=EXAMPLE_DIR),
    )


class AgentDisplay:
    """Manages the display of agent progress."""

    def __init__(self):
        self.printed_count = 0
        self.current_status = ""
        self.spinner = Spinner("dots", text="Thinking...")

    def update_status(self, status: str):
        self.current_status = status
        self.spinner = Spinner("dots", text=status)

    def print_message(self, msg):
        """Print a message with nice formatting."""
        if isinstance(msg, HumanMessage):
            console.print(Panel(str(msg.content), title="You", border_style="blue"))

        elif isinstance(msg, AIMessage):
            content = msg.content
            if isinstance(content, list):
                text_parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
                content = "\n".join(text_parts)

            if content and content.strip():
                console.print(Panel(Markdown(content), title="Agent", border_style="green"))

            if msg.tool_calls:
                for tc in msg.tool_calls:
                    name = tc.get("name", "unknown")
                    args = tc.get("args", {})

                    if name == "task":
                        desc = args.get("description", "researching...")
                        console.print(f"  [bold magenta]>> Researching:[/] {desc[:60]}...")
                        self.update_status(f"Researching: {desc[:40]}...")
                    elif name in ("generate_cover", "generate_social_image"):
                        console.print(f"  [bold cyan]>> Generating image...[/]")
                        self.update_status("Generating image...")
                    elif name == "write_file":
                        path = args.get("file_path", "file")
                        console.print(f"  [bold yellow]>> Writing:[/] {path}")
                    elif name == "web_search":
                        query = args.get("query", "")
                        console.print(f"  [bold blue]>> Searching:[/] {query[:50]}...")
                        self.update_status(f"Searching: {query[:30]}...")

        elif isinstance(msg, ToolMessage):
            name = getattr(msg, "name", "")
            if name in ("generate_cover", "generate_social_image"):
                if "saved" in msg.content.lower():
                    console.print(f"  [green]✓ Image saved[/]")
                else:
                    console.print(f"  [red]✗ Image failed: {msg.content}[/]")
            elif name == "write_file":
                console.print(f"  [green]✓ File written[/]")
            elif name == "task":
                console.print(f"  [green]✓ Research complete[/]")
            elif name == "web_search":
                if "error" not in msg.content.lower():
                    console.print(f"  [green]✓ Found results[/]")


async def main():
    """Run the content writer agent with streaming output."""
    if len(sys.argv) > 1:
        task = " ".join(sys.argv[1:])
    else:
        task = "Write a blog post about how AI agents are transforming software development"

    console.print()
    console.print("[bold blue]Content Builder Agent[/]")
    console.print(f"[dim]Task: {task}[/]")
    console.print()

    agent = create_content_writer()
    display = AgentDisplay()

    console.print()

    # Use Live display for spinner during waiting periods
    with Live(display.spinner, console=console, refresh_per_second=10, transient=True) as live:
        async for chunk in agent.astream(
            {"messages": [("user", task)]},
            config={"configurable": {"thread_id": "content-writer-demo"}},
            stream_mode="values",
        ):
            if "messages" in chunk:
                messages = chunk["messages"]
                if len(messages) > display.printed_count:
                    # Temporarily stop spinner to print
                    live.stop()
                    for msg in messages[display.printed_count:]:
                        display.print_message(msg)
                    display.printed_count = len(messages)
                    # Resume spinner
                    live.start()
                    live.update(display.spinner)

    console.print()
    console.print("[bold green]✓ Done![/]")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/]")
