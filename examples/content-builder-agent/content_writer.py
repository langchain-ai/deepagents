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


async def _expand_query(query: str) -> list[str]:
    """Expand a single query into 3 diverse sub-queries using gpt-4o-mini.

    Args:
        query: The original search query to expand.

    Returns:
        A list of 3 sub-queries, or `[query]` on any failure.
    """
    import json

    try:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        system = (
            "You are a search query expander. Given a topic, return a JSON array of exactly 3 "
            "search queries that are diverse and complementary to the original. Each query "
            "should target a different angle (e.g., definition/overview, recent developments, "
            "practical examples). Output only a raw JSON array of strings — no markdown, no commentary."
        )
        response = await llm.ainvoke([("system", system), ("human", query)])
        return json.loads(response.content)
    except Exception:
        return [query]


async def _search_tavily(query: str, max_results: int, topic: str) -> list[dict]:
    """Search via Tavily and normalise results.

    Args:
        query: Search query string.
        max_results: Maximum number of results to return.
        topic: Either "general" or "news".

    Returns:
        List of result dicts with keys url, title, content, source.
    """
    try:
        from tavily import TavilyClient

        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            return []
        client = TavilyClient(api_key=api_key)
        raw = await asyncio.to_thread(
            client.search, query, max_results=max_results, topic=topic
        )
        return [
            {
                "url": r["url"],
                "title": r.get("title", ""),
                "content": r.get("content", ""),
                "source": "tavily",
            }
            for r in raw.get("results", [])
        ]
    except Exception:
        return []


async def _search_exa(query: str, max_results: int) -> list[dict]:
    """Search via Exa and normalise results.

    Args:
        query: Search query string.
        max_results: Maximum number of results to return.

    Returns:
        List of result dicts with keys url, title, content, source.
    """
    try:
        from exa_py import AsyncExa

        api_key = os.environ.get("EXA_API_KEY")
        if not api_key:
            return []
        exa = AsyncExa(api_key=api_key)
        response = await exa.search(
            query,
            num_results=max_results,
            type="auto",
            contents={"highlights": True},
        )
        results = []
        for r in response.results:
            highlights = " ".join(r.highlights or []) if hasattr(r, "highlights") and r.highlights else ""
            results.append(
                {
                    "url": r.url,
                    "title": r.title or "",
                    "content": highlights or getattr(r, "text", ""),
                    "source": "exa",
                }
            )
        return results
    except Exception:
        return []


async def _search_ddg(query: str, max_results: int) -> list[dict]:
    """Search via DuckDuckGo and normalise results.

    Args:
        query: Search query string.
        max_results: Maximum number of results to return.

    Returns:
        List of result dicts with keys url, title, content, source.
    """
    try:
        from duckduckgo_search import DDGS

        raw = await asyncio.to_thread(DDGS().text, query, max_results=max_results)
        return [
            {
                "url": r["href"],
                "title": r.get("title", ""),
                "content": r.get("body", ""),
                "source": "duckduckgo",
            }
            for r in (raw or [])
        ]
    except Exception:
        return []


def _auto_select_providers() -> list[str]:
    """Select up to 2 available search providers based on env vars.

    Returns:
        List of up to 2 provider names in priority order: tavily, exa, duckduckgo.
    """
    available = []
    if os.environ.get("TAVILY_API_KEY"):
        available.append("tavily")
    if os.environ.get("EXA_API_KEY"):
        available.append("exa")
    available.append("duckduckgo")
    return available[:2]


@tool
async def web_search_multi(
    queries: list[str],
    providers: list[str] | None = None,
    max_results: int = 5,
    topic: Literal["general", "news"] = "general",
) -> dict:
    """Search multiple providers concurrently and return merged, deduplicated results.

    Args:
        queries: One or more search query strings to run.
        providers: Provider names to use. Auto-selected if not specified.
        max_results: Results per (query, provider) pair.
        topic: "general" for most queries, "news" for current events.

    Returns:
        Dict with keys: results (list), query_count (int), provider_count (int).
    """
    if providers is None:
        providers = _auto_select_providers()

    dispatch = {
        "tavily": _search_tavily,
        "exa": _search_exa,
        "duckduckgo": _search_ddg,
    }

    async def _run(query: str, provider: str) -> list[dict]:
        try:
            fn = dispatch[provider]
            if provider == "tavily":
                return await fn(query, max_results, topic)
            return await fn(query, max_results)
        except Exception:
            return []

    tasks = [_run(q, p) for q in queries for p in providers if p in dispatch]
    batches = await asyncio.gather(*tasks)

    seen_urls: set[str] = set()
    merged: list[dict] = []
    for batch in batches:
        for result in batch:
            url = result.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                merged.append(result)

    return {
        "results": merged,
        "query_count": len(queries),
        "provider_count": len(providers),
    }


@tool
async def web_search_auto(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news"] = "general",
) -> dict:
    """Search the web with automatic query expansion and multi-provider fan-out.

    Args:
        query: The original search query.
        max_results: Results per (query, provider) pair.
        topic: "general" for most queries, "news" for current events.

    Returns:
        Merged, deduplicated results with keys: results, query_count, provider_count,
        original_query, expanded_queries.
    """
    sub_queries = await _expand_query(query)
    all_queries = [query] + sub_queries

    result = await web_search_multi.ainvoke(
        {
            "queries": all_queries,
            "max_results": max_results,
            "topic": topic,
        }
    )
    result["original_query"] = query
    result["expanded_queries"] = sub_queries
    return result


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
        "web_search": web_search_auto,
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
                    elif name in ("web_search_auto", "web_search_multi"):
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
