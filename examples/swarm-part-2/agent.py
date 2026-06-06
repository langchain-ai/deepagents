"""Swarm part 2 example with URL-based company research subagents.

The default task expects `/companies.txt` to exist in the example's `output/`
backend root, with one `name,url` pair per line.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import httpx
from deepagents import create_deep_agent
from deepagents.backends.composite import CompositeBackend
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.filesystem import FilesystemMiddleware
from langchain_core.tools import BaseTool, tool
from langchain_quickjs import (
    CodeInterpreterMiddleware,
    SwarmSubAgent,
    create_swarm_task_tool,
)
from markdownify import markdownify

THIS_DIR = Path(__file__).parent
DEFAULT_MODEL = "claude-sonnet-4-6"
MAX_WEBPAGE_CHARS = 20_000
DEFAULT_TASK = (
    "Use the swarm skill to research companies listed in `/companies.txt` "
    "(this file lives under the example `output/` directory). Inside eval: "
    "read `/companies.txt`, split non-empty lines, and build a tasks array "
    "of `{ id: `c_${i}`, company: <name>, url: <url> }` from CSV lines "
    "formatted `name,url`. Run swarm with `subagentType: 'researcher'` and "
    "instruction `Fetch {url} and research {company}. Return latest business "
    "summary, primary product lines, HQ location, and 2-4 credible source URLs.` "
    "Use responseSchema with required fields: `company` (string), `summary` "
    "(string), `products` (array of strings), `hq` (string), `sources` "
    "(array of strings). Return run summary plus aggregated stats."
)


def _truncate(text: str, max_chars: int = MAX_WEBPAGE_CHARS) -> str:
    """Clamp long tool output so each row stays within a reasonable size.

    Args:
        text: Source text to truncate.
        max_chars: Maximum number of characters to keep.

    Returns:
        Original text when within the limit, otherwise a truncated variant.
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[truncated]"


@tool(parse_docstring=True)
def fetch_webpage(url: str, timeout_seconds: float = 15.0) -> str:
    """Fetch a webpage and return markdown content.

    Args:
        url: Absolute URL to fetch.
        timeout_seconds: HTTP timeout for the request.

    Returns:
        Markdown content extracted from the HTML response.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }
    try:
        response = httpx.get(url, headers=headers, timeout=timeout_seconds)
        response.raise_for_status()
    except Exception as exc:
        msg = f"Failed to fetch {url}: {exc}"
        return msg

    markdown = markdownify(response.text)
    return _truncate(markdown)


def _build_subagents(filesystem_tools: dict[str, BaseTool]) -> list[SwarmSubAgent]:
    """Build swarm subagent specs for local review and URL-driven research.

    Args:
        filesystem_tools: Built-in filesystem tools keyed by tool name.

    Returns:
        Configured swarm subagents for `create_swarm_task_tool`.
    """
    return [
        SwarmSubAgent(
            name="reviewer",
            description="Reviews local files and returns concise structured analysis.",
            system_prompt=(
                "You are a careful reviewer. Use tools when needed. "
                "Follow the requested response schema exactly and return only "
                "the requested fields."
            ),
            tools=[
                filesystem_tools["read_file"],
                filesystem_tools["glob"],
                filesystem_tools["write_file"],
                filesystem_tools["edit_file"],
            ],
        ),
        SwarmSubAgent(
            name="researcher",
            description=(
                "Researches companies from provided URLs and returns source-backed structured output."
            ),
            system_prompt=(
                "You are a web researcher. Use the provided URL first, then any "
                "additional URLs explicitly included in the task row. Return "
                "schema-compliant output only."
            ),
            tools=[
                fetch_webpage,
                filesystem_tools["read_file"],
                filesystem_tools["write_file"],
                filesystem_tools["edit_file"],
            ],
        ),
    ]


def _build_agent(model: str) -> object:
    """Construct the Deep Agent graph for this example.

    The graph uses a composite backend that routes `/skills/` to the local
    skills filesystem while keeping all other reads/writes in `output/`.

    Args:
        model: Model identifier passed to Deep Agents and swarm dispatch.

    Returns:
        Compiled Deep Agent runnable.
    """
    output_backend = FilesystemBackend(root_dir=str(THIS_DIR / "output"))
    skill_backend = FilesystemBackend(
        root_dir=str(THIS_DIR / "skills"), virtual_mode=True
    )
    backend = CompositeBackend(
        default=output_backend,
        routes={"/skills/": skill_backend},
    )
    filesystem_tools = {
        tool.name: tool
        for tool in FilesystemMiddleware(backend=backend).tools
    }
    swarm_tool = create_swarm_task_tool(
        subagents=_build_subagents(filesystem_tools),
        default_model=model,
    )
    return create_deep_agent(
        model=model,
        backend=backend,
        skills=["/skills/"],
        middleware=[
            CodeInterpreterMiddleware(
                ptc=[
                    swarm_tool,
                    "read_file",
                    "write_file",
                    "edit_file",
                    "glob",
                    "execute",
                ],
                skills_backend=backend,
                timeout=None,
            )
        ],
    )


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for this example script.

    Returns:
        Parsed command-line namespace.
    """
    parser = argparse.ArgumentParser(description="Swarm part 2 example agent")
    parser.add_argument("task", nargs="?", default=DEFAULT_TASK)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    return parser.parse_args()


async def _amain() -> None:
    """Run one agent invocation and print each emitted message block."""
    args = _parse_args()
    agent = _build_agent(args.model)
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": args.task}]},
    )
    for message in result["messages"]:
        print(f"--- {type(message).__name__} ---")
        print(message.content)


if __name__ == "__main__":
    asyncio.run(_amain())
