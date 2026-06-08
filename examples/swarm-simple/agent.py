"""Swarm simple example.

A minimal swarm agent that reads skill scripts from disk and synthesizes
its own eval code — no TS bundling, no files attached to the REPL.

The agent:
1. Sees the swarm skill in its system prompt.
2. Reads SKILL.md, scripts/table.ts, and scripts/executor.ts via read_file.
3. Writes a single eval block that defines the helpers inline and dispatches
   work in parallel via tools.task().
4. Collects and returns results.

No Python-side subagent wiring is needed — tools.task is a built-in PTC tool.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends.composite import CompositeBackend
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.state import StateBackend
from langchain_quickjs import CodeInterpreterMiddleware

THIS_DIR = Path(__file__).parent
DEFAULT_MODEL = "claude-sonnet-4-5"
DEFAULT_TASK = (
    "Use the swarm skill to classify the sentiment of these customer reviews in parallel: "
    "[{id: 'r1', text: 'Absolutely love this product!'}, "
    "{id: 'r2', text: 'Terrible experience, would not recommend.'}, "
    "{id: 'r3', text: 'It was okay, nothing special.'}, "
    "{id: 'r4', text: 'Best purchase I have made all year!'}, "
    "{id: 'r5', text: 'Product broke after one week.'}]. "
    "Ask each subagent to return JSON with a 'sentiment' field (positive/negative/neutral). "
    "Print the results."
)


def _build_agent(model: str) -> object:
    skill_backend = FilesystemBackend(root_dir=str(THIS_DIR / "skills"), virtual_mode=True)
    backend = CompositeBackend(
        default=StateBackend(),
        routes={"/skills/": skill_backend},
    )
    return create_deep_agent(
        model=model,
        backend=backend,
        skills=["/skills/"],
        middleware=[
            CodeInterpreterMiddleware(
                ptc=["task", "read_file", "write_file", "glob"],
                # No skills_backend — the REPL cannot `await import("@/skills/swarm")`.
                # The agent reads the skill scripts via read_file and synthesizes a single
                # eval block that defines the helpers inline and calls tools.task() directly.
                timeout=None,
            )
        ],
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Swarm simple example agent")
    parser.add_argument("task", nargs="?", default=DEFAULT_TASK)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    return parser.parse_args()


async def _amain() -> None:
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
