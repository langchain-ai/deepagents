"""Swarm example — fan out subagent tasks from inside the REPL.

The agent reads the swarm skill scripts via read_file, understands the
runSwarm pattern, and writes its own eval block that calls tools.task()
in parallel. No skill module is bundled into the REPL.

Usage:
    uv run python swarm_agent.py
    uv run python swarm_agent.py "Summarize notes/a.md, notes/b.md, notes/c.md"
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

SKILLS_DIR = str(Path(__file__).parent / "skills")
DEFAULT_MODEL = "claude-sonnet-4-6"


def _build_agent(model: str) -> object:
    """Build a Deep Agent with the swarm skill.

    Backend: ``CompositeBackend`` routing ``/skills/*`` to a real filesystem
    (so ``SkillsMiddleware`` can scan SKILL.md at build time) and everything
    else to ``StateBackend`` (keeps demo self-cleaning, avoids macOS SIP).

    No ``skills_backend`` is passed to ``CodeInterpreterMiddleware`` — the
    REPL has no bundled skill modules. The agent reads the skill scripts via
    ``read_file`` and synthesizes its own eval code.
    """
    skill_backend = FilesystemBackend(root_dir=SKILLS_DIR, virtual_mode=True)
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
                ptc=["task", "read_file"],
                timeout=None,
            )
        ],
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "task",
        nargs="?",
        default=(
            "Use the swarm skill to run these three tasks in parallel and "
            "report the results: (1) write the number 1 to /tmp_swarm/a, "
            "(2) write the number 2 to /tmp_swarm/b, "
            "(3) write the number 3 to /tmp_swarm/c."
        ),
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    return parser.parse_args()


async def _amain() -> None:
    args = _parse_args()
    agent = _build_agent(args.model)
    # ``ainvoke`` (not ``invoke``) so REPL tool calls stay on the
    # caller's thread. ``ToolNode``'s sync path routes tool calls
    # through a ``ThreadPoolExecutor``, and ``quickjs_rs.Context`` is
    # ``!Send`` — invoking the REPL's ``eval`` tool from a different
    # thread than the one that built the context panics.
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": args.task}]},
    )
    for message in result["messages"]:
        print(f"--- {type(message).__name__} ---")
        print(message.content)


if __name__ == "__main__":
    asyncio.run(_amain())
