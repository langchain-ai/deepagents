"""Swarm example — fan out subagent tasks from inside the REPL.

This example bundles a single skill, `swarm`, that ships a TypeScript
entrypoint. When the agent runs `await import("@/skills/swarm")`, the
REPL middleware installs the skill as an ES module and the model can
call `runSwarm({ tasks: [...] })` to dispatch many `tools.task(...)`
calls in parallel with bounded concurrency.

Nothing in the Python driver here is swarm-specific — the whole pattern
lives inside `skills/swarm/index.ts`. That's the point: a skill is just
code the agent pulls in when it needs it.

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
from langchain_quickjs import REPLMiddleware

SKILLS_DIR = str(Path(__file__).parent / "skills")
DEFAULT_MODEL = "claude-sonnet-4-6"


def _build_agent(model: str) -> object:
    """Build a Deep Agent with a subagent + the swarm skill.

    Backend shape: ``CompositeBackend`` that sends ``/skills/*`` to a
    ``FilesystemBackend`` rooted at this example's ``skills/`` dir, and
    everything else to ``StateBackend``. Task files the model writes
    (``/tmp_swarm/a`` in the default demo) land in agent state rather
    than on the host — avoiding SIP/read-only-root issues on macOS and
    keeping the demo self-cleaning across runs. The skills backend is
    still a real filesystem because ``SkillsMiddleware`` scans for
    SKILL.md at agent-build time, before any graph state exists.
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
            REPLMiddleware(
                ptc=["task"],
                skills_backend=backend,
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
