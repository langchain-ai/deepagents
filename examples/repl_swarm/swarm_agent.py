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
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend

from deepagents_repl import REPLMiddleware

SKILLS_ROOT = str(Path(__file__).parent / "skills")


def _build_agent(model: str | None) -> object:
    """Build a Deep Agent with a subagent + the swarm skill."""
    backend = FilesystemBackend(root_dir=str(Path(__file__).parent), virtual_mode=False)
    return create_deep_agent(
        model=model,
        backend=backend,
        skills=[SKILLS_ROOT],
        middleware=[REPLMiddleware(ptc=True, skills_backend=backend)],
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
    parser.add_argument("--model", default=None)
    return parser.parse_args()


def _main() -> None:
    args = _parse_args()
    agent = _build_agent(args.model)
    result = agent.invoke(
        {"messages": [{"role": "user", "content": args.task}]},
    )
    for message in result["messages"]:
        print(f"--- {type(message).__name__} ---")
        print(message.content)


if __name__ == "__main__":
    _main()
