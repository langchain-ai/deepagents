"""Swarm example — fan out subagent tasks from inside the REPL.

This example bundles a single skill, `swarm`, that ships a TypeScript
entrypoint. When the agent runs `await import("@/skills/swarm")`, the
REPL middleware installs the skill as an ES module and the model can
call `create(...)` / `run(...)` to dispatch many subagent or one-shot
LLM calls in parallel with bounded concurrency.

Nothing in the Python driver here is swarm-specific — the whole pattern
lives inside `skills/swarm/scripts/index.ts`. That's the point: a skill is just
code the agent pulls in when it needs it.

Usage:
    uv run python swarm_agent.py
    uv run python swarm_agent.py --preset sentiment-classification
    uv run python swarm_agent.py "Summarize notes/a.md, notes/b.md, notes/c.md"
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from textwrap import dedent

from deepagents import create_deep_agent
from deepagents.backends.composite import CompositeBackend
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.state import StateBackend
from langchain_quickjs import CodeInterpreterMiddleware

SKILLS_DIR = str(Path(__file__).parent / "skills")
DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_PRESET = "demo"

TASK_PRESETS: dict[str, str] = {
    "demo": (
        "Use the swarm skill to run these three tasks in parallel and "
        "report the results: (1) write the number 1 to /tmp_swarm/a, "
        "(2) write the number 2 to /tmp_swarm/b, "
        "(3) write the number 3 to /tmp_swarm/c."
    ),
    "sentiment-classification": dedent(
        """
        Classify the sentiment of these customer reviews using swarm.

        Reviews:
        1. "Love the new update, easy fix for the login bug!"
        2. "Works fine, nothing special."
        3. "Crashes every time I open it. Uninstalling."
        4. "Pretty good overall, a few rough edges."
        5. "Absolutely brilliant, ship it!"
        6. "Worst experience I've ever had with an app."
        7. "Decent but needs more features."
        8. "The team clearly cares about quality."

        Create a swarm table with these reviews, classify each as
        positive/negative/neutral with a confidence score, then tell me
        the distribution and average confidence.
        """
    ).strip(),
    "code-review": dedent(
        """
        Review the TypeScript files in /skills/swarm/scripts/ for real bugs
        using swarm.

        These files implement a table-based swarm orchestration library.
        Focus on issues that could cause incorrect dispatch behavior, data
        loss, or resource leaks in a long-running process.

        Use a reviewer subagent with structured output and summarize the
        top findings by severity.
        """
    ).strip(),
    "review-verify-filter": dedent(
        """
        Do a thorough two-pass bug audit of /skills/swarm/scripts/**/*.ts
        using swarm.

        Pass 1: dispatch each file to a bug-finder subagent for initial findings.
        Pass 2: flatten findings into a new table and dispatch each to a verifier
        subagent that independently confirms or rejects it.

        Report only confirmed findings and include:
        - file path
        - severity
        - verification reasoning
        """
    ).strip(),
}


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
            CodeInterpreterMiddleware(
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
        default=None,
        help="Optional free-form task. Overrides `--preset` when provided.",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(TASK_PRESETS),
        default=DEFAULT_PRESET,
        help="Use a built-in task preset.",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="Print available presets and exit.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    return parser.parse_args()


async def _amain() -> None:
    args = _parse_args()
    if args.list_presets:
        print("Available presets:")
        for name in sorted(TASK_PRESETS):
            print(f"  - {name}")
        return
    task = args.task if args.task is not None else TASK_PRESETS[args.preset]
    agent = _build_agent(args.model)
    # ``ainvoke`` (not ``invoke``) so REPL tool calls stay on the
    # caller's thread. ``ToolNode``'s sync path routes tool calls
    # through a ``ThreadPoolExecutor``, and ``quickjs_rs.Context`` is
    # ``!Send`` — invoking the REPL's ``eval`` tool from a different
    # thread than the one that built the context panics.
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": task}]},
    )
    for message in result["messages"]:
        print(f"--- {type(message).__name__} ---")
        print(message.content)


if __name__ == "__main__":
    asyncio.run(_amain())
