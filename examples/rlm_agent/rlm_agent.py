"""Recursive REPL Mode — recursion over `create_deep_agent` + REPL PTC.

`create_rlm_agent` is a thin wrapper over `create_deep_agent` that
attaches a `REPLMiddleware` (with programmatic tool calling enabled)
at every level of a nested subagent chain, down to a caller-chosen
``max_depth``. Each level exposes the parent's tool set inside the
REPL as ``tools.<name>``, so one ``eval`` call can orchestrate a
whole fan-out of sub-calls — including ``tools.task(...)`` to dispatch
to the next-deeper level.

The pattern is useful when:

- A task decomposes into independent subtasks that the top-level
  agent could plan in one shot if it could write code. The REPL at
  depth 0 turns that planning into a concrete ``Promise.all``.
- Those subtasks themselves decompose further. Each recursive subagent
  gets its own REPL + PTC, so the decomposition can continue without
  bespoke middleware wiring at every level.

Usage:
    from rlm_agent import create_rlm_agent

    agent = create_rlm_agent(
        model="claude-sonnet-4-6",
        tools=[...],
        max_depth=2,
    )
    result = agent.invoke({"messages": [{"role": "user", "content": "..."}]})

Run this file directly for a toy demo that adds two numbers via a
tool, asks the agent to compute sums in parallel, and prints the trace.
"""

from __future__ import annotations

import argparse
from typing import Any

from deepagents import create_deep_agent
from deepagents.middleware.subagents import CompiledSubAgent, SubAgent
from langchain_core.tools import BaseTool, tool

from deepagents_repl import REPLMiddleware

_MAX_DEPTH_LIMIT = 8  # guard against typos that would build thousands of agents


def create_rlm_agent(
    *,
    model: str | None = None,
    tools: list[BaseTool] | None = None,
    subagents: list[SubAgent | CompiledSubAgent] | None = None,
    max_depth: int = 1,
    system_prompt: str | None = None,
    **kwargs: Any,
) -> Any:
    """Build a Deep Agent with a recursive REPL-backed subagent chain.

    At each depth, the returned agent has:

    - Whatever `tools`, `subagents`, `system_prompt`, and other
      `create_deep_agent` kwargs the caller passed.
    - A `REPLMiddleware(ptc=True)` so the model can write
      `await tools.<name>(...)` inside the `eval` tool to orchestrate
      the parent's tool set.
    - A synthetic `recursive` subagent that points at a depth-N-1
      copy of this same construction — until `max_depth` hits 0, at
      which point the subagent chain ends (the leaf still has a REPL
      but no recursive child).

    Every level shares the caller's `tools` and `subagents` so the
    recursion is pure — there's no "this tool is only at depth 2"
    surprise. To tailor the tool set per depth, wire it in yourself
    via the `subagents` kwarg; `create_rlm_agent` is deliberately
    unopinionated past the recursion itself.

    Args:
        model: Passed through to `create_deep_agent`. Default
            (`None`) inherits whatever `create_deep_agent` defaults
            to — today, claude-sonnet-4-6.
        tools: Additional tools available at every level. Merged with
            the deep-agent built-ins.
        subagents: Extra subagents available at every level, merged
            alongside the `recursive` subagent this helper injects.
        max_depth: How many levels of nested recursion to build. `0`
            means "no recursive subagent" — the returned agent still
            has REPL + PTC, but cannot delegate to a deeper copy.
            Capped at `_MAX_DEPTH_LIMIT` to prevent accidental
            explosion.
        system_prompt: Custom system prompt. Appended to the default.
        **kwargs: Forwarded to `create_deep_agent` unchanged.

    Returns:
        A compiled Deep Agent graph.
    """
    if max_depth < 0:
        msg = "max_depth must be >= 0"
        raise ValueError(msg)
    if max_depth > _MAX_DEPTH_LIMIT:
        msg = f"max_depth {max_depth} exceeds safety cap {_MAX_DEPTH_LIMIT}"
        raise ValueError(msg)

    merged_subagents: list[SubAgent | CompiledSubAgent] = list(subagents or [])
    if max_depth > 0:
        # Build the deeper agent first and plug it in as a
        # CompiledSubAgent so we don't need to re-plumb tools and
        # middleware at the SubAgent (TypedDict) layer.
        deeper = create_rlm_agent(
            model=model,
            tools=tools,
            subagents=subagents,
            max_depth=max_depth - 1,
            system_prompt=system_prompt,
            **kwargs,
        )
        merged_subagents.append(
            CompiledSubAgent(
                name="recursive",
                description=(
                    "Delegate a subtask to a deeper recursive agent. The "
                    "deeper agent has the same tools and its own `eval` "
                    "REPL; use it when a subtask itself decomposes into "
                    "independent parallel steps."
                ),
                runnable=deeper,
            )
        )

    return create_deep_agent(
        model=model,
        tools=tools,
        subagents=merged_subagents,
        system_prompt=system_prompt,
        middleware=[REPLMiddleware(ptc=True)],
        **kwargs,
    )


# ---- demo driver --------------------------------------------------


@tool
def add(a: int, b: int) -> int:
    """Add two integers and return their sum."""
    return a + b


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "task",
        nargs="?",
        default=(
            "Use the eval REPL to compute, in parallel, "
            "add(1,2) + add(3,4) + add(5,6) + add(7,8). "
            "Return the final grand total."
        ),
    )
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--model", default=None)
    return parser.parse_args()


def _main() -> None:
    args = _parse_args()
    agent = create_rlm_agent(
        model=args.model,
        tools=[add],
        max_depth=args.max_depth,
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": args.task}]},
    )
    for message in result["messages"]:
        print(f"--- {type(message).__name__} ---")
        print(message.content)


if __name__ == "__main__":
    _main()
