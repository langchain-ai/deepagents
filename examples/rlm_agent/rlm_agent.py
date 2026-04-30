"""Recursive Agents (RLM style).

`create_deep_agent` with a compiled general-purpose chain.

`create_rlm_agent` builds a Deep Agent whose own middleware stack
includes `REPLMiddleware(ptc=[...])`, and — for `max_depth > 0` —
replaces the default `general-purpose` subagent with a
`CompiledSubAgent` whose runnable is itself a depth-(N-1) RLM agent.
The model delegates via `tools.task({subagent_type: "general-purpose",
...})` and reaches the deeper compiled agent, which has its own REPL
and its own compiled `general-purpose`, and so on until `max_depth`
bottoms out. At the leaf, `general-purpose` is the plain built-in —
no further recursion.

The pattern is useful when a task decomposes into independent
sub-tasks whose decomposition you can't predict in advance. At each
level the agent can parallelize with `Promise.all(tools.<x>(...))`
inside `eval`, or fan out via `tools.task(subagent_type="general-purpose",
...)` to a fully realized deeper agent whose own REPL can fan out again.

Usage:
    from rlm_agent import create_rlm_agent

    agent = create_rlm_agent(
        model="claude-sonnet-4-6",
        tools=[...],
        max_depth=2,
    )
    result = agent.invoke({"messages": [{"role": "user", "content": "..."}]})

Run this file directly for a toy demo that asks the agent to compute
several sums in parallel via the REPL.
"""

from __future__ import annotations

import argparse
from typing import Any

from deepagents import create_deep_agent
from deepagents.middleware.subagents import (
    GENERAL_PURPOSE_SUBAGENT,
    CompiledSubAgent,
    SubAgent,
)
from langchain_core.tools import BaseTool, tool
from langchain_quickjs import REPLMiddleware

_MAX_DEPTH_LIMIT = 8  # guard against typos that would build thousands of agents


def create_rlm_agent(
    *,
    model: str | None = None,
    tools: list[BaseTool] | None = None,
    subagents: list[SubAgent | CompiledSubAgent] | None = None,
    max_depth: int = 1,
    **kwargs: Any,
) -> Any:
    """Build a Deep Agent with a recursive compiled general-purpose chain.

    At every depth the agent itself has PTC configured with an explicit
    include-list (`task` plus user-provided tool names), so the model
    can write `eval` + `Promise.all(tools.<x>(...))` to parallelize.
    For `max_depth > 0` the default `general-purpose` subagent is
    replaced with a `CompiledSubAgent` pointing at a depth-(N-1) RLM
    agent. The model reaches it the usual way:
    `tools.task({subagent_type: "general-purpose", ...})`.

    Args:
        model: Passed through to `create_deep_agent`.
        tools: Tools available at every level, merged with the
            deep-agent built-ins.
        subagents: Extra subagents. Must NOT contain a spec named
            `general-purpose` — this helper manages that name. Pass
            other subagent specs freely; they're carried through to
            every depth unchanged.
        max_depth: How many levels of recursion to build. `0` means
            "no recursion" — the returned agent still has REPL + PTC,
            but its `general-purpose` is the unmodified built-in.
            Capped at `_MAX_DEPTH_LIMIT`.
        **kwargs: Forwarded to `create_deep_agent`.

    Returns:
        A compiled Deep Agent graph.

    Raises:
        ValueError: If `max_depth` is negative, above the cap, or if
            `subagents` contains an entry named `general-purpose`.
    """
    if max_depth < 0:
        msg = "max_depth must be >= 0"
        raise ValueError(msg)
    if max_depth > _MAX_DEPTH_LIMIT:
        msg = f"max_depth {max_depth} exceeds safety cap {_MAX_DEPTH_LIMIT}"
        raise ValueError(msg)
    for spec in subagents or []:
        if spec.get("name") == GENERAL_PURPOSE_SUBAGENT["name"]:
            msg = (
                "create_rlm_agent manages the `general-purpose` subagent "
                "itself; do not pass one via `subagents`."
            )
            raise ValueError(msg)

    return _build(
        model=model,
        tools=tools,
        extra_subagents=list(subagents or []),
        max_depth=max_depth,
        **kwargs,
    )


def _build(
    *,
    model: str | None,
    tools: list[BaseTool] | None,
    extra_subagents: list[SubAgent | CompiledSubAgent],
    max_depth: int,
    **kwargs: Any,
) -> Any:
    """Recursive builder. One call compiles one level of the chain.

    At depth 0, the general-purpose subagent is left to the default
    `create_deep_agent` auto-injection — no override, no REPL on it,
    no deeper peer. At depth N > 0, we build the depth-(N-1) agent
    first and register it as a `CompiledSubAgent` under the name
    `general-purpose`, which satisfies the "already has a
    general-purpose" check in `create_deep_agent` and suppresses the
    auto-injected default at this level.
    """
    if max_depth == 0:
        ptc_tool_names = sorted({*(t.name for t in tools or []), "task"})
        return create_deep_agent(
            model=model,
            tools=tools,
            subagents=extra_subagents,
            middleware=[REPLMiddleware(ptc=ptc_tool_names)],
            **kwargs,
        )

    deeper = _build(
        model=model,
        tools=tools,
        extra_subagents=extra_subagents,
        max_depth=max_depth - 1,
        **kwargs,
    )
    compiled_gp = CompiledSubAgent(
        name=GENERAL_PURPOSE_SUBAGENT["name"],
        description=GENERAL_PURPOSE_SUBAGENT["description"],
        runnable=deeper,
    )
    ptc_tool_names = sorted({*(t.name for t in tools or []), "task"})
    return create_deep_agent(
        model=model,
        tools=tools,
        subagents=[compiled_gp, *extra_subagents],
        middleware=[REPLMiddleware(ptc=ptc_tool_names)],
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
    parser.add_argument("--model", default="anthropic:claude-haiku-4-5")
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
