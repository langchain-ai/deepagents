"""Recursive REPL Mode (RLM) — `create_deep_agent` + a self-recursive general-purpose subagent.

`create_rlm_agent` builds a Deep Agent whose `general-purpose`
subagent has `REPLMiddleware(ptc=True)` attached. That subagent can
delegate a sub-task to a `deeper-agent` peer — a structurally
separate compiled agent one level shallower, built by this helper
itself — via `tools.task(subagent_type="deeper-agent", ...)` inside
`eval`. Each `deeper-agent` has the same shape and its own
`deeper-agent` peer, until `max_depth` bottoms out. The leaf still
has REPL + PTC but no deeper peer.

The pattern is useful when a task decomposes into independent
sub-tasks whose decomposition you can't predict in advance. At each
level the agent can choose: do the work inline, parallelize with
`Promise.all(tools.<x>(...))` inside `eval`, or fan out via
`tools.task(subagent_type="deeper-agent", ...)` to a peer whose own
REPL can fan out again.

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

from deepagents_repl import REPLMiddleware

_MAX_DEPTH_LIMIT = 8  # guard against typos that would build thousands of agents

_BASE_SYSTEM_ADDENDUM = (
    "\n\n"
    "You are running in Recursive REPL Mode. You have an `eval` tool that "
    "runs JavaScript and a programmatic tool-calling layer that exposes "
    "agent tools under `tools.<name>(...)`. Use `eval` for any work that "
    "benefits from parallel fan-out: write one script that issues many "
    "`tools.<x>(...)` calls inside `Promise.all(...)` instead of making "
    "the same tool calls serially across turns."
)

_RECURSE_SYSTEM_ADDENDUM = (
    "\n\n"
    "You have {remaining} level(s) of recursion budget remaining. You can "
    "delegate a sub-task to the `deeper-agent` subagent by calling "
    "`await tools.task({{subagent_type: 'deeper-agent', description: "
    "'...'}})`. It has the same tools and its own `eval` REPL, so use it "
    "when a sub-task itself decomposes into independent parallel steps. "
    "Prefer one `Promise.all(...)` over a deeper chain — recursion is "
    "for structural decomposition, not for issuing more calls."
)

_LEAF_SYSTEM_ADDENDUM = (
    "\n\n"
    "You are at the bottom of the recursion chain. No `deeper-agent` "
    "subagent is available here; do the work inline, using `eval` to "
    "parallelize tool calls when you can."
)


def create_rlm_agent(
    *,
    model: str | None = None,
    tools: list[BaseTool] | None = None,
    subagents: list[SubAgent | CompiledSubAgent] | None = None,
    max_depth: int = 1,
    **kwargs: Any,
) -> Any:
    """Build a Deep Agent with a self-recursive general-purpose subagent.

    Args:
        model: Passed through to `create_deep_agent`.
        tools: Tools available at every level, merged with the
            deep-agent built-ins.
        subagents: Extra subagents. Must NOT contain a spec named
            `general-purpose` — this helper provides its own at every
            level. Pass other subagent specs freely; they're carried
            through to every depth unchanged.
        max_depth: How many levels of recursion to build. `0` means
            "no recursion" — the returned agent still has REPL + PTC
            at the general-purpose subagent, but that subagent cannot
            delegate to a deeper peer. Capped at `_MAX_DEPTH_LIMIT`.
        **kwargs: Forwarded to `create_deep_agent`.

    Returns:
        A compiled Deep Agent graph.

    Raises:
        ValueError: If `max_depth` is negative, above the cap, or if
            `subagents` contains an entry named `general-purpose`
            (RLM owns that name and overriding it would break the
            recursion contract).
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
    """Recursive builder. Each call compiles one level.

    At depth N > 0, the general-purpose subagent's recursion budget
    is wired through a `CompiledSubAgent` pointing at a depth-(N-1)
    build of this same chain. The compiled peer is installed under a
    private name (`deeper-agent`) so the model sees one
    `general-purpose` entry plus one fan-out target — the private
    name appears in the task tool's subagent listing, and the
    system-prompt addendum explains when to use it.
    """
    if max_depth == 0:
        gp_middleware_addendum = _BASE_SYSTEM_ADDENDUM + _LEAF_SYSTEM_ADDENDUM
        level_subagents: list[SubAgent | CompiledSubAgent] = [
            _general_purpose_spec(gp_middleware_addendum),
            *extra_subagents,
        ]
        return create_deep_agent(
            model=model,
            tools=tools,
            subagents=level_subagents,
            **kwargs,
        )

    deeper = _build(
        model=model,
        tools=tools,
        extra_subagents=extra_subagents,
        max_depth=max_depth - 1,
        **kwargs,
    )
    deeper_entry = CompiledSubAgent(
        name="deeper-agent",
        description=(
            "Delegate a sub-task to a peer `general-purpose` agent one level deeper. "
            "Use when the sub-task itself decomposes into independent parallel steps."
        ),
        runnable=deeper,
    )
    gp_middleware_addendum = _BASE_SYSTEM_ADDENDUM + _RECURSE_SYSTEM_ADDENDUM.format(
        remaining=max_depth,
    )
    level_subagents = [
        _general_purpose_spec(gp_middleware_addendum),
        *extra_subagents,
        deeper_entry,
    ]
    return create_deep_agent(
        model=model,
        tools=tools,
        subagents=level_subagents,
        **kwargs,
    )


def _general_purpose_spec(system_addendum: str) -> SubAgent:
    """Return a general-purpose SubAgent spec with REPL + the given addendum.

    Overrides the default built-in general-purpose (which `create_deep_agent`
    provides automatically when no user spec claims the name). We override
    for two reasons: (1) attach `REPLMiddleware(ptc=True)`, (2) tack on a
    system-prompt addendum describing the recursion contract for this level.
    """
    base_prompt = GENERAL_PURPOSE_SUBAGENT["system_prompt"]
    return SubAgent(  # ty: ignore[missing-typed-dict-key]
        name=GENERAL_PURPOSE_SUBAGENT["name"],
        description=GENERAL_PURPOSE_SUBAGENT["description"],
        system_prompt=base_prompt + system_addendum,
        middleware=[REPLMiddleware(ptc=True)],
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
