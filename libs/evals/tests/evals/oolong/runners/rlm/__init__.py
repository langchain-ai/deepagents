"""RLM runner — ``create_deep_agent`` with a recursive compiled
``general-purpose`` chain.

Each level has ``REPLMiddleware(ptc=True)`` so the model can write
``eval`` + ``Promise.all(tools.<x>(...))`` to parallelize. At depths
> 0 the default ``general-purpose`` subagent is replaced with a
``CompiledSubAgent`` whose runnable is itself a depth-(N-1) RLM agent,
giving the model a real deeper graph to dispatch into via
``tools.task({subagent_type: "general-purpose", ...})``.

Mirrors :func:`examples.rlm_agent.rlm_agent.create_rlm_agent`. The
builder is inlined rather than imported because ``examples/`` is not
installable and the eval harness must not depend on example code at
runtime.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from deepagents import create_deep_agent

from tests.evals.oolong.runners._common import (
    SYSTEM_PROMPT,
    RunnerContext,
    noop_teardown,
)

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langgraph.graph.state import CompiledStateGraph

    from tests.evals.oolong.data_utils import OolongTask


RLM_MAX_DEPTH = 3
"""How many recursion levels to build.

``1`` means the root agent has its ``general-purpose`` replaced by a
compiled depth-0 agent (plain REPL+PTC, no further recursion). Raise
this if you want to exercise deeper chains, but note that every level
doubles the per-task token spend roughly proportionally.
"""


def _build_recursive(
    *,
    model: BaseChatModel,
    max_depth: int,
) -> CompiledStateGraph:
    """Recursive builder. One call compiles one level of the chain.

    At depth 0, ``general-purpose`` is the default auto-injection —
    no override, no REPL on it, no deeper peer. At depth N > 0, we
    build the depth-(N-1) agent first and register it as a
    ``CompiledSubAgent`` under the name ``general-purpose``, which
    satisfies the "already has a general-purpose" check in
    ``create_deep_agent`` and suppresses the auto-injected default at
    this level.
    """
    from deepagents.middleware.subagents import (
        GENERAL_PURPOSE_SUBAGENT,
        CompiledSubAgent,
    )
    from deepagents_repl import REPLMiddleware

    # ``timeout=None`` disables the REPL's per-eval deadline. RLM agents
    # dispatch ``tools.task`` from inside ``eval``, which blocks on the
    # deeper agent finishing — routinely longer than the default 5s,
    # and the interrupt would kill the eval mid-``Promise.all``.
    if max_depth == 0:
        return create_deep_agent(
            model=model,
            system_prompt=SYSTEM_PROMPT,
            middleware=[REPLMiddleware(ptc=True, timeout=None)],
        )

    deeper = _build_recursive(model=model, max_depth=max_depth - 1)
    compiled_gp = CompiledSubAgent(
        name=GENERAL_PURPOSE_SUBAGENT["name"],
        description=GENERAL_PURPOSE_SUBAGENT["description"],
        runnable=deeper,
    )
    return create_deep_agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        subagents=[compiled_gp],
        middleware=[REPLMiddleware(ptc=True, timeout=None)],
    )


async def build_runner(
    *,
    model: BaseChatModel,
    task: OolongTask,
) -> RunnerContext:
    """Build the RLM runner at depth :data:`RLM_MAX_DEPTH`."""
    agent = _build_recursive(model=model, max_depth=RLM_MAX_DEPTH)
    return RunnerContext(
        agent=agent,
        query_addendum="",
        initial_files={"/context.txt": task.context_window_text},
        teardown=noop_teardown,
    )


__all__ = ["RLM_MAX_DEPTH", "build_runner"]
