"""Swarm runner — REPL + PTC + the ``swarm`` skill from a real directory.

Backend shape mirrors :mod:`examples.repl_swarm.swarm_agent`:
``CompositeBackend`` that sends ``/skills/*`` to a
``FilesystemBackend`` rooted at the local ``skills/`` directory, and
everything else (including Oolong's ``/context.txt``) to
``StateBackend``. The task's context continues to flow through state
via ``run_agent_async(initial_files=...)`` — same path baseline and
rlm use — so swarm stays structurally symmetric at the
test-harness level.

The ``skills/`` directory here is a symlink to
``examples/repl_swarm/skills/`` — one source of truth for SKILL.md
and index.ts. Edit there and the change shows up in both the demo
and the eval.

Unlike ``rlm``, there's no compiled deeper ``general-purpose`` here
— the swarm skill already fans out to ``tools.task`` through the
default auto-injected ``general-purpose``, which gives the skill
somewhere to dispatch to. Adding a deeper compiled peer would
complicate the comparison without clear benefit for the one-shot
Oolong aggregation tasks.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from deepagents import create_deep_agent
from deepagents.backends.composite import CompositeBackend
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.state import StateBackend

from tests.evals.oolong.runners._common import (
    SYSTEM_PROMPT,
    RunnerContext,
    noop_teardown,
)

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from tests.evals.oolong.data_utils import OolongTask


_SKILLS_DIR = Path(__file__).parent / "skills"


SWARM_QUERY_ADDENDUM = """

You also have a ``swarm`` skill available. Import it with \
``const { runSwarm } = await import("@/skills/swarm")`` from inside an \
``eval`` call. ``runSwarm({tasks: [{description: "..."}, ...]})`` \
dispatches each task to the ``general-purpose`` subagent in parallel \
with bounded concurrency and returns a summary of their results. Use \
it when the question can be decomposed into independent sub-queries \
over /context.txt that can each be answered by a simpler subagent, \
then aggregated back. Prefer one ``runSwarm`` call over many \
sequential ``tools.task`` calls."""
"""Query nudge for the swarm runner. Tells the model the skill
exists and when to reach for it; the answer format rules from the
base prompt still apply. Lives in the user query (not the system
prompt) so the hint is scoped to the task at hand, matching the
shape of the reference snippet."""


async def build_runner(
    *,
    model: BaseChatModel,
    task: OolongTask,  # noqa: ARG001  # seeded by run_agent_async upstream
) -> RunnerContext:
    """Build the swarm runner."""
    from deepagents_repl import REPLMiddleware

    skill_backend = FilesystemBackend(
        root_dir=str(_SKILLS_DIR), virtual_mode=True
    )
    backend = CompositeBackend(
        default=StateBackend(),
        routes={"/skills/": skill_backend},
    )
    # Pass the composite (not the raw ``skill_backend``) as
    # ``skills_backend`` so the REPL's resolver goes through the
    # ``/skills/`` route with prefix stripping. Otherwise it would ask
    # ``skill_backend`` for ``/skills/swarm`` directly, which with
    # ``virtual_mode=True`` resolves to ``<skills_dir>/skills/swarm``
    # and fails with "no JS/TS files".
    #
    # ``timeout=None`` disables the REPL's per-eval deadline. Swarm
    # dispatches several ``tools.task`` calls from inside ``eval`` and
    # the default 5s cap would kill the eval mid-``Promise.all``.
    agent = create_deep_agent(
        model=model,
        backend=backend,
        skills=["/skills/"],
        system_prompt=SYSTEM_PROMPT,
        middleware=[REPLMiddleware(ptc=True, skills_backend=backend, timeout=None)],
    )
    return RunnerContext(
        agent=agent,
        query_addendum=SWARM_QUERY_ADDENDUM,
        initial_files={"/context.txt": task.context_window_text},
        teardown=noop_teardown,
    )


__all__ = ["SWARM_QUERY_ADDENDUM", "build_runner"]
