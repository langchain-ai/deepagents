"""Swarm-eval agent factory.

Self-contained copy of the swarm-runner shape — agent wiring +
system prompt — so this eval doesn't reach across into
``oolong/``. Skill source lives under ``skills/``, which is a
symlink to ``examples/repl_swarm/skills/`` (one source of truth for
SKILL.md + index.ts).

Shape: REPL + PTC at the root, ``CompositeBackend`` that routes
``/skills/*`` to a ``FilesystemBackend`` rooted at this package's
symlinked ``skills/`` directory and sends everything else to
``StateBackend``. The task's item files (``/items/NN.txt``) and any
other agent writes flow through state as usual — the filesystem
backend only serves skill source.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from deepagents import create_deep_agent
from deepagents.backends.composite import CompositeBackend
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.state import StateBackend

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langgraph.graph.state import CompiledStateGraph


_SKILLS_DIR = Path(__file__).parent / "skills"


SYSTEM_PROMPT = """\
You are a classification agent. You have access to a ``swarm`` skill \
that fans out independent subtasks to parallel subagents.

Import the skill inside an ``eval`` call:
``const { runSwarm } = await import("@/skills/swarm")``.
Then call ``runSwarm({ tasks: [{ description: "..." }, ...] })`` with \
one task per item you need to process. Each task description should be \
self-contained — include any file paths or context the subagent needs.

Return your final answer in the exact format the user requested, with \
no extra commentary."""


def build_swarm_agent(*, model: BaseChatModel) -> CompiledStateGraph:
    """Build a REPL+PTC agent with the ``swarm`` skill mounted under ``/skills/``.

    Backend is a ``CompositeBackend`` that routes ``/skills/*`` to a
    ``FilesystemBackend`` rooted at the local (symlinked) skills
    directory and everything else to ``StateBackend``. Callers seed
    task-scoped files (e.g. ``/items/NN.txt``) via
    ``run_agent_async(initial_files=...)`` as usual; those flow
    through the composite's default route into state.
    """
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
    # the skill backend for ``/skills/swarm`` directly, which with
    # ``virtual_mode=True`` resolves to ``<skills_dir>/skills/swarm``
    # and fails with "no JS/TS files".
    #
    # ``timeout=None`` disables the REPL's per-eval deadline. ``runSwarm``
    # awaits multiple ``tools.task`` calls from inside ``eval``, and the
    # default 5s cap would kill the eval mid-``Promise.all``.
    return create_deep_agent(
        model=model,
        backend=backend,
        skills=["/skills/"],
        system_prompt=SYSTEM_PROMPT,
        middleware=[REPLMiddleware(ptc=True, skills_backend=backend, timeout=None)],
    )


__all__ = ["SYSTEM_PROMPT", "build_swarm_agent"]
