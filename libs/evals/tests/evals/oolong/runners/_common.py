"""Shared types and constants for every Oolong runner.

Every runner module in this package imports from here and exports an
async ``build_runner`` matching :class:`RunnerBuilder`. The shared
system prompt lives here so baseline/rlm/swarm/shell reason under the
same constraints — they only differ in how they're wired, not in what
they're told.

A ``RunnerContext`` owns the compiled agent, any query addendum the
runner wants to append to the task question, and a teardown callable
for runner-owned external resources (e.g. LangSmith sandboxes).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Awaitable, Callable, Protocol

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langgraph.graph.state import CompiledStateGraph

    from tests.evals.oolong.data_utils import OolongTask


SYSTEM_PROMPT = """\
You are a precise data analyst. You have access to a file /context.txt \
that contains labelled data.

When asked a question about this data, you MUST:
1. Read the file /context.txt to see the full dataset.
2. Carefully analyse every single data point — do not skip, estimate, \
or approximate.
3. Count and classify exactly as instructed by the question.
4. Return ONLY the final answer in the exact format requested — no \
explanation, no extra text.

Be precise. The questions ask about aggregate statistics (counts, \
frequencies, comparisons). Every data point matters."""


@dataclass(frozen=True)
class RunnerContext:
    """What a runner hands back to the test body.

    Attributes:
        agent: Compiled agent ready for ``run_agent_async``.
        query_addendum: String appended to the task question. Swarm
            uses this to nudge the model toward ``runSwarm``; other
            runners leave it empty.
        initial_files: Files to seed via ``run_agent_async``'s state
            path. ``None`` for runners that seed via an external
            sandbox (shell) — those already uploaded before returning.
        teardown: Called after the task run. No-op for stateless
            runners; closes sandboxes for runners that own them.
    """

    agent: CompiledStateGraph
    query_addendum: str
    initial_files: dict[str, str] | None
    teardown: Callable[[], None]


class RunnerBuilder(Protocol):
    """Factory contract every runner subpackage exports as ``build_runner``.

    Takes the model and the task (some runners need the task's
    ``context_window_text`` to seed an external sandbox before the
    agent runs). Returns a :class:`RunnerContext`.
    """

    async def __call__(
        self,
        *,
        model: BaseChatModel,
        task: OolongTask,
    ) -> RunnerContext: ...


def noop_teardown() -> None:
    """Teardown for runners that don't own external resources."""


__all__ = [
    "SYSTEM_PROMPT",
    "RunnerBuilder",
    "RunnerContext",
    "noop_teardown",
]
