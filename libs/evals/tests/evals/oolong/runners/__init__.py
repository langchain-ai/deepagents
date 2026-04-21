"""Agent factories for the Oolong evals.

Four variants, parametrized over in the test suite. Each lives in its
own subpackage so it can carry its own assets (skill files, sandbox
templates) next to its code:

- :mod:`.baseline` — plain ``create_deep_agent``. Matches the JS
  Oolong setup, which uses ``getDefaultRunner()`` with no REPL or
  subagent tricks.
- :mod:`.rlm` — proper recursive RLM: each level has REPL + PTC, and
  the inner ``general-purpose`` is a compiled depth-(N-1) RLM agent.
  See :data:`.rlm.RLM_MAX_DEPTH`.
- :mod:`.swarm` — REPL + PTC + the ``swarm`` skill served from a
  ``FilesystemBackend`` mounted at ``/skills/`` via a
  ``CompositeBackend`` (everything else goes to state). Model can
  ``await import("@/skills/swarm")`` and call ``runSwarm(...)`` to
  fan tasks out via ``tools.task``. The skill directory is a symlink
  to ``examples/repl_swarm/skills/`` — one source of truth.
- :mod:`.shell` — agent backed by a per-task LangSmith sandbox. Model
  gets the full ``FilesystemMiddleware`` tool set including an
  ``execute`` shell tool that runs inside the sandbox.

Each subpackage exports ``build_runner(model=, task=)`` — async because
runners that own external sandboxes (shell) need to seed files before
returning. The shared test body iterates the runner registry and
dispatches per-task so one pytest invocation runs every runner side
by side; LangSmith ends up with one experiment per (runner, task)
pair.
"""

from __future__ import annotations

from tests.evals.oolong.runners._common import (
    SYSTEM_PROMPT,
    RunnerBuilder,
    RunnerContext,
    noop_teardown,
)
from tests.evals.oolong.runners.baseline import build_runner as build_baseline_runner
from tests.evals.oolong.runners.rlm import build_runner as build_rlm_runner
from tests.evals.oolong.runners.shell import build_runner as build_shell_runner
from tests.evals.oolong.runners.swarm import (
    SWARM_QUERY_ADDENDUM,
    build_runner as build_swarm_runner,
)

RUNNERS: dict[str, RunnerBuilder] = {
    "baseline": build_baseline_runner,
    "rlm": build_rlm_runner,
    "swarm": build_swarm_runner,
    "shell": build_shell_runner,
}
"""Registry keyed by runner name. The test suite parametrizes over
``RUNNERS.keys()`` so one pytest invocation runs every runner against
every task."""


__all__ = [
    "RUNNERS",
    "SWARM_QUERY_ADDENDUM",
    "SYSTEM_PROMPT",
    "RunnerBuilder",
    "RunnerContext",
    "build_baseline_runner",
    "build_rlm_runner",
    "build_shell_runner",
    "build_swarm_runner",
    "noop_teardown",
]
