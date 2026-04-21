"""Baseline runner — plain ``create_deep_agent`` with no REPL tricks.

Matches the JS Oolong setup, which uses ``getDefaultRunner()``. The
control condition every other runner is compared against.
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

    from tests.evals.oolong.data_utils import OolongTask


async def build_runner(
    *,
    model: BaseChatModel,
    task: OolongTask,
) -> RunnerContext:
    """Build the plain-harness baseline runner."""
    agent = create_deep_agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
    )
    return RunnerContext(
        agent=agent,
        query_addendum="",
        initial_files={"/context.txt": task.context_window_text},
        teardown=noop_teardown,
    )


__all__ = ["build_runner"]
