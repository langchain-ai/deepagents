from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from deepagents import create_deep_agent

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

from tests.evals.utils import TrajectoryScorer, final_text_contains, run_agent, tool_call


@pytest.mark.langsmith
def test_write_todos_sequential_updates_returns_text(model: BaseChatModel) -> None:
    """Creates a 5-item todo list and updates it 5 times, then responds with text."""
    agent = create_deep_agent(model=model)
    run_agent(
        agent,
        model=model,
        query=(
            "Create a TODO list with exactly 5 items using the write_todos tool. "
            "Theme: morning routine. Use these exact items in this exact order: "
            "1) Make coffee 2) Drink water 3) Check calendar 4) Write a short plan 5) Start first task. "
            "Then update the TODO list 5 times sequentially (one write_todos call per step). "
            "For update i (1-5), mark item i as completed and leave the others unchanged. "
            "After the final update, reply with the single word DONE."
        ),
        scorer=TrajectoryScorer()
        .expect(
            tool_calls=[tool_call(name="write_todos", step=i) for i in range(1, 7)],
        )
        .success(final_text_contains("DONE")),
    )


@pytest.mark.langsmith
def test_write_todos_three_steps_returns_text(model: BaseChatModel) -> None:
    """Creates a 3-item todo list and updates it twice, then responds with text."""
    agent = create_deep_agent(model=model)
    run_agent(
        agent,
        model=model,
        query=(
            "Create a TODO list with exactly 3 items using the write_todos tool. "
            "Theme: quick setup. Use these exact items in this exact order: "
            "1) Open editor 2) Pull latest changes 3) Run tests. "
            "Then update the TODO list 2 times sequentially (one write_todos call per step). "
            "For update i (1-2), mark item i as completed and leave the others unchanged. "
            "After the final update, reply with the single word DONE."
        ),
        scorer=TrajectoryScorer()
        .expect(
            tool_calls=[tool_call(name="write_todos", step=i) for i in range(1, 5)],
        )
        .success(final_text_contains("DONE")),
    )
