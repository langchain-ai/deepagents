from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from deepagents import create_deep_agent

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
from tests.evals.utils import (
    TrajectoryScorer,
    agent_steps,
    final_text_contains,
    run_agent,
    tool_call_requests,
)


@pytest.mark.langsmith
def test_custom_system_prompt(model: BaseChatModel) -> None:
    """Custom system prompt is reflected in the answer."""
    agent = create_deep_agent(model=model, system_prompt="Your name is Foo Bar.")
    run_agent(
        agent,
        query="what is your name",
        model=model,
        # 1 step: answer directly.
        # 0 tool calls: no files/tools needed.
        scorer=TrajectoryScorer()
        .expect(agent_steps(1), tool_call_requests(0))
        .success(final_text_contains("Foo Bar")),
    )
