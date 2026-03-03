from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from deepagents import create_deep_agent

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
from tests.evals.utils import TrajectoryExpectations, run_agent


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
        expect=TrajectoryExpectations(num_agent_steps=1, num_tool_call_requests=0).require_final_text_contains(
            "Foo Bar",
        ),
    )
