from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from langchain_core.tools import tool

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

from deepagents import create_deep_agent

from tests.evals.utils import (
    TrajectoryScorer,
    final_text_contains,
    run_agent,
    tool_call,
)


@tool
def get_weather_fake(location: str) -> str:  # noqa: ARG001
    """Return a fixed weather response for eval scenarios."""
    return "It's sunny at 89 degrees F"


@pytest.mark.langsmith
def test_task_calls_weather_subagent(model: BaseChatModel) -> None:
    """Requests a named subagent via task."""
    agent = create_deep_agent(
        model=model,
        subagents=[
            {
                "name": "weather_agent",
                "description": "Use this agent to get the weather",
                "system_prompt": "You are a weather agent.",
                "tools": [get_weather_fake],
                "model": "anthropic:claude-sonnet-4-6",
            }
        ],
    )
    run_agent(
        agent,
        query="Use the weather_agent subagent to get the weather in Tokyo.",
        model=model,
        # 1st step: request a subagent via the task tool.
        # 2nd step: answer using the subagent's tool result.
        # 1 tool call request: task.
        scorer=(
            TrajectoryScorer()
            .expect(
                agent_steps=2,
                tool_call_requests=1,
                tool_calls=[
                    tool_call(name="task", step=1, args_contains={"subagent_type": "weather_agent"})
                ],
            )
            .success(final_text_contains("89"))
        ),
    )


@pytest.mark.langsmith
def test_task_calls_general_purpose_subagent(model: BaseChatModel) -> None:
    """Requests the general-purpose subagent via task."""
    agent = create_deep_agent(model=model, tools=[get_weather_fake])
    run_agent(
        agent,
        query="Use the general purpose subagent to get the weather in Tokyo.",
        model=model,
        # 1st step: request a subagent via the task tool.
        # 2nd step: answer using the subagent's tool result.
        # 1 tool call request: task.
        scorer=(
            TrajectoryScorer()
            .expect(
                agent_steps=2,
                tool_call_requests=1,
                tool_calls=[
                    tool_call(
                        name="task", step=1, args_contains={"subagent_type": "general-purpose"}
                    )
                ],
            )
            .success(final_text_contains("89"))
        ),
    )
