from __future__ import annotations

import pytest
from langchain_core.tools import tool

from deepagents import create_deep_agent
from tests.evals.utils import TrajectoryExpectations, run_agent


@tool
def get_weather_fake(location: str) -> str:  # noqa: ARG001
    """Return a fixed weather response for eval scenarios."""
    return "It's sunny at 89 degrees F"


@pytest.mark.langsmith
def test_task_calls_weather_subagent(model: str) -> None:
    """Requests a named subagent via task."""
    agent = create_deep_agent(
        model=model,
        subagents=[
            {
                "name": "weather_agent",
                "description": "Use this agent to get the weather",
                "system_prompt": "You are a weather agent.",
                "tools": [get_weather_fake],
                "model": "anthropic:claude-sonnet-4-5-20250929",
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
        expect=(
            TrajectoryExpectations(num_agent_steps=2, num_tool_call_requests=1)
            .require_tool_call(step=1, name="task", args_contains={"subagent_type": "weather_agent"})
            .require_final_text_contains("89")
        ),
    )


@pytest.mark.langsmith
def test_task_calls_general_purpose_subagent(model: str) -> None:
    """Requests the general-purpose subagent via task."""
    agent = create_deep_agent(model=model, tools=[get_weather_fake])
    run_agent(
        agent,
        query="Use the general purpose subagent to get the weather in Tokyo.",
        model=model,
        # 1st step: request a subagent via the task tool.
        # 2nd step: answer using the subagent's tool result.
        # 1 tool call request: task.
        expect=(
            TrajectoryExpectations(num_agent_steps=2, num_tool_call_requests=1)
            .require_tool_call(step=1, name="task", args_contains={"subagent_type": "general-purpose"})
            .require_final_text_contains("89")
        ),
    )
