from __future__ import annotations

import pytest

from deepagents import create_deep_agent
from tests.evals.utils import TrajectoryExpectations, run_agent


@pytest.mark.langsmith
def test_task_calls_weather_subagent() -> None:
    """Requests a named subagent via task."""
    agent = create_deep_agent(
        subagents=[
            {
                "name": "weather_agent",
                "description": "Use this agent to get the weather",
                "system_prompt": "You are a weather agent.",
                "tools": [],
                "model": "anthropic:claude-sonnet-4-5-20250929",
            }
        ]
    )
    run_agent(
        agent,
        query="Use the weather_agent subagent to get the weather in Tokyo.",
        # 1 step: request a subagent via the task tool.
        # 1 tool call request: task.
        expect=TrajectoryExpectations(num_agent_steps=1, num_tool_call_requests=1).require_tool_call(
            step=1,
            name="task",
            args_contains={"subagent_type": "weather_agent"},
        ),
    )


@pytest.mark.langsmith
def test_task_calls_general_purpose_subagent() -> None:
    """Requests the general-purpose subagent via task."""
    agent = create_deep_agent()
    run_agent(
        agent,
        query="Use the general purpose subagent to get the weather in Tokyo.",
        # 1 step: request a subagent via the task tool.
        # 1 tool call request: task.
        expect=TrajectoryExpectations(num_agent_steps=1, num_tool_call_requests=1).require_tool_call(
            step=1,
            name="task",
            args_contains={"subagent_type": "general-purpose"},
        ),
    )
