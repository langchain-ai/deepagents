from __future__ import annotations

import pytest

from deepagents import create_deep_agent
from tests.evals.utils import TrajectoryExpectations, run_agent


@pytest.mark.langsmith
def test_custom_system_prompt() -> None:
    """Verify that custom system provided prompt works is incorporated."""
    agent = create_deep_agent(system_prompt="Your name is Foo Bar.")
    trajectory = run_agent(
        agent,
        query="what is your name",
        # 1 step: answer directly.
        # 0 tool calls: no files/tools needed.
        expect=TrajectoryExpectations(num_agent_steps=1, num_tool_call_requests=0),
    )
    # Extra asserts for whether action is successful or fails.
    assert "Foo Bar" in trajectory.steps[-1].action.text
