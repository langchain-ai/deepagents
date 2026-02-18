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
        expect=TrajectoryExpectations(num_agent_steps=1, num_tool_call_requests=0),
    )
    # Extra asserts for whether action is successful or fails.
    assert "Foo Bar" in trajectory.steps[-1].action.text


@pytest.mark.langsmith
def test_write_to_a_file() -> None:
    """Verify write to a file works without triggering unnecessary actions."""
    agent = create_deep_agent(system_prompt="Your name is Foo Bar.")
    trajectory = run_agent(
        agent,
        query="write your name to a file called /foo.md and then tell me your name",
        # Tool call request to edit file.
        expect=TrajectoryExpectations(num_agent_steps=2, num_tool_call_requests=1),
    )
    assert "Foo Bar" in trajectory.files["/foo.md"]
    # Extra asserts for whether action is successful or fails.
    assert "Foo Bar" in trajectory.steps[-1].action.text


@pytest.mark.langsmith
def test_read_third_word_second_line_from_file() -> None:
    """Verify reading from a seeded file works via StateBackend-style file state."""
    agent = create_deep_agent()
    trajectory = run_agent(
        agent,
        initial_files={"/foo.md": "alpha beta gamma\none two three four\n"},
        query="Read /foo.md and tell me the 3rd word on the 2nd line.",
        # 1 step to request a tool call to read foo.md
        # 2nd step to output the result of the question
        expect=TrajectoryExpectations(num_agent_steps=2, num_tool_call_requests=1),
    )
    assert "three" in trajectory.steps[-1].action.text
