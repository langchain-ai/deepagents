from __future__ import annotations

import pytest

from deepagents import create_deep_agent
from tests.evals.utils import TrajectoryExpectations, run_agent


@pytest.mark.langsmith
def test_read_file_seeded_state_backend_file() -> None:
    agent = create_deep_agent()
    trajectory = run_agent(
        agent,
        initial_files={"/foo.md": "alpha beta gamma\none two three four\n"},
        query="Read /foo.md and tell me the 3rd word on the 2nd line.",
        # 1st step: request a tool call to read /foo.md.
        # 2nd step: answer the question using the file contents.
        # 1 tool call request: read_file.
        expect=TrajectoryExpectations(num_agent_steps=2, num_tool_call_requests=1),
    )
    assert "three" in trajectory.steps[-1].action.text


@pytest.mark.langsmith
def test_write_file_simple() -> None:
    agent = create_deep_agent(system_prompt="Your name is Foo Bar.")
    trajectory = run_agent(
        agent,
        query="Write your name to a file called /foo.md and then tell me your name.",
        # 1st step: request a tool call to write /foo.md.
        # 2nd step: tell the user the name.
        # 1 tool call request: write_file.
        expect=TrajectoryExpectations(num_agent_steps=2, num_tool_call_requests=1),
    )
    assert "Foo Bar" in trajectory.files["/foo.md"]
    assert "Foo Bar" in trajectory.steps[-1].action.text
