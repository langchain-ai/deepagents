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


@pytest.mark.langsmith
def test_write_files_in_parallel() -> None:
    agent = create_deep_agent()
    trajectory = run_agent(
        agent,
        query='Write "bar" to /a.md and "bar" to /b.md. Do the writes in parallel, then confirm you did it.',
        # 1st step: request 2 write_file tool calls in parallel.
        # 2nd step: confirm the writes.
        # 2 tool call requests: write_file to /a.md and write_file to /b.md.
        expect=TrajectoryExpectations(num_agent_steps=2, num_tool_call_requests=2),
    )
    assert trajectory.files["/a.md"] == "bar"
    assert trajectory.files["/b.md"] == "bar"

    step1_tool_calls = trajectory.steps[0].action.tool_calls
    assert len(step1_tool_calls) == 2
    assert {tc["name"] for tc in step1_tool_calls} == {"write_file"}
    assert {tc["args"]["file_path"] for tc in step1_tool_calls} == {"/a.md", "/b.md"}
