from __future__ import annotations

import uuid

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from deepagents import create_deep_agent
from tests.evals.utils import run_agent
from tests.utils import get_soccer_scores, get_weather, sample_tool

SAMPLE_TOOL_CONFIG = {
    "sample_tool": True,
    "get_weather": False,
    "get_soccer_scores": {"allowed_decisions": ["approve", "reject"]},
}


@pytest.mark.langsmith
def test_hitl_agent(model: str) -> None:
    """Test that agent respects interrupt_on configuration and waits for human approval."""
    checkpointer = MemorySaver()
    agent = create_deep_agent(
        model=model,
        tools=[sample_tool, get_weather, get_soccer_scores],
        interrupt_on=SAMPLE_TOOL_CONFIG,
        checkpointer=checkpointer,
    )

    thread_id = str(uuid.uuid4())
    query = "Call the sample tool, get the weather in New York and get scores for the latest soccer games in parallel"

    # First invocation hits interrupt
    trajectory = run_agent(
        agent,
        model=model,
        query=query,
        thread_id=thread_id,
    )

    # Check state for interrupts
    config = {"configurable": {"thread_id": thread_id}}
    state = agent.get_state(config)

    # Verify all three tool calls were requested
    tool_calls = []
    for step in trajectory.steps:
        tool_calls.extend(step.action.tool_calls)

    assert any(tc["name"] == "sample_tool" for tc in tool_calls)
    assert any(tc["name"] == "get_weather" for tc in tool_calls)
    assert any(tc["name"] == "get_soccer_scores" for tc in tool_calls)

    # Verify interrupts are present
    assert state.interrupts is not None
    assert len(state.interrupts) > 0
    interrupt_value = state.interrupts[0].value
    action_requests = interrupt_value["action_requests"]
    assert len(action_requests) == 2
    assert any(action_request["name"] == "sample_tool" for action_request in action_requests)
    assert any(action_request["name"] == "get_soccer_scores" for action_request in action_requests)
    review_configs = interrupt_value["review_configs"]
    assert any(
        review_config["action_name"] == "sample_tool" and review_config["allowed_decisions"] == ["approve", "edit", "reject"]
        for review_config in review_configs
    )
    assert any(
        review_config["action_name"] == "get_soccer_scores" and review_config["allowed_decisions"] == ["approve", "reject"]
        for review_config in review_configs
    )

    # Resume with approvals - this continues from the interrupted state
    result = agent.invoke(Command(resume={"decisions": [{"type": "approve"}, {"type": "approve"}]}), config=config)

    # Verify all tool results are present after approval
    tool_results = [msg for msg in result.get("messages", []) if msg.type == "tool"]
    assert any(tool_result.name == "sample_tool" for tool_result in tool_results)
    assert any(tool_result.name == "get_weather" for tool_result in tool_results)
    assert any(tool_result.name == "get_soccer_scores" for tool_result in tool_results)


@pytest.mark.langsmith
def test_subagent_with_hitl(model: str) -> None:
    """Test that subagent respects parent's interrupt_on configuration."""
    checkpointer = MemorySaver()
    agent = create_deep_agent(
        model=model,
        tools=[sample_tool, get_weather, get_soccer_scores],
        interrupt_on=SAMPLE_TOOL_CONFIG,
        checkpointer=checkpointer,
    )

    thread_id = str(uuid.uuid4())
    query = (
        "Use the task tool to kick off the general-purpose subagent. "
        "Tell it to call the sample tool, get the weather in New York "
        "and get scores for the latest soccer games in parallel"
    )

    # First invocation hits interrupt
    _ = run_agent(
        agent,
        model=model,
        query=query,
        thread_id=thread_id,
    )

    # Check state for interrupts
    config = {"configurable": {"thread_id": thread_id}}
    state = agent.get_state(config)

    # Verify interrupts are present
    assert state.interrupts is not None
    assert len(state.interrupts) > 0
    interrupt_value = state.interrupts[0].value
    action_requests = interrupt_value["action_requests"]
    assert len(action_requests) == 2
    assert any(action_request["name"] == "sample_tool" for action_request in action_requests)
    assert any(action_request["name"] == "get_soccer_scores" for action_request in action_requests)
    review_configs = interrupt_value["review_configs"]
    assert any(
        review_config["action_name"] == "sample_tool" and review_config["allowed_decisions"] == ["approve", "edit", "reject"]
        for review_config in review_configs
    )
    assert any(
        review_config["action_name"] == "get_soccer_scores" and review_config["allowed_decisions"] == ["approve", "reject"]
        for review_config in review_configs
    )

    # Resume with approvals
    _ = agent.invoke(Command(resume={"decisions": [{"type": "approve"}, {"type": "approve"}]}), config=config)

    # Verify no more interrupts after approval
    state_after = agent.get_state(config)
    assert len(state_after.interrupts) == 0


@pytest.mark.langsmith
def test_subagent_with_custom_interrupt_on(model: str) -> None:
    """Test that subagent can have its own custom interrupt_on configuration."""
    checkpointer = MemorySaver()
    agent = create_deep_agent(
        model=model,
        tools=[sample_tool, get_weather, get_soccer_scores],
        interrupt_on=SAMPLE_TOOL_CONFIG,
        checkpointer=checkpointer,
        subagents=[
            {
                "name": "task_handler",
                "description": "A subagent that can handle all sorts of tasks",
                "system_prompt": "You are a task handler. You can handle all sorts of tasks.",
                "tools": [sample_tool, get_weather, get_soccer_scores],
                "interrupt_on": {"sample_tool": False, "get_weather": True, "get_soccer_scores": True},
            },
        ],
    )

    thread_id = str(uuid.uuid4())
    query = (
        "Use the task tool to kick off the task_handler subagent. "
        "Tell it to call the sample tool, get the weather in New York "
        "and get scores for the latest soccer games in parallel"
    )

    # First invocation hits interrupt
    _ = run_agent(
        agent,
        model=model,
        query=query,
        thread_id=thread_id,
    )

    # Check state for interrupts
    config = {"configurable": {"thread_id": thread_id}}
    state = agent.get_state(config)

    # Verify interrupts are present for get_weather and get_soccer_scores, but NOT sample_tool
    assert state.interrupts is not None
    assert len(state.interrupts) > 0
    interrupt_value = state.interrupts[0].value
    action_requests = interrupt_value["action_requests"]
    assert len(action_requests) == 2
    assert any(action_request["name"] == "get_weather" for action_request in action_requests)
    assert any(action_request["name"] == "get_soccer_scores" for action_request in action_requests)
    # sample_tool should NOT be in the interrupt requests since it's disabled in subagent config
    assert not any(action_request["name"] == "sample_tool" for action_request in action_requests)

    review_configs = interrupt_value["review_configs"]
    assert any(
        review_config["action_name"] == "get_weather" and review_config["allowed_decisions"] == ["approve", "edit", "reject"]
        for review_config in review_configs
    )
    assert any(
        review_config["action_name"] == "get_soccer_scores" and review_config["allowed_decisions"] == ["approve", "edit", "reject"]
        for review_config in review_configs
    )

    # Resume with approvals
    _ = agent.invoke(Command(resume={"decisions": [{"type": "approve"}, {"type": "approve"}]}), config=config)

    # Verify no more interrupts after approval
    state_after = agent.get_state(config)
    assert len(state_after.interrupts) == 0
