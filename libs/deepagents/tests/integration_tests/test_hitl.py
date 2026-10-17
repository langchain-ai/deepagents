"""Integration tests for human-in-the-loop interrupt configuration.

Verifies that `create_deep_agent`'s `interrupt_on` config correctly pauses
execution for approval, respects per-tool decision settings, and propagates
interrupt config through subagent delegation.

These tests assert SDK structural state (interrupt payload shape, decision
list contents, post-resume tool messages), not model behavior quality. Any
model that emits the three parallel tool calls satisfies them.
"""

from __future__ import annotations

import uuid

import pytest
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from deepagents import create_deep_agent

pytestmark = pytest.mark.requires("langchain_anthropic")


@tool(description="Use this tool to get the weather")
def get_weather(location: str) -> str:
    return f"The weather in {location} is sunny."


@tool(description="Use this tool to get the latest soccer scores")
def get_soccer_scores(team: str) -> str:
    return f"The latest soccer scores for {team} are 2-1."


@tool(description="Sample tool")
def sample_tool(sample_input: str) -> str:
    return sample_input


SAMPLE_TOOL_CONFIG = {
    "sample_tool": True,
    "get_weather": False,
    "get_soccer_scores": {"allowed_decisions": ["approve", "reject"]},
}

MODEL_ID = "claude-sonnet-4-6"


def _model():
    return init_chat_model(MODEL_ID)


def _all_tool_calls(messages: list) -> list[dict]:
    calls: list[dict] = []
    for msg in messages:
        if isinstance(msg, AIMessage):
            calls.extend(msg.tool_calls or [])
    return calls


def test_hitl_agent() -> None:
    """Top-level `interrupt_on` pauses the graph and exposes the right review configs."""
    checkpointer = MemorySaver()
    agent = create_deep_agent(
        model=_model(),
        tools=[sample_tool, get_weather, get_soccer_scores],
        interrupt_on=SAMPLE_TOOL_CONFIG,
        checkpointer=checkpointer,
    )

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    query = "Call the sample tool, get the weather in New York and get scores for the latest soccer games in parallel"

    agent.invoke({"messages": [{"role": "user", "content": query}]}, config=config)

    state = agent.get_state(config)

    tool_calls = _all_tool_calls(state.values.get("messages", []))
    assert any(tc["name"] == "sample_tool" for tc in tool_calls)
    assert any(tc["name"] == "get_weather" for tc in tool_calls)
    assert any(tc["name"] == "get_soccer_scores" for tc in tool_calls)

    assert state.interrupts is not None
    assert len(state.interrupts) > 0
    interrupt_value = state.interrupts[0].value
    action_requests = interrupt_value["action_requests"]
    assert len(action_requests) == 2
    assert any(ar["name"] == "sample_tool" for ar in action_requests)
    assert any(ar["name"] == "get_soccer_scores" for ar in action_requests)
    review_configs = interrupt_value["review_configs"]
    assert any(rc["action_name"] == "sample_tool" and rc["allowed_decisions"] == ["approve", "edit", "reject", "respond"] for rc in review_configs)
    assert any(rc["action_name"] == "get_soccer_scores" and rc["allowed_decisions"] == ["approve", "reject"] for rc in review_configs)

    result = agent.invoke(
        Command(resume={"decisions": [{"type": "approve"}, {"type": "approve"}]}),
        config=config,
    )

    tool_results = [msg for msg in result.get("messages", []) if msg.type == "tool"]
    assert any(tr.name == "sample_tool" for tr in tool_results)
    assert any(tr.name == "get_weather" for tr in tool_results)
    assert any(tr.name == "get_soccer_scores" for tr in tool_results)


def test_subagent_with_hitl() -> None:
    """Subagent (general-purpose) inherits parent's `interrupt_on` and triggers approval."""
    checkpointer = MemorySaver()
    agent = create_deep_agent(
        model=_model(),
        tools=[sample_tool, get_weather, get_soccer_scores],
        interrupt_on=SAMPLE_TOOL_CONFIG,
        checkpointer=checkpointer,
    )

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    query = (
        "Use the task tool to kick off the general-purpose subagent. "
        "Tell it to call the sample tool, get the weather in New York "
        "and get scores for the latest soccer games in parallel"
    )

    agent.invoke({"messages": [{"role": "user", "content": query}]}, config=config)

    state = agent.get_state(config)

    assert state.interrupts is not None
    assert len(state.interrupts) > 0
    interrupt_value = state.interrupts[0].value
    action_requests = interrupt_value["action_requests"]
    assert len(action_requests) == 2
    assert any(ar["name"] == "sample_tool" for ar in action_requests)
    assert any(ar["name"] == "get_soccer_scores" for ar in action_requests)
    review_configs = interrupt_value["review_configs"]
    assert any(rc["action_name"] == "sample_tool" and rc["allowed_decisions"] == ["approve", "edit", "reject", "respond"] for rc in review_configs)
    assert any(rc["action_name"] == "get_soccer_scores" and rc["allowed_decisions"] == ["approve", "reject"] for rc in review_configs)

    agent.invoke(
        Command(resume={"decisions": [{"type": "approve"}, {"type": "approve"}]}),
        config=config,
    )

    state_after = agent.get_state(config)
    assert len(state_after.interrupts) == 0


def test_subagent_with_custom_interrupt_on() -> None:
    """Declarative subagent overrides parent's `interrupt_on` with its own config."""
    checkpointer = MemorySaver()
    agent = create_deep_agent(
        model=_model(),
        tools=[sample_tool, get_weather, get_soccer_scores],
        interrupt_on=SAMPLE_TOOL_CONFIG,
        checkpointer=checkpointer,
        subagents=[
            {
                "name": "task_handler",
                "description": "A subagent that can handle all sorts of tasks",
                "system_prompt": "You are a task handler. You can handle all sorts of tasks.",
                "tools": [sample_tool, get_weather, get_soccer_scores],
                "interrupt_on": {
                    "sample_tool": False,
                    "get_weather": True,
                    "get_soccer_scores": True,
                },
            },
        ],
    )

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    query = (
        "Use the task tool to kick off the task_handler subagent. "
        "Tell it to call the sample tool, get the weather in New York "
        "and get scores for the latest soccer games in parallel"
    )

    agent.invoke({"messages": [{"role": "user", "content": query}]}, config=config)

    state = agent.get_state(config)

    assert state.interrupts is not None
    assert len(state.interrupts) > 0
    interrupt_value = state.interrupts[0].value
    action_requests = interrupt_value["action_requests"]
    assert len(action_requests) == 2
    assert any(ar["name"] == "get_weather" for ar in action_requests)
    assert any(ar["name"] == "get_soccer_scores" for ar in action_requests)
    # `sample_tool` is disabled in the subagent config; should not be in interrupt
    assert not any(ar["name"] == "sample_tool" for ar in action_requests)

    review_configs = interrupt_value["review_configs"]
    assert any(rc["action_name"] == "get_weather" and rc["allowed_decisions"] == ["approve", "edit", "reject", "respond"] for rc in review_configs)
    assert any(
        rc["action_name"] == "get_soccer_scores" and rc["allowed_decisions"] == ["approve", "edit", "reject", "respond"] for rc in review_configs
    )

    agent.invoke(
        Command(resume={"decisions": [{"type": "approve"}, {"type": "approve"}]}),
        config=config,
    )

    state_after = agent.get_state(config)
    assert len(state_after.interrupts) == 0
