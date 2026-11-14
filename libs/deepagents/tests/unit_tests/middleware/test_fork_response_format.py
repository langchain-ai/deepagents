"""Structured-output formats in the provider requests made by forked subagents."""

import json
from typing import Any

import pytest
from langchain.agents.structured_output import ProviderStrategy, ToolStrategy
from langchain_core.messages import HumanMessage, ToolMessage
from pydantic import BaseModel

from deepagents.graph import create_deep_agent
from deepagents.middleware.subagents import SUBAGENT_RESPONSE_FORMAT_CONFIG_KEY
from tests.unit_tests.middleware.skill_tools_support import call, invoke, stub_anthropic


class ParentOutput(BaseModel):
    answer: str


class ChildOutput(BaseModel):
    detail: str


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize(
    ("parent_format", "child_format", "same_tools", "same_output_config"),
    [
        (None, None, True, True),
        (ToolStrategy(ParentOutput), None, False, True),
        (ToolStrategy(ParentOutput), ToolStrategy(ParentOutput), True, True),
        (ToolStrategy(ParentOutput), ToolStrategy(ChildOutput), False, True),
        (None, ToolStrategy(ChildOutput), False, True),
        (ProviderStrategy(ParentOutput), None, True, False),
        (ProviderStrategy(ParentOutput), ProviderStrategy(ParentOutput), True, True),
        (ProviderStrategy(ParentOutput), ProviderStrategy(ChildOutput), True, False),
    ],
    ids=["none", "tool-omitted", "tool-matching", "tool-override", "tool-child-only", "provider-omitted", "provider-matching", "provider-override"],
)
def test_fork_response_formats_at_provider(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    parent_format: ToolStrategy[Any] | ProviderStrategy[Any] | None,
    child_format: ToolStrategy[Any] | ProviderStrategy[Any] | None,
    *,
    same_tools: bool,
    same_output_config: bool,
) -> None:
    if child_format is None:
        child_turn: Any = "child text"
        expected_child = "child text"
    elif child_format.schema is ParentOutput:
        child_turn = [call("ParentOutput", "child_output", answer="child answer")]
        expected_child = {"answer": "child answer"}
    else:
        child_turn = [call("ChildOutput", "child_output", detail="child detail")]
        expected_child = {"detail": "child detail"}
    if isinstance(child_format, ProviderStrategy):
        child_turn = json.dumps(expected_child)

    parent_turn: Any = "parent text" if parent_format is None else [call("ParentOutput", "parent_output", answer="parent answer")]
    if isinstance(parent_format, ProviderStrategy):
        parent_turn = json.dumps({"answer": "parent answer"})

    model, stub = stub_anthropic(
        monkeypatch,
        [[call("task", "delegate", description="do work", subagent_type="worker")], child_turn, parent_turn],
    )
    spec: dict[str, Any] = {"name": "worker", "description": "Does work.", "mode": "fork"}
    if child_format is not None:
        spec["response_format"] = child_format
    agent = create_deep_agent(model=model, subagents=[spec], response_format=parent_format)
    result = invoke(agent, {"messages": [HumanMessage("start")]}, mode)

    parent, child, _ = stub.bodies
    assert (parent["tools"] == child["tools"]) is same_tools
    assert (parent.get("output_config") == child.get("output_config")) is same_output_config
    if isinstance(parent_format, ToolStrategy) and isinstance(child_format, ToolStrategy):
        assert parent.get("tool_choice") == child.get("tool_choice")
    assert parent["system"] == child["system"]
    task_result = next(message for message in result["messages"] if isinstance(message, ToolMessage) and message.tool_call_id == "delegate")
    if isinstance(expected_child, dict):
        assert json.loads(task_result.content) == expected_child
    else:
        assert task_result.content == expected_child
    if parent_format is not None:
        assert result["structured_response"] == ParentOutput(answer="parent answer")
    else:
        assert result["messages"][-1].text == "parent text"


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("dynamic_matches_parent", [True, False])
def test_dynamic_format_overrides_fork_spec(monkeypatch: pytest.MonkeyPatch, mode: str, *, dynamic_matches_parent: bool) -> None:
    dynamic_schema = ParentOutput if dynamic_matches_parent else ChildOutput
    static_schema = ChildOutput if dynamic_matches_parent else ParentOutput
    output = {"answer": "child answer"} if dynamic_matches_parent else {"detail": "child detail"}
    model, stub = stub_anthropic(
        monkeypatch,
        [
            [call("task", "delegate", description="do work", subagent_type="worker")],
            [call(dynamic_schema.__name__, "child_output", **output)],
            [call("ParentOutput", "parent_output", answer="parent answer")],
        ],
    )
    agent = create_deep_agent(
        model=model,
        response_format=ToolStrategy(ParentOutput),
        subagents=[{"name": "worker", "description": "Does work.", "mode": "fork", "response_format": ToolStrategy(static_schema)}],
    )
    result = invoke(
        agent,
        {"messages": [HumanMessage("start")]},
        mode,
        {"configurable": {SUBAGENT_RESPONSE_FORMAT_CONFIG_KEY: ToolStrategy(dynamic_schema)}},
    )

    parent, child, _ = stub.bodies
    assert (parent["tools"] == child["tools"]) is dynamic_matches_parent
    task_result = next(message for message in result["messages"] if isinstance(message, ToolMessage) and message.tool_call_id == "delegate")
    assert json.loads(task_result.content) == output
