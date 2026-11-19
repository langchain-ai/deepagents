"""Synchronous tool execution with the cost-aware JavaScript middleware."""

from typing import Literal

import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from deepagents_code._fake_models import _ToolBindingFakeModel
from deepagents_code._js_cost import CostAwareCodeInterpreterMiddleware


@tool
def ordinary_tool() -> str:
    """Return a result without using JavaScript."""
    return "ordinary result"


@pytest.mark.filterwarnings("ignore:The class `CodeInterpreterMiddleware` is in beta")
@pytest.mark.parametrize("mode", ["thread", "turn", "call"])
@pytest.mark.parametrize("streaming", [False, True], ids=["invoke", "stream"])
@pytest.mark.parametrize("tool_name", ["js_eval", "ordinary_tool"])
def test_sync_tool_execution(
    mode: Literal["thread", "turn", "call"], streaming: bool, tool_name: str
) -> None:
    interpreter = CostAwareCodeInterpreterMiddleware(tool_name="js_eval", mode=mode)
    model = _ToolBindingFakeModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": tool_name,
                            "args": {"code": "1+1"} if tool_name == "js_eval" else {},
                            "id": "test-call",
                        }
                    ],
                ),
                AIMessage(content="done"),
            ]
        ),
        disable_streaming=True,
    )
    try:
        agent = create_agent(model, tools=[ordinary_tool], middleware=[interpreter])
        inputs = {"messages": [("user", "Run the tool.")]}
        if streaming:
            result = list(agent.stream(inputs, stream_mode="values"))[-1]
        else:
            result = agent.invoke(inputs)
        messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        assert len(messages) == 1
        assert messages[0].status == "success"
        assert messages[0].tool_call_id == "test-call"
        assert messages[0].content == (
            "<result>2</result>" if tool_name == "js_eval" else "ordinary result"
        )
    finally:
        interpreter._registry.close()
