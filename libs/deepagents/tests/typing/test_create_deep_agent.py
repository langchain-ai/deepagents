"""Static typing tests for create_deep_agent overloads.

Checked by mypy; assert_type calls are no-ops at runtime so pytest collects
this file without executing any real agent logic.
"""

from typing import Any, assert_type

from langchain.agents import AgentState
from langchain.agents.middleware.types import _InputAgentState, _OutputAgentState
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from deepagents import create_deep_agent
from deepagents.graph import DeepAgentState


class _MyState(DeepAgentState):
    custom_field: str
    tags: list[str]


class _MyResponse(BaseModel):
    answer: str


# Overload 1: explicit state_schema narrows to the subclass.
agent_custom = create_deep_agent("anthropic:claude-sonnet-4-6", state_schema=_MyState)
assert_type(
    agent_custom,
    CompiledStateGraph[_MyState, None, _InputAgentState, _OutputAgentState[Any]],
)

# Overload 2: no state_schema + response_format narrows structured_response.
agent_response = create_deep_agent("anthropic:claude-sonnet-4-6", response_format=_MyResponse)
assert_type(
    agent_response,
    CompiledStateGraph[AgentState[_MyResponse], None, _InputAgentState, _OutputAgentState[_MyResponse]],
)

# Overload 2: no args — state is AgentState[Any], response Any.
agent_default = create_deep_agent("anthropic:claude-sonnet-4-6")
assert_type(
    agent_default,
    CompiledStateGraph[AgentState[Any], None, _InputAgentState, _OutputAgentState[Any]],
)
