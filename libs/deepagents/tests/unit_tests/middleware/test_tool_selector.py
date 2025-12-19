from __future__ import annotations

from typing import Any

from langchain.agents.middleware.types import ModelRequest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from deepagents.middleware.tool_selector import ToolSelectorConfig, ToolSelectorMiddleware


@tool(description="List files in a directory.")
def ls(path: str) -> str:
    return path


@tool(description="Read a file by path.")
def read_file(path: str) -> str:
    return path


@tool(description="Alpha tool for alpha keyword.")
def alpha_tool(query: str) -> str:
    return query


@tool(description="Beta tool for beta keyword.")
def beta_tool(query: str) -> str:
    return query


@tool(description="Extra tool added by other middleware.")
def unindexed_tool(query: str) -> str:
    return query


def _build_request(messages: list[Any], tools: list[Any]) -> ModelRequest:
    model = GenericFakeChatModel(messages=iter([]))
    return ModelRequest(model=model, messages=messages, tools=tools)


def test_tool_selector_keeps_always_tools_and_selects_top_k() -> None:
    tools = [ls, read_file, alpha_tool, beta_tool]
    config = ToolSelectorConfig(k=1, fallback_k=2, last_n_messages=1, always_tool_names=("ls", "read_file"))
    middleware = ToolSelectorMiddleware(tools=tools, config=config)
    request = _build_request(messages=[HumanMessage(content="Use alpha tool")], tools=tools)

    selected_tools = middleware.wrap_model_call(request, lambda req: req.tools)
    selected_names = {tool.name for tool in selected_tools}

    assert "ls" in selected_names
    assert "read_file" in selected_names
    assert "alpha_tool" in selected_names
    assert "beta_tool" not in selected_names


def test_tool_selector_preserves_unindexed_tools_by_default() -> None:
    tools = [alpha_tool]
    config = ToolSelectorConfig(k=1, fallback_k=1, last_n_messages=1, always_tool_names=())
    middleware = ToolSelectorMiddleware(tools=tools, config=config)
    request = _build_request(messages=[HumanMessage(content="alpha")], tools=[alpha_tool, unindexed_tool])

    selected_tools = middleware.wrap_model_call(request, lambda req: req.tools)
    selected_names = {tool.name for tool in selected_tools}

    assert "alpha_tool" in selected_names
    assert "unindexed_tool" in selected_names


def test_tool_selector_allows_empty_selection() -> None:
    tools = [alpha_tool]
    config = ToolSelectorConfig(
        k=0,
        fallback_k=0,
        last_n_messages=1,
        always_tool_names=(),
        allow_empty_selection=True,
    )
    middleware = ToolSelectorMiddleware(tools=tools, config=config)
    request = _build_request(messages=[HumanMessage(content="alpha")], tools=[alpha_tool])

    selected_tools = middleware.wrap_model_call(request, lambda req: req.tools)

    assert selected_tools == []
