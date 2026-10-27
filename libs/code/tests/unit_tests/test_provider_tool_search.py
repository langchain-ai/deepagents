"""Network-free native-search request and execution coverage."""

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import httpx
import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from deepagents_code._cli_context import CLIContextSchema
from deepagents_code.configurable_model import ConfigurableModelMiddleware
from deepagents_code.provider_tool_search import MCPProviderToolSearchMiddleware

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig


@tool
def core_read() -> str:
    """Read the local source."""
    return "source"


@tool
def mcp_lookup(query: str) -> str:
    """Look up a remote document."""
    return f"document: {query}"


def response(output: list[dict]) -> dict:
    return {
        "id": "resp_test",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "model": "gpt-5.4",
        "output": output,
        "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
    }


def text_output() -> list[dict]:
    return [
        {
            "type": "message",
            "id": "msg_test",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "done", "annotations": []}],
        }
    ]


def model_with_transport(
    payloads: list[dict],
    outputs: list[list[dict]],
    *,
    base_url: str | None = None,
    model_name: str = "gpt-5.4",
) -> ChatOpenAI:
    def handle(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        body = response(outputs.pop(0))
        body["id"] = f"resp_{len(payloads)}"
        return httpx.Response(200, json=body)

    return ChatOpenAI(
        model=model_name,
        base_url=base_url,
        api_key="fake",
        http_client=httpx.Client(transport=httpx.MockTransport(handle)),
    )


@pytest.mark.parametrize(
    ("base_url", "model_name"),
    [
        (None, "gpt-5.4"),
        ("https://gateway.smith.langchain.com/openai/v1", "gpt-6.1-sol"),
        ("https://gateway.smith.langchain.com/v1", "openai/gpt-6.1-sol"),
    ],
)
def test_serialization_execution_approval_and_resume(
    base_url: str | None, model_name: str
) -> None:
    payloads: list[dict] = []
    search = [
        {
            "type": "tool_search_call",
            "id": "ts_1",
            "status": "completed",
            "arguments": {"query": "document"},
            "execution": "server",
        },
        {
            "type": "tool_search_output",
            "id": "tso_1",
            "status": "completed",
            "tools": [],
            "execution": "server",
        },
        {
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "mcp_lookup",
            "arguments": '{"query":"hello"}',
            "status": "completed",
        },
    ]
    model = model_with_transport(
        payloads,
        [search, text_output(), text_output()],
        base_url=base_url,
        model_name=model_name,
    )
    saver = InMemorySaver()
    graph = create_agent(
        model,
        tools=[core_read, mcp_lookup],
        middleware=[
            MCPProviderToolSearchMiddleware([mcp_lookup]),
            HumanInTheLoopMiddleware(interrupt_on={"mcp_lookup": True}),
        ],
        checkpointer=saver,
    )
    config: RunnableConfig = {"configurable": {"thread_id": "native-search"}}
    paused = graph.invoke({"messages": [HumanMessage("find hello")]}, config)
    assert paused["__interrupt__"]
    assert not any(
        getattr(message, "type", "") == "tool" for message in paused["messages"]
    )
    tools = payloads[0]["tools"]
    assert tools[0]["name"] == "core_read"
    assert not tools[0].get("defer_loading")
    assert tools[1]["name"] == "mcp_lookup"
    assert tools[1]["defer_loading"] is True
    assert tools[-1] == {"type": "tool_search"}
    assert "parameters" in tools[1]
    assert not (mcp_lookup.extras or {}).get("defer_loading")
    approved = graph.invoke(
        Command(resume={"decisions": [{"type": "approve"}]}), config
    )
    assert any(
        getattr(message, "content", None) == "document: hello"
        for message in approved["messages"]
    )
    assert any(item["type"] == "tool_search_output" for item in payloads[1]["input"])
    rebuilt = create_agent(
        model,
        tools=[core_read, mcp_lookup],
        middleware=[MCPProviderToolSearchMiddleware([mcp_lookup])],
        checkpointer=saver,
    )
    rebuilt.invoke({"messages": [HumanMessage("continue")]}, config)
    assert any(item["type"] == "tool_search_output" for item in payloads[2]["input"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model": "gpt-4.1"},
        {"model": "gpt-unknown"},
        {
            "model": "gpt-5.4",
            "base_url": "https://gateway.smith.langchain.com/openai/v1",
            "use_responses_api": False,
        },
        {"model": "gpt-5.4", "use_responses_api": False},
        {
            "model": "google/gemini-3-pro",
            "base_url": "https://gateway.smith.langchain.com/v1",
        },
        {
            "model": "gpt-6.1-sol",
            "base_url": "https://gateway.smith.langchain.com/google/v1",
        },
        {
            "model": "gpt-6.1-sol",
            "base_url": "https://gateway.smith.langchain.com.evil.example/openai/v1",
        },
    ],
)
def test_unsupported_requests_remain_eager(kwargs: dict) -> None:
    middleware = MCPProviderToolSearchMiddleware([mcp_lookup])
    request = ModelRequest(
        model=ChatOpenAI(api_key="fake", **kwargs),
        tools=[core_read, mcp_lookup],
        messages=[],
    )
    prepared = middleware._prepare(request)
    assert prepared.tools == [core_read, mcp_lookup]
    assert not (mcp_lookup.extras or {}).get("defer_loading")


def test_switching_and_filtered_tools() -> None:
    middleware = MCPProviderToolSearchMiddleware([mcp_lookup])
    model = ChatOpenAI(model="gpt-5.4", api_key="fake")
    request = ModelRequest(model=model, tools=[core_read, mcp_lookup], messages=[])
    native = middleware._prepare(request)
    assert native.tools[-1] == {"type": "tool_search"}
    unsupported = request.override(model=MagicMock())
    assert middleware._prepare(unsupported).tools == [core_read, mcp_lookup]
    assert middleware._prepare(request.override(tools=[core_read])).tools == [core_read]
    assert middleware._prepare(request).tools[-1] == {"type": "tool_search"}


def test_cross_provider_search_history_is_not_replayed() -> None:
    original = AIMessage(
        content=[
            {"type": "tool_search_call", "id": "ts_1"},
            {"type": "tool_search_output", "id": "tso_1"},
            {"type": "text", "text": "found"},
        ]
    )
    request = ModelRequest(model=MagicMock(), tools=[mcp_lookup], messages=[original])
    prepared = MCPProviderToolSearchMiddleware([mcp_lookup])._prepare(request)
    assert prepared.messages[0].content == [{"type": "text", "text": "found"}]
    assert len(original.content) == 3


def test_runtime_model_switch_serialization(monkeypatch: pytest.MonkeyPatch) -> None:
    payloads: list[dict] = []
    model = model_with_transport(
        payloads, [text_output(), text_output(), text_output()]
    )
    unsupported = model.model_copy(
        update={"model_name": "gpt-4.1", "use_responses_api": True}
    )
    resolved = MagicMock(model=unsupported, model_spec="openai:gpt-4.1")
    monkeypatch.setattr(
        "deepagents_code.config.create_model", lambda *_args, **_kwargs: resolved
    )
    graph = create_agent(
        model,
        tools=[core_read, mcp_lookup],
        middleware=[
            ConfigurableModelMiddleware(
                persist_model_state=False, openai_prompt_cache_key=False
            ),
            MCPProviderToolSearchMiddleware([mcp_lookup]),
        ],
    )
    graph.invoke({"messages": [HumanMessage("hello")]})
    from deepagents_code.configurable_model import _apply_overrides

    switched = _apply_overrides(
        ModelRequest(
            model=model,
            tools=[core_read, mcp_lookup],
            messages=[],
            runtime=MagicMock(context=CLIContextSchema(model="openai:gpt-4.1")),
        ),
        openai_prompt_cache_key=False,
        cli_max_retries=None,
    )
    switched_graph = create_agent(
        switched.request.model,
        tools=[core_read, mcp_lookup],
        middleware=[MCPProviderToolSearchMiddleware([mcp_lookup])],
    )
    switched_graph.invoke({"messages": [HumanMessage("hello")]})
    graph.invoke({"messages": [HumanMessage("hello")]})
    assert payloads[0]["tools"][-1] == {"type": "tool_search"}
    assert all(tool.get("type") != "tool_search" for tool in payloads[1]["tools"])
    assert all(not tool.get("defer_loading") for tool in payloads[1]["tools"])
    assert payloads[2]["tools"][-1] == {"type": "tool_search"}


@pytest.mark.parametrize("setting", [None, "false"])
def test_cli_graph_default_and_opt_out(
    tmp_path, monkeypatch: pytest.MonkeyPatch, setting: str | None
) -> None:
    from deepagents_code.agent import create_cli_agent

    if setting is None:
        monkeypatch.delenv("DEEPAGENTS_CODE_PROVIDER_TOOL_SEARCH", raising=False)
    else:
        monkeypatch.setenv("DEEPAGENTS_CODE_PROVIDER_TOOL_SEARCH", setting)
    payloads: list[dict] = []
    model = model_with_transport(payloads, [text_output()])
    model.use_responses_api = True
    graph, _ = create_cli_agent(
        model,
        "native-search-test",
        cwd=tmp_path,
        tools=[mcp_lookup],
        mcp_tools=[mcp_lookup],
        auto_approve=True,
        enable_memory=False,
        enable_skills=False,
        enable_shell=False,
    )
    graph.invoke({"messages": [HumanMessage("hello")]})
    tools = payloads[0]["tools"]
    deferred = setting is None
    assert (
        bool(
            next(tool for tool in tools if tool.get("name") == "mcp_lookup").get(
                "defer_loading"
            )
        )
        is deferred
    )
    assert any(tool.get("type") == "tool_search" for tool in tools) is deferred
    assert not next(tool for tool in tools if tool.get("name") == "read_file").get(
        "defer_loading"
    )
