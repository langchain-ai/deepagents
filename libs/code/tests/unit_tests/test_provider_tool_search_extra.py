"""Provider serialization, streaming, and interpreter deferral."""

import json
import warnings
from unittest.mock import MagicMock

import anthropic
import httpx
import pytest
from langchain.agents import create_agent
from langchain.agents.middleware.types import AgentState, ModelRequest
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from deepagents_code._js_cost import CostAwareCodeInterpreterMiddleware
from deepagents_code.provider_tool_search import MCPProviderToolSearchMiddleware


@tool
def mcp_lookup(query: str) -> str:
    """REMOTE_SCHEMA_SENTINEL: find a document."""
    return query


@pytest.mark.filterwarnings(
    "default:The model 'claude-(sonnet-4-20250514|opus-4-20250514|opus-4-1-20250805)' "
    "is deprecated:DeprecationWarning"
)
@pytest.mark.parametrize(
    ("model_name", "deferred"),
    [
        ("claude-sonnet-4-20250514", False),
        ("claude-opus-4-20250514", False),
        ("claude-opus-4-1-20250805", False),
        ("claude-opus-4.1", False),
        ("claude-sonnet-4-5", True),
        ("claude-sonnet-4-5-20250929", True),
        ("claude-opus-4-5-20251101", True),
        ("claude-haiku-4-5-20251001", True),
        ("claude-sonnet-4-6", True),
        ("claude-opus-4.8", True),
        ("claude-sonnet-5", True),
        ("claude-sonnet-5-5", True),
        ("claude-opus-5", True),
        ("claude-opus-5-5", True),
        ("claude-haiku-5-5", True),
        ("claude-fable-5-1", True),
        ("claude-mythos-5-1", True),
    ],
)
def test_anthropic_transport(model_name: str, *, deferred: bool) -> None:
    payloads = []

    def handle(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": model_name,
                "content": [{"type": "text", "text": "done"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 1},
            },
        )

    model = ChatAnthropic(
        model=model_name,
        api_key="fake",
        max_tokens=128,
    )
    model.__dict__["_client"] = anthropic.Anthropic(
        api_key="fake", http_client=httpx.Client(transport=httpx.MockTransport(handle))
    )
    graph = create_agent(
        model,
        tools=[mcp_lookup],
        middleware=[MCPProviderToolSearchMiddleware([mcp_lookup])],
    )
    graph.invoke({"messages": [HumanMessage("hello")]})
    tools = payloads[0]["tools"]
    if deferred:
        assert tools[0]["defer_loading"] is True
        assert tools[-1]["type"] == "tool_search_tool_bm25_20251119"
    else:
        assert len(tools) == 1
        assert not tools[0].get("defer_loading")
        assert tools[0]["name"] == "mcp_lookup"


def test_quickjs_bindings_without_schema_reference() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        interpreter = CostAwareCodeInterpreterMiddleware(ptc=["mcp_lookup"])

    class InterpreterState(AgentState):
        _quickjs_slot_id: str

    state: InterpreterState = {"messages": [], "_quickjs_slot_id": "test-search"}
    request: ModelRequest = ModelRequest(
        model=ChatOpenAI(model="gpt-5.4", api_key="fake"),
        tools=[mcp_lookup],
        messages=[],
        state=state,
    )
    prepared = MCPProviderToolSearchMiddleware([mcp_lookup])._prepare(request)
    prompt = interpreter._prepare_for_call(prepared)
    slot = interpreter._slot_id(request.state)
    assert interpreter._ptc_tools_by_slot[slot][0].name == "mcp_lookup"
    assert "REMOTE_SCHEMA_SENTINEL" not in prompt
    eager_prompt = interpreter._prepare_for_call(request.override(model=MagicMock()))
    assert "REMOTE_SCHEMA_SENTINEL" in eager_prompt


def test_openai_search_stream_roundtrip() -> None:
    item = {
        "type": "tool_search_call",
        "id": "ts_1",
        "status": "completed",
        "arguments": {"query": "document"},
        "execution": "server",
    }
    output = {
        "type": "tool_search_output",
        "id": "tso_1",
        "status": "completed",
        "tools": [],
        "execution": "server",
    }
    body = {
        "id": "resp_stream",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "model": "gpt-5.4",
        "output": [item, output],
        "usage": {"input_tokens": 10, "output_tokens": 1, "total_tokens": 11},
    }
    events = [
        {
            "type": "response.created",
            "response": {**body, "output": [], "status": "in_progress"},
        },
        {"type": "response.output_item.added", "output_index": 0, "item": item},
        {"type": "response.output_item.done", "output_index": 0, "item": item},
        {"type": "response.output_item.added", "output_index": 1, "item": output},
        {"type": "response.output_item.done", "output_index": 1, "item": output},
        {"type": "response.completed", "response": body},
    ]
    payloads = []

    def handle(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        data = "".join(
            f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events
        )
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=data
        )

    model = ChatOpenAI(
        model="gpt-5.4",
        api_key="fake",
        output_version="responses/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(handle)),
    )
    request = MCPProviderToolSearchMiddleware([mcp_lookup])._prepare(
        ModelRequest(model=model, tools=[mcp_lookup], messages=[HumanMessage("hello")])
    )
    chunks = list(model.bind_tools(request.tools).stream(request.messages))
    aggregate = chunks[0]
    for chunk in chunks[1:]:
        aggregate += chunk
    assert isinstance(aggregate, AIMessage)
    assert not aggregate.tool_calls
    assert any(
        block.get("type") == "tool_search_output"
        for block in aggregate.content
        if isinstance(block, dict)
    )
    replay = model._get_request_payload(
        [HumanMessage("hello"), aggregate, HumanMessage("continue")],
        tools=payloads[0]["tools"],
    )
    assert any(block["type"] == "tool_search_output" for block in replay["input"])


@pytest.mark.parametrize(
    "model_name", ["gpt-6-astra", "gpt-6-sol", "gpt-6-luna", "gpt-6.1-sol"]
)
async def test_async_provider_request(model_name: str) -> None:
    payloads = []

    def handle(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "resp_async",
                "object": "response",
                "created_at": 0,
                "status": "completed",
                "model": model_name,
                "output": [],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        model = ChatOpenAI(model=model_name, api_key="fake", http_async_client=client)
        graph = create_agent(
            model,
            tools=[mcp_lookup],
            middleware=[MCPProviderToolSearchMiddleware([mcp_lookup])],
        )
        await graph.ainvoke({"messages": [HumanMessage("hello")]})
    assert payloads[0]["tools"][0]["defer_loading"] is True
    assert payloads[0]["tools"][-1] == {"type": "tool_search"}
