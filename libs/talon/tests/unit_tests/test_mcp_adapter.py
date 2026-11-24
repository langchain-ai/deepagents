"""Exercise Talon's MCP migration against the real adapter without network access."""

from __future__ import annotations

import asyncio
import copy
import json
import os
import sys
from typing import TYPE_CHECKING

import pytest
from fastmcp import Context, FastMCP
from fastmcp.client.transports import FastMCPTransport
from fastmcp.exceptions import ToolError
from fastmcp.tools.function_tool import FunctionTool
from langchain.agents.middleware import wrap_model_call
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool, StructuredTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from mcp.client.session import ClientSession
from mcp.types import (
    ElicitRequest,
    ElicitRequestFormParams,
    ElicitRequestURLParams,
    InputRequiredResult,
)

from deepagents_talon import mcp
from deepagents_talon.authorization import current_authorization_invocation
from deepagents_talon.config import TalonConfig
from deepagents_talon.cron.scope import is_read_only
from deepagents_talon.interfaces import AgentRequest
from deepagents_talon.mcp_middleware import talon_mcp_middleware
from deepagents_talon.runtime import DeepAgentRuntime

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

    from langchain_core.messages import BaseMessage
    from langgraph.graph.state import CompiledStateGraph

    from deepagents_talon.interfaces import ToolApprovalRequest


async def _load_tools(
    server: FastMCP, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> mcp.MCPTools:
    async def connection(*_args: object) -> tuple[FastMCPTransport, str]:
        return FastMCPTransport(server), "stdio"

    path = tmp_path / "mcp.json"
    path.write_text(json.dumps({"mcpServers": {"remote": {"command": "unused"}}}))
    config = TalonConfig.from_env(
        {"AGENT_ASSISTANT_ID": "test", "DEEPAGENTS_TALON_MCP_CONFIG": str(path)},
        base_home=tmp_path,
    )
    monkeypatch.setattr(mcp, "_connection", connection)
    return await mcp.load_mcp_tools(config)


def _graph(tools: list[BaseTool]) -> CompiledStateGraph:
    builder = StateGraph(MessagesState)
    builder.add_node(
        "tools", ToolNode(tools, awrap_tool_call=talon_mcp_middleware().awrap_tool_call)
    )
    builder.add_edge(START, "tools")
    builder.add_edge("tools", END)
    return builder.compile(checkpointer=InMemorySaver())


@pytest.mark.parametrize("fail", [False, True])
async def test_real_adapter_invokes_prefixed_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, fail: bool
) -> None:
    server = FastMCP("test")
    received: list[tuple[str, str]] = []

    @server.tool
    async def search(query: str, optional: str = "default") -> str:
        received.append((query, optional))
        if fail:
            msg = "search unavailable"
            raise ToolError(msg)
        return "found"

    loaded = await _load_tools(server, tmp_path, monkeypatch)
    graph = _graph(list(loaded.tools))
    result = await graph.ainvoke(
        {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "call-42",
                            "name": "remote_search",
                            "args": {"query": "", "optional": ""},
                        }
                    ],
                )
            ]
        },
        {"configurable": {"thread_id": "test"}},
    )

    message = result["messages"][-1]
    assert received == [("", "default")]
    assert message.name == "remote_search"
    assert message.tool_call_id == "call-42"
    assert message.status == ("error" if fail else "success")
    assert message.content[0]["text"] == ("search unavailable" if fail else "found")
    assert current_authorization_invocation() is None


@pytest.mark.parametrize(
    "payload_schema",
    [
        {"type": "object"},
        {"type": "object", "properties": {}},
        {"type": ["object", "null"], "properties": {}},
    ],
)
async def test_open_arguments_survive_provider_conversion_and_invocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, payload_schema: dict[str, object]
) -> None:
    server = FastMCP("test")

    async def search(payload: dict[str, object]) -> dict[str, object]:
        return payload

    remote = FunctionTool.from_function(search)
    remote.parameters["properties"]["payload"] = payload_schema
    server.add_tool(remote)
    loaded = await _load_tools(server, tmp_path, monkeypatch)
    tool = loaded.tools[0]
    schema = convert_to_openai_tool(tool)["function"]["parameters"]
    assert schema["properties"]["payload"]["additionalProperties"] is True
    assert schema["properties"]["payload"]["type"] == payload_schema["type"]
    assert remote.parameters["properties"]["payload"] == payload_schema
    payload = {"page": 2, "filters": {"tags": ["open", "assigned"]}}

    result = await _graph(list(loaded.tools)).ainvoke(
        {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[{"id": "call-1", "name": tool.name, "args": {"payload": payload}}],
                )
            ]
        },
        {"configurable": {"thread_id": "test"}},
    )

    message = result["messages"][-1]
    assert message.status == "success"
    assert message.artifact["structured_content"] == payload


@pytest.mark.parametrize("responses", [False, True])
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_mcp_binding_preserves_optional_nested_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, responses: bool, asynchronous: bool
) -> None:
    server = FastMCP("audio")

    async def audio_create(
        lyrics: str, style: str, controls: dict[str, object] | None = None
    ) -> dict[str, object]:
        return {"lyrics": lyrics, "style": style, "controls": controls}

    remote = FunctionTool.from_function(audio_create, name="audio-create")
    remote.parameters = {
        "type": "object",
        "properties": {
            "lyrics": {"type": "string"},
            "style": {"type": "string"},
            "controls": {
                "type": ["object", "null"],
                "default": None,
                "properties": {
                    "audio_influence": {"type": "number", "default": 0},
                    "style_influence": {"type": ["number", "null"], "default": 0.5},
                },
            },
        },
        "required": ["lyrics", "style"],
    }
    original = copy.deepcopy(remote.parameters)
    server.add_tool(remote)
    loaded = await _load_tools(server, tmp_path, monkeypatch)
    tool = loaded.tools[0]
    local = StructuredTool(name="local", description="Local", args_schema=original)
    model = ChatOpenAI(model="gpt-4.1", api_key="test", use_responses_api=responses)
    request = ModelRequest(
        model=model, messages=[HumanMessage("write a song")], tools=[tool, local]
    )
    middleware = talon_mcp_middleware()

    def bind(passed: ModelRequest) -> ModelRequest:
        bound = model.bind_tools(passed.tools)
        payload = model._get_request_payload(passed.messages, **bound.kwargs)
        definitions = payload["tools"]
        remote_definition = definitions[0] if responses else definitions[0]["function"]
        local_definition = definitions[1] if responses else definitions[1]["function"]
        assert remote_definition.get("strict") is (False if responses else None)
        assert "strict" not in local_definition
        assert remote_definition["parameters"] == original
        assert local_definition["parameters"] == original
        return passed

    if asynchronous:

        async def handle(passed: ModelRequest) -> ModelRequest:
            return bind(passed)

        await middleware.awrap_model_call(request, handle)
    else:
        middleware.wrap_model_call(request, bind)
    assert tool.args_schema == original
    assert remote.parameters == original
    assert loaded.servers[0].tools[0].input_schema == original

    forwarded: list[dict[str, object]] = []
    call_tool = ClientSession.call_tool

    async def capture(client, name, arguments, **kwargs: object):
        forwarded.append(copy.deepcopy(arguments))
        return await call_tool(client, name, arguments, **kwargs)

    monkeypatch.setattr(ClientSession, "call_tool", capture)
    for controls in [
        {},
        {"controls": {}},
        {"controls": None},
        {"controls": {"style_influence": None}},
        {"controls": {"audio_influence": 0}},
    ]:
        arguments = {"lyrics": "A new day", "style": "folk", **controls}
        result = await _graph(list(loaded.tools)).ainvoke(
            {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[{"id": "audio", "name": tool.name, "args": arguments}],
                    )
                ]
            },
            {"configurable": {"thread_id": "audio"}},
        )
        assert result["messages"][-1].status == "success"
        assert forwarded[-1] == arguments


@pytest.mark.parametrize(
    ("provider", "responses"),
    [("openai", None), ("openai", True), ("openai", False), ("anthropic", None)],
)
async def test_runtime_discovers_and_executes_mcp_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, provider: str, *, responses: bool | None
) -> None:
    server = FastMCP("records")
    events: list[str] = []

    @server.tool
    async def search(query: str = "default") -> str:
        assert current_authorization_invocation() is not None
        events.append(query)
        return "found"

    loaded = await _load_tools(server, tmp_path, monkeypatch)
    if provider == "openai":
        model = ChatOpenAI(
            model="gpt-5.4",
            base_url="https://api.openai.com/v1",
            api_key="test",
            use_responses_api=responses,
        )
    else:
        model = ChatAnthropic(
            model="claude-sonnet-4-6", api_key="test", base_url="https://api.anthropic.com"
        )

    async def generate(
        self: ChatOpenAI | ChatAnthropic, messages: list[BaseMessage], **kwargs: object
    ) -> ChatResult:
        payload = self._get_request_payload(messages, tools=kwargs["tools"])
        definitions = payload["tools"]
        deferred = responses is not False
        assert (
            any(item["type"].startswith("tool_search") for item in definitions if "type" in item)
            is deferred
        )
        functions = {
            fn["name"]: fn for item in definitions if (fn := item.get("function", item)).get("name")
        }
        assert not functions["read_file"].get("defer_loading")
        remote = functions["remote_search"]
        assert bool(remote.get("defer_loading")) is deferred
        if deferred and provider == "openai":
            assert remote["strict"] is False
        message = (
            AIMessage(content="done")
            if events
            else AIMessage(
                content="", tool_calls=[{"id": "lookup", "name": "remote_search", "args": {}}]
            )
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    monkeypatch.setattr(type(model), "_agenerate", generate)
    monkeypatch.setattr(
        "deepagents_talon.runtime._resolve_model_from_env", lambda *_a, **_kw: model
    )
    runtime = DeepAgentRuntime(
        model="test:model",
        assistant_dir=tmp_path,
        tools=loaded.tools,
        middleware=(talon_mcp_middleware(),),
        include_web_tools=False,
        skills=(),
        memory=(),
    )
    runtime.approval_store.update({"remote_search": True}, runtime.approval_store.ensure().revision)

    async def approve(request: ToolApprovalRequest) -> str:
        events.extend(action["name"] for action in request.action_requests)
        return "approve"

    await runtime.start()
    try:
        result = await runtime.invoke(
            AgentRequest("chat", "find records", approval_handler=approve)
        )
        assert result.text == "done"
        assert events == ["remote_search", "default"]
    finally:
        await runtime.stop()


@pytest.mark.parametrize("grant_read", [True, False])
async def test_search_only_defers_visible_mcp_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, grant_read: bool
) -> None:
    server = FastMCP("records")

    @server.tool
    async def lookup() -> str:
        return "found"

    @server.tool
    async def write() -> str:
        return "written"

    loaded = await _load_tools(server, tmp_path, monkeypatch)
    model = ChatOpenAI(
        model="gpt-5.4",
        api_key="test",
        base_url="https://api.openai.com/v1",
        use_responses_api=True,
    )

    async def generate(
        self: ChatOpenAI, messages: list[BaseMessage], **kwargs: object
    ) -> ChatResult:
        definitions = self._get_request_payload(messages, tools=kwargs["tools"])["tools"]
        assert any(tool["type"] == "tool_search" for tool in definitions) is grant_read
        remote = {
            tool["name"]: tool for tool in definitions if tool.get("name", "").startswith("remote_")
        }
        assert set(remote) == ({"remote_lookup"} if grant_read else set())
        if grant_read:
            assert remote["remote_lookup"]["defer_loading"] is True
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="done"))])

    monkeypatch.setattr(ChatOpenAI, "_agenerate", generate)
    monkeypatch.setattr(
        "deepagents_talon.runtime._resolve_model_from_env", lambda *_a, **_kw: model
    )

    @wrap_model_call
    async def filter_tools(
        request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
    ) -> ModelResponse:
        names = {"read_file", "remote_lookup"} if grant_read else {"read_file"}
        allowed = [
            tool for tool in request.tools if isinstance(tool, BaseTool) and tool.name in names
        ]
        return await handler(request.override(tools=allowed))

    runtime = DeepAgentRuntime(
        model="test:model",
        assistant_dir=tmp_path,
        tools=loaded.tools,
        middleware=(talon_mcp_middleware(), filter_tools),
        include_web_tools=False,
        skills=(),
        memory=(),
    )
    await runtime.start()
    try:
        await runtime.invoke(AgentRequest("chat", "check records"))
    finally:
        await runtime.stop()


async def test_reload_keeps_inflight_tools_callable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = FastMCP("test")
    started, release = asyncio.Event(), asyncio.Event()

    @server.tool
    async def wait() -> str:
        started.set()
        await release.wait()
        return "done"

    loaded = await _load_tools(server, tmp_path, monkeypatch)
    async with asyncio.TaskGroup() as tasks:
        pending = tasks.create_task(loaded.tools[0].ainvoke({}))
        try:
            await started.wait()
            refreshed = await _load_tools(server, tmp_path, monkeypatch)
        finally:
            release.set()

    assert pending.result()[0]["text"] == "done"
    assert (await refreshed.tools[0].ainvoke({}))[0]["text"] == "done"


async def test_real_elicitation_cancels_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = FastMCP("test")

    @server.tool
    async def ask(ctx: Context) -> InputRequiredResult | str:
        if ctx.input_responses is None:
            return InputRequiredResult(
                input_requests={
                    "question": ElicitRequest(
                        params=ElicitRequestFormParams(
                            message="Enter a value",
                            requested_schema={"type": "object", "properties": {}},
                        )
                    ),
                    "url": ElicitRequest(
                        params=ElicitRequestURLParams(
                            mode="url",
                            message="Open a page",
                            url="https://example.com/input",
                        )
                    ),
                }
            )
        assert set(ctx.input_responses) == {"question", "url"}
        assert all(response.action == "cancel" for response in ctx.input_responses.values())
        return "cancelled"

    loaded = await _load_tools(server, tmp_path, monkeypatch)
    graph = _graph(list(loaded.tools))
    config = {"configurable": {"thread_id": "test"}}
    state = await graph.ainvoke(
        {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "call-1",
                            "name": "remote_ask",
                            "args": {},
                        }
                    ],
                )
            ]
        },
        config,
    )
    assert state["__interrupt__"][0].value["type"] == "mcp_elicitation"
    assert current_authorization_invocation() is None

    async def unexpected_approval(_request: object) -> None:
        pytest.fail("MCP input must not be handled as tool approval")

    runtime = DeepAgentRuntime(model="unused")
    resume = await runtime._build_approval_resume(
        AgentRequest(conversation_id="test", text="test", approval_handler=unexpected_approval),
        state["__interrupt__"],
    )
    result = await graph.ainvoke(resume, config)

    assert not result.get("__interrupt__")
    assert result["messages"][-1].content[0]["text"] == "cancelled"
    assert current_authorization_invocation() is None


@pytest.mark.parametrize("requests", [None, [], [{}], [{"key": "x"}, {"key": "x"}]])
def test_malformed_elicitation_is_rejected(requests: object) -> None:
    with pytest.raises(ValueError, match="MCP elicitation interrupt"):
        mcp._cancel_mcp_elicitation({"type": "mcp_elicitation", "requests": requests})


@pytest.mark.timeout(30)
async def test_stdio_processes_exit_after_discovery_and_invocation(tmp_path: Path) -> None:
    pidfile = tmp_path / "server.pid"
    script = tmp_path / "server.py"
    script.write_text(
        "import os, sys\n"
        "from pathlib import Path\n"
        "from fastmcp import FastMCP\n"
        "Path(sys.argv[1]).write_text(str(os.getpid()))\n"
        "server = FastMCP('lifecycle')\n"
        "@server.tool\n"
        "def ping() -> str:\n"
        "    return 'pong'\n"
        "server.run(transport='stdio', show_banner=False)\n"
    )
    transport = mcp._stdio_connection(
        "local", {"command": sys.executable, "args": [str(script), str(pidfile)]}
    )
    adapter = mcp.MCPAdapter(transport)
    try:
        tools = await adapter.list_tools()
        with pytest.raises(ProcessLookupError):
            os.kill(int(pidfile.read_text()), 0)
        for _ in range(2):
            assert (await tools[0].ainvoke({}))[0]["text"] == "pong"
            with pytest.raises(ProcessLookupError):
                os.kill(int(pidfile.read_text()), 0)
    finally:
        await mcp.FastMCPClient(transport).close()


@pytest.mark.parametrize("official", [False, True])
async def test_unannotated_langsmith_reads_require_official_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, official: bool
) -> None:
    server = FastMCP("langsmith")

    @server.tool
    def fetch_runs() -> str:
        return "runs"

    @server.tool
    def create_run() -> str:
        return "created"

    async def connection(*_args: object) -> tuple[FastMCPTransport, str]:
        return FastMCPTransport(server), "streamable_http"

    url = "https://api.smith.langchain.com/mcp" if official else "https://example.org/mcp"
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps({"mcpServers": {"langsmith": {"url": url}}}))
    monkeypatch.setattr(mcp, "_connection", connection)
    config = TalonConfig.from_env({"DEEPAGENTS_TALON_MCP_CONFIG": str(path)}, base_home=tmp_path)
    loaded = await mcp.load_mcp_tools(config)
    assert {t.name: is_read_only(t) for t in loaded.tools} == {
        "langsmith_fetch_runs": official,
        "langsmith_create_run": False,
    }
