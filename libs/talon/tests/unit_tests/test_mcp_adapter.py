"""Exercise Talon's MCP migration against the real adapter without network access."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import TYPE_CHECKING

import pytest
from fastmcp import Context, FastMCP
from fastmcp.client.transports import FastMCPTransport
from fastmcp.exceptions import ToolError
from fastmcp.tools.function_tool import FunctionTool
from langchain_core.messages import AIMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from mcp.types import (
    ElicitRequest,
    ElicitRequestFormParams,
    ElicitRequestURLParams,
    InputRequiredResult,
)

from deepagents_talon import mcp
from deepagents_talon.authorization import current_authorization_invocation
from deepagents_talon.config import TalonConfig
from deepagents_talon.interfaces import AgentRequest
from deepagents_talon.mcp_middleware import talon_mcp_middleware
from deepagents_talon.runtime import DeepAgentRuntime

if TYPE_CHECKING:
    from pathlib import Path

    from langchain_core.tools import BaseTool
    from langgraph.graph.state import CompiledStateGraph


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
