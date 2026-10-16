from __future__ import annotations

import asyncio
import os
import py_compile
import sys
from types import SimpleNamespace

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from deepagents_talon.interfaces import AgentRequest
from deepagents_talon.local_tools import LocalToolError
from deepagents_talon.runtime import DeepAgentRuntime
from tests.unit_tests.test_local_tools import _TOOL_MODULE


@pytest.fixture
def runtime_factory(tmp_path, monkeypatch):
    created = []

    def create_graph(**kwargs: object):
        graph = SimpleNamespace(tools={item.name: item for item in kwargs["tools"]})
        created.append(graph)
        return graph

    monkeypatch.setattr("deepagents_talon.runtime.create_deep_agent", create_graph)
    monkeypatch.setattr(
        "deepagents_talon.runtime._resolve_model_from_env",
        lambda *_a, **_k: FakeMessagesListChatModel(responses=[AIMessage(content="done")]),
    )

    def make_runtime(**kwargs: object):
        return DeepAgentRuntime(
            model="test:model",
            tools_dirs=[tmp_path],
            env={},
            skills=(),
            memory=(),
            include_web_tools=False,
            **kwargs,
        )

    return make_runtime, created


async def test_reload_updates_code_and_schema_and_adds_removes_tools(tmp_path, runtime_factory):
    make_runtime, created = runtime_factory
    path = tmp_path / "tools.py"
    path.write_text(_TOOL_MODULE, encoding="utf-8")
    runtime = make_runtime()
    await runtime.start()
    previous = created[-1]
    try:
        path.write_text(
            _TOOL_MODULE.replace("value: str", "name: str").replace(
                "return value", "return name.upper()"
            ),
            encoding="utf-8",
        )
        added = tmp_path / "added.py"
        added.write_text(_TOOL_MODULE.replace("local_echo", "added_echo"), encoding="utf-8")
        await runtime.reload_local_tools()
        current = created[-1]
        assert set(current.tools["local_echo"].args) == {"name"}
        assert current.tools["local_echo"].invoke({"name": "new"}) == "NEW"
        assert current.tools["added_echo"].invoke({"value": "added"}) == "added"
        assert previous.tools["local_echo"].invoke({"value": "old"}) == "old"
        added.unlink()
        await runtime.reload_local_tools()
        assert "added_echo" not in created[-1].tools
        path.unlink()
        await runtime.reload_local_tools()
        assert "local_echo" not in created[-1].tools
    finally:
        await runtime.stop()


async def test_reload_ignores_same_timestamp_same_size_bytecode(tmp_path, runtime_factory):
    make_runtime, created = runtime_factory
    path = tmp_path / "tools.py"
    before = _TOOL_MODULE.replace("return value", "return 'old'")
    after = before.replace("'old'", "'new'")
    path.write_text(before, encoding="utf-8")
    timestamp = path.stat()
    py_compile.compile(str(path), doraise=True)
    runtime = make_runtime()
    await runtime.start()
    try:
        path.write_text(after, encoding="utf-8")
        os.utime(path, ns=(timestamp.st_atime_ns, timestamp.st_mtime_ns))
        await runtime.reload_local_tools()
        assert created[-1].tools["local_echo"].invoke({"value": "unused"}) == "new"
    finally:
        await runtime.stop()


@pytest.mark.parametrize(
    "source",
    ["raise RuntimeError('private-secret')", _TOOL_MODULE.replace("local_echo", "task")],
)
async def test_failed_reload_preserves_graph_inventory_and_modules(
    tmp_path, runtime_factory, source
):
    make_runtime, created = runtime_factory
    path = tmp_path / "tools.py"
    path.write_text(_TOOL_MODULE, encoding="utf-8")
    runtime = make_runtime()
    await runtime.start()
    previous = created[-1]
    modules = set(sys.modules)
    try:
        path.write_text(source, encoding="utf-8")
        with pytest.raises(LocalToolError):
            await runtime.reload_local_tools()
        assert runtime._graph is previous
        assert previous.tools["local_echo"].invoke({"value": "usable"}) == "usable"
        inventory = await previous.tools["get_agent_tools"].ainvoke({})
        assert inventory["saved_changes_inactive"] is True
        assert not any(
            name.startswith("_talon_local_tools_") for name in set(sys.modules) - modules
        )
        path.write_text(_TOOL_MODULE, encoding="utf-8")
        await runtime.reload_local_tools()
        inventory = await created[-1].tools["get_agent_tools"].ainvoke({})
        assert inventory["saved_changes_inactive"] is False
    finally:
        await runtime.stop()


async def test_reload_revalidates_subagent_attachments(tmp_path, runtime_factory):
    make_runtime, created = runtime_factory
    path = tmp_path / "tools.py"
    path.write_text(_TOOL_MODULE, encoding="utf-8")
    runtime = make_runtime(
        subagents=[
            {
                "name": "local",
                "description": "Local",
                "system_prompt": "Local",
                "tool_names": ["local_echo"],
            }
        ]
    )
    await runtime.start()
    previous = created[-1]
    try:
        path.unlink()
        with pytest.raises(ValueError, match="Subagent attachment is unavailable"):
            await runtime.reload_local_tools()
        assert runtime._graph is previous
        path.write_text(
            _TOOL_MODULE.replace("return value", "return value.upper()"), encoding="utf-8"
        )
        await runtime.reload_local_tools()
        assert runtime._attachments[1]["tools"] == ["local_echo"]
        assert created[-1].tools["local_echo"].invoke({"value": "new"}) == "NEW"
    finally:
        await runtime.stop()


async def test_reload_preserves_mcp_tools_and_latest_generation_across_refresh(
    tmp_path, runtime_factory
):
    make_runtime, created = runtime_factory
    path = tmp_path / "tools.py"
    path.write_text(_TOOL_MODULE, encoding="utf-8")

    @tool
    def mcp_echo(value: str) -> str:
        """Echo via MCP."""
        return value

    async def refresh():
        return [mcp_echo]

    runtime = make_runtime(tools=[mcp_echo], refresh_tools=refresh, reload_tools=refresh)
    await runtime.start()
    try:
        path.write_text(
            _TOOL_MODULE.replace("return value", "return value.upper()"), encoding="utf-8"
        )
        await runtime.reload_local_tools()
        latest = created[-1].tools["local_echo"]
        assert created[-1].tools["mcp_echo"] is mcp_echo
        await runtime._refresh_runtime_tools()
        await runtime.reload_mcp_configuration()
        assert created[-1].tools["mcp_echo"] is mcp_echo
        assert created[-1].tools["local_echo"] is latest
    finally:
        await runtime.stop()


async def test_reload_keeps_active_turn_on_original_graph(tmp_path, runtime_factory, monkeypatch):
    make_runtime, created = runtime_factory
    path = tmp_path / "tools.py"
    path.write_text(_TOOL_MODULE, encoding="utf-8")
    runtime = make_runtime()
    await runtime.start()
    started, release = asyncio.Event(), asyncio.Event()

    async def invoke(_request, _activity):
        pinned = runtime._invocation_graph.get()
        started.set()
        await release.wait()
        assert pinned is runtime._invocation_graph.get()
        return pinned.tools["local_echo"].invoke({"value": "response"})

    monkeypatch.setattr(runtime, "_invoke_until_text", invoke)
    turn = asyncio.create_task(runtime.invoke(AgentRequest("chat", "hello")))
    try:
        await asyncio.wait_for(started.wait(), timeout=1)
        path.write_text(
            _TOOL_MODULE.replace("return value", "return value.upper()"), encoding="utf-8"
        )
        await runtime.reload_local_tools()
        release.set()
        assert (await turn).text == "response"
        assert (await runtime.invoke(AgentRequest("chat", "next"))).text == "RESPONSE"
        assert len(created) == 2
    finally:
        release.set()
        await turn
        await runtime.stop()


async def test_shutdown_releases_all_successful_generations(tmp_path, runtime_factory):
    make_runtime, _created = runtime_factory
    (tmp_path / "tools.py").write_text(_TOOL_MODULE, encoding="utf-8")
    modules = set(sys.modules)
    runtime = make_runtime()
    await runtime.start()
    await runtime.reload_local_tools()
    await runtime.reload_local_tools()
    added = {name for name in set(sys.modules) - modules if name.startswith("_talon_local_tools_")}
    assert len(added) == 3
    await runtime.stop()
    assert not added.intersection(sys.modules)
