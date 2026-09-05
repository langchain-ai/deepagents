from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain.tools import ToolRuntime
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from deepagents_talon.subagent_registry import SubagentRegistry, VersionedAsyncSubagents


def _runtime(state=None):
    return ToolRuntime(
        state=state or {},
        context=None,
        tool_call_id="call",
        store=None,
        stream_writer=lambda _: None,
        config={"configurable": {"thread_id": "chat"}},
    )


async def _call(middleware, name, args, state=None):
    target = next(tool for tool in middleware.tools if tool.name == name)
    runtime = _runtime(state)
    request = ToolCallRequest(
        tool_call={"type": "tool_call", "name": name, "id": "call", "args": args},
        tool=target,
        state=runtime.state,
        runtime=runtime,
    )

    async def handler(resolved):
        return await target.coroutine(**resolved.tool_call["args"], runtime=runtime)

    return await middleware.awrap_tool_call(request, handler)


async def test_edit_remove_restart_preserve_original_remote_target(tmp_path, monkeypatch):
    connections = []

    def client(*, url, headers):
        assert headers["x-auth-scheme"] == "langsmith"
        connections.append(url)
        result = MagicMock()
        result.threads.create = AsyncMock(return_value={"thread_id": "thread-original"})
        result.runs.create = AsyncMock(return_value={"run_id": "run-original"})
        result.runs.get = AsyncMock(return_value={"status": "running"})
        result.runs.cancel = AsyncMock()
        return result

    monkeypatch.setattr("deepagents.middleware.async_subagents.get_client", client)
    path = tmp_path / "registry.json"
    registry = SubagentRegistry(path)
    original = {
        "name": "researcher",
        "description": "Research",
        "graph_id": "old",
        "url": "https://old.example",
    }
    revisions, current = registry.prepare([original])
    registry.commit(revisions)
    launched = await _call(
        VersionedAsyncSubagents(revisions, current),
        "start_async_task",
        {
            "subagent_type": "researcher",
            "description": "work",
        },
    )
    assert isinstance(launched, Command)
    state = {"async_tasks": launched.update["async_tasks"]}

    replacement = {**original, "graph_id": "new", "url": "https://new.example"}
    revisions, current = registry.prepare([replacement])
    registry.commit(revisions)
    middleware = VersionedAsyncSubagents(revisions, current)
    await _call(
        middleware, "start_async_task", {"subagent_type": "researcher", "description": "new work"}
    )
    assert connections[-1] == "https://new.example"

    registry = SubagentRegistry(path)
    revisions, current = registry.prepare([])
    middleware = VersionedAsyncSubagents(revisions, current)
    checked = await _call(middleware, "check_async_task", {"task_id": "thread-original"}, state)
    assert isinstance(checked, Command)
    assert connections[-1] == "https://old.example"
    await _call(
        middleware,
        "update_async_task",
        {"task_id": "thread-original", "message": "follow up"},
        state,
    )
    await _call(middleware, "cancel_async_task", {"task_id": "thread-original"}, state)
    refused = await _call(
        middleware, "start_async_task", {"subagent_type": "researcher", "description": "work"}
    )
    assert isinstance(refused, ToolMessage)
    assert "unavailable" in refused.content
    assert path.stat().st_mode & 0o777 == 0o600


def test_failed_prepare_does_not_modify_registry(tmp_path):
    path = tmp_path / "registry.json"
    registry = SubagentRegistry(path)
    with pytest.raises(ValueError, match="Invalid remote subagent configuration"):
        registry.prepare([{"name": "bad", "description": "Invalid"}])
    assert not path.exists()


def test_registry_rejects_symlink(tmp_path):
    target = tmp_path / "other.json"
    target.write_text("{}")
    path = tmp_path / "registry.json"
    path.symlink_to(target)
    with pytest.raises(ValueError, match="Invalid subagent registry"):
        SubagentRegistry(path)
