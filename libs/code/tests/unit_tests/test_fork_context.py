"""Tests for scoped fork inheritance and generated assignment guidance."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import Mock

import pytest
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage

from deepagents_code._fork_context import ForkAssignmentMiddleware, inherited_snapshot
from deepagents_code.local_context import LocalContextMiddleware


def _request(
    snapshot: str = "parent", agent: str = "general-purpose"
) -> ToolCallRequest:
    return ToolCallRequest(
        tool_call={
            "name": "task",
            "args": {"description": "assignment", "subagent_type": agent},
            "id": "call",
            "type": "tool_call",
        },
        tool=None,
        state={"_local_context": snapshot},
        runtime=Mock(),
    )


@pytest.mark.parametrize("agent", ["general-purpose", "custom"])
def test_generated_assignment_does_not_mutate_parent(agent: str) -> None:
    request = _request(agent=agent)
    received = []
    middleware = ForkAssignmentMiddleware("Report the complete result.")

    def handler(updated: ToolCallRequest) -> ToolMessage:
        received.append(updated)
        return ToolMessage(content="done", tool_call_id="call")

    middleware.wrap_tool_call(request, handler)

    assert request.tool_call["args"]["description"] == "assignment"
    expected = (
        "assignment\n\nReport the complete result."
        if agent == "general-purpose"
        else "assignment"
    )
    assert received[0].tool_call["args"]["description"] == expected


async def test_async_generated_assignment() -> None:
    request = _request()
    middleware = ForkAssignmentMiddleware("Return a self-contained answer.")

    async def handler(updated: ToolCallRequest) -> ToolMessage:
        await asyncio.sleep(0)
        assert updated.tool_call["args"]["description"].endswith(
            "Return a self-contained answer."
        )
        return ToolMessage(content="done", tool_call_id="call")

    result = await middleware.awrap_tool_call(request, handler)

    assert isinstance(result, ToolMessage)
    assert request.tool_call["args"]["description"] == "assignment"


@pytest.mark.parametrize("fails", [False, True])
def test_sync_snapshot_binding_is_released(fails: bool) -> None:
    middleware = LocalContextMiddleware(backend=Mock())

    def handler(request: ToolCallRequest) -> ToolMessage:
        snapshot = inherited_snapshot()
        assert snapshot is not None
        assert snapshot.context == request.state["_local_context"]
        if fails:
            msg = "failed task"
            raise ValueError(msg)
        return ToolMessage(content="done", tool_call_id="call")

    if fails:
        with pytest.raises(ValueError, match="failed task"):
            middleware.wrap_tool_call(_request(), handler)
    else:
        middleware.wrap_tool_call(_request(), handler)

    assert inherited_snapshot() is None


async def test_concurrent_bindings_are_isolated_and_cancelled_binding_released() -> (
    None
):
    middleware = LocalContextMiddleware(backend=Mock())
    started = asyncio.Event()
    release = asyncio.Event()

    async def handler(request: ToolCallRequest) -> ToolMessage:
        snapshot = inherited_snapshot()
        assert snapshot is not None
        assert snapshot.context == request.state["_local_context"]
        started.set()
        await release.wait()
        assert inherited_snapshot() == snapshot
        return ToolMessage(content="done", tool_call_id="call")

    async def run(snapshot: str) -> None:
        try:
            await middleware.awrap_tool_call(_request(snapshot), handler)
        finally:
            assert inherited_snapshot() is None

    first = asyncio.create_task(run("first"))
    await started.wait()
    second = asyncio.create_task(run("second"))
    await asyncio.sleep(0)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    release.set()
    await second
    assert inherited_snapshot() is None


async def test_fork_inherits_latest_fingerprint_without_duplicate_facts() -> None:
    import hashlib

    from deepagents.backends.protocol import ExecuteResponse

    class Backend:
        async def aexecute(self, command: str, **kwargs: object) -> ExecuteResponse:
            del command, kwargs
            return ExecuteResponse(output="already observed", exit_code=0)

    middleware = LocalContextMiddleware(backend=Backend())
    request = _request()
    request.state["_latest_local_context_fingerprint"] = hashlib.sha256(
        b"already observed"
    ).hexdigest()

    async def handler(request: ToolCallRequest) -> ToolMessage:
        del request
        child: Any = {"messages": [], "_deepagents_forked_context": True}
        result = await middleware.abefore_agent(child, Mock())
        assert result is not None
        assert result["_local_context"] == "parent"
        assert "messages" not in result
        assert result["_local_context_fork_initialized"] is True
        return ToolMessage(content="done", tool_call_id="call")

    await middleware.awrap_tool_call(request, handler)
