"""Local fork parity with a parent's remote async tools."""

import json
from collections.abc import Awaitable, Callable, Sequence
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.runtime import Runtime

from deepagents.backends import StateBackend
from deepagents.graph import create_deep_agent
from deepagents.middleware.async_subagents import AsyncSubAgentMiddleware, AsyncSubAgentState
from deepagents.middleware.subagents import SubAgentMiddleware
from tests.unit_tests.chat_model import GenericFakeChatModel


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("middleware_kind", ["default", "override"])
async def test_fork_inherits_async_tools_without_parent_tasks(mode: str, middleware_kind: str) -> None:
    """A fork sees the parent's async tool definitions but none of the parent's tasks."""
    calls = [
        {"name": "check_async_task", "args": {"task_id": "parent-task"}, "id": "check", "type": "tool_call"},
        {"name": "cancel_async_task", "args": {"task_id": "parent-task"}, "id": "cancel", "type": "tool_call"},
    ]
    captured: dict[str, Any] = {}

    class Record(AgentMiddleware):
        def __init__(self, label: str) -> None:
            self.label = label

        def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
            captured.setdefault(self.label, request.state.get("async_tasks"))
            return handler(request)

        async def awrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]) -> ModelResponse:
            captured.setdefault(self.label, request.state.get("async_tasks"))
            return await handler(request)

    parent = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "task", "args": {"description": "continue", "subagent_type": "worker"}, "id": "worker", "type": "tool_call"}
                    ],
                ),
                AIMessage(content="parent done"),
            ]
        )
    )
    worker = GenericFakeChatModel(messages=iter([AIMessage(content="", tool_calls=calls), AIMessage(content="worker done")]))
    remote = {"name": "remote", "description": "Remote", "graph_id": "remote"}

    class MutateForkTasks(AgentMiddleware):
        def before_agent(self, state: AsyncSubAgentState, runtime: Runtime) -> dict[str, Any]:
            captured["before_worker"] = state.get("async_tasks")
            return {"async_tasks": {"fork-task": {"status": "success"}}}

        async def abefore_agent(self, state: AsyncSubAgentState, runtime: Runtime) -> dict[str, Any]:
            return self.before_agent(state, runtime)

    middleware: list[AgentMiddleware] = [Record("parent")]
    if middleware_kind == "override":
        middleware.insert(0, AsyncSubAgentMiddleware(async_subagents=[remote], system_prompt="Remote guidance"))
    agent = create_deep_agent(
        model=parent,
        middleware=middleware,
        subagents=[
            {"name": "worker", "description": "Continue", "model": worker, "mode": "fork", "middleware": [Record("worker"), MutateForkTasks()]},
            remote,
        ],
        state_schema=AsyncSubAgentState,
    )
    task = {
        "task_id": "parent-task",
        "agent_name": "remote",
        "thread_id": "parent-task",
        "run_id": "run",
        "status": "success",
        "created_at": "then",
        "last_checked_at": "then",
        "last_updated_at": "then",
    }
    inputs = {"messages": [HumanMessage(content="start")], "async_tasks": {"parent-task": task}}
    client = MagicMock()
    with (
        patch("deepagents.middleware.async_subagents.get_sync_client", return_value=client),
        patch("deepagents.middleware.async_subagents.get_client", return_value=client),
    ):
        result = agent.invoke(inputs) if mode == "sync" else await agent.ainvoke(inputs)

    def payload(model: ChatAnthropic | ChatOpenAI, tools: Sequence[BaseTool]) -> str:
        bound = model.bind_tools(tools)
        return json.dumps(model._get_request_payload([HumanMessage(content="test")], **bound.kwargs)["tools"], separators=(",", ":"))

    for model in (ChatAnthropic(model="claude-sonnet-4-6", api_key="test"), ChatOpenAI(model="gpt-4o", api_key="test")):
        assert payload(model, parent.call_history[0]["tools"]) == payload(model, worker.call_history[0]["tools"])
    assert parent.call_history[0]["messages"][0] == worker.call_history[0]["messages"][0]
    assert captured["parent"] == {"parent-task": task}
    assert not captured["before_worker"]
    assert captured["worker"] == {"fork-task": {"status": "success"}}
    assert result["async_tasks"] == {"parent-task": task}
    replies = [m for m in worker.call_history[1]["messages"] if isinstance(m, ToolMessage) and m.tool_call_id in {c["id"] for c in calls}]
    assert len(replies) == len(calls)
    assert all("belong to the agent that started you" in str(reply.content) for reply in replies)
    assert not client.mock_calls


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("entrypoint", ["deep_agent", "direct"])
@pytest.mark.parametrize("parent_kind", ["default", "custom", "shared", "none"])
async def test_fork_uses_its_own_async_middleware(mode: str, entrypoint: str, parent_kind: str) -> None:
    parent = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(content="", tool_calls=[{"name": "task", "args": {"description": "continue", "subagent_type": "worker"}, "id": "fork"}]),
                AIMessage(content="parent done"),
            ]
        )
    )
    worker = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "start_async_task", "args": {"description": "do work", "subagent_type": "fork-remote"}, "id": "start"}],
                ),
                AIMessage(content="", tool_calls=[{"name": "cancel_async_task", "args": {"task_id": "fork-task"}, "id": "cancel"}]),
                AIMessage(content="worker done"),
            ]
        )
    )
    fork_remote = AsyncSubAgentMiddleware(
        async_subagents=[{"name": "fork-remote", "description": "Fork remote", "graph_id": "fork-graph", "url": "http://remote"}]
    )
    spec = {"name": "worker", "description": "Continue", "model": worker, "mode": "fork", "middleware": [fork_remote], "tools": []}
    parent_remote = {"name": "parent-remote", "description": "Parent remote", "graph_id": "parent-graph"}
    parent_middleware: list[AgentMiddleware] = []
    if parent_kind == "custom":
        parent_middleware.append(AsyncSubAgentMiddleware(async_subagents=[parent_remote]))
    elif parent_kind == "shared":
        parent_middleware.append(fork_remote)
    if entrypoint == "deep_agent":
        agent = create_deep_agent(
            model=parent,
            middleware=parent_middleware,
            subagents=[spec, *([parent_remote] if parent_kind == "default" else [])],
            state_schema=AsyncSubAgentState,
        )
    else:
        if parent_kind == "default":
            parent_middleware.append(AsyncSubAgentMiddleware(async_subagents=[parent_remote]))
        agent = create_agent(
            model=parent,
            middleware=[SubAgentMiddleware(backend=StateBackend(), subagents=[spec], state_schema=AsyncSubAgentState), *parent_middleware],
            state_schema=AsyncSubAgentState,
        )

    sync_client = MagicMock()
    sync_client.threads.create.return_value = {"thread_id": "fork-task"}
    sync_client.runs.create.return_value = {"run_id": "fork-run"}
    async_client = MagicMock()
    async_client.threads.create = AsyncMock(return_value={"thread_id": "fork-task"})
    async_client.runs.create = AsyncMock(return_value={"run_id": "fork-run"})
    async_client.runs.cancel = AsyncMock()
    async_client.runs.get = AsyncMock(return_value={"run_id": "fork-run", "status": "running"})
    inputs = {
        "messages": [HumanMessage(content="start")],
        "async_tasks": {
            "parent-task": {
                "task_id": "parent-task",
                "agent_name": "parent-remote",
                "thread_id": "parent-task",
                "run_id": "parent-run",
                "status": "success",
            }
        },
    }
    with (
        patch("deepagents.middleware.async_subagents.get_sync_client", return_value=sync_client),
        patch("deepagents.middleware.async_subagents.get_client", return_value=async_client),
    ):
        result = agent.invoke(inputs) if mode == "sync" else await agent.ainvoke(inputs)

    replies = {m.tool_call_id: str(m.content) for call in worker.call_history for m in call["messages"] if isinstance(m, ToolMessage)}
    assert "Launched async subagent. task_id: fork-task" in replies["start"]
    assert "Cancelled async subagent task: fork-task" in replies["cancel"]
    assert result["async_tasks"] == inputs["async_tasks"]
    client = sync_client if mode == "sync" else async_client
    client.threads.create.assert_called_once()
    client.runs.create.assert_called_once()
    assert client.runs.create.call_args.kwargs["assistant_id"] == "fork-graph"
    client.runs.cancel.assert_called_once_with(thread_id="fork-task", run_id="fork-run")
