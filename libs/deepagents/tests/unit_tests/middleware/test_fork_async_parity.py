"""Local fork parity with a parent's remote async tools."""

import json
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

import pytest
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.runtime import Runtime

from deepagents.graph import create_deep_agent
from deepagents.middleware.async_subagents import AsyncSubAgentMiddleware, AsyncSubAgentState
from tests.unit_tests.chat_model import GenericFakeChatModel


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("middleware_kind", ["default", "override"])
async def test_fork_mirrors_async_tools_but_cannot_use_them(mode: str, middleware_kind: str) -> None:
    calls = [
        {"name": "start_async_task", "args": {"description": "nested", "subagent_type": "remote"}, "id": "start", "type": "tool_call"},
        {"name": "check_async_task", "args": {"task_id": "parent-task"}, "id": "check", "type": "tool_call"},
        {"name": "update_async_task", "args": {"task_id": "parent-task", "message": "update"}, "id": "update", "type": "tool_call"},
        {"name": "cancel_async_task", "args": {"task_id": "parent-task"}, "id": "cancel", "type": "tool_call"},
        {"name": "list_async_tasks", "args": {}, "id": "list", "type": "tool_call"},
        {"name": "task", "args": {"description": "nested", "subagent_type": "worker"}, "id": "task", "type": "tool_call"},
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
    if mode == "sync":
        result = agent.invoke(inputs)
    else:
        result = await agent.ainvoke(inputs)

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
    replies = [
        message
        for message in worker.call_history[1]["messages"]
        if isinstance(message, ToolMessage) and message.tool_call_id in {call["id"] for call in calls}
    ]
    assert len(replies) == len(calls)
    assert all("cannot" in str(reply.content) for reply in replies)


def test_fork_rejects_async_tools_that_cannot_be_guarded() -> None:
    class CustomTool(BaseTool):
        name: str = "custom"
        description: str = "Custom async tool."

        def _run(self) -> str:
            return "unsafe"

    remote = AsyncSubAgentMiddleware(async_subagents=[{"name": "remote", "description": "Remote", "graph_id": "remote"}])
    remote.tools = [CustomTool()]
    model = GenericFakeChatModel(messages=iter([AIMessage(content="done")]))
    with pytest.raises(ValueError, match="Cannot safely mirror non-StructuredTool"):
        create_deep_agent(
            model=model,
            middleware=[remote],
            subagents=[{"name": "worker", "description": "Continue", "model": model, "mode": "fork"}],
        )
