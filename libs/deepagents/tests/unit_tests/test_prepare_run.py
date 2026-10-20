"""Application-controlled preparation of SDK runs."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from deepagents import create_deep_agent
from deepagents.middleware import ParentNotifierMiddleware
from deepagents.middleware._parent_notifier import _ParentNotifierMiddleware
from deepagents.middleware.async_subagents import AsyncSubAgentMiddleware, RunKind, RunOptions, _notify_parent, _task_event
from tests.unit_tests.chat_model import GenericFakeChatModel
from tests.unit_tests.test_async_subagents import _callback_config, _get_tool, _make_runtime, _make_runtime_with_task, _make_spec


@pytest.mark.parametrize("operation", ["start", "update"])
@pytest.mark.parametrize("invocation", ["sync", "async"])
@pytest.mark.parametrize("async_callback", [False, True])
async def test_prepare_task_run(operation: str, invocation: str, *, async_callback: bool) -> None:
    runtime = _make_runtime() if operation == "start" else _make_runtime_with_task()
    runtime.config = {"configurable": {"thread_id": "parent"}}
    calls = []

    def prepare(kind: RunKind, options: RunOptions, config: RunnableConfig, *, tool_call_id: str | None = None) -> RunOptions:
        calls.append((kind, config, tool_call_id))
        return {**options, "webhook": "https://example.com/private", "context": {"tenant": "test"}, "multitask_strategy": "enqueue"}

    async def aprepare(kind: RunKind, options: RunOptions, config: RunnableConfig, *, tool_call_id: str | None = None) -> RunOptions:
        return prepare(kind, options, config, tool_call_id=tool_call_id)

    client = MagicMock()
    client.threads.create.return_value = {"thread_id": "thread_abc"}
    client.runs.create.return_value = {"run_id": "new_run"}
    if invocation == "async":
        client.threads.create = AsyncMock(return_value={"thread_id": "thread_abc"})
        client.runs.create = AsyncMock(return_value={"run_id": "new_run"})
    middleware = AsyncSubAgentMiddleware(async_subagents=[_make_spec()], prepare_run=aprepare if async_callback else prepare)
    tool = _get_tool(middleware.tools, f"{operation}_async_task")
    arguments = {"description": "research", "subagent_type": "test-agent"} if operation == "start" else {"task_id": "thread_abc", "message": "more"}
    getter = "get_client" if invocation == "async" else "get_sync_client"
    with patch(f"deepagents.middleware.async_subagents.{getter}", return_value=client):
        result = await tool.coroutine(**arguments, runtime=runtime) if invocation == "async" else tool.func(**arguments, runtime=runtime)
    assert calls == [(operation, runtime.config, "tc_test")]
    kwargs = client.runs.create.call_args.kwargs
    assert kwargs["thread_id"] == "thread_abc"
    assert kwargs["assistant_id"] == "my_graph"
    assert kwargs["webhook"] == "https://example.com/private"
    assert kwargs["context"] == {"tenant": "test"}
    assert kwargs["multitask_strategy"] == "enqueue"
    assert isinstance(result, Command)
    assert result.update["async_tasks"]["thread_abc"]["run_id"] == "new_run"
    assert "private" not in str(result.update)


@pytest.mark.parametrize("operation", ["start", "update"])
@pytest.mark.parametrize("invocation", ["sync", "async"])
async def test_suppressed_task_has_no_false_tracking(operation: str, invocation: str) -> None:
    prepare = AsyncMock(return_value=None)
    middleware = AsyncSubAgentMiddleware(async_subagents=[_make_spec()], prepare_run=prepare)
    runtime = _make_runtime() if operation == "start" else _make_runtime_with_task()
    original = dict(runtime.state)
    client = MagicMock()
    client.threads.create.return_value = {"thread_id": "thread_abc"}
    if invocation == "async":
        client.threads.create = AsyncMock(return_value={"thread_id": "thread_abc"})
    tool = _get_tool(middleware.tools, f"{operation}_async_task")
    arguments = {"description": "research", "subagent_type": "test-agent"} if operation == "start" else {"task_id": "thread_abc", "message": "more"}
    getter = "get_client" if invocation == "async" else "get_sync_client"
    with patch(f"deepagents.middleware.async_subagents.{getter}", return_value=client):
        result = await tool.coroutine(**arguments, runtime=runtime) if invocation == "async" else tool.func(**arguments, runtime=runtime)
    assert "suppressed" in result
    assert not isinstance(result, Command)
    assert runtime.state == original
    client.runs.create.assert_not_called()


@pytest.mark.parametrize("status", ["success", "message"])
async def test_parent_delivery_takeover_precedes_sdk_access(status: str) -> None:
    prepare = AsyncMock(return_value=None)
    config = _callback_config()
    event = _task_event(config, "researcher", status, message="question")
    with patch("deepagents.middleware.async_subagents.get_client") as client:
        accepted = await _notify_parent(config, event, {"tenant": "test"}, prepare_run=prepare)
    assert accepted
    client.assert_not_called()
    args = prepare.call_args.args
    assert args[0] == ("message" if status == "message" else "wake")
    assert args[1]["thread_id"] == "lead_thread"
    assert args[1]["assistant_id"] == "lead_assistant"
    assert args[1]["context"] == {"tenant": "test"}
    assert args[1]["input"]["messages"][0]["deepagents_notification"]["status"] == status
    assert args[2] == config


@pytest.mark.parametrize("status", ["success", "message"])
async def test_parent_sdk_run_uses_prepared_options(status: str) -> None:
    client = MagicMock()
    client.threads.get = AsyncMock(return_value={"status": "idle"})
    client.runs.create = AsyncMock()
    config = _callback_config()

    def prepare(kind: RunKind, options: RunOptions, config: RunnableConfig, *, tool_call_id: str | None = None) -> RunOptions:
        assert kind == ("message" if status == "message" else "wake")
        assert config["configurable"]["thread_id"] == "child_thread"
        assert tool_call_id is None
        return {**options, "webhook": "https://example.com/notify", "context": {"tenant": "other"}, "multitask_strategy": "enqueue"}

    with patch("deepagents.middleware.async_subagents.get_client", return_value=client):
        assert await _notify_parent(config, _task_event(config, "researcher", status), prepare_run=prepare)
    assert client.runs.create.call_args.kwargs["webhook"] == "https://example.com/notify"
    assert client.runs.create.call_args.kwargs["context"] == {"tenant": "other"}
    assert client.runs.create.call_args.kwargs["multitask_strategy"] == "enqueue"


@pytest.mark.parametrize("invocation", ["sync", "async"])
async def test_create_deep_agent_forwards_start_preparation(invocation: str) -> None:
    call = {"name": "start_async_task", "args": {"description": "dig", "subagent_type": "test-agent"}, "id": "start_id"}
    prepare = AsyncMock(return_value=None)
    model = GenericFakeChatModel(messages=iter([AIMessage(content="", tool_calls=[call]), AIMessage(content="done")]))
    middleware = AsyncSubAgentMiddleware(async_subagents=[_make_spec()], prepare_run=prepare)
    agent = create_deep_agent(model=model, subagents=[_make_spec()], middleware=[middleware])
    client = MagicMock()
    client.threads.create.return_value = {"thread_id": "new_thread"}
    if invocation == "async":
        client.threads.create = AsyncMock(return_value={"thread_id": "new_thread"})
    getter = "get_client" if invocation == "async" else "get_sync_client"
    with patch(f"deepagents.middleware.async_subagents.{getter}", return_value=client):
        inputs = {"messages": [HumanMessage(content="dig")]}
        result = await agent.ainvoke(inputs) if invocation == "async" else agent.invoke(inputs)
    assert prepare.call_args.args[0] == "start"
    assert prepare.call_args.kwargs["tool_call_id"] == "start_id"
    assert not result.get("async_tasks")
    client.runs.create.assert_not_called()


async def test_create_deep_agent_forwards_message_and_completion_preparation() -> None:
    call = {"name": "message_parent", "args": {"message": "Which file?"}, "id": "message_id"}
    prepare = AsyncMock(return_value=None)
    model = GenericFakeChatModel(messages=iter([AIMessage(content="", tool_calls=[call]), AIMessage(content="done")]))
    agent = create_deep_agent(model=model, middleware=[ParentNotifierMiddleware(prepare_run=prepare)])
    with patch("deepagents.middleware.async_subagents.get_client") as client:
        result = await agent.ainvoke({"messages": [HumanMessage(content="dig")]}, _callback_config())
    assert [call.args[0] for call in prepare.call_args_list] == ["message", "wake"]
    assert prepare.call_args_list[0].kwargs["tool_call_id"] == "message_id"
    assert result["messages"][-1].content == "done"
    client.assert_not_called()


@pytest.mark.parametrize("parent", [None, {"thread_id": "other_deployment"}])
async def test_no_valid_parent_does_not_call_application(parent: dict[str, Any] | None) -> None:
    prepare = AsyncMock(return_value=None)
    runtime = _make_runtime()
    runtime.config = _callback_config(parent)
    result = await _ParentNotifierMiddleware(prepare_run=prepare)._amessage_parent("hello", runtime)
    assert "No launching agent" in result or "not delivered" in result
    prepare.assert_not_called()


async def test_nested_graph_does_not_call_application() -> None:
    prepare = AsyncMock(return_value=None)
    runtime = _make_runtime()
    runtime.config = _callback_config(checkpoint_ns="task:1|tools:2")
    result = await _ParentNotifierMiddleware(prepare_run=prepare)._amessage_parent("hello", runtime)
    assert "No launching agent" in result
    prepare.assert_not_called()


async def test_callback_errors_do_not_expose_credentials(caplog: pytest.LogCaptureFixture) -> None:
    prepare = AsyncMock(side_effect=ValueError("credential-in-webhook"))
    middleware = AsyncSubAgentMiddleware(async_subagents=[_make_spec()], prepare_run=prepare)
    tool = _get_tool(middleware.tools, "update_async_task")
    result = await tool.coroutine(task_id="thread_abc", message="more", runtime=_make_runtime_with_task())
    assert "Failed" in result
    assert "credential-in-webhook" not in result + caplog.text
    config = _callback_config()
    assert not await _notify_parent(config, _task_event(config, "researcher", "success"), prepare_run=prepare)
    assert "credential-in-webhook" not in caplog.text
