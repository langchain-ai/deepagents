"""Async subagent tasks run by an application's own executor instead of an Agent Server."""

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from deepagents import create_deep_agent
from deepagents.middleware import (
    AsyncSubAgentMiddleware,
    AsyncTask,
    AsyncTaskExecutor,
    TaskRequest,
    TaskSnapshot,
    child_config,
    deliver_to_parent,
)
from tests.unit_tests.chat_model import GenericFakeChatModel
from tests.unit_tests.test_async_subagents import _get_tool, _make_runtime_with_task, _make_spec


class FakeExecutor:
    """Records what the middleware asks of it and reports scripted snapshots."""

    def __init__(self, snapshot: TaskSnapshot | None = None) -> None:
        self.requests: list[TaskRequest] = []
        self.sent: list[tuple[str, str, str | None]] = []
        self.cancelled: list[str] = []
        self.snapshot: TaskSnapshot = snapshot or {"status": "running"}

    async def start(self, request: TaskRequest) -> str:
        self.requests.append(request)
        return "workflow-1"

    async def status(self, task: AsyncTask) -> TaskSnapshot:  # one scripted task
        return self.snapshot

    async def send(self, task: AsyncTask, message: str, *, tool_call_id: str | None = None) -> str | None:
        self.sent.append((task["task_id"], message, tool_call_id))
        return "run-2"

    async def cancel(self, task: AsyncTask) -> None:
        self.cancelled.append(task["task_id"])


class BrokenExecutor(FakeExecutor):
    async def start(self, request: TaskRequest) -> str:  # always fails
        msg = "token=secret"
        raise RuntimeError(msg)

    async def status(self, task: AsyncTask) -> TaskSnapshot:  # always fails
        msg = "unreachable"
        raise RuntimeError(msg)


DONE: TaskSnapshot = {"status": "success", "run_id": "run-1", "result": [{"type": "text", "text": "found it"}]}


def _middleware(executor: AsyncTaskExecutor) -> AsyncSubAgentMiddleware:
    return AsyncSubAgentMiddleware(async_subagents=[_make_spec(url=None, graph_id="research-workflow")], executor=executor)


def _parent_runtime() -> ToolRuntime:
    return ToolRuntime(
        state={},
        context={"team": "eng"},
        tool_call_id="tc_start",
        store=None,
        stream_writer=lambda _: None,
        config={"configurable": {"thread_id": "parent"}, "metadata": {"assistant_id": "lead"}},
    )


def _tracked(task_id: str = "workflow-1", status: str = "running") -> dict[str, AsyncTask]:
    runtime = _make_runtime_with_task(task_id=task_id, run_id=task_id, status=status)
    return runtime.state["async_tasks"]


async def test_start_hands_the_executor_the_task_and_its_parent() -> None:
    executor = FakeExecutor()
    start = _get_tool(_middleware(executor).tools, "start_async_task")

    result = await start.coroutine(description="research it", subagent_type="test-agent", runtime=_parent_runtime())

    (request,) = executor.requests
    assert request["parent"] == {"thread_id": "parent", "assistant_id": "lead"}
    assert request["context"] == {"team": "eng"}
    assert request["tool_call_id"] == "tc_start"
    assert request["spec"]["graph_id"] == "research-workflow"
    assert request["messages"][-1]["content"] == "research it"
    assert isinstance(result, Command)
    assert result.update["async_tasks"]["workflow-1"]["status"] == "running"


async def test_failed_start_tracks_nothing_and_hides_the_error() -> None:
    start = _get_tool(_middleware(BrokenExecutor()).tools, "start_async_task")

    result = await start.coroutine(description="research it", subagent_type="test-agent", runtime=_parent_runtime())

    assert result == "Failed to launch async subagent 'test-agent': RuntimeError"


async def test_check_reports_the_executors_snapshot() -> None:
    check = _get_tool(_middleware(FakeExecutor(DONE)).tools, "check_async_task")
    runtime = _make_runtime_with_task(task_id="workflow-1", run_id="workflow-1")

    result = await check.coroutine(task_id="workflow-1", runtime=runtime)

    assert json.loads(result.update["messages"][0].content)["result"] == "found it"
    assert result.update["async_tasks"]["workflow-1"]["status"] == "success"


async def test_update_and_cancel_go_through_the_executor() -> None:
    executor = FakeExecutor()
    tools = _middleware(executor).tools
    runtime = _make_runtime_with_task(task_id="workflow-1", run_id="workflow-1")

    updated = await _get_tool(tools, "update_async_task").coroutine(task_id="workflow-1", message="also check X", runtime=runtime)
    cancelled = await _get_tool(tools, "cancel_async_task").coroutine(task_id="workflow-1", runtime=runtime)

    assert executor.sent == [("workflow-1", "also check X", "tc_test")]
    assert updated.update["async_tasks"]["workflow-1"]["run_id"] == "run-2"
    assert executor.cancelled == ["workflow-1"]
    assert cancelled.update["async_tasks"]["workflow-1"]["status"] == "cancelled"


async def test_busy_parent_gets_the_outcome_before_its_next_model_call_once() -> None:
    middleware = _middleware(FakeExecutor(DONE))
    state: dict[str, Any] = {"messages": [], "async_tasks": _tracked()}

    first = await middleware.abefore_model(state, None)  # type: ignore[arg-type]
    state["async_tasks"] = {**state["async_tasks"], **first["async_tasks"]}
    second = await middleware.abefore_model(state, None)  # type: ignore[arg-type]

    assert "found it" in json.dumps(first["messages"][0]["content"])
    assert first["async_tasks"]["workflow-1"]["status"] == "success"
    assert second is None


async def test_unreachable_executor_skips_the_task_until_the_next_check() -> None:
    middleware = _middleware(BrokenExecutor())

    assert await middleware.abefore_model({"messages": [], "async_tasks": _tracked()}, None) is None  # type: ignore[arg-type]


PARENT = {"thread_id": "parent", "assistant_id": "lead"}


def _idle_server() -> MagicMock:
    client = MagicMock()
    client.threads.get = AsyncMock(return_value={"status": "idle"})
    client.runs.create = AsyncMock()
    return client


async def test_deliver_hands_the_app_the_notification_input() -> None:
    delivered: list[tuple[Any, dict[str, Any]]] = []

    async def deliver(parent: Any, run_input: dict[str, Any]) -> bool:  # noqa: ANN401  # matches `Deliver`
        delivered.append((parent, run_input))
        return True

    with patch("deepagents.middleware.async_subagents.get_client", side_effect=AssertionError("no SDK call")):
        started = await deliver_to_parent({"thread_id": "parent"}, task_id="workflow-1", agent_name="test-agent", snapshot=DONE, deliver=deliver)

    ((parent, run_input),) = delivered
    (notification,) = run_input["messages"]
    assert started is True
    assert parent == {"thread_id": "parent"}
    assert notification["deepagents_notification"] == {"task_id": "workflow-1", "agent_name": "test-agent", "status": "success", "run_id": "run-1"}
    assert "found it" in json.dumps(notification["content"])


async def test_deliver_wakes_an_idle_agent_server_parent() -> None:
    client = _idle_server()

    with patch("deepagents.middleware.async_subagents.get_client", return_value=client):
        started = await deliver_to_parent(PARENT, task_id="workflow-1", agent_name="test-agent", snapshot=DONE)

    assert started is True
    assert client.runs.create.call_args.kwargs["assistant_id"] == "lead"
    assert client.runs.create.call_args.kwargs["multitask_strategy"] == "reject"


async def test_deliver_leaves_a_busy_agent_server_parent_alone() -> None:
    client = _idle_server()
    client.threads.get = AsyncMock(return_value={"status": "busy"})

    with patch("deepagents.middleware.async_subagents.get_client", return_value=client):
        assert await deliver_to_parent(PARENT, task_id="workflow-1", agent_name="test-agent", snapshot=DONE) is False

    client.runs.create.assert_not_called()


@pytest.mark.parametrize(
    ("parent", "snapshot"),
    [(PARENT, {"status": "running"}), ({"thread_id": "parent"}, DONE)],
    ids=["nothing-to-report", "no-way-to-reach-the-parent"],
)
async def test_deliver_skips_what_it_cannot_deliver(parent: dict[str, str], snapshot: TaskSnapshot) -> None:
    with patch("deepagents.middleware.async_subagents.get_client", side_effect=AssertionError("no SDK call")):
        assert await deliver_to_parent(parent, task_id="workflow-1", agent_name="test-agent", snapshot=snapshot) is False


@pytest.mark.parametrize(
    ("snapshot", "shown"),
    [
        ({"status": "waiting", "interrupts": [{"id": "q", "value": "Which repo?"}]}, "Which repo?"),
        ({"status": "error", "error": "workflow failed"}, "workflow failed"),
    ],
    ids=["waiting", "error"],
)
async def test_busy_parent_hears_about_pauses_and_failures(snapshot: TaskSnapshot, shown: str) -> None:
    update = await _middleware(FakeExecutor(snapshot)).abefore_model({"messages": [], "async_tasks": _tracked()}, None)  # type: ignore[arg-type]

    assert update["async_tasks"]["workflow-1"]["status"] == snapshot["status"]
    assert shown in json.dumps(update["messages"][0]["content"])


async def test_list_can_filter_on_any_reported_status() -> None:
    lister = _get_tool(_middleware(FakeExecutor({"status": "timeout"})).tools, "list_async_tasks")
    runtime = _make_runtime_with_task(task_id="workflow-1", run_id="workflow-1", status="pending")

    result = await lister.coroutine(runtime=runtime, status_filter="pending")

    assert "status: timeout" in result.update["messages"][0].content


class ChildRunningExecutor(FakeExecutor):
    """Runs a real deep agent child the way the docs say."""

    def __init__(self, task_id: str) -> None:
        super().__init__()
        self.task_id = task_id
        self.child = create_deep_agent(model=GenericFakeChatModel(messages=iter([AIMessage(content="found it")])))

    async def start(self, request: TaskRequest) -> str:
        await self.child.ainvoke({"messages": request["messages"]}, child_config(request, task_id=self.task_id, run_id="workflow-run-1"))
        return self.task_id


async def test_child_run_with_child_config_wakes_its_agent_server_parent_for_its_task() -> None:
    """Parent launch, a real child's own report, then the parent recording the outcome."""
    executor = ChildRunningExecutor("workflow-7")
    middleware = _middleware(executor)
    server = _idle_server()

    with patch("deepagents.middleware.async_subagents.get_client", return_value=server):
        launched = await _get_tool(middleware.tools, "start_async_task").coroutine(
            description="research it", subagent_type="test-agent", runtime=_parent_runtime()
        )

    wake = server.runs.create.call_args.kwargs
    (notification,) = wake["input"]["messages"]
    assert (wake["thread_id"], wake["assistant_id"]) == ("parent", "lead")
    assert notification["deepagents_notification"]["task_id"] == "workflow-7"
    message = HumanMessage(content=notification["content"], additional_kwargs={"deepagents_notification": notification["deepagents_notification"]})
    recorded = middleware.before_agent({"messages": [message], "async_tasks": launched.update["async_tasks"]}, None)  # type: ignore[arg-type]
    assert recorded["async_tasks"]["workflow-7"]["status"] == "success"
