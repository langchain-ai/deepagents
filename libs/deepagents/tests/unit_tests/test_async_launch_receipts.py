"""Lost launch acknowledgements retain owned handles without guessing run IDs."""

import asyncio
import json
from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphBubbleUp
from langgraph.types import Command

from deepagents import create_deep_agent
from deepagents.middleware.async_subagents import AsyncSubAgentMiddleware, AsyncSubAgentState, _launches_reducer
from tests.unit_tests.chat_model import GenericFakeChatModel
from tests.unit_tests.test_async_subagents import _make_runtime, _make_spec

KEY = "_unconfirmed_async_launches"


@pytest.fixture
def remote() -> Iterator[tuple[MagicMock, MagicMock, dict[str, Any]]]:
    state: dict[str, Any] = {"visible": False, "runs": [], "error": httpx.ReadError("lost response")}
    sync, asynchronous = MagicMock(), MagicMock()
    sync.__enter__.return_value = sync
    asynchronous.__aenter__ = AsyncMock(return_value=asynchronous)
    asynchronous.__aexit__ = AsyncMock(return_value=False)

    def start(**kwargs: Any) -> dict[str, Any]:
        run = {"run_id": "accepted-run", "thread_id": "owned-thread", "status": "running", "metadata": kwargs.get("metadata", {})}
        state["runs"].append(run)
        if state["error"]:
            raise state["error"]
        return run

    def read(**_kwargs: Any) -> list[dict[str, Any]]:
        if state.get("read_error"):
            raise state["read_error"]
        return state["runs"] if state["visible"] else []

    sync.threads.create.return_value = {"thread_id": "owned-thread"}
    sync.runs.create.side_effect = start
    sync.runs.list.side_effect = read
    sync.runs.get.side_effect = lambda **_: state["runs"][0]
    asynchronous.threads.create = AsyncMock(return_value={"thread_id": "owned-thread"})
    asynchronous.runs.create = AsyncMock(side_effect=start)
    asynchronous.runs.list = AsyncMock(side_effect=read)
    asynchronous.runs.get = AsyncMock(side_effect=lambda **_: state["runs"][0])
    asynchronous.runs.cancel = AsyncMock()
    with (
        patch("deepagents.middleware.async_subagents.get_sync_client", return_value=sync),
        patch("deepagents.middleware.async_subagents.get_client", return_value=asynchronous),
    ):
        yield sync, asynchronous, state


async def launch(middleware: AsyncSubAgentMiddleware, *, asynchronous: bool) -> Command | str:
    tool = middleware.tools[0]
    kwargs = {"description": "Process one synthetic job", "subagent_type": "worker", "runtime": _make_runtime()}
    return await tool.coroutine(**kwargs) if asynchronous else tool.func(**kwargs)


async def operate(middleware: AsyncSubAgentMiddleware, name: str, state: dict[str, Any], *, asynchronous: bool, **kwargs: Any) -> Command | str:
    tool = next(tool for tool in middleware.tools if tool.name == name)
    runtime = _make_runtime()
    runtime.state = state
    return await tool.coroutine(runtime=runtime, **kwargs) if asynchronous else tool.func(runtime=runtime, **kwargs)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("visible", [False, True])
async def test_lost_reply_retains_owned_task(remote: tuple, *, asynchronous: bool, visible: bool) -> None:
    sync, async_client, server = remote
    server["visible"] = visible
    middleware = AsyncSubAgentMiddleware(async_subagents=[_make_spec("worker")])
    result = await launch(middleware, asynchronous=asynchronous)
    assert isinstance(result, Command)
    assert len(server["runs"]) == 1
    assert (async_client if asynchronous else sync).runs.create.call_count == 1
    assert "owned-thread" in result.update["messages"][0].content
    if visible:
        assert result.update["async_tasks"]["owned-thread"]["run_id"] == "accepted-run"
        assert KEY not in result.update
    else:
        assert "async_tasks" not in result.update
        receipt = result.update[KEY]["owned-thread"]
        assert receipt["launch_id"] == server["runs"][0]["metadata"]["deepagents_launch_id"]
        assert set(receipt) == {"task_id", "thread_id", "agent_name", "launch_id", "created_at"}
        assert result.update["messages"][0].status == "error"
        assert "unconfirmed" in result.update["messages"][0].content


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("action", ["check_async_task", "list_async_tasks", "cancel_async_task", "update_async_task"])
async def test_later_operation_recovers_checkpointed_launch(remote: tuple, *, asynchronous: bool, action: str) -> None:
    _, _, server = remote
    middleware = AsyncSubAgentMiddleware(async_subagents=[_make_spec("worker")])
    result = await launch(middleware, asynchronous=asynchronous)
    checkpoint = json.loads(json.dumps({KEY: result.update[KEY]}))
    server.update(visible=True, error=None)
    # A new middleware instance must recover from the checkpoint, not a local cache.
    restored = AsyncSubAgentMiddleware(async_subagents=[_make_spec("worker")])
    kwargs = {} if action == "list_async_tasks" else {"task_id": "owned-thread"}
    if action == "update_async_task":
        kwargs["message"] = "Explicit follow-up"
    result = await operate(restored, action, checkpoint, asynchronous=asynchronous, **kwargs)
    assert isinstance(result, Command)
    task = result.update["async_tasks"]["owned-thread"]
    assert task["run_id"] == "accepted-run"
    assert task["status"] == ("cancelled" if action == "cancel_async_task" else "running")
    assert _launches_reducer(checkpoint[KEY], result.update[KEY]) == {}
    assert len(server["runs"]) == (2 if action == "update_async_task" else 1)
    assert "async_tasks" not in checkpoint  # Tool runtime state is not mutated.


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("action", ["check_async_task", "list_async_tasks", "cancel_async_task", "update_async_task"])
async def test_unconfirmed_operation_preserves_receipt_and_does_not_mutate_remote(remote: tuple, *, asynchronous: bool, action: str) -> None:
    sync, async_client, server = remote
    middleware = AsyncSubAgentMiddleware(async_subagents=[_make_spec("worker")])
    started = await launch(middleware, asynchronous=asynchronous)
    state = {KEY: started.update[KEY]}
    kwargs = {} if action == "list_async_tasks" else {"task_id": "owned-thread"}
    if action == "update_async_task":
        kwargs["message"] = "Follow-up"
    result = await operate(middleware, action, state, asynchronous=asynchronous, **kwargs)
    assert isinstance(result, Command)
    assert "owned-thread" in result.update["messages"][0].content
    assert "unconfirmed" in result.update["messages"][0].content
    assert result.update["messages"][0].status == "error"
    assert not result.update.get("async_tasks")
    assert "cancelled" not in result.update["messages"][0].content
    assert len(server["runs"]) == 1
    (async_client if asynchronous else sync).runs.cancel.assert_not_called()
    assert state[KEY] == started.update[KEY]


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("case", ["read_failure", "wrong_marker", "wrong_thread", "multiple"])
async def test_recovery_does_not_guess_unrelated_or_ambiguous_run(remote: tuple, *, asynchronous: bool, case: str) -> None:
    _, _, server = remote
    middleware = AsyncSubAgentMiddleware(async_subagents=[_make_spec("worker")])
    result = await launch(middleware, asynchronous=asynchronous)
    state = {KEY: result.update[KEY]}
    server["visible"] = True
    if case == "read_failure":
        server["read_error"] = httpx.ReadError("recovery unavailable")
    elif case == "wrong_marker":
        server["runs"][0]["metadata"] = {"deepagents_launch_id": "another-launch"}
    elif case == "wrong_thread":
        server["runs"][0]["thread_id"] = "not-owned"
    else:
        server["runs"].append({**server["runs"][0], "run_id": "another-run"})
    result = await operate(middleware, "check_async_task", state, asynchronous=asynchronous, task_id="owned-thread")
    assert not result.update.get("async_tasks")
    assert "unconfirmed" in result.update["messages"][0].content


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_recovered_record_survives_following_status_read_failure(remote: tuple, *, asynchronous: bool) -> None:
    sync, async_client, server = remote
    middleware = AsyncSubAgentMiddleware(async_subagents=[_make_spec("worker")])
    started = await launch(middleware, asynchronous=asynchronous)
    server["visible"] = True
    (async_client if asynchronous else sync).runs.get.side_effect = httpx.ReadError("status read failed")
    result = await operate(middleware, "check_async_task", {KEY: started.update[KEY]}, asynchronous=asynchronous, task_id="owned-thread")
    assert result.update["async_tasks"]["owned-thread"]["run_id"] == "accepted-run"
    assert "Failed to get run status" in result.update["messages"][0].content
    assert result.update[KEY] == {"owned-thread": None}


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("status", [400, 401, 408, 429, 500, 503])
async def test_definite_rejection_is_distinct_from_uncertain_outcome(remote: tuple, *, asynchronous: bool, status: int) -> None:
    _, _, server = remote
    server["error"] = httpx.HTTPStatusError("rejected", request=httpx.Request("POST", "https://example.test"), response=httpx.Response(status))
    result = await launch(AsyncSubAgentMiddleware(async_subagents=[_make_spec("worker")]), asynchronous=asynchronous)
    if status == 408 or status >= 500:
        assert isinstance(result, Command)
        assert KEY in result.update
    else:
        assert isinstance(result, str)
        assert "Failed to launch" in result


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_unknown_task_does_not_become_trusted(remote: tuple, *, asynchronous: bool) -> None:
    sync, async_client, _ = remote
    middleware = AsyncSubAgentMiddleware(async_subagents=[_make_spec("worker")])
    result = await operate(middleware, "cancel_async_task", {}, asynchronous=asynchronous, task_id="arbitrary-thread")
    assert "No tracked task found" in result
    (async_client if asynchronous else sync).runs.cancel.assert_not_called()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("error", [GraphBubbleUp(), asyncio.CancelledError()])
async def test_launch_does_not_swallow_control_flow(remote: tuple, *, asynchronous: bool, error: BaseException) -> None:
    _, _, server = remote
    server["error"] = error
    with pytest.raises(type(error)):
        await launch(AsyncSubAgentMiddleware(async_subagents=[_make_spec("worker")]), asynchronous=asynchronous)


async def test_polling_promotes_owned_receipt_and_persists_completed_outcome(remote: tuple) -> None:
    _, async_client, server = remote
    middleware = AsyncSubAgentMiddleware(async_subagents=[_make_spec("worker")])
    started = await launch(middleware, asynchronous=True)
    server["visible"] = True
    server["runs"][0]["status"] = "success"
    async_client.threads.get = AsyncMock(return_value={"status": "idle", "values": {"messages": [{"type": "ai", "content": "Complete"}]}})
    update = await middleware.abefore_model({KEY: started.update[KEY]}, None)
    assert update["async_tasks"]["owned-thread"]["status"] == "success"
    assert update[KEY] == {"owned-thread": None}
    assert "Complete" in str(update["messages"])


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_real_graph_checkpoints_receipt_and_injects_runtime_into_recovery_tools(remote: tuple, *, asynchronous: bool) -> None:
    _, _, server = remote
    checkpoint = InMemorySaver()
    config = {"configurable": {"thread_id": "parent"}}
    model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "start_async_task", "args": {"description": "job", "subagent_type": "worker"}, "id": "launch", "type": "tool_call"}
                    ],
                ),
                AIMessage(content="Launch unconfirmed"),
                AIMessage(
                    content="", tool_calls=[{"name": "check_async_task", "args": {"task_id": "owned-thread"}, "id": "check", "type": "tool_call"}]
                ),
                AIMessage(content="Task recovered"),
            ]
        )
    )
    middleware = AsyncSubAgentMiddleware(async_subagents=[_make_spec("worker")])
    graph = create_agent(model, middleware=[middleware], checkpointer=checkpoint)
    if asynchronous:
        await graph.ainvoke({"messages": [HumanMessage("Delegate")]}, config)
        saved = await graph.aget_state(config)
    else:
        graph.invoke({"messages": [HumanMessage("Delegate")]}, config)
        saved = graph.get_state(config)
    assert saved.values[KEY]["owned-thread"]["thread_id"] == "owned-thread"
    assert not saved.values.get("async_tasks")
    server["visible"] = True
    result = (
        await graph.ainvoke({"messages": [HumanMessage("Check")]}, config)
        if asynchronous
        else graph.invoke({"messages": [HumanMessage("Check")]}, config)
    )
    saved = await graph.aget_state(config) if asynchronous else graph.get_state(config)
    assert saved.values[KEY] == {}
    assert result["async_tasks"]["owned-thread"]["run_id"] == "accepted-run"
    assert len(server["runs"]) == 1
    checks = [msg for msg in result["messages"] if isinstance(msg, ToolMessage) and msg.tool_call_id == "check"]
    assert len(checks) == 1
    assert "No tracked task" not in str(checks[0].content)


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_parallel_unconfirmed_launches_keep_both_checkpoint_receipts(remote: tuple, *, asynchronous: bool) -> None:
    sync, async_client, server = remote
    (async_client if asynchronous else sync).threads.create.side_effect = [{"thread_id": "first-thread"}, {"thread_id": "second-thread"}]
    calls = [
        {"name": "start_async_task", "args": {"description": f"Job {index}", "subagent_type": "worker"}, "id": f"launch-{index}", "type": "tool_call"}
        for index in range(2)
    ]
    model = GenericFakeChatModel(messages=iter([AIMessage(content="", tool_calls=calls), AIMessage(content="Unconfirmed")]))
    graph = create_agent(model, middleware=[AsyncSubAgentMiddleware(async_subagents=[_make_spec("worker")])], checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "parallel-parent"}}
    if asynchronous:
        await graph.ainvoke({"messages": [HumanMessage("Delegate two jobs")]}, config)
        saved = await graph.aget_state(config)
    else:
        graph.invoke({"messages": [HumanMessage("Delegate two jobs")]}, config)
        saved = graph.get_state(config)
    assert set(saved.values[KEY]) == {"first-thread", "second-thread"}
    assert len(server["runs"]) == 2
    assert not saved.values.get("async_tasks")


@pytest.mark.usefixtures("remote")
async def test_launch_receipt_is_not_inherited_by_local_subagent() -> None:
    captured = []

    class Capture(AgentMiddleware):
        state_schema = AsyncSubAgentState

        def before_agent(self, state, runtime):
            captured.append(dict(state))

    parent = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "start_async_task",
                            "args": {"description": "Remote job", "subagent_type": "worker"},
                            "id": "remote",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "task", "args": {"description": "Local job", "subagent_type": "child"}, "id": "local", "type": "tool_call"}],
                ),
                AIMessage(content="Finished"),
            ]
        )
    )
    graph = create_deep_agent(
        parent,
        checkpointer=InMemorySaver(),
        subagents=[
            _make_spec("worker"),
            {
                "name": "child",
                "description": "Local child",
                "system_prompt": "Complete the job",
                "model": GenericFakeChatModel(messages=iter([AIMessage(content="Child finished")])),
                "middleware": [Capture()],
            },
        ],
    )
    config = {"configurable": {"thread_id": "private-parent"}}
    graph.invoke({"messages": [HumanMessage("Delegate")]}, config)
    assert captured
    assert not captured[0].get(KEY)
    assert "owned-thread" in graph.get_state(config).values[KEY]
