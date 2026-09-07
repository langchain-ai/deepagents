from __future__ import annotations

import asyncio
import contextvars
import logging
from types import SimpleNamespace

import pytest
from langchain.tools import ToolRuntime
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool

from deepagents_talon.archive import ArchiveScope
from deepagents_talon.authorization import (
    current_authorization_handler,
    reset_authorization_handler,
    set_authorization_handler,
)
from deepagents_talon.background import _IN_SUBAGENT, BackgroundSubagents
from deepagents_talon.cron import CronOrigin
from deepagents_talon.host import TalonHost
from deepagents_talon.interfaces import AgentRequest, ChannelMessage
from deepagents_talon.runtime import _CRON_ORIGIN, _HISTORY_SCOPE, DeepAgentRuntime
from tests.conftest import RecordingChannel
from tests.test_host import _config


class ToolModel(FakeMessagesListChatModel):
    def bind_tools(self, _tools, **_kwargs: object):
        return self


def _delegate(name="researcher"):
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "task",
                "id": "launch",
                "args": {"subagent_type": name, "description": "research"},
            }
        ],
    )


def _runtime(monkeypatch, child, responses):
    model = ToolModel(responses=responses)
    monkeypatch.setattr("deepagents_talon.runtime._resolve_model_from_env", lambda *_a, **_k: model)
    return DeepAgentRuntime(
        model="test:parent",
        include_web_tools=False,
        skills=(),
        memory=(),
        subagents=[
            {"name": "researcher", "description": "Research", "runnable": RunnableLambda(child)}
        ],
    )


async def test_chat_continues_then_main_processes_background_result(tmp_path, monkeypatch):
    entered, release = asyncio.Event(), asyncio.Event()
    child_threads = []

    async def child(_state, config):
        child_threads.append(config["configurable"]["thread_id"])
        entered.set()
        await release.wait()
        return {"messages": [AIMessage(content="raw research result")]}

    runtime = _runtime(
        monkeypatch,
        child,
        [
            _delegate(),
            AIMessage(content="Working on it"),
            AIMessage(content="Still here"),
            AIMessage(content="Processed research"),
        ],
    )
    channel = RecordingChannel()
    host = TalonHost(config=_config(tmp_path), agent=runtime, channels=[channel])
    await host.start()
    try:
        await host.receive_message(channel, ChannelMessage("chat", "research"))
        await asyncio.wait_for(entered.wait(), 2)
        await asyncio.wait_for(host._tasks["chat"], 2)
        await host.receive_message(channel, ChannelMessage("chat", "hello"))
        await asyncio.wait_for(host._tasks["chat"], 2)
        assert runtime.background.owners() == {"chat"}
        assert not runtime.background.results("chat")
        release.set()
        await asyncio.gather(*(job.worker for job in runtime.background._jobs.values()))
        await host._dispatch_background_results()
        await asyncio.wait_for(host._tasks["chat"], 2)
        assert channel.sent == [
            ("chat", "Working on it"),
            ("chat", "Still here"),
            ("chat", "Processed research"),
        ]
        state = await runtime._graph.aget_state({"configurable": {"thread_id": "chat"}})
        assert any(
            "raw research result" in str(message.content) for message in state.values["messages"]
        )
        assert child_threads[0] != "chat"
        assert not runtime.background.results("chat")
    finally:
        release.set()
        await host.stop()


@pytest.mark.parametrize("command", ["/stop", "/new"])
async def test_commands_cancel_only_this_threads_children_when_main_idle(
    tmp_path, monkeypatch, command
):
    entered = asyncio.Queue()
    cancelled = []

    async def child(state):
        owner = state["messages"][-1].content
        await entered.put(owner)
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(owner)

    runtime = _runtime(
        monkeypatch,
        child,
        [_delegate(), AIMessage(content="Started"), _delegate(), AIMessage(content="Started")],
    )
    channel = RecordingChannel()
    host = TalonHost(config=_config(tmp_path), agent=runtime, channels=[channel])
    await host.start()
    try:
        for owner in ("one", "two"):
            await host.receive_message(channel, ChannelMessage(owner, "research"))
            await asyncio.wait_for(entered.get(), 2)
            await asyncio.wait_for(host._tasks[owner], 2)
        assert channel.sent == [("one", "Started"), ("two", "Started")]
        await host.receive_message(channel, ChannelMessage("one", command))
        assert len(cancelled) == 1
        assert runtime.background.owners() == {"two"}
        assert not runtime.background.results("one")
        if command == "/new":
            assert host._agent_conversation_id("one") != "one"
    finally:
        await host.stop()
    assert len(cancelled) == 2


def _request(owner, task_tool, **args: object):
    config = {"configurable": {"thread_id": owner, "checkpoint_ns": "parent"}}
    runtime = ToolRuntime(
        state={},
        config=config,
        context=None,
        stream_writer=lambda _: None,
        tool_call_id="call",
        store=None,
    )
    return ToolCallRequest(
        tool_call={"type": "tool_call", "name": task_tool.name, "id": "call", "args": args},
        tool=task_tool,
        state={},
        runtime=runtime,
    )


async def _unused_handler(_request):
    msg = "Background delegation should detach"
    raise AssertionError(msg)


async def test_inspect_cancel_ownership_and_capacity(monkeypatch):
    monkeypatch.setattr("deepagents_talon.background._MAX_RUNNING", 1)

    @tool
    async def task() -> str:
        """Wait for cancellation."""
        await asyncio.Event().wait()
        return "done"

    background = BackgroundSubagents()
    try:
        await background.awrap_tool_call(_request("one", task), _unused_handler)
        await asyncio.sleep(0)
        listing, cancel = background.tools
        jobs = await listing.ainvoke({"runtime": _request("one", task).runtime})
        assert jobs[0]["status"] == "running"
        task_id = jobs[0]["task_id"]
        assert (
            await cancel.ainvoke({"task_id": task_id, "runtime": _request("two", task).runtime})
            == "Unknown subagent for this conversation."
        )
        refused = await background.awrap_tool_call(_request("two", task), _unused_handler)
        assert "capacity" in refused.content
        assert (
            await cancel.ainvoke({"task_id": task_id, "runtime": _request("one", task).runtime})
            == "cancelled"
        )
        assert not background.results("one")
    finally:
        await background.cancel()


async def test_cancel_finished_subagent_preserves_result():
    @tool
    async def task() -> str:
        """Return completed research."""
        return "completed research"

    background = BackgroundSubagents()
    await background.awrap_tool_call(_request("one", task), _unused_handler)
    await asyncio.gather(*(job.worker for job in background._jobs.values()))
    listing, cancel = background.tools
    runtime = _request("one", task).runtime
    jobs = await listing.ainvoke({"runtime": runtime})
    task_id = jobs[0]["task_id"]
    assert jobs[0]["status"] == "finished"
    assert await cancel.ainvoke({"task_id": task_id, "runtime": runtime}) == "finished"
    results = background.results("one")
    assert "completed research" in results[task_id]
    assert background.owners() == {"one"}
    assert not background.results("two")
    background.acknowledge(results)
    assert not background.results("one")
    assert not background.owners()


@pytest.mark.parametrize("owner", ["one", None])
async def test_conversation_cancel_discards_finished_results(owner):
    @tool
    async def task() -> str:
        """Return completed research."""
        return "completed research"

    background = BackgroundSubagents()
    for thread in ("one", "two"):
        await background.awrap_tool_call(_request(thread, task), _unused_handler)
    await asyncio.gather(*(job.worker for job in background._jobs.values()))
    assert background.results("one")
    assert background.results("two")
    assert await background.cancel(owner)
    assert not background.results("one")
    assert bool(background.results("two")) == (owner == "one")
    assert background.owners() == ({"two"} if owner == "one" else set())


async def test_remote_stream_uses_original_target_and_cancels_on_thread_stop(monkeypatch):
    connected, disconnected = asyncio.Event(), asyncio.Event()
    targets = []

    async def stream(*_args: object, **kwargs: object):
        assert kwargs["on_disconnect"] == "cancel"
        connected.set()
        try:
            await asyncio.Event().wait()
            yield SimpleNamespace(event="values", data={})
        finally:
            disconnected.set()

    def client(**kwargs: object):
        targets.append(kwargs["url"])
        return SimpleNamespace(runs=SimpleNamespace(stream=stream))

    monkeypatch.setattr("deepagents_talon.background.get_client", client)
    background = BackgroundSubagents()
    old = background.configured(
        [
            {
                "name": "remote",
                "description": "research",
                "graph_id": "g",
                "url": "https://old.example",
            }
        ]
    )
    background.configured(
        [
            {
                "name": "remote",
                "description": "research",
                "graph_id": "g",
                "url": "https://new.example",
            }
        ]
    )

    @tool
    async def start_async_task() -> str:
        """Start remote work."""
        return "unused"

    await old.awrap_tool_call(
        _request("one", start_async_task, subagent_type="remote", description="work"),
        _unused_handler,
    )
    await asyncio.wait_for(connected.wait(), 2)
    assert await background.cancel("one")
    assert disconnected.is_set()
    assert targets == ["https://old.example"]


async def test_interrupted_main_keeps_worker_and_retries_unprocessed_result(monkeypatch):
    release, paused = asyncio.Event(), asyncio.Event()

    async def child(_state):
        await release.wait()
        return {"messages": [AIMessage(content="research result")]}

    runtime = _runtime(
        monkeypatch,
        child,
        [
            _delegate(),
            AIMessage(content="Started"),
            AIMessage(content="Hello"),
            AIMessage(content="Processing"),
            AIMessage(content="Processed"),
        ],
    )
    await runtime.start()
    original = runtime._invoke_until_text

    async def pause_after_checkpoint(request, activity):
        result = await original(request, activity)
        paused.set()
        await asyncio.Event().wait()
        return result

    monkeypatch.setattr(runtime, "_invoke_until_text", pause_after_checkpoint)
    try:
        turn = asyncio.create_task(runtime.invoke(AgentRequest("chat", "delegate")))
        await asyncio.wait_for(paused.wait(), 2)
        turn.cancel()
        await asyncio.gather(turn, return_exceptions=True)
        await runtime.recover_interrupted("chat")
        assert runtime.background.owners() == {"chat"}
        monkeypatch.setattr(runtime, "_invoke_until_text", original)
        assert (await runtime.invoke(AgentRequest("chat", "hello"))).text == "Hello"
        release.set()
        await asyncio.gather(*(job.worker for job in runtime.background._jobs.values()))
        result_ids = set(runtime.background.results("chat"))
        paused.clear()
        monkeypatch.setattr(runtime, "_invoke_until_text", pause_after_checkpoint)
        turn = asyncio.create_task(runtime.invoke(AgentRequest("chat", "process results")))
        await asyncio.wait_for(paused.wait(), 2)
        turn.cancel()
        await asyncio.gather(turn, return_exceptions=True)
        assert set(runtime.background.results("chat")) == result_ids
        monkeypatch.setattr(runtime, "_invoke_until_text", original)
        assert (await runtime.invoke(AgentRequest("chat", "finish"))).text == "Processed"
        state = await runtime._graph.aget_state({"configurable": {"thread_id": "chat"}})
        assert sum(message.id in result_ids for message in state.values["messages"]) == 1
        assert not runtime.background.results("chat")
    finally:
        release.set()
        await runtime.stop()


async def test_background_worker_inherits_caller_context_and_isolates_its_own():
    marker = contextvars.ContextVar("marker", default="unset")

    @tool
    async def task() -> str:
        """Report the context the worker runs in."""
        return f"{marker.get()}/{_IN_SUBAGENT.get()}"

    background = BackgroundSubagents()
    marker.set("main conversation")
    await background.awrap_tool_call(_request("one", task), _unused_handler)
    await asyncio.gather(*(job.worker for job in background._jobs.values()))

    assert [job.result for job in background._jobs.values()] == ["main conversation/True"]
    assert _IN_SUBAGENT.get() is False


async def test_background_failure_is_logged_and_reported_without_arguments(caplog):
    @tool
    async def task(credential: str) -> str:
        """Fail while holding a credential."""
        assert credential
        msg = "upstream rejected the request"
        raise RuntimeError(msg)

    background = BackgroundSubagents()
    with caplog.at_level(logging.ERROR, logger="deepagents_talon.background"):
        await background.awrap_tool_call(
            _request("one", task, credential="sk-not-a-real-key"), _unused_handler
        )
        await asyncio.gather(*(job.worker for job in background._jobs.values()))

    (job,) = background._jobs.values()
    assert job.result == "Subagent failed before returning a result."
    assert "RuntimeError: upstream rejected the request" in caplog.text
    assert "sk-not-a-real-key" not in caplog.text


async def test_background_timeout_is_reported_separately_from_failure(monkeypatch):
    monkeypatch.setattr("deepagents_talon.background._TASK_TIMEOUT_SECONDS", 0.01)

    @tool
    async def task() -> str:
        """Never return."""
        await asyncio.Event().wait()
        return "done"

    background = BackgroundSubagents()
    await background.awrap_tool_call(_request("one", task), _unused_handler)
    await asyncio.gather(*(job.worker for job in background._jobs.values()))

    (job,) = background._jobs.values()
    assert job.result == "Subagent ran out of time before returning a result."
    assert not job.cancelled


async def test_repeatedly_failing_turns_drop_the_unprocessed_result(monkeypatch):
    async def child(_state):
        return {"messages": [AIMessage(content="research result")]}

    runtime = _runtime(monkeypatch, child, [_delegate(), AIMessage(content="Started")])
    await runtime.start()
    try:
        assert (await runtime.invoke(AgentRequest("chat", "delegate"))).text == "Started"
        await asyncio.gather(*(job.worker for job in runtime.background._jobs.values()))
        assert runtime.background.results("chat")

        async def fail(_request, _activity):
            msg = "model failed"
            raise RuntimeError(msg)

        monkeypatch.setattr(runtime, "_invoke_until_text", fail)
        failures = 0
        while runtime.background.results("chat") and failures < 8:
            with pytest.raises(RuntimeError):
                await runtime.invoke(AgentRequest("chat", "process results"))
            failures += 1

        assert failures == 3
        assert "chat" not in runtime.background.owners()
        (job,) = runtime.background._jobs.values()
        assert "never reached the user" in job.result
        assert "research result" in job.result
    finally:
        await runtime.stop()


async def test_background_worker_keeps_scoped_state_but_not_the_authorization_handler():
    async def authorize(_event):
        return None

    @tool
    async def task() -> str:
        """Report the scoped state this worker inherited."""
        return f"{_HISTORY_SCOPE.get()}|{_CRON_ORIGIN.get()}|{current_authorization_handler()}"

    scope = ArchiveScope(talon_history_channel="whatsapp", talon_history_chat="chat")
    origin = CronOrigin("chat")
    background = BackgroundSubagents()
    _HISTORY_SCOPE.set(scope)
    _CRON_ORIGIN.set(origin)
    token = set_authorization_handler(authorize)
    try:
        await background.awrap_tool_call(_request("one", task), _unused_handler)
        await asyncio.gather(*(job.worker for job in background._jobs.values()))
        assert current_authorization_handler() is authorize
    finally:
        reset_authorization_handler(token)

    (job,) = background._jobs.values()
    assert job.result == f"{scope}|{origin}|None"
