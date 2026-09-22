from __future__ import annotations

import asyncio
import contextvars
import logging
from types import SimpleNamespace

import pytest
from langchain.tools import ToolRuntime
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langgraph.types import Command

from deepagents_talon.archive import ArchiveScope
from deepagents_talon.authorization import (
    current_authorization_handler,
    reset_authorization_handler,
    set_authorization_handler,
)
from deepagents_talon.background import (
    _FAILED_RESULT,
    _IN_SUBAGENT,
    _INSTRUCTIONS,
    _MAX_RESULT_CHARACTERS,
    _SCHEDULED_INSTRUCTIONS,
    _SCHEDULED_TURN,
    _TIMED_OUT_RESULT,
    BackgroundSubagents,
)
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
        await asyncio.wait_for(host._tasks["test:chat"], 2)
        await host.receive_message(channel, ChannelMessage("chat", "hello"))
        await asyncio.wait_for(host._tasks["test:chat"], 2)
        assert runtime.background.owners() == {"test:chat"}
        assert not runtime.background.results("test:chat")
        release.set()
        await asyncio.gather(*(job.worker for job in runtime.background._jobs.values()))
        await host._dispatch_background_results()
        await asyncio.wait_for(host._tasks["test:chat"], 2)
        assert channel.sent == [
            ("chat", "Working on it"),
            ("chat", "Still here"),
            ("chat", "Processed research"),
        ]
        state = await runtime._graph.aget_state({"configurable": {"thread_id": "test:chat"}})
        assert any(
            "raw research result" in str(message.content) for message in state.values["messages"]
        )
        assert child_threads[0] != "test:chat"
        assert not runtime.background.results("test:chat")
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
            await asyncio.wait_for(host._tasks[f"test:{owner}"], 2)
        assert channel.sent == [("one", "Started"), ("two", "Started")]
        await host.receive_message(channel, ChannelMessage("one", command))
        assert len(cancelled) == 1
        assert runtime.background.owners() == {"test:two"}
        assert not runtime.background.results("test:one")
        if command == "/new":
            assert host._agent_conversation_id("test:one") != "test:one"
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


async def test_invoke_reports_the_results_it_acknowledged(monkeypatch):
    """The runtime hands back what it acknowledged so a host can undo it.

    Acknowledgement records that the model consumed a result, which the runtime
    knows; whether the user was told depends on the reply being delivered, which
    only the host knows. The ids travel so the two can be reconciled.
    """

    async def child(_state):
        return {"messages": [AIMessage(content="research result")]}

    runtime = _runtime(
        monkeypatch,
        child,
        [
            _delegate(),
            AIMessage(content="Working on it"),
            AIMessage(content="Processed research"),
        ],
    )
    await runtime.start()
    try:
        launched = await runtime.invoke(AgentRequest(conversation_id="chat", text="research"))
        assert launched.background_results == ()

        await asyncio.gather(*(job.worker for job in runtime.background._jobs.values()))
        pending = set(runtime.background.results("chat"))
        assert pending

        processed = await runtime.invoke(AgentRequest(conversation_id="chat", text="anything else"))

        assert set(processed.background_results) == pending
        assert not runtime.background.results("chat")

        runtime.background.requeue(processed.background_results)

        assert set(runtime.background.results("chat")) == pending
    finally:
        await runtime.stop()


async def test_requeue_returns_only_the_results_it_is_given():
    """Re-queueing is scoped to one turn's ids, not to everything acknowledged.

    A conversation can hold results from several turns. Only the turn whose reply
    was discarded goes back to the queue; anything an earlier turn delivered stays
    acknowledged, so it is never reported to the user twice.
    """

    @tool
    async def task() -> str:
        """Return completed research."""
        return "completed research"

    background = BackgroundSubagents()
    for thread in ("one", "two"):
        await background.awrap_tool_call(_request(thread, task), _unused_handler)
    await asyncio.gather(*(job.worker for job in background._jobs.values()))
    first = background.results("one")
    second = background.results("two")
    background.acknowledge(first)
    background.acknowledge(second)
    assert not background.results("one")
    assert not background.results("two")
    assert not background.owners()

    background.requeue(first)

    assert background.results("one") == first
    assert not background.results("two")
    assert background.owners() == {"one"}


async def test_requeue_skips_cancelled_and_unknown_results():
    """Nothing is resurrected that the conversation has no use for.

    `/stop` discards a thread's results deliberately, and a result already pruned
    is gone. Re-queueing either would be an undelivered turn reviving work the user
    stopped, so both are left alone.
    """

    @tool
    async def task() -> str:
        """Return completed research."""
        return "completed research"

    background = BackgroundSubagents()
    await background.awrap_tool_call(_request("one", task), _unused_handler)
    await asyncio.gather(*(job.worker for job in background._jobs.values()))
    results = background.results("one")
    background.acknowledge(results)
    assert await background.cancel("one")

    background.requeue(results)
    background.requeue(["subagent-never-existed"])

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


async def test_host_shutdown_completes_when_a_worker_outlives_cancellation(tmp_path, monkeypatch):
    async def child(_state):
        return {"messages": [AIMessage(content="research result")]}

    runtime = _runtime(monkeypatch, child, [AIMessage(content="Hello")])
    channel = RecordingChannel()
    host = TalonHost(config=_config(tmp_path), agent=runtime, channels=[channel])
    await host.start()

    async def refuse(_owner=None):
        return False

    monkeypatch.setattr(runtime.background, "cancel", refuse)
    await host.stop()

    # The runtime raises rather than closing resources a live worker may write to,
    # and the host treats that as a component failure so shutdown still finishes.
    assert channel.stopped is True
    assert host._stopped.is_set()
    assert runtime._graph is not None


def _scheduled():
    """Enter a scheduled turn, as the runtime does for a cron run."""
    return _SCHEDULED_TURN.set(True)


async def test_scheduled_delegation_runs_inline_and_creates_no_job():
    """A scheduled run waits for its subagent instead of detaching it.

    Also guards the middleware order this path depends on: `TaskTools` wraps
    `BackgroundSubagents`, so `handler` here is the subagent itself. Were that ever
    inverted, `_IN_SUBAGENT` would make every scheduled delegation refuse instead.
    """
    seen = []

    async def handler(request):
        seen.append(_IN_SUBAGENT.get())
        return ToolMessage("findings", tool_call_id=request.tool_call["id"])

    @tool
    async def task() -> str:
        """Delegate."""
        return "unused"

    background = BackgroundSubagents()
    token = _scheduled()
    try:
        result = await background.awrap_tool_call(
            _request("job:talon-cron", task, subagent_type="researcher", description="go"),
            handler,
        )
    finally:
        _SCHEDULED_TURN.reset(token)

    assert result.content == "findings"
    # The subagent ran with delegation closed to it, and nothing was left behind to
    # deliver later.
    assert seen == [True]
    assert background._jobs == {}
    assert _IN_SUBAGENT.get() is False


async def test_scheduled_fan_out_is_concurrent():
    """Delegations gathered in one assistant message must not serialize on the lock."""
    inside = asyncio.Event()
    both = asyncio.Event()
    peak = 0
    running = 0

    async def handler(request):
        nonlocal peak, running
        running += 1
        peak = max(peak, running)
        inside.set()
        if peak == 2:
            both.set()
        await asyncio.wait_for(both.wait(), 2)
        running -= 1
        return ToolMessage("done", tool_call_id=request.tool_call["id"])

    @tool
    async def task() -> str:
        """Delegate."""
        return "unused"

    background = BackgroundSubagents()
    token = _scheduled()
    try:
        results = await asyncio.wait_for(
            asyncio.gather(
                *(
                    background.awrap_tool_call(
                        _request("job:talon-cron", task, subagent_type="researcher"), handler
                    )
                    for _ in range(2)
                )
            ),
            2,
        )
    finally:
        _SCHEDULED_TURN.reset(token)

    assert peak == 2
    assert [item.content for item in results] == ["done", "done"]


async def test_inline_fan_out_queues_beyond_the_slot_limit(monkeypatch):
    """Inline delegation escapes the job table, so it needs its own ceiling.

    Queued rather than refused: a scheduled run has nobody to retry a refusal, so
    every delegation must eventually produce a real result.
    """
    monkeypatch.setattr("deepagents_talon.background._MAX_INLINE_RUNNING", 2)
    peak = 0
    running = 0

    async def handler(request):
        nonlocal peak, running
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0)
        running -= 1
        return ToolMessage("done", tool_call_id=request.tool_call["id"])

    @tool
    async def task() -> str:
        """Delegate."""
        return "unused"

    background = BackgroundSubagents()
    token = _scheduled()
    try:
        results = await asyncio.wait_for(
            asyncio.gather(
                *(
                    background.awrap_tool_call(
                        _request("job:talon-cron", task, subagent_type="researcher"), handler
                    )
                    for _ in range(5)
                )
            ),
            5,
        )
    finally:
        _SCHEDULED_TURN.reset(token)

    assert peak == 2
    assert [item.content for item in results] == ["done"] * 5


async def test_inline_timeout_is_reported_separately_from_failure():
    async def handler(_request):
        await asyncio.Event().wait()

    @tool
    async def task() -> str:
        """Delegate."""
        return "unused"

    background = BackgroundSubagents(inline_timeout=0.01)
    token = _scheduled()
    try:
        result = await asyncio.wait_for(
            background.awrap_tool_call(
                _request("job:talon-cron", task, subagent_type="researcher"), handler
            ),
            2,
        )
    finally:
        _SCHEDULED_TURN.reset(token)

    assert result.content == _TIMED_OUT_RESULT
    assert result.status == "error"


async def test_inline_failure_is_reported_without_arguments(caplog):
    async def handler(_request):
        msg = "boom sk-secret-token"
        raise RuntimeError(msg)

    @tool
    async def task() -> str:
        """Delegate."""
        return "unused"

    background = BackgroundSubagents()
    token = _scheduled()
    try:
        with caplog.at_level(logging.ERROR):
            result = await background.awrap_tool_call(
                _request(
                    "job:talon-cron", task, subagent_type="researcher", description="sk-secret-arg"
                ),
                handler,
            )
    finally:
        _SCHEDULED_TURN.reset(token)

    assert result.content == _FAILED_RESULT
    assert "sk-secret-arg" not in result.content
    assert "sk-secret-arg" not in caplog.text


async def test_inline_result_is_truncated():
    """The scheduled thread is reused on every fire, so one result cannot fill it."""

    async def handler(request):
        return Command(
            update={"messages": [ToolMessage("x" * 100_000, tool_call_id=request.tool_call["id"])]}
        )

    @tool
    async def task() -> str:
        """Delegate."""
        return "unused"

    background = BackgroundSubagents()
    token = _scheduled()
    try:
        result = await background.awrap_tool_call(
            _request("job:talon-cron", task, subagent_type="researcher"), handler
        )
    finally:
        _SCHEDULED_TURN.reset(token)

    assert len(result.update["messages"][0].content) == _MAX_RESULT_CHARACTERS


async def test_scheduled_run_does_not_consume_background_capacity(monkeypatch):
    """Cron delegations no longer compete with chat for the worker slots."""
    monkeypatch.setattr("deepagents_talon.background._MAX_RUNNING", 1)

    async def handler(request):
        return ToolMessage("findings", tool_call_id=request.tool_call["id"])

    @tool
    async def task() -> str:
        """Delegate."""
        await asyncio.Event().wait()
        return "done"

    background = BackgroundSubagents()
    token = _scheduled()
    try:
        await background.awrap_tool_call(
            _request("job:talon-cron", task, subagent_type="researcher"), handler
        )
    finally:
        _SCHEDULED_TURN.reset(token)

    chat = await background.awrap_tool_call(
        _request("chat", task, subagent_type="researcher"), _unused_handler
    )
    assert str(chat.content).startswith("Started background subagent")
    assert await background.cancel("chat")


async def test_scheduled_start_async_task_streams_the_remote(monkeypatch):
    """The SDK tool would hand back a task id this turn has no way to resolve."""
    created = []

    async def stream(*_args: object, **kwargs: object):
        assert kwargs["on_disconnect"] == "cancel"
        yield SimpleNamespace(
            event="values", data={"messages": [{"role": "assistant", "content": "remote findings"}]}
        )

    def client(**_kwargs: object):
        return SimpleNamespace(
            runs=SimpleNamespace(
                stream=stream, create=lambda **_k: created.append(_k) or SimpleNamespace()
            )
        )

    monkeypatch.setattr("deepagents_talon.background.get_client", client)

    @tool
    async def start_async_task() -> str:
        """Start remote work."""
        return "unused"

    background = BackgroundSubagents().configured(
        [{"name": "remote", "description": "research", "graph_id": "g", "url": "https://e.example"}]
    )
    token = _scheduled()
    try:
        result = await asyncio.wait_for(
            background.awrap_tool_call(
                _request(
                    "job:talon-cron", start_async_task, subagent_type="remote", description="w"
                ),
                _unused_handler,
            ),
            2,
        )
    finally:
        _SCHEDULED_TURN.reset(token)

    assert result.content == "remote findings"
    assert created == []


def _override(**kwargs: object) -> SimpleNamespace:
    return SimpleNamespace(**kwargs)


async def test_scheduled_prompt_replaces_the_background_instructions():
    captured = {}

    async def handler(request):
        captured["system"] = request.system_message.text
        captured["tools"] = [getattr(item, "name", "") for item in request.tools]
        return "response"

    @tool
    async def list_subagents() -> str:
        """Inspect."""
        return "none"

    @tool
    async def researcher_tool() -> str:
        """Work."""
        return "done"

    request = SimpleNamespace(
        system_message=None,
        tools=[list_subagents, researcher_tool],
        override=_override,
    )
    background = BackgroundSubagents()
    token = _scheduled()
    try:
        await background.awrap_model_call(request, handler)
    finally:
        _SCHEDULED_TURN.reset(token)

    assert _SCHEDULED_INSTRUCTIONS in captured["system"]
    assert _INSTRUCTIONS not in captured["system"]
    # Both could only ever report nothing on a run that owns no jobs.
    assert captured["tools"] == ["researcher_tool"]


async def test_inline_timeout_does_not_escape_the_graph(monkeypatch):
    """A deadline that raised would be retried, relaunching every sibling delegation.

    The tool node re-raises anything that is not a tool invocation error, and the
    runtime treats a timeout as retryable, so an escaping deadline re-runs the graph.
    The proof is that the model is answered: a raise leaves the tool call unanswered,
    because the tool node writes no message on the way out.
    """

    async def child(_state):
        await asyncio.Event().wait()
        return {"messages": [AIMessage(content="never")]}

    runtime = _runtime(monkeypatch, child, [_delegate(), AIMessage(content="scan complete")])
    runtime.background = BackgroundSubagents(inline_timeout=0.01)
    await runtime.start()
    try:
        result = await asyncio.wait_for(
            runtime.invoke(
                AgentRequest(
                    conversation_id="job:talon-cron", text="scan", metadata={"trigger": "cron"}
                )
            ),
            10,
        )
        state = await runtime._graph.aget_state({"configurable": {"thread_id": "job:talon-cron"}})
    finally:
        await runtime.stop()

    assert result.text == "scan complete"
    answers = [
        message.content
        for message in state.values["messages"]
        if isinstance(message, ToolMessage) and message.tool_call_id == "launch"
    ]
    assert answers == [_TIMED_OUT_RESULT]


@pytest.mark.parametrize("failing", [False, True])
async def test_scheduled_flag_does_not_outlive_its_turn(monkeypatch, failing):
    """A leaked flag would make the next chat turn on this process delegate inline."""

    async def child(_state):
        return {"messages": [AIMessage(content="findings")]}

    responses = [AIMessage(content="scan complete")]
    runtime = _runtime(monkeypatch, child, responses)
    if failing:

        async def explode(*_args: object, **_kwargs: object) -> None:
            msg = "turn failed"
            raise RuntimeError(msg)

        monkeypatch.setattr(runtime, "_invoke_until_text", explode)

    await runtime.start()
    try:
        request = AgentRequest(
            conversation_id="job:talon-cron", text="scan", metadata={"trigger": "cron"}
        )
        if failing:
            with pytest.raises(RuntimeError):
                await runtime.invoke(request)
        else:
            await runtime.invoke(request)
    finally:
        await runtime.stop()

    assert _SCHEDULED_TURN.get() is False


async def test_chat_turn_is_not_scheduled(monkeypatch):
    """Only a cron turn inlines; a chat delivery turn keeps detaching."""
    seen = []

    async def child(_state):
        return {"messages": [AIMessage(content="findings")]}

    runtime = _runtime(monkeypatch, child, [AIMessage(content="done")])
    original = runtime.background.awrap_model_call

    async def spy(request, handler):
        seen.append(_SCHEDULED_TURN.get())
        return await original(request, handler)

    monkeypatch.setattr(runtime.background, "awrap_model_call", spy)
    await runtime.start()
    try:
        await runtime.invoke(
            AgentRequest(
                conversation_id="chat",
                text="hello",
                metadata={"background_delivery": True},
            )
        )
    finally:
        await runtime.stop()

    assert seen == [False]
