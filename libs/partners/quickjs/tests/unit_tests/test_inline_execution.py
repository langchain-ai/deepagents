"""Tests for `execution="inline"`: REPL work on the caller's thread and loop."""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import pytest
from deepagents import create_deep_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from quickjs_rs import Runtime

from langchain_quickjs import CodeInterpreterMiddleware
from langchain_quickjs._repl import _Registry, _ThreadREPL
from langchain_quickjs._worker import InlineWorker
from tests._common import FakeChatModel

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from contextvars import Context

    from langchain_quickjs._worker import ExecutionMode

# ---------------------------------------------------------------------------
# InlineWorker
# ---------------------------------------------------------------------------


async def _running_loop() -> asyncio.AbstractEventLoop:
    return asyncio.get_running_loop()


def test_run_sync_without_a_loop_reuses_one_private_loop() -> None:
    worker = InlineWorker()
    first = worker.run_sync(_running_loop())
    assert worker.run_sync(_running_loop()) is first
    worker.close()
    assert first.is_closed()
    worker.close()  # idempotent


def test_run_sync_serializes_callers_on_different_threads() -> None:
    worker = InlineWorker()
    in_flight = 0
    peak = 0
    guard = threading.Lock()

    async def hold() -> None:
        nonlocal in_flight, peak
        with guard:
            in_flight += 1
            peak = max(peak, in_flight)
        await asyncio.sleep(0.02)
        with guard:
            in_flight -= 1

    threads = [
        threading.Thread(target=worker.run_sync, args=(hold(),)) for _ in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert not any(thread.is_alive() for thread in threads)
    worker.close()
    assert peak == 1


async def test_run_sync_from_a_running_loop_rejects_suspending_work() -> None:
    """An eval parks inside a TaskGroup, which cannot be closed synchronously."""

    async def suspends() -> None:
        async with asyncio.TaskGroup() as group:
            group.create_task(asyncio.sleep(0))
            await asyncio.sleep(0)

    with pytest.raises(RuntimeError, match="suspends"):
        InlineWorker().run_sync(suspends())


async def test_run_async_reraises_a_swallowed_cancellation() -> None:
    worker = InlineWorker()
    started = asyncio.Event()

    async def swallows() -> str:
        started.set()
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            return "swallowed"
        return "not cancelled"

    task = asyncio.create_task(worker.run_async(swallows()))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


# ---------------------------------------------------------------------------
# Middleware and registry wiring
# ---------------------------------------------------------------------------


def test_rejects_invalid_execution() -> None:
    with pytest.raises(ValueError, match="must be one of"):
        CodeInterpreterMiddleware(execution="fibers")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("execution", "timeout", "warned"),
    [("inline", None, True), ("inline", 5.0, False), ("worker", None, False)],
)
def test_warns_only_for_inline_without_timeout(
    caplog: pytest.LogCaptureFixture,
    execution: ExecutionMode,
    timeout: float | None,
    *,
    warned: bool,
) -> None:
    with caplog.at_level("WARNING", logger="langchain_quickjs.middleware"):
        CodeInterpreterMiddleware(execution=execution, timeout=timeout)
    assert ("blocks the caller's event loop" in caplog.text) is warned


def test_evict_closes_the_private_loop() -> None:
    registry = _Registry(
        memory_limit=64 * 1024 * 1024,
        timeout=5.0,
        capture_console=True,
        max_stdout_chars=4000,
        execution="inline",
    )
    try:
        assert registry.get("slot").eval_sync("6 * 7").result == "42"
        worker = registry._slots["slot"].worker
        assert isinstance(worker, InlineWorker)
        loop = worker._loop
        assert loop is not None
        registry.evict("slot")
        assert worker._loop is None
        assert loop.is_closed()
    finally:
        registry.close()


def _inline_repl(worker: InlineWorker, runtime: Runtime) -> _ThreadREPL:
    return _ThreadREPL(
        worker,
        runtime,
        timeout=5.0,
        capture_console=True,
        max_stdout_chars=4000,
        subagents_enabled=False,
    )


async def test_eval_sync_inline_refuses_a_running_loop() -> None:
    runtime = Runtime()
    repl = _inline_repl(InlineWorker(), runtime)
    try:
        with pytest.raises(RuntimeError, match="eval_async"):
            repl.eval_sync("1 + 1")
    finally:
        await repl.aclose()
        runtime.close()


# ---------------------------------------------------------------------------
# A loop that cannot be woken from another thread
# ---------------------------------------------------------------------------


class _NoCrossThreadWakeupLoop(asyncio.SelectorEventLoop):
    """Refuse cross-thread wake-ups, as a loop only its owner can drive would."""

    def call_soon_threadsafe(
        self,
        callback: Callable[..., object],
        *args: object,
        context: Context | None = None,
    ) -> asyncio.Handle:
        del callback, args, context
        msg = "this loop cannot be woken from another thread"
        raise RuntimeError(msg)


@tool("current_loop_id")
async def current_loop_id() -> str:
    """Return the id of the loop the tool runs on."""
    await asyncio.sleep(0)
    return str(id(asyncio.get_running_loop()))


@pytest.mark.timeout(30)
def test_inline_repl_completes_on_a_loop_without_cross_thread_wakeups() -> None:
    loop = _NoCrossThreadWakeupLoop()
    worker = InlineWorker()

    async def scenario() -> str:
        runtime = Runtime()
        repl = _inline_repl(worker, runtime)
        try:
            repl.install_tools([current_loop_id])
            outcome = await repl.eval_async(
                "await tools.currentLoopId({})",
                outer_loop=asyncio.get_running_loop(),
            )
            await repl.acreate_snapshot()
        finally:
            await repl.aclose()
            runtime.close()
        assert outcome.error_type is None, outcome.error_message
        return outcome.result or ""

    try:
        result = loop.run_until_complete(scenario())
    finally:
        loop.close()
        worker.close()
    assert str(id(loop)) in result


# ---------------------------------------------------------------------------
# End to end through the middleware
# ---------------------------------------------------------------------------


def _worker_thread_names() -> set[str]:
    return {
        thread.name
        for thread in threading.enumerate()
        if thread.name.startswith("quickjs-worker")
    }


@tool("live_worker_threads")
async def live_worker_threads() -> str:
    """Return the names of live QuickJS worker threads, comma-separated."""
    await asyncio.sleep(0)
    return ",".join(sorted(_worker_thread_names()))


@tool
def list_user_ids() -> list[int]:
    """List user IDs."""
    return [1, 21, 35]


def _script(code: str, *, final_message: str = "done") -> Iterator[AIMessage]:
    return iter(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "eval",
                        "args": {"code": code},
                        "id": "call_1",
                        "type": "tool_call",
                    },
                ],
            ),
            AIMessage(content=final_message),
        ]
    )


def _eval_tool_message(result: dict[str, Any]) -> ToolMessage:
    messages = [
        m for m in result["messages"] if isinstance(m, ToolMessage) and m.name == "eval"
    ]
    assert messages, "expected at least one eval ToolMessage"
    return messages[-1]


def _result_text(result: dict[str, Any]) -> str:
    content = _eval_tool_message(result).content
    assert isinstance(content, str), content
    assert "<error" not in content, content
    assert "<result" in content, content
    return content


def _agent(code: str, middleware: CodeInterpreterMiddleware, **kwargs: Any) -> Any:
    return create_deep_agent(
        model=FakeChatModel(messages=_script(code)),
        middleware=[middleware],
        **kwargs,
    )


def _threads_started_by_eval(result: dict[str, Any], before: set[str]) -> set[str]:
    """Worker threads the probe tool saw that did not exist before the run."""
    text = _result_text(result)
    start = text.index("<result") + len("<result")
    body = text[start:].split(">", 1)[1]
    names = {n for n in body.split("</result")[0].split(",") if n}
    return names - before


async def test_inline_async_eval_runs_ptc_on_the_caller_loop() -> None:
    result = await _agent(
        "await tools.currentLoopId({})",
        CodeInterpreterMiddleware(execution="inline", ptc=[current_loop_id]),
    ).ainvoke({"messages": [HumanMessage(content="go")]})
    assert str(id(asyncio.get_running_loop())) in _result_text(result)


@pytest.mark.parametrize(
    ("execution", "starts_thread"), [("worker", True), ("inline", False)]
)
async def test_worker_thread_starts_only_in_worker_mode(
    execution: ExecutionMode, starts_thread: bool
) -> None:
    before = _worker_thread_names()
    result = await _agent(
        "await tools.liveWorkerThreads({})",
        CodeInterpreterMiddleware(execution=execution, ptc=[live_worker_threads]),
    ).ainvoke({"messages": [HumanMessage(content="go")]})
    started = _threads_started_by_eval(result, before)
    assert bool(started) is starts_thread, _result_text(result)


def test_inline_sync_invoke_runs_ptc_on_a_private_loop() -> None:
    result = _agent(
        "const ids = await tools.listUserIds({});\nids.join(',');",
        CodeInterpreterMiddleware(execution="inline", ptc=[list_user_ids]),
    ).invoke({"messages": [HumanMessage(content="go")]})
    assert "1,21,35" in _result_text(result)


async def test_inline_sync_invoke_from_a_thread_with_a_running_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sync `invoke` inside async code (a notebook, say) must clean up quietly."""
    closed_loops: list[asyncio.AbstractEventLoop | None] = []
    original_close = InlineWorker.close

    def spying_close(self: InlineWorker) -> None:
        closed_loops.append(self._loop)
        original_close(self)

    monkeypatch.setattr(InlineWorker, "close", spying_close)
    middleware = CodeInterpreterMiddleware(execution="inline", ptc=[list_user_ids])
    result = _agent(
        "const ids = await tools.listUserIds({});\nids.join(',');",
        middleware,
    ).invoke({"messages": [HumanMessage(content="go")]})
    assert "1,21,35" in _result_text(result)
    assert not middleware._registry._slots
    loops = [loop for loop in closed_loops if loop is not None]
    assert loops, "the slot's private loop was never handed to close()"
    assert all(loop.is_closed() for loop in loops)


def test_inline_sync_parallel_agents_across_threads() -> None:
    def _run(index: int) -> tuple[int, dict[str, Any]]:
        result = _agent(
            f"{index} * 10", CodeInterpreterMiddleware(execution="inline")
        ).invoke({"messages": [HumanMessage(content="go")]})
        return index, result

    with ThreadPoolExecutor(max_workers=8) as executor:
        runs = list(executor.map(_run, range(12)))

    assert len(runs) == 12
    for index, result in runs:
        assert str(index * 10) in _result_text(result)


async def test_inline_mode_call_resets_state_between_evals() -> None:
    script = iter(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "eval",
                        "args": {"code": "globalThis.marker = 1"},
                        "id": "call_1",
                        "type": "tool_call",
                    },
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "eval",
                        "args": {"code": "typeof globalThis.marker"},
                        "id": "call_2",
                        "type": "tool_call",
                    },
                ],
            ),
            AIMessage(content="done"),
        ]
    )
    agent = create_deep_agent(
        model=FakeChatModel(messages=script),
        middleware=[CodeInterpreterMiddleware(execution="inline", mode="call")],
    )
    result = await agent.ainvoke({"messages": [HumanMessage(content="go")]})
    assert "undefined" in _result_text(result)


def _task_agent(seen: list[int]) -> Any:
    """An inline agent whose `researcher` subagent records the loop it ran on."""

    def _sync(state: dict[str, Any], config: Any) -> dict[str, Any]:
        del state, config
        return {"messages": [AIMessage(content="subagent-ok")]}

    async def _async(state: dict[str, Any], config: Any) -> dict[str, Any]:
        del state, config
        seen.append(id(asyncio.get_running_loop()))
        return {"messages": [AIMessage(content="subagent-ok")]}

    return create_deep_agent(
        model=FakeChatModel(
            messages=_script(
                "await task({description: 'say hi', subagentType: 'researcher'})"
            )
        ),
        subagents=[
            {
                "name": "researcher",
                "description": "returns one short answer",
                "runnable": RunnableLambda(_sync, afunc=_async),
            }
        ],
        middleware=[CodeInterpreterMiddleware(execution="inline")],
    )


async def test_inline_task_dispatch_runs_subagent_on_caller_loop() -> None:
    seen: list[int] = []
    result = await _task_agent(seen).ainvoke(
        {"messages": [HumanMessage(content="go")]},
        config={"configurable": {"thread_id": "inline-task"}},
    )
    assert "subagent-ok" in _result_text(result)
    assert seen == [id(asyncio.get_running_loop())]


def test_inline_sync_invoke_task_dispatch_runs_subagent() -> None:
    seen: list[int] = []
    result = _task_agent(seen).invoke(
        {"messages": [HumanMessage(content="go")]},
        config={"configurable": {"thread_id": "inline-task-sync"}},
    )
    assert "subagent-ok" in _result_text(result)
    assert len(seen) == 1


async def test_inline_mode_thread_restores_snapshot_across_turns() -> None:
    script = iter(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "eval",
                        "args": {"code": "const counter = 40"},
                        "id": "call_1",
                        "type": "tool_call",
                    },
                ],
            ),
            AIMessage(content="turn 1 done"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "eval",
                        "args": {"code": "counter + 2"},
                        "id": "call_2",
                        "type": "tool_call",
                    },
                ],
            ),
            AIMessage(content="turn 2 done"),
        ]
    )
    agent = create_deep_agent(
        model=FakeChatModel(messages=script),
        middleware=[CodeInterpreterMiddleware(execution="inline")],
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "inline-snapshot"}}
    await agent.ainvoke({"messages": [HumanMessage(content="one")]}, config=config)
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="two")]}, config=config
    )
    assert "42" in _result_text(result)
