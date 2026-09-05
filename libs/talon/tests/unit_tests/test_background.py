from __future__ import annotations

import asyncio

import pytest
from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import interrupt

from deepagents_talon.background import LocalTaskSupervisor


def _runtime(owner="chat"):
    return ToolRuntime(
        state={"messages": [HumanMessage(content="parent context")]},
        context=None,
        tool_call_id="call",
        store=None,
        stream_writer=lambda _: None,
        config={"configurable": {"thread_id": owner}},
    )


def _factory(saver, worker):
    def factory(definition):
        async def run(state):
            return await worker(state, definition)

        graph = StateGraph(MessagesState)
        graph.add_node("worker", run)
        graph.add_edge(START, "worker")
        graph.add_edge("worker", END)
        return graph.compile(checkpointer=saver)

    return factory


async def _wait(supervisor, task_id, status):
    await asyncio.wait_for(asyncio.shield(supervisor._workers[task_id]), timeout=2)
    assert supervisor._task(task_id, "chat")["status"] == status


async def test_worker_survives_parent_and_keeps_definition(tmp_path):
    release = asyncio.Event()
    entered = asyncio.Event()
    seen = []

    async def worker(state, definition):
        seen.extend(message.content for message in state["messages"])
        entered.set()
        await release.wait()
        return {"messages": [AIMessage(content=definition["system_prompt"])]}

    supervisor = LocalTaskSupervisor(tmp_path / "tasks.sqlite", _factory(InMemorySaver(), worker))
    definition = {"name": "researcher", "system_prompt": "original"}
    try:
        launched = await supervisor.start_task(definition, "research", _runtime())
        await entered.wait()
        # New configuration is deliberately a different object, just like graph reload.
        definition = {"name": "researcher", "system_prompt": "replacement"}
        release.set()
        await _wait(supervisor, launched["task_id"], "success")
        record = supervisor._task(launched["task_id"], "chat")
        assert record["result"] == "original"
        assert seen == ["parent context", "research"]
        with pytest.raises(ValueError, match="Unknown local task"):
            await supervisor.cancel(launched["task_id"], "other-chat")
        assert supervisor.pending_results()[0]["id"] == launched["task_id"]
        supervisor.acknowledge(launched["task_id"])
        assert supervisor.pending_results() == []
    finally:
        await supervisor.close()


async def test_restart_requires_explicit_resume_and_retains_prompt(tmp_path):
    entered = asyncio.Event()

    async def blocked(_state, _definition):
        entered.set()
        await asyncio.Event().wait()

    database = tmp_path / "tasks.sqlite"
    checkpoint = str(tmp_path / "checkpoints.sqlite")
    async with AsyncSqliteSaver.from_conn_string(checkpoint) as saver:
        supervisor = LocalTaskSupervisor(database, _factory(saver, blocked))
        launched = await supervisor.start_task({"system_prompt": "original"}, "work", _runtime())
        await entered.wait()
        await supervisor.close()

    calls = []

    async def resumed(_state, definition):
        calls.append(definition)
        return {"messages": [AIMessage(content=definition["system_prompt"])]}

    async with AsyncSqliteSaver.from_conn_string(checkpoint) as saver:
        supervisor = LocalTaskSupervisor(database, _factory(saver, resumed))
        try:
            assert calls == []
            assert supervisor._task(launched["task_id"], "chat")["status"] == "interrupted"
            await supervisor.resume(launched["task_id"], "chat", None)
            await _wait(supervisor, launched["task_id"], "success")
            assert calls == [{"system_prompt": "original"}]
        finally:
            await supervisor.close()


async def test_worker_limits_and_explicit_cancel(tmp_path):
    async def blocked(_state, _definition):
        await asyncio.Event().wait()

    supervisor = LocalTaskSupervisor(tmp_path / "tasks.sqlite", _factory(InMemorySaver(), blocked))
    try:
        tasks = [await supervisor.start_task({}, "work", _runtime()) for _ in range(4)]
        assert "error" in await supervisor.start_task({}, "extra", _runtime())
        cancelled = await supervisor.cancel(tasks[0]["task_id"], "chat")
        assert cancelled["status"] == "cancelled"
        assert supervisor._task(tasks[0]["task_id"], "chat")["status"] == "cancelled"
    finally:
        await supervisor.close()


async def test_paused_approval_requires_explicit_decision(tmp_path):
    async def worker(_state, _definition):
        answer = interrupt({"action_requests": [{"name": "effect"}, {"name": "effect2"}]})
        assert answer == {"decisions": [{"type": "reject"}, {"type": "reject"}]}
        return {"messages": [AIMessage(content="denied")]}

    supervisor = LocalTaskSupervisor(tmp_path / "tasks.sqlite", _factory(InMemorySaver(), worker))
    try:
        launched = await supervisor.start_task({}, "work", _runtime())
        await _wait(supervisor, launched["task_id"], "interrupted")
        previous = supervisor.pending_results()[0]
        assert "error" in await supervisor.resume(launched["task_id"], "chat", None)
        await supervisor.resume(launched["task_id"], "chat", "reject")
        await _wait(supervisor, launched["task_id"], "success")
        assert supervisor._task(launched["task_id"], "chat")["result"] == "denied"
        supervisor.acknowledge(launched["task_id"], revision=previous["revision"])
        assert supervisor.pending_results()[0]["status"] == "success"
    finally:
        await supervisor.close()


async def test_follow_up_interrupts_worker_on_same_thread(tmp_path):
    entered = asyncio.Event()

    async def worker(state, definition):
        if state["messages"][-1].content == "first":
            entered.set()
            await asyncio.Event().wait()
        return {
            "messages": [
                AIMessage(content=definition["system_prompt"] + ":" + state["messages"][-1].content)
            ]
        }

    supervisor = LocalTaskSupervisor(tmp_path / "tasks.sqlite", _factory(InMemorySaver(), worker))
    try:
        launched = await supervisor.start_task({"system_prompt": "original"}, "first", _runtime())
        await entered.wait()
        updated = await supervisor.update(launched["task_id"], "chat", "second")
        assert updated["task_id"] == launched["task_id"]
        await _wait(supervisor, launched["task_id"], "success")
        assert supervisor._task(launched["task_id"], "chat")["result"] == "original:second"
    finally:
        await supervisor.close()
