from __future__ import annotations

import asyncio

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from deepagents_talon.background import BackgroundSubagents
from deepagents_talon.interfaces import AgentRequest
from deepagents_talon.subagent_control import (
    _MAX_EVENTS,
    _MAX_MESSAGE_CHARACTERS,
    _MAX_MESSAGES,
    SubagentControl,
)
from deepagents_talon.subagents import _compile_fresh
from tests.unit_tests.test_background import ToolModel, _request, _unused_handler
from tests.unit_tests.test_research_subagents import (
    _call,
    _runtime,
    _write_agent,
)


async def test_local_worker_receives_followup_after_running_tool():
    entered, release = asyncio.Event(), asyncio.Event()
    observed = []

    @tool
    async def research() -> str:
        """Wait for research."""
        entered.set()
        await release.wait()
        return "private tool output"

    class RecordingModel(ToolModel):
        async def _agenerate(self, messages, *args: object, **kwargs: object):
            observed.append(messages)
            return await super()._agenerate(messages, *args, **kwargs)

    model = RecordingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": "research", "args": {}, "id": "research-call"}],
            ),
            AIMessage(content="Summary so far"),
        ]
    )
    spec = {"name": "researcher", "description": "Research", "tools": [research]}
    graph = _compile_fresh(spec, model, {})["runnable"]

    @tool
    async def task() -> str:
        """Run a local researcher."""
        result = await graph.ainvoke({"messages": [{"role": "user", "content": "Research"}]})
        return result["messages"][-1].content

    background = BackgroundSubagents().configured([spec])
    request = _request("owner", task, subagent_type="researcher")
    inspect, message = background.tools[2:]
    try:
        await background.awrap_tool_call(request, _unused_handler)
        await asyncio.wait_for(entered.wait(), 2)
        task_id = next(iter(background._jobs))
        inputs = {"task_id": task_id, "runtime": request.runtime}
        before = await inspect.ainvoke(inputs)
        assert before["events"][-1]["event"] == "tool_started"
        assert before["events"][-1]["tool"] == "research"
        receipt = await message.ainvoke({**inputs, "message": "Stop researching; summarize now"})
        assert receipt["status"] == "queued"
        assert (await inspect.ainvoke(inputs))["messages"] == [receipt]
        assert not background._jobs[task_id].worker.done()
        release.set()
        await asyncio.wait_for(background._jobs[task_id].worker, 2)
        after = await inspect.ainvoke(inputs)
        assert after["messages"] == [{**receipt, "status": "consumed"}]
        assert any("Stop researching; summarize now" in str(m.content) for m in observed[-1])
        assert background._jobs[task_id].result == "Summary so far"
        assert "private tool output" not in str(after)
        assert "Stop researching" not in str(after)
        assert all(event["timestamp"] for event in after["events"])
        assert "error" in await message.ainvoke({**inputs, "message": "too late"})
    finally:
        release.set()
        await background.cancel()


async def test_control_tools_enforce_ownership_and_opaque_limitations():
    @tool
    async def task() -> str:
        """Wait."""
        await asyncio.Event().wait()
        return "done"

    background = BackgroundSubagents()
    request = _request("owner", task)
    inspect, message = background.tools[2:]
    try:
        await background.awrap_tool_call(request, _unused_handler)
        await asyncio.sleep(0)
        task_id = next(iter(background._jobs))
        for owner in ("another", "", None):
            inputs = {"task_id": task_id, "runtime": _request(owner, task).runtime}
            assert await inspect.ainvoke(inputs) == {
                "error": "Unknown subagent for this conversation."
            }
            assert await message.ainvoke({**inputs, "message": "hurry"}) == {
                "error": "Unknown subagent for this conversation."
            }
        inputs = {"task_id": task_id, "runtime": request.runtime}
        assert (
            "only for Talon-built"
            in (await message.ainvoke({**inputs, "message": "hurry"}))["error"]
        )
        assert (await inspect.ainvoke(inputs))["activity"] == "lifecycle only"
        for limit in (0, _MAX_EVENTS + 1):
            assert "error" in await inspect.ainvoke({**inputs, "limit": limit})
        await background.cancel()
        assert (await inspect.ainvoke(inputs))["status"] == "cancelled"
    finally:
        await background.cancel()


@pytest.mark.parametrize("message", ["   ", "x" * (_MAX_MESSAGE_CHARACTERS + 1)])
def test_message_size_limits(message):
    control = SubagentControl()
    assert "error" in control.enqueue(message)
    assert not control.messages


def test_control_retention_never_evicts_queued_messages():
    control = SubagentControl()
    for _ in range(_MAX_MESSAGES):
        assert control.enqueue("hurry")["status"] == "queued"
    assert "error" in control.enqueue("overflow")
    first = control.messages[0]
    first["status"] = "consumed"
    assert control.enqueue("next")["status"] == "queued"
    assert len(control.messages) == _MAX_MESSAGES
    assert first not in control.messages
    for _ in range(_MAX_EVENTS + 1):
        control.record("model_started")
    assert len(control.events) == _MAX_EVENTS


async def test_runtime_marks_local_agents_messageable(tmp_path, monkeypatch):
    _write_agent(tmp_path, "[research]")
    entered, release = asyncio.Event(), asyncio.Event()

    @tool
    async def research() -> str:
        """Research."""
        entered.set()
        await release.wait()
        return "evidence"

    parent = ToolModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[_call("task", subagent_type="researcher", description="Research")],
            ),
            AIMessage(content="Started"),
        ]
    )
    child = ToolModel(
        responses=[
            AIMessage(content="", tool_calls=[_call("research")]),
            AIMessage(content="Summary"),
        ]
    )
    runtime = _runtime(tmp_path, monkeypatch, parent, child, tools=[research])
    await runtime.start()
    try:
        await runtime.invoke(AgentRequest(conversation_id="owner", text="Research"))
        await asyncio.wait_for(entered.wait(), 2)
        task_id = next(iter(runtime.background._jobs))
        inspect, message = runtime.background.tools[2:]
        tool_runtime = _request("owner", research).runtime
        inputs = {"task_id": task_id, "runtime": tool_runtime}
        assert (await inspect.ainvoke(inputs))["messaging_supported"] is True
        assert (await message.ainvoke({**inputs, "message": "Summarize now"}))["status"] == "queued"
        release.set()
        await asyncio.wait_for(runtime.background._jobs[task_id].worker, 2)
        assert (await inspect.ainvoke(inputs))["messages"][0]["status"] == "consumed"
    finally:
        release.set()
        await runtime.stop()
