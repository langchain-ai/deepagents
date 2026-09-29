"""Checkpointed cost ownership for JavaScript subagent dispatch."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Literal, NotRequired

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence
    from pathlib import Path

    from deepagents.middleware.subagents import SubAgent
    from langchain.agents.middleware import AgentMiddleware
    from langchain_core.messages.ai import UsageMetadata
    from langchain_core.runnables import RunnableConfig
    from langchain_core.tools import BaseTool
    from langgraph.checkpoint.base import (
        ChannelVersions,
        Checkpoint,
        CheckpointMetadata,
    )
    from langgraph.graph.state import CompiledStateGraph

import pytest
from deepagents.backends import StateBackend
from deepagents.middleware import SubAgentMiddleware
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command, interrupt

from deepagents_code import cost_tracking
from deepagents_code._fake_models import _ToolBindingFakeModel
from deepagents_code._js_cost import CostAwareCodeInterpreterMiddleware
from deepagents_code.cost_tracking import CostTrackingMiddleware

_INTERPRETERS: list[CostAwareCodeInterpreterMiddleware] = []
_CONFIG: RunnableConfig = {"configurable": {"thread_id": "js-cost"}}


def _usage() -> UsageMetadata:
    return {
        "input_tokens": 1000,
        "output_tokens": 100,
        "total_tokens": 1100,
        "input_token_details": {"cache_creation": 100, "cache_read": 400},
        "output_token_details": {"reasoning": 40},
    }


_ESTIMATE = cost_tracking._CostEstimate(
    total_cost_usd=1.0,
    input_cost_usd=0.7,
    output_cost_usd=0.3,
    input_tokens=1000,
    output_tokens=100,
    cache_creation_tokens=100,
    cache_read_tokens=400,
    reasoning_tokens=40,
    cache_creation_cost_usd=0.2,
    cache_read_cost_usd=0.1,
    reasoning_cost_usd=0.12,
)


def _message(message_id: str) -> AIMessage:
    return AIMessage(content="done", id=message_id, usage_metadata=_usage())


def _fake_model(*messages: AIMessage) -> _ToolBindingFakeModel:
    return _ToolBindingFakeModel(messages=iter(messages), disable_streaming=True)


@pytest.fixture(autouse=True)
def controlled_cost(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(cost_tracking, "_estimate_cost", lambda *_args: _ESTIMATE)
    monkeypatch.setattr(cost_tracking, "pricing_data_available", lambda: True)
    token = cost_tracking._RECORDER_VAR.set(cost_tracking._SessionCostRecorder())
    yield
    cost_tracking._RECORDER_VAR.reset(token)
    for interpreter in _INTERPRETERS:
        interpreter._registry.close()
    _INTERPRETERS.clear()


def _call(code: str, *, message_id: str = "parent-call") -> AIMessage:
    return _tool_call(
        "js_eval", {"code": code}, message_id=message_id, call_id=message_id
    )


def _tool_call(
    name: str,
    args: dict[str, object] | None = None,
    *,
    message_id: str | None = None,
    call_id: str | None = None,
) -> AIMessage:
    return AIMessage(
        content="",
        id=message_id or name,
        usage_metadata=_usage(),
        tool_calls=[{"name": name, "args": args or {}, "id": call_id or name}],
    )


def _child(*messages: AIMessage, tools: Sequence[BaseTool] = ()) -> CompiledStateGraph:
    return create_agent(
        _fake_model(*messages),
        tools=tools,
        middleware=[CostTrackingMiddleware(nested=True)],
    )


def _parent(
    code: str,
    children: dict[str, Any],
    saver: InMemorySaver | AsyncSqliteSaver | None,
    *,
    resuming: bool = False,
    nested: bool = False,
    messages: list[AIMessage] | None = None,
    mode: Literal["thread", "turn", "call"] = "thread",
) -> CompiledStateGraph:
    interpreter = CostAwareCodeInterpreterMiddleware(tool_name="js_eval", mode=mode)
    _INTERPRETERS.append(interpreter)
    middleware: list[AgentMiddleware[Any, Any, Any]] = [
        CostTrackingMiddleware(nested=nested),
        SubAgentMiddleware(
            backend=StateBackend(),
            subagents=[
                {"name": name, "description": name, "runnable": child}
                for name, child in children.items()
            ],
            private_state_keys=frozenset({"_session_cost_usd"}),
        ),
        interpreter,
    ]
    return create_agent(
        _fake_model(
            *(
                messages
                if messages is not None
                else [*([] if resuming else [_call(code)]), _message("parent-done")]
            )
        ),
        middleware=middleware,
        checkpointer=saver,
    )


async def _total(agent: CompiledStateGraph) -> float:
    snapshot = await agent.aget_state(_CONFIG)
    return snapshot.values.get("_session_cost_usd", 0.0)


async def _assert_accounting(
    agent: CompiledStateGraph,
    requests: int,
    *,
    priced: int | None = None,
    charged: int | None = None,
    reported: int | None = None,
) -> None:
    priced = requests if priced is None else priced
    charged = priced if charged is None else charged
    reported = requests if reported is None else reported
    snapshot = await agent.aget_state(_CONFIG)
    assert snapshot.values.get("_session_cost_usd", 0.0) == pytest.approx(charged)
    assert snapshot.values["_session_cost_breakdown"] == pytest.approx(
        {
            "version": 1,
            "request_count": requests,
            "priced_request_count": priced,
            "input_tokens": 1000 * reported,
            "output_tokens": 100 * reported,
            "cache_creation_tokens": 100 * reported,
            "cache_read_tokens": 400 * reported,
            "reasoning_tokens": 40 * reported,
            "input_cost_usd": 0.7 * charged,
            "output_cost_usd": 0.3 * charged,
            "total_cost_usd": float(charged),
            "cache_creation_cost_usd": 0.2 * charged,
            "cache_read_cost_usd": 0.1 * charged,
            "reasoning_cost_usd": 0.12 * charged,
            "input_tokens_complete": reported == requests,
            "output_tokens_complete": reported == requests,
            "cache_creation_tokens_complete": reported == requests,
            "cache_read_tokens_complete": reported == requests,
            "reasoning_tokens_complete": reported == requests,
            "input_cost_complete": priced == requests,
            "output_cost_complete": priced == requests,
            "cache_creation_cost_complete": priced == requests,
            "cache_read_cost_complete": priced == requests,
            "reasoning_cost_complete": priced == requests,
            "historical_complete": True,
        }
    )


def _fresh_runtime() -> None:
    for interpreter in _INTERPRETERS:
        interpreter._registry.close()
    _INTERPRETERS.clear()
    cost_tracking._RECORDER_VAR.set(cost_tracking._SessionCostRecorder())


@pytest.mark.filterwarnings("ignore:The class `CodeInterpreterMiddleware` is in beta")
@pytest.mark.parametrize("mode", ["thread", "turn", "call"])
async def test_js_subagent_cost_is_durable(mode) -> None:
    agent = _parent(
        'await task({description:"work", subagentType:"child"})',
        {"child": _child(_message("child"))},
        InMemorySaver(),
        mode=mode,
    )
    await agent.ainvoke({"messages": [HumanMessage("go")]}, _CONFIG)
    snapshot = await agent.aget_state(_CONFIG)
    assert snapshot.values["_session_cost_usd"] == pytest.approx(3.0)
    await _assert_accounting(agent, 3)
    assert snapshot.values["_session_cost_transfers"] == {}
    from deepagents_code.app import _format_cost_breakdown_table

    table = _format_cost_breakdown_table(
        snapshot.values["_session_cost_usd"], snapshot.values["_session_cost_breakdown"]
    )
    assert "Entire-thread estimated breakdown" in table
    assert "cache creation" in table
    assert "cache read" in table
    assert "reasoning" in table
    assert "partial" not in table.lower()
    assert [m.id for m in snapshot.values["messages"] if isinstance(m, AIMessage)] == [
        "parent-call",
        "parent-done",
    ]
    assert [
        m.name for m in snapshot.values["messages"] if isinstance(m, ToolMessage)
    ] == ["js_eval"]
    _fresh_runtime()
    await agent.ainvoke(None, _CONFIG)
    assert await _total(agent) == pytest.approx(3.0)
    await _assert_accounting(agent, 3)


@pytest.mark.parametrize(
    ("parallel", "failed_sibling"), [(False, False), (True, False), (False, True)]
)
async def test_completed_sibling_interrupt_fresh_runtime(
    parallel: bool, failed_sibling: bool
) -> None:
    @tool
    def approval() -> str:
        """Pause for approval."""
        return str(interrupt("approve?"))

    def interrupted(resuming: bool = False) -> CompiledStateGraph:
        messages = [
            _tool_call("approval", message_id="approval-call", call_id="approve"),
            _message("approved"),
        ]
        return _child(
            *(messages[1:] if resuming else messages),
            tools=[approval],
        )

    saver = InMemorySaver()
    code = (
        'await task({description:"done", subagentType:"done"});'
        'await task({description:"pause", subagentType:"pause"})'
    )
    if parallel:
        code = (
            'await Promise.all([task({description:"done", subagentType:"done"}),'
            'task({description:"pause", subagentType:"pause"})])'
        )
    done = _child(_message("done"))
    if failed_sibling:

        @tool
        def fail() -> str:
            """Fail after durable model accounting."""
            msg = "sibling failed"
            raise RuntimeError(msg)

        done = _child(_tool_call("fail", message_id="failed"), tools=[fail])
        code = code.replace(
            'task({description:"done", subagentType:"done"})',
            'task({description:"done", subagentType:"done"}).catch(() => "failed")',
        )
    agent = _parent(code, {"done": done, "pause": interrupted()}, saver)
    result = await agent.ainvoke({"messages": [HumanMessage("go")]}, _CONFIG)
    assert len(result["__interrupt__"]) == 1
    assert await _total(agent) == pytest.approx(1.0)
    await _assert_accounting(agent, 1)
    _fresh_runtime()
    resumed = _parent(
        code,
        {
            "done": _child(),
            "pause": interrupted(True),
        },
        saver,
        resuming=True,
    )
    await resumed.ainvoke(Command(resume="yes"), _CONFIG)
    assert await _total(resumed) == pytest.approx(5.0)
    await _assert_accounting(resumed, 5)
    await resumed.ainvoke(None, _CONFIG)
    assert await _total(resumed) == pytest.approx(5.0)
    await _assert_accounting(resumed, 5)


async def test_nested_dispatch_and_multiple_eval_tools() -> None:
    inner = _parent(
        'await task({description:"leaf", subagentType:"leaf"})',
        {"leaf": _child(_message("leaf"))},
        None,
        nested=True,
    )
    code = 'await task({description:"inner", subagentType:"inner"})'
    first = _call(code)
    second = _call(
        'await task({description:"other", subagentType:"other"})',
        message_id="second-call",
    )
    agent = _parent(
        code,
        {"inner": inner, "other": _child(_message("other"))},
        InMemorySaver(),
        messages=[first, second, _message("done")],
    )
    await agent.ainvoke({"messages": [HumanMessage("go")]}, _CONFIG)
    assert await _total(agent) == pytest.approx(7.0)
    await _assert_accounting(agent, 7)


async def test_completed_cost_survives_eval_failure() -> None:
    code = (
        'await task({description:"done", subagentType:"done"});throw new Error("boom")'
    )
    agent = _parent(code, {"done": _child(_message("done"))}, InMemorySaver())
    result = await agent.ainvoke({"messages": [HumanMessage("go")]}, _CONFIG)
    assert await _total(agent) == pytest.approx(3.0)
    await _assert_accounting(agent, 3)
    assert "boom" in result["messages"][-2].content


async def test_cancel_after_child_checkpoint_then_resume() -> None:
    waiting = asyncio.Event()
    release = asyncio.Event()

    @tool
    async def wait() -> str:
        """Wait until released."""
        waiting.set()
        await release.wait()
        return "released"

    def child(resuming: bool = False) -> CompiledStateGraph:
        messages = [
            _tool_call("wait", message_id="waiting"),
            _message("released"),
        ]
        return _child(
            *(messages[1:] if resuming else messages),
            tools=[wait],
        )

    saver = InMemorySaver()
    code = (
        'await task({description:"done", subagentType:"done"});'
        'await task({description:"wait", subagentType:"wait"})'
    )
    agent = _parent(code, {"done": _child(_message("done")), "wait": child()}, saver)
    invocation = asyncio.create_task(
        agent.ainvoke({"messages": [HumanMessage("go")]}, _CONFIG)
    )
    await asyncio.wait_for(waiting.wait(), timeout=10)
    invocation.cancel()
    with pytest.raises(asyncio.CancelledError):
        await invocation
    assert await _total(agent) == pytest.approx(1.0)
    await _assert_accounting(agent, 1)
    _fresh_runtime()
    release.set()
    resumed = _parent(
        code,
        {
            "done": _child(),
            "wait": child(True),
        },
        saver,
        resuming=True,
    )
    await resumed.ainvoke(None, _CONFIG)
    assert await _total(resumed) == pytest.approx(5.0)
    await _assert_accounting(resumed, 5)


async def test_failed_child_preserves_checkpointed_cost() -> None:
    @tool
    def fail() -> str:
        """Fail after the child's model checkpoint."""
        msg = "child failed"
        raise RuntimeError(msg)

    child = _child(_tool_call("fail", message_id="failed"), tools=[fail])
    agent = _parent(
        'await task({description:"fail", subagentType:"fail"})',
        {"fail": child},
        InMemorySaver(),
    )
    result = await agent.ainvoke({"messages": [HumanMessage("go")]}, _CONFIG)
    assert "child failed" in result["messages"][-2].content
    assert await _total(agent) == pytest.approx(3.0)
    await _assert_accounting(agent, 3)


@pytest.mark.parametrize("durability", ["async", "exit"])
@pytest.mark.parametrize("nested", [False, True])
async def test_dispatch_failure_cancels_sibling_without_losing_durable_cost(
    durability, nested
) -> None:
    waiting = asyncio.Event()

    @tool
    async def wait() -> str:
        """Wait until the parallel dispatch fails."""
        waiting.set()
        await asyncio.Event().wait()
        return "unreachable"

    @tool
    async def fail() -> str:
        """Fail once the sibling has checkpointed its model cost."""
        await waiting.wait()
        msg = "parallel failure"
        raise RuntimeError(msg)

    code = (
        'await Promise.all([task({description:"wait",subagentType:"wait"}),'
        'task({description:"fail",subagentType:"fail"})])'
    )
    waiter = _child(_tool_call("wait"), tools=[wait])
    if nested:
        waiter = _parent(
            'await task({description:"done", subagentType:"done"});'
            'await task({description:"wait", subagentType:"wait"})',
            {"done": _child(_message("done")), "wait": waiter},
            None,
            nested=True,
        )
    agent = _parent(
        code,
        {"wait": waiter, "fail": _child(_tool_call("fail"), tools=[fail])},
        InMemorySaver(),
    )
    result = await agent.ainvoke(
        {"messages": [HumanMessage("go")]}, _CONFIG, durability=durability
    )
    assert "parallel failure" in result["messages"][-2].content
    assert await _total(agent) == pytest.approx(6.0 if nested else 4.0)
    await _assert_accounting(agent, 6 if nested else 4)


async def test_failed_nested_dispatch_preserves_unclaimed_transfer() -> None:
    code = 'await task({description:"leaf", subagentType:"leaf"})'
    inner = _parent(
        code,
        {"leaf": _child(_message("leaf"))},
        None,
        nested=True,
        messages=[_call(code)],
    )
    agent = _parent(
        'await task({description:"inner", subagentType:"inner"})',
        {"inner": inner},
        InMemorySaver(),
    )
    await agent.ainvoke({"messages": [HumanMessage("go")]}, _CONFIG)
    assert await _total(agent) == pytest.approx(4.0)
    await _assert_accounting(agent, 4)


async def test_structured_result_and_unrelated_state_are_isolated() -> None:
    from langchain.agents.middleware import AgentMiddleware, AgentState

    class State(AgentState):
        unrelated: NotRequired[str]

    class ChildState(AgentMiddleware):
        state_schema = State

        def after_agent(self, state, runtime) -> dict[str, str]:  # noqa: ARG002  # Middleware hook signature.
            return {"unrelated": "child-only"}

    interpreter = CostAwareCodeInterpreterMiddleware(tool_name="js_eval", mode="call")
    _INTERPRETERS.append(interpreter)
    child_middleware: list[AgentMiddleware[Any, Any, Any]] = [
        CostTrackingMiddleware(nested=True),
        ChildState(),
    ]
    child: SubAgent = {
        "tools": [],
        "name": "structured",
        "description": "structured",
        "system_prompt": "Return an answer.",
        "model": _fake_model(
            _tool_call(
                "subagent_response",
                {"answer": 42},
                message_id="structured",
                call_id="answer",
            )
        ),
        "middleware": child_middleware,
    }
    code = (
        'const r = await task({description:"answer", subagentType:"structured",'
        'responseSchema:{type:"object",properties:{answer:{type:"integer"}},'
        'required:["answer"]}}); r.answer + 1'
    )
    middleware: list[AgentMiddleware[Any, Any, Any]] = [
        CostTrackingMiddleware(),
        SubAgentMiddleware(
            backend=StateBackend(),
            subagents=[child],
            private_state_keys=frozenset({"_session_cost_usd"}),
        ),
        interpreter,
    ]
    agent = create_agent(
        _fake_model(_call(code), _message("done")),
        middleware=middleware,
        state_schema=State,
        checkpointer=InMemorySaver(),
    )
    result = await agent.ainvoke(
        {"messages": [HumanMessage("go")], "unrelated": "parent-only"}, _CONFIG
    )
    assert "43" in result["messages"][-2].content
    snapshot = await agent.aget_state(_CONFIG)
    assert snapshot.values["unrelated"] == "parent-only"
    assert "structured_response" not in snapshot.values
    assert await _total(agent) == pytest.approx(3.0)
    await _assert_accounting(agent, 3)


async def test_direct_task_still_transfers_costs() -> None:
    message = _tool_call(
        "task",
        {"description": "work", "subagent_type": "child"},
        message_id="direct",
        call_id="direct",
    )
    agent = _parent(
        "",
        {"child": _child(_message("child"))},
        InMemorySaver(),
        messages=[message, _message("done")],
    )
    await agent.ainvoke({"messages": [HumanMessage("go")]}, _CONFIG)
    assert await _total(agent) == pytest.approx(3.0)
    await _assert_accounting(agent, 3)


@pytest.mark.parametrize("durability", ["async", "exit"])
async def test_completion_order_does_not_swap_results_on_sqlite_resume(
    tmp_path, durability, monkeypatch
) -> None:
    dispatched = []
    release = asyncio.Event()

    @tool
    async def delay() -> str:
        """Let D finish before A on the first invocation."""
        await release.wait()
        return "ready"

    @tool
    def approval() -> str:
        """Interrupt after both dependent dispatches finish."""
        return str(interrupt("approve?"))

    from langchain.agents.middleware import AgentMiddleware

    class Dispatch(AgentMiddleware):
        def before_agent(self, state, runtime) -> None:  # noqa: ARG002  # Middleware hook signature.
            name = state["messages"][0].content
            dispatched.append(name)
            if name == "D":
                release.set()

    def children(resuming: bool = False) -> dict[str, CompiledStateGraph]:
        result = {}
        for name in "ABCD":
            messages = (
                []
                if resuming
                else [
                    AIMessage(
                        content="RESULT_" + name, id=name, usage_metadata=_usage()
                    )
                ]
            )
            if name == "A" and not resuming:
                messages.insert(
                    0,
                    _tool_call("delay"),
                )
            middleware: list[AgentMiddleware[Any, Any, Any]] = [
                CostTrackingMiddleware(nested=True),
                Dispatch(),
            ]
            result[name] = create_agent(
                _fake_model(*messages),
                tools=[delay] if name == "A" else [],
                middleware=middleware,
            )
        messages = [_message("approved")]
        if not resuming:
            messages.insert(
                0,
                _tool_call("approval"),
            )
        result["approval"] = _child(*messages, tools=[approval])
        return result

    code = (
        "const answers = await Promise.all(["
        'task({description:"A",subagentType:"A"}).then(()=>task({description:"C",subagentType:"C"})),'
        'task({description:"B",subagentType:"B"}).then(()=>task({description:"D",subagentType:"D"}))]);'
        'await task({description:"approve",subagentType:"approval"}); answers'
    )
    database = str(tmp_path / "checkpoints.sqlite")
    async with AsyncSqliteSaver.from_conn_string(database) as saver:
        agent = _parent(code, children(), saver)
        result = await agent.ainvoke(
            {"messages": [HumanMessage("go")]}, _CONFIG, durability=durability
        )
        assert result["__interrupt__"]
        assert dispatched.index("D") < dispatched.index("C")
    _fresh_runtime()
    from deepagents_code import _js_cost

    original = _js_cost._cost_task
    resumed_dispatches = []
    c_started = asyncio.Event()

    def reordered(tool, runtime, owner, active) -> StructuredTool:
        proxy = original(tool, runtime, owner, active)
        invoke = proxy.coroutine
        assert invoke is not None

        async def dispatch(description, subagent_type, runtime) -> object:
            if description == "B":
                await c_started.wait()
            resumed_dispatches.append(description)
            if description == "C":
                c_started.set()
            return await invoke(description, subagent_type, runtime)

        return proxy.model_copy(update={"coroutine": dispatch})

    monkeypatch.setattr(_js_cost, "_cost_task", reordered)
    async with AsyncSqliteSaver.from_conn_string(database) as saver:
        agent = _parent(code, children(True), saver, resuming=True)
        result = await agent.ainvoke(
            Command(resume="yes"), _CONFIG, durability=durability
        )
        output = result["messages"][-2].content
        assert output.index("RESULT_C") < output.index("RESULT_D")
        assert resumed_dispatches.index("C") < resumed_dispatches.index("D")
        assert await _total(agent) == pytest.approx(9.0)
        await _assert_accounting(agent, 9)
    _fresh_runtime()
    async with AsyncSqliteSaver.from_conn_string(database) as saver:
        agent = _parent(code, children(True), saver, resuming=True)
        await agent.ainvoke(None, _CONFIG, durability=durability)
        assert await _total(agent) == pytest.approx(9.0)
        await _assert_accounting(agent, 9)


@pytest.mark.parametrize("durability", ["async", "exit"])
async def test_identical_parallel_requests_are_distinct(durability) -> None:
    agent = _parent(
        'await Promise.all([task({description:"same",subagentType:"child"}),'
        'task({description:"same",subagentType:"child"})])',
        {"child": _child(_message("one"), _message("two"))},
        InMemorySaver(),
    )
    await agent.ainvoke(
        {"messages": [HumanMessage("go")]}, _CONFIG, durability=durability
    )
    assert await _total(agent) == pytest.approx(4.0)
    await _assert_accounting(agent, 4)


async def test_direct_grandchild_is_not_double_counted() -> None:
    direct = _tool_call(
        "task",
        {"description": "leaf", "subagent_type": "leaf"},
        message_id="direct",
        call_id="leaf",
    )
    inner = _parent(
        "",
        {"leaf": _child(_message("leaf"))},
        None,
        nested=True,
        messages=[direct, _message("inner")],
    )
    agent = _parent(
        'await task({description:"inner",subagentType:"inner"})',
        {"inner": inner},
        InMemorySaver(),
    )
    await agent.ainvoke({"messages": [HumanMessage("go")]}, _CONFIG)
    assert await _total(agent) == pytest.approx(5.0)
    await _assert_accounting(agent, 5)


@pytest.mark.parametrize("durability", ["async", "exit"])
async def test_cancellation_settles_inflight_receipt_before_sqlite_close(
    tmp_path, durability
) -> None:
    writing = asyncio.Event()
    release = asyncio.Event()

    class SlowSaver(AsyncSqliteSaver):
        async def aput(
            self, config, checkpoint, metadata, new_versions
        ) -> RunnableConfig:
            if "js_cost_owner" in metadata:
                writing.set()
                await release.wait()
            return await super().aput(config, checkpoint, metadata, new_versions)

    database = str(tmp_path / "cancel.sqlite")
    code = 'await task({description:"child",subagentType:"child"})'
    async with SlowSaver.from_conn_string(database) as saver:
        agent = _parent(code, {"child": _child(_message("child"))}, saver)
        invocation = asyncio.create_task(
            agent.ainvoke(
                {"messages": [HumanMessage("go")]}, _CONFIG, durability=durability
            )
        )
        await asyncio.wait_for(writing.wait(), timeout=10)
        invocation.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await invocation
    _fresh_runtime()
    async with AsyncSqliteSaver.from_conn_string(database) as saver:
        agent = _parent(
            code,
            {"child": _child()},
            saver,
            resuming=True,
        )
        await agent.ainvoke(None, _CONFIG, durability=durability)
        assert await _total(agent) == pytest.approx(3.0)
        await _assert_accounting(agent, 3)


_FREE_ESTIMATE = replace(
    _ESTIMATE,
    total_cost_usd=0.0,
    input_cost_usd=0.0,
    output_cost_usd=0.0,
    cache_creation_cost_usd=0.0,
    cache_read_cost_usd=0.0,
    reasoning_cost_usd=0.0,
)


@pytest.mark.parametrize(
    ("pricing", "priced", "charged"),
    [("free", 4, 0), ("unpriceable", 0, 0), ("mixed", 3, 3), ("mixed_free", 3, 2)],
)
async def test_parallel_usage_with_partial_or_zero_pricing(
    monkeypatch: pytest.MonkeyPatch, pricing: str, priced: int, charged: int
) -> None:
    def estimate(
        _usage: Mapping[str, object] | None, model: str, _provider: str
    ) -> cost_tracking._CostEstimate | None:
        if pricing == "free":
            return _FREE_ESTIMATE
        if pricing == "unpriceable" or model == "unpriceable-child":
            return None
        if pricing == "mixed_free" and model == "free-child":
            return _FREE_ESTIMATE
        return _ESTIMATE

    monkeypatch.setattr(cost_tracking, "_estimate_cost", estimate)
    unpriceable = _message("unpriceable")
    unpriceable.response_metadata = {"model_name": "unpriceable-child"}
    free = _message("free")
    free.response_metadata = {"model_name": "free-child"}
    agent = _parent(
        'await Promise.all([task({description:"one",subagentType:"one"}),'
        'task({description:"two",subagentType:"two"})])',
        {"one": _child(unpriceable), "two": _child(free)},
        InMemorySaver(),
    )
    await agent.ainvoke({"messages": [HumanMessage("go")]}, _CONFIG)
    await _assert_accounting(agent, 4, priced=priced, charged=charged)
    from deepagents_code.app import _format_cost_breakdown_table

    snapshot = await agent.aget_state(_CONFIG)
    table = _format_cost_breakdown_table(
        snapshot.values.get("_session_cost_usd", 0.0),
        snapshot.values["_session_cost_breakdown"],
    )
    assert "Entire-thread estimated breakdown" in table
    assert ("Some requests were unpriceable; costs are partial." in table) == (
        priced < 4
    )
    assert ("(partial)" in table) == (priced < 4)
    _fresh_runtime()
    await agent.ainvoke(None, _CONFIG)
    await _assert_accounting(agent, 4, priced=priced, charged=charged)


@pytest.mark.parametrize("priced", [False, True], ids=["unpriceable", "free"])
async def test_zero_dollar_receipts_survive_sqlite_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, priced: bool
) -> None:
    monkeypatch.setattr(
        cost_tracking,
        "_estimate_cost",
        lambda *_args: _FREE_ESTIMATE if priced else None,
    )

    @tool
    def approval() -> str:
        """Pause after recording zero dollars of usage."""
        return str(interrupt("approve?"))

    code = 'await task({description:"work",subagentType:"child"})'
    database = str(tmp_path / "zero.sqlite")
    async with AsyncSqliteSaver.from_conn_string(database) as saver:
        agent = _parent(
            code, {"child": _child(_tool_call("approval"), tools=[approval])}, saver
        )
        result = await agent.ainvoke({"messages": [HumanMessage("go")]}, _CONFIG)
        assert result["__interrupt__"]
    _fresh_runtime()
    async with AsyncSqliteSaver.from_conn_string(database) as saver:
        agent = _parent(
            code,
            {"child": _child(_message("approved"), tools=[approval])},
            saver,
            resuming=True,
        )
        await agent.ainvoke(Command(resume="yes"), _CONFIG)
        await _assert_accounting(agent, 4, priced=4 if priced else 0, charged=0)
        await agent.ainvoke(None, _CONFIG)
        await _assert_accounting(agent, 4, priced=4 if priced else 0, charged=0)


async def test_missing_child_categories_remain_incomplete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    partial = replace(
        _ESTIMATE,
        cache_creation_tokens=None,
        cache_read_tokens=None,
        reasoning_tokens=None,
        cache_creation_cost_usd=None,
        cache_read_cost_usd=None,
        reasoning_cost_usd=None,
    )
    monkeypatch.setattr(
        cost_tracking,
        "_estimate_cost",
        lambda usage, *_args: _ESTIMATE if "input_token_details" in usage else partial,
    )
    child_message = _message("partial")
    child_message.usage_metadata = {
        "input_tokens": 1000,
        "output_tokens": 100,
        "total_tokens": 1100,
    }
    agent = _parent(
        'await task({description:"partial",subagentType:"child"})',
        {"child": _child(child_message)},
        InMemorySaver(),
    )
    await agent.ainvoke({"messages": [HumanMessage("go")]}, _CONFIG)
    snapshot = await agent.aget_state(_CONFIG)
    assert snapshot.values["_session_cost_usd"] == pytest.approx(3.0)
    breakdown = snapshot.values["_session_cost_breakdown"]
    assert breakdown["historical_complete"] is True
    assert breakdown["request_count"] == breakdown["priced_request_count"] == 3
    assert breakdown["input_tokens"] == 3000
    assert breakdown["output_tokens"] == 300
    assert breakdown["input_cost_usd"] == pytest.approx(2.1)
    assert breakdown["output_cost_usd"] == pytest.approx(0.9)
    for category, tokens, cost in (
        ("cache_creation", 200, 0.4),
        ("cache_read", 800, 0.2),
        ("reasoning", 80, 0.24),
    ):
        assert breakdown[f"{category}_tokens"] == tokens
        assert breakdown[f"{category}_cost_usd"] == pytest.approx(cost)
        assert breakdown[f"{category}_tokens_complete"] is False
        assert breakdown[f"{category}_cost_complete"] is False


async def test_missing_child_usage_remains_incomplete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cost_tracking,
        "_estimate_cost",
        lambda usage, *_args: _ESTIMATE if usage else None,
    )
    agent = _parent(
        'await task({description:"unknown usage",subagentType:"child"})',
        {"child": _child(AIMessage(content="done", id="unknown"))},
        InMemorySaver(),
    )
    await agent.ainvoke({"messages": [HumanMessage("go")]}, _CONFIG)
    await _assert_accounting(agent, 3, priced=2, reported=2)


async def test_legacy_dollar_only_receipt_survives_sqlite_resume(
    tmp_path: Path,
) -> None:
    class LegacySaver(AsyncSqliteSaver):
        async def aput(
            self,
            config: RunnableConfig,
            checkpoint: Checkpoint,
            metadata: CheckpointMetadata,
            new_versions: ChannelVersions,
        ) -> RunnableConfig:
            if "js_cost_owner" in metadata:
                # Emulate the persisted format before breakdown receipts existed.
                checkpoint["channel_values"].pop("breakdown", None)
                checkpoint["channel_versions"].pop("breakdown", None)
            return await super().aput(config, checkpoint, metadata, new_versions)

    @tool
    def approval() -> str:
        """Pause with a dollar-only receipt on disk."""
        return str(interrupt("approve?"))

    code = 'await task({description:"work",subagentType:"child"})'
    database = str(tmp_path / "legacy.sqlite")
    async with LegacySaver.from_conn_string(database) as saver:
        agent = _parent(
            code, {"child": _child(_tool_call("approval"), tools=[approval])}, saver
        )
        result = await agent.ainvoke({"messages": [HumanMessage("go")]}, _CONFIG)
        assert result["__interrupt__"]
    _fresh_runtime()
    async with AsyncSqliteSaver.from_conn_string(database) as saver:
        agent = _parent(
            code,
            {"child": _child(_message("approved"), tools=[approval])},
            saver,
            resuming=True,
        )
        await agent.ainvoke(Command(resume="yes"), _CONFIG)
        snapshot = await agent.aget_state(_CONFIG)
        assert snapshot.values["_session_cost_usd"] == pytest.approx(4.0)
        breakdown = snapshot.values["_session_cost_breakdown"]
        assert breakdown["historical_complete"] is False
        assert breakdown["total_cost_usd"] == pytest.approx(4.0)
        assert breakdown["request_count"] == breakdown["priced_request_count"] == 3
        assert breakdown["input_tokens"] == 3000
        assert breakdown["output_tokens"] == 300
        assert breakdown["input_cost_usd"] == pytest.approx(2.1)
        assert breakdown["output_cost_usd"] == pytest.approx(0.9)
        assert breakdown["cache_creation_tokens"] == 300
        assert breakdown["cache_read_tokens"] == 1200
        assert breakdown["reasoning_tokens"] == 120
        assert breakdown["cache_creation_cost_usd"] == pytest.approx(0.6)
        assert breakdown["cache_read_cost_usd"] == pytest.approx(0.3)
        assert breakdown["reasoning_cost_usd"] == pytest.approx(0.36)
        from deepagents_code.app import _format_cost_breakdown_table

        assert _format_cost_breakdown_table(4.0, breakdown) == ""
        await agent.ainvoke(None, _CONFIG)
        replayed = await agent.aget_state(_CONFIG)
        assert replayed.values["_session_cost_usd"] == pytest.approx(4.0)
        assert replayed.values["_session_cost_breakdown"] == breakdown
