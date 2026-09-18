"""Checkpointed cost ownership for JavaScript subagent dispatch."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Literal, NotRequired

if TYPE_CHECKING:
    from collections.abc import Iterator

    from deepagents.middleware.subagents import SubAgent
    from langchain.agents.middleware import AgentMiddleware
    from langchain_core.messages.ai import UsageMetadata
    from langchain_core.runnables import RunnableConfig
    from langgraph.graph.state import CompiledStateGraph

import pytest
from deepagents.backends import StateBackend
from deepagents.middleware import SubAgentMiddleware
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt

from deepagents_code import cost_tracking
from deepagents_code._fake_models import _ToolBindingFakeModel
from deepagents_code._js_cost import CostAwareCodeInterpreterMiddleware
from deepagents_code.cost_tracking import CostTrackingMiddleware

_INTERPRETERS: list[CostAwareCodeInterpreterMiddleware] = []
_CONFIG: RunnableConfig = {"configurable": {"thread_id": "js-cost"}}


def _usage() -> UsageMetadata:
    return {"input_tokens": 1000, "output_tokens": 100, "total_tokens": 1100}


def _message(message_id: str) -> AIMessage:
    return AIMessage(content="done", id=message_id, usage_metadata=_usage())


def _fake_model(*messages: AIMessage) -> _ToolBindingFakeModel:
    return _ToolBindingFakeModel(messages=iter(messages), disable_streaming=True)


@pytest.fixture(autouse=True)
def controlled_cost(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(cost_tracking, "estimate_cost", lambda *_args: 1.0)
    monkeypatch.setattr(cost_tracking, "pricing_data_available", lambda: True)
    token = cost_tracking._RECORDER_VAR.set(cost_tracking._SessionCostRecorder())
    yield
    cost_tracking._RECORDER_VAR.reset(token)
    for interpreter in _INTERPRETERS:
        interpreter._registry.close()
    _INTERPRETERS.clear()


def _call(code: str, *, message_id: str = "parent-call") -> AIMessage:
    return AIMessage(
        content="",
        id=message_id,
        usage_metadata=_usage(),
        tool_calls=[{"name": "js_eval", "args": {"code": code}, "id": message_id}],
    )


def _child(name: str) -> CompiledStateGraph:
    return create_agent(
        _fake_model(_message(name)),
        middleware=[CostTrackingMiddleware(nested=True)],
    )


def _parent(
    code: str,
    children: dict[str, Any],
    saver: InMemorySaver | None,
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
        {"child": _child("child")},
        InMemorySaver(),
        mode=mode,
    )
    await agent.ainvoke({"messages": [HumanMessage("go")]}, _CONFIG)
    snapshot = await agent.aget_state(_CONFIG)
    assert snapshot.values["_session_cost_usd"] == pytest.approx(3.0)
    assert snapshot.values["_session_cost_transfers"] == {}
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


@pytest.mark.parametrize(
    "code",
    [
        (
            'await Promise.all([task({description:"a", subagentType:"a"}),'
            'task({description:"b", subagentType:"b"})])'
        ),
        (
            'await task({description:"a", subagentType:"a"});'
            'await task({description:"b", subagentType:"b"})'
        ),
    ],
)
async def test_parallel_and_sequential(code: str) -> None:
    agent = _parent(code, {"a": _child("a"), "b": _child("b")}, InMemorySaver())
    await agent.ainvoke({"messages": [HumanMessage("go")]}, _CONFIG)
    assert await _total(agent) == pytest.approx(4.0)


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
            AIMessage(
                content="",
                id="approval-call",
                usage_metadata=_usage(),
                tool_calls=[{"name": "approval", "args": {}, "id": "approve"}],
            ),
            _message("approved"),
        ]
        return create_agent(
            _fake_model(*(messages[1:] if resuming else messages)),
            tools=[approval],
            middleware=[CostTrackingMiddleware(nested=True)],
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
    done = _child("done")
    if failed_sibling:

        @tool
        def fail() -> str:
            """Fail after durable model accounting."""
            msg = "sibling failed"
            raise RuntimeError(msg)

        done = create_agent(
            _fake_model(
                AIMessage(
                    content="",
                    id="failed",
                    usage_metadata=_usage(),
                    tool_calls=[{"name": "fail", "args": {}, "id": "fail"}],
                )
            ),
            tools=[fail],
            middleware=[CostTrackingMiddleware(nested=True)],
        )
        code = code.replace(
            'task({description:"done", subagentType:"done"})',
            'task({description:"done", subagentType:"done"}).catch(() => "failed")',
        )
    agent = _parent(code, {"done": done, "pause": interrupted()}, saver)
    result = await agent.ainvoke({"messages": [HumanMessage("go")]}, _CONFIG)
    assert len(result["__interrupt__"]) == 1
    assert await _total(agent) == pytest.approx(1.0)
    _fresh_runtime()
    resumed = _parent(
        code,
        {
            "done": create_agent(
                _fake_model(), middleware=[CostTrackingMiddleware(nested=True)]
            ),
            "pause": interrupted(True),
        },
        saver,
        resuming=True,
    )
    await resumed.ainvoke(Command(resume="yes"), _CONFIG)
    assert await _total(resumed) == pytest.approx(5.0)
    await resumed.ainvoke(None, _CONFIG)
    assert await _total(resumed) == pytest.approx(5.0)


async def test_nested_dispatch_and_multiple_eval_tools() -> None:
    inner = _parent(
        'await task({description:"leaf", subagentType:"leaf"})',
        {"leaf": _child("leaf")},
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
        {"inner": inner, "other": _child("other")},
        InMemorySaver(),
        messages=[first, second, _message("done")],
    )
    await agent.ainvoke({"messages": [HumanMessage("go")]}, _CONFIG)
    assert await _total(agent) == pytest.approx(7.0)


async def test_completed_cost_survives_eval_failure() -> None:
    code = (
        'await task({description:"done", subagentType:"done"});throw new Error("boom")'
    )
    agent = _parent(code, {"done": _child("done")}, InMemorySaver())
    result = await agent.ainvoke({"messages": [HumanMessage("go")]}, _CONFIG)
    assert await _total(agent) == pytest.approx(3.0)
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
            AIMessage(
                content="",
                id="waiting",
                usage_metadata=_usage(),
                tool_calls=[{"name": "wait", "args": {}, "id": "wait"}],
            ),
            _message("released"),
        ]
        return create_agent(
            _fake_model(*(messages[1:] if resuming else messages)),
            tools=[wait],
            middleware=[CostTrackingMiddleware(nested=True)],
        )

    saver = InMemorySaver()
    code = (
        'await task({description:"done", subagentType:"done"});'
        'await task({description:"wait", subagentType:"wait"})'
    )
    agent = _parent(code, {"done": _child("done"), "wait": child()}, saver)
    invocation = asyncio.create_task(
        agent.ainvoke({"messages": [HumanMessage("go")]}, _CONFIG)
    )
    await asyncio.wait_for(waiting.wait(), timeout=10)
    invocation.cancel()
    with pytest.raises(asyncio.CancelledError):
        await invocation
    assert await _total(agent) == pytest.approx(1.0)
    _fresh_runtime()
    release.set()
    resumed = _parent(
        code,
        {
            "done": create_agent(
                _fake_model(), middleware=[CostTrackingMiddleware(nested=True)]
            ),
            "wait": child(True),
        },
        saver,
        resuming=True,
    )
    await resumed.ainvoke(None, _CONFIG)
    assert await _total(resumed) == pytest.approx(5.0)


async def test_failed_child_preserves_checkpointed_cost() -> None:
    @tool
    def fail() -> str:
        """Fail after the child's model checkpoint."""
        msg = "child failed"
        raise RuntimeError(msg)

    child = create_agent(
        _fake_model(
            AIMessage(
                content="",
                id="failed",
                usage_metadata=_usage(),
                tool_calls=[{"name": "fail", "args": {}, "id": "fail"}],
            )
        ),
        tools=[fail],
        middleware=[CostTrackingMiddleware(nested=True)],
    )
    agent = _parent(
        'await task({description:"fail", subagentType:"fail"})',
        {"fail": child},
        InMemorySaver(),
    )
    result = await agent.ainvoke({"messages": [HumanMessage("go")]}, _CONFIG)
    assert "child failed" in result["messages"][-2].content
    assert await _total(agent) == pytest.approx(3.0)


async def test_dispatch_failure_cancels_sibling_without_losing_durable_cost() -> None:
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

    def child(name: str, operation) -> CompiledStateGraph:
        return create_agent(
            _fake_model(
                AIMessage(
                    content="",
                    id=name,
                    usage_metadata=_usage(),
                    tool_calls=[{"name": name, "args": {}, "id": name}],
                )
            ),
            tools=[operation],
            middleware=[CostTrackingMiddleware(nested=True)],
        )

    code = (
        'await Promise.all([task({description:"wait",subagentType:"wait"}),'
        'task({description:"fail",subagentType:"fail"})])'
    )
    agent = _parent(
        code,
        {"wait": child("wait", wait), "fail": child("fail", fail)},
        InMemorySaver(),
    )
    result = await agent.ainvoke({"messages": [HumanMessage("go")]}, _CONFIG)
    assert "parallel failure" in result["messages"][-2].content
    assert await _total(agent) == pytest.approx(4.0)


async def test_failed_nested_dispatch_preserves_unclaimed_transfer() -> None:
    code = 'await task({description:"leaf", subagentType:"leaf"})'
    inner = _parent(
        code, {"leaf": _child("leaf")}, None, nested=True, messages=[_call(code)]
    )
    agent = _parent(
        'await task({description:"inner", subagentType:"inner"})',
        {"inner": inner},
        InMemorySaver(),
    )
    await agent.ainvoke({"messages": [HumanMessage("go")]}, _CONFIG)
    assert await _total(agent) == pytest.approx(4.0)


async def test_structured_result_and_unrelated_state_are_isolated() -> None:
    from langchain.agents.middleware import AgentMiddleware, AgentState

    class State(AgentState):
        unrelated: NotRequired[str]

    class ChildState(AgentMiddleware):
        state_schema = State

        def after_agent(self, state, runtime) -> dict[str, str]:
            assert state["messages"]
            assert runtime is not None
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
            AIMessage(
                content="",
                id="structured",
                usage_metadata=_usage(),
                tool_calls=[
                    {
                        "name": "subagent_response",
                        "args": {"answer": 42},
                        "id": "answer",
                    }
                ],
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


@pytest.mark.parametrize("amount", [True, -1, float("inf"), float("nan"), "1"])
def test_invalid_costs_are_not_transferred(amount: object) -> None:
    from deepagents_code._js_cost import _cost_transfers

    result = Command(
        update={
            "_session_cost_transfers": {
                "tools:a|dispatch:b": {"owner_scope": "tools:a", "cost_usd": amount}
            }
        }
    )
    assert _cost_transfers(result, "tools:a|dispatch:b", "") == {}


def test_only_exact_child_scope_and_owner_are_accepted() -> None:
    from deepagents_code._js_cost import _cost_transfers

    result = Command(
        update={
            "_session_cost_transfers": {
                "tools:a|dispatch:b": {"owner_scope": "tools:a", "cost_usd": 1.0},
                "tools:a|dispatch:c": {"owner_scope": "tools:a", "cost_usd": 100.0},
            }
        }
    )
    assert _cost_transfers(result, "tools:a|dispatch:b", "root") == {
        "tools:a|dispatch:b": {"owner_scope": "root", "cost_usd": 1.0}
    }
    assert _cost_transfers(result, "tools:wrong|dispatch:b", "root") == {}


async def test_direct_task_still_transfers_costs() -> None:
    message = AIMessage(
        content="",
        id="direct",
        usage_metadata=_usage(),
        tool_calls=[
            {
                "name": "task",
                "args": {"description": "work", "subagent_type": "child"},
                "id": "direct",
            }
        ],
    )
    agent = _parent(
        "",
        {"child": _child("child")},
        InMemorySaver(),
        messages=[message, _message("done")],
    )
    await agent.ainvoke({"messages": [HumanMessage("go")]}, _CONFIG)
    assert await _total(agent) == pytest.approx(3.0)
