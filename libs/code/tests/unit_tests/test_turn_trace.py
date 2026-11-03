"""Exercise retrospective turn updates through real LangGraph tracing."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langchain_core.tracers.langchain import LangChainTracer
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_config
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from langsmith import Client, tracing_context
from pydantic import BaseModel

from deepagents_code import turn_trace
from deepagents_code.turn_trace import capture_turn

if TYPE_CHECKING:
    from collections.abc import Iterator

    from langchain.agents.middleware.types import ModelResponse
    from langchain_core.runnables import RunnableConfig
    from langsmith.run_trees import RunTree


class _State(BaseModel):
    value: int


@pytest.fixture
def traced_graph(monkeypatch):
    monkeypatch.setattr(turn_trace, "_turns", turn_trace.OrderedDict())
    client = MagicMock(spec=Client)
    updates = []
    client.update_run.side_effect = lambda *args, **kwargs: updates.append(
        (args, deepcopy(kwargs))
    )
    tracer = LangChainTracer(client=client)
    captured = []
    model = FakeListChatModel(responses=["done"])

    @tool
    def echo(value: str) -> str:
        """Echo the provided value."""
        return value

    def capture(state: _State) -> dict[str, int]:
        captured.append(capture_turn(get_config()))
        return {"value": state.value}

    def work(state: _State) -> dict[str, int]:
        RunnableLambda(lambda _: model.invoke("hello")).invoke(None)
        echo.invoke({"value": "hello"})
        return {"value": state.value}

    def pause(state: _State) -> dict[str, int]:
        capture(state)
        interrupt("approve")
        return {"value": state.value}

    builder = StateGraph(_State)
    for name, node in [("capture", capture), ("work", work), ("pause", pause)]:
        builder.add_node(name, node)
    builder.add_edge(START, "capture")
    builder.add_edge("capture", "work")
    builder.add_edge("work", "pause")
    builder.add_edge("pause", END)
    return builder.compile(checkpointer=InMemorySaver()), tracer, captured, updates


def _config(tracer: LangChainTracer, turn: str) -> RunnableConfig:
    return {
        "callbacks": [tracer],
        "configurable": {"thread_id": "thread"},
        "metadata": {
            "thread_id": "thread",
            "turn_id": turn,
            "turn_number": 1,
            "unrelated": "keep",
            "git_repo": "legacy",
            "ls_git_repo": "stale",
        },
    }


def _runs(roots) -> Iterator[RunTree]:
    pending = list(roots)
    while pending:
        run = pending.pop()
        pending.extend(run.child_runs)
        yield run


def test_open_root_metadata_removal_preserves_children_and_completed_roots(
    traced_graph,
):
    graph, tracer, captured, updates = traced_graph
    graph.invoke({"value": 1}, _config(tracer, "turn-one"))
    trace = captured[-1]
    assert trace is not None
    completed = next(iter(trace.roots.values()))
    before = {run.id: deepcopy(run.metadata) for run in _runs([completed])}
    live = completed.create_child(name="next-invocation", run_type="chain")
    child = live.create_child(name="model", run_type="chain")
    trace.roots[str(live.id)] = live
    updates.clear()
    trace.update(
        {"open_swe_git_targets": [{"repo": "work", "branch": "main"}]},
        remove=("ls_git_repo",),
    )
    assert "ls_git_repo" not in live.metadata
    assert live.metadata["unrelated"] == "keep"
    assert live.metadata["open_swe_git_targets"] == [{"repo": "work", "branch": "main"}]
    assert child.metadata["ls_git_repo"] == "stale"
    assert all(
        run.metadata == before[run.id] for run in _runs([completed]) if run.id in before
    )
    assert live.end_time is None
    assert not updates
    graph.invoke({"value": 2}, _config(tracer, "turn-two"))
    assert captured[-1] is not trace


def test_disabled_and_unscoped_traces_are_not_captured(traced_graph):
    graph, tracer, captured, _ = traced_graph
    with tracing_context(enabled=False):
        graph.invoke({"value": 1}, _config(tracer, "turn-one"))
    assert captured
    assert all(trace is None for trace in captured)
    captured.clear()
    config = _config(tracer, "turn-two")
    del config["metadata"]["turn_id"]
    graph.invoke({"value": 2}, config)
    assert captured
    assert all(trace is None for trace in captured)


@pytest.mark.parametrize("ask_user", [False, True])
async def test_final_model_payloads_survive_git_reconciliation(monkeypatch, ask_user):
    import json
    from queue import PriorityQueue

    from langchain.agents import create_agent
    from langchain.agents.middleware import wrap_model_call
    from langchain_core.language_models.fake_chat_models import (
        FakeMessagesListChatModel,
    )
    from langchain_core.messages import AIMessage
    from langsmith._internal._operations import (
        SerializedRunOperation,
        combine_serialized_queue_operations,
    )

    from deepagents_code._fake_models import _ToolBindingFakeModel
    from deepagents_code.git_trace import GitTraceMiddleware

    monkeypatch.setattr(turn_trace, "_turns", turn_trace.OrderedDict())
    client = Client(
        api_url="https://example.invalid", api_key="test", auto_batch_tracing=False
    )
    # Keep the real serialized batch queue deterministic without a network worker.
    client.tracing_queue = PriorityQueue()
    tracer = LangChainTracer(client=client)

    @tool("ask_user")
    def question() -> str:
        """Ask for clarification before answering."""
        return interrupt("clarify")

    @wrap_model_call
    async def model_wrapper(request, handler) -> ModelResponse:
        return await handler(request)

    answer = AIMessage(
        content="done",
        usage_metadata={
            "input_tokens": 240000,
            "output_tokens": 10,
            "total_tokens": 240010,
        },
    )
    messages = [answer]
    if ask_user:
        messages.insert(
            0,
            AIMessage(
                content="",
                tool_calls=[{"name": "ask_user", "args": {}, "id": "question"}],
            ),
        )

    class Model(FakeMessagesListChatModel):
        bind_tools = _ToolBindingFakeModel.bind_tools

    middleware = GitTraceMiddleware(MagicMock(), "/unused")
    agent = create_agent(
        Model(responses=messages),
        tools=[question],
        middleware=[middleware, model_wrapper],
        checkpointer=InMemorySaver(),
    )
    try:
        config = _config(tracer, "turn-one")
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "hello"}]}, config
        )
        if ask_user:
            assert result["__interrupt__"]
            result = await agent.ainvoke(Command(resume="continue"), config)
        assert result["messages"][-1].content == "done"
        queued = []
        while not client.tracing_queue.empty():
            queued.append(client.tracing_queue.get_nowait().item)
            client.tracing_queue.task_done()
        operations = combine_serialized_queue_operations(queued)
        runs = []
        for operation in operations:
            assert isinstance(operation, SerializedRunOperation)
            body = operation.deserialize_run_info()
            if operation.outputs is not None:
                body["outputs"] = json.loads(operation.outputs)
            runs.append(body)
        roots = [run for run in runs if not run.get("parent_run_id")]
        assert len(roots) == (2 if ask_user else 1)
        assert "ls_git_repo" not in roots[-1]["extra"]["metadata"]
        if ask_user:
            assert roots[0]["extra"]["metadata"]["ls_git_repo"] == "stale"
        assert all(run.get("end_time") for run in runs)
        model_runs = [run for run in runs if run["name"] == "model"]
        wrappers = [run for run in runs if "model_wrapper" in run["name"]]
        llms = [run for run in runs if run["run_type"] == "llm"]
        assert model_runs
        assert wrappers
        assert llms
        assert all(run.get("outputs") for run in [*model_runs, *wrappers, *llms])
        assert (
            llms[-1]["outputs"]["generations"][0][0]["message"]["kwargs"][
                "usage_metadata"
            ]
            == answer.usage_metadata
        )
        children = {run["id"] for run in [*model_runs, *wrappers, *llms]}
        assert all(
            operation.outputs is not None
            for operation in queued
            if str(operation.id) in children and operation.operation == "patch"
        )
    finally:
        client.close(timeout=0)
