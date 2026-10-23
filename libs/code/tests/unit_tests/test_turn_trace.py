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
from langsmith import Client, agent_address, tracing_context
from pydantic import BaseModel

from deepagents_code import turn_trace
from deepagents_code.turn_trace import capture_turn

if TYPE_CHECKING:
    from collections.abc import Iterator

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


def _config(tracer: LangChainTracer, turn: str) -> dict:
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


def test_backfills_root_llm_tool_and_nested_runs_across_hitl(traced_graph):
    graph, tracer, captured, updates = traced_graph
    config = _config(tracer, "turn-one")
    graph.invoke({"value": 1}, config)
    first = captured[-1]
    assert first is not None
    graph.invoke(Command(resume=True), config)
    assert captured[-1] is first
    assert len(first.roots) == 2
    runs = list(_runs(first.roots.values()))
    assert {run.run_type for run in runs} == {"chain", "llm", "tool"}
    assert any(run.name == "RunnableLambda" for run in runs)
    before = {run.id: (run.end_time, run.outputs, run.error) for run in runs}
    updates.clear()
    first.update({"ls_git_repo": "work-repo", "ls_git_branch": "work-branch"})
    assert len(updates) == len(runs)
    for _, patch in updates:
        identity = patch["run_id"]
        assert patch["end_time"] == before[identity][0]
        assert patch["inputs"] is None
        assert patch["outputs"] == before[identity][1] or not before[identity][1]
        assert patch["error"] == before[identity][2]
        metadata = patch["extra"]["metadata"]
        assert metadata["unrelated"] == "keep"
        assert metadata["git_repo"] == "legacy"
        assert metadata["turn_id"] == "turn-one"
        assert metadata["ls_git_repo"] == "work-repo"
        assert metadata["ls_git_branch"] == "work-branch"
    assert all((run.end_time, run.outputs, run.error) == before[run.id] for run in runs)
    graph.invoke({"value": 2}, _config(tracer, "turn-two"))
    second = captured[-1]
    assert second is not None
    assert second is not first
    second.update({}, remove=("ls_git_repo", "ls_git_branch"))
    assert all(
        "ls_git_repo" not in run.metadata for run in _runs(second.roots.values())
    )
    assert all(run.metadata["ls_git_repo"] == "work-repo" for run in runs)


def test_active_runs_are_not_ended_and_telemetry_failure_is_silent(traced_graph):
    graph, tracer, captured, updates = traced_graph
    graph.invoke({"value": 1}, _config(tracer, "turn-one"))
    trace = captured[-1]
    assert trace is not None
    root = next(iter(trace.roots.values()))
    root.end_time = None
    updates.clear()
    trace.update({"ls_git_repo": "work"})
    assert root.end_time is None
    assert all(patch["run_id"] != root.id for _, patch in updates)
    tracer.client.update_run.side_effect = RuntimeError("telemetry unavailable")
    trace.update({"ls_git_repo": "still-work"})
    assert all(run.metadata["ls_git_repo"] == "still-work" for run in _runs([root]))


def test_replica_backfill_preserves_completed_payload_and_timing(traced_graph):
    graph, tracer, captured, updates = traced_graph
    with tracing_context(
        replicas=[
            {"project_name": "primary", "primary": True},
            {"project_name": "mirror"},
        ]
    ):
        graph.invoke({"value": 1}, _config(tracer, "turn-one"))
    trace = captured[-1]
    assert trace is not None
    runs = list(_runs(trace.roots.values()))
    updates.clear()
    trace.update({"ls_git_repo": "work"})
    assert len(updates) == 2 * len(runs)
    assert {patch["session_name"] for _, patch in updates} == {"primary", "mirror"}
    assert sorted(patch["end_time"] for _, patch in updates) == sorted(
        [run.end_time for run in runs] * 2
    )
    for _, patch in updates:
        assert patch["inputs"] is None
        assert patch["extra"]["metadata"]["ls_git_repo"] == "work"


@pytest.mark.filterwarnings("ignore:Addressing runs with.*is in beta")
@pytest.mark.parametrize("destination", ["project", "address"])
def test_real_sdk_payload_preserves_destination_and_timing(monkeypatch, destination):
    from langsmith.run_trees import RunTree

    transport = MagicMock()
    monkeypatch.setattr(Client, "_update_run", transport)
    client = Client(
        api_url="https://example.invalid", api_key="test", auto_batch_tracing=False
    )
    try:
        completed = RunTree(
            name="completed",
            run_type="chain",
            ls_client=client,
            inputs={"excluded": "input"},
            project_name="source" if destination == "project" else None,
            address=agent_address("dcode", "production")
            if destination == "address"
            else None,
        )
        completed.end(outputs={"unchanged": True})
        original_end = completed.end_time
        assert original_end is not None
        live = RunTree(name="live", run_type="chain", ls_client=client)
        turn_trace.TurnTrace(roots={"completed": completed, "live": live}).update(
            {"ls_git_repo": "example/repo"}
        )
        transport.assert_called_once()
        body = transport.call_args.args[0]
        assert body["end_time"] == original_end.isoformat()
        assert live.end_time is None
        assert "inputs" not in body
        assert body["outputs"] == {"unchanged": True}
        assert completed.outputs == {"unchanged": True}
        assert body["extra"]["metadata"]["ls_git_repo"] == "example/repo"
        if destination == "project":
            assert body["session_name"] == "source"
        else:
            assert body["address"] == "lrn:agents/dcode/environments/production"
    finally:
        client.close()


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
