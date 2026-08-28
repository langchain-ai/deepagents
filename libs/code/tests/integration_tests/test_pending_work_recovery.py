"""Integration coverage for abandoning stale pending graph work."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, TypedDict, cast
from unittest.mock import AsyncMock

import pytest

if TYPE_CHECKING:
    from pathlib import Path

    from langchain_core.runnables import RunnableConfig

from deepagents_code.client.remote_client import RemoteAgent


class _State(TypedDict):
    messages: list[Any]


@pytest.mark.timeout(120)
async def test_abandon_pending_tool_work_never_executes_tool(
    tmp_path: Path,
) -> None:
    from langchain_core.messages import AIMessage
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph

    calls: list[str] = []

    def model(state: _State) -> dict[str, object]:
        del state
        return {}

    def tools(state: _State) -> dict[str, object]:
        del state
        calls.append("executed")
        (tmp_path / "side-effect").write_text("ran")
        return {}

    builder = StateGraph(cast("Any", _State))
    builder.add_node("model", model)
    builder.add_node("tools", tools)
    builder.add_edge(START, "model")
    builder.add_edge("model", "tools")
    builder.add_edge("tools", END)
    graph = builder.compile(checkpointer=InMemorySaver())
    config: RunnableConfig = {"configurable": {"thread_id": "pending-recovery"}}
    await graph.aupdate_state(
        config,
        {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "shell", "args": {}, "id": "call-1"}],
                )
            ]
        },
        as_node="model",
    )
    before = await graph.aget_state(config)
    assert before.next == ("tools",)

    agent = RemoteAgent(url="http://localhost:8123", graph_name="agent")
    local_graph = cast("Any", graph)
    local_graph._validate_client = lambda: SimpleNamespace(
        runs=SimpleNamespace(list=AsyncMock(return_value=[]))
    )
    agent._graph = local_graph
    await agent.aabandon_pending_work(config)

    after = await graph.aget_state(config)
    assert after.next == ()
    assert after.tasks == ()
    assert calls == []
    assert not (tmp_path / "side-effect").exists()
    cancelled = after.values["messages"][-1]
    assert cancelled.tool_call_id == "call-1"
    assert cancelled.status == "error"
