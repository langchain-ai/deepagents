"""Checkpointed JavaScript fan-out resumes logical subagents without restarting work."""

from __future__ import annotations

import asyncio
from collections import Counter
from typing import Any

from deepagents import create_deep_agent
from deepagents.middleware.subagents import CompiledSubAgent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command, interrupt

from langchain_quickjs import CodeInterpreterMiddleware
from tests._common import FakeChatModel


def _child(
    name: str,
    pauses: int,
    work: Counter[str],
    approved: dict[str, list[str]],
    predecessor: asyncio.Event | None,
) -> CompiledSubAgent:
    async def perform_work(state: MessagesState) -> dict[str, Any]:
        work[name] += 1
        return {"messages": [AIMessage(content=f"{name} result")]}

    async def post_work_hook(state: MessagesState) -> dict[str, Any]:
        if predecessor is not None:
            await predecessor.wait()
        decisions = [interrupt({"child": name, "hook": i}) for i in range(pauses)]
        approved[name] = decisions
        return {}

    graph = StateGraph(MessagesState)
    graph.add_node("work", perform_work)
    graph.add_node("post_work_hook", post_work_hook)
    graph.add_edge(START, "work")
    graph.add_edge("work", "post_work_hook")
    graph.add_edge("post_work_hook", END)
    return CompiledSubAgent(
        name=name,
        description=f"Run {name} work and its completion hooks.",
        runnable=graph.compile(),
    )


async def test_fanout_keeps_three_ids_across_checkpoint_resumes() -> None:
    """Completed work survives sibling hooks and repeated parent-node reexecution."""
    work: Counter[str] = Counter()
    approved: dict[str, list[str]] = {}
    finished = {name: asyncio.Event() for name in ("fast", "review", "cost")}
    children = [
        _child("fast", 0, work, approved, None),
        _child("review", 1, work, approved, finished["fast"]),
        _child("cost", 3, work, approved, finished["review"]),
    ]
    middleware = CodeInterpreterMiddleware(tool_name="js_eval")
    code = (
        "await Promise.all(['fast', 'review', 'cost'].map(subagentType => "
        "task({description: 'Do the assigned work', label: 'Worker', "
        "subagentType})))"
    )
    model = FakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "js_eval",
                            "args": {"code": code},
                            "id": "fanout_eval",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="All work returned."),
            ]
        )
    )
    agent = create_deep_agent(
        model=model,
        middleware=[middleware],
        subagents=children,
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "subagent-replay"}}
    events: list[dict[str, Any]] = []
    names: dict[str, str] = {}
    rounds: list[list[dict[str, Any]]] = []
    payload: dict[str, Any] | Command = {
        "messages": [HumanMessage(content="Run the three workers.")]
    }
    try:
        for _ in range(6):
            attempt: list[dict[str, Any]] = []
            async for mode, chunk in agent.astream(
                payload, config, stream_mode=["custom", "updates"]
            ):
                if mode != "custom" or chunk.get("type") != "subagent":
                    continue
                events.append(chunk)
                attempt.append(chunk)
                if chunk["phase"] == "start":
                    names[chunk["id"]] = chunk["subagent_type"]
                elif chunk["phase"] == "complete":
                    finished[names[chunk["id"]]].set()
            rounds.append(attempt)
            snapshot = await agent.aget_state(config)
            if not snapshot.interrupts:
                break
            payload = Command(
                resume={item.id: "approved" for item in snapshot.interrupts}
            )
        else:
            msg = "Subagent hooks did not finish after five resumes"
            raise AssertionError(msg)
    finally:
        middleware._registry.close()

    assert len(rounds) == 5
    assert work == {"fast": 1, "review": 1, "cost": 1}
    assert approved == {"fast": [], "review": ["approved"], "cost": ["approved"] * 3}
    assert len(names) == 3
    assert {event["eval_id"] for event in events} == {"fanout_eval"}
    assert {
        names[event["id"]] for event in rounds[0] if event["phase"] == "complete"
    } == {"fast"}
    for attempt in rounds:
        assert {
            (event["id"], event["subagent_type"])
            for event in attempt
            if event["phase"] == "start"
        } == set(names.items())
    assert {event["id"] for event in rounds[-1] if event["phase"] == "complete"} == set(
        names
    )
    assert not snapshot.next
    assert snapshot.values["messages"][-1].content == "All work returned."
    results = [
        message
        for message in snapshot.values["messages"]
        if isinstance(message, ToolMessage) and message.tool_call_id == "fanout_eval"
    ]
    assert len(results) == 1
    assert "<error" not in results[0].content
    for name in finished:
        assert f"{name} result" in results[0].content
