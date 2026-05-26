"""Unit tests for _messages_delta_reducer."""

from __future__ import annotations

from typing import Annotated

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.channels.delta import DeltaChannel
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from deepagents._messages_reducer import _messages_delta_reducer


def test_idless_message_gets_id_and_order_preserved() -> None:
    existing = [AIMessage(content="hello", id="existing-1")]
    new_msg = HumanMessage(content="follow-up")  # no id

    result = _messages_delta_reducer(existing, [[new_msg]])

    assert len(result) == 2
    assert result[0].id == "existing-1"
    assert result[1] is new_msg
    assert result[1].id is not None


# ---------------------------------------------------------------------------
# Cross-invocation ID stability
# ---------------------------------------------------------------------------


def _build_graph(checkpointer: object) -> object:
    State = TypedDict(  # noqa: UP013
        "State",
        {"messages": Annotated[list, DeltaChannel(_messages_delta_reducer, snapshot_frequency=50)]},
    )  # type: ignore[call-overload]

    turn = [0]

    def agent(_state: dict) -> dict:  # type: ignore[type-arg]
        turn[0] += 1
        return {"messages": [AIMessage(content=f"reply-{turn[0]}", id=f"ai-{turn[0]}")]}

    return (
        StateGraph(State)
        .add_node("agent", agent)
        .add_edge(START, "agent")
        .add_edge("agent", END)
        .compile(checkpointer=checkpointer)
    )


def test_human_message_id_stable_across_invocations_sync() -> None:
    """The same HumanMessage must keep its ID when a thread is resumed.

    Without the LangGraph fix (ensure_message_ids in put_writes), the
    checkpoint stores id=None and every resumed invocation replays through
    the reducer, assigning a fresh UUID — so the same human input appears
    with a different ID in each LangSmith trace.
    """
    saver = InMemorySaver()
    graph = _build_graph(saver)
    config = {"configurable": {"thread_id": "stability-sync"}}

    graph.invoke({"messages": [HumanMessage(content="write a hello world script")]}, config)
    state1 = graph.get_state(config)
    id_turn1 = next(m.id for m in state1.values["messages"] if isinstance(m, HumanMessage))

    graph.invoke({"messages": [HumanMessage(content="add error handling")]}, config)
    state2 = graph.get_state(config)
    id_turn2 = next(
        m.id
        for m in state2.values["messages"]
        if isinstance(m, HumanMessage) and m.content == "write a hello world script"
    )

    assert id_turn1 is not None, "HumanMessage should have been assigned an ID"
    assert id_turn1 == id_turn2, (
        f"HumanMessage ID changed across invocations on the same thread: "
        f"turn 1={id_turn1!r}, turn 2={id_turn2!r}. "
        "Checkpoint is storing id=None — see langchain-ai/langgraph#7913."
    )


@pytest.mark.anyio
async def test_human_message_id_stable_across_invocations_async() -> None:
    """Same check via ainvoke (AsyncPregelLoop path)."""
    saver = InMemorySaver()
    graph = _build_graph(saver)
    config = {"configurable": {"thread_id": "stability-async"}}

    await graph.ainvoke(
        {"messages": [HumanMessage(content="write a hello world script")]}, config
    )
    state1 = await graph.aget_state(config)
    id_turn1 = next(m.id for m in state1.values["messages"] if isinstance(m, HumanMessage))

    await graph.ainvoke(
        {"messages": [HumanMessage(content="add error handling")]}, config
    )
    state2 = await graph.aget_state(config)
    id_turn2 = next(
        m.id
        for m in state2.values["messages"]
        if isinstance(m, HumanMessage) and m.content == "write a hello world script"
    )

    assert id_turn1 is not None
    assert id_turn1 == id_turn2, (
        f"Async: HumanMessage ID changed across invocations: "
        f"turn 1={id_turn1!r}, turn 2={id_turn2!r}"
    )
