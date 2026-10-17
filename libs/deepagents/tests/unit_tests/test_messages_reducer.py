"""Tests for DeltaChannel message ID stability.

IDs are assigned by LangGraph's ensure_message_ids() before writes are
serialised to the checkpoint. These tests verify the end-to-end property:
get_state() always returns messages with stable, non-None IDs — both within
a single invocation and across resumed threads.
"""

from __future__ import annotations

from typing import Annotated

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.channels.delta import DeltaChannel
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from deepagents._messages_reducer import _messages_delta_reducer


def _build_graph(checkpointer: object) -> object:
    State = TypedDict(  # noqa: UP013
        "State",
        {"messages": Annotated[list, DeltaChannel(_messages_delta_reducer, snapshot_frequency=50)]},
    )  # type: ignore[call-overload]

    turn = [0]

    def agent(_state: dict) -> dict:  # type: ignore[type-arg]
        turn[0] += 1
        return {"messages": [AIMessage(content=f"reply-{turn[0]}", id=f"ai-{turn[0]}")]}

    return StateGraph(State).add_node("agent", agent).add_edge(START, "agent").add_edge("agent", END).compile(checkpointer=checkpointer)


def test_get_state_messages_have_ids() -> None:
    """Every message returned by get_state() must have a non-None id."""
    saver = InMemorySaver()
    graph = _build_graph(saver)
    config = {"configurable": {"thread_id": "has-ids"}}

    graph.invoke({"messages": [HumanMessage(content="hello")]}, config)

    state = graph.get_state(config)
    for msg in state.values["messages"]:
        assert msg.id is not None, f"{type(msg).__name__} has id=None after get_state()"


def test_dict_style_invoke_messages_have_stable_ids() -> None:
    """Dict-style input (API / over-the-wire format) must also yield stable IDs.

    When the graph is invoked with {"role": "user", "content": "..."} dicts
    instead of BaseMessage objects, ensure_message_ids() coerces them before
    serialisation so the checkpoint never stores id=None.
    """
    saver = InMemorySaver()
    graph = _build_graph(saver)
    config = {"configurable": {"thread_id": "dict-style"}}

    graph.invoke({"messages": [{"role": "user", "content": "hello"}]}, config)

    state = graph.get_state(config)
    human = next(m for m in state.values["messages"] if isinstance(m, HumanMessage))
    assert human.id is not None, "dict-style HumanMessage should have a stable ID"

    # ID must be stable across repeated get_state() calls
    ids = [next(m.id for m in graph.get_state(config).values["messages"] if isinstance(m, HumanMessage)) for _ in range(3)]
    assert len(set(ids)) == 1, f"dict-style HumanMessage id unstable: {ids}"


def test_human_message_id_stable_across_invocations_sync() -> None:
    """The same HumanMessage must keep its ID when a thread is resumed."""
    saver = InMemorySaver()
    graph = _build_graph(saver)
    config = {"configurable": {"thread_id": "stability-sync"}}

    graph.invoke({"messages": [HumanMessage(content="write a hello world script")]}, config)
    id_turn1 = next(m.id for m in graph.get_state(config).values["messages"] if isinstance(m, HumanMessage))

    graph.invoke({"messages": [HumanMessage(content="add error handling")]}, config)
    id_turn2 = next(
        m.id for m in graph.get_state(config).values["messages"] if isinstance(m, HumanMessage) and m.content == "write a hello world script"
    )

    assert id_turn1 is not None
    assert id_turn1 == id_turn2, (
        f"HumanMessage ID changed across invocations: turn 1={id_turn1!r}, turn 2={id_turn2!r}. "
        "Checkpoint is storing id=None — see langchain-ai/langgraph#7913."
    )


def test_reducer_handles_none_base_state() -> None:
    """`DeltaChannel.replay_writes` passes `state=None` when the earliest
    checkpoint for a thread did not seed `messages: []`. The reducer must
    treat that the same as an empty base.

    Regression test for https://github.com/langchain-ai/deepagents/issues/3564.
    """  # noqa: D205
    msg = HumanMessage(content="hi", id="h1")
    result = _messages_delta_reducer(None, [[msg]])
    assert result == [msg]

    # Empty writes against a None base should also not crash.
    assert _messages_delta_reducer(None, []) == []
    assert _messages_delta_reducer(None, [[]]) == []


def test_reducer_handles_none_base_state_with_dict_messages() -> None:
    """None-base replay still coerces raw over-the-wire message payloads."""
    result = _messages_delta_reducer(None, [[{"role": "user", "content": "hi"}]])

    assert len(result) == 1
    assert isinstance(result[0], HumanMessage)
    assert result[0].content == "hi"


@pytest.mark.anyio
async def test_human_message_id_stable_across_invocations_async() -> None:
    """Same check via ainvoke (AsyncPregelLoop path)."""
    saver = InMemorySaver()
    graph = _build_graph(saver)
    config = {"configurable": {"thread_id": "stability-async"}}

    await graph.ainvoke({"messages": [HumanMessage(content="write a hello world script")]}, config)
    id_turn1 = next(m.id for m in (await graph.aget_state(config)).values["messages"] if isinstance(m, HumanMessage))

    await graph.ainvoke({"messages": [HumanMessage(content="add error handling")]}, config)
    id_turn2 = next(
        m.id for m in (await graph.aget_state(config)).values["messages"] if isinstance(m, HumanMessage) and m.content == "write a hello world script"
    )

    assert id_turn1 is not None
    assert id_turn1 == id_turn2, f"Async: HumanMessage ID changed across invocations: turn 1={id_turn1!r}, turn 2={id_turn2!r}"
