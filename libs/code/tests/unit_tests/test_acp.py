"""Tests for dcode-specific ACP approval context."""

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, cast

from langchain_core.messages import HumanMessage
from langgraph.store.memory import InMemoryStore

if TYPE_CHECKING:
    from langgraph.pregel import Pregel

    from deepagents_code._cli_context import CLIContextSchema


async def test_auto_graph_adds_trusted_prompt_context() -> None:
    class Graph:
        checkpointer = object()

        async def astream(
            self, value: dict[str, object], **kwargs: object
        ) -> AsyncIterator[object]:
            self.call = {"value": value, **kwargs}
            return
            yield

    graph = Graph()
    store = InMemoryStore()
    from deepagents_code.acp import _AutoGraph, _prompt

    token = _prompt.set("Update parser.py")
    try:
        chunks = [
            chunk
            async for chunk in _AutoGraph(
                cast("Pregel[Any, Any, Any, Any]", graph), store
            ).astream(
                {"messages": [{"role": "user", "content": "expanded"}]},
                config={"configurable": {"thread_id": "session-1"}},
            )
        ]
    finally:
        _prompt.reset(token)

    assert not chunks
    context = cast("CLIContextSchema", graph.call["context"])
    assert context.approval_mode == "auto"
    assert context.thread_id == "session-1"
    assert context.approval_mode_key
    item = store.get(("deepagents_code", "approval_mode"), context.approval_mode_key)
    assert item
    assert item.value == {"mode": "auto"}
    message = cast("dict[str, Any]", graph.call["value"])["messages"][0]
    assert isinstance(message, HumanMessage)
    metadata = message.additional_kwargs["deepagents_code_user_prompt"]
    assert metadata["literal_user_text"] == "Update parser.py"
    assert metadata["turn_id"] == context.turn_id
