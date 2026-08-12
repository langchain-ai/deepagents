"""Tests for dcode-specific ACP approval context."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

from acp.schema import TextContentBlock
from langchain_core.messages import HumanMessage
from langgraph.store.memory import InMemoryStore


async def test_auto_graph_adds_trusted_prompt_context() -> None:
    class Graph:
        checkpointer = object()

        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        async def astream(
            self, value: dict[str, object], **kwargs: object
        ) -> AsyncIterator[object]:
            self.calls.append({"value": value, **kwargs})
            return
            yield

        async def aget_state(self, _config: dict[str, object]) -> object:
            return SimpleNamespace(next=(), interrupts=[])

    graph = Graph()
    store = InMemoryStore()
    from deepagents_code.acp import _AutoGraph, _prompt

    wrapped = _AutoGraph(graph, store)  # type: ignore[arg-type]
    token = _prompt.set(("session-1", "Update parser.py"))
    try:
        chunks = [
            chunk
            async for chunk in wrapped.astream(
                {"messages": [{"role": "user", "content": "expanded"}]},
                config={"configurable": {"thread_id": "session-1"}},
            )
        ]
    finally:
        _prompt.reset(token)

    assert not chunks
    call = graph.calls[0]
    context = call["context"]
    assert context.approval_mode == "auto"
    assert context.thread_id == "session-1"
    item = store.get(("deepagents_code", "approval_mode"), context.approval_mode_key)
    assert item.value == {"mode": "auto"}
    message = call["value"]["messages"][0]
    assert isinstance(message, HumanMessage)
    metadata = message.additional_kwargs["deepagents_code_user_prompt"]
    assert metadata["literal_user_text"] == "Update parser.py"
    assert metadata["turn_id"] == context.turn_id
