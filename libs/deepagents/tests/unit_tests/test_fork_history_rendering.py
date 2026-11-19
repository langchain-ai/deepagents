"""Tests for model-visible fork history under context management."""

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.summarization import SummarizationMiddleware
from tests.unit_tests.chat_model import GenericFakeChatModel


def _models() -> tuple[GenericFakeChatModel, GenericFakeChatModel]:
    parent = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "task", "args": {"description": "Inspect", "subagent_type": "worker"}, "id": "task"}],
                ),
                AIMessage(content="Parent done"),
            ]
        )
    )
    child = GenericFakeChatModel(messages=iter([AIMessage(content="Child done")]))
    return parent, child


@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
async def test_fork_task_moves_argument_out_of_truncation_window(*, use_async: bool) -> None:
    parent, child = _models()
    agent = create_deep_agent(
        model=parent,
        subagents=[{"name": "worker", "description": "Continue the conversation", "mode": "fork", "model": child}],
    )
    value = "x" * 2200
    history = [
        HumanMessage(content="Prepare files"),
        AIMessage(content="", tool_calls=[{"name": "write_file", "args": {"file_path": "/old", "content": value}, "id": "old"}]),
        ToolMessage(content="Written", tool_call_id="old"),
        AIMessage(content="", tool_calls=[{"name": "write_file", "args": {"file_path": "/recent", "content": value}, "id": "recent"}]),
        ToolMessage(content="Written", tool_call_id="recent"),
        *[HumanMessage(content=f"Question {i}") if i % 2 == 0 else AIMessage(content=f"Answer {i}") for i in range(18)],
    ]
    if use_async:
        await agent.ainvoke({"messages": history})
    else:
        agent.invoke({"messages": history})

    def arguments(model: GenericFakeChatModel) -> dict[str, str]:
        return {
            call["id"]: call["args"]["content"]
            for message in model.call_history[0]["messages"]
            if isinstance(message, AIMessage)
            for call in message.tool_calls
            if call["name"] == "write_file"
        }

    truncated = "x" * 20 + "...(argument truncated)"
    assert arguments(parent) == {"old": truncated, "recent": value}
    assert arguments(child) == {"old": truncated, "recent": truncated}


@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
async def test_fork_task_can_trigger_compaction(tmp_path: Path, *, use_async: bool) -> None:
    parent, child = _models()
    backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
    middleware = SummarizationMiddleware(
        model=parent,
        backend=backend,
        trigger=("messages", 6),
        keep=("messages", 2),
        summarizer=lambda _messages: "Condensed history",
    )
    agent = create_deep_agent(
        model=parent,
        backend=backend,
        middleware=[middleware],
        subagents=[{"name": "worker", "description": "Continue the conversation", "mode": "fork", "model": child}],
    )
    history = [
        HumanMessage(content="First question"),
        AIMessage(content="First answer"),
        HumanMessage(content="Second question"),
        AIMessage(content="Second answer"),
        HumanMessage(content="Third question"),
    ]
    if use_async:
        await agent.ainvoke({"messages": history})
    else:
        agent.invoke({"messages": history})

    parent_messages = parent.call_history[0]["messages"]
    child_messages = child.call_history[0]["messages"]
    assert any(message.content == "First question" for message in parent_messages)
    assert not any(message.content == "First question" for message in child_messages)
    assert any("Condensed history" in message.text for message in child_messages)
    assert any("Inspect" in message.text for message in child_messages)
