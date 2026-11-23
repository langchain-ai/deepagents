from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from deepagents.middleware.summarization import SummarizationMiddleware
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from deepagents_talon.interfaces import AgentRequest
from deepagents_talon.memory import TalonMemoryMiddleware
from deepagents_talon.runtime import DeepAgentRuntime
from tests.unit_tests.test_skill_reload import PromptRecorder, _patch_model

if TYPE_CHECKING:
    from pathlib import Path

    from langchain_core.outputs import ChatResult


class MemoryRecorder(PromptRecorder):
    inputs: list[list[BaseMessage]] = []  # noqa: RUF012  # Pydantic copies instance fields.

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> ChatResult:
        self.inputs.append(list(messages))
        return super()._generate(messages, *args, **kwargs)


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_memory_reloads_all_sources_on_checkpointed_turns(
    tmp_path: Path, *, asynchronous: bool
) -> None:
    sources = [tmp_path / "first.md", tmp_path / "second.md"]
    for index, path in enumerate(sources):
        path.write_text(f"original preference {index}")
    model = MemoryRecorder(responses=[AIMessage(content=f"reply {index}") for index in range(4)])
    backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=False)
    graph = create_deep_agent(
        model=model,
        backend=backend,
        memory=[str(path) for path in sources],
        middleware=[
            TalonMemoryMiddleware(backend=backend, sources=[str(path) for path in sources])
        ],
        checkpointer=InMemorySaver(),
    )
    for turn, thread in enumerate(["chat", "chat", "chat", "fresh"]):
        if turn == 1:
            for index, path in enumerate(sources):
                path.write_text(f"updated preference {index}")
        config = {"configurable": {"thread_id": thread}}
        payload = {"messages": [{"role": "user", "content": "hello"}]}
        if asynchronous:
            await graph.ainvoke(payload, config)
        else:
            graph.invoke(payload, config)

    assert model.prompts[0] == model.prompts[1] == model.prompts[2]
    assert "updated preference 0" in model.prompts[3]
    assert model.inputs[1][: len(model.inputs[0])] == model.inputs[0]
    assert model.inputs[2][:-2] == model.inputs[1]
    for turn in (1, 2):
        updates = [
            message for message in model.inputs[turn][1:] if "<agent_memory>" in message.text
        ]
        assert len(updates) == 1
        update = updates[0]
        assert "file data from disk" in update.text
        for index in range(len(sources)):
            assert f"updated preference {index}" in update.text
        position = model.inputs[turn].index(update)
        assert isinstance(model.inputs[turn][position + 1], HumanMessage)
    assert all("<agent_memory>" not in message.text for message in model.inputs[0][1:])
    assert all("<agent_memory>" not in message.text for message in model.inputs[3][1:])


@pytest.mark.parametrize("tool_name", ["edit_file", "write_file"])
async def test_runtime_sees_file_tool_memory_edits_on_next_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tool_name: str
) -> None:
    source = tmp_path / "memory.md"
    if tool_name == "edit_file":
        source.write_text("retired destination")
        args = {
            "file_path": str(source),
            "old_string": "retired destination",
            "new_string": "current destination",
        }
    else:
        args = {"file_path": str(source), "content": "current destination"}
    model = MemoryRecorder(
        responses=[
            AIMessage(
                content="", tool_calls=[{"name": tool_name, "args": args, "id": "save-memory"}]
            ),
            AIMessage(content="saved"),
            AIMessage(content="next turn"),
        ]
    )
    _patch_model(monkeypatch, model)
    runtime = DeepAgentRuntime(
        model="test:model",
        assistant_dir=tmp_path,
        memory=[str(source)],
        backend=FilesystemBackend(root_dir=tmp_path, virtual_mode=False),
        checkpointer=InMemorySaver(),
        include_web_tools=False,
        env={},
    )
    await runtime.start()
    try:
        await runtime.invoke(AgentRequest("chat", "save the preference"))
        assert source.read_text() == "current destination"
        await runtime.invoke(AgentRequest("chat", "use the preference"))
    finally:
        await runtime.stop()

    assert model.prompts[0] == model.prompts[1] == model.prompts[2]
    updates = [message for message in model.inputs[-1][1:] if "<agent_memory>" in message.text]
    assert len(updates) == 1
    assert "current destination" in updates[0].text
    assert "retired destination" not in updates[0].text
    assert "file data from disk" in updates[0].text


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("compact", [False, True])
async def test_memory_snapshot_refreshes_when_history_is_rebuilt(
    tmp_path: Path, *, asynchronous: bool, compact: bool
) -> None:
    source = tmp_path / "memory.md"
    source.write_text("original memory")
    backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=False)
    model = MemoryRecorder(responses=[AIMessage(content=f"reply {index}") for index in range(4)])
    summarizer = SummarizationMiddleware(
        model=PromptRecorder(responses=[AIMessage(content="summary")]),
        backend=backend,
        trigger=None if compact else ("messages", 5),
        keep=("messages", 2),
    )
    graph = create_deep_agent(
        model=model,
        backend=backend,
        memory=[str(source)],
        middleware=[summarizer, TalonMemoryMiddleware(backend=backend, sources=[str(source)])],
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "chat"}}
    for turn in range(3):
        if turn == 1:
            source.write_text("updated memory")
        if turn == 2 and compact:
            await graph.aupdate_state(
                config,
                {
                    "_summarization_event": {
                        "cutoff_index": 2,
                        "summary_message": HumanMessage(content="compacted history"),
                        "file_path": None,
                        "created_at_index": 4,
                    }
                },
            )
        payload = {"messages": [{"role": "user", "content": f"turn {turn}"}]}
        if asynchronous:
            await graph.ainvoke(payload, config)
        else:
            graph.invoke(payload, config)

    assert model.prompts[0] == model.prompts[1]
    assert "updated memory" in model.inputs[1][-2].text
    assert "updated memory" in model.prompts[2]
    assert "original memory" not in model.prompts[2]
    assert all("<agent_memory>" not in message.text for message in model.inputs[2][1:])
    snapshot = await graph.aget_state(config)
    assert snapshot.values["memory_prompt"]["contents"] == {str(source): "updated memory"}
    assert snapshot.values["memory_prompt"]["updates"] == {}


async def test_unchanged_memory_does_not_add_updates(tmp_path: Path) -> None:
    source = tmp_path / "memory.md"
    source.write_text("unchanged memory")
    backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=False)
    model = MemoryRecorder(responses=[AIMessage(content=f"reply {index}") for index in range(4)])
    graph = create_deep_agent(
        model=model,
        backend=backend,
        memory=[str(source)],
        middleware=[TalonMemoryMiddleware(backend=backend, sources=[str(source)])],
        checkpointer=InMemorySaver(),
    )
    for _ in range(2):
        await graph.ainvoke(
            {"messages": [{"role": "user", "content": "hello"}]},
            {"configurable": {"thread_id": "chat"}},
        )
    assert model.prompts[0] == model.prompts[1]
    assert all("<agent_memory>" not in message.text for message in model.inputs[1][1:])
