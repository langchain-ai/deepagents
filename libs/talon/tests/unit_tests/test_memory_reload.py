from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from deepagents_talon.interfaces import AgentRequest
from deepagents_talon.memory import TalonMemoryMiddleware
from deepagents_talon.runtime import DeepAgentRuntime
from tests.unit_tests.test_skill_reload import PromptRecorder, _patch_model

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_memory_reloads_all_sources_on_checkpointed_turns(
    tmp_path: Path, *, asynchronous: bool
) -> None:
    sources = [tmp_path / "first.md", tmp_path / "second.md"]
    for index, path in enumerate(sources):
        path.write_text(f"original preference {index}")
    model = PromptRecorder(responses=[AIMessage(content="ok")])
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

    for index in range(len(sources)):
        assert f"original preference {index}" in model.prompts[0]
        for prompt in model.prompts[1:]:
            assert f"updated preference {index}" in prompt
            assert f"original preference {index}" not in prompt
    assert all(prompt.count("</agent_memory>") == 1 for prompt in model.prompts)
    assert model.prompts[1] == model.prompts[2] == model.prompts[3]


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
    model = PromptRecorder(
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

    assert "current destination" in model.prompts[-1]
    assert "retired destination" not in model.prompts[-1]
    assert model.prompts[-1].count("</agent_memory>") == 1


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("saved", [False, True])
@pytest.mark.parametrize("tool_name", ["edit_file", "write_file"])
async def test_correction_gets_one_extra_opportunity_to_save(
    tmp_path: Path, tool_name: str, *, asynchronous: bool, saved: bool
) -> None:
    source = tmp_path / "memory.md"
    args = {"file_path": str(source)}
    if tool_name == "edit_file":
        source.write_text("old preference")
        args.update(old_string="old preference", new_string="Research fully before answering.")
    else:
        args["content"] = "Research fully before answering."
    write = AIMessage(
        content="",
        tool_calls=[{"name": tool_name, "args": args, "id": "save-correction"}],
    )
    acknowledgement = AIMessage(content="You're right, I should have researched fully.")
    responses = (
        [write, acknowledgement] if saved else [acknowledgement, write, AIMessage(content="saved")]
    )
    model = PromptRecorder(responses=responses)
    backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=False)
    graph = create_deep_agent(
        model=model,
        backend=backend,
        middleware=[TalonMemoryMiddleware(backend=backend, sources=[str(source)])],
    )
    payload = {"messages": [{"role": "user", "content": "You stopped too early."}]}
    result = await graph.ainvoke(payload) if asynchronous else graph.invoke(payload)

    assert source.read_text() == "Research fully before answering."
    assert len(model.prompts) == (2 if saved else 3)
    assert result["messages"][-1].content == (acknowledgement.content if saved else "saved")


@pytest.mark.parametrize(
    ("feedback", "reply"),
    [
        ("You stopped too early.", "I will research more."),
        ("Please investigate further.", "I should have checked the evidence."),
        ("Please investigate further.", "You\u2019re right."),
    ],
)
async def test_correction_reminder_is_bounded_and_resets_each_turn(
    tmp_path: Path, feedback: str, reply: str
) -> None:
    model = PromptRecorder(
        responses=[AIMessage(content=reply) for _ in range(4)] + [AIMessage(content="hello")]
    )
    backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=False)
    graph = create_deep_agent(
        model=model,
        backend=backend,
        middleware=[TalonMemoryMiddleware(backend=backend, sources=[str(tmp_path / "memory.md")])],
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "chat"}}
    for turn in range(2):
        result = await graph.ainvoke({"messages": [{"role": "user", "content": feedback}]}, config)
        assert result["messages"][-1].content == reply
        assert len(model.prompts) == 2 * (turn + 1)
    await graph.ainvoke({"messages": [{"role": "user", "content": "hello"}]}, config)
    assert len(model.prompts) == 5


@pytest.mark.parametrize(
    ("target", "old_string"),
    [
        ("memory.md", "missing text"),
        ("unrelated.md", "old preference"),
    ],
)
async def test_failed_or_unrelated_write_does_not_suppress_reminder(
    tmp_path: Path, target: str, old_string: str
) -> None:
    for name in ["memory.md", "unrelated.md"]:
        (tmp_path / name).write_text("old preference")
    model = PromptRecorder(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "edit_file",
                        "args": {
                            "file_path": str(tmp_path / target),
                            "old_string": old_string,
                            "new_string": "new preference",
                        },
                        "id": "edit",
                    }
                ],
            ),
            AIMessage(content="You're right."),
            AIMessage(content="finished"),
        ]
    )
    backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=False)
    graph = create_deep_agent(
        model=model,
        backend=backend,
        middleware=[TalonMemoryMiddleware(backend=backend, sources=[str(tmp_path / "memory.md")])],
    )
    result = await graph.ainvoke(
        {"messages": [{"role": "user", "content": "You stopped too early."}]}
    )
    assert len(model.prompts) == 3
    assert result["messages"][-1].content == "finished"


async def test_previous_turn_memory_write_does_not_suppress_correction(tmp_path: Path) -> None:
    source = tmp_path / "memory.md"
    model = PromptRecorder(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "write_file",
                        "args": {"file_path": str(source), "content": "preference"},
                        "id": "previous-save",
                    }
                ],
            ),
            AIMessage(content="saved"),
            AIMessage(content="I should have investigated."),
            AIMessage(content="finished"),
            AIMessage(content="hello"),
        ]
    )
    backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=False)
    graph = create_deep_agent(
        model=model,
        backend=backend,
        middleware=[TalonMemoryMiddleware(backend=backend, sources=[str(source)])],
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "chat"}}
    await graph.ainvoke({"messages": [{"role": "user", "content": "save my preference"}]}, config)
    assert source.read_text() == "preference"
    result = await graph.ainvoke(
        {"messages": [{"role": "user", "content": "You stopped too early."}]}, config
    )
    assert result["messages"][-1].content == "finished"
    assert len(model.prompts) == 4
    await graph.ainvoke({"messages": [{"role": "user", "content": "hello"}]}, config)
    assert len(model.prompts) == 5


@pytest.mark.parametrize("sources", [[], ["memory.md"]])
async def test_ordinary_turn_does_not_get_reminder(tmp_path: Path, sources: list[str]) -> None:
    model = PromptRecorder(responses=[AIMessage(content="hello")])
    backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=False)
    graph = create_deep_agent(
        model=model,
        backend=backend,
        middleware=[TalonMemoryMiddleware(backend=backend, sources=sources)],
    )
    await graph.ainvoke({"messages": [{"role": "user", "content": "hello"}]})
    assert len(model.prompts) == 1


async def test_correction_without_memory_sources_does_not_get_reminder(tmp_path: Path) -> None:
    model = PromptRecorder(responses=[AIMessage(content="You're right.")])
    backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=False)
    graph = create_deep_agent(
        model=model,
        backend=backend,
        middleware=[TalonMemoryMiddleware(backend=backend, sources=[])],
    )
    result = await graph.ainvoke(
        {"messages": [{"role": "user", "content": "You stopped too early."}]}
    )
    assert result["messages"][-1].content == "You're right."
    assert len(model.prompts) == 1
