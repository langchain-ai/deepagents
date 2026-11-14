"""Fork requests as serialized by real provider models, with stubbed HTTP responses."""

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from deepagents.backends.filesystem import FilesystemBackend
from deepagents.graph import create_deep_agent
from deepagents.middleware._utils import _FORK_INSTRUCTIONS_KEY
from deepagents.middleware.summarization import SummarizationMiddleware
from tests.unit_tests.middleware.skill_tools_support import call, invoke, stub_anthropic, stub_openai, write_skill


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("instructions", [None, "FORK_INSTRUCTIONS"], ids=["plain", "instructions"])
@pytest.mark.parametrize("blocks", [False, True], ids=["string-system", "block-system"])
@pytest.mark.parametrize("memory", [False, True], ids=["no-memory", "memory"])
@pytest.mark.parametrize("ending", ["user", "tool"])
@pytest.mark.parametrize("supported", [False, True], ids=["leading-only", "mid-conversation"])
def test_anthropic_fork_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode: str,
    instructions: str | None,
    ending: str,
    *,
    blocks: bool,
    memory: bool,
    supported: bool,
) -> None:
    model, stub = stub_anthropic(
        monkeypatch,
        [[call("task", "call_1", description="FORK_TASK", subagent_type="worker")], "worker done", "parent done"],
        profile={"mid_conversation_system_messages": supported},
    )
    system = (
        SystemMessage(
            content=[{"type": "text", "text": "PARENT_BASE", "cache_control": {"type": "ephemeral"}}, {"type": "text", "text": "PARENT_TAIL"}]
        )
        if blocks
        else "PARENT_BASE"
    )
    worker = {"name": "worker", "description": "Continues with context.", "mode": "fork"}
    if instructions is not None:
        worker["system_prompt"] = instructions
    (tmp_path / "AGENTS.md").write_text("MEMORY_CONTENT")
    write_skill(tmp_path, "example")
    agent = create_deep_agent(
        model=model,
        system_prompt=system,
        backend=FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True),
        memory=["/AGENTS.md"] if memory else None,
        skills=["/skills/"] if memory else None,
        subagents=[worker],
    )
    if ending == "user":
        history = [HumanMessage(content="EARLIER"), AIMessage(content="ANSWER"), HumanMessage(content="PARENT_REQUEST")]
    else:
        history = [
            HumanMessage(content="EARLIER"),
            AIMessage(content="", tool_calls=[call("read_file", "previous_call", file_path="/note")]),
            ToolMessage(
                content=[{"type": "text", "text": "PREVIOUS_TOOL_RESULT", "cache_control": {"type": "ephemeral"}}],
                tool_call_id="previous_call",
            ),
        ]
    invoke(agent, {"messages": history}, mode, {"recursion_limit": 25})

    parent, fork = stub.bodies[:2]
    assert parent["tools"] == fork["tools"]
    assert parent["cache_control"] == fork["cache_control"]
    assert parent["system"][-1]["cache_control"]["type"] == "ephemeral"
    if blocks:
        assert parent["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert ("MEMORY_CONTENT" in str(parent["system"])) == memory
    assert ("example" in str(parent["system"])) == memory
    if instructions and not supported:
        if blocks:
            assert fork["system"][:2] == system.content
            assert fork["system"][2]["text"] == "\n\nFORK_INSTRUCTIONS"
            if memory:
                assert fork["system"][3:] == parent["system"][2:]
        else:
            assert fork["system"][0]["text"] == "PARENT_BASE\n\nFORK_INSTRUCTIONS"
            assert fork["system"][1:] == parent["system"][1:]
    else:
        assert parent["system"] == fork["system"]
    messages = fork["messages"]
    if instructions and supported:
        assert messages[-1] == {"role": "system", "content": "FORK_INSTRUCTIONS"}
        messages = messages[:-1]
    assert messages[:-1] == parent["messages"][:-1]
    assert parent["messages"][-1]["role"] == messages[-1]["role"] == "user"
    prior, task = messages[-1]["content"]
    if ending == "user":
        assert parent["messages"][-1]["content"] == "PARENT_REQUEST"
        assert prior == {"type": "text", "text": "PARENT_REQUEST"}
    else:
        assert parent["messages"][-1]["content"] == [prior]
        assert prior == {
            "type": "tool_result",
            "content": [{"type": "text", "text": "PREVIOUS_TOOL_RESULT"}],
            "tool_use_id": "previous_call",
            "is_error": False,
            "cache_control": {"type": "ephemeral"},
        }
    assert task["type"] == "text"
    assert task["text"].endswith("FORK_TASK")
    assert "FORK_INSTRUCTIONS" not in str(messages)
    assert ("FORK_INSTRUCTIONS" in str(fork["system"])) == bool(instructions and not supported)


@pytest.mark.parametrize("instructions", [None, "FORK_INSTRUCTIONS"], ids=["plain", "instructions"])
@pytest.mark.parametrize("ending", ["user", "tool"])
@pytest.mark.parametrize("supported", [False, True], ids=["leading-only", "mid-conversation"])
def test_openai_fork_payload(instructions: str | None, ending: str, *, supported: bool) -> None:
    model, stub = stub_openai(
        [[call("task", "call_1", description="FORK_TASK", subagent_type="worker")], "worker done", "parent done"],
        profile={"mid_conversation_system_messages": supported},
    )
    worker = {"name": "worker", "description": "Continues with context.", "mode": "fork"}
    if instructions is not None:
        worker["system_prompt"] = instructions
    agent = create_deep_agent(model=model, system_prompt="PARENT_BASE", subagents=[worker])
    if ending == "user":
        history = [HumanMessage(content="PARENT_REQUEST")]
    else:
        history = [
            HumanMessage(content="EARLIER"),
            AIMessage(content="", tool_calls=[call("read_file", "previous_call", file_path="/note")]),
            ToolMessage(content="PREVIOUS_TOOL_RESULT", tool_call_id="previous_call"),
        ]
    agent.invoke({"messages": history}, {"recursion_limit": 25})

    parent, fork = stub.bodies[:2]
    assert parent["tools"] == fork["tools"]
    messages = fork["input"]
    if instructions and supported:
        assert messages[-1] == {"type": "message", "role": "system", "content": "FORK_INSTRUCTIONS"}
        messages = messages[:-1]
    if instructions and not supported:
        assert messages[0] == {**parent["input"][0], "content": "PARENT_BASE\n\nFORK_INSTRUCTIONS"}
        assert messages[1:-1] == parent["input"][1:]
    else:
        assert messages[:-1] == parent["input"]
    if ending == "tool":
        assert parent["input"][-1] == {"type": "function_call_output", "output": "PREVIOUS_TOOL_RESULT", "call_id": "previous_call"}
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"].endswith("FORK_TASK")
    assert "FORK_INSTRUCTIONS" not in messages[-1]["content"]


@pytest.mark.parametrize("profile", [None, {}, {"mid_conversation_system_messages": False}, {"mid_conversation_system_messages": True}])
def test_fork_uses_its_own_model_profile(monkeypatch: pytest.MonkeyPatch, profile: dict[str, bool] | None) -> None:
    model, stub = stub_anthropic(
        monkeypatch,
        [[call("task", "call_1", description="FORK_TASK", subagent_type="worker")], "worker done", "parent done"],
        profile={"mid_conversation_system_messages": not bool((profile or {}).get("mid_conversation_system_messages"))},
    )
    worker_model = model.model_copy(update={"profile": profile})
    agent = create_deep_agent(
        model=model,
        system_prompt="PARENT_BASE",
        subagents=[{"name": "worker", "description": "Continues.", "mode": "fork", "model": worker_model, "system_prompt": "FORK_INSTRUCTIONS"}],
    )
    agent.invoke({"messages": [HumanMessage(content="PARENT_REQUEST")]})
    parent, fork = stub.bodies[:2]
    if (profile or {}).get("mid_conversation_system_messages"):
        assert fork["system"] == parent["system"]
        assert fork["messages"][-1] == {"role": "system", "content": "FORK_INSTRUCTIONS"}
    else:
        assert fork["system"][0]["text"] == "PARENT_BASE\n\nFORK_INSTRUCTIONS"
        assert fork["messages"][-1]["role"] == "user"


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("keep", [1, 3, 4, 5, 100], ids=["evict-addendum", "keep-addendum", "keep-task", "keep-parent-tail", "no-compaction"])
def test_anthropic_fork_addendum_survives_compaction(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: str, keep: int) -> None:
    model, stub = stub_anthropic(
        monkeypatch,
        [
            [call("task", "call_1", description="FORK_TASK", subagent_type="worker")],
            [call("ls", "fork_1", path="/")],
            [call("ls", "fork_2", path="/")],
            "worker done",
            "parent done",
        ],
        profile={"mid_conversation_system_messages": True},
    )
    summary_model, summary_stub = stub_openai(["SUMMARY", "SUMMARY"])
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
    summarization = SummarizationMiddleware(model=summary_model, backend=backend, trigger=("messages", 6), keep=("messages", keep))
    agent = create_deep_agent(
        model=model,
        system_prompt="PARENT_BASE",
        backend=backend,
        subagents=[
            {"name": "worker", "description": "Continues.", "mode": "fork", "system_prompt": "FORK_INSTRUCTIONS", "middleware": [summarization]}
        ],
    )
    invoke(agent, {"messages": [HumanMessage(content="EARLIER"), AIMessage(content="ANSWER"), HumanMessage(content="PARENT_REQUEST")]}, mode)

    parent, *forks, resumed_parent = stub.bodies
    assert len(forks) == 3
    assert bool(summary_stub.bodies) == (keep < 100)
    for fork in forks:
        assert fork["system"] == parent["system"]
        assert fork["tools"] == parent["tools"]
        messages = fork["messages"]
        systems = [i for i, message in enumerate(messages) if message["role"] == "system"]
        assert len(systems) == 1
        index = systems[0]
        assert messages[index]["content"] == "FORK_INSTRUCTIONS"
        assert messages[index - 1]["role"] == "user"
        assert index == len(messages) - 1 or messages[index + 1]["role"] == "assistant"
    assert "FORK_INSTRUCTIONS" not in str(resumed_parent)


@pytest.mark.parametrize("cutoff", [1, 2], ids=["retained", "evicted"])
@pytest.mark.parametrize("empty_assistant", ["", [{"type": "refusal", "refusal": "filtered"}], [{"type": "thinking", "thinking": "old reasoning"}]])
def test_compacted_fork_serializes_after_filtered_assistant(
    monkeypatch: pytest.MonkeyPatch, empty_assistant: str | list[dict[str, str]], cutoff: int
) -> None:
    model, stub = stub_anthropic(monkeypatch, ["done"], profile={"mid_conversation_system_messages": True})
    instructions = SystemMessage(content="FORK_INSTRUCTIONS", additional_kwargs={_FORK_INSTRUCTIONS_KEY: True})
    raw = [HumanMessage(content="task"), instructions, AIMessage(content=empty_assistant), HumanMessage(content="continue")]
    event = {"cutoff_index": cutoff, "created_at_index": len(raw), "summary_message": HumanMessage(content="SUMMARY"), "file_path": None}
    effective = SummarizationMiddleware._apply_event_to_messages(raw, event)
    model.invoke([SystemMessage(content="PARENT_BASE"), *effective])
    messages = stub.bodies[0]["messages"]
    assert [m["role"] for m in messages] == ["user", "user", "system"]
    assert messages[-2]["content"] == "continue"
    assert messages[-1]["content"] == "FORK_INSTRUCTIONS"


def test_compacted_fork_keeps_new_thinking_after_restored_system(monkeypatch: pytest.MonkeyPatch) -> None:
    model, stub = stub_anthropic(monkeypatch, ["done", "done"], profile={"mid_conversation_system_messages": True})
    instructions = SystemMessage(content="FORK_INSTRUCTIONS", additional_kwargs={_FORK_INSTRUCTIONS_KEY: True})
    raw = [HumanMessage(content="task"), instructions, AIMessage(content="old answer"), HumanMessage(content="continue")]
    event = {"cutoff_index": 1, "created_at_index": len(raw), "summary_message": HumanMessage(content="SUMMARY"), "file_path": None}
    effective = SummarizationMiddleware._apply_event_to_messages(raw, event)
    model.invoke([SystemMessage(content="PARENT_BASE"), *effective])
    raw.extend(
        [
            AIMessage(
                content=[{"type": "thinking", "thinking": "new reasoning", "signature": "sig"}],
                tool_calls=[call("ls", "ls_1", path="/")],
            ),
            ToolMessage(content="listed", tool_call_id="ls_1"),
        ]
    )
    model.invoke([SystemMessage(content="PARENT_BASE"), *SummarizationMiddleware._apply_event_to_messages(raw, event)])
    first, second = [body["messages"] for body in stub.bodies]
    assert second[: len(first)] == first
    assert second[-2]["role"] == "assistant"
    assert {"type": "thinking", "thinking": "new reasoning", "signature": "sig"} in second[-2]["content"]
    assert second[-1]["role"] == "user"


def test_anthropic_system_addendum_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    model, stub = stub_anthropic(monkeypatch, ["done", "done", "done", "done"])
    cached = {"type": "ephemeral", "ttl": "5m"}
    prompts = [
        "PARENT_BASE",
        "PARENT_BASE\n\nFORK_INSTRUCTIONS",
        SystemMessage(content=[{"type": "text", "text": "PARENT_BASE"}, {"type": "text", "text": "\n\nFORK_INSTRUCTIONS"}]),
        SystemMessage(content=[{"type": "text", "text": "PARENT_BASE", "cache_control": cached}, {"type": "text", "text": "\n\nFORK_INSTRUCTIONS"}]),
    ]
    for prompt in prompts:
        agent = create_deep_agent(model=model, system_prompt=prompt)
        agent.invoke({"messages": [HumanMessage(content="PARENT_REQUEST")]})

    parent, joined, split, marked = [body["system"] for body in stub.bodies]
    assert parent == [{"type": "text", "text": "PARENT_BASE", "cache_control": cached}]
    assert joined == [{"type": "text", "text": "PARENT_BASE\n\nFORK_INSTRUCTIONS", "cache_control": cached}]
    assert split[0] == {"type": "text", "text": "PARENT_BASE"}
    assert split[1]["cache_control"] == cached
    assert marked[0] == parent[0]
    assert marked[1]["text"] == "\n\nFORK_INSTRUCTIONS"
    assert marked[1]["cache_control"] == cached
