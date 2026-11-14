"""Fork requests as serialized by real provider models, with stubbed HTTP responses."""

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from deepagents.backends.filesystem import FilesystemBackend
from deepagents.graph import create_deep_agent
from tests.unit_tests.middleware.skill_tools_support import call, invoke, stub_anthropic, stub_openai


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("instructions", [None, "FORK_INSTRUCTIONS"], ids=["plain", "instructions"])
@pytest.mark.parametrize("blocks", [False, True], ids=["string-system", "block-system"])
@pytest.mark.parametrize("memory", [False, True], ids=["no-memory", "memory"])
@pytest.mark.parametrize("ending", ["user", "tool"])
def test_anthropic_fork_payload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: str, instructions: str | None, ending: str, *, blocks: bool, memory: bool
) -> None:
    model, stub = stub_anthropic(
        monkeypatch,
        [[call("task", "call_1", description="FORK_TASK", subagent_type="worker")], "worker done", "parent done"],
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
    agent = create_deep_agent(
        model=model,
        system_prompt=system,
        backend=FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True),
        memory=["/AGENTS.md"] if memory else None,
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
    assert parent["system"] == fork["system"]
    assert parent["cache_control"] == fork["cache_control"]
    assert parent["system"][-1]["cache_control"]["type"] == "ephemeral"
    if blocks:
        assert parent["system"][0]["cache_control"] == {"type": "ephemeral"}
    if memory:
        assert "MEMORY_CONTENT" in parent["system"][-1]["text"]
    assert fork["messages"][:-1] == parent["messages"][:-1]
    assert parent["messages"][-1]["role"] == fork["messages"][-1]["role"] == "user"
    prior, task = fork["messages"][-1]["content"]
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
    assert task["text"].endswith(f"{instructions}\n\nFORK_TASK" if instructions else "FORK_TASK")
    assert ("FORK_INSTRUCTIONS" in task["text"]) == (instructions is not None)
    assert "FORK_INSTRUCTIONS" not in str(fork["system"])


@pytest.mark.parametrize("instructions", [None, "FORK_INSTRUCTIONS"], ids=["plain", "instructions"])
@pytest.mark.parametrize("ending", ["user", "tool"])
def test_openai_fork_payload(instructions: str | None, ending: str) -> None:
    model, stub = stub_openai([[call("task", "call_1", description="FORK_TASK", subagent_type="worker")], "worker done", "parent done"])
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
    assert fork["input"][:-1] == parent["input"]
    if ending == "tool":
        assert parent["input"][-1] == {"type": "function_call_output", "output": "PREVIOUS_TOOL_RESULT", "call_id": "previous_call"}
    assert fork["input"][-1]["role"] == "user"
    assert fork["input"][-1]["content"].endswith("FORK_TASK")
    assert ("FORK_INSTRUCTIONS" in fork["input"][-1]["content"]) == (instructions is not None)


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
