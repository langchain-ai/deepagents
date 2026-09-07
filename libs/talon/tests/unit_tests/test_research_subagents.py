from __future__ import annotations

import asyncio
import json
import shlex

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from pydantic import PrivateAttr

from deepagents_talon.interfaces import AgentRequest
from deepagents_talon.runtime import DeepAgentRuntime
from deepagents_talon.subagents import prepare_subagents


class ToolModel(FakeMessagesListChatModel):
    _seen: list = PrivateAttr(default_factory=list)
    _tools: list = PrivateAttr(default_factory=list)

    def bind_tools(self, tools, **_kwargs: object):
        self._tools.append([item.name for item in tools])
        return self

    def _generate(self, messages, *args: object, **kwargs: object):
        self._seen.append(messages)
        return super()._generate(messages, *args, **kwargs)


def _call(name, **args: object):
    return {"name": name, "id": name, "args": args}


def _write_agent(root, tools="[]", *, name="researcher"):
    path = root / "agents" / name / "AGENTS.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\ndescription: Research\nmodel: test:child\ntools: {tools}\n"
        "---\nAnswer the delegated question."
    )
    return path


def _runtime(root, monkeypatch, parent, child, **kwargs: object):
    monkeypatch.setattr(
        "deepagents_talon.runtime._resolve_model_from_env", lambda *_a, **_k: parent
    )

    def compile_child(**options: object):
        options["model"] = child
        return create_agent(**options)

    monkeypatch.setattr("deepagents_talon.subagents.create_agent", compile_child)
    return DeepAgentRuntime(
        model="test:parent",
        assistant_dir=root,
        include_web_tools=False,
        skills=(),
        **{"memory": (), **kwargs},
    )


@pytest.mark.parametrize("background", [False, True])
@pytest.mark.parametrize(
    ("name", "attached"),
    [
        ("researcher", True),
        ("researcher", False),
        ("general-purpose", False),
        ("general-purpose", True),
    ],
)
async def test_research_boundaries(tmp_path, monkeypatch, background, name, attached):
    _write_agent(tmp_path, "[lookup]" if attached else "[]")
    private = "PRIVATE-PARENT-MARKER"
    memory = tmp_path / "memory.md"
    memory.write_text(private)
    output = tmp_path / "output.txt"
    skill = tmp_path / "skill.md"
    skill.write_text("Use lookup for research.")
    selected = ["lookup", "read_file"] if name == "general-purpose" and attached else ["lookup"]
    launch = {"tools": selected} if name == "general-purpose" and attached else {}
    effects = []

    @tool
    def lookup() -> str:
        """Return research evidence."""
        effects.append("lookup")
        return "Source: fixture; evidence found"

    forbidden = [
        _call(name)
        for name in (
            "execute",
            "search_conversations",
            "reload_subagent_configuration",
            "task",
            "start_async_task",
        )
    ]
    skill_calls = [_call("read_file", file_path=str(skill))] if "read_file" in selected else []
    child = ToolModel(
        responses=[
            AIMessage(content="", tool_calls=[_call("lookup"), *skill_calls, *forbidden]),
            AIMessage(content="Research complete"),
        ]
        if attached
        else [AIMessage(content="No tools")]
    )
    parent = ToolModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    _call("task", subagent_type=name, description="Find evidence", **launch)
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[_call("write_file", file_path=str(output), content="main works")],
            ),
            AIMessage(content="Done"),
        ]
    )
    runtime = _runtime(tmp_path, monkeypatch, parent, child, tools=[lookup], memory=[str(memory)])
    if not background:
        monkeypatch.setattr(runtime.background, "configured", lambda _: AgentMiddleware())
    await runtime.start()
    try:
        await runtime.invoke(AgentRequest("chat", f"Parent history contains {private}"))
        await asyncio.gather(*(job.worker for job in runtime.background._jobs.values()))
        assert output.read_text() == "main works"
        assert memory.read_text() == private
        assert effects == (["lookup"] if attached else [])
        assert child._tools == ([selected, selected] if attached else [])
        assert child._seen[0][-1].content == "Find evidence"
        assert private not in str(child._seen)
        assert "Parent history" not in str(child._seen[0])
        if name == "general-purpose" and attached:
            assert "Use lookup for research." in str(child._seen)
        messages = child._seen[-1]
        denied = [message for message in messages if getattr(message, "status", None) == "error"]
        assert {message.name for message in denied} == (
            {call["name"] for call in forbidden} if attached else set()
        )
        inventory = runtime._graph.nodes["tools"].bound.tools_by_name["get_agent_tools"].invoke({})
        agent = next(item for item in inventory["agents"] if item["name"] == name)
        if name == "general-purpose":
            assert "read_file" in agent["selectable_tools"]
            assert "task" not in agent["selectable_tools"]
        else:
            assert agent["tools"] == (["lookup"] if attached else [])
        assert private not in json.dumps(inventory)
    finally:
        await runtime.stop()


@pytest.mark.parametrize("name", ["researcher", "general-purpose"])
async def test_explicit_shell_access(tmp_path, monkeypatch, name):
    _write_agent(tmp_path, "[execute]")
    output = tmp_path / "child-output"
    launch = {"tools": ["execute"]} if name == "general-purpose" else {}
    parent = ToolModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    _call("task", subagent_type=name, description="Create the file", **launch)
                ],
            ),
            AIMessage(content="Done"),
        ]
    )
    child = ToolModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[_call("execute", command=f"printf done > {shlex.quote(str(output))}")],
            ),
            AIMessage(content="Created"),
        ]
    )
    runtime = _runtime(tmp_path, monkeypatch, parent, child)
    await runtime.start()
    try:
        await runtime.invoke(AgentRequest("chat", "Create the file"))
        await asyncio.gather(*(job.worker for job in runtime.background._jobs.values()))
        assert output.read_text() == "done"
    finally:
        await runtime.stop()


def _inventory(runtime):
    return runtime._graph.nodes["tools"].bound.tools_by_name["get_agent_tools"].invoke({})


@pytest.mark.parametrize(
    "definition", [{"runnable": RunnableLambda(lambda state: state)}, {"graph_id": "remote"}]
)
def test_opaque_general_purpose_is_rejected(definition):
    spec = {"name": "general-purpose", "description": "Custom agent", **definition}
    with pytest.raises(ValueError, match="must use a name other than 'general-purpose'"):
        prepare_subagents([spec], [], "test:model", None)


@pytest.mark.parametrize("source", ["local", "supplied", "compiled"])
async def test_fork_is_rejected(tmp_path, source):
    spec = {"name": "researcher", "description": "Research", "mode": "fork"}
    if source == "local":
        path = _write_agent(tmp_path)
        path.write_text(
            path.read_text().replace("description: Research", "mode: fork\ndescription: Research")
        )
    elif source == "compiled":
        spec["runnable"] = RunnableLambda(lambda state: state)
    runtime = DeepAgentRuntime(
        model="test:model", assistant_dir=tmp_path, subagents=[] if source == "local" else [spec]
    )
    with pytest.raises(ValueError, match="fresh context"):
        await runtime.start()


@pytest.mark.parametrize(
    "selection",
    [{"tools": ["missing"]}, {"tools": ["task"]}, {"tools": ["read_file", "read_file"]}],
)
async def test_general_requires_valid_selection(tmp_path, monkeypatch, selection):
    parent = ToolModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    _call("task", subagent_type="general-purpose", description="Work", **selection)
                ],
            ),
            AIMessage(content="Done"),
        ]
    )
    child = ToolModel(responses=[AIMessage(content="Must not run")])
    runtime = _runtime(tmp_path, monkeypatch, parent, child)
    await runtime.start()
    try:
        await runtime.invoke(AgentRequest("chat", "Work"))
        await asyncio.gather(*(job.worker for job in runtime.background._jobs.values()))
        assert not child._seen
        assert "Specify tools" in next(iter(runtime.background.results("chat").values()))
    finally:
        await runtime.stop()


async def test_attachment_reload_and_invalid_edits_retain_effective_graph(tmp_path, monkeypatch):
    _write_agent(tmp_path, "[first]")

    @tool
    def first() -> str:
        """First source."""
        return "first"

    @tool
    def second() -> str:
        """Second source."""
        return "second"

    model = ToolModel(responses=[AIMessage(content="Done")])
    runtime = _runtime(tmp_path, monkeypatch, model, model, tools=[first, second])
    await runtime.start()
    try:
        old_view = runtime._graph.nodes["tools"].bound.tools_by_name["get_agent_tools"]
        _write_agent(tmp_path, "[second]")
        assert _inventory(runtime)["saved_changes_inactive"]
        assert runtime._attachments[1]["tools"] == ["first"]
        await runtime.reload_subagent_configuration()
        assert not _inventory(runtime)["saved_changes_inactive"]
        assert runtime._attachments[1]["tools"] == ["second"]
        previous = old_view.invoke({})
        assert previous["current_turn_uses_previous_graph"]
        assert previous["agents"][1]["tools"] == ["first"]
        assert previous["latest_agents"][1]["tools"] == ["second"]
        active = runtime._graph
        for tools in ("null", "lookup", "[lookup, lookup]", "[1]", "[missing]"):
            _write_agent(tmp_path, tools)
            result = await runtime._subagent_reload_tool().ainvoke({})
            assert result["status"] == "failed"
            assert "inactive" in result["message"]
            assert runtime._graph is active
            assert _inventory(runtime)["saved_changes_inactive"]
            assert runtime._attachments[1]["tools"] == ["second"]
    finally:
        await runtime.stop()
