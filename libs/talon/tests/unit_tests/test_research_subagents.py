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
from deepagents_talon.tool_approvals import ToolApprovalStore


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
        skills=(),
        **{"include_web_tools": False, "memory": (), **kwargs},
    )


@pytest.mark.parametrize("background", [False, True])
@pytest.mark.parametrize(
    ("name", "attached"),
    [
        ("researcher", True),
        ("researcher", False),
        ("prepared", False),
        ("prepared", True),
    ],
)
async def test_research_boundaries(tmp_path, monkeypatch, background, name, attached):
    _write_agent(tmp_path, "[lookup]" if attached else "[]")
    _write_agent(tmp_path, name="prepared")
    private = "PRIVATE-PARENT-MARKER"
    memory = tmp_path / "memory.md"
    memory.write_text(private)
    output = tmp_path / "output.txt"
    skill = tmp_path / "skill.md"
    skill.write_text("Use lookup for research.")
    selected = ["lookup", "read_file"] if name == "prepared" and attached else ["lookup"]
    launch = {"tools": selected} if name == "prepared" and attached else {}
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
        if name == "prepared" and attached:
            assert "Use lookup for research." in str(child._seen)
        messages = child._seen[-1]
        denied = [message for message in messages if getattr(message, "status", None) == "error"]
        assert {message.name for message in denied} == (
            {call["name"] for call in forbidden} if attached else set()
        )
        inventory = await _inventory(runtime)
        agent = next(item for item in inventory["agents"] if item["name"] == name)
        if name == "prepared":
            assert "read_file" in agent["selectable_tools"]
            assert "task" not in agent["selectable_tools"]
        else:
            assert agent["tools"] == (["lookup"] if attached else [])
        assert private not in json.dumps(inventory)
    finally:
        await runtime.stop()


@pytest.mark.parametrize("name", ["researcher", "prepared"])
async def test_explicit_shell_access(tmp_path, monkeypatch, name):
    _write_agent(tmp_path, "[execute]")
    _write_agent(tmp_path, name="prepared")
    output = tmp_path / "child-output"
    launch = {"tools": ["execute"]} if name == "prepared" else {}
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


async def _inventory(runtime):
    return await runtime._graph.nodes["tools"].bound.tools_by_name["get_agent_tools"].ainvoke({})


@pytest.mark.parametrize("configured", [False, True])
async def test_no_implicit_general_purpose_agent(tmp_path, monkeypatch, configured):
    path = _write_agent(tmp_path) if configured else None
    parent = ToolModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[_call("task", subagent_type="general-purpose", description="Work")],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    _call(
                        "task",
                        subagent_type="general-purpose",
                        description="Work",
                        tools=["execute"],
                    )
                ],
            ),
            AIMessage(content="Done"),
        ]
    )
    child = ToolModel(responses=[AIMessage(content="Must not run")])
    runtime = _runtime(tmp_path, monkeypatch, parent, child)
    await runtime.start()
    try:
        tools = runtime._graph.nodes["tools"].bound.tools_by_name
        assert ("task" in tools) == configured
        assert {agent["name"] for agent in (await _inventory(runtime))["agents"]} == (
            {"main", "researcher"} if configured else {"main"}
        )
        if configured:
            assert "general-purpose" not in tools["task"].description
        await runtime.invoke(AgentRequest("chat", "Work"))
        await asyncio.gather(*(job.worker for job in runtime.background._jobs.values()))
        assert not child._seen
        if path is not None:
            path.unlink()
            await runtime.reload_subagent_configuration()
            assert "task" not in runtime._graph.nodes["tools"].bound.tools_by_name
    finally:
        await runtime.stop()


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
@pytest.mark.parametrize("name", ["researcher", "prepared"])
async def test_task_requires_valid_selection(tmp_path, monkeypatch, selection, name):
    _write_agent(tmp_path)
    _write_agent(tmp_path, name="prepared")
    parent = ToolModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[_call("task", subagent_type=name, description="Work", **selection)],
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


@pytest.mark.parametrize("protected", [False, True])
async def test_named_task_adds_tools_without_changing_defaults(tmp_path, monkeypatch, protected):
    path = _write_agent(tmp_path, "[first]")
    original = path.read_text()
    effects = []

    @tool
    def first() -> str:
        """Read configured evidence."""
        effects.append("first")
        return "Source: configured"

    @tool
    def second() -> str:
        """Perform an additional operation."""
        effects.append("second")
        return "Source: additional"

    child = ToolModel(
        responses=[
            AIMessage(content="", tool_calls=[_call("first")]),
            AIMessage(content="", tool_calls=[_call("second")]),
            AIMessage(content="Done"),
        ]
    )
    parent = ToolModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    _call(
                        "task",
                        subagent_type="researcher",
                        description="Research",
                        tools=["first", "second"],
                    )
                ],
            ),
            AIMessage(content="Delegated"),
            AIMessage(
                content="",
                tool_calls=[
                    _call("task", subagent_type="researcher", description="Research again")
                ],
            ),
            AIMessage(content="Delegated"),
        ]
    )
    store = ToolApprovalStore(tmp_path / "tools.json")
    snapshot = store.ensure()
    store.update({"second": protected}, snapshot.revision)
    runtime = _runtime(
        tmp_path,
        monkeypatch,
        parent,
        child,
        tools=[first, second],
        approval_store=store,
    )
    await runtime.start()
    try:
        await runtime.invoke(AgentRequest("chat", "Research"))
        await asyncio.gather(*(job.worker for job in runtime.background._jobs.values()))
        assert effects == (["first"] if protected else ["first", "second"])
        assert set(child._tools[0]) == {"first", "second"}
        assert "Answer the delegated question." in str(child._seen[0][0])
        if protected:
            assert "needs tool approval" in str(runtime.background.results("chat"))
        child.i = 0
        await runtime.invoke(AgentRequest("chat", "Research again"))
        await asyncio.gather(*(job.worker for job in runtime.background._jobs.values()))
        assert effects == (["first", "first"] if protected else ["first", "second", "first"])
        assert child._tools[-1] == ["first"]
        assert path.read_text() == original
        assert next(
            item for item in (await _inventory(runtime))["agents"] if item["name"] == "researcher"
        )["tools"] == ["first"]
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
        assert (await _inventory(runtime))["saved_changes_inactive"]
        assert runtime._attachments[1]["tools"] == ["first"]
        await runtime.reload_subagent_configuration()
        assert not (await _inventory(runtime))["saved_changes_inactive"]
        assert runtime._attachments[1]["tools"] == ["second"]
        previous = await old_view.ainvoke({})
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
            assert (await _inventory(runtime))["saved_changes_inactive"]
            assert runtime._attachments[1]["tools"] == ["second"]
    finally:
        await runtime.stop()


@pytest.mark.parametrize("declared", [False, True])
async def test_web_tools_follow_the_declared_capability_not_the_agent_name(
    tmp_path, monkeypatch, declared
):
    path = tmp_path / "agents" / "external-research" / "AGENTS.md"
    path.parent.mkdir(parents=True)
    frontmatter = "---\ndescription: Public research\ntools: []\n"
    path.write_text(
        f"{frontmatter}web: true\n---\nResearch." if declared else f"{frontmatter}---\nResearch."
    )
    model = ToolModel(responses=[AIMessage(content="Done")])
    runtime = _runtime(tmp_path, monkeypatch, model, model, include_web_tools=True)
    await runtime.start()
    try:
        agents = {item["name"]: item["tools"] for item in (await _inventory(runtime))["agents"]}
        assert agents["external-research"] == (["fetch_url"] if declared else [])
    finally:
        await runtime.stop()


async def test_declared_web_capability_travels_with_a_renamed_agent(tmp_path, monkeypatch):
    path = tmp_path / "agents" / "public-digging" / "AGENTS.md"
    path.parent.mkdir(parents=True)
    path.write_text("---\ndescription: Public research\ntools: []\nweb: true\n---\nResearch.")
    model = ToolModel(responses=[AIMessage(content="Done")])
    runtime = _runtime(tmp_path, monkeypatch, model, model, include_web_tools=True)
    await runtime.start()
    try:
        agents = {item["name"]: item["tools"] for item in (await _inventory(runtime))["agents"]}
        assert agents["public-digging"] == ["fetch_url"]
    finally:
        await runtime.stop()


async def test_per_task_agent_runs_with_an_explicit_config_and_survives_no_messages(
    tmp_path, monkeypatch
):
    captured = {}

    class Recorder:
        async def ainvoke(self, payload, config=None):
            captured["config"] = config
            captured["payload"] = payload
            return {"messages": []}

    _write_agent(tmp_path)
    parent = ToolModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    _call(
                        "task",
                        subagent_type="researcher",
                        description="Find evidence",
                        tools=["current_time"],
                    )
                ],
            ),
            AIMessage(content="Done"),
        ]
    )
    child = ToolModel(responses=[AIMessage(content="Must not run")])
    runtime = _runtime(tmp_path, monkeypatch, parent, child)
    await runtime.start()
    # Patched after the graph is built so only the per-task compile is intercepted.
    monkeypatch.setattr(
        "deepagents_talon.subagents._compile_fresh",
        lambda *_args, **_kwargs: {
            "name": "researcher",
            "description": "x",
            "runnable": Recorder(),
        },
    )
    try:
        await runtime.invoke(AgentRequest("chat", "Work"))
        await asyncio.gather(*(job.worker for job in runtime.background._jobs.values()))
        results = list(runtime.background.results("chat").values())
    finally:
        await runtime.stop()

    assert captured["config"] == {"recursion_limit": 500}
    assert "Subagent returned no result." in results[0]
