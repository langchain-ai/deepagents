from __future__ import annotations

import asyncio

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph

from deepagents_talon.interfaces import AgentRequest
from deepagents_talon.runtime import DeepAgentRuntime


@pytest.fixture(autouse=True)
def stub_child_compilation(monkeypatch):
    monkeypatch.setattr(
        "deepagents_talon.subagents._compile_fresh",
        lambda spec, *_args: {**spec, "runnable": RunnableLambda(lambda state: state)},
    )


def _write_agent(path, prompt):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\ndescription: Research tasks\n---\n{prompt}")


def _graph_factory(entered=None, release=None):
    def create(**kwargs: object):
        names = ",".join(
            agent.get("system_prompt", agent.get("graph_id", ""))
            for agent in kwargs["subagents"] or []
            if agent["name"] != "general-purpose"
        )

        async def reply(state):
            if entered is not None and names == "old":
                entered.set()
                await release.wait()
            return {"messages": [AIMessage(content=f"{names or 'none'}:{len(state['messages'])}")]}

        graph = StateGraph(MessagesState)
        graph.add_node("reply", reply)
        graph.add_edge(START, "reply")
        graph.add_edge("reply", END)
        return graph.compile(checkpointer=kwargs["checkpointer"])

    return create


async def test_reload_add_edit_delete_and_failure_preserve_sqlite_history(tmp_path, monkeypatch):
    path = tmp_path / "agents" / "researcher" / "AGENTS.md"
    monkeypatch.setattr("deepagents_talon.runtime.create_deep_agent", _graph_factory())
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "checkpoints.sqlite")) as saver:
        runtime = DeepAgentRuntime(
            model="test:model",
            assistant_dir=tmp_path,
            checkpointer=saver,
            include_web_tools=False,
            skills=(),
            memory=(),
        )
        await runtime.start()
        reload_tool = runtime._subagent_reload_tool()
        try:
            assert (await runtime.invoke(AgentRequest("chat", "one"))).text == "none:1"
            _write_agent(path, "old")
            assert (await runtime.invoke(AgentRequest("chat", "two"))).text == "none:3"
            assert await reload_tool.ainvoke({}) == {"status": "reloaded", "available": "next_turn"}
            assert (await runtime.invoke(AgentRequest("chat", "three"))).text == "old:5"
            _write_agent(path, "new")
            assert (await runtime.invoke(AgentRequest("chat", "four"))).text == "old:7"
            await reload_tool.ainvoke({})
            assert (await runtime.invoke(AgentRequest("chat", "five"))).text == "new:9"
            path.write_text("unfinished edit")
            assert (await reload_tool.ainvoke({}))["status"] == "failed"
            assert (await runtime.invoke(AgentRequest("chat", "six"))).text == "new:11"
            path.unlink()
            assert (await runtime.invoke(AgentRequest("chat", "seven"))).text == "new:13"
            await reload_tool.ainvoke({})
            assert (await runtime.invoke(AgentRequest("chat", "eight"))).text == "none:15"
        finally:
            await runtime.stop()


async def test_remote_loader_runs_only_at_startup_and_explicit_reload(monkeypatch):
    definitions = [{"name": "remote", "description": "Research", "graph_id": "old"}]
    loads = []

    def load_subagents():
        loads.append(True)
        return definitions

    monkeypatch.setattr("deepagents_talon.runtime.create_deep_agent", _graph_factory())
    runtime = DeepAgentRuntime(
        model="test:model",
        load_subagents=load_subagents,
        include_web_tools=False,
        skills=(),
        memory=(),
    )
    await runtime.start()
    try:
        assert (await runtime.invoke(AgentRequest("chat", "one"))).text == "old:1"
        definitions[0]["graph_id"] = "new"
        assert (await runtime.invoke(AgentRequest("chat", "two"))).text == "old:3"
        assert len(loads) == 1
        await runtime._subagent_reload_tool().ainvoke({})
        assert (await runtime.invoke(AgentRequest("chat", "three"))).text == "new:5"
        assert len(loads) == 2
    finally:
        await runtime.stop()


async def test_reload_keeps_active_turn_on_original_graph(tmp_path, monkeypatch):
    entered, release = asyncio.Event(), asyncio.Event()
    path = tmp_path / "agents" / "researcher" / "AGENTS.md"
    _write_agent(path, "old")
    monkeypatch.setattr(
        "deepagents_talon.runtime.create_deep_agent", _graph_factory(entered, release)
    )
    runtime = DeepAgentRuntime(
        model="test:model", assistant_dir=tmp_path, include_web_tools=False, skills=(), memory=()
    )
    await runtime.start()
    active = asyncio.create_task(runtime.invoke(AgentRequest("chat", "work")))
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        _write_agent(path, "new")
        await runtime.reload_subagent_configuration()
        assert (await runtime.invoke(AgentRequest("other-chat", "work"))).text == "new:1"
        release.set()
        assert (await active).text == "old:1"
    finally:
        release.set()
        await active
        await runtime.stop()


async def test_compile_failure_retains_previous_configuration(tmp_path, monkeypatch):
    path = tmp_path / "agents" / "researcher" / "AGENTS.md"
    _write_agent(path, "old")
    create = _graph_factory()

    def sometimes_fails(**kwargs: object):
        if kwargs["subagents"][0]["system_prompt"] == "bad":
            msg = "Compilation failed"
            raise RuntimeError(msg)
        return create(**kwargs)

    monkeypatch.setattr("deepagents_talon.runtime.create_deep_agent", sometimes_fails)
    runtime = DeepAgentRuntime(
        model="test:model", assistant_dir=tmp_path, include_web_tools=False, skills=(), memory=()
    )
    await runtime.start()
    try:
        _write_agent(path, "bad")
        with pytest.raises(RuntimeError, match="Compilation failed"):
            await runtime.reload_subagent_configuration()
        assert (await runtime.invoke(AgentRequest("chat", "work"))).text == "old:1"
    finally:
        await runtime.stop()
