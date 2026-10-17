from __future__ import annotations

import asyncio

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from deepagents_talon.interfaces import AgentRequest
from deepagents_talon.runtime import DeepAgentRuntime
from deepagents_talon.tool_approvals import ToolApprovalStore
from tests.archive_helpers import make_runtime, make_saver


class ToolModel(FakeMessagesListChatModel):
    def bind_tools(self, _tools, **_kwargs: object):
        return self


@pytest.mark.parametrize("name", ["researcher", "prepared"])
async def test_real_graph_launch_and_child_approval(tmp_path, monkeypatch, name):
    path = tmp_path / "agents" / "researcher" / "AGENTS.md"
    path.parent.mkdir(parents=True)
    path.write_text(
        "---\ndescription: Research\nmodel: test:child\n"
        "tools: [sensitive_effect]\n---\nResearch carefully."
    )
    prepared = tmp_path / "agents" / "prepared" / "AGENTS.md"
    prepared.parent.mkdir(parents=True)
    prepared.write_text("---\ndescription: Prepared task\n---\nComplete the task.")
    effects = []

    @tool
    def sensitive_effect() -> str:
        """Perform a protected action."""
        effects.append("effect")
        return "done"

    parent = ToolModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "task",
                        "id": "launch",
                        "args": {
                            "subagent_type": name,
                            "description": "work",
                            **({"tools": ["sensitive_effect"]} if name == "prepared" else {}),
                        },
                    }
                ],
            ),
            AIMessage(content="Started background work"),
        ]
    )
    child = ToolModel(
        responses=[
            AIMessage(
                content="", tool_calls=[{"name": "sensitive_effect", "id": "effect", "args": {}}]
            ),
            AIMessage(content="Finished"),
        ]
    )
    monkeypatch.setattr(
        "deepagents_talon.runtime._resolve_model_from_env",
        lambda model, *_args, **_kwargs: child if model == "test:child" else parent,
    )
    monkeypatch.setattr(
        "deepagents.graph.resolve_model", lambda model: child if model == "test:child" else model
    )
    monkeypatch.setattr(
        "deepagents_talon.subagents.create_agent",
        lambda **kwargs: create_agent(**{**kwargs, "model": child}),
    )
    store = ToolApprovalStore(tmp_path / "tools.json")
    snapshot = store.ensure()
    store.update({"sensitive_effect": True}, snapshot.revision)
    runtime = DeepAgentRuntime(
        model="test:parent",
        assistant_dir=tmp_path,
        tools=[sensitive_effect],
        approval_store=store,
        include_web_tools=False,
        skills=(),
        memory=(),
    )
    approvals = []

    async def approve(request):
        approvals.extend(item["name"] for item in request.action_requests)
        return "approve"

    await runtime.start()
    try:
        result = await runtime.invoke(AgentRequest("chat", "delegate", approval_handler=approve))
        assert result.text == "Started background work"
        assert approvals == []
        await asyncio.gather(*(job.worker for job in runtime.background._jobs.values()))
        assert effects == []
        results = runtime.background.results("chat")
        assert len(results) == 1
        assert "approval" in next(iter(results.values()))
    finally:
        await runtime.stop()


async def test_background_subagent_keeps_the_hosts_history_scope(tmp_path, monkeypatch):
    path = tmp_path / "agents" / "researcher" / "AGENTS.md"
    path.parent.mkdir(parents=True)
    path.write_text(
        "---\ndescription: Research\nmodel: test:child\n"
        "tools: [search_conversations]\n---\nSearch this chat's history."
    )
    parent = ToolModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "task",
                        "id": "launch",
                        "args": {"subagent_type": "researcher", "description": "recall"},
                    }
                ],
            ),
            AIMessage(content="Started background work"),
        ]
    )
    child = ToolModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "search_conversations", "id": "recall", "args": {"query": "orchard"}}
                ],
            ),
            AIMessage(content="Reviewed the history"),
        ]
    )
    monkeypatch.setattr(
        "deepagents_talon.runtime._resolve_model_from_env",
        lambda model, *_args, **_kwargs: child if model == "test:child" else parent,
    )
    monkeypatch.setattr(
        "deepagents.graph.resolve_model", lambda model: child if model == "test:child" else model
    )
    monkeypatch.setattr(
        "deepagents_talon.subagents.create_agent",
        lambda **kwargs: create_agent(**{**kwargs, "model": child}),
    )

    async with make_saver(str(tmp_path / "history.sqlite"), InMemorySaver) as saver:
        scopes = []
        search_page = saver.archive.search_page

        async def record(scope, **kwargs: object):
            scopes.append(scope)
            return await search_page(scope, **kwargs)

        monkeypatch.setattr(saver.archive, "search_page", record)
        runtime = make_runtime(saver, tmp_path)
        await runtime.start()
        try:
            result = await runtime.invoke(
                AgentRequest(
                    "chat",
                    "recall the orchard",
                    metadata={"history_channel": "whatsapp", "history_chat": "chat"},
                )
            )
            assert result.text == "Started background work"
            await asyncio.gather(*(job.worker for job in runtime.background._jobs.values()))
            results = [job.result for job in runtime.background._jobs.values()]
        finally:
            await runtime.stop()

    assert results == ["Reviewed the history"]
    assert scopes == [{"talon_history_channel": "whatsapp", "talon_history_chat": "chat"}]
