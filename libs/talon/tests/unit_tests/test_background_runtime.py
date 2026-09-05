from __future__ import annotations

import asyncio

import pytest
from langchain_core._api import LangChainBetaWarning
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from deepagents_talon.interfaces import AgentRequest
from deepagents_talon.runtime import DeepAgentRuntime


class ToolModel(FakeMessagesListChatModel):
    def bind_tools(self, _tools, **_kwargs: object):
        return self


async def test_real_graph_launch_and_child_approval(tmp_path, monkeypatch):
    path = tmp_path / "agents" / "researcher" / "AGENTS.md"
    path.parent.mkdir(parents=True)
    path.write_text("---\ndescription: Research\nmodel: test:child\n---\nResearch carefully.")
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
                        "name": "start_local_task",
                        "id": "launch",
                        "args": {
                            "subagent_type": "researcher",
                            "description": "work",
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
    runtime = DeepAgentRuntime(
        model="test:parent",
        assistant_dir=tmp_path,
        tools=[sensitive_effect],
        interrupt_on={"sensitive_effect": True},
        include_web_tools=False,
        skills=(),
        memory=(),
    )
    approvals = []

    async def approve(request):
        approvals.extend(item["name"] for item in request.action_requests)
        return "approve"

    with pytest.warns(LangChainBetaWarning, match="forked subagents"):
        await runtime.start()
    try:
        result = await runtime.invoke(AgentRequest("chat", "delegate", approval_handler=approve))
        assert result.text == "Started background work"
        assert approvals == ["start_local_task"]
        supervisor = runtime.local_tasks
        workers = list(supervisor._workers.items())
        await asyncio.gather(*(worker for _task_id, worker in workers))
        task_id = supervisor.pending_results()[0]["id"]
        assert supervisor._task(task_id, "chat")["status"] == "interrupted"
        assert effects == []
        await supervisor.resume(task_id, "chat", "approve")
        await asyncio.gather(*list(supervisor._workers.values()))
        assert supervisor._task(task_id, "chat")["status"] == "success"
        assert effects == ["effect"]
    finally:
        await runtime.stop()
