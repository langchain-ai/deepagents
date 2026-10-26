"""One-off model consultations are opt-in, approved, and context-limited."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from deepagents_talon.background import _IN_SUBAGENT
from deepagents_talon.interfaces import AgentRequest, ToolApprovalRequest
from deepagents_talon.runtime import DeepAgentRuntime
from deepagents_talon.tool_approvals import ToolApprovalStore
from tests.unit_tests.test_research_subagents import ToolModel, _call, _inventory

if TYPE_CHECKING:
    from pathlib import Path


def _runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, parent: ToolModel, helper: ToolModel
) -> DeepAgentRuntime:
    monkeypatch.setattr(
        "deepagents_talon.runtime._resolve_model_from_env",
        lambda spec, *_args, **_kwargs: helper if spec == "test:helper" else parent,
    )
    return DeepAgentRuntime(
        model="test:parent",
        env={"DEEPAGENTS_TALON_HELP_MODEL": "test:helper"},
        assistant_dir=tmp_path,
        skills=(),
        memory=(),
        include_web_tools=False,
        max_retries=1,
    )


async def test_smart_model_switch_updates_future_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = ToolModel(responses=[AIMessage(content="done")])
    helper = ToolModel(responses=[AIMessage(content="Advice")])
    runtime = _runtime(tmp_path, monkeypatch, parent, helper)
    await runtime.start()
    try:
        assert await runtime.select_smart_model(None)
        assert runtime.smart_model is None
        assert "ask_for_help" not in (await _inventory(runtime))["agents"][0]["tools"]
        assert await runtime.select_smart_model("test:parent")
        assert runtime.smart_model == "test:parent"
        assert "ask_for_help" in (await _inventory(runtime))["agents"][0]["tools"]
        assert not await runtime.select_smart_model("unavailable:model")
        assert runtime.smart_model == "test:parent"
    finally:
        await runtime.stop()


async def test_help_tool_is_opt_in(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = ToolModel(responses=[AIMessage(content="done")])
    monkeypatch.setattr("deepagents_talon.runtime._resolve_model_from_env", lambda *_a, **_k: model)
    runtime = DeepAgentRuntime(
        model="test:parent", env={}, assistant_dir=tmp_path, skills=(), memory=()
    )
    await runtime.start()
    try:
        inventory = await _inventory(runtime)
        assert "ask_for_help" not in inventory["agents"][0]["tools"]
    finally:
        await runtime.stop()


@pytest.mark.parametrize("decision", ["approve", "reject"])
async def test_help_sends_only_approved_question(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, decision: str
) -> None:
    parent = ToolModel(
        responses=[
            AIMessage(content="", tool_calls=[_call("ask_for_help", question="Why did X fail?")]),
            AIMessage(content="done"),
        ]
    )
    helper = ToolModel(responses=[AIMessage(content="Try checking X.")])
    runtime = _runtime(tmp_path, monkeypatch, parent, helper)
    seen: list[ToolApprovalRequest] = []

    async def decide(request: ToolApprovalRequest) -> str:
        seen.append(request)
        return decision

    await runtime.start()
    try:
        inventory = await _inventory(runtime)
        assert "ask_for_help" in inventory["agents"][0]["tools"]
        assert all(
            "ask_for_help" not in agent.get("selectable_tools", []) for agent in inventory["agents"]
        )
        await runtime.invoke(
            AgentRequest(
                "chat",
                "PRIVATE HISTORY MARKER",
                metadata={"tool_approval_operator": True},
                approval_handler=decide,
            )
        )
        assert len(seen) == 1
        assert seen[0].action_requests[0]["args"] == {"question": "Why did X fail?"}
        if decision == "approve":
            messages = helper._seen[0]
            assert len(messages) == 2
            assert isinstance(messages[0], SystemMessage)
            assert isinstance(messages[1], HumanMessage)
            assert messages[1].content == "Why did X fail?"
            assert "PRIVATE HISTORY MARKER" not in str(messages)
        else:
            assert helper._seen == []
    finally:
        await runtime.stop()


@pytest.mark.parametrize("operator", [True, False])
async def test_help_does_not_send_without_approval_handler_or_operator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, operator: bool
) -> None:
    parent = ToolModel(
        responses=[
            AIMessage(content="", tool_calls=[_call("ask_for_help", question="Private data")]),
            AIMessage(content="done"),
        ]
    )
    helper = ToolModel(responses=[AIMessage(content="Unexpected")])
    runtime = _runtime(tmp_path, monkeypatch, parent, helper)
    await runtime.start()
    try:
        await runtime.invoke(
            AgentRequest("chat", "work", metadata={"tool_approval_operator": operator})
        )
        assert helper._seen == []
    finally:
        await runtime.stop()


async def test_help_still_requires_approval_if_policy_disables_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = ToolModel(
        responses=[
            AIMessage(content="", tool_calls=[_call("ask_for_help", question="Why?")]),
            AIMessage(content="done"),
        ]
    )
    helper = ToolModel(responses=[AIMessage(content="Advice")])
    store = ToolApprovalStore(tmp_path / "tools.json")
    snapshot = store.ensure()
    store.update({"ask_for_help": False}, snapshot.revision)
    runtime = _runtime(tmp_path, monkeypatch, parent, helper)
    seen: list[ToolApprovalRequest] = []

    async def reject(request: ToolApprovalRequest) -> str:
        seen.append(request)
        return "reject"

    await runtime.start()
    try:
        await runtime.invoke(
            AgentRequest(
                "chat",
                "work",
                metadata={"tool_approval_operator": True},
                approval_handler=reject,
            )
        )
        assert len(seen) == 1
        assert seen[0].action_requests[0]["name"] == "ask_for_help"
        assert helper._seen == []
    finally:
        await runtime.stop()


async def test_help_refuses_delegated_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = ToolModel(responses=[AIMessage(content="done")])
    helper = ToolModel(responses=[AIMessage(content="Unexpected")])
    runtime = _runtime(tmp_path, monkeypatch, parent, helper)
    await runtime.start()
    try:
        tool = runtime._graph.nodes["tools"].bound.tools_by_name["ask_for_help"]
        token = _IN_SUBAGENT.set(True)
        try:
            result = await tool.ainvoke({"question": "Private data"})
            assert "Only an operator" in result
            assert helper._seen == []
        finally:
            _IN_SUBAGENT.reset(token)
    finally:
        await runtime.stop()
