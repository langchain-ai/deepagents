"""Skill-only MCP tools in the production dcode agent stack."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import PrivateAttr

from deepagents_code.agent import create_cli_agent
from deepagents_code.skills.invocation import build_skill_invocation_envelope

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from langchain_core.runnables import RunnableConfig


class _RecordingModel(GenericFakeChatModel):
    _schemas: list[set[str]] = PrivateAttr(default_factory=list)

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> _RecordingModel:
        _ = tool_choice, kwargs
        self._schemas.append(
            {item.name for item in tools if isinstance(item, BaseTool)}
        )
        return self


def _call(name: str, **args: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": name}])


@pytest.mark.parametrize(
    "activation", ["read", "pin", "approval", "disabled", "unread"]
)
async def test_skill_only_tool_disclosure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, activation: str
) -> None:
    calls: list[str] = []

    @tool
    def scoped_tool() -> str:
        """Read the skill's records."""
        calls.append("scoped")
        return "skill result"

    @tool
    def eager_tool() -> str:
        """Always available."""
        return "eager result"

    scoped_tool.metadata = {
        "_deepagents_code_mcp": True,
        "_deepagents_code_skill_only": True,
    }
    root = tmp_path / "skills"
    path = root / "records" / "SKILL.md"
    path.parent.mkdir(parents=True)
    path.write_text(
        "---\nname: records\ndescription: Read records\nmetadata:\n"
        "  include_tools: scoped_tool\n---\nUse scoped_tool.\n"
    )
    monkeypatch.setattr(
        "deepagents_code.agent.get_skill_sources", lambda **_: [(str(root), "Test")]
    )
    responses = []
    if activation == "read":
        responses.append(_call("read_file", file_path=str(path)))
    responses.extend([_call("scoped_tool"), AIMessage(content="done")])
    model = _RecordingModel(
        messages=iter(responses),
        profile={"max_input_tokens": 200000},
        disable_streaming=True,
    )
    graph, _ = create_cli_agent(
        model,
        "skill-tools-test",
        cwd=tmp_path,
        tools=[scoped_tool, eager_tool],
        mcp_tools=[scoped_tool, eager_tool],
        auto_approve=activation != "approval",
        enable_memory=False,
        enable_skills=activation != "disabled",
        enable_shell=False,
        checkpointer=InMemorySaver(),
    )
    state: dict[str, Any] = {"messages": [HumanMessage("read records")]}
    if activation in {"pin", "approval"}:
        envelope = build_skill_invocation_envelope("records")
        state = {
            "messages": [HumanMessage(envelope.prompt, **envelope.message_kwargs)],
            "pinned_skills": ["records"],
        }
    config: RunnableConfig = {"configurable": {"thread_id": "skill-tools-test"}}
    result = await graph.ainvoke(state, config)
    if activation == "approval":
        assert calls == []
        [approval] = result["__interrupt__"]
        assert approval.value["action_requests"][0]["name"] == "scoped_tool"
        result = await graph.ainvoke(
            Command(resume={"decisions": [{"type": "approve"}]}), config
        )
    assert all("eager_tool" in names for names in model._schemas)
    if activation in {"disabled", "unread"}:
        assert calls == []
        assert all("scoped_tool" not in names for names in model._schemas)
        assert any(
            isinstance(message, ToolMessage) and message.status == "error"
            for message in result["messages"]
        )
    else:
        assert calls == ["scoped"]
        assert "scoped_tool" in model._schemas[-1]
        if activation == "read":
            assert "scoped_tool" not in model._schemas[0]
        assert any(
            isinstance(message, ToolMessage) and message.content == "skill result"
            for message in result["messages"]
        )


@pytest.mark.parametrize(
    ("subagent", "forked"),
    [("researcher", "0"), ("general-purpose", "0"), ("general-purpose", "1")],
)
async def test_delegated_skill_tool_retains_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, subagent: str, forked: str
) -> None:
    cancelled = asyncio.Event()

    @tool
    async def scoped_tool() -> str:
        """Simulate a stalled MCP server."""
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
        return "unreachable"

    scoped_tool.metadata = {
        "_deepagents_code_mcp": True,
        "_deepagents_code_skill_only": True,
    }
    root = tmp_path / "skills"
    path = root / "records" / "SKILL.md"
    path.parent.mkdir(parents=True)
    path.write_text(
        "---\nname: records\ndescription: Read records\nmetadata:\n"
        "  include_tools: scoped_tool\n---\nUse scoped_tool.\n"
    )
    monkeypatch.setenv("DEEPAGENTS_CODE_FORKED_SUBAGENTS", forked)
    monkeypatch.setattr(
        "deepagents_code.agent.get_skill_sources", lambda **_: [(str(root), "Test")]
    )
    monkeypatch.setattr(
        "deepagents_code.config_manifest.resolve_mcp_tool_timeout", lambda: 0.01
    )
    monkeypatch.setattr(
        "deepagents_code.agent.list_subagents",
        lambda **_: [
            {
                "name": "researcher",
                "description": "Research",
                "system_prompt": "Research",
                "model": None,
            }
        ],
    )
    model = _RecordingModel(
        messages=iter(
            [
                _call("task", description="read records", subagent_type=subagent),
                _call("read_file", file_path=str(path)),
                _call("scoped_tool"),
                AIMessage(content="child recovered"),
                AIMessage(content="parent recovered"),
            ]
        ),
        profile={"max_input_tokens": 200000},
        disable_streaming=True,
    )
    graph, _ = create_cli_agent(
        model,
        "skill-tools-test",
        cwd=tmp_path,
        tools=[scoped_tool],
        mcp_tools=[scoped_tool],
        auto_approve=True,
        enable_memory=False,
        enable_shell=False,
    )
    result = await asyncio.wait_for(
        graph.ainvoke({"messages": [HumanMessage("delegate research")]}), timeout=5
    )
    assert cancelled.is_set()
    assert "scoped_tool" not in model._schemas[0]
    assert any("scoped_tool" in names for names in model._schemas)
    assert result["messages"][-1].content == "parent recovered"
