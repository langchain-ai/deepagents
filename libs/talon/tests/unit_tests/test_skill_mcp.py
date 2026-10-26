"""Exercise skill-scoped MCP disclosure through Talon's real runtime."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from fastmcp import FastMCP
from fastmcp.client.transports import FastMCPTransport
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from pydantic import Field

from deepagents_talon import mcp
from deepagents_talon.authorization import current_authorization_invocation
from deepagents_talon.config import TalonConfig
from deepagents_talon.interfaces import AgentRequest
from deepagents_talon.mcp_config import MCPConfigStore
from deepagents_talon.mcp_middleware import talon_mcp_middleware
from deepagents_talon.runtime import DeepAgentRuntime

if TYPE_CHECKING:
    from pathlib import Path


class ToolRecorder(FakeMessagesListChatModel):
    schemas: list[list[str]] = Field(default_factory=list)

    def bind_tools(self, tools: list, **_kwargs: object) -> ToolRecorder:
        self.schemas.append([item.name for item in tools])
        return self


def call(name: str, **args: object) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "id": name, "args": args}])


@pytest.mark.parametrize("activation", ["read", "pin", "none", "disabled", "eager"])
@pytest.mark.parametrize("decision", ["approve", "reject"])
async def test_disclosure_preserves_approval_and_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, activation: str, decision: str
) -> None:
    server = FastMCP("records")
    effects: list[str] = []

    @server.tool
    async def search() -> str:
        assert current_authorization_invocation() is not None
        effects.append("searched")
        return "found"

    async def connection(*_args: object) -> tuple[FastMCPTransport, str]:
        return FastMCPTransport(server), "stdio"

    monkeypatch.setattr(mcp, "_connection", connection)
    config_path = tmp_path / "mcp.json"
    config_path.write_text(
        json.dumps(
            {"mcpServers": {"records": {"command": "unused", "skillOnly": activation != "eager"}}}
        )
    )
    config = TalonConfig.from_env(
        {"DEEPAGENTS_TALON_MCP_CONFIG": str(config_path)}, base_home=tmp_path
    )
    loaded = await mcp.load_mcp_tools(config)
    skill = tmp_path / "skills" / "records" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text(
        "---\nname: records\ndescription: Find records\n"
        "metadata:\n  include_tools: records_search\n---\nSearch records."
    )
    responses = [call("records_search"), AIMessage(content="done")]
    if activation == "read":
        responses.insert(0, call("read_file", file_path=str(skill)))
    model = ToolRecorder(responses=responses)
    monkeypatch.setattr("deepagents_talon.runtime._resolve_model_from_env", lambda *_a, **_k: model)
    runtime = DeepAgentRuntime(
        model="test:model",
        assistant_dir=tmp_path,
        tools=loaded.tools,
        middleware=(talon_mcp_middleware(),),
        skills=() if activation == "disabled" else None,
        include_web_tools=False,
        memory=(),
    )
    policy = runtime.approval_store.ensure()
    runtime.approval_store.update({"records_search": True}, policy.revision)
    approvals: list[str] = []

    async def decide(request: object) -> str:
        approvals.extend(action["name"] for action in request.action_requests)
        assert not effects
        return decision

    await runtime.start()
    try:
        result = await runtime.invoke(
            AgentRequest(
                "chat",
                "find records",
                metadata={"trigger": "cron", "pinned_skills": ["records"]}
                if activation == "pin"
                else {},
                approval_handler=decide,
            )
        )
        assert result.text == "done"
    finally:
        await runtime.stop()
    activated = activation in {"read", "pin", "eager"}
    assert ("records_search" in model.schemas[0]) is (activation in {"pin", "eager"})
    assert any("records_search" in names for names in model.schemas) is activated
    assert effects == (
        ["searched"] if activation in {"read", "eager"} and decision == "approve" else []
    )
    assert approvals == ([] if activation == "pin" else ["records_search"])


@pytest.mark.parametrize("value", ["true", 1, None, []])
def test_skill_only_requires_a_boolean(tmp_path: Path, value: object) -> None:
    path = tmp_path / "mcp.json"
    path.write_text(
        json.dumps({"mcpServers": {"records": {"command": "unused", "skillOnly": value}}})
    )
    with pytest.raises(mcp.MCPConfigError, match="skillOnly must be a boolean"):
        mcp._load_config(path, {})


@pytest.mark.parametrize("value", [True, False, "true"])
def test_config_management_validates_and_preserves_skill_only(
    tmp_path: Path, value: object
) -> None:
    path = tmp_path / "mcp.json"
    store = MCPConfigStore(path, lambda: None)
    view, update = store.tools()
    result = update.invoke(
        {
            "server_name": "records",
            "server": {"command": "unused", "skillOnly": value},
            "expected_revision": view.invoke({})["revision"],
        }
    )
    if isinstance(value, bool):
        assert result["status"] == "updated"
        assert view.invoke({})["mcpServers"]["records"]["skillOnly"] is value
    else:
        assert result["status"] != "updated"
        assert not path.exists()


async def test_reload_reclassifies_tools_for_an_existing_conversation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    @tool
    def records_search() -> str:
        """Find records."""
        return "found"

    model = ToolRecorder(responses=[AIMessage(content="done")])
    monkeypatch.setattr("deepagents_talon.runtime._resolve_model_from_env", lambda *_a, **_k: model)
    replacement = [records_search]

    async def reload_tools() -> list:
        return replacement

    runtime = DeepAgentRuntime(
        model="test:model",
        assistant_dir=tmp_path,
        tools=[records_search],
        reload_tools=reload_tools,
        include_web_tools=False,
        memory=(),
    )
    await runtime.start()
    try:
        await runtime.invoke(AgentRequest("chat", "before"))
        assert "records_search" in model.schemas[-1]
        replacement = [
            records_search.model_copy(update={"metadata": {"_deepagents_talon_skill_only": True}})
        ]
        await runtime.reload_mcp_configuration()
        await runtime.invoke(AgentRequest("chat", "after"))
        assert "records_search" not in model.schemas[-1]
        replacement = []
        await runtime.reload_mcp_configuration()
        await runtime.invoke(AgentRequest("chat", "removed"))
        assert "records_search" not in model.schemas[-1]
    finally:
        await runtime.stop()
