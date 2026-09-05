from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Self

import pytest
from deepagents.backends import StateBackend
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from deepagents_talon.config import TalonConfig
from deepagents_talon.interfaces import AgentRequest
from deepagents_talon.mcp import MCPToolProvider
from deepagents_talon.mcp_config import MCP_CONFIG_AUTO_APPROVE_ENV, MCPConfigStore
from deepagents_talon.runtime import DeepAgentRuntime

if TYPE_CHECKING:
    from pathlib import Path

    from deepagents_talon.interfaces import ToolApprovalDecision, ToolApprovalRequest


@pytest.fixture
def config_tools(tmp_path: Path):
    path = tmp_path / "private" / ".mcp.json"
    updates: list[bool] = []
    store = MCPConfigStore(path, lambda: updates.append(True))
    return path, *store.tools(), updates


def test_view_redacts_literals_without_expansion(config_tools, monkeypatch):
    path, view, _, _ = config_tools
    path.parent.mkdir()
    server = {
        "transport": "http",
        "url": "https://user:private@example.test/mcp?token=private",
        "headers": {"Authorization": "Bearer private", "X-Key": "${CREDENTIAL}"},
        "env": {"KEY": "${CREDENTIAL:-private}", "OTHER": "prefix-${CREDENTIAL}"},
        "args": ["--key=private"],
        "command": "private",
        "unknown": {"nested": "private"},
    }
    path.write_text(json.dumps({"mcpServers": {"example": server}, "metadata": "private"}))
    monkeypatch.setenv("CREDENTIAL", "expanded-private")

    result = view.invoke({})

    assert "private" not in json.dumps(result)
    assert result["mcpServers"]["example"]["headers"]["X-Key"] == "${CREDENTIAL}"
    assert result["mcpServers"]["example"]["transport"] == "http"
    assert result["mcpServers"]["example"]["command"] == "<redacted>"
    assert "metadata" not in result


def test_update_preserves_secrets_and_unrelated_settings(config_tools):
    path, view, update, updates = config_tools
    path.parent.mkdir()
    original = {
        "mcpServers": {
            "example": {"url": "https://example.test/mcp", "headers": {"X-Key": "stored-value"}},
            "other": {"command": "server", "args": ["stored-value"]},
        },
        "metadata": {"keep": True},
    }
    path.write_text(json.dumps(original))
    result = view.invoke({})
    server = result["mcpServers"]["example"]
    server["allowedTools"] = ["read_*"]

    assert update.invoke(
        {"server_name": "example", "server": server, "expected_revision": result["revision"]}
    ) == {"status": "updated", "available": "next_turn"}

    expected = json.loads(json.dumps(original))
    expected["mcpServers"]["example"]["allowedTools"] = ["read_*"]
    assert json.loads(path.read_text()) == expected
    assert path.stat().st_mode & 0o777 == 0o600
    assert updates == [True]
    assert view.invoke({})["revision"] != result["revision"]


def test_add_remove_and_stale_revision(config_tools):
    path, view, update, updates = config_tools
    revision = view.invoke({})["revision"]
    arguments = {
        "server_name": "example",
        "server": {"command": "server"},
        "expected_revision": revision,
    }
    assert update.invoke(arguments)["status"] == "updated"
    assert update.invoke(arguments)["status"] == "conflict"
    assert (
        update.invoke(
            {**arguments, "server": None, "expected_revision": view.invoke({})["revision"]}
        )["status"]
        == "updated"
    )
    assert json.loads(path.read_text()) == {"mcpServers": {}}
    assert updates == [True, True]


def test_external_edit_invalidates_pending_update(config_tools):
    path, view, update, updates = config_tools
    revision = view.invoke({})["revision"]
    path.parent.mkdir()
    path.write_text('{"mcpServers": {}, "operator": true}')
    original = path.read_bytes()
    result = update.invoke(
        {"server_name": "example", "server": {"command": "server"}, "expected_revision": revision}
    )
    assert result["status"] == "conflict"
    assert path.read_bytes() == original
    assert updates == []


@pytest.mark.parametrize(
    "server",
    [
        {"url": "<redacted>"},
        {"command": "server", "env": {"PYTHONPATH": "/untrusted"}},
        {"url": "https://example.test", "auth": "oauth", "headers": {"Authorization": "value"}},
        {"url": "https://example.test", "allowedTools": []},
        {"command": "server", "allowedTools": ["*"], "disabledTools": ["*"]},
        {"url": "https://example.test", "unknown": "value"},
        {"transport": "invalid"},
        {"transport": []},
        {"args": "bad"},
        {"command": 123},
        {"command": "${INVALID"},
        {"url": "https://example.test", "env": []},
    ],
)
def test_invalid_update_leaves_file_unchanged(config_tools, server):
    path, view, update, updates = config_tools
    result = update.invoke(
        {
            "server_name": "example",
            "server": server,
            "expected_revision": view.invoke({})["revision"],
        }
    )
    assert result["status"] == "error"
    assert not path.exists()
    assert updates == []


@pytest.mark.parametrize("content", ["{private invalid", "[]", '{"mcpServers": []}'])
def test_malformed_config_errors_are_redacted(config_tools, content):
    path, view, update, updates = config_tools
    path.parent.mkdir()
    path.write_text(content)
    assert view.invoke({})["status"] == "error"
    result = update.invoke({"server_name": "example", "server": None, "expected_revision": "x"})
    assert result["status"] == "error"
    assert "private" not in json.dumps(result)
    assert path.read_text() == content
    assert updates == []


def test_symlink_cannot_read_or_update_another_file(config_tools, tmp_path):
    path, view, update, updates = config_tools
    revision = view.invoke({})["revision"]
    target = tmp_path / "target"
    target.write_text('{"mcpServers": {}}')
    path.parent.mkdir()
    path.symlink_to(target)
    assert view.invoke({})["status"] == "error"
    assert (
        update.invoke({"server_name": "example", "server": None, "expected_revision": revision})[
            "status"
        ]
        == "error"
    )
    assert target.read_text() == '{"mcpServers": {}}'
    assert updates == []


def test_failed_atomic_replace_preserves_file_and_cleans_temp(config_tools, monkeypatch):
    path, view, update, updates = config_tools
    path.parent.mkdir()
    path.write_text('{"mcpServers": {}}')

    def fail_replace(*_args: object):
        msg = "private error"
        raise OSError(msg)

    monkeypatch.setattr(type(path), "replace", fail_replace)
    result = update.invoke(
        {
            "server_name": "example",
            "server": {"command": "server"},
            "expected_revision": view.invoke({})["revision"],
        }
    )
    assert result["status"] == "error"
    assert "private" not in json.dumps(result)
    assert path.read_text() == '{"mcpServers": {}}'
    assert not list(path.parent.glob(".mcp-*"))
    assert updates == []


def test_concurrent_updates_do_not_overwrite_each_other(config_tools):
    path, view, update, updates = config_tools
    revision = view.invoke({})["revision"]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                update.invoke,
                [
                    {
                        "server_name": name,
                        "server": {"command": "server"},
                        "expected_revision": revision,
                    }
                    for name in ("one", "two")
                ],
            )
        )
    assert sorted(result["status"] for result in results) == ["conflict", "updated"]
    assert len(json.loads(path.read_text())["mcpServers"]) == 1
    assert updates == [True]


async def test_provider_keeps_management_tools_after_refresh(tmp_path):
    config = TalonConfig(
        "test", tmp_path, env={"DEEPAGENTS_TALON_MCP_CONFIG": str(tmp_path / "mcp")}
    )
    provider = MCPToolProvider(config)
    tools = {tool.name: tool for tool in (await provider.load()).tools}
    revision = tools["get_mcp_configuration"].invoke({})["revision"]
    result = tools["update_mcp_server"].invoke(
        {
            "server_name": "absent",
            "server": None,
            "expected_revision": revision,
        }
    )
    assert result["status"] == "updated"
    refreshed = await provider.refresh_if_needed()
    assert refreshed is not None
    assert {tool.name for tool in refreshed} == tools.keys()
    assert await provider.refresh_if_needed() is None


class ToolCallingModel(FakeMessagesListChatModel):
    def bind_tools(self, *_args: object, **_kwargs: object) -> Self:
        return self


@pytest.mark.parametrize("after_reload", [False, True])
@pytest.mark.parametrize(
    ("decision", "auto_approve", "trigger", "writes"),
    [
        ("approve", None, "channel", True),
        ("reject", None, "channel", False),
        (None, None, "channel", False),
        ("approve", None, "cron", False),
        (None, "true", "channel", True),
        (None, "false", "channel", False),
        (None, "typo", "channel", False),
    ],
)
async def test_runtime_gates_real_config_writes(  # noqa: PLR0913  # Independent approval/reload cases.
    config_tools, decision, auto_approve, trigger, writes, after_reload
):
    path, view, update, _ = config_tools
    arguments = {
        "server_name": "example",
        "server": {"command": "server"},
        "expected_revision": view.invoke({})["revision"],
    }
    model = ToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": "update_mcp_server", "args": arguments, "id": "call"}],
            ),
            AIMessage(content="done"),
        ]
    )
    approvals: list[ToolApprovalRequest] = []

    async def approve(request: ToolApprovalRequest) -> ToolApprovalDecision:
        assert not path.exists()
        approvals.append(request)
        return decision

    async def reload_tools():
        return [view, update]

    runtime = DeepAgentRuntime(
        model=model,
        tools=[] if after_reload else [view, update],
        reload_tools=reload_tools,
        backend=StateBackend(),
        env={} if auto_approve is None else {MCP_CONFIG_AUTO_APPROVE_ENV: auto_approve},
        interrupt_on={"update_mcp_server": False},
        include_web_tools=False,
        skills=(),
        memory=(),
    )
    await runtime.start()
    try:
        if after_reload:
            await runtime.reload_mcp_configuration()
        await runtime.invoke(
            AgentRequest(
                conversation_id="chat",
                text="configure MCP",
                metadata={"trigger": trigger},
                approval_handler=approve if decision is not None else None,
            )
        )
    finally:
        await runtime.stop()
    assert path.exists() is writes
    assert bool(approvals) is (decision is not None and trigger != "cron")


async def test_pending_approval_rejects_concurrent_edit(config_tools):
    path, view, update, _ = config_tools
    arguments = {
        "server_name": "example",
        "server": {"command": "server"},
        "expected_revision": view.invoke({})["revision"],
    }
    model = ToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": "update_mcp_server", "args": arguments, "id": "call"}],
            ),
            AIMessage(content="done"),
        ]
    )
    waiting, resume = asyncio.Event(), asyncio.Event()

    async def approve(_request: ToolApprovalRequest) -> ToolApprovalDecision:
        waiting.set()
        await resume.wait()
        return "approve"

    runtime = DeepAgentRuntime(
        model=model,
        tools=[view, update],
        backend=StateBackend(),
        env={},
        include_web_tools=False,
        skills=(),
        memory=(),
    )
    await runtime.start()
    task = asyncio.create_task(
        runtime.invoke(
            AgentRequest(
                conversation_id="chat",
                text="configure MCP",
                approval_handler=approve,
            )
        )
    )
    try:
        await asyncio.wait_for(waiting.wait(), timeout=5)
        path.parent.mkdir()
        path.write_text('{"mcpServers": {}, "operator": true}')
        resume.set()
        await task
        assert json.loads(path.read_text()) == {"mcpServers": {}, "operator": True}
    finally:
        resume.set()
        await task
        await runtime.stop()
