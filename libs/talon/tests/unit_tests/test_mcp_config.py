from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Self

import pytest
from deepagents.backends import StateBackend
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from deepagents_talon.interfaces import AgentRequest
from deepagents_talon.mcp_config import (
    MCP_CONFIG_AUTO_APPROVE_ENV,
    WORKSPACE_ENV,
    MCPConfigStore,
    agent_workspace_root,
)
from deepagents_talon.runtime import DeepAgentRuntime

if TYPE_CHECKING:
    from deepagents_talon.interfaces import ToolApprovalDecision, ToolApprovalRequest


@pytest.fixture
def config_tools(tmp_path: Path):
    path = tmp_path / "private" / ".mcp.json"
    updates: list[bool] = []
    store = MCPConfigStore(path, lambda: updates.append(True))
    return path, *store.tools(), updates


def test_redacted_round_trip_preserves_secrets_and_other_settings(config_tools, monkeypatch):
    path, view, update, updates = config_tools
    path.parent.mkdir()
    original = {
        "mcpServers": {
            "example": {
                "url": "https://user:private@example.test/mcp?token=private",
                "headers": {"Authorization": "Bearer private", "X-Key": "${CREDENTIAL}"},
                "env": {"KEY": "${CREDENTIAL:-private}"},
                "args": ["private"],
                "command": "private",
                "transport": "http",
            },
            "other": {"command": "server"},
        },
        "metadata": "private",
    }
    path.write_text(json.dumps(original))
    monkeypatch.setenv("CREDENTIAL", "expanded-private")
    result = view.invoke({})
    assert "private" not in json.dumps(result)
    server = result["mcpServers"]["example"]
    assert server["headers"]["X-Key"] == "${CREDENTIAL}"
    assert server["transport"] == "http"
    server["allowedTools"] = ["read_*"]
    response = update.invoke(
        {"server_name": "example", "server": server, "expected_revision": result["revision"]}
    )
    assert response == {"status": "updated", "available": "after_successful_reload"}
    original["mcpServers"]["example"]["allowedTools"] = ["read_*"]
    assert json.loads(path.read_text()) == original
    assert path.stat().st_mode & 0o777 == 0o600
    assert updates == [True]


def test_add_remove_and_stale_revision(config_tools):
    path, view, update, updates = config_tools
    revision = view.invoke({})["revision"]
    arguments = {
        "server_name": "example",
        "server": {"command": "server"},
        "expected_revision": revision,
    }
    assert update.invoke(arguments)["status"] == "updated"
    revision = view.invoke({})["revision"]
    path.write_text('{"mcpServers": {"operator": {"command": "server"}}}')
    assert update.invoke({**arguments, "expected_revision": revision})["status"] == "conflict"
    assert "example" not in json.loads(path.read_text())["mcpServers"]
    result = update.invoke(
        {
            "server_name": "operator",
            "server": None,
            "expected_revision": view.invoke({})["revision"],
        }
    )
    assert result["status"] == "updated"
    assert json.loads(path.read_text()) == {"mcpServers": {}}
    assert updates == [True, True]


@pytest.mark.parametrize(
    "server",
    [
        {"url": "<redacted>"},
        {"url": "https://example.test", "auth": "oauth", "headers": {"Authorization": "value"}},
        {"url": "https://example.test", "unknown": "value"},
        {"transport": []},
        {"command": "${INVALID"},
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


class ToolCallingModel(FakeMessagesListChatModel):
    def bind_tools(self, *_args: object, **_kwargs: object) -> Self:
        return self


@pytest.mark.parametrize(
    ("decision", "auto_approve", "trigger", "writes"),
    [
        ("approve", None, "channel", True),
        ("reject", None, "channel", False),
        (None, None, "channel", False),
        ("approve", None, "cron", False),
        (None, "true", "channel", True),
        (None, "typo", "channel", False),
    ],
)
async def test_runtime_gates_real_config_writes(
    config_tools, decision, auto_approve, trigger, writes
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
        tools=[],
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


def test_config_store_warns_about_a_path_inside_the_agent_workspace(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The docstring invariant is reported; Round 2 turns it into an error."""
    workspace = tmp_path / "workspace"
    path = workspace / "nested" / ".mcp.json"

    with caplog.at_level(logging.WARNING, logger="deepagents_talon.mcp_config"):
        store = MCPConfigStore(path, lambda: None, agent_root=workspace)

    assert store._path == path
    assert "MCP configuration" in caplog.text
    assert str(workspace) in caplog.text
    assert WORKSPACE_ENV in caplog.text


def test_config_store_works_when_talon_runs_from_the_home_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression guard: the default config path sits under CWD when launched from $HOME."""
    monkeypatch.delenv(WORKSPACE_ENV, raising=False)
    monkeypatch.chdir(tmp_path)

    view, _update = MCPConfigStore(tmp_path / ".deepagents" / ".mcp.json", lambda: None).tools()

    assert view.invoke({})["mcpServers"] == {}


def test_agent_workspace_root_prefers_the_configured_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(WORKSPACE_ENV, raising=False)

    assert agent_workspace_root({WORKSPACE_ENV: str(tmp_path)}) == tmp_path.resolve()
    assert agent_workspace_root({}) == Path.cwd().resolve()
