from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import pytest

from deepagents_talon.config import TalonConfig
from deepagents_talon.mcp import MCPConfigError, load_mcp_tools, mcp_config_path

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class DummyTool:
    name: str
    description: str = ""


class FakeMCPClient:
    calls: ClassVar[list[dict[str, object]]] = []

    def __init__(self, connections: dict[str, object], **kwargs: object) -> None:
        self.connections = connections
        self.calls.append({"connections": connections, **kwargs})

    async def get_tools(self, *, server_name: str | None = None) -> list[DummyTool]:
        assert server_name is not None
        return [DummyTool(f"{server_name}_read", "Read files")]


def _write_config(path: Path, servers: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"mcpServers": servers}), encoding="utf-8")


def _config(tmp_path: Path, env: dict[str, str] | None = None) -> TalonConfig:
    return TalonConfig.from_env(
        {
            "AGENT_ASSISTANT_ID": "test",
            "DEEPAGENTS_TALON_WORKSPACE": str(tmp_path / "workspace"),
            **(env or {}),
        },
        base_home=tmp_path,
    )


def test_mcp_config_path_uses_standard_path_or_env_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    monkeypatch.setattr("deepagents_talon.mcp.Path.home", lambda: home)

    assert mcp_config_path(_config(tmp_path)) == home / ".deepagents" / ".mcp.json"

    custom = tmp_path / "custom.mcp.json"
    assert (
        mcp_config_path(_config(tmp_path, {"DEEPAGENTS_TALON_MCP_CONFIG": str(custom)})) == custom
    )


async def test_load_mcp_tools_uses_standard_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeMCPClient.calls.clear()
    home = tmp_path / "home"
    monkeypatch.setattr("deepagents_talon.mcp.Path.home", lambda: home)
    monkeypatch.setattr("deepagents_talon.mcp.MultiServerMCPClient", FakeMCPClient)
    _write_config(
        home / ".deepagents" / ".mcp.json",
        {"remote": {"type": "http", "url": "https://example.com/mcp"}},
    )
    result = await load_mcp_tools(_config(tmp_path))

    assert [tool.name for tool in result.tools] == ["remote_read"]
    assert [server.name for server in result.servers] == ["remote"]
    assert FakeMCPClient.calls[0]["connections"] == {
        "remote": {
            "transport": "streamable_http",
            "url": "https://example.com/mcp",
            "timeout": 30.0,
        }
    }


async def test_explicit_config_interpolates_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeMCPClient.calls.clear()
    monkeypatch.setattr("deepagents_talon.mcp.MultiServerMCPClient", FakeMCPClient)
    config_path = tmp_path / "custom.mcp.json"
    _write_config(
        config_path,
        {
            "remote": {
                "type": "sse",
                "url": "${MCP_URL}",
                "headers": {"Authorization": "Bearer ${MCP_TOKEN}"},
            }
        },
    )

    monkeypatch.setenv("MCP_URL", "https://example.com/sse")
    monkeypatch.setenv("MCP_TOKEN", "secret")

    await load_mcp_tools(_config(tmp_path, {"DEEPAGENTS_TALON_MCP_CONFIG": str(config_path)}))

    assert FakeMCPClient.calls[0]["connections"] == {
        "remote": {
            "transport": "sse",
            "url": "https://example.com/sse",
            "timeout": 30.0,
            "headers": {"Authorization": "Bearer secret"},
        }
    }


@pytest.mark.parametrize(
    ("document", "match"),
    [
        ([], "must contain a JSON object"),
        ({}, "must contain an mcpServers object"),
        ({"mcpServers": {"bad/name": {"command": "x"}}}, "server name"),
        (
            {"mcpServers": {"unsafe": {"command": "x", "env": {"LD_PRELOAD": "x"}}}},
            "cannot set LD_PRELOAD",
        ),
    ],
)
async def test_invalid_config_fails_before_connecting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    document: object,
    match: str,
) -> None:
    FakeMCPClient.calls.clear()
    monkeypatch.setattr("deepagents_talon.mcp.MultiServerMCPClient", FakeMCPClient)
    config_path = tmp_path / "invalid.mcp.json"
    config_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(MCPConfigError, match=match):
        await load_mcp_tools(_config(tmp_path, {"DEEPAGENTS_TALON_MCP_CONFIG": str(config_path)}))

    assert FakeMCPClient.calls == []


async def test_server_connection_error_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FailingMCPClient(FakeMCPClient):
        async def get_tools(self, *, server_name: str | None = None) -> list[DummyTool]:
            assert server_name is not None
            msg = "connection failed"
            raise RuntimeError(msg)

    config_path = tmp_path / "custom.mcp.json"
    _write_config(config_path, {"remote": {"url": "https://example.com/mcp"}})
    monkeypatch.setattr("deepagents_talon.mcp.MultiServerMCPClient", FailingMCPClient)

    result = await load_mcp_tools(
        _config(tmp_path, {"DEEPAGENTS_TALON_MCP_CONFIG": str(config_path)})
    )

    assert result.tools == ()
    assert result.servers[0].status == "error"
    assert result.servers[0].error == "connection failed"
