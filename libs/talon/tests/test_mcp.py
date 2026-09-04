from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Self

import pytest
from mcp.client.auth import OAuthFlowError

from deepagents_talon.config import TalonConfig
from deepagents_talon.mcp import (
    MCPConfigError,
    load_mcp_tools,
    login_mcp_server,
    mcp_config_path,
)

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


async def test_tool_allowlist_filters_loaded_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class MultipleToolClient(FakeMCPClient):
        async def get_tools(self, *, server_name: str | None = None) -> list[DummyTool]:
            assert server_name is not None
            return [DummyTool(f"{server_name}_read"), DummyTool(f"{server_name}_write")]

    config_path = tmp_path / "custom.mcp.json"
    _write_config(
        config_path,
        {
            "remote": {
                "url": "https://example.com/mcp",
                "allowedTools": ["read"],
            }
        },
    )
    monkeypatch.setattr("deepagents_talon.mcp.MultiServerMCPClient", MultipleToolClient)

    result = await load_mcp_tools(
        _config(tmp_path, {"DEEPAGENTS_TALON_MCP_CONFIG": str(config_path)})
    )

    assert [tool.name for tool in result.tools] == ["remote_read"]
    assert [tool.name for tool in result.servers[0].tools] == ["remote_read"]


async def test_oauth_connection_uses_stored_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = object()

    class FakeStorage:
        def __init__(self, server_name: str, *, server_url: str) -> None:
            assert (server_name, server_url) == ("remote", "https://example.com/mcp")

        async def get_tokens(self) -> object:
            return object()

    monkeypatch.setattr("deepagents_talon.mcp.FileTokenStorage", FakeStorage)
    monkeypatch.setattr("deepagents_talon.mcp.build_oauth_provider", lambda **_kwargs: provider)
    monkeypatch.setattr("deepagents_talon.mcp.MultiServerMCPClient", FakeMCPClient)
    config_path = tmp_path / "custom.mcp.json"
    _write_config(
        config_path,
        {"remote": {"url": "https://example.com/mcp", "auth": "oauth"}},
    )

    await load_mcp_tools(_config(tmp_path, {"DEEPAGENTS_TALON_MCP_CONFIG": str(config_path)}))

    connection = FakeMCPClient.calls[-1]["connections"]
    assert isinstance(connection, dict)
    assert connection["remote"]["auth"] is provider


async def test_login_uses_talon_config_and_interactive_oauth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "custom.mcp.json"
    _write_config(
        config_path,
        {"remote": {"url": "https://example.com/mcp", "auth": "oauth"}},
    )
    provider = object()
    calls: list[dict[str, object]] = []

    class LoginClient:
        def __init__(self, connections: dict[str, object]) -> None:
            calls.append(connections)

        def session(self, server_name: str):
            assert server_name == "remote"

            class Session:
                async def __aenter__(self) -> Self:
                    return self

                async def __aexit__(self, *_args: object) -> None:
                    return None

            return Session()

    monkeypatch.setattr("deepagents_talon.mcp.MultiServerMCPClient", LoginClient)
    monkeypatch.setattr("deepagents_talon.mcp._MCP_LOAD_TIMEOUT_SECONDS", 1)
    monkeypatch.setattr("deepagents_talon.mcp.FileTokenStorage", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        "deepagents_talon.mcp.build_oauth_provider",
        lambda **kwargs: provider if kwargs["interactive"] else None,
    )

    result = await login_mcp_server(_config(tmp_path), "remote", str(config_path))

    assert result == 0
    assert calls == [
        {
            "remote": {
                "transport": "streamable_http",
                "url": "https://example.com/mcp",
                "timeout": 30.0,
                "auth": provider,
            }
        }
    ]


async def test_login_reports_oauth_failure_without_details(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = tmp_path / "custom.mcp.json"
    _write_config(config_path, {"remote": {"url": "https://example.com/mcp"}})

    failure = OAuthFlowError("secret token exchange response")

    async def fail_login(*_args: object) -> None:
        raise failure

    monkeypatch.setattr("deepagents_talon.mcp._open_mcp_session", fail_login)
    monkeypatch.setattr("deepagents_talon.mcp.FileTokenStorage", lambda *_args, **_kwargs: object())
    monkeypatch.setattr("deepagents_talon.mcp.build_oauth_provider", lambda **_kwargs: object())

    result = await login_mcp_server(_config(tmp_path), "remote", str(config_path))

    assert result == 1
    assert capsys.readouterr().err == "MCP login failed: OAuthFlowError\n"


async def test_login_does_not_timeout_interactive_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "custom.mcp.json"
    _write_config(config_path, {"remote": {"url": "https://example.com/mcp"}})

    async def slow_login(*_args: object) -> None:
        await asyncio.sleep(0.02)

    monkeypatch.setattr("deepagents_talon.mcp._open_mcp_session", slow_login)
    monkeypatch.setattr("deepagents_talon.mcp._MCP_LOAD_TIMEOUT_SECONDS", 0.001)
    monkeypatch.setattr("deepagents_talon.mcp.FileTokenStorage", lambda *_args, **_kwargs: object())
    monkeypatch.setattr("deepagents_talon.mcp.build_oauth_provider", lambda **_kwargs: object())

    assert await login_mcp_server(_config(tmp_path), "remote", str(config_path)) == 0


async def test_login_reports_missing_server_without_deepagents_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = tmp_path / "custom.mcp.json"
    _write_config(config_path, {})

    result = await login_mcp_server(_config(tmp_path), "missing", str(config_path))

    assert result == 1
    assert "was not found" in capsys.readouterr().err


async def test_invalid_stdio_environment_does_not_block_valid_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "custom.mcp.json"
    _write_config(
        config_path,
        {
            "unsafe": {"command": "x", "env": {"LD_PRELOAD": "x"}},
            "valid": {"url": "https://example.com/mcp"},
        },
    )
    monkeypatch.setattr("deepagents_talon.mcp.MultiServerMCPClient", FakeMCPClient)

    result = await load_mcp_tools(
        _config(tmp_path, {"DEEPAGENTS_TALON_MCP_CONFIG": str(config_path)})
    )

    assert [tool.name for tool in result.tools] == ["valid_read"]
    assert result.servers[0].error == "MCP stdio server 'unsafe' cannot set LD_PRELOAD"


async def test_invalid_server_does_not_block_valid_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "custom.mcp.json"
    _write_config(
        config_path,
        {
            "invalid": {"transport": "websocket", "url": "https://example.com"},
            "valid": {"url": "https://example.com/mcp"},
        },
    )
    monkeypatch.setattr("deepagents_talon.mcp.MultiServerMCPClient", FakeMCPClient)

    result = await load_mcp_tools(
        _config(tmp_path, {"DEEPAGENTS_TALON_MCP_CONFIG": str(config_path)})
    )

    assert [tool.name for tool in result.tools] == ["valid_read"]
    assert result.servers[0].status == "error"
    assert result.servers[1].status == "ok"
