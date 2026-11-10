"""Exercise persisted MCP enablement and tool unloading."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from fastmcp import FastMCP
from fastmcp.client.transports import FastMCPTransport

from deepagents_talon import mcp
from deepagents_talon.config import TalonConfig

if TYPE_CHECKING:
    from pathlib import Path


async def test_disabled_servers_skip_environment_connections_and_authentication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "mcp.json"
    path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "remote": {"url": "${UNSET_MCP_URL}", "auth": "oauth", "enabled": False},
                    "local": {"command": "${UNSET_MCP_COMMAND}", "enabled": False},
                }
            }
        )
    )
    monkeypatch.delenv("UNSET_MCP_URL", raising=False)
    monkeypatch.delenv("UNSET_MCP_COMMAND", raising=False)

    async def unexpected_connection(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Disabled MCP servers must not connect")

    monkeypatch.setattr(mcp, "_connection", unexpected_connection)
    config = TalonConfig.from_env({"DEEPAGENTS_TALON_MCP_CONFIG": str(path)}, base_home=tmp_path)
    provider = mcp.MCPToolProvider(config)
    loaded = await provider.load()
    assert all(server.status == "disabled" and not server.tools for server in loaded.servers)
    tools = {tool.name: tool for tool in loaded.tools}
    assert not {"remote_read", "local_read", "authenticate_mcp_server"} & tools.keys()
    assert tools["get_mcp_server_status"].invoke({}) == (
        {"server_name": "remote", "status": "disabled", "can_authenticate": False},
        {"server_name": "local", "status": "disabled", "can_authenticate": False},
    )
    provider._oauth_servers = frozenset({"remote"})
    assert (await provider._authenticate("remote", "stale-turn"))["status"] == "failed"
    assert await mcp.login_mcp_server(config, "remote") == 1
    assert "is disabled" in capsys.readouterr().err


async def test_configuration_toggle_unloads_and_restores_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = FastMCP("records")

    @server.tool
    def read() -> str:
        return "records"

    async def connection(*_args: object) -> tuple[FastMCPTransport, str]:
        return FastMCPTransport(server), "stdio"

    monkeypatch.setattr(mcp, "_connection", connection)
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps({"mcpServers": {"records": {"command": "${MCP_COMMAND}"}}}))
    monkeypatch.setenv("MCP_COMMAND", "unused")
    config = TalonConfig.from_env({"DEEPAGENTS_TALON_MCP_CONFIG": str(path)}, base_home=tmp_path)
    provider = mcp.MCPToolProvider(config)
    tools = {tool.name: tool for tool in (await provider.load()).tools}
    assert "records_read" in tools
    for enabled in (False, True):
        view = tools["get_mcp_configuration"].invoke({})
        settings = view["mcpServers"]["records"] | {"enabled": enabled}
        assert (
            tools["update_mcp_server"].invoke(
                {
                    "server_name": "records",
                    "server": settings,
                    "expected_revision": view["revision"],
                }
            )["status"]
            == "updated"
        )
        refreshed = await provider.refresh_if_needed()
        assert refreshed is not None
        tools = {tool.name: tool for tool in refreshed}
        assert ("records_read" in tools) is enabled
        assert tools["get_mcp_server_status"].invoke({})[0]["status"] == (
            "ok" if enabled else "disabled"
        )
        assert json.loads(path.read_text())["mcpServers"]["records"] == {
            "command": "${MCP_COMMAND}",
            "enabled": enabled,
        }
        assert await provider.refresh_if_needed() is None


@pytest.mark.parametrize("value", ["false", 0, 1, None, [], {}])
def test_enabled_requires_a_boolean(tmp_path: Path, value: object) -> None:
    path = tmp_path / "mcp.json"
    path.write_text(
        json.dumps({"mcpServers": {"records": {"command": "unused", "enabled": value}}})
    )
    with pytest.raises(mcp.MCPConfigError, match="enabled must be a boolean"):
        mcp._load_config(path, {})
