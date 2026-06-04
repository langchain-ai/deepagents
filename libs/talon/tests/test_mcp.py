from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import pytest

from deepagents_talon.config import TalonConfig
from deepagents_talon.mcp import (
    MCPConfigError,
    load_mcp_tools,
    load_mcp_tools_from_config,
    write_mcp_server_config,
)

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class DummyTool:
    name: str


class DummyMCPClient:
    connections: ClassVar[list[dict[str, object]]] = []

    def __init__(self, *, connections: dict[str, dict[str, object]]) -> None:
        self.connections.append(connections)

    async def get_tools(self) -> list[DummyTool]:
        return [
            DummyTool("files_read"),
            DummyTool("files_write"),
            DummyTool("search"),
        ]


async def test_load_mcp_tools_filters_allowed_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("deepagents_talon.mcp.MultiServerMCPClient", DummyMCPClient)

    result = await load_mcp_tools_from_config(
        {
            "mcpServers": {
                "files": {
                    "type": "stdio",
                    "command": "server",
                    "allowedTools": ["read", "search"],
                },
            },
        },
    )

    assert [tool.name for tool in result.tools] == ["files_read", "search"]
    assert result.servers[0].tool_count == 2
    assert DummyMCPClient.connections[-1]["files"]["transport"] == "stdio"


async def test_load_mcp_tools_filters_disabled_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("deepagents_talon.mcp.MultiServerMCPClient", DummyMCPClient)

    result = await load_mcp_tools_from_config(
        {
            "mcpServers": {
                "files": {
                    "type": "http",
                    "url": "https://tools.example/mcp",
                    "disabledTools": ["write"],
                },
            },
        },
    )

    assert [tool.name for tool in result.tools] == ["files_read", "search"]
    assert DummyMCPClient.connections[-1]["files"]["transport"] == "streamable_http"


def test_write_mcp_server_config_creates_config(tmp_path: Path) -> None:
    path = tmp_path / "tools.json"

    write_mcp_server_config(
        path=path,
        name="linear",
        server={
            "type": "http",
            "url": "https://mcp.example/mcp",
            "headers": {"Authorization": "Bearer ${LINEAR_TOKEN}"},
        },
    )

    data = json.loads(path.read_text())
    assert data["mcpServers"]["linear"]["url"] == "https://mcp.example/mcp"


def test_write_mcp_server_config_rejects_both_filters(tmp_path: Path) -> None:
    with pytest.raises(MCPConfigError):
        write_mcp_server_config(
            path=tmp_path / "tools.json",
            name="bad",
            server={
                "type": "stdio",
                "command": "server",
                "allowedTools": ["read"],
                "disabledTools": ["write"],
            },
        )


async def test_load_mcp_tools_reads_manifest_and_env_headers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("deepagents_talon.mcp.MultiServerMCPClient", DummyMCPClient)
    monkeypatch.setenv("TOKEN", "secret")
    config = TalonConfig.from_env({"AGENT_ASSISTANT_ID": "test"}, base_home=tmp_path)
    config.ensure_home()
    (config.manifest_dir / "tools.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "remote": {
                        "transport": "sse",
                        "url": "https://tools.example/sse",
                        "headers": {"Authorization": "Bearer ${TOKEN}"},
                    },
                },
            },
        ),
    )

    result = await load_mcp_tools(config)

    assert [tool.name for tool in result.tools] == ["files_read", "files_write", "search"]
    assert DummyMCPClient.connections[-1]["remote"]["headers"] == {
        "Authorization": "Bearer secret",
    }
