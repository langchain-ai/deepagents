from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeAlias

import pytest

from deepagents_code.mcp_tools import MCPServerInfo as CodeMCPServerInfo, MCPToolInfo
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


FakeCodeLoaderResult: TypeAlias = tuple[list[DummyTool], None, list[CodeMCPServerInfo]]


async def _fake_code_loader(data: dict[str, Any]) -> FakeCodeLoaderResult:
    tools = [
        DummyTool("files_read"),
        DummyTool("files_write"),
        DummyTool("search"),
    ]
    infos = [
        CodeMCPServerInfo(
            name=name,
            transport=str(server.get("type") or server.get("transport") or "stdio"),
            tools=tuple(MCPToolInfo(name=tool.name, description="") for tool in tools),
        )
        for name, server in data["mcpServers"].items()
        if isinstance(server, dict)
    ]
    return tools, None, infos


async def test_load_mcp_tools_from_config_delegates_to_code_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[dict[str, Any]] = []

    async def fake_loader(data: dict[str, Any]) -> FakeCodeLoaderResult:
        seen.append(data)
        return await _fake_code_loader(data)

    monkeypatch.setattr("deepagents_talon.mcp.get_mcp_tools_from_config", fake_loader)

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

    assert seen[0]["mcpServers"]["files"]["allowedTools"] == ["read", "search"]
    assert [tool.name for tool in result.tools] == ["files_read", "files_write", "search"]
    assert result.servers[0].tool_count == 3


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
    seen: list[dict[str, Any]] = []

    async def fake_loader(data: dict[str, Any]) -> FakeCodeLoaderResult:
        seen.append(data)
        return await _fake_code_loader(data)

    monkeypatch.setattr("deepagents_talon.mcp.get_mcp_tools_from_config", fake_loader)
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
    assert seen[0]["mcpServers"]["remote"]["headers"] == {
        "Authorization": "Bearer ${TOKEN}",
    }
