from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeAlias

from deepagents_code.mcp_tools import MCPServerInfo as CodeMCPServerInfo, MCPToolInfo
from deepagents_talon.config import TalonConfig
from deepagents_talon.mcp import load_mcp_tools, load_mcp_tools_from_config

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


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
    assert len(result.servers[0].tools) == 3


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
