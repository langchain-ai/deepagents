from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias

from deepagents_code.mcp_tools import MCPServerInfo as CodeMCPServerInfo, MCPToolInfo
from deepagents_talon.config import TalonConfig
from deepagents_talon.mcp import discover_mcp_config_paths, load_mcp_tools

if TYPE_CHECKING:
    import pytest


@dataclass(frozen=True)
class DummyTool:
    name: str


FakeCodeLoaderResult: TypeAlias = tuple[list[DummyTool], None, list[CodeMCPServerInfo]]


def _fake_code_loader(path: str) -> FakeCodeLoaderResult:
    data = json.loads(Path(path).read_text())
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


async def test_load_mcp_tools_reads_manifest_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[Path] = []

    async def fake_loader(path: str) -> FakeCodeLoaderResult:
        seen.append(Path(path))
        return _fake_code_loader(path)

    monkeypatch.setattr("deepagents_talon.mcp.get_mcp_tools", fake_loader)
    config = TalonConfig.from_env({"AGENT_ASSISTANT_ID": "test"}, base_home=tmp_path)
    config.ensure_home()
    tools_path = config.manifest_dir / ".mcp.json"
    tools_path.write_text(
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
    assert seen == [tools_path]


async def test_load_mcp_tools_prefers_env_config_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[Path] = []

    async def fake_loader(path: str) -> FakeCodeLoaderResult:
        seen.append(Path(path))
        return _fake_code_loader(path)

    monkeypatch.setattr("deepagents_talon.mcp.get_mcp_tools", fake_loader)
    env_path = tmp_path / "custom-tools.json"
    env_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "custom": {
                        "type": "stdio",
                        "command": "server",
                    },
                },
            },
        ),
    )
    config = TalonConfig.from_env(
        {
            "AGENT_ASSISTANT_ID": "test",
            "DEEPAGENTS_TALON_MCP_CONFIG": str(env_path),
        },
        base_home=tmp_path,
    )
    config.ensure_home()
    (config.manifest_dir / ".mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "manifest": {
                        "type": "stdio",
                        "command": "server",
                    },
                },
            },
        ),
    )

    await load_mcp_tools(config)

    assert seen == [env_path]


def test_discover_mcp_config_paths_ignores_legacy_tools_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    config = TalonConfig.from_env({"AGENT_ASSISTANT_ID": "test"}, base_home=tmp_path)
    config.ensure_home()
    (config.manifest_dir / "tools.json").write_text("{}", encoding="utf-8")
    config_path = config.manifest_dir / ".mcp.json"
    config_path.write_text('{"mcpServers": {}}', encoding="utf-8")

    assert discover_mcp_config_paths(config) == [config_path]
