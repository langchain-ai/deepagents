from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

from deepagents_code.mcp_tools import MCPServerInfo as CodeMCPServerInfo, MCPToolInfo
from deepagents_code.project_utils import ProjectContext
from deepagents_talon.config import TalonConfig
from deepagents_talon.mcp import load_mcp_tools

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


@dataclass(frozen=True)
class DummyTool:
    name: str


FakeCodeLoaderResult: TypeAlias = tuple[list[DummyTool], None, list[CodeMCPServerInfo]]


async def test_load_mcp_tools_delegates_to_deepagents_code_discovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    env_path = tmp_path / "custom.mcp.json"
    calls: list[dict[str, object]] = []
    tools = [DummyTool("files_read")]
    infos = [
        CodeMCPServerInfo(
            name="remote",
            transport="streamable_http",
            tools=(MCPToolInfo(name="files_read", description=""),),
        )
    ]

    async def fake_resolver(**kwargs: object) -> FakeCodeLoaderResult:
        calls.append(kwargs)
        return tools, None, infos

    monkeypatch.setattr("deepagents_talon.mcp.resolve_and_load_mcp_tools", fake_resolver)
    config = TalonConfig.from_env(
        {
            "AGENT_ASSISTANT_ID": "test",
            "DEEPAGENTS_TALON_WORKSPACE": str(workspace),
            "DEEPAGENTS_TALON_MCP_CONFIG": str(env_path),
        },
        base_home=tmp_path,
    )

    result = await load_mcp_tools(config)

    assert result.tools == tuple(tools)
    assert result.servers == tuple(infos)
    assert len(calls) == 1
    assert calls[0]["explicit_config_path"] == str(env_path)
    assert calls[0]["trust_project_mcp"] is None
    project_context = calls[0]["project_context"]
    assert isinstance(project_context, ProjectContext)
    assert project_context.user_cwd == workspace.resolve()
