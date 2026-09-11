"""Optional MCP dependencies do not prevent using the base package."""

from importlib.metadata import PackageNotFoundError

import pytest

from deepagents_code import mcp_tools
from deepagents_code.client.commands.mcp import run_mcp_login, run_mcp_login_list


@pytest.fixture(params=[None, "2.1.1"])
def unavailable_mcp(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = mcp_tools.metadata.version

    def version(name: str) -> str:
        if name == "mcp":
            if request.param is None:
                raise PackageNotFoundError(name)
            return request.param
        return original(name)

    monkeypatch.setattr(mcp_tools.metadata, "version", version)


@pytest.mark.usefixtures("unavailable_mcp")
async def test_loader_reports_missing_extra_without_connecting() -> None:
    tools, manager, servers = await mcp_tools._load_tools_from_config(
        {"mcpServers": {"docs": {"url": "https://example.com/mcp"}}}
    )

    assert tools == []
    assert manager is None
    assert len(servers) == 1
    assert servers[0].name == "docs"
    assert servers[0].status == "error"
    assert servers[0].error is not None
    assert "deepagents-code[mcp]" in servers[0].error


@pytest.mark.usefixtures("unavailable_mcp")
async def test_login_reports_missing_extra(capsys: pytest.CaptureFixture[str]) -> None:
    assert await run_mcp_login(server="docs", config_path=None) == 1
    assert "deepagents-code[mcp]" in capsys.readouterr().err
    assert await run_mcp_login_list(config_path=None) == 1
    assert "deepagents-code[mcp]" in capsys.readouterr().err
