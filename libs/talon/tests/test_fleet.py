from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from deepagents_talon.__main__ import _agent_runtime
from deepagents_talon.config import TalonConfig
from deepagents_talon.cron import CronJobStore
from deepagents_talon.fleet import FleetAgentComponents, load_fleet_agent_components
from deepagents_talon.mcp import MCPTools
from deepagents_talon.runtime import DeepAgentRuntime

if TYPE_CHECKING:
    import pytest


def fleet_tool() -> str:
    """Fleet tool stub."""
    return "fleet"


def local_tool() -> str:
    """Local MCP tool stub."""
    return "local"


async def test_load_fleet_agent_components_coerces_public_loader_payload(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LANGSMITH_TENANT_ID", raising=False)
    subagent = {
        "name": "researcher",
        "description": "Research tasks",
        "system_prompt": "Research carefully.",
    }

    async def fake_load_agent_components(path):
        assert path == tmp_path
        assert os.environ["LANGSMITH_TENANT_ID"] == "tenant"
        return {
            "model": "fleet:model",
            "system_prompt": "fleet prompt",
            "tools": [fleet_tool],
            "subagents": [subagent],
            "interrupt_on": {"fleet_tool": True},
        }

    monkeypatch.setattr(
        "deepagents_talon.fleet.load_agent_components",
        fake_load_agent_components,
    )

    components = await load_fleet_agent_components(
        tmp_path,
        env={"LANGSMITH_TENANT_ID": "tenant"},
    )

    assert components.model == "fleet:model"
    assert components.system_prompt == "fleet prompt"
    assert components.tools == (fleet_tool,)
    assert components.subagents == (subagent,)
    assert components.interrupt_on == {"fleet_tool": True}
    assert "LANGSMITH_TENANT_ID" not in os.environ


async def test_agent_runtime_loads_fleet_components(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fleet_dir = tmp_path / "fleet"
    fleet_dir.mkdir()
    subagent = {
        "name": "researcher",
        "description": "Research tasks",
        "system_prompt": "Research carefully.",
    }
    seen: dict[str, Any] = {}

    async def fake_load_fleet(
        path,
        *,
        env,
    ) -> FleetAgentComponents:
        seen["path"] = path
        seen["env"] = env
        return FleetAgentComponents(
            model="fleet:model",
            system_prompt="fleet prompt",
            tools=(fleet_tool,),
            subagents=(subagent,),
            interrupt_on={"fleet_tool": True},
        )

    async def fail_load_mcp(_config):
        msg = "local MCP loader should not run for Fleet sources"
        raise AssertionError(msg)

    monkeypatch.setattr("deepagents_talon.__main__.load_fleet_agent_components", fake_load_fleet)
    monkeypatch.setattr("deepagents_talon.__main__.load_mcp_tools", fail_load_mcp)

    config = TalonConfig.from_env(
        {
            "AGENT_ASSISTANT_ID": "test",
            "DEEPAGENTS_TALON_FLEET_DIR": str(fleet_dir),
            "BUILTIN_MCP_URL": "https://tools.example/mcp",
        },
        base_home=tmp_path,
    )
    runtime = await _agent_runtime(
        config,
        CronJobStore(assistant_id="test", cron_dir=tmp_path / "cron"),
    )

    assert isinstance(runtime, DeepAgentRuntime)
    assert runtime.model == "fleet:model"
    assert runtime.system_prompt == "fleet prompt"
    assert runtime.tools == (fleet_tool,)
    assert runtime.subagents == (subagent,)
    assert runtime.assistant_dir is None
    assert seen["path"] == fleet_dir
    assert seen["env"]["BUILTIN_MCP_URL"] == "https://tools.example/mcp"


async def test_agent_runtime_allows_fleet_model_override(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fleet_dir = tmp_path / "fleet"
    fleet_dir.mkdir()

    async def fake_load_fleet(
        path,
        *,
        env,
    ) -> FleetAgentComponents:
        del path, env
        return FleetAgentComponents(
            model="fleet:model",
            system_prompt="fleet prompt",
            tools=(),
            subagents=(),
            interrupt_on=None,
        )

    monkeypatch.setattr("deepagents_talon.__main__.load_fleet_agent_components", fake_load_fleet)
    config = TalonConfig.from_env(
        {
            "AGENT_ASSISTANT_ID": "test",
            "AGENT_MODEL": "override:model",
            "DEEPAGENTS_TALON_FLEET_DIR": str(fleet_dir),
        },
        base_home=tmp_path,
    )

    runtime = await _agent_runtime(
        config,
        CronJobStore(assistant_id="test", cron_dir=tmp_path / "cron"),
    )

    assert isinstance(runtime, DeepAgentRuntime)
    assert runtime.model == "override:model"


async def test_agent_runtime_keeps_non_fleet_local_mcp_path(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail_load_fleet(path, *, env):
        del path, env
        msg = "Fleet loader should not run without a Fleet source"
        raise AssertionError(msg)

    async def fake_load_mcp(_config) -> MCPTools:
        return MCPTools(tools=(local_tool,), servers=())

    monkeypatch.setattr("deepagents_talon.__main__.load_fleet_agent_components", fail_load_fleet)
    monkeypatch.setattr("deepagents_talon.__main__.load_mcp_tools", fake_load_mcp)
    config = TalonConfig.from_env(
        {
            "AGENT_ASSISTANT_ID": "test",
            "AGENT_MODEL": "local:model",
        },
        base_home=tmp_path,
    )

    runtime = await _agent_runtime(
        config,
        CronJobStore(assistant_id="test", cron_dir=tmp_path / "cron"),
    )

    assert isinstance(runtime, DeepAgentRuntime)
    assert runtime.model == "local:model"
    assert runtime.tools == (local_tool,)
    assert runtime.assistant_dir == config.manifest_dir
