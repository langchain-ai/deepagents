from __future__ import annotations

import json
import logging
import os
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

from langchain.agents.middleware.types import AgentMiddleware

from deepagents_talon.__main__ import _agent_runtime
from deepagents_talon.config import TalonConfig
from deepagents_talon.cron import CronJobStore
from deepagents_talon.fleet import FleetAgentComponents, load_fleet_agent_components
from deepagents_talon.fleet_manifest import load_fleet_run_manifest, manifest_path
from deepagents_talon.mcp import MCPTools
from deepagents_talon.runtime import INTERRUPT_ON_TOOLS_ENV_KEY, DeepAgentRuntime

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


class PassthroughMiddleware(AgentMiddleware):
    """Middleware stub for runtime wiring assertions."""


def fleet_tool() -> str:
    """Fleet tool stub."""
    return "fleet"


def local_tool() -> str:
    """Local MCP tool stub."""
    return "local"


def _skill_content(name: str, description: str) -> str:
    return f"""---
name: {name}
description: {description}
---

# {name}
"""


def _write_required_fleet_files(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "AGENTS.md").write_text("Fleet prompt", encoding="utf-8")
    (path / "config.json").write_text(json.dumps({"model": "fleet:model"}), encoding="utf-8")


async def test_load_fleet_agent_components_coerces_public_loader_payload(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_required_fleet_files(tmp_path)
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


async def test_load_fleet_agent_components_adds_static_skills_loader(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fleet_dir = tmp_path / "fleet"
    _write_required_fleet_files(fleet_dir)
    fleet_skill = fleet_dir / "skills" / "fleet-skill"
    fleet_skill.mkdir(parents=True)
    (fleet_skill / "SKILL.md").write_text(
        _skill_content("fleet-skill", "Fleet skill description"),
        encoding="utf-8",
    )
    operator_dir = tmp_path / "operator-skills"
    operator_skill = operator_dir / "operator-skill"
    operator_skill.mkdir(parents=True)
    (operator_skill / "SKILL.md").write_text(
        _skill_content("operator-skill", "Operator skill description"),
        encoding="utf-8",
    )

    async def fake_load_agent_components(path):
        assert path == fleet_dir
        return {
            "model": "fleet:model",
            "system_prompt": "fleet prompt",
            "tools": [],
            "subagents": [],
            "interrupt_on": None,
        }

    monkeypatch.setattr(
        "deepagents_talon.fleet.load_agent_components",
        fake_load_agent_components,
    )

    components = await load_fleet_agent_components(
        fleet_dir,
        env={"DEEPAGENTS_TALON_SKILLS_DIRS": str(operator_dir)},
    )

    assert components.skills == (str(fleet_dir / "skills"), str(operator_dir))
    assert len(components.middleware) == 1

    loader = cast("Any", components.middleware[0])
    update = loader.before_agent({"skills_metadata": []}, object())
    assert update is not None
    assert set(update["files"]) == {
        str(fleet_dir / "skills" / "fleet-skill" / "SKILL.md"),
        str(operator_dir / "operator-skill" / "SKILL.md"),
    }
    assert {skill["name"] for skill in update["skills_metadata"]} == {
        "fleet-skill",
        "operator-skill",
    }

    files_only = loader.before_agent({"skills_metadata": [{"name": "already-loaded"}]}, object())
    assert files_only is not None
    assert "files" in files_only
    assert "skills_metadata" not in files_only


async def test_load_fleet_agent_components_logs_mcp_surface(
    tmp_path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fleet_dir = tmp_path / "fleet"
    _write_required_fleet_files(fleet_dir)
    (fleet_dir / "tools.json").write_text(
        json.dumps(
            {
                "tools": [
                    {
                        "name": "search",
                        "mcp_server_url": "https://builtin.example/catalog?token=raw-secret",
                        "auth_type": "builtin",
                    },
                    {
                        "name": "missing",
                        "mcp_server_url": "https://missing.example/mcp?token=missing-secret",
                        "auth_type": "headers",
                        "headers": {"Authorization": "Bearer header-secret"},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    subagent_dir = fleet_dir / "subagents" / "researcher"
    subagent_dir.mkdir(parents=True)
    (subagent_dir / "tools.json").write_text(
        json.dumps(
            {
                "tools": [
                    {
                        "name": "calendar",
                        "mcp_server_url": "https://calendar.example/mcp?token=oauth-secret",
                        "auth_type": "oauth",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    async def fake_load_agent_components(path):
        assert path == fleet_dir
        return {
            "model": "fleet:model",
            "system_prompt": "fleet prompt",
            "tools": [SimpleNamespace(name="search")],
            "subagents": [{"tools": [SimpleNamespace(name="calendar")]}],
            "interrupt_on": None,
        }

    monkeypatch.setattr(
        "deepagents_talon.fleet.load_agent_components",
        fake_load_agent_components,
    )

    with caplog.at_level(logging.INFO, logger="deepagents_talon.fleet"):
        await load_fleet_agent_components(
            fleet_dir,
            env={"BUILTIN_MCP_URL": "https://builtin.example/mcp?api_key=builtin-secret"},
        )

    event = _talon_events(caplog, event="fleet.mcp_surface")[0]
    assert event["server_count"] == 3
    server_payload = cast("list[dict[str, object]]", event["servers"])
    servers = {cast("str", server["auth_path"]): server for server in server_payload}
    assert servers["builtin"]["endpoint"] == "https://builtin.example/mcp"
    assert servers["builtin"]["status"] == "loaded"
    assert servers["builtin"]["loaded_tools"] == ["search"]
    assert servers["oauth"]["endpoint"] == "https://calendar.example/mcp"
    assert servers["oauth"]["status"] == "loaded"
    assert servers["oauth"]["scopes"] == ["subagent:researcher"]
    assert servers["headers"]["endpoint"] == "https://missing.example/mcp"
    assert servers["headers"]["status"] == "skipped"
    assert servers["headers"]["requested_tools"] == ["missing"]
    assert "secret" not in caplog.text


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
    middleware = PassthroughMiddleware()

    async def fake_load_fleet(
        path,
        *,
        env,
    ) -> FleetAgentComponents:
        seen.setdefault("paths", []).append(path)
        seen["env"] = env
        return FleetAgentComponents(
            model="fleet:model",
            system_prompt="fleet prompt",
            tools=(fleet_tool,),
            subagents=(cast("Any", subagent),),
            interrupt_on={"fleet_tool": True},
            skills=("/fleet/skills",),
            middleware=(middleware,),
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
            INTERRUPT_ON_TOOLS_ENV_KEY: "bash, execute",
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
    assert runtime.skills == ("/fleet/skills",)
    assert runtime.middleware == (middleware,)
    assert runtime.interrupt_on == {
        "fleet_tool": True,
        "bash": True,
        "execute": True,
    }
    assert runtime.assistant_dir is None
    assert seen["paths"] == [fleet_dir]
    assert seen["env"]["BUILTIN_MCP_URL"] == "https://tools.example/mcp"
    assert runtime.reload_agent_components is not None
    manifest = load_fleet_run_manifest(manifest_path(config.home))
    assert manifest.assistant_id == "test"
    assert manifest.fleet_dir == str(fleet_dir)
    assert manifest.local_mcp_config_path == str(tmp_path / "test" / ".mcp.json")
    assert manifest.approval_tool_names == ("fleet_tool",)

    refreshed = await runtime.reload_agent_components()

    assert refreshed.model == "fleet:model"
    assert refreshed.tools == (fleet_tool,)
    assert refreshed.middleware == (middleware,)
    assert refreshed.interrupt_on == {
        "fleet_tool": True,
        "bash": True,
        "execute": True,
    }
    assert seen["paths"] == [fleet_dir, fleet_dir]
    assert load_fleet_run_manifest(manifest_path(config.home)).created_at == manifest.created_at


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
        return MCPTools(tools=(cast("Any", local_tool),), servers=())

    monkeypatch.setattr("deepagents_talon.__main__.load_fleet_agent_components", fail_load_fleet)
    monkeypatch.setattr("deepagents_talon.__main__.load_mcp_tools", fake_load_mcp)
    config = TalonConfig.from_env(
        {
            "AGENT_ASSISTANT_ID": "test",
            "AGENT_MODEL": "local:model",
            INTERRUPT_ON_TOOLS_ENV_KEY: "local_tool, custom_mcp_tool",
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
    assert runtime.interrupt_on == {"local_tool": True, "custom_mcp_tool": True}


def _talon_events(
    caplog: pytest.LogCaptureFixture,
    *,
    event: str,
) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for message in caplog.messages:
        if not message.startswith("talon_event "):
            continue
        payload = json.loads(message.removeprefix("talon_event "))
        if payload.get("event") == event:
            events.append(payload)
    return events
