"""Tests for the Harbor Switchyard LangGraph agent."""

from __future__ import annotations

import pytest
from harbor.agents.installed.langgraph import LangGraph
from harbor.models.agent.context import AgentContext

from deepagents_harbor.switchyard_agent import SwitchyardLangGraph
from deepagents_harbor.switchyard_environment import (
    SwitchyardDockerEnvironment,
    SwitchyardLangSmithEnvironment,
)


async def test_setup_always_removes_temporary_egress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = object.__new__(SwitchyardLangGraph)
    environment = object.__new__(SwitchyardLangSmithEnvironment)
    calls: list[str] = []

    async def fake_setup(_self, _environment) -> None:
        calls.append("setup")

    async def fake_isolate() -> None:
        calls.append("isolate")

    monkeypatch.setattr(LangGraph, "setup", fake_setup)
    environment.isolate_main_after_setup = fake_isolate

    await agent.setup(environment)

    assert calls == ["setup", "isolate"]


async def test_setup_removes_egress_when_install_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = object.__new__(SwitchyardLangGraph)
    environment = object.__new__(SwitchyardLangSmithEnvironment)
    isolated = False

    async def fake_setup(_self, _environment) -> None:
        msg = "install failed"
        raise RuntimeError(msg)

    async def fake_isolate() -> None:
        nonlocal isolated
        isolated = True

    monkeypatch.setattr(LangGraph, "setup", fake_setup)
    environment.isolate_main_after_setup = fake_isolate

    with pytest.raises(RuntimeError, match="install failed"):
        await agent.setup(environment)

    assert isolated is True


async def test_setup_uses_native_docker_without_langsmith_isolation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = object.__new__(SwitchyardLangGraph)
    environment = object.__new__(SwitchyardDockerEnvironment)
    calls: list[str] = []

    async def fake_setup(_self, _environment) -> None:
        calls.append("setup")

    monkeypatch.setattr(LangGraph, "setup", fake_setup)

    await agent.setup(environment)

    assert calls == ["setup"]


async def test_docker_run_closes_sidecar_after_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = object.__new__(SwitchyardLangGraph)
    environment = object.__new__(SwitchyardDockerEnvironment)
    calls: list[str] = []

    async def fake_run(_self, _instruction, _environment, _context) -> None:
        calls.append("run")

    async def fake_capture() -> None:
        calls.append("capture")

    monkeypatch.setattr(LangGraph, "run", fake_run)
    environment.capture_and_stop_switchyard = fake_capture

    await agent.run("task", environment, AgentContext())

    assert calls == ["run", "capture"]


async def test_docker_run_closes_sidecar_when_agent_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = object.__new__(SwitchyardLangGraph)
    environment = object.__new__(SwitchyardDockerEnvironment)
    closed = False

    async def fake_run(_self, _instruction, _environment, _context) -> None:
        msg = "agent failed"
        raise RuntimeError(msg)

    async def fake_capture() -> None:
        nonlocal closed
        closed = True

    monkeypatch.setattr(LangGraph, "run", fake_run)
    environment.capture_and_stop_switchyard = fake_capture

    with pytest.raises(RuntimeError, match="agent failed"):
        await agent.run("task", environment, AgentContext())

    assert closed is True
