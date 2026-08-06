"""Tests for the Harbor Switchyard LangGraph agent."""

from __future__ import annotations

import pytest
from harbor.agents.installed.langgraph import LangGraph

from deepagents_harbor.switchyard_agent import SwitchyardLangGraph
from deepagents_harbor.switchyard_environment import SwitchyardLangSmithEnvironment


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
