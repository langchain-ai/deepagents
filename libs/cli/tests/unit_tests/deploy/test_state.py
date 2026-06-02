"""Tests for user-local deploy state."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

import deepagents_cli.deploy.state as state_module
from deepagents_cli.deploy.state import State

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _state_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(state_module, "_STATE_ROOT", tmp_path / "deploy-state")


def test_load_missing_returns_empty(tmp_path: Path) -> None:
    state = State.load(tmp_path, endpoint="https://api.smith.langchain.com")
    assert state.agent_id is None
    assert state.revision is None
    assert state.endpoint == "https://api.smith.langchain.com"
    assert state.mcp_servers == {}


def test_save_writes_schema_versioned_json(tmp_path: Path) -> None:
    state = State.load(tmp_path, endpoint="https://api.smith.langchain.com")
    state.save(agent_id="abc", revision="rev1")
    data = json.loads(state.state_path.read_text())
    assert data["schema_version"] == 1
    assert data["project_root"] == str(tmp_path)
    assert data["agent_id"] == "abc"
    assert data["revision"] == "rev1"
    assert data["endpoint"] == "https://api.smith.langchain.com"
    assert "last_deployed_at" in data
    assert data["mcp_servers"] == {}
    assert not (tmp_path / ".deepagents" / "state.json").exists()


def test_save_then_reload_roundtrips(tmp_path: Path) -> None:
    s1 = State.load(tmp_path, endpoint="https://example.invalid")
    s1.mcp_servers = {"https://tools.example/": "srv-1"}
    s1.save(agent_id="aid", revision="r1")
    s2 = State.load(tmp_path, endpoint="https://example.invalid")
    assert s2.agent_id == "aid"
    assert s2.revision == "r1"
    assert s2.endpoint == "https://example.invalid"
    assert s2.mcp_servers == {"https://tools.example/": "srv-1"}


def test_endpoint_is_part_of_state_key(tmp_path: Path) -> None:
    first = State.load(tmp_path, endpoint="https://first.invalid")
    second = State.load(tmp_path, endpoint="https://second.invalid")
    assert first.state_path != second.state_path


def test_reset_clears_existing(tmp_path: Path) -> None:
    State.load(tmp_path, endpoint="https://api.invalid").save(
        agent_id="abc", revision="r1"
    )
    fresh = State.load(tmp_path, endpoint="https://api.invalid", reset=True)
    assert fresh.agent_id is None
    assert not fresh.state_path.exists()


def test_clear_agent_removes_id(tmp_path: Path) -> None:
    s = State.load(tmp_path, endpoint="https://api.invalid")
    s.save(agent_id="abc", revision="r1")
    s.clear_agent()
    reloaded = State.load(tmp_path, endpoint="https://api.invalid")
    assert reloaded.agent_id is None
    assert reloaded.revision is None


def test_unknown_schema_version_raises(tmp_path: Path) -> None:
    state = State.load(tmp_path, endpoint="https://api.invalid")
    state.state_path.parent.mkdir(parents=True)
    state.state_path.write_text(json.dumps({"schema_version": 99, "agent_id": "x"}))
    with pytest.raises(ValueError, match="schema_version"):
        State.load(tmp_path, endpoint="https://api.invalid")
