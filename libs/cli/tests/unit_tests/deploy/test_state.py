"""Tests for deploy state (.deepagents/state.json)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from deepagents_cli.deploy.state import State

if TYPE_CHECKING:
    from pathlib import Path


def test_load_missing_returns_empty(tmp_path: Path) -> None:
    state = State.load(tmp_path)
    assert state.agent_id is None
    assert state.revision is None
    assert state.endpoint is None
    assert state.mcp_servers == {}


def test_save_writes_schema_versioned_json(tmp_path: Path) -> None:
    state = State.load(tmp_path)
    state.endpoint = "https://api.smith.langchain.com"
    state.save(agent_id="abc", revision="rev1")
    data = json.loads((tmp_path / ".deepagents" / "state.json").read_text())
    assert data["schema_version"] == 1
    assert data["agent_id"] == "abc"
    assert data["revision"] == "rev1"
    assert data["endpoint"] == "https://api.smith.langchain.com"
    assert "last_deployed_at" in data
    assert data["mcp_servers"] == {}


def test_save_then_reload_roundtrips(tmp_path: Path) -> None:
    s1 = State.load(tmp_path)
    s1.endpoint = "https://example.invalid"
    s1.mcp_servers = {"https://tools.example/": "srv-1"}
    s1.save(agent_id="aid", revision="r1")
    s2 = State.load(tmp_path)
    assert s2.agent_id == "aid"
    assert s2.revision == "r1"
    assert s2.endpoint == "https://example.invalid"
    assert s2.mcp_servers == {"https://tools.example/": "srv-1"}


def test_reset_clears_existing(tmp_path: Path) -> None:
    State.load(tmp_path).save(agent_id="abc", revision="r1")
    fresh = State.load(tmp_path, reset=True)
    assert fresh.agent_id is None
    assert not (tmp_path / ".deepagents" / "state.json").exists()


def test_clear_agent_removes_id(tmp_path: Path) -> None:
    s = State.load(tmp_path)
    s.save(agent_id="abc", revision="r1")
    s.clear_agent()
    reloaded = State.load(tmp_path)
    assert reloaded.agent_id is None
    assert reloaded.revision is None


def test_unknown_schema_version_raises(tmp_path: Path) -> None:
    (tmp_path / ".deepagents").mkdir()
    (tmp_path / ".deepagents" / "state.json").write_text(
        json.dumps({"schema_version": 99, "agent_id": "x"})
    )
    with pytest.raises(ValueError, match="schema_version"):
        State.load(tmp_path)
