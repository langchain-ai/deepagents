"""Benign scripted application checks; these do not evaluate model judgment."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from deepagents_talon.config import TalonConfig
from deepagents_talon.runtime import DeepAgentRuntime
from tests.unit_tests.test_research_subagents import ToolModel, _inventory, _runtime

_FIXTURES = json.loads((Path(__file__).parent / "fixtures" / "hardening_reviews.json").read_text())


def test_skill_installation_and_discovery_preserve_user_changes(tmp_path: Path) -> None:
    config = TalonConfig("review", tmp_path / "assistant")
    config.ensure_home()
    runtime = DeepAgentRuntime(model="test:parent", assistant_dir=config.home, env={})
    sources = runtime._resolve_skills()
    assert sources == [str(config.home / "skills")]
    skill = Path(sources[0]) / "configuration-hardening"
    files = [skill / "SKILL.md", skill / "references" / "hardening-model.md"]
    for path in files:
        assert path.is_file()
        assert path.stat().st_mode & 0o777 == 0o600
        path.write_text("Operator customization")
    before = [path.stat().st_mtime_ns for path in files]
    config.ensure_home()
    assert [path.read_text() for path in files] == ["Operator customization"] * 2
    assert [path.stat().st_mtime_ns for path in files] == before
    files[1].unlink()
    config.ensure_home()
    assert files[1].is_file()
    assert files[0].read_text() == "Operator customization"


@pytest.mark.parametrize("fixture", _FIXTURES, ids=lambda item: item["name"])
async def test_review_apply_verify_and_rollback(tmp_path, monkeypatch, fixture):
    config = TalonConfig("review", tmp_path / "assistant")
    config.ensure_home()
    model = ToolModel(responses=[AIMessage(content="Done")])
    runtime = _runtime(config.home, monkeypatch, model, model)
    path = config.agents_dir / "internal-research" / "AGENTS.md"
    original = path.read_text()
    unrelated = config.agents_dir / "external-research" / "AGENTS.md"
    untouched = unrelated.read_bytes()
    await runtime.start()
    try:
        before = _inventory(runtime)
        old_inventory = runtime._attachment_tool(runtime._attachments)
        path.write_text(original.replace("tools: []", f"tools: {json.dumps(fixture['tools'])}"))
        assert _inventory(runtime)["saved_changes_inactive"]
        result = await runtime._subagent_reload_tool().ainvoke({})
        assert result["status"] == fixture["status"]
        after = _inventory(runtime)
        agents = {agent["name"]: agent["tools"] for agent in after["latest_agents"]}
        assert agents["internal-research"] == fixture["expected"]
        assert after["saved_changes_inactive"] == (fixture["status"] == "failed")
        assert old_inventory.invoke({})["current_turn_uses_previous_graph"] == (
            fixture["status"] == "reloaded"
        )
        assert not {"execute", "write_file", "task"} & set(agents["internal-research"])
        assert "execute" in agents["main"]
        assert unrelated.read_bytes() == untouched
        graph = runtime._graph
        assert _inventory(runtime) == after
        assert runtime._graph is graph
        path.write_text(original)
        await runtime.reload_subagent_configuration()
        assert _inventory(runtime)["agents"] == before["agents"]
        assert not _inventory(runtime)["saved_changes_inactive"]
    finally:
        await runtime.stop()
