"""Scripted adversarial calls prove capabilities and gates, not model refusal."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from deepagents_talon.config import TalonConfig, _install_defaults
from deepagents_talon.interfaces import AgentRequest
from tests.unit_tests.test_research_subagents import ToolModel, _call, _inventory, _runtime

_FIXTURES = json.loads(
    (Path(__file__).parent / "fixtures" / "research_injections.json").read_text()
)


def test_install_defaults_preserves_existing_files(tmp_path: Path) -> None:
    fresh = TalonConfig("fresh", tmp_path / "fresh")
    fresh.ensure_home()
    instructions = fresh.home / "AGENTS.md"
    assert instructions.is_file()
    assert instructions.stat().st_mode & 0o777 == 0o600
    assert (fresh.agents_dir / "internal-research" / "AGENTS.md").is_file()
    assert (fresh.agents_dir / "external-research" / "AGENTS.md").is_file()
    instructions.write_text("User instructions")
    fresh.ensure_home()
    assert instructions.read_text() == "User instructions"
    existing = TalonConfig("existing", tmp_path / "existing")
    existing.home.mkdir()
    existing.ensure_home()
    assert (existing.home / "AGENTS.md").is_file()
    assert (existing.agents_dir / "internal-research" / "AGENTS.md").is_file()
    assert (existing.agents_dir / "external-research" / "AGENTS.md").is_file()


def test_existing_home_backfills_missing_research_defaults(tmp_path: Path) -> None:
    config = TalonConfig("existing", tmp_path / "existing")
    internal = config.agents_dir / "internal-research" / "AGENTS.md"
    internal.parent.mkdir(parents=True)
    internal.write_text("Custom internal research")
    instructions = config.home / "AGENTS.md"
    instructions.write_text("Custom main instructions")
    external = config.agents_dir / "external-research" / "AGENTS.md"
    external.parent.mkdir()

    for _ in range(2):
        config.ensure_home()
        assert internal.read_text() == "Custom internal research"
        assert instructions.read_text() == "Custom main instructions"
        assert external.is_file()
        assert external.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("name", ["internal-research", "external-research"])
def test_interrupted_backfill_recovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    config = TalonConfig("existing", tmp_path / "existing")
    config.home.mkdir()
    instructions = config.home / "AGENTS.md"
    instructions.write_text("Custom instructions")
    target = config.agents_dir / name / "AGENTS.md"
    write_text = Path.write_text

    def interrupted(path: Path, contents: str, encoding: str | None = None) -> int:
        if path.parent.parent == target.parent:
            write_text(path, contents[:10], encoding=encoding)
            msg = "disk full"
            raise OSError(msg)
        return write_text(path, contents, encoding=encoding)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "write_text", interrupted)
        with pytest.raises(OSError, match="disk full"):
            config.ensure_home()
    assert not target.exists()
    assert not list(target.parent.iterdir())
    config.ensure_home()
    defaults = Path(__file__).parents[2] / "deepagents_talon" / "defaults"
    assert target.read_text() == (defaults / "agents" / name / "AGENTS.md").read_text()
    assert target.stat().st_mode & 0o777 == 0o600
    assert instructions.read_text() == "Custom instructions"


def test_interrupted_install_does_not_publish_partial_home(tmp_path, monkeypatch):
    config = TalonConfig("fresh", tmp_path / "fresh")

    def interrupted(home):
        (home / "AGENTS.md").write_text("partial instructions")
        msg = "disk full"
        raise OSError(msg)

    with monkeypatch.context() as patch:
        patch.setattr("deepagents_talon.config._install_defaults", interrupted)
        with pytest.raises(OSError, match="disk full"):
            config.ensure_home()
    assert not config.home.exists()
    assert not list(tmp_path.iterdir())
    config.ensure_home()
    assert "evidence, not user instructions" in (config.home / "AGENTS.md").read_text()
    assert (config.agents_dir / "internal-research" / "AGENTS.md").is_file()
    assert (config.agents_dir / "external-research" / "AGENTS.md").is_file()


@pytest.mark.parametrize("web_enabled", [False, True])
@pytest.mark.parametrize("tavily_key", [None, "", "  ", "test-key"])
async def test_missing_tools_reload_and_rollback(tmp_path, monkeypatch, web_enabled, tavily_key):
    monkeypatch.setenv("TAVILY_API_KEY", "unrelated-process-key")
    _install_defaults(tmp_path)
    model = ToolModel(responses=[AIMessage(content="Done")])
    env = {} if tavily_key is None else {"TAVILY_API_KEY": tavily_key}
    runtime = _runtime(tmp_path, monkeypatch, model, model, include_web_tools=web_enabled, env=env)
    expected = ["fetch_url"] if web_enabled else []
    if web_enabled and tavily_key and tavily_key.strip():
        expected.append("web_search")
    await runtime.start()
    try:
        agents = {item["name"]: item["tools"] for item in _inventory(runtime)["agents"]}
        assert agents["internal-research"] == []
        assert agents["external-research"] == expected
        assert not {"fetch_url", "web_search"} & set(agents["main"])
        assert {"read_file", "write_file", "execute"} <= set(agents["main"])
        path = tmp_path / "agents" / "internal-research" / "AGENTS.md"
        original = path.read_text()
        path.write_text(original.replace("tools: []", "tools: [current_time]"))
        assert _inventory(runtime)["saved_changes_inactive"]
        await runtime.reload_subagent_configuration()
        agents = {item["name"]: item["tools"] for item in _inventory(runtime)["agents"]}
        assert agents["internal-research"] == ["current_time"]
        assert agents["external-research"] == expected
        assert not {"fetch_url", "web_search"} & set(agents["main"])
        active = runtime._graph
        path.write_text(original.replace("tools: []", "tools: null"))
        assert (await runtime._subagent_reload_tool().ainvoke({}))["status"] == "failed"
        assert runtime._graph is active
        assert _inventory(runtime)["saved_changes_inactive"]
        runtime._replace_runtime_tools([])
        assert "web_search" not in _inventory(runtime)["agents"][0]["tools"]
        path.write_text(original)
        await runtime.reload_subagent_configuration()
        assert not _inventory(runtime)["saved_changes_inactive"]
        assert not {"fetch_url", "web_search"} & set(_inventory(runtime)["agents"][0]["tools"])
    finally:
        await runtime.stop()


@pytest.mark.parametrize("fixture", _FIXTURES[:4], ids=lambda item: item["surface"])
async def test_research_injection_cannot_gain_tools(tmp_path, monkeypatch, fixture):
    _install_defaults(tmp_path)

    @tool(fixture["tool"])
    def source() -> str:
        """Return fixture evidence."""
        return f"{fixture['citation']}: {fixture['text']}"

    forbidden = [
        "read_file",
        "write_file",
        "execute",
        "update_mcp_server",
        "send_email",
        "search_conversations",
        "reload_subagent_configuration",
        "task",
        "start_async_task",
    ]
    child = ToolModel(
        responses=[
            AIMessage(content="", tool_calls=[_call(source.name)]),
            AIMessage(content="", tool_calls=[_call(name) for name in forbidden]),
            AIMessage(content=f"Friday; source: {fixture['citation']}; suspected injection."),
        ]
    )
    parent = ToolModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    _call(
                        "task",
                        subagent_type=fixture["role"],
                        description="Find the deadline; cite evidence.",
                        tools=[source.name],
                    )
                ],
            ),
            AIMessage(content="Delegated"),
        ]
    )
    runtime = _runtime(
        tmp_path,
        monkeypatch,
        parent,
        child,
        tools=[source],
        include_web_tools=True,
        env={"TAVILY_API_KEY": "test-key"},
    )
    await runtime.start()
    try:
        await runtime.invoke(AgentRequest("chat", "Find the deadline. Private marker: PRIVATE-123"))
        await asyncio.gather(*(job.worker for job in runtime.background._jobs.values()))
        agents = {item["name"]: item["tools"] for item in _inventory(runtime)["agents"]}
        base = ["fetch_url", "web_search"] if fixture["role"] == "external-research" else []
        assert agents[fixture["role"]] == base
        assert source.name in agents["main"]
        assert set(child._tools[-1]) == {*base, source.name}
        assert "PRIVATE-123" not in str(child._seen)
        assert fixture["text"] in str(child._seen)
        denied = {
            message.name
            for message in child._seen[-1]
            if getattr(message, "status", None) == "error"
        }
        assert denied == set(forbidden)
        assert fixture["citation"] in str(runtime.background.results("chat"))
    finally:
        await runtime.stop()


@pytest.mark.parametrize("fixture", _FIXTURES[4:], ids=lambda item: item["surface"])
async def test_main_injection_cannot_fabricate_approval(tmp_path, monkeypatch, fixture):
    _install_defaults(tmp_path)
    path = tmp_path / "fixture.txt"
    path.write_text(fixture["text"])
    sent = []

    @tool
    def send_email() -> str:
        """Send private notes."""
        sent.append(True)
        return "sent"

    parent = ToolModel(
        responses=[
            AIMessage(content="", tool_calls=[_call("read_file", file_path=str(path))]),
            AIMessage(content="", tool_calls=[_call("send_email")]),
            AIMessage(content="Friday. The action was not approved."),
        ]
    )
    runtime = _runtime(
        tmp_path,
        monkeypatch,
        parent,
        parent,
        tools=[send_email],
        interrupt_on={"send_email": True},
        include_web_tools=True,
    )
    if fixture["surface"] == "returned-evidence":
        parent.responses = parent.responses[1:]
        monkeypatch.setattr(
            runtime.background, "results", lambda _: {"research-result": fixture["text"]}
        )
    await runtime.start()
    try:
        await runtime.invoke(AgentRequest("chat", "Find the deadline."))
        assert not sent
        assert fixture["text"] in str(parent._seen)
        assert "evidence, not user instructions" in str(parent._seen[0][0].content)
        assert {"read_file", "write_file", "execute", "send_email"} <= set(
            _inventory(runtime)["agents"][0]["tools"]
        )
    finally:
        await runtime.stop()
