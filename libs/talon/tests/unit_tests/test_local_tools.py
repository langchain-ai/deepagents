from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from deepagents_talon.__main__ import _agent_runtime
from deepagents_talon.config import TalonConfig
from deepagents_talon.local_tools import LocalToolError, load_local_tools
from deepagents_talon.runtime import DeepAgentRuntime

if TYPE_CHECKING:
    from pathlib import Path

_TOOL_MODULE = '''from langchain_core.tools import tool

@tool
def local_echo(value: str) -> str:
    """Echo a value locally."""
    return value
'''


async def test_load_tools_filters_helpers_and_supports_sync_async_and_custom_tools(tmp_path: Path):
    (tmp_path / "tools.py").write_text(
        _TOOL_MODULE
        + '''
from dataclasses import dataclass
from langchain_core.tools import BaseTool

@dataclass
class Helper:
    value: str

@tool
async def async_echo(value: str) -> str:
    """Echo asynchronously."""
    return Helper(value).value

class CustomTool(BaseTool):
    name: str = "custom_echo"
    description: str = "Echo using a custom tool."

    def _run(self, value: str) -> str:
        return value

custom = CustomTool()
alias = local_echo
_private = tool("hidden", description="Hidden")(lambda: "hidden")

def helper():
    raise AssertionError("helpers must not be invoked")
''',
        encoding="utf-8",
    )
    (tmp_path / "_ignored.py").write_text("raise AssertionError", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "ignored.py").write_text("raise AssertionError", encoding="utf-8")

    modules = set(sys.modules)
    with load_local_tools([tmp_path, tmp_path]) as loaded:
        tools = {item.name: item for item in loaded}
        assert set(tools) == {"local_echo", "async_echo", "custom_echo"}
        assert tools["local_echo"].invoke({"value": "sync"}) == "sync"
        assert await tools["async_echo"].ainvoke({"value": "async"}) == "async"
        assert tools["custom_echo"].invoke({"value": "custom"}) == "custom"
    assert not any(name.startswith("_talon_local_tools_") for name in set(sys.modules) - modules)


def test_load_tools_uses_directory_and_filename_order(tmp_path: Path):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    for directory, filename, name in [
        (first, "z.py", "z"),
        (first, "a.py", "a"),
        (second, "b.py", "b"),
    ]:
        (directory / filename).write_text(
            _TOOL_MODULE.replace("local_echo", name), encoding="utf-8"
        )

    with load_local_tools([first, second]) as tools:
        assert [item.name for item in tools] == ["a", "z", "b"]
    with load_local_tools([]) as tools:
        assert tools == ()


def test_load_tools_rejects_duplicate_names(tmp_path: Path):
    for filename in ("a.py", "b.py"):
        (tmp_path / filename).write_text(_TOOL_MODULE, encoding="utf-8")
    with (
        pytest.raises(LocalToolError, match=r"Duplicate local tool 'local_echo'.*a\.py.*b\.py"),
        load_local_tools([tmp_path]),
    ):
        pytest.fail("invalid tools must fail before loading")


def test_load_tools_rejects_missing_or_non_directory_paths(tmp_path: Path):
    file = tmp_path / "file.py"
    file.touch()
    for path in (tmp_path / "missing", file):
        with pytest.raises(LocalToolError, match="not a directory"), load_local_tools([path]):
            pytest.fail("invalid paths must fail before loading")


@pytest.mark.parametrize("source", ["raise RuntimeError('private-secret')", "not valid python!"])
def test_failed_import_is_redacted_and_removed(tmp_path: Path, source: str):
    (tmp_path / "broken.py").write_text(source, encoding="utf-8")
    modules = set(sys.modules)
    with (
        pytest.raises(LocalToolError, match=r"Cannot import local tools.*broken\.py") as error,
        load_local_tools([tmp_path]),
    ):
        pytest.fail("invalid tools must fail before loading")
    assert "private-secret" not in str(error.value)
    assert not any(name.startswith("_talon_local_tools_") for name in set(sys.modules) - modules)


def test_load_tools_rejects_symlink_escape(tmp_path: Path):
    tools = tmp_path / "tools"
    tools.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("raise AssertionError('must not import')", encoding="utf-8")
    (tools / "escape.py").symlink_to(outside)
    with pytest.raises(LocalToolError, match="must remain inside"), load_local_tools([tools]):
        pytest.fail("symlink escapes must fail before loading")


async def test_runtime_preserves_local_tools_and_subagent_attachments_on_mcp_reload(
    tmp_path, monkeypatch
):
    (tmp_path / "tools.py").write_text(_TOOL_MODULE, encoding="utf-8")
    captured = []

    def create_graph(**kwargs: object):
        captured.append(kwargs)
        return SimpleNamespace()

    async def reload_tools():
        return ()

    monkeypatch.setattr("deepagents_talon.runtime.create_deep_agent", create_graph)
    monkeypatch.setattr(
        "deepagents_talon.runtime._resolve_model_from_env",
        lambda *_a, **_k: FakeMessagesListChatModel(responses=[AIMessage(content="done")]),
    )
    runtime = DeepAgentRuntime(
        model="test:model",
        tools_dirs=[tmp_path],
        refresh_tools=reload_tools,
        reload_tools=reload_tools,
        subagents=[
            {
                "name": "local",
                "description": "Local tools",
                "system_prompt": "Local",
                "tool_names": ["local_echo"],
            }
        ],
        env={},
        skills=(),
        memory=(),
        include_web_tools=False,
    )
    await runtime.start()
    original = next(item for item in captured[0]["tools"] if item.name == "local_echo")
    (tmp_path / "tools.py").write_text(
        "raise AssertionError('must not reimport')", encoding="utf-8"
    )
    await runtime._refresh_runtime_tools()
    await runtime.reload_mcp_configuration()
    await runtime.reload_subagent_configuration()
    for kwargs in captured:
        loaded = next(item for item in kwargs["tools"] if item.name == "local_echo")
        assert loaded is original
        assert loaded.invoke({"value": "available"}) == "available"
    assert "local_echo" in runtime._attachments[1]["tools"]
    await runtime.stop()


@pytest.mark.parametrize(
    "name",
    ["current_time", "read_file", "write_todos", "task", "get_agent_tools", "fetch_url", "compact"],
)
async def test_runtime_rejects_builtin_collisions(tmp_path: Path, name: str):
    (tmp_path / "tools.py").write_text(_TOOL_MODULE.replace("local_echo", name), encoding="utf-8")
    runtime = DeepAgentRuntime(
        model="test:model", tools_dirs=[tmp_path], env={}, skills=(), memory=()
    )
    modules = set(sys.modules)
    with pytest.raises(LocalToolError, match=f"Local tool name '{name}' conflicts"):
        await runtime.start()
    assert not any(value.startswith("_talon_local_tools_") for value in set(sys.modules) - modules)


async def test_mcp_collision_preserves_previous_graph(tmp_path: Path, monkeypatch):
    (tmp_path / "tools.py").write_text(_TOOL_MODULE, encoding="utf-8")
    monkeypatch.setattr("deepagents_talon.runtime.create_deep_agent", lambda **_: SimpleNamespace())

    @tool("local_echo", description="An MCP tool with the same name.")
    def mcp_tool() -> str:
        return "mcp"

    async def reload_tools():
        return [mcp_tool]

    runtime = DeepAgentRuntime(
        model="test:model",
        tools_dirs=[tmp_path],
        reload_tools=reload_tools,
        env={},
        skills=(),
        memory=(),
        include_web_tools=False,
    )
    await runtime.start()
    previous = runtime._graph
    with pytest.raises(LocalToolError, match="Local tool name 'local_echo' conflicts"):
        await runtime.reload_mcp_configuration()
    assert runtime._graph is previous
    assert runtime.tools == ()
    await runtime.stop()


async def test_runtime_stop_releases_imported_modules(tmp_path, monkeypatch):
    (tmp_path / "tools.py").write_text(_TOOL_MODULE, encoding="utf-8")
    monkeypatch.setattr("deepagents_talon.runtime.create_deep_agent", lambda **_: SimpleNamespace())
    modules = set(sys.modules)
    for _ in range(2):
        runtime = DeepAgentRuntime(
            model="test:model",
            tools_dirs=[tmp_path],
            env={},
            skills=(),
            memory=(),
        )
        await runtime.start()
        assert any(name.startswith("_talon_local_tools_") for name in set(sys.modules) - modules)
        await runtime.stop()
        assert not any(
            name.startswith("_talon_local_tools_") for name in set(sys.modules) - modules
        )


async def test_startup_wires_configured_directories_and_echo_skips_imports(tmp_path, monkeypatch):
    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / "tools.py").write_text(_TOOL_MODULE, encoding="utf-8")
    config = TalonConfig.from_env(
        {"AGENT_MODEL": "test:model", "DEEPAGENTS_TALON_TOOLS_DIRS": str(tools)},
        base_home=tmp_path,
    )

    async def load_mcp(_self):
        return SimpleNamespace(servers=(), tools=())

    monkeypatch.setattr("deepagents_talon.__main__.MCPToolProvider.load", load_mcp)
    runtime = await _agent_runtime(config)
    assert isinstance(runtime, DeepAgentRuntime)
    assert runtime.tools_dirs == (tools,)
    echo_config = TalonConfig.from_env(
        {"DEEPAGENTS_TALON_TOOLS_DIRS": str(tmp_path / "missing")},
        base_home=tmp_path,
    )
    echo = await _agent_runtime(echo_config)
    await echo.start()
