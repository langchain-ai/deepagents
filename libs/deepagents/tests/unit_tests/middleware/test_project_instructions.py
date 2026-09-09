"""Tests for project instruction discovery."""

from pathlib import Path

from langchain.agents.middleware.types import ToolCallRequest
from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.project_instructions import (
    ProjectInstructionsData,
    ProjectInstructionsMiddleware,
    _merge_project_instructions,
)


def _middleware(tmp_path: Path) -> ProjectInstructionsMiddleware:
    return ProjectInstructionsMiddleware(
        backend=FilesystemBackend(root_dir=tmp_path, virtual_mode=True),
        project_root="/",
        cwd="/src",
        backend_root="/",
    )


def _virtual(path: Path, root: Path) -> str:
    return "/" + path.relative_to(root).as_posix()


def _request(
    name: str,
    file_path: str | None,
    state: dict,
    *,
    args: dict | None = None,
) -> ToolCallRequest:
    runtime = ToolRuntime(
        state=state,
        context=None,
        tool_call_id="call_1",
        store=None,
        stream_writer=lambda _: None,
        config={},
    )
    tool_args = args if args is not None else {"file_path": file_path}
    return ToolCallRequest(
        runtime=runtime,
        tool_call={"id": "call_1", "name": name, "args": tool_args},
        state=state,
        tool=None,
    )


def test_loads_ambient_instructions_from_root_through_cwd(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "AGENTS.md").write_text("root", encoding="utf-8")
    (tmp_path / "src" / "AGENTS.md").write_text("src", encoding="utf-8")
    outside = tmp_path.parent / "AGENTS.md"
    outside.write_text("outside", encoding="utf-8")

    update = _middleware(tmp_path).before_agent({}, None, {})  # type: ignore[arg-type]

    contents = update["project_instructions"]["contents"]
    assert list(contents.values()) == ["root", "src"]
    assert "outside" not in contents.values()


def test_nested_read_discovers_checkpointed_scoped_instructions(tmp_path: Path) -> None:
    nested = tmp_path / "src" / "feature"
    nested.mkdir(parents=True)
    (nested / "AGENTS.md").write_text("feature", encoding="utf-8")
    middleware = _middleware(tmp_path)
    state = middleware.before_agent({}, None, {})  # type: ignore[arg-type]
    calls = 0

    def handler(request: ToolCallRequest) -> ToolMessage:
        nonlocal calls
        calls += 1
        return ToolMessage(content="file", tool_call_id=request.tool_call["id"])

    result = middleware.wrap_tool_call(_request("read_file", _virtual(nested / "code.py", tmp_path), state), handler)

    assert calls == 1
    assert isinstance(result, Command)
    assert result.update is not None
    contents = result.update["project_instructions"]["contents"]
    assert contents[_virtual(nested / "AGENTS.md", tmp_path)] == "feature"


def test_mutation_retries_after_new_instructions_are_visible(tmp_path: Path) -> None:
    nested = tmp_path / "src" / "feature"
    nested.mkdir(parents=True)
    (nested / "AGENTS.md").write_text("feature", encoding="utf-8")
    middleware = _middleware(tmp_path)
    state = middleware.before_agent({}, None, {})  # type: ignore[arg-type]
    calls = 0

    def handler(request: ToolCallRequest) -> ToolMessage:
        nonlocal calls
        calls += 1
        return ToolMessage(content="written", tool_call_id=request.tool_call["id"])

    request = _request("write_file", _virtual(nested / "code.py", tmp_path), state)
    result = middleware.wrap_tool_call(request, handler)

    assert calls == 0
    assert isinstance(result, Command)
    assert result.update is not None
    [message] = result.update["messages"]
    assert "retry the tool call" in message.content

    result = middleware.wrap_tool_call(
        _request(
            "write_file",
            _virtual(nested / "code.py", tmp_path),
            {"project_instructions": result.update["project_instructions"]},
        ),
        handler,
    )
    assert calls == 1
    assert isinstance(result, ToolMessage)


def test_prompt_preserves_scope_and_trust_boundary(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "AGENTS.md").write_text("root", encoding="utf-8")
    middleware = _middleware(tmp_path)
    state = middleware.before_agent({}, None, {})  # type: ignore[arg-type]

    prompt = middleware._format_prompt(state)

    assert prompt is not None
    assert 'applies_to="/"' in prompt
    assert "subordinate to system policy and explicit user instructions" in prompt
    assert "root" in prompt


def test_hidden_instructions_apply_to_project_root(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / ".deepagents").mkdir()
    hidden = tmp_path / ".deepagents" / "AGENTS.md"
    hidden.write_text("hidden", encoding="utf-8")

    update = _middleware(tmp_path).before_agent({}, None, {})  # type: ignore[arg-type]

    data = update["project_instructions"]
    assert data["contents"]["/.deepagents/AGENTS.md"] == "hidden"
    assert data["scopes"]["/.deepagents/AGENTS.md"] == "/"


def test_search_tools_discover_nested_instructions(tmp_path: Path) -> None:
    nested = tmp_path / "src" / "feature"
    nested.mkdir(parents=True)
    instructions = nested / "AGENTS.md"
    instructions.write_text("feature", encoding="utf-8")
    middleware = _middleware(tmp_path)

    for name, args in (
        ("ls", {"path": "/src/feature"}),
        ("glob", {"pattern": "*.py", "path": "/src/feature"}),
        ("grep", {"pattern": "x", "path": "/src/feature"}),
    ):
        state = middleware.before_agent({}, None, {})  # type: ignore[arg-type]
        result = middleware.wrap_tool_call(
            _request(name, None, state, args=args),
            lambda request: ToolMessage(content="result", tool_call_id=request.tool_call["id"]),
        )

        assert isinstance(result, Command)
        assert result.update is not None
        assert result.update["project_instructions"]["contents"][_virtual(instructions, tmp_path)] == "feature"


def test_search_tool_without_path_defaults_to_cwd(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    instructions = tmp_path / "src" / "AGENTS.md"
    instructions.write_text("src", encoding="utf-8")
    middleware = _middleware(tmp_path)

    result = middleware.wrap_tool_call(
        _request("glob", None, {}, args={"pattern": "*.py"}),
        lambda request: ToolMessage(content="result", tool_call_id=request.tool_call["id"]),
    )

    assert isinstance(result, Command)
    assert result.update is not None
    assert result.update["project_instructions"]["contents"][_virtual(instructions, tmp_path)] == "src"


def test_before_agent_refreshes_changed_and_deleted_instructions(tmp_path: Path) -> None:
    nested = tmp_path / "src" / "feature"
    nested.mkdir(parents=True)
    ambient = tmp_path / "AGENTS.md"
    discovered = nested / "AGENTS.md"
    ambient.write_text("old root", encoding="utf-8")
    discovered.write_text("old feature", encoding="utf-8")
    middleware = _middleware(tmp_path)
    state = middleware.before_agent({}, None, {})  # type: ignore[arg-type]
    result = middleware.wrap_tool_call(
        _request("read_file", _virtual(nested / "code.py", tmp_path), state),
        lambda request: ToolMessage(content="file", tool_call_id=request.tool_call["id"]),
    )
    assert isinstance(result, Command)
    assert result.update is not None
    prior = {"project_instructions": result.update["project_instructions"]}
    ambient.write_text("new root", encoding="utf-8")
    discovered.unlink()

    update = middleware.before_agent(prior, None, {})  # type: ignore[arg-type]

    contents = update["project_instructions"]["contents"]
    assert contents[_virtual(ambient, tmp_path)] == "new root"
    assert _virtual(discovered, tmp_path) not in contents


def test_parallel_discovery_merge_does_not_resurrect_removed_files() -> None:
    left = ProjectInstructionsData(
        project_root="/project",
        cwd="/project",
        contents={"/project/removed/AGENTS.md": "old"},
        scopes={"/project/removed/AGENTS.md": "/project/removed"},
    )
    removed = ProjectInstructionsData(
        project_root="/project",
        cwd="/project",
        contents={},
        scopes={},
        removed={"/project/removed/AGENTS.md"},
    )
    discovered = ProjectInstructionsData(
        project_root="/project",
        cwd="/project",
        contents={"/project/new/AGENTS.md": "new"},
        scopes={"/project/new/AGENTS.md": "/project/new"},
    )

    merged = _merge_project_instructions(left, removed)
    merged = _merge_project_instructions(merged, discovered)

    assert merged["contents"] == {"/project/new/AGENTS.md": "new"}


def test_execute_retries_after_scanning_nested_instructions(tmp_path: Path) -> None:
    nested = tmp_path / "src" / "feature"
    nested.mkdir(parents=True)
    instructions = nested / "AGENTS.md"
    instructions.write_text("feature", encoding="utf-8")
    middleware = _middleware(tmp_path)
    state = middleware.before_agent({}, None, {})  # type: ignore[arg-type]
    calls = 0

    def handler(request: ToolCallRequest) -> ToolMessage:
        nonlocal calls
        calls += 1
        return ToolMessage(content="executed", tool_call_id=request.tool_call["id"])

    result = middleware.wrap_tool_call(
        _request("execute", None, state, args={"command": "touch src/feature/x"}),
        handler,
    )

    assert calls == 0
    assert isinstance(result, Command)
    assert result.update is not None
    assert result.update["project_instructions"]["contents"][_virtual(instructions, tmp_path)] == "feature"
