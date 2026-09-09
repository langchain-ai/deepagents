"""Tests for project instruction discovery."""

from pathlib import Path

from langchain.agents.middleware.types import ToolCallRequest
from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.project_instructions import ProjectInstructionsMiddleware


def _middleware(tmp_path: Path) -> ProjectInstructionsMiddleware:
    return ProjectInstructionsMiddleware(
        backend=FilesystemBackend(root_dir=tmp_path, virtual_mode=True),
        project_root=str(tmp_path),
        cwd=str(tmp_path / "src"),
        backend_root="/",
    )


def _request(name: str, file_path: str, state: dict) -> ToolCallRequest:
    runtime = ToolRuntime(
        state=state,
        context=None,
        tool_call_id="call_1",
        store=None,
        stream_writer=lambda _: None,
        config={},
    )
    return ToolCallRequest(
        runtime=runtime,
        tool_call={"id": "call_1", "name": name, "args": {"file_path": file_path}},
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

    result = middleware.wrap_tool_call(_request("read_file", str(nested / "code.py"), state), handler)

    assert calls == 1
    assert isinstance(result, Command)
    assert result.update is not None
    contents = result.update["project_instructions"]["contents"]
    assert contents[str(nested / "AGENTS.md")] == "feature"


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

    request = _request("write_file", str(nested / "code.py"), state)
    result = middleware.wrap_tool_call(request, handler)

    assert calls == 0
    assert isinstance(result, Command)
    assert result.update is not None
    [message] = result.update["messages"]
    assert "retry the tool call" in message.content

    result = middleware.wrap_tool_call(
        _request(
            "write_file",
            str(nested / "code.py"),
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
    assert f'applies_to="{tmp_path}"' in prompt
    assert "subordinate to system policy and explicit user instructions" in prompt
    assert "root" in prompt
