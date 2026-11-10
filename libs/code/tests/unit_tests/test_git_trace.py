"""Exercise checkout attribution through the agent middleware lifecycle."""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest
from deepagents.backends.filesystem import FilesystemBackend
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from deepagents_code import git_trace
from deepagents_code._fake_models import _ToolBindingFakeModel
from deepagents_code.git_trace import GitTraceMiddleware

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig
    from langgraph.graph.state import CompiledStateGraph

    from deepagents_code.git_evidence import GitSnapshot


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _repo(path: Path, name: str = "repo") -> Path:
    path.mkdir()
    _git(path, "init", "-b", "main")
    _git(path, "config", "user.name", "Test")
    _git(path, "config", "user.email", "test@example.com")
    _git(path, "remote", "add", "origin", f"https://github.com/example/{name}.git")
    (path / "work.txt").write_text("original")
    _git(path, "add", ".")
    _git(path, "commit", "-m", "initial")
    return path


def _config(turn: str = "turn-1") -> RunnableConfig:
    return {
        "configurable": {"thread_id": "thread-1"},
        "metadata": {"thread_id": "thread-1", "turn_id": turn, "existing": "kept"},
    }


async def test_first_actual_work_pair_and_turn_isolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _repo(tmp_path / "a", "a")
    second = _repo(tmp_path / "b", "b")
    traces: dict[str, MagicMock] = {}

    def capture(config: RunnableConfig) -> MagicMock:
        return traces.setdefault(config["metadata"]["turn_id"], MagicMock())

    monkeypatch.setattr(git_trace, "capture_turn", capture)
    middleware = GitTraceMiddleware(
        lambda: FilesystemBackend(root_dir=tmp_path, virtual_mode=False), str(tmp_path)
    )

    @tool
    def edit(file_path: str) -> str:
        """Edit a file at the observed checkout path."""
        Path(file_path).write_text("edited", encoding="utf-8")
        return "edited"

    responses = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "edit",
                    "args": {"file_path": str(repo / "work.txt")},
                    "id": str(index),
                }
            ],
        )
        for index, repo in enumerate([second, first])
    ]
    agent = create_agent(
        _ToolBindingFakeModel(messages=iter([*responses, AIMessage(content="done")])),
        tools=[edit],
        middleware=[middleware],
    )
    await agent.ainvoke(
        {"messages": [{"role": "user", "content": "edit both"}]}, _config()
    )
    metadata = traces["turn-1"].update.call_args.args[0]
    assert metadata["ls_git_repo"] == "example/b"
    assert metadata["open_swe_git_targets"] == [
        {"repo": "example/b", "branch": "main"},
        {"repo": "example/a", "branch": "main"},
    ]
    calls = traces["turn-1"].update.call_count
    reader = create_agent(
        _ToolBindingFakeModel(messages=iter([AIMessage(content="read only")])),
        middleware=[middleware],
    )
    await reader.ainvoke(
        {"messages": [{"role": "user", "content": "explain"}]}, _config("turn-2")
    )
    assert traces["turn-2"].update.call_args.args[0] == {}
    assert traces["turn-1"].update.call_count == calls


async def test_disabled_tracing_does_not_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(git_trace, "capture_turn", lambda _config: None)
    probe = MagicMock(side_effect=AssertionError("tracing disabled"))
    middleware = GitTraceMiddleware(probe, "/unused")
    agent = create_agent(
        _ToolBindingFakeModel(messages=iter([AIMessage(content="answer")])),
        middleware=[middleware],
    )
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "hi"}]}, _config()
    )
    assert result["messages"][-1].content == "answer"
    probe.assert_not_called()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    ("name", "args", "metadata"),
    [
        ("read_file", {"file_path": "/repo/file"}, None),
        ("ls", {"file_path": "/repo"}, None),
        ("glob", {"file_path": "/repo"}, None),
        ("grep", {"file_path": "/repo"}, None),
        ("write_todos", {}, None),
        ("ask_user", {}, None),
        ("custom_read", {"file_path": "/repo/file"}, {"readOnlyHint": True}),
    ],
)
async def test_traced_readonly_tools_do_not_touch_backend(
    monkeypatch: pytest.MonkeyPatch,
    asynchronous: bool,
    name: str,
    args: dict[str, str],
    metadata: dict[str, bool] | None,
) -> None:
    trace = MagicMock()
    monkeypatch.setattr(git_trace, "capture_turn", lambda _config: trace)
    backend = MagicMock()
    middleware = GitTraceMiddleware(backend, "/repo")

    @tool(name)
    def inspect_file(file_path: str = "") -> str:
        """Return the requested path without changing the checkout."""
        return f"read:{file_path}"

    inspect_file.metadata = metadata
    model = _ToolBindingFakeModel(
        messages=iter(
            [
                AIMessage(
                    content="", tool_calls=[{"name": name, "args": args, "id": "read"}]
                ),
                AIMessage(content="done"),
            ]
        )
    )
    agent = create_agent(model, tools=[inspect_file], middleware=[middleware])
    value = {"messages": [{"role": "user", "content": "inspect"}]}
    result = (
        await agent.ainvoke(value, _config())
        if asynchronous
        else agent.invoke(value, _config())
    )
    assert result["messages"][-1].content == "done"
    assert result["messages"][-2].content == f"read:{args.get('file_path', '')}"
    backend.assert_not_called()


async def test_cancelled_tool_does_not_wait_for_after_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(git_trace, "capture_turn", lambda _config: MagicMock())
    middleware = GitTraceMiddleware(MagicMock(), "/repo")
    probe = AsyncMock(return_value=[])
    monkeypatch.setattr(middleware, "_aprobe", probe)
    request = MagicMock()
    request.tool_call = {"name": "execute", "args": {"command": "build"}}
    request.runtime.config = _config()
    cancelled = asyncio.CancelledError("user cancelled")
    handler = AsyncMock(side_effect=cancelled)
    with pytest.raises(asyncio.CancelledError) as error:
        await middleware.awrap_tool_call(request, handler)
    assert error.value is cancelled
    probe.assert_awaited_once()


def test_failed_tool_does_not_run_after_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(git_trace, "capture_turn", lambda _config: MagicMock())
    middleware = GitTraceMiddleware(MagicMock(), "/repo")
    probe = MagicMock(return_value=[])
    monkeypatch.setattr(middleware, "_probe", probe)
    request = MagicMock()
    request.tool_call = {"name": "execute", "args": {"command": "build"}}
    request.runtime.config = _config()
    handler = MagicMock(side_effect=RuntimeError("tool failed"))
    with pytest.raises(RuntimeError, match="tool failed"):
        middleware.wrap_tool_call(request, handler)
    probe.assert_called_once()


def _tool_agent(
    middleware: GitTraceMiddleware, calls: list[tuple[str, dict[str, str]]]
) -> CompiledStateGraph:
    @tool
    def execute(command: str) -> str:
        """Run a shell command in the checkout."""
        return subprocess.run(
            command, shell=True, check=True, capture_output=True, text=True
        ).stdout

    @tool
    def write_file(file_path: str, content: str) -> str:
        """Write a file in the checkout."""
        Path(file_path).write_text(content, encoding="utf-8")
        return "written"

    responses = [
        AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": str(i)}])
        for i, (name, args) in enumerate(calls)
    ]
    return create_agent(
        _ToolBindingFakeModel(messages=iter([*responses, AIMessage(content="done")])),
        tools=[execute, write_file],
        middleware=[middleware],
    )


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_turn_probes_once_per_call_after_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, asynchronous: bool
) -> None:
    repo = _repo(tmp_path / "repo")
    (repo / "dirty.txt").write_text("preexisting")
    other = _repo(tmp_path / "other", "other")
    (other / "work.txt").write_text("preexisting dirt")
    trace = MagicMock()
    monkeypatch.setattr(git_trace, "capture_turn", lambda _config: trace)
    probes: list[str] = []
    sync_probe, async_probe = git_trace.probe_git, git_trace.aprobe_git

    def probe(backend, cwd: str, args=None) -> list[GitSnapshot]:
        probes.append(cwd)
        return sync_probe(backend, cwd, args)

    async def aprobe(backend, cwd: str, args=None) -> list[GitSnapshot]:
        probes.append(cwd)
        return await async_probe(backend, cwd, args)

    monkeypatch.setattr(git_trace, "probe_git", probe)
    monkeypatch.setattr(git_trace, "aprobe_git", aprobe)
    middleware = GitTraceMiddleware(
        lambda: FilesystemBackend(root_dir=repo, virtual_mode=False), str(repo)
    )
    reads = [("execute", {"command": "git status --short"})] * 3
    calls = [
        *reads,
        ("write_file", {"file_path": str(repo / "work.txt"), "content": "edited"}),
        *reads,
        ("execute", {"command": f"cat {other / 'work.txt'}"}),
    ]
    agent = _tool_agent(middleware, calls)
    value = {"messages": [{"role": "user", "content": "work"}]}
    if asynchronous:
        await agent.ainvoke(value, _config())
    else:
        agent.invoke(value, _config())
    assert len(probes) == len(calls) + 2
    metadata = trace.update.call_args.args[0]
    assert metadata["open_swe_git_targets"] == [
        {"repo": "example/repo", "branch": "main"}
    ]


def test_repeated_probe_failures_back_off_then_disable(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    clock = [0.0]
    monkeypatch.setattr(git_trace.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(git_trace, "capture_turn", lambda _config: MagicMock())
    probe = MagicMock(side_effect=subprocess.TimeoutExpired("git", 2))
    monkeypatch.setattr(git_trace, "probe_git", probe)
    middleware = GitTraceMiddleware(MagicMock(), "/repo")
    request = MagicMock()
    request.tool_call = {"name": "execute", "args": {"command": "build"}}
    request.runtime.config = _config()
    handler = MagicMock(return_value="ok")
    counts = []
    with caplog.at_level("INFO", logger=git_trace.__name__):
        for _ in range(4):
            for _ in range(3):
                assert middleware.wrap_tool_call(request, handler) == "ok"
                counts.append(probe.call_count)
            clock[0] += 3600
    assert counts == [1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3]
    assert handler.call_count == 12
    assert [record.levelname for record in caplog.records] == ["INFO"]


def test_unsupported_backend_disables_probing_immediately(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(git_trace, "capture_turn", lambda _config: MagicMock())
    probe = MagicMock(side_effect=NotImplementedError("no bounded execution"))
    monkeypatch.setattr(git_trace, "probe_git", probe)
    middleware = GitTraceMiddleware(MagicMock(), "/repo")
    request = MagicMock()
    request.tool_call = {"name": "execute", "args": {"command": "build"}}
    request.runtime.config = _config()
    with caplog.at_level("INFO", logger=git_trace.__name__):
        for _ in range(3):
            middleware.wrap_tool_call(request, MagicMock(return_value="ok"))
    assert probe.call_count == 1
    assert [record.levelname for record in caplog.records] == ["INFO"]
    assert git_trace._env_vars.GIT_ATTRIBUTION in caplog.records[0].getMessage()


def test_stale_concurrent_snapshot_cannot_add_targets() -> None:
    from deepagents_code._git import parse_repository_metadata
    from deepagents_code.git_evidence import GitSnapshot as Snapshot

    repository = parse_repository_metadata("https://github.com/example/repo.git")
    main = Snapshot("/repo", repository, "main", "a", "clean")
    feature = Snapshot("/repo", repository, "feature", "a", "clean")
    turn = git_trace._Turn(MagicMock())
    turn.observe([main], 1, 1, created=False)
    turn.observe([feature], 3, 2, created=True)
    turn.observe([main], 2, 3, created=True)
    turn.observe([feature], 4, 4, created=True)
    assert turn.metadata()["open_swe_git_targets"] == [
        {"repo": "example/repo", "branch": "feature"}
    ]
