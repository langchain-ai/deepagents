"""Verify observed Git work reaches real agent trace trees retrospectively."""

from __future__ import annotations

import subprocess
from copy import deepcopy
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
from deepagents.backends.filesystem import FilesystemBackend
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langchain_core.tracers.langchain import LangChainTracer
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from langsmith import Client, tracing_context
from langsmith.run_trees import RunTree

from deepagents_code import git_trace, turn_trace
from deepagents_code._fake_models import _ToolBindingFakeModel
from deepagents_code.git_trace import GitTraceMiddleware

if TYPE_CHECKING:
    from pathlib import Path

    from langchain_core.runnables import RunnableConfig
    from langchain_core.tools import BaseTool

    from deepagents_code.git_evidence import GitSnapshot


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def checkout(tmp_path):
    _git(tmp_path, "init", "-b", "main")
    _git(tmp_path, "config", "user.name", "Test")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "remote", "add", "origin", "https://github.com/example/work.git")
    (tmp_path / "work.txt").write_text("original", encoding="utf-8")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-m", "initial")
    return tmp_path


@pytest.fixture
def transport(monkeypatch):
    monkeypatch.setattr(turn_trace, "_turns", turn_trace.OrderedDict())
    client = MagicMock(spec=Client)
    saved = {}
    initial = {}
    patches = []

    def create(**kwargs: object) -> None:
        initial[str(kwargs["id"])] = deepcopy(kwargs)
        saved[str(kwargs["id"])] = deepcopy(kwargs)

    def update(run_id=None, **kwargs: object) -> None:
        identity = str(run_id or kwargs.pop("run_id"))
        patches.append((identity, deepcopy(kwargs)))
        saved[identity].update(deepcopy(kwargs))

    client.create_run.side_effect = create
    client.update_run.side_effect = update
    return LangChainTracer(client=client), saved, initial, patches


def _config(tracer: LangChainTracer, turn: str = "one") -> RunnableConfig:
    return {
        "callbacks": [tracer],
        "configurable": {"thread_id": "thread"},
        "metadata": {
            "thread_id": "thread",
            "turn_id": turn,
            "turn_number": 1 if turn == "one" else 2,
            "git_repo": "example/user-start",
            "git_branch": "user-start-branch",
            "ls_git_repo": "example/user-start",
            "ls_git_branch": "user-start-branch",
            "unrelated": "keep",
            "ls_attribution_identifier": "configured-user",
        },
    }


def _edit_tool(checkout: Path) -> BaseTool:
    @tool
    def edit(file_path: str) -> str:
        """Create a work branch and edit the requested checkout file."""
        _git(checkout, "switch", "-c", "feature/observed")
        RunnableLambda(
            lambda _: (checkout / "work.txt").write_text("changed", encoding="utf-8")
        ).invoke(file_path)
        return "work completed"

    return edit


def _messages(checkout: Path) -> list[AIMessage]:
    return [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "edit",
                    "args": {"file_path": str(checkout / "work.txt")},
                    "id": "edit-once",
                }
            ],
        ),
        AIMessage(content="done"),
        AIMessage(content="no changes"),
    ]


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_real_agent_backfills_approval_siblings_and_isolates_next_turn(
    checkout, transport, asynchronous
) -> None:
    tracer, saved, initial, patches = transport
    middleware = GitTraceMiddleware(
        lambda: FilesystemBackend(root_dir=checkout, virtual_mode=False), str(checkout)
    )
    agent = create_agent(
        _ToolBindingFakeModel(messages=iter(_messages(checkout))),
        tools=[_edit_tool(checkout)],
        middleware=[middleware, HumanInTheLoopMiddleware(interrupt_on={"edit": True})],
        checkpointer=InMemorySaver(),
    )
    config = _config(tracer)
    value = {"messages": [{"role": "user", "content": "edit the checkout"}]}
    first = (
        await agent.ainvoke(value, config)
        if asynchronous
        else agent.invoke(value, config)
    )
    assert first["__interrupt__"]
    earlier = set(saved)
    assert any(saved[key]["run_type"] == "llm" for key in earlier)
    resume = Command(resume={"decisions": [{"type": "approve"}]})
    result = (
        await agent.ainvoke(resume, config)
        if asynchronous
        else agent.invoke(resume, config)
    )
    assert result["messages"][-1].content == "done"
    assert (checkout / "work.txt").read_text() == "changed"
    roots = [run for run in saved.values() if not run.get("parent_run_id")]
    assert len(roots) == 2
    assert {run["run_type"] for run in saved.values()} == {"chain", "llm", "tool"}
    assert any(run["name"] == "RunnableLambda" for run in saved.values())
    for run in saved.values():
        metadata = run["extra"]["metadata"]
        assert metadata["ls_git_repo"] == "example/work"
        assert metadata["ls_git_branch"] == "feature/observed"
        assert metadata["git_repo"] == "example/user-start"
        assert metadata["git_branch"] == "user-start-branch"
        assert metadata["unrelated"] == "keep"
        assert metadata["ls_attribution_identifier"] == "configured-user"
        assert metadata["turn_id"] == "one"
    assert all(
        initial[key]["extra"]["metadata"]["ls_git_repo"] == "example/user-start"
        for key in earlier
    )
    assert any(
        key in earlier
        and patch.get("extra", {}).get("metadata", {}).get("ls_git_repo")
        == "example/work"
        for key, patch in patches
    )
    completed = deepcopy(saved)
    next_value = {"messages": [{"role": "user", "content": "explain only"}]}
    result = (
        await agent.ainvoke(next_value, _config(tracer, "two"))
        if asynchronous
        else agent.invoke(next_value, _config(tracer, "two"))
    )
    assert result["messages"][-1].content == "no changes"
    assert all(saved[key] == run for key, run in completed.items())
    later = [run for key, run in saved.items() if key not in completed]
    assert later
    assert all("ls_git_repo" not in run["extra"]["metadata"] for run in later)
    assert all(
        run["extra"]["metadata"]["git_repo"] == "example/user-start" for run in later
    )


@pytest.mark.parametrize("failure", ["probe", "unknown_before", "update"])
async def test_optional_observation_failure_preserves_real_agent_output(
    checkout, transport, monkeypatch, failure
) -> None:
    tracer, saved, _, _ = transport
    successful_probes = []
    if failure == "unknown_before":
        original = git_trace.aprobe_git
        failed = False

        async def probe(backend, cwd, args=None) -> list[GitSnapshot]:
            nonlocal failed
            if not failed:
                failed = True
                message = "initial snapshot unavailable"
                raise RuntimeError(message)
            result = await original(backend, cwd, args)
            successful_probes.extend(result)
            return result

        monkeypatch.setattr(git_trace, "aprobe_git", probe)
    elif failure == "probe":
        monkeypatch.setattr(
            git_trace,
            "aprobe_git",
            MagicMock(side_effect=RuntimeError("probe unavailable")),
        )
    else:
        monkeypatch.setattr(
            turn_trace.TurnTrace,
            "update",
            MagicMock(side_effect=RuntimeError("telemetry unavailable")),
        )
    middleware = GitTraceMiddleware(
        lambda: FilesystemBackend(root_dir=checkout, virtual_mode=False), str(checkout)
    )
    agent = create_agent(
        _ToolBindingFakeModel(messages=iter(_messages(checkout))),
        tools=[_edit_tool(checkout)],
        middleware=[middleware],
    )
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "edit the checkout"}]},
        _config(tracer),
    )
    assert result["messages"][-1].content == "done"
    assert (checkout / "work.txt").read_text() == "changed"
    assert any(message.content == "work completed" for message in result["messages"])
    if failure == "unknown_before":
        assert successful_probes
        assert any(snapshot.repository is not None for snapshot in successful_probes)
        assert all(
            "ls_git_repo" not in run["extra"]["metadata"] for run in saved.values()
        )


async def test_mismatching_enclosing_trace_is_not_backfilled(
    checkout, transport
) -> None:
    tracer, saved, initial, patches = transport
    outer = RunTree(
        name="another-turn",
        run_type="chain",
        inputs={},
        extra={
            "metadata": {
                "thread_id": "thread",
                "turn_id": "other",
                "ls_git_repo": "untouched",
            }
        },
        ls_client=tracer.client,
    )
    before = deepcopy(outer.metadata)
    middleware = GitTraceMiddleware(
        lambda: FilesystemBackend(root_dir=checkout, virtual_mode=False), str(checkout)
    )
    agent = create_agent(
        _ToolBindingFakeModel(messages=iter(_messages(checkout))),
        tools=[_edit_tool(checkout)],
        middleware=[middleware],
    )
    with tracing_context(parent=outer):
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "edit the checkout"}]},
            _config(tracer),
        )
    assert result["messages"][-1].content == "done"
    assert outer.metadata == before
    assert outer.end_time is None
    assert all(identity != str(outer.id) for identity, _ in patches)
    assert saved
    assert all(
        run["extra"]["metadata"]["ls_git_repo"] == "example/work"
        for run in saved.values()
    )
    assert all(
        run["extra"]["metadata"]["turn_id"]
        == initial[identity]["extra"]["metadata"]["turn_id"]
        for identity, run in saved.items()
    )
