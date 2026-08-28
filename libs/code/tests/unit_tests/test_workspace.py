"""Tests for durable dcode workspace bindings."""

from __future__ import annotations

import asyncio

import pytest

from deepagents_code.workspace import (
    WorkspaceConflictError,
    bind_thread_workspace,
    require_thread_workspace,
)


@pytest.fixture(autouse=True)
def workspace_database(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Point bindings at an isolated SQLite database."""
    database = tmp_path / "sessions.db"
    monkeypatch.setenv("DEEPAGENTS_CODE_SERVER_DB_PATH", str(database))
    return database


async def test_binding_is_idempotent(tmp_path) -> None:
    """The same thread and effective workspace return one binding."""
    config = {"enable_shell": True}

    first = await bind_thread_workspace("thread-1", str(tmp_path), config)
    second = await bind_thread_workspace("thread-1", str(tmp_path), config)

    assert second == first


async def test_binding_rejects_another_workspace(tmp_path) -> None:
    """A thread cannot silently move to another working directory."""
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    await bind_thread_workspace("thread-1", str(first), {})

    with pytest.raises(WorkspaceConflictError, match="already bound"):
        await bind_thread_workspace("thread-1", str(second), {})


async def test_concurrent_first_bind_has_one_winner(tmp_path) -> None:
    """A first-bind race cannot mix two workspace claims."""
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()

    results = await asyncio.gather(
        bind_thread_workspace("thread-1", str(first), {}),
        bind_thread_workspace("thread-1", str(second), {}),
        return_exceptions=True,
    )

    assert sum(not isinstance(result, Exception) for result in results) == 1
    assert sum(isinstance(result, WorkspaceConflictError) for result in results) == 1


async def test_run_context_must_match_binding(tmp_path) -> None:
    """Execution rejects a stale or substituted workspace descriptor."""
    config = {"enable_shell": True}
    binding = await bind_thread_workspace("thread-1", str(tmp_path), config)

    assert (
        await require_thread_workspace("thread-1", binding.to_payload(), config)
    ) == binding

    changed = binding.to_payload()
    changed["cwd"] = "/tmp/substituted"
    with pytest.raises(WorkspaceConflictError, match="does not match"):
        await require_thread_workspace("thread-1", changed, config)


def test_relative_workspace_is_rejected() -> None:
    """Client-controlled relative paths never inherit the server cwd."""
    from deepagents_code.workspace import resolve_workspace

    with pytest.raises(ValueError, match="absolute path"):
        resolve_workspace("relative/path", {})
