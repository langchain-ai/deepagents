"""Tests for durable dcode workspace bindings."""

from __future__ import annotations

import asyncio
import sqlite3

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


async def test_binding_rejects_resource_policy_change(tmp_path) -> None:
    """A thread cannot silently change its privileged resource policy."""
    binding = await bind_thread_workspace(
        "thread-1",
        str(tmp_path),
        {"auto_approve": False},
        config_fingerprint="config-a",
    )

    with pytest.raises(WorkspaceConflictError, match="already bound"):
        await bind_thread_workspace(
            "thread-1",
            str(tmp_path),
            {"auto_approve": True},
            config_fingerprint="config-b",
        )
    with pytest.raises(WorkspaceConflictError, match="configuration does not match"):
        await require_thread_workspace(
            "thread-1", binding.to_payload(), config_fingerprint="config-b"
        )


async def test_binding_persists_only_non_secret_policy(
    tmp_path, workspace_database
) -> None:
    """Durable bindings do not contain model credentials or prompt material."""
    from deepagents_code._server_config import ServerConfig

    config = ServerConfig(
        model_params={"api_key": "secret-value"},
        system_prompt="secret-prompt",
        auto_approve=True,
    ).to_workspace_payload()
    binding = await bind_thread_workspace("thread-1", str(tmp_path), config)

    with sqlite3.connect(workspace_database) as conn:
        stored = conn.execute(
            "SELECT workspace_config_json FROM dcode_thread_workspaces"
        ).fetchone()[0]
    assert "secret-value" not in stored
    assert "secret-prompt" not in stored
    assert binding.workspace_config()["auto_approve"] is True


async def test_current_schema_migrates_on_reopen(tmp_path, workspace_database) -> None:
    """Databases created before policy fingerprinting upgrade in place."""
    with sqlite3.connect(workspace_database) as conn:
        conn.execute(
            """
            CREATE TABLE dcode_thread_workspaces (
                thread_id TEXT PRIMARY KEY NOT NULL,
                schema_version INTEGER NOT NULL,
                workspace_id TEXT NOT NULL,
                cwd TEXT NOT NULL,
                project_root TEXT,
                generation INTEGER NOT NULL,
                resource_key TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
    config = {"enable_shell": True}
    binding = await bind_thread_workspace("thread-1", str(tmp_path), config)

    assert binding.schema_version == 2
    assert binding.config_fingerprint
    assert (await require_thread_workspace("thread-1", binding.to_payload())) == binding
    with sqlite3.connect(workspace_database) as conn:
        stored_version = conn.execute(
            "SELECT schema_version FROM dcode_thread_workspaces WHERE thread_id = ?",
            ("thread-1",),
        ).fetchone()[0]
    assert stored_version == 2


def test_relative_workspace_is_rejected() -> None:
    """Client-controlled relative paths never inherit the server cwd."""
    from deepagents_code.workspace import resolve_workspace

    with pytest.raises(ValueError, match="absolute path"):
        resolve_workspace("relative/path", {})
