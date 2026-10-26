"""Tests for durable dcode workspace bindings."""

from __future__ import annotations

import asyncio
import sqlite3
from contextlib import closing

import pytest

from deepagents_code.workspace import (
    WorkspaceConflictError,
    bind_thread_workspace,
    get_thread_workspace,
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

    with pytest.raises(WorkspaceConflictError, match="configuration changed"):
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


async def test_binding_rejects_resolved_project_policy_change(tmp_path) -> None:
    binding = await bind_thread_workspace(
        "thread-1",
        str(tmp_path),
        {"trust_project_mcp": False},
        config_fingerprint="config-a",
    )

    expected = (
        "Cannot host this workspace because the project's resolved policy differs "
        "from the policy recorded when this workspace was bound."
    )
    with pytest.raises(WorkspaceConflictError) as exc_info:
        await bind_thread_workspace(
            "thread-1",
            str(tmp_path),
            {"trust_project_mcp": True},
            config_fingerprint="config-b",
        )

    assert str(exc_info.value) == expected
    assert binding.workspace_config()["trust_project_mcp"] is False


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

    with closing(sqlite3.connect(workspace_database)) as conn, conn:
        stored = conn.execute(
            "SELECT workspace_config_json FROM dcode_thread_workspaces"
        ).fetchone()[0]
    assert "secret-value" not in stored
    assert "secret-prompt" not in stored
    assert binding.workspace_config()["auto_approve"] is True


async def test_current_schema_migrates_on_reopen(tmp_path, workspace_database) -> None:
    """Databases created before policy fingerprinting upgrade in place."""
    with closing(sqlite3.connect(workspace_database)) as conn, conn:
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

    assert binding.schema_version == 4
    assert binding.config_fingerprint
    assert binding.policy_fingerprint
    assert (await require_thread_workspace("thread-1", binding.to_payload())) == binding
    with closing(sqlite3.connect(workspace_database)) as conn, conn:
        stored_version = conn.execute(
            "SELECT schema_version FROM dcode_thread_workspaces WHERE thread_id = ?",
            ("thread-1",),
        ).fetchone()[0]
    assert stored_version == 4


async def test_a_stale_schema_row_rebinds_instead_of_conflicting(
    tmp_path, workspace_database
) -> None:
    """A version-2 row's fingerprint used launch-project policy values.

    Those rows were bound with the launch config's fingerprint regardless of
    directory, so comparing one against a resolved fingerprint reported drift
    forever and the thread could never re-bind.
    """
    binding = await bind_thread_workspace(
        "thread-1", str(tmp_path), {"trust_project_mcp": True}
    )
    with closing(sqlite3.connect(workspace_database)) as conn, conn:
        conn.execute(
            """
            UPDATE dcode_thread_workspaces
            SET schema_version = 2, config_fingerprint = 'stale-launch-fingerprint'
            WHERE thread_id = ?
            """,
            ("thread-1",),
        )

    rebound = await bind_thread_workspace(
        "thread-1", str(tmp_path), {"trust_project_mcp": False}
    )

    assert rebound.schema_version == 4
    assert rebound.workspace_config()["trust_project_mcp"] is False
    assert rebound.config_fingerprint != "stale-launch-fingerprint"
    assert (await require_thread_workspace("thread-1", rebound.to_payload())) == rebound
    assert rebound.workspace_id == binding.workspace_id


@pytest.mark.parametrize(
    ("original", "changed"),
    [
        ({"auto_approve": False}, {"auto_approve": True}),
        ({"shell_allow_list": ["ls"]}, {"shell_allow_list": ["*"]}),
        ({"allow_fs_tools": ["read_file"]}, {"allow_fs_tools": None}),
        ({"no_mcp": True}, {"no_mcp": False}),
    ],
)
async def test_migration_rejects_session_policy_drift(
    tmp_path, workspace_database, original, changed
) -> None:
    """Project migration cannot replace a thread's recorded session controls."""
    original["trust_project_mcp"] = True
    changed["trust_project_mcp"] = False
    await bind_thread_workspace("thread-1", str(tmp_path), original)
    with closing(sqlite3.connect(workspace_database)) as conn, conn:
        conn.execute("UPDATE dcode_thread_workspaces SET schema_version = 2")
    stored = await get_thread_workspace("thread-1")

    with pytest.raises(WorkspaceConflictError, match="server configuration changed"):
        await bind_thread_workspace("thread-1", str(tmp_path), changed)

    assert await get_thread_workspace("thread-1") == stored


async def test_legacy_v3_row_migrates_only_when_nothing_changed(
    tmp_path, workspace_database
) -> None:
    """A v3 binding upgrades to v4 only when its full fingerprint still matches.

    Exact fingerprint equality proves the whole (model + policy) config is
    unchanged, so upgrading is safe. The row is rewritten with the new policy
    and runtime fingerprints; conversation checkpoints live in separate tables
    and are untouched.
    """
    config = {"enable_shell": True, "auto_approve": False}
    binding = await bind_thread_workspace("thread-1", str(tmp_path), config)
    # Simulate a v3 row: drop the v4 fingerprints, keep the full fingerprint.
    with closing(sqlite3.connect(workspace_database)) as conn, conn:
        conn.execute(
            """
            UPDATE dcode_thread_workspaces
            SET schema_version = 3, policy_fingerprint = '', runtime_fingerprint = ''
            WHERE thread_id = ?
            """,
            ("thread-1",),
        )

    # Simulate surviving conversation history in a separate table; the
    # migration must leave it byte-for-byte intact.
    with closing(sqlite3.connect(workspace_database)) as conn, conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS checkpoints (thread_id TEXT, state TEXT)"
        )
        conn.execute(
            "INSERT INTO checkpoints VALUES ('thread-1', 'conversation-state')"
        )

    rebound = await bind_thread_workspace("thread-1", str(tmp_path), config)

    assert rebound.schema_version == 4
    assert rebound.policy_fingerprint
    assert rebound.runtime_fingerprint
    assert rebound.workspace_id == binding.workspace_id
    with closing(sqlite3.connect(workspace_database)) as conn, conn:
        history = conn.execute(
            "SELECT state FROM checkpoints WHERE thread_id = 'thread-1'"
        ).fetchone()
    assert history == ("conversation-state",)


@pytest.mark.parametrize(
    ("original", "changed"),
    [
        ({"auto_approve": False}, {"auto_approve": True}),
        ({"trust_project_extensions": True}, {"trust_project_extensions": False}),
        ({"trust_project_mcp": True}, {"trust_project_mcp": False}),
        ({"extension_paths": ["/original"]}, {"extension_paths": ["/changed"]}),
        ({"sandbox_setup": "/original"}, {"sandbox_setup": "/changed"}),
    ],
)
async def test_legacy_v3_row_with_policy_change_is_rejected(
    tmp_path, workspace_database, original, changed
) -> None:
    """A v3 row whose full fingerprint no longer matches cannot be proven safe."""
    await bind_thread_workspace("thread-1", str(tmp_path), original)
    with closing(sqlite3.connect(workspace_database)) as conn, conn:
        conn.execute(
            """
            UPDATE dcode_thread_workspaces
            SET schema_version = 3, policy_fingerprint = '', runtime_fingerprint = ''
            WHERE thread_id = ?
            """,
            ("thread-1",),
        )

    # A real policy change flips the full fingerprint, so the legacy row is
    # unprovable and is rejected rather than assumed equivalent.
    with pytest.raises(WorkspaceConflictError):
        await bind_thread_workspace("thread-1", str(tmp_path), changed)


async def test_model_only_change_does_not_rebind_current_schema(tmp_path) -> None:
    """On the current schema a model change keeps the binding (policy unchanged).

    The payload omits model settings, so the policy fingerprint is stable
    across a model switch; only the runtime fingerprint changes.
    """
    from deepagents_code._server_config import ServerConfig

    bound = ServerConfig(model="anthropic:claude-a", auto_approve=False)
    binding = await bind_thread_workspace(
        "thread-1",
        str(tmp_path),
        bound.to_workspace_payload(),
        config_fingerprint=bound.runtime_fingerprint(),
    )

    switched = ServerConfig(model="openai:gpt-b", auto_approve=False)
    rebound = await bind_thread_workspace(
        "thread-1",
        str(tmp_path),
        switched.to_workspace_payload(),
        config_fingerprint=switched.runtime_fingerprint(),
    )

    assert rebound.policy_fingerprint == binding.policy_fingerprint
    assert rebound.workspace_id == binding.workspace_id


def test_relative_workspace_is_rejected() -> None:
    """Client-controlled relative paths never inherit the server cwd."""
    from deepagents_code.workspace import resolve_workspace

    with pytest.raises(ValueError, match="absolute path"):
        resolve_workspace("relative/path", {})
