"""Durable thread-to-workspace bindings for the dcode server."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path, PurePath
from typing import Any, TypedDict, cast

from deepagents_code._env_vars import SERVER_ENV_PREFIX

_SCHEMA_VERSION = 1
_MAX_PATH_LENGTH = 4096
_MAX_CONFIG_LENGTH = 64_000


class WorkspacePayload(TypedDict):
    """JSON-safe workspace descriptor carried in LangGraph runtime context."""

    schema_version: int
    workspace_id: str
    cwd: str
    project_root: str | None
    generation: int
    resource_key: str


@dataclass(frozen=True)
class WorkspaceBinding:
    """Server-authoritative workspace assigned to one LangGraph thread."""

    schema_version: int
    workspace_id: str
    cwd: str
    project_root: str | None
    generation: int
    resource_key: str

    def to_payload(self) -> WorkspacePayload:
        """Return the JSON-safe runtime-context representation."""
        return cast("WorkspacePayload", asdict(self))


def _database_path() -> Path:
    value = os.environ.get(f"{SERVER_ENV_PREFIX}DB_PATH")
    if value:
        return Path(value)
    from deepagents_code.sessions import get_db_path

    return get_db_path()


def _canonical_directory(value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value or len(value) > _MAX_PATH_LENGTH:
        msg = f"workspace.{field} must be a non-empty absolute path"
        raise ValueError(msg)
    candidate = Path(value)
    if not candidate.is_absolute() or ".." in PurePath(value).parts:
        msg = f"workspace.{field} must be an absolute path without traversal"
        raise ValueError(msg)
    if os.name != "nt":
        from deepagents.backends.utils import validate_path

        validate_path(value)
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        msg = f"workspace.{field} is unavailable: {value}"
        raise ValueError(msg) from exc
    if not resolved.is_dir():
        msg = f"workspace.{field} is not a directory: {value}"
        raise ValueError(msg)
    if os.name != "nt":
        from deepagents.backends.utils import validate_path

        validate_path(str(resolved))
    return resolved


def _fingerprint(value: object) -> str:
    try:
        serialized = json.dumps(value, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        msg = "workspace configuration must be JSON serializable"
        raise ValueError(msg) from exc
    if len(serialized) > _MAX_CONFIG_LENGTH:
        msg = "workspace configuration is too large"
        raise ValueError(msg)
    return hashlib.sha256(serialized.encode()).hexdigest()


def resolve_workspace(
    cwd: object,
    workspace_config: object | None = None,
) -> WorkspaceBinding:
    """Resolve and validate a client workspace claim.

    Returns:
        A canonical, fingerprinted descriptor.

    Raises:
        ValueError: If a path or configuration field is invalid.
    """
    if workspace_config is not None and not isinstance(workspace_config, dict):
        msg = "workspace_config must be an object"
        raise ValueError(msg)
    canonical_cwd = _canonical_directory(cwd, field="cwd")
    from deepagents_code.project_utils import find_project_root

    project_root = find_project_root(canonical_cwd)
    if project_root is not None:
        project_root = _canonical_directory(str(project_root), field="project_root")
    workspace_id = _fingerprint(
        {
            "cwd": str(canonical_cwd),
            "project_root": str(project_root) if project_root else None,
        }
    )
    resource_key = _fingerprint({"workspace_id": workspace_id})
    return WorkspaceBinding(
        schema_version=_SCHEMA_VERSION,
        workspace_id=workspace_id,
        cwd=str(canonical_cwd),
        project_root=str(project_root) if project_root else None,
        generation=1,
        resource_key=resource_key,
    )


def _initialize(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS dcode_thread_workspaces (
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


def _row_binding(row: sqlite3.Row) -> WorkspaceBinding:
    return WorkspaceBinding(
        schema_version=row["schema_version"],
        workspace_id=row["workspace_id"],
        cwd=row["cwd"],
        project_root=row["project_root"],
        generation=row["generation"],
        resource_key=row["resource_key"],
    )


def _bind(thread_id: str, proposed: WorkspaceBinding) -> WorkspaceBinding:
    with sqlite3.connect(_database_path(), timeout=5) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("BEGIN IMMEDIATE")
        _initialize(conn)
        conn.execute(
            """
            INSERT OR IGNORE INTO dcode_thread_workspaces (
                thread_id, schema_version, workspace_id, cwd, project_root,
                generation, resource_key
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                thread_id,
                proposed.schema_version,
                proposed.workspace_id,
                proposed.cwd,
                proposed.project_root,
                proposed.generation,
                proposed.resource_key,
            ),
        )
        row = conn.execute(
            "SELECT * FROM dcode_thread_workspaces WHERE thread_id = ?",
            (thread_id,),
        ).fetchone()
        if row is None:
            msg = f"workspace binding was not persisted for thread {thread_id}"
            raise RuntimeError(msg)
        existing = _row_binding(row)
        if existing.workspace_id != proposed.workspace_id:
            msg = f"thread {thread_id} is already bound to {existing.cwd}"
            raise WorkspaceConflictError(msg)
        return existing


def _read(thread_id: str) -> WorkspaceBinding | None:
    with sqlite3.connect(_database_path(), timeout=5) as conn:
        conn.row_factory = sqlite3.Row
        _initialize(conn)
        row = conn.execute(
            "SELECT * FROM dcode_thread_workspaces WHERE thread_id = ?",
            (thread_id,),
        ).fetchone()
        return _row_binding(row) if row is not None else None


class WorkspaceConflictError(RuntimeError):
    """A thread was claimed from a different workspace."""


async def bind_thread_workspace(
    thread_id: str,
    cwd: object,
    workspace_config: object | None = None,
) -> WorkspaceBinding:
    """Atomically create or verify a thread workspace binding.

    Returns:
        The immutable binding for the thread.

    Raises:
        ValueError: If the thread or workspace claim is invalid.
    """
    if not isinstance(thread_id, str) or not thread_id:
        msg = "thread_id must be non-empty"
        raise ValueError(msg)
    proposed = await asyncio.to_thread(resolve_workspace, cwd, workspace_config)
    return await asyncio.to_thread(_bind, thread_id, proposed)


async def get_thread_workspace(thread_id: str) -> WorkspaceBinding | None:
    """Read a thread's durable workspace binding.

    Returns:
        The binding, or `None` when the thread is unbound.
    """
    if not isinstance(thread_id, str) or not thread_id:
        return None
    return await asyncio.to_thread(_read, thread_id)


async def require_thread_workspace(
    thread_id: str,
    payload: object,
    workspace_config: object | None = None,
) -> WorkspaceBinding:
    """Validate run context against the durable workspace binding.

    Returns:
        The server-authoritative binding.

    Raises:
        TypeError: If workspace context is not an object.
        WorkspaceConflictError: If the context or current workspace has changed.
    """
    if not isinstance(payload, dict):
        msg = "workspace context is required"
        raise TypeError(msg)
    data = cast("dict[str, Any]", payload)
    existing = await get_thread_workspace(thread_id)
    if existing is None:
        msg = f"thread {thread_id} has no workspace binding"
        raise WorkspaceConflictError(msg)
    expected = existing.to_payload()
    if any(data.get(key) != value for key, value in expected.items()):
        msg = f"workspace context does not match thread {thread_id}"
        raise WorkspaceConflictError(msg)
    resolved = await asyncio.to_thread(
        resolve_workspace, existing.cwd, workspace_config
    )
    if resolved.workspace_id != existing.workspace_id:
        msg = f"workspace identity changed for thread {thread_id}"
        raise WorkspaceConflictError(msg)
    return existing
