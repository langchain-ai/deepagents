"""Durable, server-authoritative thread workspace bindings."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sqlite3
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path, PurePath
from typing import TYPE_CHECKING, Any, TypedDict, cast

from deepagents_code._env_vars import SERVER_ENV_PREFIX

if TYPE_CHECKING:
    from collections.abc import Mapping

_SCHEMA_VERSION = 3
"""Binding schema generation.

Version 3 resolves project policy per workspace. Version 2 used the launch
config's fingerprint for every directory, so its project policy values may
differ from the newly resolved ones. After checking workspace identity and
session policy, `_bind` migrates old rows instead of rejecting that mismatch
as configuration drift.
"""
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
    config_fingerprint: str


@dataclass(frozen=True)
class WorkspaceBinding:
    """Server-authoritative workspace and resource policy for one thread."""

    schema_version: int
    workspace_id: str
    cwd: str
    project_root: str | None
    generation: int
    resource_key: str
    config_fingerprint: str
    workspace_config_json: str

    def to_payload(self) -> WorkspacePayload:
        """Return the public runtime-context representation."""
        payload = asdict(self)
        payload.pop("workspace_config_json")
        return cast("WorkspacePayload", payload)

    def workspace_config(self) -> dict[str, Any]:
        """Return the persisted, server-authoritative resource policy."""
        return cast("dict[str, Any]", json.loads(self.workspace_config_json))


class WorkspaceConflictError(RuntimeError):
    """A workspace claim or runtime conflicts with server resource policy."""

    @classmethod
    def from_reason(cls, reason: str) -> WorkspaceConflictError:
        """Build a workspace-hosting refusal with a stated reason.

        Returns:
            A conflict with the standard workspace-hosting message.
        """
        msg = f"Cannot host this workspace because {reason}."
        return cls(msg)


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


def canonical_workspace_config(value: object | None) -> tuple[str, str]:
    """Return bounded canonical JSON and its SHA-256 fingerprint.

    Raises:
        TypeError: If the configuration is not an object.
        ValueError: If it cannot be serialized or exceeds the size limit.
    """
    if value is None:
        value = {}
    if not isinstance(value, dict):
        msg = "workspace_config must be an object"
        raise TypeError(msg)
    try:
        serialized = _canonical_json(value)
    except (TypeError, ValueError) as exc:
        msg = "workspace configuration must be JSON serializable"
        raise ValueError(msg) from exc
    if len(serialized) > _MAX_CONFIG_LENGTH:
        msg = "workspace configuration is too large"
        raise ValueError(msg)
    return serialized, hashlib.sha256(serialized.encode()).hexdigest()


def _canonical_json(value: object) -> str:
    """Return JSON with consistent key ordering and spacing for fingerprinting."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def canonical_fingerprint(value: object) -> str:
    """Fingerprint `value` with the canonical workspace serialization.

    Returns:
        The SHA-256 hex digest of the canonical JSON encoding.
    """
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def resolve_workspace(
    cwd: object,
    workspace_config: object | None = None,
    *,
    config_fingerprint: str | None = None,
) -> WorkspaceBinding:
    """Resolve a client-supplied cwd into a canonical workspace binding.

    `cwd` is untrusted and is validated here. `workspace_config` is not: every
    caller passes server-resolved policy. A client claim is verified against
    server policy in `offload_api.workspace` and never reaches this function.

    Returns:
        A canonical, fingerprinted binding including the resource policy.
    """
    config_json, policy_fingerprint = canonical_workspace_config(workspace_config)
    config_fingerprint = config_fingerprint or policy_fingerprint
    canonical_cwd = _canonical_directory(cwd, field="cwd")
    from deepagents_code.project_utils import find_project_root

    project_root = find_project_root(canonical_cwd)
    if project_root is not None:
        project_root = _canonical_directory(str(project_root), field="project_root")
    workspace_id = canonical_fingerprint(
        {
            "cwd": str(canonical_cwd),
            "project_root": str(project_root) if project_root else None,
        }
    )
    resource_key = canonical_fingerprint(
        {"workspace_id": workspace_id, "config_fingerprint": config_fingerprint}
    )
    return WorkspaceBinding(
        schema_version=_SCHEMA_VERSION,
        workspace_id=workspace_id,
        cwd=str(canonical_cwd),
        project_root=str(project_root) if project_root else None,
        generation=1,
        resource_key=resource_key,
        config_fingerprint=config_fingerprint,
        workspace_config_json=config_json,
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
            config_fingerprint TEXT NOT NULL,
            workspace_config_json TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    columns = {
        row[1]
        for row in conn.execute("PRAGMA table_info(dcode_thread_workspaces)").fetchall()
    }
    if "config_fingerprint" not in columns:
        conn.execute(
            "ALTER TABLE dcode_thread_workspaces "
            "ADD COLUMN config_fingerprint TEXT NOT NULL DEFAULT ''"
        )
    if "workspace_config_json" not in columns:
        conn.execute(
            "ALTER TABLE dcode_thread_workspaces "
            "ADD COLUMN workspace_config_json TEXT NOT NULL DEFAULT '{}'"
        )


def _row_binding(row: sqlite3.Row) -> WorkspaceBinding:
    return WorkspaceBinding(
        schema_version=row["schema_version"],
        workspace_id=row["workspace_id"],
        cwd=row["cwd"],
        project_root=row["project_root"],
        generation=row["generation"],
        resource_key=row["resource_key"],
        config_fingerprint=row["config_fingerprint"],
        workspace_config_json=row["workspace_config_json"],
    )


def _is_migratable(existing: WorkspaceBinding) -> bool:
    """Whether a row's recorded policy predates the current schema.

    Older rows may contain launch-project policy instead of workspace policy.
    `_binding_differs` checks workspace identity and recorded session policy
    before allowing migration.

    Returns:
        `True` when the row has no fingerprint yet, or an older schema.
    """
    return not existing.config_fingerprint or existing.schema_version < _SCHEMA_VERSION


def _binding_differs(existing: WorkspaceBinding, proposed: WorkspaceBinding) -> bool:
    if existing.workspace_id != proposed.workspace_id:
        return True
    if not _is_migratable(existing):
        return existing.config_fingerprint != proposed.config_fingerprint
    if not existing.config_fingerprint:
        # Pre-fingerprint rows have no recorded policy to preserve.
        return False
    from deepagents_code._server_config import SESSION_WORKSPACE_FIELDS

    bound_policy = existing.workspace_config()
    proposed_policy = proposed.workspace_config()
    return any(
        bound_policy.get(key) != proposed_policy.get(key)
        for key in SESSION_WORKSPACE_FIELDS
    )


PROJECT_POLICY_DRIFT_REASON = (
    "the project's resolved policy differs from the policy recorded "
    "when this workspace was bound"
)
SERVER_CONFIG_DRIFT_REASON = (
    "the server configuration changed after this workspace was bound"
)


def drifted_project_fields(
    bound_config: Mapping[str, Any],
    current_config: Mapping[str, Any],
) -> list[str]:
    """Name the project-scoped fields that drifted from their binding.

    The refusal these feed is safe either way, but it is not diagnosable
    without the field names: the resolution reads the extension trust store on
    every call, so a transient read failure reports as a policy change. These
    values are paths and booleans, never secrets, so naming them is safe.

    Returns:
        The drifted field names, sorted; empty when the policy is unchanged.
    """
    from deepagents_code._server_config import PROJECT_WORKSPACE_FIELDS

    return sorted(
        key
        for key in PROJECT_WORKSPACE_FIELDS
        if bound_config.get(key) != current_config.get(key)
    )


def _binding_conflict(
    thread_id: str,
    existing: WorkspaceBinding,
    proposed: WorkspaceBinding,
) -> WorkspaceConflictError:
    if existing.workspace_id != proposed.workspace_id:
        return WorkspaceConflictError(
            f"thread {thread_id} is already bound to a different workspace"
        )
    if _is_migratable(existing):
        return WorkspaceConflictError.from_reason(SERVER_CONFIG_DRIFT_REASON)
    drifted = drifted_project_fields(
        existing.workspace_config(),
        proposed.workspace_config(),
    )
    reason = PROJECT_POLICY_DRIFT_REASON if drifted else SERVER_CONFIG_DRIFT_REASON
    return WorkspaceConflictError.from_reason(reason)


def _bind(thread_id: str, proposed: WorkspaceBinding) -> WorkspaceBinding:
    with closing(sqlite3.connect(_database_path(), timeout=5)) as conn, conn:
        conn.row_factory = sqlite3.Row
        conn.execute("BEGIN IMMEDIATE")
        _initialize(conn)
        conn.execute(
            """
            INSERT OR IGNORE INTO dcode_thread_workspaces (
                thread_id, schema_version, workspace_id, cwd, project_root,
                generation, resource_key, config_fingerprint, workspace_config_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                thread_id,
                proposed.schema_version,
                proposed.workspace_id,
                proposed.cwd,
                proposed.project_root,
                proposed.generation,
                proposed.resource_key,
                proposed.config_fingerprint,
                proposed.workspace_config_json,
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
        if _binding_differs(existing, proposed):
            raise _binding_conflict(thread_id, existing, proposed)
        if _is_migratable(existing):
            # Guard on the fingerprint this transaction actually read, so a
            # concurrent migration cannot be overwritten after the fact.
            conn.execute(
                """
                UPDATE dcode_thread_workspaces
                SET schema_version = ?, resource_key = ?, config_fingerprint = ?,
                    workspace_config_json = ?
                WHERE thread_id = ? AND config_fingerprint = ?
                """,
                (
                    proposed.schema_version,
                    proposed.resource_key,
                    proposed.config_fingerprint,
                    proposed.workspace_config_json,
                    thread_id,
                    existing.config_fingerprint,
                ),
            )
            return proposed
        return existing


def _read(thread_id: str) -> WorkspaceBinding | None:
    with closing(sqlite3.connect(_database_path(), timeout=5)) as conn, conn:
        conn.row_factory = sqlite3.Row
        conn.execute("BEGIN IMMEDIATE")
        _initialize(conn)
        row = conn.execute(
            "SELECT * FROM dcode_thread_workspaces WHERE thread_id = ?",
            (thread_id,),
        ).fetchone()
        return _row_binding(row) if row is not None else None


async def bind_thread_workspace(
    thread_id: str,
    cwd: object,
    workspace_config: object | None = None,
    *,
    config_fingerprint: str | None = None,
) -> WorkspaceBinding:
    """Atomically create or verify a thread workspace binding.

    Returns:
        The immutable binding for the thread.

    Raises:
        ValueError: If the thread is invalid.
    """
    if not isinstance(thread_id, str) or not thread_id:
        msg = "thread_id must be non-empty"
        raise ValueError(msg)
    proposed = await asyncio.to_thread(
        resolve_workspace,
        cwd,
        workspace_config,
        config_fingerprint=config_fingerprint,
    )
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
    *,
    config_fingerprint: str | None = None,
) -> WorkspaceBinding:
    """Validate run context against the durable workspace binding.

    Returns:
        The server-authoritative binding and persisted resource policy.

    Raises:
        TypeError: If workspace context is not an object.
        WorkspaceConflictError: If the context, policy, or workspace has changed.
    """
    if not isinstance(payload, dict) or not payload:
        msg = "workspace context is required"
        raise TypeError(msg)
    data = cast("dict[str, Any]", payload)
    claimed_fingerprint = config_fingerprint
    if workspace_config is not None:
        _, claimed_fingerprint = canonical_workspace_config(workspace_config)

    def _require() -> WorkspaceBinding:
        with closing(sqlite3.connect(_database_path(), timeout=5)) as conn, conn:
            conn.row_factory = sqlite3.Row
            conn.execute("BEGIN IMMEDIATE")
            _initialize(conn)
            row = conn.execute(
                "SELECT * FROM dcode_thread_workspaces WHERE thread_id = ?",
                (thread_id,),
            ).fetchone()
            if row is None:
                msg = f"thread {thread_id} has no workspace binding"
                raise WorkspaceConflictError(msg)
            existing = _row_binding(row)
            expected = existing.to_payload()
            if any(data.get(key) != value for key, value in expected.items()):
                msg = f"workspace context does not match thread {thread_id}"
                raise WorkspaceConflictError(msg)
            if (
                claimed_fingerprint is not None
                and claimed_fingerprint != existing.config_fingerprint
            ):
                msg = f"workspace configuration does not match thread {thread_id}"
                raise WorkspaceConflictError(msg)
            return existing

    existing = await asyncio.to_thread(_require)
    if existing.schema_version != _SCHEMA_VERSION:
        msg = f"workspace binding schema is unsupported for thread {thread_id}"
        raise WorkspaceConflictError(msg)
    resolved = await asyncio.to_thread(
        resolve_workspace,
        existing.cwd,
        existing.workspace_config(),
        config_fingerprint=existing.config_fingerprint,
    )
    if resolved.workspace_id != existing.workspace_id:
        msg = f"workspace identity changed for thread {thread_id}"
        raise WorkspaceConflictError(msg)
    return existing
