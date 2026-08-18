"""Persistent trust decisions for project-authored Python extensions."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from filelock import FileLock, Timeout

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)

_STORE_VERSION: Literal[1] = 1
_LOCK_TIMEOUT_SECONDS = 5.0
_THREAD_LOCK = threading.Lock()


def _default_store_path() -> Path:
    """Return the app-managed extension trust store path."""
    from deepagents_code.model_config import DEFAULT_STATE_DIR

    return DEFAULT_STATE_DIR / "extension_trust.json"


def _project_key(project_root: Path | str) -> str:
    """Return the canonical trust key for a project root."""
    return str(Path(project_root).expanduser().resolve())


@contextmanager
def _store_lock(path: Path) -> Iterator[None]:
    """Serialize read-merge-write trust store updates.

    Args:
        path: Trust store path.

    Yields:
        Control while both process-local and cross-process locks are held.
    """
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name != "nt":
        path.parent.chmod(0o700)
    lock = FileLock(
        str(path.with_name(f"{path.name}.lock")),
        timeout=_LOCK_TIMEOUT_SECONDS,
        thread_local=False,
    )
    with _THREAD_LOCK, lock:
        yield


def _parse_projects(data: object, *, path: Path) -> dict[str, Any]:
    """Validate the trust store structure and entries.

    Args:
        data: Parsed JSON value.
        path: Store path for error context.

    Returns:
        Structurally valid project entries.

    Raises:
        TypeError: If the top-level value or project map has the wrong type.
        ValueError: If the schema version is unsupported.
    """
    if not isinstance(data, dict):
        msg = f"Extension trust store must be an object: {path}"
        raise TypeError(msg)
    if data.get("version") != _STORE_VERSION:
        msg = f"Unsupported extension trust store version: {path}"
        raise ValueError(msg)
    projects = data.get("projects")
    if not isinstance(projects, dict):
        msg = f"Extension trust store projects must be an object: {path}"
        raise TypeError(msg)
    return {
        key: value
        for key, value in projects.items()
        if isinstance(key, str)
        and isinstance(value, dict)
        and isinstance(value.get("trusted_at"), str)
    }


def _load_projects(path: Path, *, strict: bool = False) -> dict[str, Any]:
    """Load and validate the persisted project map.

    Args:
        path: Trust store path.
        strict: Raise instead of treating malformed data as untrusted.

    Returns:
        Valid project entries, or an empty mapping when reading fails safely.

    Raises:
        OSError: If strict reading fails.
        TypeError: If strict structural validation fails.
        UnicodeDecodeError: If strict UTF-8 decoding fails.
        json.JSONDecodeError: If strict JSON parsing fails.
        ValueError: If strict parsing or structural validation fails.
    """
    try:
        data: object = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        if strict:
            raise
        logger.warning("Could not read extension trust store %s", path, exc_info=True)
        return {}

    try:
        return _parse_projects(data, path=path)
    except (TypeError, ValueError):
        if strict:
            raise
        logger.warning("Could not read extension trust store %s", path, exc_info=True)
        return {}


def _write_projects(path: Path, projects: dict[str, Any]) -> None:
    """Atomically write trust data with restrictive permissions.

    Args:
        path: Destination trust store path.
        projects: Canonical project trust entries.
    """
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                {"version": _STORE_VERSION, "projects": projects},
                handle,
                sort_keys=True,
            )
            handle.write("\n")
        if os.name != "nt":
            temporary.chmod(0o600)
        temporary.replace(path)
        if os.name != "nt":
            path.chmod(0o600)
    except BaseException:
        with suppress(OSError):
            temporary.unlink()
        raise


def is_project_extensions_trusted(
    project_root: Path | str,
    *,
    store_path: Path | None = None,
) -> bool:
    """Return whether a project may execute its extension code.

    Args:
        project_root: Project root to inspect.
        store_path: Alternate store path for tests.

    Returns:
        `True` only for a valid persisted decision on the canonical root.
    """
    projects = _load_projects(store_path or _default_store_path())
    return _project_key(project_root) in projects


def trust_project_extensions(
    project_root: Path | str,
    *,
    store_path: Path | None = None,
) -> bool:
    """Persist permission to execute one project's extension code.

    Args:
        project_root: Project root to trust.
        store_path: Alternate store path for tests.

    Returns:
        `True` when the decision was persisted. Failures return `False` and must
            not be interpreted as an implicit future grant.
    """
    path = store_path or _default_store_path()
    try:
        with _store_lock(path):
            projects = _load_projects(path, strict=True)
            projects[_project_key(project_root)] = {
                "trusted_at": datetime.now(UTC).isoformat()
            }
            _write_projects(path, projects)
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
        Timeout,
    ):
        logger.exception("Could not save extension trust store %s", path)
        return False
    return True
