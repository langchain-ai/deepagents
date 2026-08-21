"""Persistent per-project record of directories whose `.env` should be skipped.

Separate from the hooks trust store (`hooks/trust.py`): choosing not to load a
project's `.env` is an independent decision from trusting its hooks, so the two
stores do not share state. The shape and concurrency model (versioned JSON,
atomic write, process + cross-process locking) mirror that module.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from filelock import FileLock, Timeout
from pydantic import BaseModel, ConfigDict, ValidationError

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)

_STORE_VERSION: Literal[1] = 1
_STORE_LOCK_TIMEOUT_SECONDS = 5.0
_STORE_THREAD_LOCK = threading.Lock()
"""Process-local guard for skip-store mutations (see `hooks.trust`)."""


class DotenvSkipEntry(BaseModel):
    """Persisted record for one canonical project root whose `.env` is skipped."""

    model_config = ConfigDict(extra="ignore")

    skipped_at: str
    """UTC ISO-8601 timestamp when the skip was recorded."""


class DotenvSkipStore(BaseModel):
    """Versioned on-disk store of project roots that skip the project `.env`."""

    model_config = ConfigDict(extra="ignore")

    version: Literal[1] = _STORE_VERSION
    """Schema version; unsupported versions are ignored on read."""

    projects: dict[str, DotenvSkipEntry] = {}
    """Map of canonical project roots to skip entries."""


def _default_store_path() -> Path:
    from deepagents_code.model_config import DEFAULT_STATE_DIR

    return DEFAULT_STATE_DIR / "dotenv_skip.json"


def _project_key(project_root: Path | str) -> str:
    return str(Path(project_root).expanduser().resolve())


def skip_key_for_start_path(start_path: Path | None) -> str | None:
    """Resolve the skip-store key that governs the `.env` found from a directory.

    The key is the parent of the discovered `.env` file — the directory that
    actually owns the file — not the invocation directory. A non-Git project
    has no `ProjectContext.project_root`, so keying on `user_cwd` would miss
    the ancestor `.env` that `_find_dotenv_from_start_path` walks up to; keying
    on the file's parent makes "never load" follow the file whether the launch
    is from its own directory or a subdirectory.

    Args:
        start_path: Directory to start `.env` discovery from; cwd when `None`.

    Returns:
        The canonical key for the discovered `.env`, or `None` when no project
        `.env` is present (nothing to skip).
    """
    from deepagents_code.config import _find_dotenv_from_start_path

    try:
        dotenv_path = _find_dotenv_from_start_path(
            Path(start_path) if start_path is not None else Path.cwd()
        )
    except OSError:
        logger.debug("Could not locate a project .env for the skip key")
        return None
    if dotenv_path is None:
        return None
    return _project_key(dotenv_path.parent)


def _store_lock_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.lock")


@contextmanager
def _store_lock(path: Path) -> Iterator[None]:
    """Serialize read-merge-write updates to the dotenv skip store.

    Combines a single process-local threading lock with a cross-process
    `FileLock` on a sibling `.lock` file so concurrent `dcode` processes cannot
    drop each other's entries. Callers handle `filelock.Timeout` and `OSError`.
    """
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name != "nt":
        path.parent.chmod(0o700)
    file_lock = FileLock(
        str(_store_lock_path(path)),
        timeout=_STORE_LOCK_TIMEOUT_SECONDS,
        thread_local=False,
    )
    with _STORE_THREAD_LOCK, file_lock:
        yield


def _parse_projects(raw_projects: object, *, path: Path) -> dict[str, DotenvSkipEntry]:
    """Parse skip entries, skipping structurally invalid ones.

    Returns:
        Validated project map. Empty when `raw_projects` is missing or not a
        mapping.

    Raises:
        TypeError: When `raw_projects` is present but not a mapping.
    """
    if raw_projects is None:
        return {}
    if not isinstance(raw_projects, dict):
        msg = f"dotenv skip store projects must be an object: {path}"
        raise TypeError(msg)

    projects: dict[str, DotenvSkipEntry] = {}
    for key, value in raw_projects.items():
        if not isinstance(key, str):
            logger.warning(
                "Skipping non-string dotenv skip project key in %s: %r", path, key
            )
            continue
        try:
            projects[key] = DotenvSkipEntry.model_validate(value)
        except ValidationError as exc:
            logger.warning(
                "Skipping invalid dotenv skip entry for %s in %s: %s", key, path, exc
            )
    return projects


def _load_store(path: Path, *, strict: bool = False) -> DotenvSkipStore:
    """Load and validate the dotenv skip store.

    Args:
        path: Store path.
        strict: When `True`, raise on unreadable or structurally invalid stores
            so writers refuse to overwrite them.

    Returns:
        Validated store. Missing files yield an empty store; unreadable or
        unsupported stores yield an empty store when not `strict`.

    Raises:
        OSError: When `strict` and the file cannot be read.
        TypeError: When `strict` and `projects` or the top-level shape is not a
            mapping.
        ValueError: When `strict` and the version is unsupported.
        json.JSONDecodeError: When `strict` and the file is not valid JSON.
        UnicodeDecodeError: When `strict` and the file is not UTF-8 text.
    """
    try:
        raw_text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return DotenvSkipStore()
    except (OSError, UnicodeDecodeError) as exc:
        if strict:
            raise
        logger.warning("Could not read dotenv skip store %s: %s", path, exc)
        return DotenvSkipStore()

    try:
        data: object = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        if strict:
            raise
        logger.warning("Could not parse dotenv skip store %s: %s", path, exc)
        return DotenvSkipStore()

    if not isinstance(data, dict):
        msg = f"dotenv skip store must be a JSON object: {path}"
        if strict:
            raise TypeError(msg)
        logger.warning(msg)
        return DotenvSkipStore()

    version = data.get("version")
    if version != _STORE_VERSION:
        msg = f"Unsupported dotenv skip store version: {version!r}"
        if strict:
            raise ValueError(msg)
        logger.warning(
            "Ignoring dotenv skip store with unsupported version %r", version
        )
        return DotenvSkipStore()

    try:
        projects = _parse_projects(data.get("projects"), path=path)
    except TypeError:
        if strict:
            raise
        logger.warning(
            "Ignoring dotenv skip store with invalid projects field at %s", path
        )
        return DotenvSkipStore()

    return DotenvSkipStore(version=_STORE_VERSION, projects=projects)


def _write_store(path: Path, store: DotenvSkipStore) -> None:
    """Atomically write the skip store with restrictive permissions."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name != "nt":
        path.parent.chmod(0o700)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(store.model_dump(mode="json"), handle, sort_keys=True)
            handle.write("\n")
        if os.name != "nt":
            tmp_path.chmod(0o600)
        tmp_path.replace(path)
        if os.name != "nt":
            path.chmod(0o600)
    except BaseException:
        with suppress(OSError):
            tmp_path.unlink()
        raise


def is_project_dotenv_skipped(
    project_root: Path | str,
    *,
    store_path: Path | None = None,
) -> bool:
    """Return whether the project `.env` is skipped for a canonical key.

    Args:
        project_root: The skip key to inspect — the parent directory of the
            discovered `.env` (see `skip_key_for_start_path`).
        store_path: Alternate store path for tests.

    Returns:
        `True` when the key is in the skip store.
    """
    path = store_path or _default_store_path()
    store = _load_store(path)
    return _project_key(project_root) in store.projects


def skip_project_dotenv(
    project_root: Path | str,
    *,
    store_path: Path | None = None,
) -> bool:
    """Persist skipping the project `.env` for a canonical project root.

    Args:
        project_root: Project root whose `.env` should be skipped.
        store_path: Alternate store path for tests.

    Returns:
        `True` when the decision was saved.

    Note:
        Failures (unreadable store, lock timeout, I/O errors) return `False`
        without mutating the on-disk store; callers treat `False` as a real
        persistence failure, not an implicit session decision.
    """
    path = store_path or _default_store_path()
    try:
        with _store_lock(path):
            try:
                store = _load_store(path, strict=True)
            except (
                OSError,
                UnicodeDecodeError,
                json.JSONDecodeError,
                TypeError,
                ValueError,
            ):
                logger.exception(
                    "Refusing to overwrite unreadable dotenv skip store %s", path
                )
                return False

            projects = dict(store.projects)
            projects[_project_key(project_root)] = DotenvSkipEntry(
                skipped_at=datetime.now(UTC).isoformat()
            )
            _write_store(
                path,
                DotenvSkipStore(version=_STORE_VERSION, projects=projects),
            )
    except Timeout:
        logger.exception("Timed out waiting to persist dotenv skip store %s", path)
        return False
    except OSError:
        logger.exception("Could not save dotenv skip store %s", path)
        return False
    return True
