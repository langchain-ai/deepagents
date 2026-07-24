"""Persistent workspace trust for project-scoped hooks."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from deepagents_code.json_types import JSON_OBJECT_ADAPTER, JsonObject

logger = logging.getLogger(__name__)

_STORE_VERSION = 1


def _default_store_path() -> Path:
    from deepagents_code.model_config import DEFAULT_STATE_DIR

    return DEFAULT_STATE_DIR / "hooks_trust.json"


def _project_key(project_root: Path | str) -> str:
    return str(Path(project_root).expanduser().resolve())


def _load_store(path: Path, *, strict: bool = False) -> JsonObject:
    try:
        data = JSON_OBJECT_ADAPTER.validate_json(path.read_bytes())
    except FileNotFoundError:
        return {}
    except (OSError, ValidationError) as exc:
        if strict:
            raise
        logger.warning("Could not read hooks trust store %s: %s", path, exc)
        return {}
    if not data:
        return {}
    version = data.get("version")
    if version != _STORE_VERSION:
        if strict:
            msg = f"Unsupported hooks trust store version: {version!r}"
            raise ValueError(msg)
        logger.warning(
            "Ignoring hooks trust store with unsupported version %r", version
        )
        return {}
    return data


def is_project_hooks_trusted(
    project_root: Path | str,
    *,
    store_path: Path | None = None,
) -> bool:
    """Return whether project hooks are trusted for a canonical workspace root.

    Args:
        project_root: Workspace root to inspect.
        store_path: Alternate trust store path for tests.

    Returns:
        `True` when the canonical workspace root is trusted.
    """
    path = store_path or _default_store_path()
    projects = _load_store(path).get("projects")
    if not isinstance(projects, dict):
        return False
    entry = projects.get(_project_key(project_root))
    return isinstance(entry, dict) and isinstance(entry.get("trusted_at"), str)


def trust_project_hooks(
    project_root: Path | str,
    *,
    store_path: Path | None = None,
) -> bool:
    """Persist project-hook trust for a workspace root.

    Args:
        project_root: Workspace root to trust.
        store_path: Alternate trust store path for tests.

    Returns:
        `True` when trust was saved.
    """
    path = store_path or _default_store_path()
    try:
        data = _load_store(path, strict=True)
    except (OSError, ValidationError, ValueError):
        logger.exception("Refusing to overwrite unreadable hooks trust store %s", path)
        return False

    projects_value = data.get("projects")
    projects: JsonObject = (
        dict(projects_value) if isinstance(projects_value, dict) else {}
    )
    projects[_project_key(project_root)] = {"trusted_at": datetime.now(UTC).isoformat()}
    payload: JsonObject = {"version": _STORE_VERSION, "projects": projects}

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, sort_keys=True)
                handle.write("\n")
            tmp_path.replace(path)
        except BaseException:
            with suppress(OSError):
                tmp_path.unlink()
            raise
    except OSError:
        logger.exception("Could not save hooks trust store %s", path)
        return False
    return True
