"""Trust store for project-scoped extensions.

Extensions are the first project-level dcode resource that *executes* code
(skills, `AGENTS.md`, and agents are data), so the trust decision is part of the
discovery pipeline rather than a check bolted on afterwards: the project source
is not even scanned until the decision is made.

Decisions are per working directory and persisted next to the other state files
under `~/.deepagents/.state/extension_trust.json`. They are app-managed
bookkeeping, not hand-editable configuration.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_STORE_VERSION = 1
"""Schema version stamped into `extension_trust.json`; bump on incompatible changes."""


def _default_store_path() -> Path:
    """Return the default trust store path.

    Returns:
        `~/.deepagents/.state/extension_trust.json`.
    """
    return Path.home() / ".deepagents" / ".state" / "extension_trust.json"


def _project_key(project_root: Path | str) -> str:
    """Return the canonical store key for a project root.

    Returns:
        The absolute, symlink-resolved path as a string.
    """
    return str(Path(project_root).expanduser().resolve(strict=False))


def _load_projects(store_path: Path) -> dict[str, Any]:
    """Read the persisted project map.

    Returns:
        The stored project map, or an empty mapping when the store is missing or
            unreadable.
    """
    try:
        data = json.loads(store_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        logger.warning(
            "Could not read extension trust store %s", store_path, exc_info=True
        )
        return {}
    projects = data.get("projects") if isinstance(data, dict) else None
    return projects if isinstance(projects, dict) else {}


def is_project_extensions_trusted(
    project_root: Path | str,
    *,
    store_path: Path | None = None,
) -> bool:
    """Return whether project extensions are trusted for a project root.

    Args:
        project_root: Project root to inspect.
        store_path: Alternate store path for tests.

    Returns:
        `True` when the project root has a persisted trust decision.
    """
    path = store_path or _default_store_path()
    return _project_key(project_root) in _load_projects(path)


def trust_project_extensions(
    project_root: Path | str,
    *,
    store_path: Path | None = None,
) -> bool:
    """Persist a trust decision for a project root.

    Written atomically via a temporary file in the same directory so a crash
    mid-write cannot leave a partially valid store that silently grants or drops
    trust.

    Args:
        project_root: Project root to trust.
        store_path: Alternate store path for tests.

    Returns:
        `True` when the decision was persisted. `False` signals a real
            persistence failure and must not be treated as an implicit grant.
    """
    path = store_path or _default_store_path()
    projects = _load_projects(path)
    projects[_project_key(project_root)] = {"trusted_at": datetime.now(UTC).isoformat()}
    payload = json.dumps(
        {"version": _STORE_VERSION, "projects": projects},
        indent=2,
    )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            delete=False,
        ) as handle:
            handle.write(payload)
            temp_path = Path(handle.name)
        os.replace(temp_path, path)  # noqa: PTH105  # atomic rename of the temp store
    except OSError:
        logger.warning("Could not save extension trust store %s", path, exc_info=True)
        return False
    return True
