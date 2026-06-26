"""Trust store for skill directories that resolve outside trusted roots.

`load_skill_content` refuses to read a `SKILL.md` whose resolved path falls
outside every trusted skill root — this stops a symlink inside a skill
directory from reading arbitrary files. The static escape hatch is the
`DEEPAGENTS_CODE_EXTRA_SKILLS_DIRS` env var / `[skills].extra_allowed_dirs`
config allowlist.

This module adds an in-the-moment, persistent approval path (mirroring
`mcp_trust.py`): when a skill resolves outside the trusted roots, the user is
asked once to allow the resolved target directory, and the decision is
remembered. Trust is keyed by the resolved target directory, so re-pointing a
symlink at a new directory is not trusted and re-prompts.

Trust entries are app-managed bookkeeping (a set of approved directories), not
user-facing configuration, so they live alongside the other state files under
`~/.deepagents/.state/skill_trust.json` rather than in the hand-editable
`config.toml`.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_STORAGE_VERSION = 1
"""Schema version stamped into `skill_trust.json`; bump on incompatible changes."""


def _default_store_path() -> Path:
    """Return `~/.deepagents/.state/skill_trust.json`.

    Resolved at call time (not import time) so tests can redirect storage by
    monkeypatching `deepagents_code.model_config.DEFAULT_STATE_DIR` — the same
    pattern `mcp_trust._default_store_path` uses.
    """
    from deepagents_code.model_config import DEFAULT_STATE_DIR

    return DEFAULT_STATE_DIR / "skill_trust.json"


def _normalize(target_dir: Path | str) -> str:
    """Return the resolved absolute string form of a directory key."""
    return str(Path(target_dir).expanduser().resolve())


def _load_store(store_path: Path) -> dict[str, Any]:
    """Read the JSON trust store file.

    Returns:
        Parsed JSON data, or an empty dict when the file is missing,
        unreadable, or corrupt. A corrupt store degrades to "nothing
        trusted" so a bad file can't crash startup — the next write
        rewrites it cleanly.
    """
    try:
        if not store_path.exists():
            return {}
        data = json.loads(store_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        # A corrupt store silently drops every prior approval and forces a
        # re-prompt, so log at WARNING (not DEBUG) to leave a breadcrumb for
        # the otherwise-unexplained re-prompt.
        logger.warning(
            "Skill trust store %s is corrupt; treating as empty: %s", store_path, exc
        )
        return {}
    except OSError as exc:
        logger.warning(
            "Could not read skill trust store %s; treating as empty: %s",
            store_path,
            exc,
        )
        return {}
    if not isinstance(data, dict):
        logger.warning(
            "Skill trust store %s is not a JSON object; ignoring", store_path
        )
        return {}
    return data


def _save_store(data: dict[str, Any], store_path: Path) -> bool:
    """Atomic write of JSON trust data to `store_path`.

    Uses `tempfile.mkstemp` + `Path.replace` for crash safety.

    Args:
        data: Full store dict to write.
        store_path: Destination path.

    Returns:
        `True` on success, `False` on I/O failure.
    """
    try:
        store_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=store_path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            Path(tmp_path).replace(store_path)
        except BaseException:
            with contextlib.suppress(OSError):
                Path(tmp_path).unlink()
            raise
    except (OSError, ValueError):
        logger.exception("Failed to save skill trust store to %s", store_path)
        return False
    return True


def _read_dirs(store_path: Path) -> dict[str, Any]:
    """Return the `dirs` mapping from the store, or an empty dict."""
    dirs = _load_store(store_path).get("dirs", {})
    return dirs if isinstance(dirs, dict) else {}


def is_skill_dir_trusted(
    target_dir: Path | str,
    *,
    store_path: Path | None = None,
) -> bool:
    """Check whether a resolved skill directory has been trusted.

    Args:
        target_dir: Directory to check; resolved before lookup.
        store_path: Path to the trust store file. Defaults to
            `~/.deepagents/.state/skill_trust.json`.

    Returns:
        `True` if the resolved directory is present in the store.
    """
    if store_path is None:
        store_path = _default_store_path()
    return _normalize(target_dir) in _read_dirs(store_path)


def trust_skill_dir(
    target_dir: Path | str,
    *,
    store_path: Path | None = None,
) -> bool:
    """Persist trust for a resolved skill directory.

    Args:
        target_dir: Directory to trust; resolved before storing.
        store_path: Path to the trust store file. Defaults to
            `~/.deepagents/.state/skill_trust.json`.

    Returns:
        `True` if the entry was saved successfully.
    """
    if store_path is None:
        store_path = _default_store_path()

    data = _load_store(store_path)
    dirs = data.get("dirs")
    if not isinstance(dirs, dict):
        dirs = {}
    dirs[_normalize(target_dir)] = {
        "trusted_at": datetime.now(timezone.utc).isoformat()
    }
    data["version"] = _STORAGE_VERSION
    data["dirs"] = dirs
    return _save_store(data, store_path)


def revoke_skill_dir_trust(
    target_dir: Path | str,
    *,
    store_path: Path | None = None,
) -> bool:
    """Remove trust for a resolved skill directory.

    Args:
        target_dir: Directory to revoke; resolved before lookup.
        store_path: Path to the trust store file. Defaults to
            `~/.deepagents/.state/skill_trust.json`.

    Returns:
        `True` if the entry was removed (or didn't exist).
    """
    if store_path is None:
        store_path = _default_store_path()

    data = _load_store(store_path)
    dirs = data.get("dirs")
    key = _normalize(target_dir)
    if not isinstance(dirs, dict) or key not in dirs:
        return True
    del dirs[key]
    data["version"] = _STORAGE_VERSION
    data["dirs"] = dirs
    return _save_store(data, store_path)


def clear_trusted_skill_dirs(*, store_path: Path | None = None) -> bool:
    """Remove all trusted skill directories.

    Args:
        store_path: Path to the trust store file. Defaults to
            `~/.deepagents/.state/skill_trust.json`.

    Returns:
        `True` if the store was cleared (or was already empty).
    """
    if store_path is None:
        store_path = _default_store_path()

    if not _read_dirs(store_path):
        return True
    return _save_store({"version": _STORAGE_VERSION, "dirs": {}}, store_path)


def list_trusted_skill_dirs(*, store_path: Path | None = None) -> list[str]:
    """Return the sorted list of trusted skill directory paths.

    Args:
        store_path: Path to the trust store file. Defaults to
            `~/.deepagents/.state/skill_trust.json`.

    Returns:
        Sorted absolute directory paths previously trusted.
    """
    if store_path is None:
        store_path = _default_store_path()
    return sorted(_read_dirs(store_path))


def load_trusted_skill_dirs(*, store_path: Path | None = None) -> list[Path]:
    """Return trusted skill directories as resolved `Path` objects.

    Used to extend the containment allowlist passed to `load_skill_content`.

    Args:
        store_path: Path to the trust store file. Defaults to
            `~/.deepagents/.state/skill_trust.json`.

    Returns:
        Resolved directory paths; empty when nothing is trusted.
    """
    return [Path(p) for p in list_trusted_skill_dirs(store_path=store_path)]
