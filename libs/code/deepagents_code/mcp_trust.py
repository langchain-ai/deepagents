"""Trust store for project-level MCP server configurations.

Manages persistent approval of project-level MCP configs that contain stdio
servers (which execute local commands). Trust is fingerprint-based: if the
config content changes, the user must re-approve.

Trust entries are app-managed bookkeeping (a map of project root to config
fingerprint), not user-facing configuration, so they live alongside the
other state files under `~/.deepagents/.state/mcp_trust.json` rather than in
the hand-editable `config.toml`.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_STORAGE_VERSION = 1
"""Schema version stamped into `mcp_trust.json`; bump on incompatible changes."""


def _default_store_path() -> Path:
    """Return `~/.deepagents/.state/mcp_trust.json`.

    Resolved at call time (not import time) so tests can redirect storage by
    monkeypatching `deepagents_code.model_config.DEFAULT_STATE_DIR` — the same
    pattern `auth_store._auth_path` and `mcp_auth._tokens_dir` use.
    """
    from deepagents_code.model_config import DEFAULT_STATE_DIR

    return DEFAULT_STATE_DIR / "mcp_trust.json"


def compute_config_fingerprint(config_paths: list[Path]) -> str:
    """Compute a SHA-256 fingerprint over sorted, concatenated config contents.

    Args:
        config_paths: Paths to config files to fingerprint.

    Returns:
        Fingerprint string in the form `sha256:<hex>`.
    """
    hasher = hashlib.sha256()
    for path in sorted(config_paths):
        try:
            hasher.update(path.read_bytes())
        except OSError:
            logger.warning("Could not read %s for fingerprinting", path, exc_info=True)
    return f"sha256:{hasher.hexdigest()}"


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
        # the otherwise-unexplained re-prompt — consistent with the WARNING in
        # compute_config_fingerprint and the exception log in _save_store.
        logger.warning(
            "MCP trust store %s is corrupt; treating as empty: %s", store_path, exc
        )
        return {}
    except OSError as exc:
        logger.warning(
            "Could not read MCP trust store %s; treating as empty: %s",
            store_path,
            exc,
        )
        return {}
    if not isinstance(data, dict):
        logger.warning("MCP trust store %s is not a JSON object; ignoring", store_path)
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
        logger.exception("Failed to save MCP trust store to %s", store_path)
        return False
    return True


def _read_projects(store_path: Path) -> dict[str, Any]:
    """Return the `projects` mapping from the store, or an empty dict."""
    projects = _load_store(store_path).get("projects", {})
    return projects if isinstance(projects, dict) else {}


def is_project_mcp_trusted(
    project_root: str,
    fingerprint: str,
    *,
    store_path: Path | None = None,
) -> bool:
    """Check whether a project's MCP config is trusted with the given fingerprint.

    Args:
        project_root: Absolute path to the project root.
        fingerprint: Expected fingerprint string (`sha256:<hex>`).
        store_path: Path to the trust store file. Defaults to
            `~/.deepagents/.state/mcp_trust.json`.

    Returns:
        `True` if the stored fingerprint matches.
    """
    if store_path is None:
        store_path = _default_store_path()
    return _read_projects(store_path).get(project_root) == fingerprint


def trust_project_mcp(
    project_root: str,
    fingerprint: str,
    *,
    store_path: Path | None = None,
) -> bool:
    """Persist trust for a project's MCP config.

    Args:
        project_root: Absolute path to the project root.
        fingerprint: Fingerprint to store (`sha256:<hex>`).
        store_path: Path to the trust store file. Defaults to
            `~/.deepagents/.state/mcp_trust.json`.

    Returns:
        `True` if the entry was saved successfully.
    """
    if store_path is None:
        store_path = _default_store_path()

    data = _load_store(store_path)
    projects = data.get("projects")
    if not isinstance(projects, dict):
        projects = {}
    projects[project_root] = fingerprint
    data["version"] = _STORAGE_VERSION
    data["projects"] = projects
    return _save_store(data, store_path)


def revoke_project_mcp_trust(
    project_root: str,
    *,
    store_path: Path | None = None,
) -> bool:
    """Remove trust for a project's MCP config.

    Args:
        project_root: Absolute path to the project root.
        store_path: Path to the trust store file. Defaults to
            `~/.deepagents/.state/mcp_trust.json`.

    Returns:
        `True` if the entry was removed (or didn't exist).
    """
    if store_path is None:
        store_path = _default_store_path()

    data = _load_store(store_path)
    projects = data.get("projects")
    if not isinstance(projects, dict) or project_root not in projects:
        return True
    del projects[project_root]
    data["version"] = _STORAGE_VERSION
    data["projects"] = projects
    return _save_store(data, store_path)
