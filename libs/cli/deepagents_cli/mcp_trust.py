"""Trust store for project-level MCP server configurations.

Manages persistent approval of project-level MCP configs that contain stdio
servers (which execute local commands). Trust is fingerprint-based: if the
config content changes, the user must re-approve.

Trust entries are stored in `~/.deepagents/config.toml` under
`[mcp_trust.projects]`.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_DIR = Path.home() / ".deepagents"
_DEFAULT_CONFIG_PATH = _DEFAULT_CONFIG_DIR / "config.toml"


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


def _load_config(config_path: Path) -> dict[str, Any]:
    """Read the TOML config file.

    Returns:
        Parsed TOML data, or an empty dict on failure.
    """
    import tomllib

    try:
        if not config_path.exists():
            return {}
        with config_path.open("rb") as f:
            return tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        logger.debug("Could not read config %s", config_path, exc_info=True)
        return {}


def _save_config(data: dict[str, Any], config_path: Path) -> bool:
    """Atomic write of TOML data to config_path.

    Uses `tempfile.mkstemp` + `Path.replace` for crash safety.

    Args:
        data: Full TOML data dict to write.
        config_path: Destination path.

    Returns:
        `True` on success, `False` on I/O failure.
    """
    import tomli_w

    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=config_path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                tomli_w.dump(data, f)
            Path(tmp_path).replace(config_path)
        except BaseException:
            with contextlib.suppress(OSError):
                Path(tmp_path).unlink()
            raise
    except (OSError, ValueError):
        logger.exception("Failed to save config to %s", config_path)
        return False
    return True


def is_project_mcp_trusted(
    project_root: str,
    fingerprint: str,
    *,
    config_path: Path | None = None,
) -> bool:
    """Check whether a project's MCP config is trusted with the given fingerprint.

    Args:
        project_root: Absolute path to the project root.
        fingerprint: Expected fingerprint string (`sha256:<hex>`).
        config_path: Path to the trust config file.

    Returns:
        `True` if the stored fingerprint matches.
    """
    if config_path is None:
        config_path = _DEFAULT_CONFIG_PATH

    data = _load_config(config_path)
    projects = data.get("mcp_trust", {}).get("projects", {})
    return projects.get(project_root) == fingerprint


def trust_project_mcp(
    project_root: str,
    fingerprint: str,
    *,
    config_path: Path | None = None,
) -> bool:
    """Persist trust for a project's MCP config.

    Args:
        project_root: Absolute path to the project root.
        fingerprint: Fingerprint to store (`sha256:<hex>`).
        config_path: Path to the trust config file.

    Returns:
        `True` if the entry was saved successfully.
    """
    if config_path is None:
        config_path = _DEFAULT_CONFIG_PATH

    data = _load_config(config_path)
    if "mcp_trust" not in data:
        data["mcp_trust"] = {}
    if "projects" not in data["mcp_trust"]:
        data["mcp_trust"]["projects"] = {}
    data["mcp_trust"]["projects"][project_root] = fingerprint
    return _save_config(data, config_path)


def revoke_project_mcp_trust(
    project_root: str,
    *,
    config_path: Path | None = None,
) -> bool:
    """Remove trust for a project's MCP config.

    Args:
        project_root: Absolute path to the project root.
        config_path: Path to the trust config file.

    Returns:
        `True` if the entry was removed (or didn't exist).
    """
    if config_path is None:
        config_path = _DEFAULT_CONFIG_PATH

    data = _load_config(config_path)
    projects = data.get("mcp_trust", {}).get("projects", {})
    if project_root not in projects:
        return True
    del data["mcp_trust"]["projects"][project_root]
    return _save_config(data, config_path)
