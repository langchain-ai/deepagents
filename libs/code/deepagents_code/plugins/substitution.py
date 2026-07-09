"""Variable substitution for plugin-provided configuration."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)
_WARNED_USER_CONFIG: set[str] = set()


@dataclass(frozen=True, slots=True, kw_only=True)
class PluginSubstitutionContext:
    """Values available while expanding plugin configuration."""

    plugin_root: Path
    plugin_data: Path
    project_dir: Path | None = None
    session_id: str | None = None
    skill_dir: Path | None = None


def plugin_environment(
    *,
    plugin_root: Path,
    plugin_data: Path,
    project_dir: Path | None = None,
    session_id: str | None = None,
    skill_dir: Path | None = None,
) -> dict[str, str]:
    """Build environment variables exposed to plugin subprocesses.

    Args:
        plugin_root: Plugin root directory.
        plugin_data: Plugin data directory.
        project_dir: Optional project directory.
        session_id: Optional session identifier.
        skill_dir: Optional invoked skill directory.

    Returns:
        Environment variables for plugin subprocesses.
    """
    plugin_data.mkdir(parents=True, exist_ok=True)
    root = str(plugin_root.resolve())
    data = str(plugin_data.resolve())
    env = {
        "CLAUDE_PLUGIN_ROOT": root,
        "CLAUDE_PLUGIN_DATA": data,
        "PLUGIN_ROOT": root,
        "PLUGIN_DATA": data,
        "DEEPAGENTS_PLUGIN_ROOT": root,
        "DEEPAGENTS_PLUGIN_DATA": data,
    }
    if project_dir is not None:
        env["CLAUDE_PROJECT_DIR"] = str(project_dir.resolve())
    if session_id is not None:
        env["CLAUDE_SESSION_ID"] = session_id
    if skill_dir is not None:
        env["CLAUDE_SKILL_DIR"] = str(skill_dir.resolve())
    return env


def substitute_string(
    value: str,
    *,
    plugin_root: Path,
    plugin_data: Path,
    project_dir: Path | None = None,
    session_id: str | None = None,
    skill_dir: Path | None = None,
    warning_key: str | None = None,
) -> str:
    """Substitute plugin path variables in a string.

    Args:
        value: String to transform.
        plugin_root: Plugin root directory.
        plugin_data: Plugin data directory.
        project_dir: Optional project directory.
        session_id: Optional session identifier.
        skill_dir: Optional invoked skill directory.
        warning_key: Identifier used to emit unsupported-option warnings once.

    Returns:
        String with supported plugin variables substituted.
    """
    env = plugin_environment(
        plugin_root=plugin_root,
        plugin_data=plugin_data,
        project_dir=project_dir,
        session_id=session_id,
        skill_dir=skill_dir,
    )
    result = value
    for key, replacement in env.items():
        result = result.replace(f"${{{key}}}", replacement)
    key = warning_key or str(plugin_root)
    if "${user_config." in result and key not in _WARNED_USER_CONFIG:
        _WARNED_USER_CONFIG.add(key)
        logger.warning(
            "Plugin userConfig substitution is not supported yet; "
            "leaving ${user_config.*} literal"
        )
    return result


def substitute_json(
    value: object,
    *,
    plugin_root: Path,
    plugin_data: Path,
    project_dir: Path | None = None,
    session_id: str | None = None,
    skill_dir: Path | None = None,
    warning_key: str | None = None,
) -> object:
    """Substitute plugin variables throughout a JSON-compatible value.

    Args:
        value: JSON-compatible value to transform.
        plugin_root: Plugin root directory.
        plugin_data: Plugin data directory.
        project_dir: Optional project directory.
        session_id: Optional session identifier.
        skill_dir: Optional invoked skill directory.
        warning_key: Identifier used to emit unsupported-option warnings once.

    Returns:
        Value with strings recursively substituted.
    """
    if isinstance(value, str):
        return substitute_string(
            value,
            plugin_root=plugin_root,
            plugin_data=plugin_data,
            project_dir=project_dir,
            session_id=session_id,
            skill_dir=skill_dir,
            warning_key=warning_key,
        )
    if isinstance(value, list):
        return [
            substitute_json(
                item,
                plugin_root=plugin_root,
                plugin_data=plugin_data,
                project_dir=project_dir,
                session_id=session_id,
                skill_dir=skill_dir,
                warning_key=warning_key,
            )
            for item in value
        ]
    if isinstance(value, dict):
        return {
            key: substitute_json(
                item,
                plugin_root=plugin_root,
                plugin_data=plugin_data,
                project_dir=project_dir,
                session_id=session_id,
                skill_dir=skill_dir,
                warning_key=warning_key,
            )
            for key, item in value.items()
        }
    return value
