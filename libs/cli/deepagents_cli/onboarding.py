"""First-run onboarding state for the interactive CLI."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from deepagents_cli._env_vars import DEBUG_ONBOARDING, is_env_truthy
from deepagents_cli.model_config import DEFAULT_STATE_DIR

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

ONBOARDING_MARKER_FILENAME = "onboarding_complete"
"""Marker filename under `~/.deepagents/.state` after onboarding has completed."""

ONBOARDING_NAME_MEMORY_START = "<!-- deepagents:onboarding-name:start -->"
"""Start marker for the managed onboarding name memory block."""

ONBOARDING_NAME_MEMORY_END = "<!-- deepagents:onboarding-name:end -->"
"""End marker for the managed onboarding name memory block."""


def onboarding_marker_path(state_dir: Path | None = None) -> Path:
    """Return the first-run onboarding marker path.

    Args:
        state_dir: Optional state directory override for tests.

    Returns:
        Path to the onboarding completion marker.
    """
    return (state_dir or DEFAULT_STATE_DIR) / ONBOARDING_MARKER_FILENAME


def has_completed_onboarding(state_dir: Path | None = None) -> bool:
    """Return whether the user has completed onboarding.

    Args:
        state_dir: Optional state directory override for tests.

    Returns:
        `True` when the onboarding marker exists, otherwise `False`.
    """
    try:
        return onboarding_marker_path(state_dir).exists()
    except OSError:
        logger.warning("Could not inspect onboarding marker", exc_info=True)
        return False


def mark_onboarding_complete(state_dir: Path | None = None) -> bool:
    """Persist that onboarding has completed.

    Args:
        state_dir: Optional state directory override for tests.

    Returns:
        `True` when the marker was written, otherwise `False`.
    """
    path = onboarding_marker_path(state_dir)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("1\n", encoding="utf-8")
    except OSError:
        logger.warning("Could not write onboarding marker at %s", path, exc_info=True)
        return False
    return True


def write_onboarding_name_memory(
    name: str,
    assistant_id: str,
    *,
    memory_path: Path | None = None,
) -> bool:
    """Persist the optional onboarding name into user agent memory.

    Empty or whitespace-only names are skipped (no file is written).

    Args:
        name: Submitted user name.
        assistant_id: Agent identifier whose user memory should be updated.
        memory_path: Optional memory file override for tests.

    Returns:
        `True` when memory was written, otherwise `False`.
    """
    clean = _normalize_memory_name(name)
    if not clean:
        return False

    if memory_path is None:
        from deepagents_cli.config import settings

        path = settings.get_user_agent_md_path(assistant_id)
    else:
        path = memory_path

    block = _onboarding_name_memory_block(clean)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            existing = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            existing = ""
        except UnicodeDecodeError:
            # Existing memory file is not valid UTF-8. Overwriting would clobber
            # whatever the user has there, so abort and let them resolve it.
            logger.warning(
                "Existing memory file %s is not valid UTF-8; skipping onboarding "
                "name memory write to avoid clobbering user content",
                path,
                exc_info=True,
            )
            return False
        path.write_text(
            _upsert_onboarding_name_memory(existing, block),
            encoding="utf-8",
        )
    except OSError:
        logger.warning(
            "Could not write onboarding name memory at %s",
            path,
            exc_info=True,
        )
        return False
    return True


def _normalize_memory_name(name: str) -> str:
    """Normalize whitespace in a name before writing it to memory.

    Returns:
        Name with leading/trailing whitespace stripped and internal runs
            collapsed to single spaces.
    """
    return " ".join(name.split())


def _onboarding_name_memory_block(name: str) -> str:
    """Return the managed memory block for an onboarding name."""
    quoted = json.dumps(name)
    return (
        f"{ONBOARDING_NAME_MEMORY_START}\n"
        f"- The user's preferred name is {quoted}.\n"
        f"{ONBOARDING_NAME_MEMORY_END}"
    )


def _upsert_onboarding_name_memory(existing: str, block: str) -> str:
    """Insert or replace the managed onboarding name memory block.

    Returns:
        Updated memory file content.
    """
    start = existing.find(ONBOARDING_NAME_MEMORY_START)
    end = existing.find(ONBOARDING_NAME_MEMORY_END)
    if start != -1 and end != -1 and start < end:
        end += len(ONBOARDING_NAME_MEMORY_END)
        prefix = existing[:start].rstrip()
        suffix = existing[end:].strip()
        parts = [part for part in (prefix, block, suffix) if part]
        return "\n\n".join(parts).rstrip() + "\n"

    base = existing.rstrip()
    if not base:
        return f"## User Preferences\n\n{block}\n"
    if "## User Preferences" in base:
        return f"{base}\n\n{block}\n"
    return f"{base}\n\n## User Preferences\n\n{block}\n"


def should_run_onboarding(state_dir: Path | None = None) -> bool:
    """Return whether onboarding should open at interactive startup.

    Args:
        state_dir: Optional state directory override for tests.

    Returns:
        `True` when the debug override is enabled or no completion marker exists.
    """
    if is_env_truthy(DEBUG_ONBOARDING):
        return True
    return not has_completed_onboarding(state_dir)
