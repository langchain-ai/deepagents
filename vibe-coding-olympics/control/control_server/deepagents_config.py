"""Helpers for event-specific Deep Agents CLI config cleanup."""

from __future__ import annotations

import contextlib
import logging
import os
import re
import tempfile
import tomllib
from pathlib import Path

logger = logging.getLogger(__name__)

CONFIG_PATH_ENV = "VIBE_DEEPAGENTS_CONFIG_PATH"
DEFAULT_CONFIG_PATH = Path.home() / ".deepagents" / "config.toml"
_RECENT_KEY_RE = re.compile(r"^\s*(?:recent|['\"]recent['\"])\s*=")
_TABLE_RE = re.compile(r"^\s*\[([^\]]+)\]")


def config_path() -> Path:
    """Return the Deep Agents CLI config path for this player process."""
    raw = os.environ.get(CONFIG_PATH_ENV, "").strip()
    if raw:
        return Path(raw).expanduser()
    return DEFAULT_CONFIG_PATH


def clear_recent_model(path: Path | None = None) -> bool:
    """Remove `[models].recent` from the Deep Agents CLI config.

    Args:
        path: Config path to mutate. Defaults to `~/.deepagents/config.toml`.

    Returns:
        `True` if the key is absent after the call, `False` if the config could
        not be read or rewritten.
    """
    target = path or config_path()
    if not target.exists():
        return True

    try:
        with target.open("rb") as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        logger.warning("Could not read Deep Agents config %s", target, exc_info=True)
        return False

    models = data.get("models")
    if not isinstance(models, dict) or "recent" not in models:
        return True

    try:
        original = target.read_text(encoding="utf-8")
        updated = _remove_recent_model_line(original)
        tomllib.loads(updated)
        _atomic_write_text(target, updated)
    except (OSError, tomllib.TOMLDecodeError):
        logger.warning(
            "Could not clear models.recent from Deep Agents config %s",
            target,
            exc_info=True,
        )
        return False
    return True


def _remove_recent_model_line(content: str) -> str:
    """Return TOML content with the direct `[models].recent` key removed."""
    lines = content.splitlines(keepends=True)
    current_table: str | None = None
    output: list[str] = []
    for line in lines:
        if match := _TABLE_RE.match(line):
            current_table = match.group(1).strip()
        if current_table == "models" and _RECENT_KEY_RE.match(line):
            continue
        output.append(line)
    return "".join(output)


def _atomic_write_text(path: Path, content: str) -> None:
    """Atomically replace `path` with `content`."""
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    tmp = Path(tmp_path)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        tmp.replace(path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise
