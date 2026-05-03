"""One-time migration of legacy CLI state files into `~/.deepagents/.state/`.

Earlier versions wrote internal state directly under `~/.deepagents/`,
mixing it with user-facing agent directories (so e.g. `mcp-tokens/`
showed up in `deepagents agents list`). State now lives in a dedicated
`.state/` subdirectory; this module moves any legacy files into place
on startup.

The migration is best-effort and idempotent: it skips entries whose
destination already exists, logs and continues on per-entry failures,
and never blocks startup on I/O errors.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from deepagents_cli.model_config import DEFAULT_CONFIG_DIR, DEFAULT_STATE_DIR
from deepagents_cli.onboarding import ONBOARDING_MARKER_FILENAME

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

logger = logging.getLogger(__name__)


_LEGACY_NAMES: tuple[str, ...] = (
    "mcp-tokens",
    "sessions.db",
    "sessions.db-wal",
    "sessions.db-shm",
    "latest_version.json",
    "update_state.json",
    "history.jsonl",
    ONBOARDING_MARKER_FILENAME,
)
"""Names directly under `~/.deepagents/` that now live in `.state/`.

`sessions.db-wal` and `sessions.db-shm` are SQLite sidecar files that may
or may not be present depending on whether the database was opened in WAL
mode and whether a checkpoint had run before shutdown.
"""


def _iter_migrations(
    config_dir: Path,
    state_dir: Path,
    names: Iterable[str],
) -> Iterable[tuple[Path, Path]]:
    for name in names:
        yield config_dir / name, state_dir / name


def migrate_legacy_state(
    *,
    config_dir: Path = DEFAULT_CONFIG_DIR,
    state_dir: Path = DEFAULT_STATE_DIR,
) -> None:
    """Move legacy state entries from `config_dir` into `state_dir`.

    Idempotent: each entry is skipped when the destination already exists
    (the migration ran on a prior invocation) or when the source does not
    exist (nothing to move). Errors on individual entries are logged and
    swallowed so a single unmovable file does not block the rest.

    Args:
        config_dir: Directory holding legacy state. Defaults to
            `~/.deepagents/`.
        state_dir: Destination directory for state files. Defaults to
            `~/.deepagents/.state/`.
    """
    try:
        if not config_dir.is_dir():
            return
    except OSError:
        logger.debug(
            "Could not stat %s; skipping state migration",
            config_dir,
            exc_info=True,
        )
        return

    pending: list[tuple[Path, Path]] = []
    for src, dst in _iter_migrations(config_dir, state_dir, _LEGACY_NAMES):
        try:
            src_exists = src.exists()
        except OSError:
            continue
        if not src_exists:
            continue
        try:
            dst_exists = dst.exists()
        except OSError:
            continue
        if dst_exists:
            # Both exist — typically a CLI downgrade after the migration
            # already ran once (the older version recreates a fresh file
            # at the legacy path) or a manually pre-populated `.state/`.
            # Clobbering either copy could lose data, so skip and warn so
            # the user can resolve it.
            logger.warning(
                "Cannot migrate %s -> %s: destination already exists. "
                "Inspect both files and either delete the obsolete one "
                "or move the legacy file in manually.",
                src,
                dst,
            )
            continue
        pending.append((src, dst))
    if not pending:
        return

    try:
        state_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.warning(
            "Could not create state directory %s; skipping state migration",
            state_dir,
            exc_info=True,
        )
        return

    for src, dst in pending:
        try:
            src.rename(dst)
        except OSError:
            logger.warning(
                "Failed to migrate %s -> %s; leaving legacy file in place",
                src,
                dst,
                exc_info=True,
            )
            continue
        logger.info("Migrated %s -> %s", src, dst)
