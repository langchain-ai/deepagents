"""Sensitive local-state retention and cleanup helpers.

Talon is an experimental runtime and is subject to change or removal at any time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from deepagents_talon.config import TalonConfig
    from deepagents_talon.cron import CronJob, CronJobStore

DEFAULT_CRON_JOB_RETENTION_DAYS = 30
DEFAULT_INBOUND_MEDIA_RETENTION_HOURS = 24

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DataLifecycleReport:
    """Summary of sensitive local-state cleanup.

    Args:
        removed_cron_jobs: Completed cron records deleted from disk.
        removed_media_files: Downloaded inbound media files deleted from disk.
    """

    removed_cron_jobs: tuple[CronJob, ...]
    removed_media_files: tuple[Path, ...]


def cleanup_sensitive_state(
    *,
    config: TalonConfig,
    cron_store: CronJobStore,
    now: datetime | None = None,
) -> DataLifecycleReport:
    """Apply retention policy for sensitive persisted Talon state.

    Args:
        config: Talon process configuration.
        cron_store: Store for assistant-scoped cron records.
        now: Current timestamp override for deterministic tests.

    Returns:
        Cleanup summary.
    """
    current = datetime.now(UTC) if now is None else _coerce_utc(now)
    cron_days = _env_non_negative_int(
        config,
        "DEEPAGENTS_TALON_CRON_RETENTION_DAYS",
        DEFAULT_CRON_JOB_RETENTION_DAYS,
    )
    media_hours = _env_non_negative_int(
        config,
        "DEEPAGENTS_TALON_INBOUND_MEDIA_RETENTION_HOURS",
        DEFAULT_INBOUND_MEDIA_RETENTION_HOURS,
    )

    removed_cron_jobs = tuple(
        cron_store.prune_completed(retain_for=timedelta(days=cron_days), now=current),
    )
    removed_media_files = tuple(
        _delete_old_files(
            config.inbound_media_dir,
            cutoff=current - timedelta(hours=media_hours),
        ),
    )
    _remove_empty_dirs(config.inbound_media_dir)
    if removed_cron_jobs or removed_media_files:
        logger.info(
            "Talon data lifecycle cleanup removed %d cron job(s) and %d media file(s)",
            len(removed_cron_jobs),
            len(removed_media_files),
        )
    return DataLifecycleReport(
        removed_cron_jobs=removed_cron_jobs,
        removed_media_files=removed_media_files,
    )


def _delete_old_files(root: Path, *, cutoff: datetime) -> list[Path]:
    if not root.exists():
        return []

    removed: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() and not path.is_symlink():
            continue
        try:
            modified = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        except OSError:
            continue
        if modified > cutoff:
            continue
        try:
            path.unlink()
        except OSError:
            logger.warning("Could not delete expired inbound media file: %s", path, exc_info=True)
        else:
            removed.append(path)
    return removed


def _remove_empty_dirs(root: Path) -> None:
    if not root.exists():
        return

    for path in sorted((item for item in root.rglob("*") if item.is_dir()), reverse=True):
        try:
            path.rmdir()
        except OSError:
            continue


def _env_non_negative_int(config: TalonConfig, key: str, default: int) -> int:
    value = config.env.get(key)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as error:
        msg = f"{key} must be a non-negative integer"
        raise ValueError(msg) from error
    if parsed < 0:
        msg = f"{key} must be a non-negative integer"
        raise ValueError(msg)
    return parsed


def _coerce_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
