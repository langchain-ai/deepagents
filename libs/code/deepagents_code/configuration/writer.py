"""User-config-only atomic TOML writes."""

from __future__ import annotations

import contextlib
import os
import tempfile
import threading
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

USER_CONFIG_WRITE_LOCK = threading.RLock()


@dataclass(frozen=True, slots=True)
class WriteResult:
    """Outcome of one user config transaction."""

    ok: bool
    changed: bool
    error: str | None = None


def update_user_config(
    mutate: Callable[[dict[str, Any]], bool],
    *,
    config_path: Path | None = None,
) -> WriteResult:
    """Serialize a read-modify-write of the user config and replace it atomically.

    Args:
        mutate: Edit applied to the table parsed inside the write lock. It must
            edit that table in place and return whether anything changed;
            overwriting it wholesale discards concurrent edits to sibling
            tables, which is the hazard the shared lock exists to prevent.
        config_path: Override the default config location; intended for tests.

    Returns:
        Transaction success, changed state, and safe error detail.
    """
    if config_path is None:
        from deepagents_code.model_config import DEFAULT_CONFIG_PATH

        config_path = DEFAULT_CONFIG_PATH
    with USER_CONFIG_WRITE_LOCK:
        try:
            if (
                config_path.parent.exists()
                and config_path.parent.stat().st_mode & 0o222 == 0
            ):
                return WriteResult(
                    False,
                    False,
                    f"could not update {config_path}: parent directory is not writable",
                )
            if config_path.exists():
                with config_path.open("rb") as handle:
                    data = tomllib.load(handle)
            else:
                data = {}
        except (OSError, tomllib.TOMLDecodeError) as exc:
            return WriteResult(False, False, f"could not update {config_path}: {exc}")
        # Outside the exception handling on purpose: a `TypeError` or
        # `ValueError` raised inside a caller's closure is a bug in that
        # caller, and reporting it as "could not update <path>" sends the user
        # to check permissions and disk space for a defect in a lambda.
        changed = mutate(data)
        if not changed:
            return WriteResult(True, False)
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(
                dir=config_path.parent,
                suffix=".tmp",
            )
            temporary_path = Path(temporary)
            try:
                import tomli_w

                with os.fdopen(descriptor, "wb") as handle:
                    tomli_w.dump(data, handle)
                temporary_path.replace(config_path)
            except BaseException:
                with contextlib.suppress(OSError):
                    temporary_path.unlink()
                raise
        except (OSError, tomllib.TOMLDecodeError, TypeError, ValueError) as exc:
            return WriteResult(False, False, f"could not update {config_path}: {exc}")
    return WriteResult(True, True)
