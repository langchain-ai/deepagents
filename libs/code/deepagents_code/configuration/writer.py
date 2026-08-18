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
    """Atomically mutate only the user config while preserving sibling tables.

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
            changed = mutate(data)
            if not changed:
                return WriteResult(True, False)
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
