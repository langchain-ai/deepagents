"""User-config-only atomic TOML writes."""

from __future__ import annotations

import contextlib
import logging
import os
import tempfile
import threading
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

USER_CONFIG_WRITE_LOCK = threading.RLock()


@dataclass(frozen=True, slots=True)
class WriteResult:
    """Outcome of one user config transaction."""

    ok: bool
    changed: bool
    error: str | None = None

    def __post_init__(self) -> None:
        """Reject outcomes that cannot describe a real transaction.

        Callers branch on `ok` alone, so a failure with no detail would surface
        as a bare "could not be saved" with nothing to act on, and a change
        recorded against a failed write would report an edit that never
        reached the file.

        Raises:
            ValueError: If the three fields do not describe one outcome.
        """
        if not self.ok and self.error is None:
            msg = "a failed write must carry an error detail"
            raise ValueError(msg)
        if self.changed and not self.ok:
            msg = "a failed write cannot have changed the file"
            raise ValueError(msg)
        if self.ok and self.error is not None:
            msg = "a successful write cannot carry an error detail"
            raise ValueError(msg)


def update_user_config(
    mutate: Callable[[dict[str, Any]], bool],
    *,
    config_path: Path | None = None,
) -> WriteResult:
    """Serialize a read-modify-write of the user config and replace it atomically.

    Writes the user tier only. The managed path is refused rather than trusted
    to be unreachable.

    A committed write to the default path also refreshes the shared process
    resolver, so later reads see the new value. That refresh is best-effort and
    never turns a landed write into a reported failure; see
    `refresh_shared_resolver`.

    Args:
        mutate: Edit applied to the table parsed inside the write lock. It must
            edit that table in place and return whether anything changed;
            overwriting it wholesale discards concurrent edits to sibling
            tables, which is the hazard the shared lock exists to prevent.
        config_path: Override the default config location; intended for tests.
            The managed-config path is rejected.

    Returns:
        Transaction success, changed state, and safe error detail.
    """
    if config_path is None:
        from deepagents_code.model_config import DEFAULT_CONFIG_PATH

        config_path = DEFAULT_CONFIG_PATH
    from deepagents_code.configuration.paths import managed_config_path

    if config_path == managed_config_path():
        # The managed tier is read-only, and `THREAT_MODEL.md` states that as a
        # security property. It held only because no caller passed this path;
        # make it structural, so the claim does not depend on every future
        # caller knowing it.
        return WriteResult(False, False, "managed config is read-only")
    with USER_CONFIG_WRITE_LOCK:
        try:
            if config_path.exists():
                with config_path.open("rb") as handle:
                    data = tomllib.load(handle)
            else:
                data = {}
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
            # `tomllib` decodes the bytes itself, so a file that is not UTF-8
            # raises `UnicodeDecodeError` rather than `TOMLDecodeError`.
            # Letting it escape turns a mis-encoded config into a traceback the
            # caller reports as a generic failure, with the real reason lost.
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
            # Imported before `mkstemp` so a missing writer dependency cannot
            # leak the descriptor: the cleanup below can unlink the temp path,
            # but only `os.fdopen` takes ownership of the descriptor, so an
            # `ImportError` between the two left it open.
            import tomli_w

            descriptor, temporary = tempfile.mkstemp(
                dir=config_path.parent,
                suffix=".tmp",
            )
            temporary_path = Path(temporary)
            try:
                handle = os.fdopen(descriptor, "wb")
            except BaseException:
                # Nothing adopted the descriptor, so close it here. Closing it
                # in the handler below instead would risk closing an unrelated
                # file: once `os.fdopen` succeeds and its `with` block exits,
                # the number is free for another thread to reuse.
                with contextlib.suppress(OSError):
                    os.close(descriptor)
                with contextlib.suppress(OSError):
                    temporary_path.unlink()
                raise
            try:
                with handle:
                    tomli_w.dump(data, handle)
                temporary_path.replace(config_path)
            except BaseException:
                with contextlib.suppress(OSError):
                    temporary_path.unlink()
                raise
        except (
            OSError,
            ImportError,
            tomllib.TOMLDecodeError,
            TypeError,
            ValueError,
        ) as exc:
            # `ImportError` covers the `tomli_w` import inside this block: an
            # install without the writer dependency must report "could not
            # update <path>" like any other write failure.
            return WriteResult(False, False, f"could not update {config_path}: {exc}")
    refresh_shared_resolver(config_path)
    return WriteResult(True, True)


def refresh_shared_resolver(config_path: Path) -> None:
    """Make a committed write visible to the shared process resolver.

    Only the default path is refreshed. `get_config_resolver` is keyed on
    `DEFAULT_CONFIG_PATH`, so reloading it after a write to an override path
    would re-read the real user config and the managed policy file - live
    filesystem reads inside tests that passed a `tmp_path` - while leaving the
    written path's own view stale anyway.

    Refreshes every tier, managed policy included. The managed snapshot is
    fetched before the resolver takes its generation lock, then installed as
    an already-refreshed replacement. An in-app preference toggle therefore
    still picks up policy installed since startup without making ordinary
    event-loop config reads wait on remote I/O. Leaving the managed provider
    alone would let the user tier advance past the policy tier, which is the
    split-generation state the whole design exists to prevent.

    Failures are logged rather than returned. The write already landed and was
    replaced into place; reporting a stale in-process view as a failed write
    sends the user to retry or hand-edit a file that is already correct.

    Every failure, not just `OSError`: a reload can also raise `ValueError`
    from the snapshot and resolved-value invariants, or `RuntimeError` from a
    provider that produced no snapshot. Callers such as
    `model_config._save_toml_field` invoke this from the success branch of
    their own write and return `bool`, so anything escaping here surfaces as a
    crash after the bytes are already on disk -- the exact outcome the
    paragraph above says must not happen.

    Args:
        config_path: Path the caller just wrote.
    """
    from deepagents_code.model_config import DEFAULT_CONFIG_PATH

    if config_path != DEFAULT_CONFIG_PATH:
        return
    from deepagents_code.configuration.resolver import get_config_resolver

    try:
        get_config_resolver(refresh_managed=True)
    except Exception as exc:  # noqa: BLE001  # Write committed; see docstring
        logger.warning(
            "Wrote %s but could not refresh the shared config resolver: %s. "
            "This process keeps serving the previous values until it restarts.",
            config_path,
            exc,
        )
