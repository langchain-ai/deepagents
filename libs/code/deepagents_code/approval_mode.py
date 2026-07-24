"""Approval-mode state shared by the Textual client and agent server."""

from __future__ import annotations

import contextlib
import inspect
import json
import logging
import os
import tempfile
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import TypedDict

from filelock import FileLock, Timeout

logger = logging.getLogger(__name__)

APPROVAL_MODE_NAMESPACE: tuple[str, str] = ("deepagents_code", "approval_mode")
"""Store namespace for per-thread approval-mode control records."""

YOLO_ACKNOWLEDGEMENT_POLICY_VERSION = "2026-07-14"
"""Version of the unrestricted-mode warning that must be acknowledged."""

AUTO_NOTICE_VERSION = "2026-07-24"
"""Version of the first-run Auto mode education notice.

Bump this string whenever the notice copy changes materially enough that
already-shown installs should see it again.
"""


class ApprovalMode(StrEnum):
    """Tool-approval policy selected for an interactive thread."""

    MANUAL = "manual"
    AUTO = "auto"
    YOLO = "yolo"


class ApprovalModePayload(TypedDict):
    """Stored approval-mode control payload."""

    mode: str


def coerce_approval_mode(value: object) -> ApprovalMode:
    """Return a validated mode, failing closed to `manual`.

    Args:
        value: Untrusted mode value from config, context, or storage.

    Returns:
        A validated `ApprovalMode`; invalid values become `ApprovalMode.MANUAL`.
    """
    try:
        return ApprovalMode(value) if isinstance(value, str) else ApprovalMode.MANUAL
    except ValueError:
        return ApprovalMode.MANUAL


def next_approval_mode(
    current: ApprovalMode | str | object,
    *,
    auto_eligible: bool,
    yolo_switcher_enabled: bool,
) -> ApprovalMode | None:
    """Return the next Shift+Tab approval mode for the active session.

    The cycle is Manual → Auto → YOLO → Manual when both Auto and the YOLO
    switcher entry are available. Auto is omitted when `auto_eligible` is false
    (for example a remote sandbox). YOLO is omitted when orgs/users disable
    `startup.yolo_switcher`. Launching with `--yolo` still leaves unrestricted
    mode when the switcher entry is disabled; Shift+Tab only exits YOLO then.

    Args:
        current: Active approval mode (mode enum or raw value).
        auto_eligible: Whether classifier-backed Auto can be selected.
        yolo_switcher_enabled: Whether unrestricted YOLO appears in the cycle.

    Returns:
        The next mode, or `None` when no alternate mode is available.
    """
    mode = (
        current if isinstance(current, ApprovalMode) else coerce_approval_mode(current)
    )
    if mode is ApprovalMode.MANUAL:
        if auto_eligible:
            return ApprovalMode.AUTO
        if yolo_switcher_enabled:
            return ApprovalMode.YOLO
        return None
    if mode is ApprovalMode.AUTO:
        if yolo_switcher_enabled:
            return ApprovalMode.YOLO
        return ApprovalMode.MANUAL
    # Only genuine YOLO reaches here; unknown/invalid values were normalized to
    # Manual above and took that branch. Exiting YOLO always returns to Manual.
    return ApprovalMode.MANUAL


def approval_mode_key(thread_id: str) -> str:
    """Return the store key for a thread's live approval mode.

    Args:
        thread_id: LangGraph thread id for the active session.

    Returns:
        Deterministic store key that does not expose the raw thread id.
    """
    return sha256(thread_id.encode("utf-8")).hexdigest()


def approval_mode_payload(
    *,
    mode: ApprovalMode | str | None = None,
    auto_approve: bool | None = None,
) -> ApprovalModePayload:
    """Return the stored approval-mode payload.

    Args:
        mode: Explicit approval mode.
        auto_approve: Compatibility input for callers using the previous Boolean
            API. `True` maps to unrestricted `yolo`, and `False` maps to `manual`.

    Returns:
        JSON-serializable store value.

    Raises:
        ValueError: If neither or both inputs are supplied, or `mode` is invalid.
    """
    if (mode is None) == (auto_approve is None):
        msg = "Provide exactly one of mode or auto_approve"
        raise ValueError(msg)
    if auto_approve is not None:
        resolved = ApprovalMode.YOLO if auto_approve else ApprovalMode.MANUAL
    else:
        try:
            resolved = ApprovalMode(mode)
        except (TypeError, ValueError) as exc:
            msg = f"Invalid approval mode: {mode!r}"
            raise ValueError(msg) from exc
    return {"mode": resolved.value}


def _item_value(item: object) -> object:
    """Extract a store item's value.

    Args:
        item: SDK or runtime store-item shape.

    Returns:
        The stored value, or `None` when the shape is unrecognized.
    """
    if isinstance(item, Mapping):
        return item.get("value")
    return getattr(item, "value", None)


def _approval_mode_from_item(item: object) -> ApprovalMode | None:
    """Extract a validated approval mode from a Store item.

    Args:
        item: SDK or runtime store-item shape.

    Returns:
        The stored mode, or `None` when the item is missing or malformed.
    """
    if item is None:
        logger.debug("Approval-mode store item is missing")
        return None

    value = _item_value(item)
    raw_mode = value.get("mode") if isinstance(value, Mapping) else None
    if isinstance(raw_mode, str):
        try:
            return ApprovalMode(raw_mode)
        except ValueError:
            pass

    logger.warning("Approval-mode store item has invalid contents")
    return None


def read_approval_mode_from_store(
    store: object, key: str | None
) -> ApprovalMode | None:
    """Read a live approval mode from the server-side LangGraph Store.

    Args:
        store: `request.runtime.store` from the graph server.
        key: Store key produced by `approval_mode_key`.

    Returns:
        A validated mode, or `None` when the record cannot be trusted. Callers
        must interpret `None` as `manual`.
    """
    if store is None:
        logger.debug("Approval-mode store is unavailable")
        return None
    if not isinstance(key, str) or not key:
        logger.debug("Approval-mode store key is missing or invalid")
        return None

    get = getattr(store, "get", None)
    if get is None:
        logger.debug("Approval-mode store does not expose get()")
        return None

    try:
        item = get(APPROVAL_MODE_NAMESPACE, key)
    except Exception:
        logger.warning("Could not read approval-mode store item", exc_info=True)
        return None
    return _approval_mode_from_item(item)


async def aread_approval_mode_from_store(
    store: object, key: str | None
) -> ApprovalMode | None:
    """Asynchronously read a live approval mode from a LangGraph Store.

    The graph server supplies an async batched Store whose synchronous methods
    reject calls from the event-loop thread. Prefer `aget()` for that runtime,
    while retaining a synchronous fallback for lightweight local test stores.

    Args:
        store: `request.runtime.store` from the graph server.
        key: Store key produced by `approval_mode_key`.

    Returns:
        A validated mode, or `None` when the record cannot be trusted. Callers
        must interpret `None` as `manual`.
    """
    if store is None:
        logger.debug("Approval-mode store is unavailable")
        return None
    if not isinstance(key, str) or not key:
        logger.debug("Approval-mode store key is missing or invalid")
        return None

    aget = getattr(store, "aget", None)
    get = getattr(store, "get", None)
    try:
        if callable(aget):
            result = aget(APPROVAL_MODE_NAMESPACE, key)
            item = await result if inspect.isawaitable(result) else result
        elif callable(get):
            item = get(APPROVAL_MODE_NAMESPACE, key)
        else:
            logger.debug("Approval-mode store does not expose get() or aget()")
            return None
    except Exception:
        logger.warning("Could not read approval-mode store item", exc_info=True)
        return None
    return _approval_mode_from_item(item)


async def awrite_approval_mode(
    agent: object,
    thread_id: str,
    *,
    mode: ApprovalMode | str | None = None,
    auto_approve: bool | None = None,
) -> str | None:
    """Persist approval mode through an agent's remote store client.

    Args:
        agent: Agent object. Remote agents expose `aput_store_item`.
        thread_id: LangGraph thread id for the active session.
        mode: Explicit approval mode.
        auto_approve: Compatibility input for the previous Boolean API.

    Returns:
        Store key written, or `None` when the agent has no store writer.
    """
    put = getattr(agent, "aput_store_item", None)
    if put is None:
        return None

    key = approval_mode_key(thread_id)
    await put(
        APPROVAL_MODE_NAMESPACE,
        key,
        approval_mode_payload(mode=mode, auto_approve=auto_approve),
    )
    return key


_APPROVAL_STATE_LOCK_TIMEOUT_SECONDS = 5.0
"""Longest a save waits for the shared install-local approval-state lock."""

_APPROVAL_STATE_THREAD_LOCKS: dict[str, threading.Lock] = {}
_APPROVAL_STATE_THREAD_LOCKS_GUARD = threading.Lock()


def yolo_acknowledgement_path() -> Path:
    """Return the installation-local acknowledgement file path.

    Returns:
        Path under the private dcode state directory.
    """
    from deepagents_code.model_config import DEFAULT_STATE_DIR

    return DEFAULT_STATE_DIR / "approval.json"


def _approval_state_lock_path(path: Path) -> Path:
    """Return the sibling lock file path that serializes approval-state saves.

    Args:
        path: Path to `approval.json`.

    Returns:
        Dedicated `.lock` path next to the state file.
    """
    return path.with_name(f"{path.name}.lock")


def _approval_state_thread_lock(path: Path) -> threading.Lock:
    """Return the process-local mutation lock for an approval-state path.

    Each `_approval_state_lock` builds a fresh `FileLock(thread_local=False)`, so
    the cross-process lock alone does not serialize threads within one dcode
    process. This threading lock guarantees only one in-process thread runs the
    read-merge-write at a time, independent of the filelock backend's cross-thread
    behavior.

    Args:
        path: Path to `approval.json`.

    Returns:
        Process-lifetime lock keyed by the normalized path string.
    """
    key = str(path)
    with _APPROVAL_STATE_THREAD_LOCKS_GUARD:
        lock = _APPROVAL_STATE_THREAD_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _APPROVAL_STATE_THREAD_LOCKS[key] = lock
        return lock


@contextmanager
def _approval_state_lock(path: Path) -> Iterator[None]:
    """Serialize read-merge-write updates to install-local approval state.

    Combines a process-local threading lock with a cross-process `FileLock` on a
    sibling `.lock` file so concurrent YOLO and Auto saves cannot drop each
    other's fields.

    Args:
        path: Path to `approval.json`.

    Yields:
        Control while the caller exclusively holds the mutation lock.

    Callers should handle `filelock.Timeout` (lock wait expired) and `OSError`
    (lock directory creation failure).
    """
    # These run during __enter__, before the lock is held. Any OSError from
    # mkdir/chmod propagates out of the `with` to the caller, which must catch
    # it (the only caller, `_merge_approval_state`, does).
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name != "nt":
        path.parent.chmod(0o700)
    file_lock = FileLock(
        str(_approval_state_lock_path(path)),
        timeout=_APPROVAL_STATE_LOCK_TIMEOUT_SECONDS,
        thread_local=False,
    )
    with _approval_state_thread_lock(path), file_lock:
        yield


def _load_approval_state(path: Path) -> dict[str, object]:
    """Load the install-local approval state file, or an empty dict.

    A missing file is the normal first-run case and returns `{}` silently.
    Unreadable, corrupt, or non-object state also returns `{}` (so callers fail
    closed and re-prompt) but is logged: the next save overwrites the file, so
    this warning is the only surviving evidence of the corruption.

    Args:
        path: Path to `approval.json`.

    Returns:
        Parsed object mapping when valid; otherwise `{}`.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        logger.warning(
            "Ignoring unreadable or corrupt approval state at %s; a re-prompt "
            "may follow and the file will be overwritten on the next save",
            path,
            exc_info=True,
        )
        return {}
    if not isinstance(data, dict):
        logger.warning(
            "Ignoring non-object approval state at %s; a re-prompt may follow "
            "and the file will be overwritten on the next save",
            path,
        )
        return {}
    return data


def _write_approval_state(
    path: Path,
    payload: Mapping[str, object],
    *,
    failure_label: str,
) -> bool:
    """Atomically write install-local approval state.

    Callers that load-then-merge must hold `_approval_state_lock` around both
    the load and this write so concurrent savers cannot clobber each other.

    Args:
        path: Destination path under the private dcode state directory.
        payload: JSON-serializable mapping to persist.
        failure_label: Human-readable label for the warning log on OSError.

    Returns:
        `True` when the private atomic write succeeds, otherwise `False`.
    """
    tmp_path: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if os.name != "nt":
            path.parent.chmod(0o700)
        fd, raw_tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        tmp_path = Path(raw_tmp_path)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, separators=(",", ":"))
            handle.write("\n")
        if os.name != "nt":
            tmp_path.chmod(0o600)
        tmp_path.replace(path)
        if os.name != "nt":
            path.chmod(0o600)
    except OSError:
        logger.warning("Could not persist %s", failure_label, exc_info=True)
        if tmp_path is not None:
            with contextlib.suppress(OSError):
                tmp_path.unlink()
        return False
    return True


def _merge_approval_state(
    path: Path,
    updates: Mapping[str, object],
    *,
    failure_label: str,
) -> bool:
    """Load, merge, and write install-local approval state under a lock.

    Args:
        path: Path to `approval.json`.
        updates: Fields to overlay onto the current state mapping.
        failure_label: Human-readable label for warning logs.

    Returns:
        `True` when the locked merge-write succeeds, otherwise `False`.
    """
    try:
        with _approval_state_lock(path):
            payload = {
                **_load_approval_state(path),
                **updates,
                "version": 1,
            }
            return _write_approval_state(
                path,
                payload,
                failure_label=failure_label,
            )
    except Timeout:
        logger.warning(
            "Timed out waiting to persist %s",
            failure_label,
            exc_info=True,
        )
        return False
    except OSError:
        logger.warning(
            "Could not lock approval state for %s", failure_label, exc_info=True
        )
        return False


def has_yolo_acknowledgement(path: Path | None = None) -> bool:
    """Return whether the current unrestricted-mode warning was accepted.

    Args:
        path: Alternate acknowledgement path for tests.

    Returns:
        `True` only for a valid record matching the current policy version.
    """
    target = path or yolo_acknowledgement_path()
    data = _load_approval_state(target)
    return (
        data.get("version") == 1
        and data.get("policy_version") == YOLO_ACKNOWLEDGEMENT_POLICY_VERSION
        and data.get("acknowledged") is True
    )


def save_yolo_acknowledgement(path: Path | None = None) -> bool:
    """Persist the current unrestricted-mode warning acknowledgement.

    Merges into any existing approval state so first-run Auto notice fields are
    preserved. Concurrent saves are serialized with a cross-process lock.

    Args:
        path: Alternate acknowledgement path for tests.

    Returns:
        `True` when the private atomic write succeeds, otherwise `False`.
    """
    target = path or yolo_acknowledgement_path()
    return _merge_approval_state(
        target,
        {
            "policy_version": YOLO_ACKNOWLEDGEMENT_POLICY_VERSION,
            "acknowledged": True,
        },
        failure_label="YOLO acknowledgement",
    )


def has_auto_mode_notice(path: Path | None = None) -> bool:
    """Return whether the current Auto first-enable notice was already shown.

    The notice self-versions on `auto_notice_version` and deliberately does not
    gate on the top-level `version` (unlike `has_yolo_acknowledgement`), so the
    two records can evolve independently within one `approval.json`.

    Args:
        path: Alternate approval-state path for tests.

    Returns:
        `True` only when a shown notice matches `AUTO_NOTICE_VERSION`.
    """
    target = path or yolo_acknowledgement_path()
    data = _load_approval_state(target)
    return (
        data.get("auto_notice_shown") is True
        and data.get("auto_notice_version") == AUTO_NOTICE_VERSION
    )


def save_auto_mode_notice(path: Path | None = None) -> bool:
    """Persist that the Auto first-enable notice was shown.

    Merges into any existing approval state so YOLO acknowledgement fields are
    preserved. Concurrent saves are serialized with a cross-process lock.
    Callers should fail open when this returns `False`.

    Args:
        path: Alternate approval-state path for tests.

    Returns:
        `True` when the private atomic write succeeds, otherwise `False`.
    """
    target = path or yolo_acknowledgement_path()
    return _merge_approval_state(
        target,
        {
            "auto_notice_version": AUTO_NOTICE_VERSION,
            "auto_notice_shown": True,
        },
        failure_label="Auto mode notice",
    )
