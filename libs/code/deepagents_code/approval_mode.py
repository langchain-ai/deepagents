"""Live approval-mode state shared through the LangGraph Store."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from hashlib import sha256
from typing import TypedDict

logger = logging.getLogger(__name__)

APPROVAL_MODE_NAMESPACE: tuple[str, str] = ("deepagents_code", "approval_mode")
"""Store namespace for per-thread approval-mode control records."""


class ApprovalModePayload(TypedDict):
    """Stored approval-mode control payload."""

    auto_approve: bool


def approval_mode_key(thread_id: str) -> str:
    """Return the store key for a thread's live approval mode.

    Args:
        thread_id: LangGraph thread id for the active session.

    Returns:
        Deterministic store key that does not expose the raw thread id.
    """
    return sha256(thread_id.encode("utf-8")).hexdigest()


def approval_mode_payload(*, auto_approve: bool) -> ApprovalModePayload:
    """Return the stored approval-mode payload.

    Args:
        auto_approve: Whether gated tool calls should skip HITL approval.

    Returns:
        JSON-serializable store value.
    """
    return {"auto_approve": auto_approve}


def _item_value(item: object) -> object:
    """Extract a store item's value from SDK and runtime item shapes.

    Returns:
        The item's stored value, or `None` when the shape is unrecognized.
    """
    if isinstance(item, Mapping):
        return item.get("value")
    return getattr(item, "value", None)


def _coerce_store_item(item: object) -> bool | None:
    """Return the `auto_approve` bool from a store item, else `None`.

    Shared by the sync and async readers so both apply the same fail-closed
    parsing: a missing item, unrecognized shape, or non-bool `auto_approve`
    yields `None`.
    """
    if item is None:
        logger.debug("Approval-mode store item is missing")
        return None

    value = _item_value(item)
    auto_approve = value.get("auto_approve") if isinstance(value, Mapping) else None
    if isinstance(auto_approve, bool):
        return auto_approve

    logger.debug("Approval-mode store item has invalid contents")
    return None


def _validate_store_key(store: object, key: str | None) -> bool:
    """Return whether `store`/`key` are usable, logging why when they are not.

    The `isinstance` guard still rejects non-string keys as defense-in-depth,
    since the value crosses the JSON/RemoteGraph boundary before reaching here.
    """
    if store is None:
        logger.debug("Approval-mode store is unavailable")
        return False
    if not isinstance(key, str) or not key:
        logger.debug("Approval-mode store key is missing or invalid")
        return False
    return True


def read_approval_mode_from_store(store: object, key: str | None) -> bool | None:
    """Read a live approval mode from the server-side LangGraph Store.

    Synchronous variant. On the LangGraph API server the runtime Store is an
    `AsyncBatchedBaseStore` whose synchronous `get()` raises on the running
    event loop, so prefer `aread_approval_mode_from_store` on async paths; this
    remains for in-process stores whose sync `get()` is safe.

    Args:
        store: `request.runtime.store` from the graph server.
        key: Store key produced by `approval_mode_key`.

    Returns:
        `True` or `False` when the store contains a valid mode, otherwise
        `None`. Callers should treat `None` as fail-closed.
    """
    if not _validate_store_key(store, key):
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
    return _coerce_store_item(item)


async def aread_approval_mode_from_store(store: object, key: str | None) -> bool | None:
    """Read a live approval mode from the LangGraph Store asynchronously.

    The `when` interrupt predicate is synchronous, but the LangGraph API
    server's runtime Store is an `AsyncBatchedBaseStore` whose synchronous
    `get()` raises `asyncio.InvalidStateError` when called on the server's
    running event loop. This awaits `aget()` (the supported interface) so the
    live mode is actually read there; a prefetch middleware calls this before
    the model turn and threads the result into state for the sync predicate.

    Args:
        store: `runtime.store` from the graph server.
        key: Store key produced by `approval_mode_key`.

    Returns:
        `True` or `False` when the store contains a valid mode, otherwise
        `None`. Callers should treat `None` as fail-closed.
    """
    if not _validate_store_key(store, key):
        return None

    aget = getattr(store, "aget", None)
    get = getattr(store, "get", None)
    try:
        if aget is not None:
            item = await aget(APPROVAL_MODE_NAMESPACE, key)
        elif get is not None:
            item = get(APPROVAL_MODE_NAMESPACE, key)
        else:
            logger.debug("Approval-mode store exposes neither aget() nor get()")
            return None
    except Exception:
        logger.warning("Could not read approval-mode store item", exc_info=True)
        return None
    return _coerce_store_item(item)


async def awrite_approval_mode(
    agent: object,
    thread_id: str,
    *,
    auto_approve: bool,
) -> str | None:
    """Persist approval mode through an agent's remote store client.

    Args:
        agent: Agent object. Remote agents expose `aput_store_item`; agents
            without a writer use run context only.
        thread_id: LangGraph thread id for the active session.
        auto_approve: Whether gated tool calls should skip HITL approval.

    Returns:
        Store key written, or `None` when the agent has no store writer.

    Notes:
        Remote agents rely on the server-side store being visible to the
        running graph before the next gated tool predicate executes.
    """
    put = getattr(agent, "aput_store_item", None)
    if put is None:
        return None

    key = approval_mode_key(thread_id)
    await put(
        APPROVAL_MODE_NAMESPACE,
        key,
        approval_mode_payload(auto_approve=auto_approve),
    )
    return key
