"""Forward delta-channel history reads through langgraph-api's saver adapter.

The LangGraph dev server wraps a custom checkpointer in
`langgraph_api._checkpointer._adapter._CustomCheckpointerAdapter`. The adapter
forwards unknown attributes to the inner saver through `__getattr__`, but
`BaseCheckpointSaver` already defines `aget_delta_channel_history`, so the
forwarding never fires. The adapter then runs the base ancestor walk: one
`aget_tuple` and one full checkpoint decode per ancestor. The SQLite saver
overrides the method with two indexed queries, and long threads pay the
difference on every state read and every run start.

This module is a best-effort patch over a private langgraph-api class. It only
patches when the adapter still inherits the base method, and it falls back to
the base walk when the inner saver has no override of its own. Remove it when
the adapter forwards these methods itself.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langgraph.checkpoint.base import BaseCheckpointSaver

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from langchain_core.runnables import RunnableConfig
    from langgraph.checkpoint.base import DeltaChannelHistory

logger = logging.getLogger(__name__)

_BASE_ASYNC = BaseCheckpointSaver.aget_delta_channel_history
_BASE_SYNC = BaseCheckpointSaver.get_delta_channel_history


def _overriding_inner(
    adapter: BaseCheckpointSaver, name: str
) -> BaseCheckpointSaver | None:
    """Return the adapter's inner saver if it overrides the base method `name`."""
    inner = adapter.__dict__.get("_inner")
    if not isinstance(inner, BaseCheckpointSaver):
        return None
    if getattr(type(inner), name) is getattr(BaseCheckpointSaver, name):
        return None
    return inner


async def _aget_delta_channel_history(
    self: BaseCheckpointSaver, *, config: RunnableConfig, channels: Sequence[str]
) -> Mapping[str, DeltaChannelHistory]:
    inner = _overriding_inner(self, "aget_delta_channel_history")
    if inner is None:
        return await _BASE_ASYNC(self, config=config, channels=channels)
    return await inner.aget_delta_channel_history(config=config, channels=channels)


def _get_delta_channel_history(
    self: BaseCheckpointSaver, *, config: RunnableConfig, channels: Sequence[str]
) -> Mapping[str, DeltaChannelHistory]:
    inner = _overriding_inner(self, "get_delta_channel_history")
    if inner is None:
        return _BASE_SYNC(self, config=config, channels=channels)
    return inner.get_delta_channel_history(config=config, channels=channels)


def install_delta_history_forwarding() -> None:
    """Make langgraph-api's checkpointer adapter use the inner saver's fast path.

    Call this in the server process before the adapter serves requests. It is
    idempotent and a no-op when the adapter is missing or already overrides
    the methods.
    """
    try:
        # The adapter has no public seam; this patch exists to reach it.
        from langgraph_api._checkpointer._adapter import (  # noqa: PLC2701
            _CustomCheckpointerAdapter,
        )
    except ImportError:
        logger.warning("langgraph-api checkpointer adapter not found; skipping patch")
        return
    adapter = _CustomCheckpointerAdapter
    if adapter.aget_delta_channel_history is _BASE_ASYNC:
        adapter.aget_delta_channel_history = _aget_delta_channel_history
    if adapter.get_delta_channel_history is _BASE_SYNC:
        adapter.get_delta_channel_history = _get_delta_channel_history
