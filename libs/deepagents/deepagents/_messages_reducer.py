"""Local copy of `_messages_delta_reducer` from langgraph PR #7729.

Vendored so that deepagents can pin to the latest published langgraph alpha
without waiting for the PR to merge and ship. Drop this module and switch
`graph.py` back to `from langgraph.graph.message import _messages_delta_reducer`
once the upstream release contains the fix.

Source: https://github.com/langchain-ai/langgraph/pull/7729
"""

from __future__ import annotations

import uuid
from typing import Any, cast

from langchain_core.messages import (
    AnyMessage,
    BaseMessage,
    BaseMessageChunk,
    RemoveMessage,
    convert_to_messages,
    message_chunk_to_message,
)
from langgraph.graph.message import REMOVE_ALL_MESSAGES


def _messages_delta_reducer(state: list[AnyMessage], writes: list[list[AnyMessage]]) -> list[AnyMessage]:  # noqa: C901, PLR0912
    """**Experimental.** Batch reducer for use with `DeltaChannel`.

    Provides full `add_messages` parity: dedup by ID, `RemoveMessage`
    tombstoning, `REMOVE_ALL_MESSAGES` reset, `BaseMessageChunk` coercion,
    and UUID assignment for ID-less messages — all in a single batched pass.

    This reducer is batching-invariant, as required by `DeltaChannel`:
    `reducer(reducer(state, xs), ys) == reducer(state, xs + ys)`.

    Raw dict / string / tuple inputs are coerced to typed `BaseMessage`
    objects so that HTTP-driven graphs work without a separate coercion step.
    """
    # Each write is either a list of message-likes or a single message-like
    # (BaseMessage / dict / str / tuple). Only lists flatten; everything
    # else is one message.
    flat: list[Any] = []
    for w in writes:
        if isinstance(w, list):
            flat.extend(w)
        else:
            flat.append(w)
    # Steady state: the reducer's own output is already typed BaseMessages
    # (never chunks), so skip convert_to_messages on the fast path.
    # Only raw input (initial dicts, deserialized blobs) hits the slow path.
    if state and isinstance(state[0], BaseMessage):
        state_msgs = state
    else:
        state_msgs = cast(
            "list[AnyMessage]",
            [message_chunk_to_message(cast("BaseMessageChunk", m)) for m in convert_to_messages(state)],
        )
    # Coerce chunks to full messages — streaming nodes can emit BaseMessageChunk.
    msgs = cast(
        "list[AnyMessage]",
        [message_chunk_to_message(cast("BaseMessageChunk", m)) for m in convert_to_messages(flat)],
    )

    # REMOVE_ALL_MESSAGES resets everything; find the last sentinel and
    # discard all state plus all writes before it.
    remove_all_idx = None
    for idx, m in enumerate(msgs):
        if isinstance(m, RemoveMessage) and m.id == REMOVE_ALL_MESSAGES:
            remove_all_idx = idx
    if remove_all_idx is not None:
        state_msgs = []
        msgs = msgs[remove_all_idx + 1 :]

    # Build index and assign missing IDs in one pass (parity with add_messages
    # so that eviction and RemoveMessage tombstoning work on ID-less messages).
    index: dict[str, int] = {}
    for i, m in enumerate(state_msgs):
        if m.id is None:
            m.id = str(uuid.uuid4())
        index[m.id] = i
    result: list[AnyMessage | None] = list(state_msgs)
    for msg in msgs:
        if msg.id is None:
            msg.id = str(uuid.uuid4())
        mid = msg.id
        if isinstance(msg, RemoveMessage):
            if mid in index:
                result[index[mid]] = None
                del index[mid]
        elif mid in index:
            result[index[mid]] = msg
        else:
            index[mid] = len(result)
            result.append(msg)
    return [m for m in result if m is not None]
