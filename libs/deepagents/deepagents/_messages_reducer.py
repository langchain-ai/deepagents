"""Local `DeltaChannel` reducer for the messages key.

Adapted from langgraph's `_messages_delta_reducer` (PR #7729). The upstream
version coerces `BaseMessageChunk` writes to full messages for parity with
`add_messages`. Deepagents never writes chunks to the messages channel —
`langchain.agents.create_agent` appends full `AIMessage` objects, and
streaming via `astream_events` operates on the output side, not the state
side — so we skip the per-message coercion.
"""

from __future__ import annotations

from typing import Any, cast

from langchain_core.messages import (
    AnyMessage,
    BaseMessage,
    RemoveMessage,
    convert_to_messages,
)
from langgraph.graph.message import REMOVE_ALL_MESSAGES


def _messages_delta_reducer(  # noqa: C901
    state: list[AnyMessage], writes: list[list[AnyMessage]]
) -> list[AnyMessage]:
    """Batch reducer for use with `DeltaChannel` on the messages key.

    Dedups by ID, tombstones via `RemoveMessage`, resets on
    `REMOVE_ALL_MESSAGES`. ID-less messages are appended without ID
    assignment — checkpointers serialize pending writes before
    `update()` runs, so IDs assigned inside the reducer never reach
    stored writes and would differ on replay, defeating deduplication.

    Raw dict / string / tuple inputs are coerced to typed `BaseMessage` so
    HTTP-driven graphs work without a separate coercion step.
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
    # Steady state: the reducer's own output is already typed BaseMessages,
    # so skip convert_to_messages on the fast path. Only raw input (initial
    # dicts, deserialized blobs) hits the slow path.
    state_msgs = state if state and isinstance(state[0], BaseMessage) else cast("list[AnyMessage]", convert_to_messages(state))
    msgs = cast("list[AnyMessage]", convert_to_messages(flat))

    # REMOVE_ALL_MESSAGES resets everything; find the last sentinel and
    # discard all state plus all writes before it.
    remove_all_idx = None
    for idx, m in enumerate(msgs):
        if isinstance(m, RemoveMessage) and m.id == REMOVE_ALL_MESSAGES:
            remove_all_idx = idx
    if remove_all_idx is not None:
        state_msgs = []
        msgs = msgs[remove_all_idx + 1 :]

    index: dict[str, int] = {m.id: i for i, m in enumerate(state_msgs) if m.id is not None}
    result: list[AnyMessage | None] = list(state_msgs)
    for msg in msgs:
        mid = msg.id
        if mid is None:
            result.append(msg)
        elif isinstance(msg, RemoveMessage):
            if mid in index:
                result[index[mid]] = None
                del index[mid]
        elif mid in index:
            result[index[mid]] = msg
        else:
            index[mid] = len(result)
            result.append(msg)
    return [m for m in result if m is not None]
