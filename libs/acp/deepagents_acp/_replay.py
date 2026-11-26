"""Rebuild the updates that show a saved conversation to a client loading it."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from acp.schema import UserMessageChunk

from deepagents_acp._messages import _content_blocks, _content_updates
from deepagents_acp._tool_calls import (
    _fail_open_calls,
    _finish_tool_call,
    _start_message_tool_calls,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterable, Iterable, Iterator

    from langgraph.types import StateSnapshot

    from deepagents_acp._tool_calls import _ToolCall


async def _history_messages(history: AsyncIterable[StateSnapshot]) -> list[Any]:
    """Return every message a thread has held, oldest first, each in its latest form.

    Keeps no snapshot past its turn in the loop, so memory grows with the
    conversation rather than with how many checkpoints it took.

    Args:
        history: The thread's state history, newest first.
    """
    latest: dict[str, Any] = {}
    order: list[str] = []
    async for snapshot in history:
        current = {message.id: message for message in snapshot.values.get("messages", [])}
        for message_id, message in current.items():
            latest.setdefault(message_id, message)
        # Walking back in time, this snapshot's messages appeared before those only newer ones hold.
        order = [*current, *(message_id for message_id in order if message_id not in current)]
    return [latest[message_id] for message_id in order]


def _user_updates(message: Any) -> Iterator[UserMessageChunk]:
    """Yield a persisted user message's content."""
    # `lc_source` marks a notification the agent inserted (local context, a
    # pinned skill, a summary), not something the user typed.
    if message.additional_kwargs.get("lc_source"):
        return
    for block in _content_blocks(message):
        yield UserMessageChunk(
            session_update="user_message_chunk", content=block, message_id=message.id
        )


def _replay_updates(messages: Iterable[Any]) -> Iterator[Any]:
    """Yield the session updates for a saved conversation, in order.

    Assistant content and tool calls go through the functions the live stream
    uses, so a replayed turn cannot drift from how it first looked. Nothing a
    live turn sends once at its end, such as usage or a title, is replayed.
    A call that never got a result is shown as failed, as a cancelled turn does.
    """
    calls: dict[str, _ToolCall] = {}
    for message in messages:
        if message.type == "human":
            # Tool-call ids are unique within a turn only; some providers reuse them.
            # An open call stays: `PatchToolCalls` answers it after the next user message.
            if not message.additional_kwargs.get("lc_source"):
                calls = {tool_id: call for tool_id, call in calls.items() if not call.finished}
            yield from _user_updates(message)
        elif message.type == "ai":
            yield from _content_updates(message, message.id)
            yield from _start_message_tool_calls(calls, message)
        elif message.type == "tool" and message.tool_call_id in calls:
            # Unlike the live stream, a result whose call is not in history is not shown.
            yield from _finish_tool_call(calls, message)
    yield from _fail_open_calls(calls)
