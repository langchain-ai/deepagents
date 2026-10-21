"""Mail from an async subagent to the agent that launched it.

The child appends to an outbox in its own checkpointed state and sends the parent
a mail wake; the parent pulls unread mail past a per-task cursor. Mail is
delivered only by the pull, so a mail wake carries no content and can be lost.
"""

from collections.abc import Mapping
from typing import Annotated, NotRequired, TypedDict, cast

from langchain.agents.middleware.types import AgentState, PrivateStateAttr


class _Mail(TypedDict):
    id: str
    """Stable across retries: the posting run and tool call."""

    run_id: str | None
    content: str


class _Outbox(TypedDict):
    mail: list[_Mail]
    """Append-only, so a position is a stable cursor."""

    notified_count: int
    """How much of `mail` the child has notified the parent about."""


def _merge_outbox(existing: _Outbox | None, update: _Outbox) -> _Outbox:
    current = existing or {}
    return {"mail": [*current.get("mail", []), *update["mail"]], "notified_count": max(current.get("notified_count", 0), update["notified_count"])}


def _merge_cursors(existing: dict[str, int] | None, update: dict[str, int]) -> dict[str, int]:
    merged = dict(existing or {})
    for task_id, cursor in update.items():
        merged[task_id] = max(merged.get(task_id, 0), cursor)
    return merged


class _OutboxState(AgentState):
    deepagents_outbox: Annotated[NotRequired[_Outbox], PrivateStateAttr, _merge_outbox]


def _post(mail: _Mail) -> _Outbox:
    """The state update that appends `mail`."""
    return {"mail": [mail], "notified_count": 0}


def _unnotified(outbox: _Outbox | None) -> int | None:
    """The `notified_count` to record once the mail wake is sent, or `None` when there's nothing new."""
    mail, notified = (outbox or {}).get("mail", []), (outbox or {}).get("notified_count", 0)
    return len(mail) if len(mail) > notified else None


def _unread(values: object, cursor: int) -> list[_Mail]:
    """Mail in a child's state values past `cursor`."""
    outbox = values.get("deepagents_outbox") if isinstance(values, Mapping) else None
    return list(cast("_Outbox", outbox).get("mail", [])[cursor:]) if isinstance(outbox, Mapping) else []
