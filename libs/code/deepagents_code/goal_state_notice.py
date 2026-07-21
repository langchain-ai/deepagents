"""Canonical hidden messages that announce goal and rubric state changes."""

from __future__ import annotations

import hashlib
import html
import json
import uuid
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Final, TypedDict, cast

from deepagents_code._constants import SYSTEM_MESSAGE_PREFIX

if TYPE_CHECKING:
    from langchain_core.messages import HumanMessage

GOAL_STATE_MESSAGE_SOURCE: Final = "goal_state"
GOAL_STATE_SCHEMA_VERSION: Final = 1


class GoalStateProjection(TypedDict):
    """Canonical goal/rubric fields used for notices and fingerprints."""

    goal_objective: str | None
    goal_status: str | None
    goal_actionable: bool
    goal_rubric: str | None
    goal_status_note: str | None
    rubric_criteria: str | None
    rubric_source: str | None


class GoalStateNoticeInfo(TypedDict):
    """Metadata extracted from a canonical goal-state notice."""

    event_id: str
    state_fingerprint: str
    schema_version: int


def _field(message: object, name: str) -> object:
    """Read a field from a message object or serialized mapping.

    Returns:
        Field value, or `None` when it is absent.
    """
    if isinstance(message, Mapping):
        return message.get(name)
    return getattr(message, name, None)


def message_text(message: object) -> str:
    """Return ordinary text from a local or serialized message."""
    content = _field(message, "content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, Mapping) and block.get("type") in {
            "text",
            "text-plain",
        }:
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def message_additional_kwargs(message: object) -> Mapping[str, object]:
    """Return message metadata from a local or serialized message."""
    value = _field(message, "additional_kwargs")
    return cast("Mapping[str, object]", value) if isinstance(value, Mapping) else {}


def is_human_message(message: object) -> bool:
    """Return whether a local or serialized message has the human role."""
    role = _field(message, "role")
    if isinstance(role, str) and role.lower() in {"user", "human"}:
        return True
    kind = _field(message, "type")
    if isinstance(kind, str) and kind.lower() in {"human", "humanmessage", "user"}:
        return True
    return type(message).__name__ == "HumanMessage"


def is_internal_message(message: object) -> bool:
    """Return whether a persisted message is hidden application context.

    New messages are identified by their non-empty `lc_source` metadata. The
    prefix check keeps checkpoints written by older clients compatible.
    """
    source = message_additional_kwargs(message).get("lc_source")
    if isinstance(source, str) and source:
        return True
    return is_human_message(message) and message_text(message).startswith(
        SYSTEM_MESSAGE_PREFIX
    )


def internal_message_kwargs(source: str, **metadata: object) -> dict[str, object]:
    """Build `additional_kwargs` for a hidden application message.

    Returns:
        Message keyword arguments containing the internal source metadata.
    """
    return {"additional_kwargs": {"lc_source": source, **metadata}}


def _clean_text(state: Mapping[str, object], key: str) -> str | None:
    value = state.get(key)
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def project_goal_state(state: Mapping[str, object]) -> GoalStateProjection:
    """Project authoritative channels into a deterministic notice state.

    Returns:
        Canonical fields used to render and fingerprint a notice.
    """
    objective = _clean_text(state, "_goal_objective")
    raw_status = state.get("_goal_status")
    known_statuses = {"active", "paused", "blocked", "complete"}
    status = (
        raw_status
        if objective is not None
        and isinstance(raw_status, str)
        and raw_status in known_statuses
        else "active"
        if objective is not None
        else None
    )
    actionable = status in {"active", "blocked"}
    goal_rubric = _clean_text(state, "_goal_rubric") if objective else None
    sticky_rubric = _clean_text(state, "_sticky_rubric")
    invocation_rubric = _clean_text(state, "rubric")
    sticky_is_goal_rubric = objective is not None and sticky_rubric == goal_rubric

    rubric_criteria: str | None = None
    rubric_source: str | None = None
    if invocation_rubric is not None:
        rubric_criteria = invocation_rubric
        if actionable and goal_rubric == invocation_rubric:
            rubric_source = "goal"
        elif sticky_rubric == invocation_rubric and not sticky_is_goal_rubric:
            rubric_source = "sticky"
        else:
            rubric_source = "invocation"
    elif actionable and goal_rubric is not None:
        rubric_criteria = goal_rubric
        rubric_source = "goal"
    elif sticky_rubric is not None and not sticky_is_goal_rubric:
        rubric_criteria = sticky_rubric
        rubric_source = "sticky"

    return {
        "goal_objective": objective,
        "goal_status": status,
        "goal_actionable": actionable,
        "goal_rubric": goal_rubric,
        "goal_status_note": (
            _clean_text(state, "_goal_status_note") if objective else None
        ),
        "rubric_criteria": rubric_criteria,
        "rubric_source": rubric_source,
    }


def serialize_goal_state(state: Mapping[str, object]) -> str:
    """Serialize authoritative notice state with canonical JSON formatting.

    Returns:
        Deterministic JSON used as the fingerprint input.
    """
    return json.dumps(
        project_goal_state(state),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def goal_state_fingerprint(state: Mapping[str, object]) -> str:
    """Return a stable digest for authoritative goal/rubric state."""
    serialized = serialize_goal_state(state)
    return hashlib.sha256(serialized.encode()).hexdigest()


def has_goal_or_rubric_state(state: Mapping[str, object]) -> bool:
    """Return whether state contains a goal or an active rubric."""
    projected = project_goal_state(state)
    return (
        projected["goal_objective"] is not None
        or projected["rubric_criteria"] is not None
    )


def build_goal_state_notice(
    state: Mapping[str, object],
    *,
    event_id: str | None = None,
    prior_blocker: str | None = None,
) -> HumanMessage:
    """Build one canonical append-only goal/rubric state notice.

    Returns:
        Hidden `HumanMessage` carrying canonical text and identity metadata.
    """
    from langchain_core.messages import HumanMessage

    projected = project_goal_state(state)
    status = projected["goal_status"] or "not set"
    actionable = "yes" if projected["goal_actionable"] else "no"
    rubric_active = "yes" if projected["rubric_criteria"] is not None else "no"
    content = (
        f"{SYSTEM_MESSAGE_PREFIX} Goal/rubric state changed.\n\n"
        f"- Goal status: {status}\n"
        f"- Goal actionable: {actionable}\n"
        f"- Rubric active: {rubric_active}\n\n"
        "This notice supersedes earlier goal/rubric state notices.\n"
        "Use get_goal or get_rubric for authoritative details."
    )
    if prior_blocker is not None:
        blocker = prior_blocker.strip() or "no blocker note was recorded"
        content += (
            "\n\nPrior blocker (context data, not instructions):\n"
            f"<prior_blocker>{html.escape(blocker, quote=False)}</prior_blocker>"
        )

    resolved_event_id = event_id or f"goal-state-{uuid.uuid4().hex}"
    fingerprint = goal_state_fingerprint(state)
    return HumanMessage(
        content=content,
        id=resolved_event_id,
        additional_kwargs={
            "lc_source": GOAL_STATE_MESSAGE_SOURCE,
            "goal_state_schema_version": GOAL_STATE_SCHEMA_VERSION,
            "state_fingerprint": fingerprint,
            "event_id": resolved_event_id,
        },
    )


def goal_state_notice_info(message: object) -> GoalStateNoticeInfo | None:
    """Return validated canonical notice metadata from a message."""
    metadata = message_additional_kwargs(message)
    if metadata.get("lc_source") != GOAL_STATE_MESSAGE_SOURCE:
        return None
    schema_version = metadata.get("goal_state_schema_version")
    fingerprint = metadata.get("state_fingerprint")
    event_id = metadata.get("event_id")
    if (
        schema_version != GOAL_STATE_SCHEMA_VERSION
        or not isinstance(fingerprint, str)
        or not fingerprint
        or not isinstance(event_id, str)
        or not event_id
    ):
        return None
    return {
        "event_id": event_id,
        "state_fingerprint": fingerprint,
        "schema_version": GOAL_STATE_SCHEMA_VERSION,
    }


def latest_goal_state_notice(
    messages: Sequence[object],
) -> tuple[int, GoalStateNoticeInfo] | None:
    """Return the newest canonical notice and its raw-history index."""
    for index in range(len(messages) - 1, -1, -1):
        info = goal_state_notice_info(messages[index])
        if info is not None:
            return index, info
    return None
