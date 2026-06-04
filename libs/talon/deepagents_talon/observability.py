"""Observability helpers for Talon runtime processes."""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}
DEFAULT_LANGSMITH_PROJECT = "deepagents-talon"


def langsmith_tracing_enabled(env: Mapping[str, str]) -> bool:
    """Return whether LangSmith tracing is configured for this process.

    Args:
        env: Environment values visible to the Talon runtime.

    Returns:
        `True` when tracing is explicitly enabled and an API key is present.
    """
    tracing = env.get("LANGSMITH_TRACING", "")
    return tracing.lower() in TRUTHY_ENV_VALUES and bool(env.get("LANGSMITH_API_KEY"))


@contextmanager
def langsmith_trace_context(
    env: Mapping[str, str],
    *,
    assistant_id: str,
    conversation_id: str,
    metadata: Mapping[str, object],
) -> Iterator[None]:
    """Open a LangSmith tracing context for a single agent run when configured.

    Args:
        env: Environment values visible to the Talon runtime.
        assistant_id: Assistant namespace for trace metadata.
        conversation_id: Conversation or thread id for trace metadata.
        metadata: Agent request metadata attached to the trace.
    """
    if not langsmith_tracing_enabled(env):
        yield
        return

    try:
        from langsmith import tracing_context  # noqa: PLC0415
    except ImportError:
        logging.getLogger(__name__).warning(
            "LangSmith tracing requested but langsmith is not installed",
        )
        yield
        return

    trigger = metadata.get("trigger")
    trace_metadata = {
        "assistant_id": assistant_id,
        "conversation_id": conversation_id,
        **dict(metadata),
    }
    tags = ["deepagents-talon", f"assistant:{assistant_id}"]
    if isinstance(trigger, str):
        tags.append(f"trigger:{trigger}")

    with tracing_context(
        project_name=env.get("LANGSMITH_PROJECT", DEFAULT_LANGSMITH_PROJECT),
        tags=tags,
        metadata=trace_metadata,
        enabled=True,
    ):
        yield


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    """Emit one structured JSON event through the standard logger.

    Args:
        logger: Logger used by the emitting subsystem.
        event: Stable event name.
        fields: JSON-serializable event fields.
    """
    payload = {"event": event, **fields}
    logger.info("talon_event %s", json.dumps(payload, sort_keys=True, default=str))
