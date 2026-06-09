"""Observability helpers for Talon runtime processes.

Talon is an experimental runtime and is subject to change or removal at any time.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from hashlib import sha256
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlsplit, urlunsplit

if TYPE_CHECKING:
    from collections.abc import Iterator

TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}
DEFAULT_LANGSMITH_PROJECT = "deepagents-talon"
REDACTED_LOG_VALUE = "[redacted]"
_SECRET_KEY_MARKERS = (
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "credential",
    "cookie",
    "oauth",
    "password",
    "secret",
    "session",
    "token",
)
_PII_KEYS = frozenset({"conversation_id", "message_id", "sender_id"})
_BEARER_RE = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/-]+=*")
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(api[_-]?key|authorization|password|secret|token)=([^&\s]+)",
)


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
    payload = {"event": event, **_redact_mapping(fields)}
    logger.info("talon_event %s", json.dumps(payload, sort_keys=True, default=str))


def redact_for_logging(value: object) -> object:
    """Return a log-safe copy of `value`.

    Args:
        value: Arbitrary structured payload destined for logs.

    Returns:
        A JSON-compatible value with obvious secrets and URL query data removed.
    """
    if isinstance(value, Mapping):
        redacted: dict[str, object] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key)
            if _is_secret_key(key):
                redacted[key] = REDACTED_LOG_VALUE
            else:
                redacted[key] = redact_for_logging(raw_value)
        return redacted

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [redact_for_logging(item) for item in value]

    if isinstance(value, str):
        return _redact_string(value)

    return value


def _redact_mapping(value: Mapping[str, object]) -> dict[str, object]:
    return cast("dict[str, object]", redact_for_logging(value))


def _is_secret_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return normalized in _PII_KEYS or any(marker in normalized for marker in _SECRET_KEY_MARKERS)


def _redact_string(value: str) -> str:
    text = _sanitize_url(value)
    text = _BEARER_RE.sub("Bearer [redacted]", text)
    return _SECRET_ASSIGNMENT_RE.sub(r"\1=[redacted]", text)


def _sanitize_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return value
    if parsed.scheme not in {"http", "https", "ws", "wss"} or not parsed.netloc:
        return value

    host = parsed.hostname or ""
    try:
        port = parsed.port
    except ValueError:
        return value
    netloc = host if port is None else f"{host}:{port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))


def stable_log_ref(value: str) -> str:
    """Return a stable non-secret reference for a sensitive identifier.

    Args:
        value: Raw identifier that should not be emitted directly.

    Returns:
        Short SHA-256-derived reference suitable for correlating log events.
    """
    return sha256(value.encode("utf-8")).hexdigest()[:12]
