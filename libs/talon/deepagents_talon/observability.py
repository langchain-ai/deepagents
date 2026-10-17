"""Observability helpers for Talon runtime processes.

Talon is an experimental runtime and is subject to change or removal at any time.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from hashlib import sha256
from itertools import islice
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlsplit, urlunsplit

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import BaseMessage

if TYPE_CHECKING:
    from collections.abc import Iterator
    from uuid import UUID

    from langchain_core.outputs import LLMResult

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
    r"(?i)\b(?P<key>[a-z_][a-z0-9_.-]*)(?P<separator>\s*=\s*)(?P<value>[^&\s]+)",
)
_SECRET_COLON_ASSIGNMENT_RE = re.compile(
    r"""(?i)(?P<prefix>["']?(?P<key>[a-z_][a-z0-9_.-]*)["']?\s*:\s*)"""
    r"""(?:"[^"]*"|'[^']*'|[^,}\s]+)""",
)
AGENT_ACTIVITY_LOGGING_ENV_KEY = "DEEPAGENTS_TALON_AGENT_ACTIVITY_LOGGING"
AGENT_ACTIVITY_PREVIEW_LIMIT = 1000
_ACTIVITY_PREVIEW_DEPTH = 5
_ACTIVITY_PREVIEW_ITEMS = 20
_ACTIVITY_PREVIEW_VALUES = 100
_ACTIVITY_TRUNCATION_MARKER = "…[truncated]"


class AgentActivityCallback(AsyncCallbackHandler):
    """Emit bounded local activity logs for one agent invocation."""

    def __init__(self, logger: logging.Logger, conversation_id: str) -> None:
        """Create an activity callback scoped to one conversation."""
        self._logger = logger
        self._conversation_ref = stable_log_ref(conversation_id)
        self._run_started_at = time.perf_counter()
        self._activity_started_at: dict[UUID, float] = {}
        self._tool_names: dict[UUID, str] = {}

    def run_started(self, trigger: object) -> None:
        """Log the start of the enclosing agent invocation."""
        fields: dict[str, object] = {"conversation_ref": self._conversation_ref}
        if isinstance(trigger, str) and trigger:
            fields["trigger"] = trigger
        log_event(self._logger, "agent.run.started", **fields)

    def run_completed(self, text: str) -> None:
        """Log successful completion of the enclosing agent invocation."""
        log_event(
            self._logger,
            "agent.run.completed",
            conversation_ref=self._conversation_ref,
            duration_ms=_duration_ms(self._run_started_at),
            text_chars=len(text),
        )

    def run_failed(self, error: BaseException) -> None:
        """Log failure of the enclosing agent invocation."""
        log_event(
            self._logger,
            "agent.run.failed",
            conversation_ref=self._conversation_ref,
            duration_ms=_duration_ms(self._run_started_at),
            error_type=type(error).__name__,
        )

    async def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        **_kwargs: Any,
    ) -> None:
        """Log the start of a non-chat model call."""
        self._thinking_started(run_id, input_count=len(prompts), serialized=serialized)

    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        **_kwargs: Any,
    ) -> None:
        """Log the start of a chat-model call without recording messages."""
        self._thinking_started(run_id, input_count=sum(map(len, messages)), serialized=serialized)

    async def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        **_kwargs: Any,
    ) -> None:
        """Log successful completion of a model call."""
        self._finish_activity(
            "agent.thinking.completed",
            run_id,
            output_count=len(response.generations),
        )

    async def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **_kwargs: Any,
    ) -> None:
        """Log failure of a model call."""
        self._finish_activity(
            "agent.thinking.failed",
            run_id,
            error_type=type(error).__name__,
        )

    async def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        inputs: dict[str, Any] | None = None,
        **_kwargs: Any,
    ) -> None:
        """Log a tool call with a bounded, redacted input preview."""
        tool_name = _serialized_name(serialized)
        self._tool_names[run_id] = tool_name
        self._start_activity(
            "agent.tool.started",
            run_id,
            tool_name=tool_name,
            input_preview=_activity_preview(inputs if inputs is not None else input_str),
        )

    async def on_tool_end(
        self,
        output: object,
        *,
        run_id: UUID,
        **_kwargs: Any,
    ) -> None:
        """Log successful completion of a tool call."""
        self._finish_tool(
            "agent.tool.completed",
            run_id,
            output_preview=_activity_preview(output),
        )

    async def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **_kwargs: Any,
    ) -> None:
        """Log failure of a tool call."""
        self._finish_tool("agent.tool.failed", run_id, error_type=type(error).__name__)

    def _thinking_started(
        self,
        run_id: UUID,
        *,
        input_count: int,
        serialized: Mapping[str, object],
    ) -> None:
        self._start_activity(
            "agent.thinking.started",
            run_id,
            input_count=input_count,
            model=_serialized_name(serialized),
        )

    def _start_activity(self, event: str, run_id: UUID, **fields: object) -> None:
        self._activity_started_at[run_id] = time.perf_counter()
        log_event(
            self._logger,
            event,
            activity_ref=stable_log_ref(str(run_id)),
            conversation_ref=self._conversation_ref,
            **fields,
        )

    def _finish_activity(self, event: str, run_id: UUID, **fields: object) -> None:
        started_at = self._activity_started_at.pop(run_id, None)
        log_event(
            self._logger,
            event,
            activity_ref=stable_log_ref(str(run_id)),
            conversation_ref=self._conversation_ref,
            duration_ms=_duration_ms(started_at),
            **fields,
        )

    def _finish_tool(self, event: str, run_id: UUID, **fields: object) -> None:
        tool_name = self._tool_names.pop(run_id, "unknown")
        self._finish_activity(event, run_id, tool_name=tool_name, **fields)


def agent_activity_logging_enabled(env: Mapping[str, str]) -> bool:
    """Return whether local agent activity events are enabled."""
    return env.get(AGENT_ACTIVITY_LOGGING_ENV_KEY, "").strip().lower() in TRUTHY_ENV_VALUES


def _duration_ms(started_at: float | None) -> int:
    if started_at is None:
        return 0
    return max(round((time.perf_counter() - started_at) * 1000), 0)


def _serialized_name(serialized: Mapping[str, object]) -> str:
    name = serialized.get("name")
    if isinstance(name, str) and name:
        return name
    identifier = serialized.get("id")
    if isinstance(identifier, Sequence) and not isinstance(identifier, (str, bytes)):
        tail = identifier[-1] if identifier else None
        if isinstance(tail, str) and tail:
            return tail
    return "unknown"


def _activity_preview(value: object) -> str:
    content = value.content if isinstance(value, BaseMessage) else value
    safe_value = _activity_log_value(
        _parse_json_value(content),
        depth=0,
        budget=[_ACTIVITY_PREVIEW_VALUES],
    )
    text = safe_value if isinstance(safe_value, str) else json.dumps(safe_value)
    if len(text) <= AGENT_ACTIVITY_PREVIEW_LIMIT:
        return text
    keep = AGENT_ACTIVITY_PREVIEW_LIMIT - len(_ACTIVITY_TRUNCATION_MARKER)
    return text[:keep] + _ACTIVITY_TRUNCATION_MARKER


def _activity_log_value(value: object, *, depth: int, budget: list[int]) -> object:
    if budget[0] <= 0:
        return _ACTIVITY_TRUNCATION_MARKER
    budget[0] -= 1
    if depth >= _ACTIVITY_PREVIEW_DEPTH:
        result: object = "<nested value>"
    elif isinstance(value, dict):
        result = _activity_log_mapping(
            cast("dict[object, object]", value),
            depth=depth,
            budget=budget,
        )
    elif isinstance(value, (list, tuple)):
        result = _activity_log_sequence(value, depth=depth, budget=budget)
    elif isinstance(value, str):
        result = _bounded_activity_string(value)
    elif isinstance(value, bytes):
        result = f"<{len(value)} bytes>"
    elif value is None or isinstance(value, (bool, int, float)):
        result = value
    else:
        result = f"<{type(value).__name__}>"
    return result


def _activity_log_sequence(
    value: Sequence[object],
    *,
    depth: int,
    budget: list[int],
) -> list[object]:
    result: list[object] = []
    for item in islice(value, _ACTIVITY_PREVIEW_ITEMS):
        if budget[0] <= 0:
            result.append(_ACTIVITY_TRUNCATION_MARKER)
            break
        result.append(_activity_log_value(item, depth=depth + 1, budget=budget))
    return result


def _activity_log_mapping(
    value: dict[object, object],
    *,
    depth: int,
    budget: list[int],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for raw_key, raw_value in islice(value.items(), _ACTIVITY_PREVIEW_ITEMS):
        if budget[0] <= 0:
            result[_ACTIVITY_TRUNCATION_MARKER] = _ACTIVITY_TRUNCATION_MARKER
            break
        key = raw_key if isinstance(raw_key, str) else f"<{type(raw_key).__name__} key>"
        result[key] = (
            REDACTED_LOG_VALUE
            if _is_secret_key(key)
            else _activity_log_value(raw_value, depth=depth + 1, budget=budget)
        )
    return result


def _bounded_activity_string(value: str) -> str:
    if len(value) > AGENT_ACTIVITY_PREVIEW_LIMIT:
        keep = AGENT_ACTIVITY_PREVIEW_LIMIT - len(_ACTIVITY_TRUNCATION_MARKER)
        value = value[:keep] + _ACTIVITY_TRUNCATION_MARKER
    return _redact_string(value)


def _parse_json_value(value: object) -> object:
    if not isinstance(value, str) or len(value) > AGENT_ACTIVITY_PREVIEW_LIMIT * 4:
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value


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
    _emit_event(logger, logging.INFO, event, fields)


def log_debug_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    """Emit one structured JSON event when debug logging is enabled.

    Args:
        logger: Logger used by the emitting subsystem.
        event: Stable event name.
        fields: JSON-serializable event fields.
    """
    if logger.isEnabledFor(logging.DEBUG):
        _emit_event(logger, logging.DEBUG, event, fields)


def _emit_event(
    logger: logging.Logger,
    level: int,
    event: str,
    fields: Mapping[str, object],
) -> None:
    payload = {"event": event, **_redact_mapping(fields)}
    logger.log(level, "talon_event %s", json.dumps(payload, sort_keys=True, default=str))


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
    text = _SECRET_ASSIGNMENT_RE.sub(_redact_equals_assignment, text)
    return _SECRET_COLON_ASSIGNMENT_RE.sub(_redact_colon_assignment, text)


def _redact_equals_assignment(match: re.Match[str]) -> str:
    if not _is_secret_key(match.group("key")):
        return match.group(0)
    return f"{match.group('key')}{match.group('separator')}{REDACTED_LOG_VALUE}"


def _redact_colon_assignment(match: re.Match[str]) -> str:
    if not _is_secret_key(match.group("key")):
        return match.group(0)
    return f'{match.group("prefix")}"{REDACTED_LOG_VALUE}"'


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
