"""Model-node retry middleware for the coding agent.

Wraps only the agent model node (not the whole agent turn) so transient model
connection failures are retried without re-running completed tool calls. Retry
counts are attached to constructed models upstream so runtime model switches
carry their provider-specific budget into each request. This module owns the
retry policy: which errors are transient, the backoff curve, and the user-facing
status surfaced while retrying.
"""

from __future__ import annotations

import logging
import math
import random
import time
from contextlib import contextmanager
from copy import copy
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.callbacks import BaseCallbackManager
from langchain_core.exceptions import ModelError
from langchain_core.runnables.config import var_child_runnable_config
from langgraph.errors import GraphBubbleUp
from langgraph.pregel._messages import (  # noqa: PLC2701  # not publicly re-exported
    StreamMessagesHandler,
)

from deepagents_code.config import (
    DEFAULT_MODEL_RETRIES,
    MODEL_RETRIES_ATTR,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterator, Mapping

    from langchain.agents.middleware.types import ModelRequest, ModelResponse
    from langchain_core.callbacks import BaseCallbackHandler
    from langgraph.pregel.protocol import StreamChunk

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_MODEL_RETRIES",
    "CodeModelRetryMiddleware",
    "aretry_model_call",
    "build_retry_event",
    "format_retry_status",
    "retry_model_call",
    "retry_status_from_event",
]

# Curve parameters mirror Codex's retry defaults.
_INITIAL_DELAY_SECONDS = 0.2
_BACKOFF_FACTOR = 2.0
_MAX_DELAY_SECONDS = 10.0
_MAX_RETRY_AFTER_SECONDS = 60.0
_JITTER_FRACTION = 0.1
_RETRYABLE_STATUS_CODES = frozenset({408, 409, 429})
_TRANSIENT_SDK_EXC_NAMES = frozenset(
    {
        "APITimeoutError",
        "APIConnectionError",
        "APIConnectionTimeoutError",
        "Aborted",
        "ConnectTimeoutError",
        "ConnectionClosedError",
        "DeadlineExceeded",
        "EndpointConnectionError",
        "ReadTimeoutError",
        "ResourceExhausted",
        "ServiceUnavailable",
    }
)

_HTTP_SERVER_ERROR_FLOOR = 500
_HTTP_SERVER_ERROR_CEILING = 600
_RETRY_STATUS_FALLBACK = "Retrying model request"


def _google_api_core_status_code(exc: Exception) -> int | None:
    """Return a numeric Google API Core status without importing its package."""
    if not any(
        base.__module__ == "google.api_core.exceptions" for base in type(exc).__mro__
    ):
        return None
    code = getattr(exc, "code", None)
    return code if isinstance(code, int) and not isinstance(code, bool) else None


class _MessageStreamTracker:
    """Track whether a model attempt emitted visible output."""

    def __init__(self) -> None:
        self.has_streamed = False
        self._tracked: list[tuple[StreamMessagesHandler, StreamMessagesHandler]] = []

    def callbacks_with_tracked_messages(
        self, callbacks: BaseCallbackManager
    ) -> BaseCallbackManager | None:
        replacements: dict[int, StreamMessagesHandler] = {}

        def forward(source: StreamMessagesHandler, chunk: StreamChunk) -> None:
            source.stream(chunk)
            self.has_streamed = True

        def replace(handler: BaseCallbackHandler) -> BaseCallbackHandler:
            if not isinstance(handler, StreamMessagesHandler):
                return handler
            key = id(handler)
            if key not in replacements:
                tracked = type(handler)(
                    lambda chunk, source=handler: forward(source, chunk),
                    handler.subgraphs,
                    parent_ns=handler.parent_ns,
                )
                tracked.seen.update(handler.seen)
                replacements[key] = tracked
                self._tracked.append((handler, tracked))
            return replacements[key]

        tracked_callbacks = copy(callbacks)
        tracked_callbacks.handlers = [replace(item) for item in callbacks.handlers]
        tracked_callbacks.inheritable_handlers = [
            replace(item) for item in callbacks.inheritable_handlers
        ]
        return tracked_callbacks if replacements else None

    def merge_seen(self) -> None:
        """Merge tracked de-duplication IDs into the original handlers."""
        for source, tracked in self._tracked:
            source.seen.update(tracked.seen)


@contextmanager
def _track_message_streams(
    tracker: _MessageStreamTracker,
) -> Iterator[_MessageStreamTracker]:
    try:
        from langgraph.config import get_config

        config = get_config()
    except RuntimeError:
        yield tracker
        return

    callbacks = config.get("callbacks")
    if not isinstance(callbacks, BaseCallbackManager):
        yield tracker
        return
    tracked_callbacks = tracker.callbacks_with_tracked_messages(callbacks)
    if tracked_callbacks is None:
        yield tracker
        return

    tracked_config = config.copy()
    tracked_config["callbacks"] = tracked_callbacks
    token = var_child_runnable_config.set(tracked_config)
    try:
        yield tracker
    finally:
        var_child_runnable_config.reset(token)
        tracker.merge_seen()


def _extract_status_code(exc: Exception) -> int | None:
    """Return an HTTP status carried by a provider error, if any."""
    status = getattr(exc, "status_code", None)
    if isinstance(status, bool):
        return None
    if isinstance(status, int):
        return status

    google_status = _google_api_core_status_code(exc)
    if google_status is not None:
        return google_status

    response = getattr(exc, "response", None)
    if response is not None:
        response_status = getattr(response, "status_code", None)
        if isinstance(response_status, int) and not isinstance(response_status, bool):
            return response_status
        if isinstance(response, dict):
            metadata = response.get("ResponseMetadata")
            if isinstance(metadata, dict):
                response_status = metadata.get("HTTPStatusCode")
                if isinstance(response_status, int) and not isinstance(
                    response_status, bool
                ):
                    return response_status

    http_status = getattr(exc, "http_status", None)
    if isinstance(http_status, int) and not isinstance(http_status, bool):
        return http_status

    return None


def _retry_after_seconds(exc: Exception) -> float | None:
    """Return a capped `Retry-After` response delay, if present."""
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers is None:
        return None
    try:
        raw = headers.get("retry-after")
    except (AttributeError, TypeError):
        return None
    if not isinstance(raw, str) or not raw.strip():
        return None

    raw = raw.strip()
    try:
        seconds = float(raw)
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(raw)
        except (TypeError, ValueError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=UTC)
        seconds = (retry_at - datetime.now(UTC)).total_seconds()

    if not math.isfinite(seconds):
        return None
    return min(max(seconds, 0.0), _MAX_RETRY_AFTER_SECONDS)


def _backoff_delay(
    attempt: int,
    *,
    initial: float,
    factor: float,
    max_delay: float,
    jitter: bool,
) -> float:
    """Return a capped exponential delay, with optional post-cap jitter."""
    delay = min(initial * (factor**attempt), max_delay)
    if jitter and delay > 0:
        jitter_amount = delay * _JITTER_FRACTION
        delay = max(0.0, delay + random.uniform(-jitter_amount, jitter_amount))  # noqa: S311  # backoff jitter, not security-sensitive
    return delay


def _compute_backoff_delay(attempt: int) -> float:
    """Return the configured backoff after a zero-indexed attempt."""
    return _backoff_delay(
        attempt,
        initial=_INITIAL_DELAY_SECONDS,
        factor=_BACKOFF_FACTOR,
        max_delay=_MAX_DELAY_SECONDS,
        jitter=True,
    )


def _retry_delay_seconds(attempt: int, exc: Exception) -> float:
    """Return a provider-directed or local backoff delay for one failure."""
    retry_after = _retry_after_seconds(exc)
    return retry_after if retry_after is not None else _compute_backoff_delay(attempt)


def _model_max_retries(model: object, fallback: int) -> int:
    """Return valid retry metadata attached to `model`, or `fallback`."""
    raw_retries = getattr(model, MODEL_RETRIES_ATTR, None)
    if (
        isinstance(raw_retries, int)
        and not isinstance(raw_retries, bool)
        and raw_retries >= 0
    ):
        return raw_retries
    return fallback


def _is_transient_sdk_error(exc: Exception) -> bool:
    return any(base.__name__ in _TRANSIENT_SDK_EXC_NAMES for base in type(exc).__mro__)


def _is_retryable_model_error(exc: Exception) -> bool:
    """Return whether a model error is transient and safe to retry."""
    if isinstance(exc, ModelError):
        return exc.is_retryable

    # Optional dependency: httpx ships with the HTTP-based providers but keep the
    # import lazy so classification never forces it at startup.
    httpx_transient: tuple[type[BaseException], ...] = ()
    try:
        import httpx
    except ImportError:
        # Raised for a genuinely absent httpx and for a broken sub-import
        # (h11, certifi). The latter silently disables the classification this
        # module exists for, so leave a trace.
        logger.debug(
            "httpx unavailable; its transport errors will not be classified "
            "as retryable",
            exc_info=True,
        )
    else:
        # Deliberately narrower than `TransportError`, whose subclasses include
        # permanent faults: `UnsupportedProtocol` (a mistyped base_url scheme),
        # `LocalProtocolError` (a malformed request), and `ProxyError` (a
        # misconfigured proxy). Retrying those burns the whole budget on an
        # error that was knowable on the first attempt.
        httpx_transient = (
            httpx.TimeoutException,
            httpx.NetworkError,
            httpx.RemoteProtocolError,
        )

    if isinstance(exc, httpx_transient):
        return True

    # A status-bearing provider error is decided solely by its code: retry only
    # 408/409/429/5xx, and never fall through to broader heuristics for a 4xx
    # that would otherwise be misclassified as a bare connection error.
    status = _extract_status_code(exc)
    if status is not None:
        return status in _RETRYABLE_STATUS_CODES or (
            _HTTP_SERVER_ERROR_FLOOR <= status < _HTTP_SERVER_ERROR_CEILING
        )

    if _is_transient_sdk_error(exc):
        return True

    # Stdlib transport faults raised directly (rare, but cheap to cover).
    return isinstance(exc, (TimeoutError, ConnectionError))


def format_retry_status(attempt: int, max_retries: int) -> str:
    """Return the concise user-facing status shown during a retry backoff.

    Carries no trailing ellipsis: the TUI spinner appends its own. Names no
    cause either, because a retry may be a rate limit or a 5xx rather than a
    dropped connection.

    Args:
        attempt: The 1-indexed retry number about to be attempted.
        max_retries: The configured maximum retry count.

    Returns:
        A short status line, e.g. `"Retrying model request 1/5"`.
    """
    return f"Retrying model request {attempt}/{max_retries}"


def _log_give_up(exc: Exception, attempts: int, max_retries: int) -> None:
    """Log why the retry loop stopped before re-raising."""
    if not _is_retryable_model_error(exc):
        logger.debug(
            "Model call failed with a non-transient %s; not retrying",
            type(exc).__name__,
            exc_info=exc,
        )
    elif max_retries:
        logger.error(
            "Model call failed after %d attempts (retry budget %d exhausted): %s",
            attempts,
            max_retries,
            type(exc).__name__,
            exc_info=exc,
        )
    else:
        logger.warning(
            "Model call failed with a transient %s but retries are disabled "
            "(retry budget 0)",
            type(exc).__name__,
            exc_info=exc,
        )


def _retry_call[ResultT](
    call: Callable[[], ResultT],
    *,
    max_retries: int,
    on_retry: Callable[[int, int], None],
    retry_guard: Callable[[Exception, int], bool] | None = None,
    delay_for: Callable[[int, Exception], float] | None = None,
) -> ResultT:
    """Run one synchronous call under the shared retry policy.

    Returns:
        The successful call result.

    Raises:
        GraphBubbleUp: If the graph signals control flow.
        RuntimeError: If the retry loop exits unexpectedly.
    """
    for attempt in range(max_retries + 1):
        try:
            return call()
        except GraphBubbleUp:
            raise
        except Exception as exc:  # classified by _is_retryable_model_error
            if retry_guard is not None and not retry_guard(exc, attempt + 1):
                raise
            if not _is_retryable_model_error(exc) or attempt >= max_retries:
                _log_give_up(exc, attempt + 1, max_retries)
                raise
            on_retry(attempt + 1, max_retries)
            delay = (delay_for or _retry_delay_seconds)(attempt, exc)
            if delay:
                time.sleep(delay)
    msg = "Unexpected: retry loop completed without returning"
    raise RuntimeError(msg)


async def _aretry_call[ResultT](
    call: Callable[[], Awaitable[ResultT]],
    *,
    max_retries: int,
    on_retry: Callable[[int, int], None],
    retry_guard: Callable[[Exception, int], bool] | None = None,
    delay_for: Callable[[int, Exception], float] | None = None,
) -> ResultT:
    """Run one asynchronous call under the shared retry policy.

    Returns:
        The successful call result.

    Raises:
        GraphBubbleUp: If the graph signals control flow.
        RuntimeError: If the retry loop exits unexpectedly.
    """
    import asyncio

    for attempt in range(max_retries + 1):
        try:
            return await call()
        except GraphBubbleUp:
            raise
        except Exception as exc:  # classified by _is_retryable_model_error
            if retry_guard is not None and not retry_guard(exc, attempt + 1):
                raise
            if not _is_retryable_model_error(exc) or attempt >= max_retries:
                _log_give_up(exc, attempt + 1, max_retries)
                raise
            on_retry(attempt + 1, max_retries)
            delay = (delay_for or _retry_delay_seconds)(attempt, exc)
            if delay:
                await asyncio.sleep(delay)
    msg = "Unexpected: retry loop completed without returning"
    raise RuntimeError(msg)


def _log_auxiliary_retry(_attempt: int, _max_retries: int) -> None:
    """Log one auxiliary-model retry."""
    logger.warning("Auxiliary model call failed; retrying")


def _allow_retry_after_stream(
    tracker: _MessageStreamTracker, exc: Exception, attempt: int
) -> bool:
    if not tracker.has_streamed:
        return True
    logger.warning(
        "Model stream failed after output began; not retrying attempt %d",
        attempt,
        exc_info=exc,
    )
    return False


def retry_model_call[ResultT](model: object, call: Callable[[], ResultT]) -> ResultT:
    """Run a non-streaming auxiliary model call with its configured retry budget.

    Args:
        model: Model carrying dcode retry metadata when dcode owns its SDK retries.
        call: Fresh invocation callable to run for each attempt.

    Returns:
        The successful call result.
    """
    return _retry_call(
        call,
        max_retries=_model_max_retries(model, 0),
        on_retry=_log_auxiliary_retry,
    )


async def aretry_model_call[ResultT](
    model: object, call: Callable[[], Awaitable[ResultT]]
) -> ResultT:
    """Run an asynchronous auxiliary model call with its configured retry budget.

    Args:
        model: Model carrying dcode retry metadata when dcode owns its SDK retries.
        call: Fresh async invocation callable to run for each attempt.

    Returns:
        The successful call result.
    """
    return await _aretry_call(
        call,
        max_retries=_model_max_retries(model, 0),
        on_retry=_log_auxiliary_retry,
    )


def retry_status_from_event(event: Mapping[Any, object]) -> str:
    """Return retry status text for an untrusted `model_retry` payload.

    Both the TUI and the headless client render this event, so the validation
    lives with the producer rather than being written twice with different
    strictness.

    Args:
        event: Custom-stream payload, not trusted to hold sane numbers.

    Returns:
        The validated status line, or a cause-free fallback for malformed data.
    """
    attempt = event.get("attempt")
    max_retries = event.get("max_retries")
    if (
        isinstance(attempt, int)
        and not isinstance(attempt, bool)
        and isinstance(max_retries, int)
        and not isinstance(max_retries, bool)
        and 1 <= attempt <= max_retries
    ):
        return format_retry_status(attempt, max_retries)
    logger.warning("Ignoring malformed model_retry payload: %r", dict(event))
    return _RETRY_STATUS_FALLBACK


def build_retry_event(attempt: int, max_retries: int) -> dict[str, object]:
    """Build the custom-stream payload announcing a model retry.

    Args:
        attempt: The 1-indexed retry number about to be attempted.
        max_retries: The configured maximum retry count.

    Returns:
        A stream-writer payload consumed by the client renderers.
    """
    return {
        "type": "model_retry",
        "attempt": attempt,
        "max_retries": max_retries,
        "message": format_retry_status(attempt, max_retries),
    }


class CodeModelRetryMiddleware(AgentMiddleware):
    """Retry transient model-node failures without replaying completed tools."""

    def __init__(self, *, max_retries: int = DEFAULT_MODEL_RETRIES) -> None:
        """Initialize the middleware with the resolved retry count.

        Args:
            max_retries: Startup fallback for retry attempts after the initial
                call. `0` disables retries unless the request's runtime-selected
                model carries a different provider-specific budget.

        Raises:
            ValueError: If `max_retries` is negative.
        """
        if max_retries < 0:
            msg = "max_retries must be >= 0"
            raise ValueError(msg)
        self.max_retries = max_retries

    def _retry_delay(self, attempt: int, exc: Exception) -> float:
        retry_after = _retry_after_seconds(exc)
        return retry_after if retry_after is not None else self._compute_delay(attempt)

    @staticmethod
    def _compute_delay(attempt: int) -> float:
        """Return the local backoff after a zero-indexed attempt."""
        return _compute_backoff_delay(attempt)

    @staticmethod
    def _emit_retry_status(
        request: ModelRequest, attempt: int, max_retries: int
    ) -> None:
        event = build_retry_event(attempt, max_retries)
        logger.warning("Model call failed; %s", event["message"])
        writer = getattr(getattr(request, "runtime", None), "stream_writer", None)
        if writer is None:
            return
        try:
            writer(event)
        except GraphBubbleUp:
            # LangGraph control flow must not be mistaken for a writer fault.
            raise
        except Exception:
            # This event is the only signal that the coming pause is a retry,
            # so losing it leaves the user watching an unexplained stall.
            logger.warning("Failed to emit model_retry stream event", exc_info=True)

    def _request_max_retries(self, request: ModelRequest) -> int:
        return _model_max_retries(getattr(request, "model", None), self.max_retries)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Retry a synchronous model-node call without replaying visible output.

        Returns:
            The successful model response.
        """
        max_retries = self._request_max_retries(request)
        stream_tracker = _MessageStreamTracker()

        def call() -> ModelResponse:
            nonlocal stream_tracker
            stream_tracker = _MessageStreamTracker()
            with _track_message_streams(stream_tracker):
                return handler(request)

        return _retry_call(
            call,
            max_retries=max_retries,
            on_retry=lambda attempt, budget: self._emit_retry_status(
                request, attempt, budget
            ),
            retry_guard=lambda exc, attempt: _allow_retry_after_stream(
                stream_tracker, exc, attempt
            ),
            delay_for=self._retry_delay,
        )

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Retry an asynchronous model-node call without replaying visible output.

        Returns:
            The successful model response.
        """
        max_retries = self._request_max_retries(request)
        stream_tracker = _MessageStreamTracker()

        async def call() -> ModelResponse:
            nonlocal stream_tracker
            stream_tracker = _MessageStreamTracker()
            with _track_message_streams(stream_tracker):
                return await handler(request)

        return await _aretry_call(
            call,
            max_retries=max_retries,
            on_retry=lambda attempt, budget: self._emit_retry_status(
                request, attempt, budget
            ),
            retry_guard=lambda exc, attempt: _allow_retry_after_stream(
                stream_tracker, exc, attempt
            ),
            delay_for=self._retry_delay,
        )
