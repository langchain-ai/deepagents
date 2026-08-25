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

from langchain.agents.middleware import ModelRetryMiddleware
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
    from langchain_core.messages import AIMessage
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

_INITIAL_DELAY_SECONDS = 0.2
"""First backoff delay, matching Codex's 200ms initial retry wait."""

_BACKOFF_FACTOR = 2.0
"""Exponential backoff multiplier (Codex uses 2.0)."""

_MAX_DELAY_SECONDS = 10.0
"""Cap on a single backoff delay so exponential growth stays bounded."""

_MAX_RETRY_AFTER_SECONDS = 60.0
"""Cap on a server-requested `Retry-After` wait.

Providers routinely ask for 20-60s on a rate limit, well past the exponential
curve. Honoring the request beats hammering the same quota, but an unbounded
wait would stall the turn indefinitely on a hostile or mistaken header.
"""

_JITTER_FRACTION = 0.1
"""Multiplicative jitter of +-10%, matching Codex's 0.9..1.1 range."""

_RETRYABLE_STATUS_CODES = frozenset({408, 409, 429})
"""Non-5xx statuses worth retrying (timeout, provider lock conflict, rate limit)."""

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
"""Provider SDK exception class names that signal a transient network fault.

Matched by class name across the MRO so optional provider packages (openai,
anthropic, ...) never have to be imported to classify their errors. These are
distinct from `APIStatusError`, which carries an HTTP status handled separately.
"""

_HTTP_SERVER_ERROR_FLOOR = 500
_HTTP_SERVER_ERROR_CEILING = 600


def _google_api_core_status_code(exc: Exception) -> int | None:
    """Return a numeric Google API Core status without importing its package."""
    if not any(
        base.__module__ == "google.api_core.exceptions" for base in type(exc).__mro__
    ):
        return None
    code = getattr(exc, "code", None)
    return code if isinstance(code, int) and not isinstance(code, bool) else None


class _MessageStreamTracker:
    """Forward LangGraph message-stream callbacks and record visible output."""

    def __init__(self) -> None:
        self.has_streamed = False

    def callbacks_with_tracked_messages(
        self, callbacks: BaseCallbackManager
    ) -> BaseCallbackManager | None:
        """Return a callback-manager copy that tracks message-stream delivery."""
        replacements: dict[int, StreamMessagesHandler] = {}

        def forward(source: StreamMessagesHandler, chunk: StreamChunk) -> None:
            data = chunk[2]
            if isinstance(data, tuple) and data:
                message_id = getattr(data[0], "id", None)
                if isinstance(message_id, str | int):
                    source.seen.add(message_id)
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
            return replacements[key]

        tracked_callbacks = copy(callbacks)
        tracked_callbacks.handlers = [replace(item) for item in callbacks.handlers]
        tracked_callbacks.inheritable_handlers = [
            replace(item) for item in callbacks.inheritable_handlers
        ]
        return tracked_callbacks if replacements else None


@contextmanager
def _track_message_streams(
    tracker: _MessageStreamTracker,
) -> Iterator[_MessageStreamTracker]:
    """Track whether one model attempt has emitted visible message output.

    The caller owns `tracker` so the retry loop can read it even when this
    context manager fails before yielding.

    Args:
        tracker: Attempt-local tracker to record message-stream delivery on.

    Yields:
        The tracker passed in by the caller.
    """
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


def _extract_status_code(exc: Exception) -> int | None:
    """Return an HTTP status code carried by a provider error, if any.

    Inspects the common attributes used across SDKs (`status_code`, Google API
    Core's numeric `code`, `response.status_code`, `http_status`) plus botocore's
    `response["ResponseMetadata"]["HTTPStatusCode"]` mapping defensively, so a
    missing or non-integer value simply yields `None`.

    Args:
        exc: The exception raised by the model call.

    Returns:
        The integer status code, or `None` when the exception carries none.
    """
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
    """Return the server-requested retry delay carried by `exc`, if any.

    Reads the `Retry-After` response header, which providers set on 429 and 503
    in either delta-seconds or HTTP-date form. Our exponential curve tops out
    around 3s per step, so ignoring the header means retrying well before the
    quota resets and spending the whole budget for nothing.

    Args:
        exc: The exception raised by the model call.

    Returns:
        A non-negative delay in seconds capped at `_MAX_RETRY_AFTER_SECONDS`,
        or `None` when the exception carries no usable header.
    """
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


def _compute_backoff_delay(attempt: int) -> float:
    """Return the jittered exponential delay after a zero-indexed attempt."""
    delay = min(_INITIAL_DELAY_SECONDS * (_BACKOFF_FACTOR**attempt), _MAX_DELAY_SECONDS)
    if delay > 0:
        jitter_amount = delay * _JITTER_FRACTION
        delay = max(0.0, delay + random.uniform(-jitter_amount, jitter_amount))  # noqa: S311  # backoff jitter, not security-sensitive
    return delay


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
    """Return whether `exc` is a name-matched transient provider SDK error.

    Args:
        exc: The exception raised by the model call.

    Returns:
        `True` when any class in the exception's MRO is a known transient
            provider timeout/connection error.
    """
    return any(base.__name__ in _TRANSIENT_SDK_EXC_NAMES for base in type(exc).__mro__)


def _is_retryable_model_error(exc: Exception) -> bool:
    """Return whether a model-node exception is a transient error worth retrying.

    LangChain's standard `ModelError.is_retryable` classification is authoritative.
    For integrations that do not yet emit standard model errors, falls back to
    transient transport/timeout faults and provider status errors that indicate an
    overloaded or momentarily unavailable backend (408, lock-timeout 409, 429,
    5xx). Deterministic client errors and dcode model-setup/config errors are
    never retried.

    Args:
        exc: The exception raised by the model call.

    Returns:
        `True` when the error is transient and should be retried.
    """
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

    Carries no trailing ellipsis: the TUI spinner appends its own, and the
    previous wording claimed a dropped connection for rate limits and 5xx
    responses, which `_is_retryable_model_error` also retries.

    Args:
        attempt: The 1-indexed retry number about to be attempted.
        max_retries: The configured maximum retry count.

    Returns:
        A short status line, e.g. ``"Retrying model request 1/5"``.
    """
    return f"Retrying model request {attempt}/{max_retries}"


def _log_give_up(exc: Exception, attempts: int, max_retries: int) -> None:
    """Record why the retry loop stopped before re-raising.

    `on_failure="error"` re-raises the original exception, which carries no
    trace of the attempts spent on it. Without this the caller sees a bare
    provider error and cannot tell a first-attempt failure from an exhausted
    budget.

    Args:
        exc: The exception about to be re-raised.
        attempts: Total model calls made, including the initial one.
        max_retries: Retry budget resolved for this model request.
    """
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


def retry_model_call[ResultT](model: object, call: Callable[[], ResultT]) -> ResultT:
    """Run a non-streaming auxiliary model call with its configured retry budget.

    Args:
        model: Model carrying dcode retry metadata when dcode owns its SDK retries.
        call: Fresh invocation callable to run for each attempt.

    Returns:
        The successful call result.

    Raises:
        GraphBubbleUp: Propagates LangGraph control flow immediately.
        RuntimeError: If the retry loop exits without returning (unreachable in
            practice). Exhausted and non-transient errors propagate unchanged.
    """
    max_retries = _model_max_retries(model, 0)
    for attempt in range(max_retries + 1):
        try:
            return call()
        except GraphBubbleUp:
            raise
        except Exception as exc:  # classified by _is_retryable_model_error
            if not _is_retryable_model_error(exc) or attempt >= max_retries:
                _log_give_up(exc, attempt + 1, max_retries)
                raise
            logger.warning("Auxiliary model call failed; retrying")
            if delay := _retry_delay_seconds(attempt, exc):
                time.sleep(delay)
    msg = "Unexpected: auxiliary retry loop completed without returning"
    raise RuntimeError(msg)


async def aretry_model_call[ResultT](
    model: object, call: Callable[[], Awaitable[ResultT]]
) -> ResultT:
    """Async variant of `retry_model_call`.

    Args:
        model: Model carrying dcode retry metadata when dcode owns its SDK retries.
        call: Fresh async invocation callable to run for each attempt.

    Returns:
        The successful call result.

    Raises:
        GraphBubbleUp: Propagates LangGraph control flow immediately.
        RuntimeError: If the retry loop exits without returning (unreachable in
            practice). Exhausted and non-transient errors propagate unchanged.
    """
    import asyncio

    max_retries = _model_max_retries(model, 0)
    for attempt in range(max_retries + 1):
        try:
            return await call()
        except GraphBubbleUp:
            raise
        except Exception as exc:  # classified by _is_retryable_model_error
            if not _is_retryable_model_error(exc) or attempt >= max_retries:
                _log_give_up(exc, attempt + 1, max_retries)
                raise
            logger.warning("Auxiliary model call failed; retrying")
            if delay := _retry_delay_seconds(attempt, exc):
                await asyncio.sleep(delay)
    msg = "Unexpected: auxiliary retry loop completed without returning"
    raise RuntimeError(msg)


def retry_status_from_event(event: Mapping[Any, object]) -> str | None:
    """Return retry status text for an untrusted `model_retry` payload.

    Both the TUI and the headless client render this event, so the validation
    lives with the producer rather than being written twice with different
    strictness.

    Args:
        event: Custom-stream payload, not trusted to hold sane numbers.

    Returns:
        The status line, or `None` when the payload is malformed. A malformed
        payload is a producer bug, so it is logged rather than passed through.
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
    return None


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


class CodeModelRetryMiddleware(ModelRetryMiddleware):
    """Retry the model node on transient errors with Codex-style backoff.

    Subclasses LangChain's `ModelRetryMiddleware` to add a user-facing status
    (`model_retry` custom-stream event) before each backoff sleep, since the base
    class exposes no on-retry hook. Retries wrap only the model node, so a retry
    never replays completed tool calls. `on_failure="error"` is fixed, so an
    exhausted retry re-raises the original exception rather than returning an
    error `AIMessage`.
    """

    def __init__(self, *, max_retries: int = DEFAULT_MODEL_RETRIES) -> None:
        """Initialize the middleware with the resolved retry count.

        Args:
            max_retries: Startup fallback for retry attempts after the initial
                call. `0` disables retries unless the request's runtime-selected
                model carries a different provider-specific budget.
        """
        super().__init__(
            max_retries=max_retries,
            retry_on=_is_retryable_model_error,
            on_failure="error",
            backoff_factor=_BACKOFF_FACTOR,
            initial_delay=_INITIAL_DELAY_SECONDS,
            max_delay=_MAX_DELAY_SECONDS,
            jitter=True,
        )

    def _retry_delay(self, attempt: int, exc: Exception) -> float:
        """Return the wait before the retry following `attempt`.

        A server-requested `Retry-After` wins over the local curve: the
        provider knows when its quota resets, and retrying sooner just spends
        the budget against a closed door.

        Args:
            attempt: The 0-indexed attempt that just failed.
            exc: The exception that attempt raised.

        Returns:
            Delay in seconds.
        """
        retry_after = _retry_after_seconds(exc)
        return retry_after if retry_after is not None else self._compute_delay(attempt)

    def _compute_delay(self, attempt: int) -> float:
        """Return the backoff delay before the retry following `attempt`.

        Args:
            attempt: The 0-indexed attempt that just failed.

        Returns:
            Delay in seconds, capped at `_MAX_DELAY_SECONDS`, with +-10% jitter.
        """
        delay = min(self.initial_delay * (self.backoff_factor**attempt), self.max_delay)
        if self.jitter and delay > 0:
            jitter_amount = delay * _JITTER_FRACTION
            delay = max(0.0, delay + random.uniform(-jitter_amount, jitter_amount))  # noqa: S311  # backoff jitter, not security-sensitive
        return delay

    @staticmethod
    def _emit_retry_status(
        request: ModelRequest, attempt: int, max_retries: int
    ) -> None:
        """Surface a concise retry status without leaking a stack trace.

        Args:
            request: The in-flight model request (carries the runtime writer).
            attempt: The 1-indexed retry number about to be attempted.
            max_retries: Retry budget resolved for this model request.

        Raises:
            GraphBubbleUp: Propagates LangGraph control flow raised through the
                writer rather than treating it as a writer fault.
        """
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
        """Resolve the retry budget attached to this request's model.

        Args:
            request: Model request after runtime model selection.

        Returns:
            The model-specific non-negative retry count, or the middleware's
            startup fallback when the model carries no valid metadata.
        """
        return _model_max_retries(getattr(request, "model", None), self.max_retries)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse | AIMessage:
        """Retry the model node on transient errors, surfacing status per retry.

        Args:
            request: Model request to execute (may be re-run on retry).
            handler: Callable that executes the model node.

        Returns:
            The successful `ModelResponse`.

        Raises:
            GraphBubbleUp: Propagates LangGraph control-flow exceptions immediately.
            RuntimeError: If the retry loop exits without returning (unreachable
                in practice). Exhausted or non-transient errors are re-raised by
                the inherited `on_failure="error"` handling.
        """
        max_retries = self._request_max_retries(request)
        for attempt in range(max_retries + 1):
            stream_tracker = _MessageStreamTracker()
            try:
                with _track_message_streams(stream_tracker):
                    response = handler(request)
            except GraphBubbleUp:
                raise
            except Exception as exc:  # classified by _is_retryable_model_error
                if stream_tracker.has_streamed:
                    logger.warning(
                        "Model stream failed after output began; "
                        "not retrying attempt %d",
                        attempt + 1,
                        exc_info=exc,
                    )
                    raise
                if not _is_retryable_model_error(exc) or attempt >= max_retries:
                    _log_give_up(exc, attempt + 1, max_retries)
                    return self._handle_failure(exc, attempt + 1)
                self._emit_retry_status(request, attempt + 1, max_retries)
                delay = self._retry_delay(attempt, exc)
                if delay > 0:
                    time.sleep(delay)
            else:
                return response
        msg = "Unexpected: retry loop completed without returning"
        raise RuntimeError(msg)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse | AIMessage:
        """Async variant of `wrap_model_call`.

        Args:
            request: Model request to execute (may be re-run on retry).
            handler: Async callable that executes the model node.

        Returns:
            The successful `ModelResponse`.

        Raises:
            GraphBubbleUp: Propagates LangGraph control-flow exceptions immediately.
            RuntimeError: If the retry loop exits without returning (unreachable
                in practice). Exhausted or non-transient errors are re-raised by
                the inherited `on_failure="error"` handling.
        """
        import asyncio

        max_retries = self._request_max_retries(request)
        for attempt in range(max_retries + 1):
            stream_tracker = _MessageStreamTracker()
            try:
                with _track_message_streams(stream_tracker):
                    response = await handler(request)
            except GraphBubbleUp:
                raise
            except Exception as exc:  # classified by _is_retryable_model_error
                if stream_tracker.has_streamed:
                    logger.warning(
                        "Model stream failed after output began; "
                        "not retrying attempt %d",
                        attempt + 1,
                        exc_info=exc,
                    )
                    raise
                if not _is_retryable_model_error(exc) or attempt >= max_retries:
                    _log_give_up(exc, attempt + 1, max_retries)
                    return self._handle_failure(exc, attempt + 1)
                self._emit_retry_status(request, attempt + 1, max_retries)
                delay = self._retry_delay(attempt, exc)
                if delay > 0:
                    await asyncio.sleep(delay)
            else:
                return response
        msg = "Unexpected: retry loop completed without returning"
        raise RuntimeError(msg)
