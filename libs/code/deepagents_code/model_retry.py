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
import random
import time
from contextlib import contextmanager
from copy import copy
from typing import TYPE_CHECKING

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
    get_glyphs,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterator

    from langchain.agents.middleware.types import ModelRequest, ModelResponse
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.messages import AIMessage
    from langgraph.pregel.protocol import StreamChunk

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_MODEL_RETRIES",
    "CodeModelRetryMiddleware",
    "build_retry_event",
    "format_retry_status",
]

_INITIAL_DELAY_SECONDS = 0.2
"""First backoff delay, matching Codex's 200ms initial retry wait."""

_BACKOFF_FACTOR = 2.0
"""Exponential backoff multiplier (Codex uses 2.0)."""

_MAX_DELAY_SECONDS = 10.0
"""Cap on a single backoff delay so exponential growth stays bounded."""

_JITTER_FRACTION = 0.1
"""Multiplicative jitter of +-10%, matching Codex's 0.9..1.1 range."""

_RETRYABLE_STATUS_CODES = frozenset({408, 409, 429})
"""Non-5xx statuses worth retrying (timeout, provider lock conflict, rate limit)."""

_TRANSIENT_SDK_EXC_NAMES = frozenset(
    {
        "APITimeoutError",
        "APIConnectionError",
        "APIConnectionTimeoutError",
        "ConnectTimeoutError",
        "ConnectionClosedError",
        "EndpointConnectionError",
        "ReadTimeoutError",
    }
)
"""Provider SDK exception class names that signal a transient network fault.

Matched by class name across the MRO so optional provider packages (openai,
anthropic, ...) never have to be imported to classify their errors. These are
distinct from `APIStatusError`, which carries an HTTP status handled separately.
"""

_HTTP_SERVER_ERROR_FLOOR = 500
_HTTP_SERVER_ERROR_CEILING = 600


class _MessageStreamBuffer:
    """Buffer LangGraph message-stream callbacks until a model attempt succeeds."""

    def __init__(self) -> None:
        self._chunks: list[tuple[StreamMessagesHandler, StreamChunk]] = []

    def callbacks_with_buffered_messages(
        self, callbacks: BaseCallbackManager
    ) -> BaseCallbackManager | None:
        """Return a callback-manager copy with message streams redirected here."""
        replacements: dict[int, StreamMessagesHandler] = {}

        def replace(handler: BaseCallbackHandler) -> BaseCallbackHandler:
            if not isinstance(handler, StreamMessagesHandler):
                return handler
            key = id(handler)
            if key not in replacements:
                buffered = type(handler)(
                    lambda chunk, source=handler: self._chunks.append((source, chunk)),
                    handler.subgraphs,
                    parent_ns=handler.parent_ns,
                )
                buffered.seen.update(handler.seen)
                replacements[key] = buffered
            return replacements[key]

        buffered_callbacks = copy(callbacks)
        buffered_callbacks.handlers = [replace(item) for item in callbacks.handlers]
        buffered_callbacks.inheritable_handlers = [
            replace(item) for item in callbacks.inheritable_handlers
        ]
        return buffered_callbacks if replacements else None

    def flush(self) -> None:
        """Publish buffered chunks and preserve LangGraph's message de-duplication."""
        for handler, chunk in self._chunks:
            data = chunk[2]
            if isinstance(data, tuple) and data:
                message_id = getattr(data[0], "id", None)
                if isinstance(message_id, str | int):
                    handler.seen.add(message_id)
            handler.stream(chunk)
        self._chunks.clear()


@contextmanager
def _buffer_message_streams() -> Iterator[_MessageStreamBuffer]:
    """Redirect the current run's LangGraph message callbacks for one attempt.

    Yields:
        The attempt-local stream buffer to flush after a successful model call.
    """
    buffer = _MessageStreamBuffer()
    try:
        from langgraph.config import get_config

        config = get_config()
    except RuntimeError:
        yield buffer
        return

    callbacks = config.get("callbacks")
    if not isinstance(callbacks, BaseCallbackManager):
        yield buffer
        return
    buffered_callbacks = buffer.callbacks_with_buffered_messages(callbacks)
    if buffered_callbacks is None:
        yield buffer
        return

    buffered_config = config.copy()
    buffered_config["callbacks"] = buffered_callbacks
    token = var_child_runnable_config.set(buffered_config)
    try:
        yield buffer
    finally:
        var_child_runnable_config.reset(token)


def _extract_status_code(exc: Exception) -> int | None:
    """Return an HTTP status code carried by a provider error, if any.

    Inspects the common attributes used across SDKs (`status_code`,
    `response.status_code`, `http_status`) plus botocore's
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
        pass
    else:
        # Covers ReadError, ConnectError, RemoteProtocolError, and every other
        # TransportError, plus connect/read/write/pool timeouts.
        httpx_transient = (httpx.TimeoutException, httpx.TransportError)

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

    Args:
        attempt: The 1-indexed retry number about to be attempted.
        max_retries: The configured maximum retry count.

    Returns:
        A short status line, e.g. ``"model connection dropped, retrying 1/5..."``.
    """
    return (
        f"model connection dropped, retrying {attempt}/{max_retries}"
        f"{get_glyphs().ellipsis}"
    )


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
    if max_retries and _is_retryable_model_error(exc):
        logger.error(
            "Model call failed after %d attempts (retry budget %d exhausted): %s",
            attempts,
            max_retries,
            type(exc).__name__,
            exc_info=exc,
        )
    else:
        logger.debug(
            "Model call failed with a non-transient %s; not retrying",
            type(exc).__name__,
            exc_info=exc,
        )


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

    def _compute_delay(self, attempt: int) -> float:
        """Return the backoff delay before the retry following `attempt`.

        Args:
            attempt: The 0-indexed attempt that just failed.

        Returns:
            Delay in seconds, capped at `_MAX_DELAY_SECONDS`, with +-10% jitter.
        """
        delay = self.initial_delay * (self.backoff_factor**attempt)
        delay = min(delay, self.max_delay)
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
        """
        event = build_retry_event(attempt, max_retries)
        logger.warning("Model call failed; %s", event["message"])
        writer = getattr(getattr(request, "runtime", None), "stream_writer", None)
        if writer is None:
            return
        try:
            writer(event)
        except Exception:
            # A UI status must never break the retry/stream loop.
            logger.debug("Failed to emit model_retry stream event", exc_info=True)

    def _request_max_retries(self, request: ModelRequest) -> int:
        """Resolve the retry budget attached to this request's model.

        Args:
            request: Model request after runtime model selection.

        Returns:
            The model-specific non-negative retry count, or the middleware's
            startup fallback when the model carries no valid metadata.
        """
        raw_retries = getattr(getattr(request, "model", None), MODEL_RETRIES_ATTR, None)
        if (
            isinstance(raw_retries, int)
            and not isinstance(raw_retries, bool)
            and raw_retries >= 0
        ):
            return raw_retries
        return self.max_retries

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
            try:
                with _buffer_message_streams() as stream_buffer:
                    response = handler(request)
            except GraphBubbleUp:
                raise
            except Exception as exc:  # noqa: BLE001  # classified by _is_retryable_model_error
                if not _is_retryable_model_error(exc) or attempt >= max_retries:
                    _log_give_up(exc, attempt + 1, max_retries)
                    return self._handle_failure(exc, attempt + 1)
                self._emit_retry_status(request, attempt + 1, max_retries)
                delay = self._compute_delay(attempt)
                if delay > 0:
                    time.sleep(delay)
            else:
                stream_buffer.flush()
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
            try:
                with _buffer_message_streams() as stream_buffer:
                    response = await handler(request)
            except GraphBubbleUp:
                raise
            except Exception as exc:  # noqa: BLE001  # classified by _is_retryable_model_error
                if not _is_retryable_model_error(exc) or attempt >= max_retries:
                    _log_give_up(exc, attempt + 1, max_retries)
                    return self._handle_failure(exc, attempt + 1)
                self._emit_retry_status(request, attempt + 1, max_retries)
                delay = self._compute_delay(attempt)
                if delay > 0:
                    await asyncio.sleep(delay)
            else:
                stream_buffer.flush()
                return response
        msg = "Unexpected: retry loop completed without returning"
        raise RuntimeError(msg)
