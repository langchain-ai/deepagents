"""Model-node retry middleware for the coding agent.

Wraps only the agent model node (not the whole agent turn) so transient model
connection failures are retried without re-running completed tool calls. Retry
count is resolved from config/CLI upstream (see `config.resolve_model_retries`);
this module owns the retry policy: which errors are transient, the backoff
curve, and the user-facing status surfaced while retrying.
"""

from __future__ import annotations

import logging
import random
import time
from typing import TYPE_CHECKING

from langchain.agents.middleware import ModelRetryMiddleware

from deepagents_code.config import DEFAULT_MODEL_RETRIES, get_glyphs

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain.agents.middleware.types import ModelRequest, ModelResponse
    from langchain_core.messages import AIMessage

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

_RETRYABLE_STATUS_CODES = frozenset({408, 429})
"""Non-5xx HTTP status codes worth retrying (request timeout, rate limit)."""

_TRANSIENT_SDK_EXC_NAMES = frozenset(
    {
        "APITimeoutError",
        "APIConnectionError",
        "APIConnectionTimeoutError",
    }
)
"""Provider SDK exception class names that signal a transient network fault.

Matched by class name across the MRO so optional provider packages (openai,
anthropic, ...) never have to be imported to classify their errors. These are
distinct from `APIStatusError`, which carries an HTTP status handled separately.
"""

_HTTP_SERVER_ERROR_FLOOR = 500
_HTTP_SERVER_ERROR_CEILING = 600


def _extract_status_code(exc: Exception) -> int | None:
    """Return an HTTP status code carried by a provider error, if any.

    Inspects the common attributes used across SDKs (`status_code`,
    `response.status_code`, `http_status`) defensively so a missing or
    non-integer attribute simply yields `None`.

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

    Retries transient transport/timeout faults and provider status errors that
    indicate an overloaded or momentarily unavailable backend (408, 429, 5xx).
    Deterministic client errors (auth, permission, bad request, validation,
    context length) and dcode model-setup/config errors are never retried, so an
    exhausted retry surfaces the original, actionable failure.

    Args:
        exc: The exception raised by the model call.

    Returns:
        `True` when the error is transient and should be retried.
    """
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
    # 408/429/5xx, and never fall through to broader heuristics for a 4xx that
    # would otherwise be misclassified as a bare connection error.
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
    prefix = f"model connection dropped, retrying {attempt}/{max_retries}"
    return f"{prefix}{get_glyphs().ellipsis}"


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
            max_retries: Retry attempts after the initial call. ``0`` disables
                retries. Resolved upstream from config/CLI.
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

    def _emit_retry_status(self, request: ModelRequest, attempt: int) -> None:
        """Surface a concise retry status without leaking a stack trace.

        Args:
            request: The in-flight model request (carries the runtime writer).
            attempt: The 1-indexed retry number about to be attempted.
        """
        event = build_retry_event(attempt, self.max_retries)
        logger.warning("Model call failed; %s", event["message"])
        writer = getattr(getattr(request, "runtime", None), "stream_writer", None)
        if writer is None:
            return
        try:
            writer(event)
        except Exception:
            # A UI status must never break the retry/stream loop.
            logger.debug("Failed to emit model_retry stream event", exc_info=True)

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
            RuntimeError: If the retry loop exits without returning (unreachable
                in practice). Exhausted or non-transient errors are re-raised by
                the inherited `on_failure="error"` handling.
        """
        for attempt in range(self.max_retries + 1):
            try:
                return handler(request)
            except Exception as exc:  # noqa: BLE001  # classified by _is_retryable_model_error
                if not _is_retryable_model_error(exc) or attempt >= self.max_retries:
                    return self._handle_failure(exc, attempt + 1)
                self._emit_retry_status(request, attempt + 1)
                delay = self._compute_delay(attempt)
                if delay > 0:
                    time.sleep(delay)
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
            RuntimeError: If the retry loop exits without returning (unreachable
                in practice). Exhausted or non-transient errors are re-raised by
                the inherited `on_failure="error"` handling.
        """
        import asyncio

        for attempt in range(self.max_retries + 1):
            try:
                return await handler(request)
            except Exception as exc:  # noqa: BLE001  # classified by _is_retryable_model_error
                if not _is_retryable_model_error(exc) or attempt >= self.max_retries:
                    return self._handle_failure(exc, attempt + 1)
                self._emit_retry_status(request, attempt + 1)
                delay = self._compute_delay(attempt)
                if delay > 0:
                    await asyncio.sleep(delay)
        msg = "Unexpected: retry loop completed without returning"
        raise RuntimeError(msg)
