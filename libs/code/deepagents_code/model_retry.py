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
from typing import TYPE_CHECKING, Literal, TypedDict, TypeVar

from langchain.agents.middleware import ModelRetryMiddleware

from deepagents_code.config import DEFAULT_MODEL_RETRIES, MODEL_RETRIES_ATTR, get_glyphs

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain.agents.middleware.types import ModelRequest, ModelResponse
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import AIMessage

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

__all__ = [
    "DEFAULT_MODEL_RETRIES",
    "CodeModelRetryMiddleware",
    "ModelRetryEvent",
    "build_retry_event",
    "format_retry_status",
]


class ModelRetryEvent(TypedDict):
    """Custom-stream payload announcing a model retry.

    The `type` discriminant lets the client and TUI renderers dispatch this
    event apart from rubric events sharing the `custom` stream mode.
    """

    type: Literal["model_retry"]
    attempt: int
    max_retries: int
    message: str


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

_NONRETRYABLE_ERROR_CODES = frozenset(
    {
        "insufficient_quota",
        "billing_hard_limit_reached",
        "billing_not_active",
        "account_deactivated",
    }
)
"""Provider error codes that are permanent despite a retryable HTTP status.

OpenAI and compatible providers return HTTP 429 for exhausted billing/quota
(`insufficient_quota`), which no amount of retrying will clear. Classifying
these by their string error code surfaces the actionable failure immediately
instead of after the full retry budget and its backoff sleeps.
"""

_THROTTLING_ERROR_CODES = frozenset(
    {
        "ThrottlingException",
        "Throttling",
        "ThrottledException",
        "TooManyRequestsException",
        "RequestLimitExceeded",
        "RequestThrottled",
        "SlowDown",
    }
)
"""botocore error codes that signal rate limiting behind a non-429 status.

AWS commonly surfaces throttling with an HTTP 400 body, which the status-code
check would otherwise treat as a fatal client error. Matching the error code
lets these retry.
"""


def _provider_error_code(exc: Exception) -> str | None:
    """Return a provider-specific string error code, if the SDK carries one.

    Covers OpenAI-style `code`/`type` string attributes and botocore's
    `response["Error"]["Code"]` mapping. Google's integer `code` (an HTTP
    status) is ignored here; it is handled by `_extract_status_code`.

    Args:
        exc: The exception raised by the model call.

    Returns:
        The string error code, or `None` when the exception carries none.
    """
    for attr in ("code", "type"):
        value = getattr(exc, attr, None)
        if isinstance(value, str) and value:
            return value
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        error = response.get("Error")
        if isinstance(error, dict):
            code = error.get("Code")
            if isinstance(code, str) and code:
                return code
    return None


def _describe_error(exc: Exception) -> str:
    """Return a short, log-safe description of a retried error.

    Combines the exception class name with any HTTP status and provider error
    code so retry logs record *what* failed without dumping a stack trace.

    Args:
        exc: The exception raised by the model call.

    Returns:
        A concise descriptor, e.g. ``"APIStatusError status=429 code=overloaded"``.
    """
    parts = [type(exc).__name__]
    status = _extract_status_code(exc)
    if status is not None:
        parts.append(f"status={status}")
    code = _provider_error_code(exc)
    if code:
        parts.append(f"code={code}")
    return " ".join(parts)


def _extract_status_code(exc: Exception) -> int | None:
    """Return an HTTP status code carried by a provider error, if any.

    Inspects the common attributes used across SDKs (`status_code`, `code`,
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

    # Google API Core exceptions (including Vertex AI's `ResourceExhausted`
    # and `ServiceUnavailable`) expose their HTTP status on `code`. Check it
    # before `response`, which may be absent or may contain a gRPC call object.
    code = getattr(exc, "code", None)
    if isinstance(code, int) and not isinstance(code, bool):
        return code

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

    Retries transient transport/timeout faults and provider status errors that
    indicate an overloaded or momentarily unavailable backend (408, 429, 5xx),
    plus rate-limit errors AWS surfaces behind a non-429 status. Deterministic
    client errors (auth, permission, bad request, validation, context length),
    permanent billing/quota exhaustion (which providers report as 429), and
    dcode model-setup/config errors are never retried, so an exhausted retry
    surfaces the original, actionable failure.

    Args:
        exc: The exception raised by the model call.

    Returns:
        `True` when the error is transient and should be retried.
    """
    # A permanent quota/billing error rides a retryable status (429); its string
    # error code is the only signal that retrying is futile, so check it first.
    error_code = _provider_error_code(exc)
    if error_code in _NONRETRYABLE_ERROR_CODES:
        return False

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

    # AWS throttling arrives as an HTTP 400 body; the error code is the only
    # signal that this 4xx is actually a rate limit worth retrying.
    if error_code in _THROTTLING_ERROR_CODES:
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
        A short status line, e.g. ``"model call failed, retrying 1/5..."``. The
            phrasing is cause-neutral because retryable errors include rate
            limits and server errors, not only dropped connections.
    """
    prefix = f"model call failed, retrying {attempt}/{max_retries}"
    return f"{prefix}{get_glyphs().ellipsis}"


def build_retry_event(attempt: int, max_retries: int) -> ModelRetryEvent:
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
    never replays completed tool calls. The retry loop is reimplemented so an
    exhausted or non-transient error re-raises the original exception (with its
    traceback intact) rather than being wrapped or returned as an error
    `AIMessage`.
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
        writer: Callable[[ModelRetryEvent], object] | None,
        attempt: int,
        max_retries: int,
        exc: Exception,
    ) -> None:
        """Log the retried error and surface a concise status to the client.

        The entire body is guarded: neither logging, event construction, nor the
        stream write may propagate, since an exception here would replace the
        original transient error and abort the remaining retries.

        Args:
            writer: Optional custom-stream writer for the in-flight call.
            attempt: The 1-indexed retry number about to be attempted.
            max_retries: Retry budget resolved for this model request.
            exc: The classified transient error triggering this retry.
        """
        try:
            logger.warning(
                "Model call failed (%s); retrying %d/%d",
                _describe_error(exc),
                attempt,
                max_retries,
            )
            if writer is None:
                return
            writer(build_retry_event(attempt, max_retries))
        except Exception:
            # A UI status must never break the retry/stream loop.
            logger.debug("Failed to emit model_retry status", exc_info=True)

    def _model_max_retries(self, model: BaseChatModel) -> int:
        """Resolve the retry budget attached to a model.

        Args:
            model: Concrete model selected for the call.

        Returns:
            The model-specific non-negative retry count, or the middleware's
            startup fallback when the model carries no valid metadata.
        """
        raw_retries = getattr(model, MODEL_RETRIES_ATTR, None)
        if (
            isinstance(raw_retries, int)
            and not isinstance(raw_retries, bool)
            and raw_retries >= 0
        ):
            return raw_retries
        return self.max_retries

    def run_with_retry(
        self,
        model: BaseChatModel,
        handler: Callable[[], _T],
        *,
        writer: Callable[[ModelRetryEvent], object] | None = None,
    ) -> _T:
        """Run a direct model call with the model's attached retry budget.

        This is used by model calls that do not pass through LangChain's model
        node, such as forced conversation summaries.

        Args:
            model: Concrete model used by `handler`.
            handler: Zero-argument callable that performs the model call.
            writer: Optional custom-stream writer for retry status events.

        Returns:
            The successful handler result.

        Raises:
            RuntimeError: If the retry loop exits without returning (unreachable
                in practice).
        """
        max_retries = self._model_max_retries(model)
        for attempt in range(max_retries + 1):
            try:
                return handler()
            except Exception as exc:  # classified by _is_retryable_model_error
                if not _is_retryable_model_error(exc) or attempt >= max_retries:
                    raise
                self._emit_retry_status(writer, attempt + 1, max_retries, exc)
                delay = self._compute_delay(attempt)
                if delay > 0:
                    time.sleep(delay)
        msg = "Unexpected: retry loop completed without returning"
        raise RuntimeError(msg)

    async def arun_with_retry(
        self,
        model: BaseChatModel,
        handler: Callable[[], Awaitable[_T]],
        *,
        writer: Callable[[ModelRetryEvent], object] | None = None,
    ) -> _T:
        """Asynchronously run a direct model call with attached retry metadata.

        Args:
            model: Concrete model used by `handler`.
            handler: Zero-argument async callable that performs the model call.
            writer: Optional custom-stream writer for retry status events.

        Returns:
            The successful handler result.

        Raises:
            RuntimeError: If the retry loop exits without returning (unreachable
                in practice).
        """
        import asyncio

        max_retries = self._model_max_retries(model)
        for attempt in range(max_retries + 1):
            try:
                return await handler()
            except Exception as exc:  # classified by _is_retryable_model_error
                if not _is_retryable_model_error(exc) or attempt >= max_retries:
                    raise
                self._emit_retry_status(writer, attempt + 1, max_retries, exc)
                delay = self._compute_delay(attempt)
                if delay > 0:
                    await asyncio.sleep(delay)
        msg = "Unexpected: retry loop completed without returning"
        raise RuntimeError(msg)

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
        """
        writer = getattr(getattr(request, "runtime", None), "stream_writer", None)
        return self.run_with_retry(
            request.model,
            lambda: handler(request),
            writer=writer,
        )

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
        """
        writer = getattr(getattr(request, "runtime", None), "stream_writer", None)
        return await self.arun_with_retry(
            request.model,
            lambda: handler(request),
            writer=writer,
        )
