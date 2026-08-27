"""Model-node retry middleware for the coding agent.

Wraps only the agent model node (not the whole agent turn) so transient model
connection failures are retried without re-running completed tool calls. Retry
counts are attached to constructed models upstream so runtime model switches
carry their provider-specific budget into each request. This module owns the
retry policy: which errors are transient, the backoff curve, and the user-facing
status surfaced while retrying.

Why not LangChain's `ModelRetryMiddleware`: it reads its retry count once at
construction, so it can't honor the provider-specific budget we stamp on each
model for runtime switches. It sleeps between attempts without saying
anything, which in a streaming terminal just looks frozen. When the budget
runs out it hands back an `AIMessage` containing the error text, so a dead
provider ends the turn disguised as a model answer. Its retry check only
inspects the raised exception, missing transport faults wrapped in exception
groups, and it jitters at 25% while ignoring `Retry-After` headers. None of
that is a knob; fixing any of it means overriding the whole loop, so we own
the loop here.
"""

from __future__ import annotations

import logging
import math
import random
import time
import uuid
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
    "INTERRUPTED_TOOL_OUTPUT",
    "CodeModelRetryMiddleware",
    "aretry_model_call",
    "build_attempt_event",
    "build_retry_event",
    "format_retry_status",
    "legacy_retry_index",
    "model_attempt_from_event",
    "model_retry_from_event",
    "retry_model_call",
    "retry_status_from_event",
]

# Tuned for interactive use: quick first retry, tight cap, modest jitter.
_INITIAL_DELAY_SECONDS = 0.2
_BACKOFF_FACTOR = 2.0
_MAX_DELAY_SECONDS = 10.0
_MAX_RETRY_AFTER_SECONDS = 60.0
_JITTER_FRACTION = 0.1
_RETRYABLE_STATUS_CODES = frozenset({408, 409, 429})
# Provider-SDK error classes that name a transient failure, keyed by the root
# package that owns the name. The package is part of the key on purpose: these
# are generic words, and matching a bare class name would classify any
# dependency's identically-named error as transient -- the same rigor the
# httpcore/aiohttp checks in `_is_http_transport_error` already apply.
_TRANSIENT_SDK_EXC_NAMES = frozenset(
    {
        ("anthropic", "APIConnectionError"),
        ("anthropic", "APIConnectionTimeoutError"),
        ("anthropic", "APITimeoutError"),
        ("botocore", "ConnectionClosedError"),
        ("botocore", "ConnectTimeoutError"),
        ("botocore", "EndpointConnectionError"),
        ("botocore", "ReadTimeoutError"),
        # `google.api_core` statuses are read by `_google_api_core_status_code`
        # first; these cover the subclasses raised without a numeric code.
        ("google", "Aborted"),
        ("google", "DeadlineExceeded"),
        ("google", "ResourceExhausted"),
        ("google", "ServiceUnavailable"),
        ("openai", "APIConnectionError"),
        ("openai", "APITimeoutError"),
        ("urllib3", "ConnectTimeoutError"),
        ("urllib3", "ReadTimeoutError"),
        ("websockets", "ConnectionClosedError"),
    }
)

_HTTP_SERVER_ERROR_FLOOR = 500
_HTTP_SERVER_ERROR_CEILING = 600
_RETRY_STATUS_FALLBACK = "Retrying model request"
# Total sleep the interactive model node may spend across one call's retries.
# Per-delay caps bound nothing (see `_delay_budget_guard`): five honoured
# `Retry-After` hints of `_MAX_RETRY_AFTER_SECONDS` each would stall a turn for
# five minutes behind a spinner. One full honoured hint still fits.
_MAX_INTERACTIVE_TOTAL_DELAY_SECONDS = 60.0
# What the product says when an attempt is superseded. Every surface renders
# some part of this set, so the wording lives with the event builders rather
# than being spelled once per client.
INTERRUPTED_TOOL_OUTPUT = "Model response interrupted before tool execution"
"""Synthetic tool output for a call superseded before the tool ran."""
RETRY_BOUNDARY_LINE = (
    "--- connection dropped; the output above is incomplete — retrying ---"
)
"""Rule printed between a failed attempt's partial output and its replay."""
RETRY_MARKER_FALLBACK = (
    "Connection dropped; the partial response above is incomplete. Retrying."
)
"""Retry marker for a payload whose attempt counts are unusable."""
TERMINAL_ATTEMPT_MARKER = (
    "The model request failed; the partial response above is incomplete."
)
"""Marker for partial output left behind by an exhausted retry budget."""
_ATTEMPT_PHASES = frozenset({"start", "complete"})
_CALL_ID_MAX_LENGTH = 64
_CALL_ID_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
)


def _google_api_core_status_code(exc: Exception) -> int | None:
    """Return a numeric Google API Core status without importing its package."""
    if not any(
        base.__module__ == "google.api_core.exceptions" for base in type(exc).__mro__
    ):
        return None
    code = getattr(exc, "code", None)
    return code if isinstance(code, int) and not isinstance(code, bool) else None


class _MessageStreamTracker:
    """Track whether a model attempt emitted output to the message stream."""

    def __init__(self) -> None:
        self.has_streamed = False
        self._tracked: list[tuple[StreamMessagesHandler, StreamMessagesHandler]] = []

    def callbacks_with_tracked_messages(
        self, callbacks: BaseCallbackManager
    ) -> BaseCallbackManager | None:
        replacements: dict[int, StreamMessagesHandler] = {}

        def forward(source: StreamMessagesHandler, chunk: StreamChunk) -> None:
            # Flag first: a writer that raises part-way through has still put
            # the chunk beyond our control, so the client must be told output
            # may have escaped even though the consumer never saw the chunk.
            self.has_streamed = True
            source.stream(chunk)

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
    # Every early return below leaves `tracker.has_streamed` permanently
    # `False`, which makes `output_may_have_started` permanently `False` and
    # silently disables the supersession marking this module exists to provide:
    # a retried attempt's partial output is then appended with no boundary. Say
    # so, at a level matched to how expected the cause is.
    try:
        from langgraph.config import get_config

        config = get_config()
    except RuntimeError:
        logger.debug(
            "No runnable config in scope; model attempts cannot detect streamed "
            "output, so a retry may append after unmarked partial output",
            exc_info=True,
        )
        yield tracker
        return

    callbacks = config.get("callbacks")
    if not isinstance(callbacks, BaseCallbackManager):
        logger.warning(
            "Runnable config carries %s under 'callbacks' rather than a "
            "BaseCallbackManager; retry supersession cannot be detected",
            type(callbacks).__name__,
        )
        yield tracker
        return
    tracked_callbacks = tracker.callbacks_with_tracked_messages(callbacks)
    if tracked_callbacks is None:
        # Routine when nothing consumes the `messages` stream mode; also what a
        # renamed or restructured `StreamMessagesHandler` would look like.
        logger.debug(
            "No message-stream handler attached; retry supersession cannot be "
            "detected for this model call"
        )
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
        # httpx/requests headers are case-insensitive; a plain dict is not, so
        # fall back to the canonical casing rather than miss the hint.
        raw = headers.get("retry-after")
        if raw is None:
            raw = headers.get("Retry-After")
    except (AttributeError, TypeError):
        logger.debug("Retry-After lookup failed on %s headers", type(exc).__name__)
        return None
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw.strip():
        # Ignoring a provider's pacing hint can escalate a rate limit into a
        # ban, so an unusable value is worth a trace.
        logger.debug("Ignoring unusable Retry-After value %r", raw)
        return None

    raw = raw.strip()
    try:
        seconds = float(raw)
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(raw)
        except (TypeError, ValueError):
            logger.debug("Ignoring unparseable Retry-After value %r", raw)
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=UTC)
        seconds = (retry_at - datetime.now(UTC)).total_seconds()

    if not math.isfinite(seconds):
        return None
    if seconds <= 0:
        # A zero or already-elapsed hint carries no wait information. Returning
        # it verbatim would skip the sleep entirely and let the whole budget
        # burn in a tight loop, so fall back to the exponential curve.
        return None
    return min(seconds, _MAX_RETRY_AFTER_SECONDS)


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
    """Return whether any base class is a known transient provider-SDK error."""
    return any(
        (base.__module__.partition(".")[0], base.__name__) in _TRANSIENT_SDK_EXC_NAMES
        for base in type(exc).__mro__
    )


def _is_http_transport_error(exc: BaseException) -> bool:
    """Return whether `exc` is a transient HTTP response transport failure."""
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

    error_type = type(exc)
    if error_type.__module__.startswith("httpcore") and error_type.__name__ in {
        "ReadError",
        "RemoteProtocolError",
    }:
        return True
    return (
        error_type.__module__ == "aiohttp.http_exceptions"
        and error_type.__name__ == "TransferEncodingError"
        and "Not enough data to satisfy transfer length header" in str(exc)
    )


def _direct_model_error_retryability(
    exc: BaseException, *, raised: bool
) -> bool | None:
    """Classify one exception before inspecting any wrapped failures.

    Args:
        exc: The exception to classify.
        raised: Whether `exc` is the exception the model call actually raised,
            rather than one reached through a group member or a cause chain.

    Returns:
        Whether the exception is retryable, or `None` when it has no direct
        signal and its group members or chain should be inspected.
    """
    if isinstance(exc, ModelError):
        return exc.is_retryable

    if _is_http_transport_error(exc):
        return True

    if not isinstance(exc, Exception):
        return None

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

    # Stdlib transport faults raised directly (rare, but cheap to cover). This
    # heuristic alone is deliberately confined to the raised exception: Python
    # sets `__context__` on anything raised inside an `except` block, so
    # honouring it here would make a permanent failure that merely surfaced
    # while handling a timeout look transient and burn the whole budget on it.
    #
    # The checks above are not confined that way, and the asymmetry is chosen,
    # not an oversight. `TimeoutError`/`ConnectionError` are broad -- every
    # `asyncio.wait_for` deadline and every socket fault in the process is one
    # -- whereas a package-qualified SDK class or an httpx transport error is
    # narrow enough that finding one in the context chain really does mean the
    # call died in transport and an SDK re-raised inside its `except`. That
    # wrap-and-reraise shape is the common one, so those stay trusted through
    # `__context__` (see `test_predicate_retries_transport_error_in_context_chain`).
    if raised and isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    return None


def _is_retryable_model_error(exc: Exception) -> bool:
    """Return whether a model error tree contains a transient failure.

    Descends into `BaseExceptionGroup` members and the cause chain, so a
    transport fault wrapped by an async task group is still found. An exception
    that classifies either way decides for its own branch and is not descended
    through, which keeps a definite `ModelError.is_retryable` verdict (an
    authentication failure, say) authoritative over whatever it happens to
    wrap.

    The stock retry check stops at the raised exception, so it would miss a
    `httpx.ConnectError` wrapped in an `ExceptionGroup`; this walk catches it.
    """
    pending: list[tuple[BaseException, bool]] = [(exc, True)]
    seen: set[int] = set()
    while pending:
        current, raised = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        retryable = _direct_model_error_retryability(current, raised=raised)
        if retryable is not None:
            if retryable:
                return True
            continue
        if isinstance(current, BaseExceptionGroup):
            pending.extend((member, False) for member in current.exceptions)
        cause = current.__cause__ or current.__context__
        if cause is not None:
            pending.append((cause, False))
    return False


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
        # `info`, not `debug`: a fault in this module's own instrumentation
        # (a `StreamMessagesHandler` signature change, say) surfaces here
        # classified as non-transient, and would otherwise reach the user as an
        # unexplained provider error with no traceback at default log levels.
        logger.info(
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
    on_retry: Callable[[int, int, Exception], None],
    retry_guard: Callable[[Exception, int, float], bool] | None = None,
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
            # Settle eligibility before consulting the guard. A guard that ran
            # first would blame the delay budget for an error that was never
            # going to be retried, and would skip the exhausted-budget log
            # entirely.
            if not _is_retryable_model_error(exc) or attempt >= max_retries:
                _log_give_up(exc, attempt + 1, max_retries)
                # Re-raise, don't convert to an `AIMessage`: a dead provider
                # should end the turn as an error, not as a reply the model
                # never made.
                raise
            # Drawn once: the backoff carries jitter, so re-deriving it for the
            # guard would authorise one delay and then sleep a different one.
            delay = _retry_delay_seconds(attempt, exc)
            if retry_guard is not None and not retry_guard(exc, attempt + 1, delay):
                raise
            on_retry(attempt + 1, max_retries, exc)
            if delay:
                time.sleep(delay)
    msg = "Unexpected: retry loop completed without returning"
    raise RuntimeError(msg)


async def _aretry_call[ResultT](
    call: Callable[[], Awaitable[ResultT]],
    *,
    max_retries: int,
    on_retry: Callable[[int, int, Exception], None],
    retry_guard: Callable[[Exception, int, float], bool] | None = None,
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
            # Settle eligibility before consulting the guard. A guard that ran
            # first would blame the delay budget for an error that was never
            # going to be retried, and would skip the exhausted-budget log
            # entirely.
            if not _is_retryable_model_error(exc) or attempt >= max_retries:
                _log_give_up(exc, attempt + 1, max_retries)
                # Always re-raise (see `_retry_call`).
                raise
            # Drawn once: the backoff carries jitter, so re-deriving it for the
            # guard would authorise one delay and then sleep a different one.
            delay = _retry_delay_seconds(attempt, exc)
            if retry_guard is not None and not retry_guard(exc, attempt + 1, delay):
                raise
            on_retry(attempt + 1, max_retries, exc)
            if delay:
                await asyncio.sleep(delay)
    msg = "Unexpected: retry loop completed without returning"
    raise RuntimeError(msg)


def _log_auxiliary_retry(attempt: int, max_retries: int, exc: Exception) -> None:
    """Log one auxiliary-model retry.

    Only the final exception survives to be re-raised, so an attempt logged
    without its cause is unrecoverable: five 429s and a 429 followed by four
    connection resets are indistinguishable after the fact.
    """
    logger.warning(
        "Auxiliary model call failed with %s (status %s); retrying %d/%d",
        type(exc).__name__,
        _extract_status_code(exc),
        attempt,
        max_retries,
        exc_info=exc,
    )


def _auxiliary_max_retries(model: object) -> int:
    """Return the auxiliary retry budget for `model`, defaulting when unstamped.

    A model that never passed through `create_model` carries no
    `MODEL_RETRIES_ATTR`. Defaulting that case to zero would make every
    auxiliary wrapper a silent passthrough -- and because
    `_install_summary_model_retries` replaces LangChain's unconditional
    three-attempt `with_retry`, compaction summarization would quietly drop to
    a single attempt. Fall back to the normal budget and say so, since a
    retry-less summarizer is invisible at runtime.

    Returns:
        The attached budget, or `DEFAULT_MODEL_RETRIES` when there is none.
    """
    resolved = _model_max_retries(model, -1)
    if resolved >= 0:
        return resolved
    logger.warning(
        "Model %s carries no dcode retry metadata; auxiliary calls fall back "
        "to %d retries and its own SDK retry loop may still be active",
        type(model).__name__,
        DEFAULT_MODEL_RETRIES,
    )
    return DEFAULT_MODEL_RETRIES


def _delay_budget_guard(
    max_total_delay: float | None,
    *,
    label: str = "Auxiliary model",
) -> Callable[[Exception, int, float], bool]:
    """Build a guard that keeps total retry sleep within `max_total_delay`.

    Callers that run under an enclosing deadline cannot afford an honoured
    `Retry-After` of up to `_MAX_RETRY_AFTER_SECONDS`: the sleep outlives the
    deadline, the task is cancelled mid-wait, and the real provider error is
    replaced by an unrelated `TimeoutError`. Refusing the retry surfaces the
    genuine cause instead, and avoids retrying a rate limit early.

    The budget is cumulative, not per-delay. Capping each wait in isolation
    bounds nothing: five waits that each clear a 5s ceiling still spend 25s,
    which is exactly how a 20s classifier deadline was overrun by the retries
    meant to fit inside it.

    Args:
        max_total_delay: Cumulative sleep ceiling, or `None` to honour the full
            policy.
        label: Sentence-leading subject for the refusal log, so an interactive
            stall reads differently from an auxiliary one.

    Returns:
        A `retry_guard` callable for the shared retry loops.
    """
    spent = 0.0

    def guard(exc: Exception, attempt: int, delay: float) -> bool:  # noqa: ARG001
        nonlocal spent
        if max_total_delay is None:
            return True
        if spent + delay <= max_total_delay:
            spent += delay
            return True
        logger.warning(
            "%s retries would wait %.1fs past the total delay budget of "
            "%.1fs; surfacing %s instead",
            label,
            spent + delay - max_total_delay,
            max_total_delay,
            type(exc).__name__,
        )
        return False

    return guard


def retry_model_call[ResultT](
    model: object,
    call: Callable[[], ResultT],
    *,
    max_total_delay: float | None = None,
) -> ResultT:
    """Run a non-streaming auxiliary model call with its configured retry budget.

    Args:
        model: Model carrying dcode retry metadata when dcode owns its SDK retries.
        call: Fresh invocation callable to run for each attempt.
        max_total_delay: Total time this caller can spend sleeping between
            attempts, for callers running under an enclosing deadline. `None`
            honours the full policy.

    Returns:
        The successful call result.
    """
    return _retry_call(
        call,
        max_retries=_auxiliary_max_retries(model),
        on_retry=_log_auxiliary_retry,
        retry_guard=_delay_budget_guard(max_total_delay),
    )


async def aretry_model_call[ResultT](
    model: object,
    call: Callable[[], Awaitable[ResultT]],
    *,
    max_total_delay: float | None = None,
) -> ResultT:
    """Run an asynchronous auxiliary model call with its configured retry budget.

    Args:
        model: Model carrying dcode retry metadata when dcode owns its SDK retries.
        call: Fresh async invocation callable to run for each attempt.
        max_total_delay: Total time this caller can spend sleeping between
            attempts, for callers running under an enclosing deadline. `None`
            honours the full policy.

    Returns:
        The successful call result.
    """
    return await _aretry_call(
        call,
        max_retries=_auxiliary_max_retries(model),
        on_retry=_log_auxiliary_retry,
        retry_guard=_delay_budget_guard(max_total_delay),
    )


def retry_counts_from_event(
    event: Mapping[Any, object],
) -> tuple[int, int] | None:
    """Validate the attempt counters of an untrusted `model_retry` payload.

    Every surface that renders a retry needs the same two numbers under the
    same range, so the check lives once with the producer rather than being
    re-derived per surface with drifting strictness.

    Args:
        event: Custom-stream payload, not trusted to hold sane numbers.

    Returns:
        The `(attempt, max_retries)` pair, or `None` when either is unusable.
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
        return (attempt, max_retries)
    return None


def retry_status_from_event(event: Mapping[Any, object]) -> str:
    """Return retry status text for an untrusted `model_retry` payload.

    Both the TUI and the headless client render this status line, so its
    validation lives with the producer rather than being written twice with
    different strictness.

    Args:
        event: Custom-stream payload, not trusted to hold sane numbers.

    Returns:
        The validated status line, or a cause-free fallback for malformed data.
    """
    counts = retry_counts_from_event(event)
    if counts is None:
        logger.warning("Ignoring malformed model_retry payload: %r", dict(event))
        return _RETRY_STATUS_FALLBACK
    return format_retry_status(*counts)


def retry_marker_from_event(event: Mapping[Any, object]) -> str:
    """Build the in-chat retry marker from validated numeric fields only.

    The event's own `message` field is untrusted render text, so the marker is
    re-derived from `attempt`/`max_retries` and never parses markup out of it.

    Always returns a marker. By the time this is called the partial reply has
    already been finalized and detached from the stream, so returning nothing
    would leave a truncated answer in the chat that reads as a complete one,
    followed by a second full answer, with nothing saying the first was cut off.
    Unusable numbers cost the "1/5" suffix, not the marker -- the same way
    `retry_status_from_event` degrades to a cause-free status line.

    Args:
        event: Custom-stream payload, not trusted to hold sane numbers.

    Returns:
        The marker line, counted when the numbers allow it.
    """
    counts = retry_counts_from_event(event)
    if counts is None:
        logger.warning(
            "Unusable retry counts in model_retry payload; marking the "
            "superseded reply without them"
        )
        return RETRY_MARKER_FALLBACK
    attempt, max_retries = counts
    return (
        "Connection dropped; the partial response above is incomplete. "
        f"Retrying {attempt}/{max_retries}."
    )


def legacy_retry_index(event: Mapping[Any, object]) -> int:
    """Identity fallback for a `model_retry` payload that names no attempt.

    A producer that predates attempt lifecycle events carries no `call_id`, so
    a consumer cannot tell a second retry of one call from a redelivery of the
    same event by correlation. The retry counter it does carry is enough: two
    retries of one call always differ, while a redelivery does not.

    Args:
        event: Custom-stream payload, not trusted to hold sane numbers.

    Returns:
        The payload's retry counter when it is a usable int, else `-1`.
    """
    attempt = event.get("attempt")
    if isinstance(attempt, int) and not isinstance(attempt, bool):
        return attempt
    return -1


def build_retry_event(
    attempt: int,
    max_retries: int,
    *,
    call_id: str | None = None,
    failed_attempt: int | None = None,
    output_may_have_started: bool = False,
) -> dict[str, object]:
    """Build the custom-stream payload announcing a model retry.

    Args:
        attempt: The 1-indexed retry number about to be attempted.
        max_retries: The configured maximum retry count.
        call_id: Opaque ID correlating every attempt of one model call. Omit
            for producers that predate attempt lifecycle events.
        failed_attempt: The 0-indexed attempt being superseded. Required to
            carry `call_id`.
        output_may_have_started: Whether the superseded attempt may have put
            message output beyond server control. Conservative by design: the
            tracker flags before forwarding a chunk.

    Returns:
        A stream-writer payload consumed by the client renderers.

    Raises:
        ValueError: If only one of `call_id` and `failed_attempt` is given.
    """
    if (call_id is None) != (failed_attempt is None):
        msg = "call_id and failed_attempt must be provided together"
        raise ValueError(msg)
    event: dict[str, object] = {
        "type": "model_retry",
        "attempt": attempt,
        "max_retries": max_retries,
        "message": format_retry_status(attempt, max_retries),
    }
    if call_id is not None:
        event["call_id"] = call_id
        event["failed_attempt"] = failed_attempt
        event["output_may_have_started"] = output_may_have_started
    return event


def build_attempt_event(call_id: str, attempt: int, *, phase: str) -> dict[str, object]:
    """Build the custom-stream payload marking one model attempt boundary.

    Args:
        call_id: Opaque ID shared by every attempt of one model call.
        attempt: The 0-indexed attempt whose boundary is marked.
        phase: `"start"` before the handler runs, `"complete"` after it
            returns successfully.

    Returns:
        A stream-writer payload consumed by the client renderers.

    Raises:
        ValueError: If `phase` is not a known lifecycle phase.
    """
    if phase not in _ATTEMPT_PHASES:
        msg = f"phase must be one of {sorted(_ATTEMPT_PHASES)}, got {phase!r}"
        raise ValueError(msg)
    return {
        "type": "model_attempt",
        "phase": phase,
        "call_id": call_id,
        "attempt": attempt,
    }


def _validated_call_id(value: object) -> str | None:
    """Return `value` as a correlation ID, or `None` when it is untrusted."""
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= _CALL_ID_MAX_LENGTH
        or any(char not in _CALL_ID_CHARS for char in value)
    ):
        return None
    return value


def model_retry_from_event(event: Mapping[Any, object]) -> dict[str, object] | None:
    """Return validated retry-correlation fields from an untrusted event."""
    call_id = _validated_call_id(event.get("call_id"))
    failed_attempt = event.get("failed_attempt")
    visible = event.get("output_may_have_started")
    if call_id is None and failed_attempt is None and visible is None:
        return None
    if (
        call_id is None
        or not isinstance(failed_attempt, int)
        or isinstance(failed_attempt, bool)
        or failed_attempt < 0
        or not isinstance(visible, bool)
    ):
        logger.warning("Ignoring malformed model_retry correlation fields")
        return None
    return {
        "call_id": call_id,
        "failed_attempt": failed_attempt,
        "output_may_have_started": visible,
    }


def model_attempt_from_event(
    event: Mapping[Any, object],
) -> dict[str, object] | None:
    """Return a validated `model_attempt` payload from an untrusted event.

    Remote and local consumers receive lifecycle events from the same custom
    stream as provider-shaped data, so every field is structurally validated
    before use. Unknown fields are ignored and unknown phases are dropped, so
    a newer server never breaks an older client.

    Args:
        event: Custom-stream payload, not trusted to hold sane values.

    Returns:
        A dict with `type`, `phase`, `call_id`, and `attempt`, or `None` for
        malformed data.
    """
    phase = event.get("phase")
    call_id = _validated_call_id(event.get("call_id"))
    attempt = event.get("attempt")
    if (
        not isinstance(phase, str)
        or phase not in _ATTEMPT_PHASES
        or call_id is None
        or not isinstance(attempt, int)
        or isinstance(attempt, bool)
        or attempt < 0
    ):
        logger.warning("Ignoring malformed model_attempt lifecycle fields")
        return None
    return {
        "type": "model_attempt",
        "phase": phase,
        "call_id": call_id,
        "attempt": attempt,
    }


class CodeModelRetryMiddleware(AgentMiddleware):
    """Retry transient model-node failures without replaying completed tools.

    Emits `model_attempt` start/complete lifecycle events around every handler
    invocation, correlated by one `call_id` per model call, so clients can
    reconcile output from a superseded attempt when a transient failure is
    retried after streaming began.
    """

    def __init__(
        self,
        *,
        max_retries: int = DEFAULT_MODEL_RETRIES,
        stream_output_is_visible: bool = True,
    ) -> None:
        """Initialize the middleware with the resolved retry count.

        Args:
            max_retries: Startup fallback for retry attempts after the initial
                call. `0` disables retries unless the request's runtime-selected
                model carries a different provider-specific budget.
            stream_output_is_visible: Whether message-stream chunks emitted by
                this model reach a user-visible consumer; it decides the
                `output_may_have_started` supersession flag on retry events.
                Keep `True` unless the entire nested stream is filtered before
                rendering.

        Raises:
            TypeError: If `max_retries` or `stream_output_is_visible` has the
                wrong type.
            ValueError: If `max_retries` is negative.
        """
        # `True >= 0` passes and `range(True + 1)` runs two attempts, so an
        # unchecked bool reads as a budget of one retry.
        if isinstance(max_retries, bool):
            msg = f"max_retries must be an int, got {type(max_retries).__name__}"
            raise TypeError(msg)
        if max_retries < 0:
            msg = "max_retries must be >= 0"
            raise ValueError(msg)
        if not isinstance(stream_output_is_visible, bool):
            msg = (
                "stream_output_is_visible must be a bool, got "
                f"{type(stream_output_is_visible).__name__}"
            )
            raise TypeError(msg)
        self.max_retries = max_retries
        self.stream_output_is_visible = stream_output_is_visible

    @staticmethod
    def _emit_stream_event(request: ModelRequest, event: dict[str, object]) -> None:
        writer = getattr(getattr(request, "runtime", None), "stream_writer", None)
        if writer is None:
            return
        try:
            writer(event)
        except GraphBubbleUp:
            # LangGraph control flow must not be mistaken for a writer fault.
            raise
        except Exception:
            # These events are the only signal that a pause is a retry and the
            # only correlation a client has between chunks and attempts, so
            # losing one must be visible in the logs without failing the run.
            logger.warning(
                "Failed to emit %s stream event", event["type"], exc_info=True
            )

    def _emit_retry_status(
        self,
        request: ModelRequest,
        attempt: int,
        max_retries: int,
        exc: Exception,
        call_id: str,
        has_streamed: bool,
    ) -> None:
        event = build_retry_event(
            attempt,
            max_retries,
            call_id=call_id,
            failed_attempt=attempt - 1,
            output_may_have_started=has_streamed and self.stream_output_is_visible,
        )
        # The user-facing event stays deliberately vague, but the log must name
        # the cause: only the last exception is re-raised, so an attempt logged
        # without its type and status leaves no way to tell a run of rate
        # limits from a run of connection resets.
        logger.warning(
            "Model call failed with %s (status %s); %s",
            type(exc).__name__,
            _extract_status_code(exc),
            event["message"],
            exc_info=exc,
        )
        self._emit_stream_event(request, event)

    def _request_max_retries(self, request: ModelRequest) -> int:
        # A `/model` switch stamps its own budget on the constructed model;
        # that wins over the startup fallback, so read it per request.
        return _model_max_retries(getattr(request, "model", None), self.max_retries)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Retry a synchronous model-node call, even after streamed output.

        Returns:
            The successful model response.
        """
        max_retries = self._request_max_retries(request)
        stream_tracker = _MessageStreamTracker()
        call_id = uuid.uuid4().hex
        current_attempt = 0

        def call() -> ModelResponse:
            nonlocal stream_tracker
            stream_tracker = _MessageStreamTracker()
            self._emit_stream_event(
                request, build_attempt_event(call_id, current_attempt, phase="start")
            )
            with _track_message_streams(stream_tracker):
                result = handler(request)
            self._emit_stream_event(
                request,
                build_attempt_event(call_id, current_attempt, phase="complete"),
            )
            return result

        def on_retry(attempt: int, budget: int, exc: Exception) -> None:
            nonlocal current_attempt
            self._emit_retry_status(
                request, attempt, budget, exc, call_id, stream_tracker.has_streamed
            )
            current_attempt = attempt

        return _retry_call(
            call,
            max_retries=max_retries,
            on_retry=on_retry,
            retry_guard=_delay_budget_guard(
                _MAX_INTERACTIVE_TOTAL_DELAY_SECONDS, label="Interactive model"
            ),
        )

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Retry an asynchronous model-node call, even after streamed output.

        Returns:
            The successful model response.
        """
        max_retries = self._request_max_retries(request)
        stream_tracker = _MessageStreamTracker()
        call_id = uuid.uuid4().hex
        current_attempt = 0

        async def call() -> ModelResponse:
            nonlocal stream_tracker
            stream_tracker = _MessageStreamTracker()
            self._emit_stream_event(
                request, build_attempt_event(call_id, current_attempt, phase="start")
            )
            with _track_message_streams(stream_tracker):
                result = await handler(request)
            self._emit_stream_event(
                request,
                build_attempt_event(call_id, current_attempt, phase="complete"),
            )
            return result

        def on_retry(attempt: int, budget: int, exc: Exception) -> None:
            nonlocal current_attempt
            self._emit_retry_status(
                request, attempt, budget, exc, call_id, stream_tracker.has_streamed
            )
            current_attempt = attempt

        return await _aretry_call(
            call,
            max_retries=max_retries,
            on_retry=on_retry,
            retry_guard=_delay_budget_guard(
                _MAX_INTERACTIVE_TOTAL_DELAY_SECONDS, label="Interactive model"
            ),
        )
