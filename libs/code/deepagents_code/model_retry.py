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
from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal, TypedDict, TypeVar

from langchain.agents.middleware import ModelRetryMiddleware
from langchain_core.callbacks import (
    AsyncCallbackManager,
    BaseCallbackHandler,
    CallbackManager,
)
from langchain_core.exceptions import ContextOverflowError
from langchain_core.runnables import ensure_config, patch_config
from langchain_core.runnables.config import set_config_context
from langgraph.errors import GraphBubbleUp
from pydantic import ValidationError

from deepagents_code.config import (
    CLI_MAX_RETRIES_KEY,
    DEFAULT_MODEL_RETRIES,
    get_glyphs,
    get_model_retries,
    is_valid_retry_count,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterator
    from typing import Any
    from uuid import UUID

    from langchain.agents.middleware.types import ModelRequest, ModelResponse
    from langchain_core.language_models import BaseChatModel
    from langchain_core.outputs import ChatGenerationChunk, GenerationChunk
    from langchain_core.runnables import RunnableConfig

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

__all__ = [
    "DEFAULT_MODEL_RETRIES",
    "CodeModelRetryMiddleware",
    "ModelRetryEvent",
    "build_retry_event",
    "exception_chain",
    "format_retry_status",
]

# Marker attribute that typed local faults can set to opt out of model-node
# retries (for example Auto's classifier wait-budget deadline). Prefer this over
# message matching so classification stays reliable across wrappers.
DCODE_MODEL_RETRYABLE_ATTR = "dcode_model_retryable"


class ModelRetryEvent(TypedDict):
    """Custom-stream payload announcing a model retry.

    The `type` discriminant lets the client and TUI renderers dispatch this
    event apart from rubric events sharing the `custom` stream mode.
    """

    type: Literal["model_retry"]
    attempt: int
    max_retries: int
    message: str


_HIDDEN_LC_SOURCES = frozenset({"summarization", "auto_mode_classifier"})
"""`lc_source` metadata values marking model output the user never sees.

Producers invoke their model with `config={"metadata": {"lc_source": ...}}`:
`_create_summary_with_retry` / `_acreate_summary_with_retry` in
`offload_middleware` (and upstream `SummarizationMiddleware._create_summary`)
tag `"summarization"`; `AutoModeHITLMiddleware._classify` tags
`"auto_mode_classifier"`. Neither reaches the transcript: the TUI renderer filters
both (`_is_summarization_chunk` / `_is_auto_mode_classifier_chunk` in
`tui/textual_adapter`), `client/non_interactive` filters summarization, and the
classifier does not run outside the interactive runtime at all. So their tokens
must not count as user-visible output when deciding whether a retry is safe.
"""


class _StreamOutputTracker(BaseCallbackHandler):
    """Record whether a model attempt emitted any *user-visible* output.

    The tracker is installed on the ambient callback config, so it observes
    every model run inside the retried handler -- not only the model node being
    guarded. Inner middleware runs its own model calls there (the summarizer
    runs inside `SummarizationMiddleware.wrap_model_call`, which this middleware
    wraps), and their tokens are hidden from the user. Counting them would latch
    `emitted` before the guarded call even starts and silently veto every retry
    on summarizing turns, so runs tagged with a `_HIDDEN_LC_SOURCES` marker are
    tracked by run id and ignored.

    Runs that arrive without a recognized marker still count, keeping the
    duplicate-output guard fail-safe for anything not positively identified as
    internal.
    """

    run_inline = True

    def __init__(self) -> None:
        """Initialize request-local streaming state."""
        self.emitted = False
        self._hidden_runs: set[UUID] = set()

    def _note_run_start(self, run_id: UUID, metadata: dict[str, Any] | None) -> None:
        """Record `run_id` when its metadata marks the run as non-user-facing."""
        if metadata is not None and metadata.get("lc_source") in _HIDDEN_LC_SOURCES:
            self._hidden_runs.add(run_id)

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Track whether this completion run is hidden from the user."""
        del serialized, prompts, parent_run_id, tags, kwargs
        self._note_run_start(run_id, metadata)

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Track whether this chat run is hidden from the user."""
        del serialized, messages, parent_run_id, tags, kwargs
        self._note_run_start(run_id, metadata)

    def on_llm_new_token(
        self,
        token: str | list[str | dict[str, Any]],
        *,
        chunk: GenerationChunk | ChatGenerationChunk | None = None,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Mark a token-stream chunk as externally visible, including empty ones."""
        del token, chunk, parent_run_id, tags, kwargs
        if run_id in self._hidden_runs:
            return
        self.emitted = True

    def on_stream_event(
        self,
        event: object,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Mark a protocol-stream event as externally visible."""
        del event, parent_run_id, tags, kwargs
        if run_id in self._hidden_runs:
            return
        self.emitted = True


def _stream_tracking_config(
    tracker: _StreamOutputTracker,
    *,
    asynchronous: bool = False,
) -> RunnableConfig:
    """Add `tracker` to a request-local copy of the active callback config.

    The model itself must remain unchanged: model clients can carry mutable
    provider or test state that a shallow Pydantic copy would fail to preserve.

    Args:
        tracker: Per-request output tracker.
        asynchronous: Whether the request uses the asynchronous callback path.

    Returns:
        A runnable config that preserves existing handlers and adds `tracker`.
    """
    config = ensure_config()
    manager = AsyncCallbackManager if asynchronous else CallbackManager
    callbacks = manager.configure(inheritable_callbacks=config.get("callbacks"))
    callbacks.add_handler(tracker)
    return patch_config(config, callbacks=callbacks)


def _runtime_model_retry_override(runtime: object) -> int | None:
    """Return the validated retry carrier from a model or tool runtime.

    Args:
        runtime: Runtime whose context may be a schema object or mapping.

    Returns:
        The non-negative retry override, or `None` when absent or invalid.
    """
    context = getattr(runtime, "context", None)
    if isinstance(context, Mapping):
        params = context.get("model_params")
    else:
        params = getattr(context, "model_params", None)
    if not isinstance(params, Mapping):
        return None
    raw = params.get(CLI_MAX_RETRIES_KEY)
    if is_valid_retry_count(raw):
        return raw
    return None


_INITIAL_DELAY_SECONDS = 0.2
"""First backoff delay, 200ms (mirrors Codex's initial wait as of 2026-01)."""

_BACKOFF_FACTOR = 2.0
"""Exponential backoff multiplier (2.0; mirrors Codex as of 2026-01)."""

_MAX_DELAY_SECONDS = 10.0
"""Cap on a single backoff delay so exponential growth stays bounded."""

_JITTER_FRACTION = 0.1
"""Multiplicative jitter of +-10% (0.9..1.1; mirrors Codex as of 2026-01)."""

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
        # Google api_core exceptions whose HTTP status is unavailable over a gRPC
        # transport (`.code` is a `grpc.StatusCode` enum, not an int), so the
        # status-code path cannot classify them. Match by name instead.
        "ResourceExhausted",
        "ServiceUnavailable",
        "DeadlineExceeded",
        "Aborted",
        "InternalServerError",
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
"""Provider error codes that are permanent regardless of the HTTP status.

OpenAI and compatible providers return HTTP 429 for exhausted billing/quota
(`insufficient_quota`), which no amount of retrying will clear. Classifying
these by their string error code surfaces the actionable failure immediately
instead of after the full retry budget and its backoff sleeps. The remaining
codes (hard billing limits, inactive billing, deactivated accounts) are equally
permanent; they usually arrive with an already-fatal 401/403, so matching them
here is a safety net that holds regardless of the status the provider attaches.
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


def _safe_getattr(obj: object, name: str) -> Any:  # noqa: ANN401  # arbitrary SDK attribute
    """Return `obj.name`, treating a raising descriptor as absent.

    Provider SDKs (and custom provider classes registered via `[models.providers]`)
    may expose error details through properties that parse a response body lazily
    and can themselves raise. Classification runs inside an `except` block, so an
    unguarded probe would replace the model failure with the probe's own error --
    the user would see that instead of the real cause, and no retry would happen.

    Args:
        obj: Object to probe, normally the exception being classified.
        name: Attribute name to read.

    Returns:
        The attribute value, or `None` when it is missing or raises.
    """
    try:
        return getattr(obj, name, None)
    except Exception:
        logger.debug("Probing %r on %r raised", name, type(obj), exc_info=True)
        return None


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
        value = _safe_getattr(exc, attr)
        if isinstance(value, str) and value:
            return value
    response = _safe_getattr(exc, "response")
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
    status = _safe_getattr(exc, "status_code")
    if isinstance(status, bool):
        return None
    if isinstance(status, int):
        return status

    # Google API Core exceptions often expose an int HTTP status on `code`
    # (REST/HTTP: ResourceExhausted→429, ServiceUnavailable→503, Aborted→409).
    # Over pure gRPC the same attribute is a non-int StatusCode enum and is
    # ignored here so name-based classification can run. Check int `code`
    # before `response`, which may be absent or may contain a gRPC call object.
    code = _safe_getattr(exc, "code")
    if isinstance(code, int) and not isinstance(code, bool):
        return code

    response = _safe_getattr(exc, "response")
    if response is not None:
        response_status = _safe_getattr(response, "status_code")
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

    http_status = _safe_getattr(exc, "http_status")
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


def _is_marked_non_retryable(exc: BaseException) -> bool:
    """Return whether `exc` opted out of model-node retries via marker attr."""
    return _safe_getattr(exc, DCODE_MODEL_RETRYABLE_ATTR) is False


_NEVER_RETRY_TYPES: tuple[type[BaseException], ...] = (
    GraphBubbleUp,
    ContextOverflowError,
    ValidationError,
)
"""Types that are never a transient model fault, regardless of the error chain.

`GraphBubbleUp` is LangGraph's control-flow signal (interrupts and other
bubble-ups), not a failure; retrying it would suppress a pending approval prompt.
`ContextOverflowError` is deterministic -- the request is too large and will be
too large on every attempt, so retrying only delays the compaction path that
actually handles it. `ValidationError` is how a structured-output call reports a
response that does not fit the schema; the same prompt reproduces it, so retrying
just spends the budget to surface the identical error.

These carry no HTTP status and no provider error code, so the status-based checks
in `_is_definitive_non_retryable` cannot recognize them. Without an explicit
entry, any transient fault elsewhere in the chain (a dropped connection recorded
as `__context__`, or a sibling under a task group) would make the chain scan vote
to retry.
"""


def _mark_non_retryable(exc: BaseException) -> None:
    """Mark `exc` so an enclosing retry driver declines to retry it.

    Retry drivers nest: the model-node middleware wraps inner middleware that runs
    its own `run_with_retry` (summarization, the rubric grader, the Auto
    classifier). Without a marker the outer driver would re-run a handler whose
    inner budget is already spent, multiplying attempts -- `retries + 1` squared
    in the worst case -- while the status line still reports the outer budget.

    Marking is best-effort: some SDK exceptions define `__slots__` or are otherwise
    read-only, and failing to annotate must never replace the original error.
    """
    try:
        setattr(exc, DCODE_MODEL_RETRYABLE_ATTR, False)
    except Exception:
        logger.debug("Could not mark %r as retry-exhausted", type(exc), exc_info=True)


def _is_status_retryable(status: int, *, error_code: str | None) -> bool:
    """Return whether an HTTP status (plus optional provider code) is retryable."""
    if error_code in _THROTTLING_ERROR_CODES:
        return True
    return status in _RETRYABLE_STATUS_CODES or (
        _HTTP_SERVER_ERROR_FLOOR <= status < _HTTP_SERVER_ERROR_CEILING
    )


def _is_definitive_non_retryable(exc: Exception) -> bool:
    """Return whether `exc` is a permanent client/auth/billing failure.

    Used as a top-level (and top-level ExceptionGroup member) guard so a
    deterministic failure wins over sibling or chained transient faults.
    """
    if _is_marked_non_retryable(exc):
        return True
    if isinstance(exc, _NEVER_RETRY_TYPES):
        return True
    error_code = _provider_error_code(exc)
    if error_code in _NONRETRYABLE_ERROR_CODES:
        return True
    status = _extract_status_code(exc)
    if status is None:
        return False
    if _is_status_retryable(status, error_code=error_code):
        return False
    # Known-transient SDK types (Google Aborted→409, etc.) are not definitive.
    return not _is_transient_sdk_error(exc)


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
    # Explicit local opt-out (for example the Auto classifier wait budget).
    if _is_marked_non_retryable(exc):
        return False

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

    error_type = type(exc)
    if error_type.__module__.startswith("httpcore") and error_type.__name__ in {
        "ReadError",
        "RemoteProtocolError",
    }:
        return True
    if (
        error_type.__module__ == "aiohttp.http_exceptions"
        and error_type.__name__ == "TransferEncodingError"
        and "Not enough data to satisfy transfer length header" in str(exc)
    ):
        return True

    # AWS throttling arrives as an HTTP 400 body; the error code is the only
    # signal that this 4xx is actually a rate limit worth retrying.
    if error_code in _THROTTLING_ERROR_CODES:
        return True

    # Status-bearing provider errors: retry 408/429/5xx, then allow known
    # transient SDK types whose status would otherwise look permanent (Google
    # Aborted → HTTP 409). Do not fall through to bare TimeoutError heuristics
    # for other 4xx codes.
    status = _extract_status_code(exc)
    if status is not None:
        if _is_status_retryable(status, error_code=error_code):
            return True
        return _is_transient_sdk_error(exc)

    if _is_transient_sdk_error(exc):
        return True

    # Stdlib transport faults raised directly (rare, but cheap to cover).
    return isinstance(exc, (TimeoutError, ConnectionError))


def exception_chain(exc: BaseException) -> Iterator[BaseException]:
    """Yield an exception, its causes, contexts, and group members once.

    Walks both `__cause__` and `__context__` (with cycle protection), plus
    `BaseExceptionGroup` members, so transient transport faults stay
    discoverable whether they are explicit causes, implicit contexts, or
    sibling tasks under asyncio/anyio.

    `raise X from None` is honored as an explicit statement that the in-flight
    context is unrelated, so it is not followed -- otherwise an incidental
    transient error could vote to retry a deliberate, deterministic failure.
    Note this is narrower than `__suppress_context__` alone: assigning `__cause__`
    also sets that flag, and there both links stay in the chain because an
    explicit cause suppresses the context only for *display*.
    """
    pending = [exc]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if (
            current.__context__ is not None
            and current.__context__ is not current.__cause__
            and not (current.__suppress_context__ and current.__cause__ is None)
        ):
            pending.append(current.__context__)


# Back-compat private alias used by internal callers/tests.
_exception_chain = exception_chain


def _contains_retryable_model_error(exc: BaseException) -> bool:
    """Return whether an exception chain contains a transient model failure."""
    return any(
        isinstance(current, Exception) and _is_retryable_model_error(current)
        for current in exception_chain(exc)
    )


def _should_retry_after_failure(exc: BaseException) -> bool:
    """Decide whether a failed model call is worth retrying.

    A definitive verdict anywhere in the exception chain wins: a permanent
    provider error code, a status-bearing deterministic client error (a 4xx other
    than 408/429 that is not an AWS throttle or known-transient SDK type), a
    never-retry type, or a typed opt-out marker. Only when nothing in the chain is
    definitively permanent does a transient fault anywhere in it vote to retry.

    Checking the whole chain in both directions matters because the two verdicts
    reach us in mirrored shapes. A permanent failure can hide *under* a transient
    wrapper (`ReadError` raised `from` a 401, or `group[group[401, ReadError]]`),
    and a transient failure can hide under an opaque wrapper. Scanning the chain
    for "permanent" first and "transient" second resolves both without letting a
    transient sibling launder a deterministic error into five pointless attempts.

    Args:
        exc: The exception raised by the model call.

    Returns:
        `True` when the failure should be retried.
    """
    chain = list(exception_chain(exc))
    if any(
        isinstance(current, Exception) and _is_definitive_non_retryable(current)
        for current in chain
    ):
        return False
    return any(
        isinstance(current, Exception) and _is_retryable_model_error(current)
        for current in chain
    )


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
        # `retry_on`/`on_failure` satisfy the base constructor's validation only;
        # both `wrap_model_call` overrides bypass the base retry loop entirely.
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
            Delay in seconds. The exponential term is capped at
                `_MAX_DELAY_SECONDS` *before* +-10% jitter is applied (mirroring
                Codex), so a capped delay can land slightly above the cap.
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
        return get_model_retries(model, self.max_retries)

    @staticmethod
    def _should_attempt_again(
        exc: Exception,
        *,
        attempt: int,
        resolved_retries: int,
        retry_if: Callable[[], bool] | None,
    ) -> bool:
        """Decide whether the loop may run another attempt after `exc`.

        Shared by the sync and async drivers so the two cannot drift. When the
        answer is no, `exc` is marked non-retryable before the caller re-raises so
        an enclosing retry driver does not restart a budget this one already spent.

        Classification is guarded: it probes attributes on a foreign exception, and
        a failure there must surface the model error, not the probe's error.

        Args:
            exc: The failure from the attempt that just completed.
            attempt: 0-indexed attempt that just failed.
            resolved_retries: Retry budget for this call.
            retry_if: Optional guard vetoing a retry (for example once output
                became visible to the user).

        Returns:
            `True` when another attempt should run.
        """
        try:
            retryable = _should_retry_after_failure(exc)
        except Exception:
            logger.exception("Retry classification failed; not retrying")
            retryable = False
        if retryable and attempt < resolved_retries:
            if retry_if is None or retry_if():
                return True
            logger.info("Not retrying: model output was already shown to the user")
        if retryable and attempt >= resolved_retries:
            # Only budget exhaustion needs the marker. A failure classified as
            # permanent already stops every driver, and a `retry_if` veto is
            # request-local -- an outer driver with its own tracker may legitimately
            # reach a different verdict.
            _mark_non_retryable(exc)
        return False

    def run_with_retry(
        self,
        model: BaseChatModel,
        handler: Callable[[], _T],
        *,
        writer: Callable[[ModelRetryEvent], object] | None = None,
        retry_if: Callable[[], bool] | None = None,
        max_retries: int | None = None,
    ) -> _T:
        """Run a model call with retries, using the model's attached budget.

        This is the shared retry driver. It backs both `wrap_model_call` (the
        LangChain model node) and direct model calls made outside that node,
        such as forced conversation summaries. On exhaustion, a non-transient
        error, or a `retry_if` veto, the original model error is re-raised with
        its traceback intact (never wrapped or swallowed).

        Args:
            model: Concrete model used by `handler`.
            handler: Zero-argument callable that performs the model call.
            writer: Optional custom-stream writer for retry status events.
            retry_if: Optional guard checked after a retryable failure. Returning
                `False` prevents another attempt after output becomes visible.
            max_retries: Request-local retry budget, or `None` to use model metadata.

        Returns:
            The successful handler result.

        Raises:
            RuntimeError: If the retry loop exits without returning (unreachable
                in practice; present only as a fail-loud type-checker guard).
        """
        resolved_retries = (
            max_retries if max_retries is not None else self._model_max_retries(model)
        )
        # A negative budget would make `range(resolved_retries + 1)` empty,
        # skipping the model call entirely and hitting the fail-loud guard
        # below. Callers pass validated values today; clamp defensively so the
        # loop always runs at least the initial attempt.
        resolved_retries = max(0, resolved_retries)
        for attempt in range(resolved_retries + 1):
            try:
                return handler()
            except Exception as exc:  # classified by _should_retry_after_failure
                if not self._should_attempt_again(
                    exc,
                    attempt=attempt,
                    resolved_retries=resolved_retries,
                    retry_if=retry_if,
                ):
                    raise
                self._emit_retry_status(writer, attempt + 1, resolved_retries, exc)
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
        retry_if: Callable[[], bool] | None = None,
        max_retries: int | None = None,
    ) -> _T:
        """Asynchronously run a model call with the model's attached budget.

        The async twin of `run_with_retry`: it backs both `awrap_model_call`
        (the LangChain model node) and direct async model calls outside that
        node, such as forced conversation summaries. On exhaustion, a
        non-transient error, or a `retry_if` veto, the original model error is
        re-raised with its traceback intact (never wrapped or swallowed).

        Args:
            model: Concrete model used by `handler`.
            handler: Zero-argument async callable that performs the model call.
            writer: Optional custom-stream writer for retry status events.
            retry_if: Optional guard checked after a retryable failure. Returning
                `False` prevents another attempt after output becomes visible.
            max_retries: Request-local retry budget, or `None` to use model metadata.

        Returns:
            The successful handler result.

        Raises:
            RuntimeError: If the retry loop exits without returning (unreachable
                in practice; present only as a fail-loud type-checker guard).
        """
        import asyncio

        resolved_retries = (
            max_retries if max_retries is not None else self._model_max_retries(model)
        )
        # See `run_with_retry`: clamp so a negative budget cannot skip the call.
        resolved_retries = max(0, resolved_retries)
        for attempt in range(resolved_retries + 1):
            try:
                return await handler()
            except Exception as exc:  # classified by _should_retry_after_failure
                if not self._should_attempt_again(
                    exc,
                    attempt=attempt,
                    resolved_retries=resolved_retries,
                    retry_if=retry_if,
                ):
                    raise
                self._emit_retry_status(writer, attempt + 1, resolved_retries, exc)
                delay = self._compute_delay(attempt)
                if delay > 0:
                    await asyncio.sleep(delay)
        msg = "Unexpected: retry loop completed without returning"
        raise RuntimeError(msg)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Retry the model node on transient errors, surfacing status per retry.

        Args:
            request: Model request to execute (may be re-run on retry).
            handler: Callable that executes the model node.

        Returns:
            The successful `ModelResponse`. Never an error `AIMessage`: failures
                propagate from `run_with_retry` with the original traceback.
        """
        writer = getattr(getattr(request, "runtime", None), "stream_writer", None)
        tracker = _StreamOutputTracker()
        config = _stream_tracking_config(tracker)

        def tracked_handler() -> ModelResponse:
            with set_config_context(config) as context:
                return context.run(handler, request)

        return self.run_with_retry(
            request.model,
            tracked_handler,
            writer=writer,
            retry_if=lambda: not tracker.emitted,
            max_retries=_runtime_model_retry_override(request.runtime),
        )

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Async variant of `wrap_model_call`.

        Args:
            request: Model request to execute (may be re-run on retry).
            handler: Async callable that executes the model node.

        Returns:
            The successful `ModelResponse`. Never an error `AIMessage`: failures
                propagate from `arun_with_retry` with the original traceback.
        """
        import asyncio

        writer = getattr(getattr(request, "runtime", None), "stream_writer", None)
        tracker = _StreamOutputTracker()
        config = _stream_tracking_config(tracker, asynchronous=True)

        async def tracked_handler() -> ModelResponse:
            async def invoke_handler() -> ModelResponse:
                return await handler(request)

            with set_config_context(config) as context:
                task = context.run(lambda: asyncio.create_task(invoke_handler()))
                return await task

        return await self.arun_with_retry(
            request.model,
            tracked_handler,
            writer=writer,
            retry_if=lambda: not tracker.emitted,
            max_retries=_runtime_model_retry_override(request.runtime),
        )
