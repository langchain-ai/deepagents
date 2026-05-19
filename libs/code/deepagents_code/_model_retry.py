"""Transient-error retry layer for chat model clients.

Some providers (notably Fireworks) periodically reject requests with a
transient HTTP 503 ``service overloaded, please try again later`` response
or drop streaming connections mid-flight (raising ``httpx.ReadError`` /
``httpx.ReadTimeout``). Without retry, those momentary blips propagate
straight up through every middleware (``TodoListMiddleware`` →
``FilesystemMiddleware`` → ``SubAgentMiddleware`` → … → the agent) and
end the user's task with no output.

This module installs a bounded retry loop on the chat-model instance
itself so every middleware-wrapped call inherits the same recovery. We
only retry on a narrow allowlist of *transient* error classes — 4xx
schema/auth errors (``InvalidRequestError``, ``BadRequestResponseError``,
``NotFoundResponseError``) are deterministic and must surface immediately.

We intentionally patch the model's bound methods in place instead of
calling ``Runnable.with_retry``, because the latter returns a
``RunnableRetry`` wrapper that no longer exposes ``bind_tools`` / the
``profile`` attribute that downstream middleware relies on.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import AsyncIterator, Callable, Iterator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

# Bounded exponential backoff with jitter. Tuned to absorb a handful of
# back-to-back 503s without keeping the user staring at a frozen UI for
# too long.
MAX_ATTEMPTS = 4
"""Maximum number of attempts (initial call + 3 retries)."""

BASE_DELAY_SECONDS = 1.0
"""Initial backoff between retries."""

MAX_DELAY_SECONDS = 30.0
"""Cap on the per-attempt sleep."""

# Substring shown by Fireworks when its inference fleet is overloaded.
# Some SDK versions raise this through a generic httpx ``HTTPStatusError``
# rather than the typed ``ServiceUnavailableError``, so we also match on
# the body text.
_FIREWORKS_OVERLOADED_TEXT = "service overloaded, please try again later"


def _is_transient_error(exc: BaseException) -> bool:
    """Return True if *exc* is a transient provider/network error.

    Matches:

    - ``fireworks.client.error.ServiceUnavailableError`` (typed 503 from
      the Fireworks SDK).
    - Any exception whose class name ends in ``ServiceUnavailableError``
      (defensive — other providers reuse the name).
    - HTTP 503 responses, including ones surfaced as
      ``httpx.HTTPStatusError`` with the Fireworks "service overloaded"
      body.
    - ``httpx.ReadError`` / ``httpx.ReadTimeout`` raised when a streaming
      connection is dropped mid-response.

    Does *not* match deterministic 4xx errors
    (``InvalidRequestError``, ``BadRequestResponseError``,
    ``NotFoundResponseError``, etc.) — those are caller bugs that must
    surface immediately.
    """
    exc_name = type(exc).__name__

    # httpx transport-level failures during streaming. Importing httpx
    # lazily because it may not be installed in minimal test envs.
    try:
        import httpx
    except ImportError:  # pragma: no cover - httpx is a langchain transitive dep
        httpx = None  # type: ignore[assignment]

    if httpx is not None and isinstance(exc, (httpx.ReadError, httpx.ReadTimeout)):
        return True

    # Typed Fireworks SDK error. Imported lazily — Fireworks is an
    # optional extra and may not be installed.
    try:
        from fireworks.client.error import (  # type: ignore[import-not-found]
            ServiceUnavailableError as _FireworksServiceUnavailable,
        )
    except ImportError:  # pragma: no cover - optional dependency
        _FireworksServiceUnavailable = None  # type: ignore[assignment]

    if _FireworksServiceUnavailable is not None and isinstance(
        exc, _FireworksServiceUnavailable
    ):
        return True

    # Defensive: another SDK might raise a class also named
    # ``ServiceUnavailableError``. Match on the class name so we cover
    # the same semantic without importing every provider.
    if exc_name.endswith("ServiceUnavailableError"):
        return True

    # Status-bearing responses (httpx.HTTPStatusError, openai-style
    # APIStatusError, etc.). We dig out the status code defensively.
    status_code = _extract_status_code(exc)
    if status_code == 503:
        return True

    # Last resort: match the Fireworks overload body text on whatever
    # message the SDK surfaced. This catches SDK versions that raise a
    # bare ``Exception`` or a generic httpx error with the body inline.
    message = str(exc).lower()
    if _FIREWORKS_OVERLOADED_TEXT in message:
        return True

    return False


def _extract_status_code(exc: BaseException) -> int | None:
    """Best-effort extraction of an HTTP status code from *exc*."""
    # httpx.HTTPStatusError carries `.response.status_code`.
    response = getattr(exc, "response", None)
    if response is not None:
        status = getattr(response, "status_code", None)
        if isinstance(status, int):
            return status
    # openai/anthropic-style errors expose `.status_code` directly.
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    return None


def _retry_sync(func: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a sync callable with bounded exponential-backoff retry."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        from tenacity import (
            RetryError,
            Retrying,
            retry_if_exception,
            stop_after_attempt,
            wait_exponential_jitter,
        )

        retrying = Retrying(
            reraise=True,
            stop=stop_after_attempt(MAX_ATTEMPTS),
            wait=wait_exponential_jitter(
                initial=BASE_DELAY_SECONDS, max=MAX_DELAY_SECONDS
            ),
            retry=retry_if_exception(_is_transient_error),
            before_sleep=_log_retry,
        )
        try:
            for attempt in retrying:
                with attempt:
                    return func(*args, **kwargs)
        except RetryError as exc:  # pragma: no cover - reraise=True covers this
            raise exc.last_attempt.exception()  # type: ignore[misc]
        return None  # pragma: no cover - unreachable

    return wrapper


def _retry_async(func: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap an async callable with bounded exponential-backoff retry."""

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        from tenacity import (
            AsyncRetrying,
            RetryError,
            retry_if_exception,
            stop_after_attempt,
            wait_exponential_jitter,
        )

        retrying = AsyncRetrying(
            reraise=True,
            stop=stop_after_attempt(MAX_ATTEMPTS),
            wait=wait_exponential_jitter(
                initial=BASE_DELAY_SECONDS, max=MAX_DELAY_SECONDS
            ),
            retry=retry_if_exception(_is_transient_error),
            before_sleep=_log_retry,
        )
        try:
            async for attempt in retrying:
                with attempt:
                    return await func(*args, **kwargs)
        except RetryError as exc:  # pragma: no cover - reraise=True covers this
            raise exc.last_attempt.exception()  # type: ignore[misc]
        return None  # pragma: no cover - unreachable

    return wrapper


def _retry_sync_iter(func: Callable[..., Iterator[Any]]) -> Callable[..., Iterator[Any]]:
    """Wrap a sync-iterator callable with retry that covers connect-time errors.

    We only retry until the first chunk is produced — once streaming has
    started, partial output has already reached the caller and a retry
    would duplicate it. ``httpx.ReadError`` mid-stream therefore still
    surfaces, but the much more common case (the request itself fails
    with a 503 before any tokens flow) is recoverable.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Iterator[Any]:
        from tenacity import (
            RetryError,
            Retrying,
            retry_if_exception,
            stop_after_attempt,
            wait_exponential_jitter,
        )

        retrying = Retrying(
            reraise=True,
            stop=stop_after_attempt(MAX_ATTEMPTS),
            wait=wait_exponential_jitter(
                initial=BASE_DELAY_SECONDS, max=MAX_DELAY_SECONDS
            ),
            retry=retry_if_exception(_is_transient_error),
            before_sleep=_log_retry,
        )
        try:
            for attempt in retrying:
                with attempt:
                    iterator = func(*args, **kwargs)
                    # Materialize the first item inside the retry block so
                    # connect-time 503s trigger another attempt.
                    try:
                        first = next(iterator)
                    except StopIteration:
                        return iter(())
                    return _chain_first(first, iterator)
        except RetryError as exc:  # pragma: no cover - reraise=True covers this
            raise exc.last_attempt.exception()  # type: ignore[misc]
        return iter(())  # pragma: no cover - unreachable

    return wrapper


def _retry_async_iter(
    func: Callable[..., AsyncIterator[Any]],
) -> Callable[..., AsyncIterator[Any]]:
    """Async-iterator variant of :func:`_retry_sync_iter`."""

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        from tenacity import (
            AsyncRetrying,
            RetryError,
            retry_if_exception,
            stop_after_attempt,
            wait_exponential_jitter,
        )

        retrying = AsyncRetrying(
            reraise=True,
            stop=stop_after_attempt(MAX_ATTEMPTS),
            wait=wait_exponential_jitter(
                initial=BASE_DELAY_SECONDS, max=MAX_DELAY_SECONDS
            ),
            retry=retry_if_exception(_is_transient_error),
            before_sleep=_log_retry,
        )
        try:
            async for attempt in retrying:
                with attempt:
                    iterator = func(*args, **kwargs)
                    try:
                        first = await iterator.__anext__()
                    except StopAsyncIteration:
                        return _empty_async_iter()
                    return _chain_first_async(first, iterator)
        except RetryError as exc:  # pragma: no cover - reraise=True covers this
            raise exc.last_attempt.exception()  # type: ignore[misc]
        return _empty_async_iter()  # pragma: no cover - unreachable

    return wrapper


def _chain_first(first: Any, rest: Iterator[Any]) -> Iterator[Any]:
    yield first
    yield from rest


async def _chain_first_async(first: Any, rest: AsyncIterator[Any]) -> AsyncIterator[Any]:
    yield first
    async for chunk in rest:
        yield chunk


async def _empty_async_iter() -> AsyncIterator[Any]:
    if False:  # pragma: no cover - empty async generator
        yield None


def _log_retry(retry_state: Any) -> None:
    """Tenacity ``before_sleep`` hook — log each retry at WARNING level."""
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    sleep_for = getattr(retry_state.next_action, "sleep", None)
    logger.warning(
        "Transient model error on attempt %s/%s (%s: %s); retrying in %.2fs",
        retry_state.attempt_number,
        MAX_ATTEMPTS,
        type(exc).__name__ if exc else "?",
        exc,
        sleep_for if sleep_for is not None else 0.0,
    )


_RETRY_INSTALLED_ATTR = "_deepagents_code_transient_retry_installed"


def install_transient_retry(model: BaseChatModel) -> BaseChatModel:
    """Patch *model* in place so transient errors trigger bounded retry.

    The retry sits at the LLM client layer, so every middleware-wrapped
    call (``TodoListMiddleware``, ``FilesystemMiddleware``,
    ``SubAgentMiddleware``, ``SummarizationMiddleware``,
    ``ConfigurableModelMiddleware``, ``AnthropicPromptCachingMiddleware``,
    …) inherits the same recovery without needing to know about it.

    Idempotent — calling this twice on the same instance is a no-op so
    repeated calls from session-resumption code paths don't compound
    backoff.

    Returns the same instance for call-site fluency.
    """
    if getattr(model, _RETRY_INSTALLED_ATTR, False):
        return model

    for sync_name in ("invoke", "_generate"):
        method = getattr(model, sync_name, None)
        if callable(method):
            try:
                setattr(model, sync_name, _retry_sync(method))
            except (AttributeError, TypeError):
                # Some chat-model classes use ``__slots__`` or otherwise
                # forbid attribute assignment. Fail open — the user
                # still gets the un-retried client rather than a crash
                # at startup.
                logger.debug(
                    "Could not install sync retry on %s.%s; skipping",
                    type(model).__name__,
                    sync_name,
                )

    for async_name in ("ainvoke", "_agenerate"):
        method = getattr(model, async_name, None)
        if callable(method):
            try:
                setattr(model, async_name, _retry_async(method))
            except (AttributeError, TypeError):
                logger.debug(
                    "Could not install async retry on %s.%s; skipping",
                    type(model).__name__,
                    async_name,
                )

    stream = getattr(model, "stream", None)
    if callable(stream):
        try:
            model.stream = _retry_sync_iter(stream)  # type: ignore[method-assign]
        except (AttributeError, TypeError):
            logger.debug("Could not install retry on %s.stream", type(model).__name__)

    astream = getattr(model, "astream", None)
    if callable(astream):
        try:
            model.astream = _retry_async_iter(astream)  # type: ignore[method-assign]
        except (AttributeError, TypeError):
            logger.debug("Could not install retry on %s.astream", type(model).__name__)

    try:
        setattr(model, _RETRY_INSTALLED_ATTR, True)
    except (AttributeError, TypeError):
        # Sentinel can't be stored — idempotency degrades to "patch
        # again, but the wrappers are themselves transparent on success
        # so the user-visible behavior is unchanged."
        logger.debug(
            "Could not record retry-installed sentinel on %s", type(model).__name__
        )

    return model
