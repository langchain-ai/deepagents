"""Model fallback chain for Talon's agent graphs.

Talon is an experimental runtime and is subject to change or removal at any time.

`TalonModelFallbackMiddleware` retries a model call on transient provider errors and,
once the retry budget for that model is spent, moves on to the next model configured in
`DEEPAGENTS_TALON_MODEL_FALLBACKS`. LangChain's `ModelFallbackMiddleware` is not used
because it falls back on every exception, including authentication and malformed-request
errors that another model cannot fix, and because it builds fallback models with a bare
`init_chat_model` call that skips Talon's provider profiles, base URL, and context size.

Talon's middleware lands ahead of the Deep Agents prompt-caching middleware, so caching
runs again for each attempt against the model actually being called: a fallback from an
Anthropic model to another provider never carries Anthropic cache markers.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from deepagents._models import get_model_identifier, get_model_provider, model_matches_spec
from langchain.agents.middleware import AgentMiddleware
from langchain_core.exceptions import ContextOverflowError
from langgraph.errors import GraphBubbleUp

from deepagents_talon.messaging import MESSAGE_HANDLER

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping, Sequence

    from langchain.agents.middleware.types import ModelRequest, ModelResponse
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

FALLBACKS_ENV_KEY = "DEEPAGENTS_TALON_MODEL_FALLBACKS"

_BAD_REQUEST_STATUS_CODE = 400
_RETRYABLE_STATUS_CODES = frozenset({408, 409, 413, 429, 500, 502, 503, 504})
_CONTEXT_MARKERS = (
    "context length",
    "context window",
    "context limit",
    "maximum context",
    "max context",
    "input too long",
    "request too large",
)
_RETRYABLE_BAD_REQUEST_MARKERS = (
    "failed to parse",
    "tool_call",
    "tool call",
    *_CONTEXT_MARKERS,
)
_RETRYABLE_MESSAGE_MARKERS = (
    *_RETRYABLE_BAD_REQUEST_MARKERS,
    "connection aborted",
    "connection closed",
    "connection lost",
    "connection refused",
    "connection reset",
    "connection timed out",
    "read timeout",
    "timed out",
    "timeout limit",
    "temporarily unavailable",
    "temporary failure",
    "try again later",
)
_MAX_BACKOFF_SECONDS = 10


class ModelFallbackExhaustedError(RuntimeError):
    """Every model in the fallback chain failed with a transient error.

    Each model already spent its retry budget, so the runtime does not retry the
    whole turn again on top of that.
    """


@dataclass(slots=True)
class FallbackTurn:
    """Fallback progress shared by every model call in one Talon turn.

    Args:
        served: Chain position that last answered, keyed by the chain's primary model.
            Later calls in the turn start there instead of paying the full retry cost
            on a model that already failed.
        notified: Whether the chat was already told a fallback answered.
    """

    served: dict[str, int] = field(default_factory=dict)
    notified: bool = False


FALLBACK_TURN: contextvars.ContextVar[FallbackTurn | None] = contextvars.ContextVar(
    "talon_fallback_turn", default=None
)


class _ModelUnavailableError(Exception):
    """One model spent its retry budget; the chain should try the next one."""

    def __init__(self, error: Exception) -> None:
        super().__init__(str(error))
        self.error = error


def fallback_specs_from_env(env: Mapping[str, str]) -> tuple[str, ...]:
    """Parse and syntax-check the configured fallback chain.

    Only the `provider:model` shape is checked here. Models are built the first time
    the chain reaches them, so a missing provider package or credential surfaces as
    that fallback failing rather than as a startup error.

    Args:
        env: Process environment to read.

    Returns:
        Fallback specs in the order they are tried.

    Raises:
        ValueError: If an entry is not a `provider:model` spec.
    """
    # Split inline rather than reuse `channels.base.split_csv`: importing the channels
    # package loads every chat platform client into the runtime.
    specs = [item.strip() for item in env.get(FALLBACKS_ENV_KEY, "").split(",") if item.strip()]
    for spec in specs:
        provider, separator, name = spec.partition(":")
        if not separator or not provider.strip() or not name.strip():
            msg = f"{FALLBACKS_ENV_KEY} entries must be provider:model specs"
            raise ValueError(msg)
    return tuple(specs)


class TalonModelFallbackMiddleware(AgentMiddleware):
    """Retry a model on transient errors, then walk the configured fallback chain.

    Args:
        specs: Fallback model specs, tried in order after the request's own model.
        build: Builds a chat model from a spec with Talon's model settings applied.
        max_retries: Attempts per model before moving to the next one.
    """

    def __init__(
        self,
        specs: Sequence[str],
        build: Callable[[str], BaseChatModel],
        *,
        max_retries: int,
    ) -> None:
        """Keep the chain; fallback models are built on first use and cached."""
        self._specs = tuple(specs)
        self._build = build
        self._max_retries = max_retries
        self._models: dict[str, BaseChatModel] = {}

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Call the request's model, falling back when it stays unavailable."""
        turn, key, chain = self._start(request)
        last: Exception | None = None
        for index in range(turn.served.get(key, 0), len(chain)):
            attempt = self._request_for(request, chain, index)
            if isinstance(attempt, Exception):
                last = attempt
                continue
            try:
                response = await self._acall(attempt, handler)
            except _ModelUnavailableError as unavailable:
                last = unavailable.error
                continue
            await self._arecord(turn, key, index, attempt.model)
            return response
        msg = "Every model in the fallback chain failed"
        raise ModelFallbackExhaustedError(msg) from last

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Call the request's model, falling back when it stays unavailable."""
        turn, key, chain = self._start(request)
        last: Exception | None = None
        for index in range(turn.served.get(key, 0), len(chain)):
            attempt = self._request_for(request, chain, index)
            if isinstance(attempt, Exception):
                last = attempt
                continue
            try:
                response = self._call(attempt, handler)
            except _ModelUnavailableError as unavailable:
                last = unavailable.error
                continue
            self._note_fallback(turn, key, index, attempt.model)
            return response
        msg = "Every model in the fallback chain failed"
        raise ModelFallbackExhaustedError(msg) from last

    def _start(self, request: ModelRequest) -> tuple[FallbackTurn, str, list[str | None]]:
        turn = FALLBACK_TURN.get() or FallbackTurn()
        fallbacks = [spec for spec in self._specs if not model_matches_spec(request.model, spec)]
        return turn, _model_key(request.model), [None, *fallbacks]

    def _request_for(
        self, request: ModelRequest, chain: Sequence[str | None], index: int
    ) -> ModelRequest | Exception:
        spec = chain[index]
        if spec is None:
            return request
        try:
            model = self._model(spec)
        except Exception as error:  # noqa: BLE001  # an unbuildable fallback counts as failing
            logger.warning("Could not build fallback model %s", spec, exc_info=True)
            return error
        return request.override(model=model)

    def _model(self, spec: str) -> BaseChatModel:
        model = self._models.get(spec)
        if model is None:
            model = self._build(spec)
            self._models[spec] = model
        return model

    async def _acall(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        for attempt in range(self._max_retries):
            try:
                return await handler(request)
            except Exception as error:  # noqa: BLE001  # classified; non-transient errors re-raise
                self._raise_unless_transient(error, request, attempt)
                await asyncio.sleep(_backoff(attempt))
        msg = "model retry loop exited unexpectedly"
        raise RuntimeError(msg)

    def _call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        for attempt in range(self._max_retries):
            try:
                return handler(request)
            except Exception as error:  # noqa: BLE001  # classified; non-transient errors re-raise
                self._raise_unless_transient(error, request, attempt)
                time.sleep(_backoff(attempt))
        msg = "model retry loop exited unexpectedly"
        raise RuntimeError(msg)

    def _raise_unless_transient(
        self, error: Exception, request: ModelRequest, attempt: int
    ) -> None:
        """Re-raise errors another attempt cannot fix; signal when this model is spent."""
        if not _falls_back(error):
            raise error
        name = get_model_identifier(request.model) or type(request.model).__name__
        if attempt + 1 >= self._max_retries:
            logger.warning(
                "Model %s is unavailable after %d attempts: %s", name, attempt + 1, error
            )
            raise _ModelUnavailableError(error) from error
        logger.warning("Retryable error from model %s; retrying: %s", name, error)

    async def _arecord(
        self, turn: FallbackTurn, key: str, index: int, model: BaseChatModel
    ) -> None:
        if not self._note_fallback(turn, key, index, model):
            return
        handler = MESSAGE_HANDLER.get()
        if handler is None:
            return
        name = get_model_identifier(model) or type(model).__name__
        try:
            await handler(f"Primary model unavailable; answering with `{name}`.")
        except Exception:  # noqa: BLE001  # a lost notice must not fail the answered call
            logger.warning("Could not deliver the model fallback notice")

    @staticmethod
    def _note_fallback(turn: FallbackTurn, key: str, index: int, model: BaseChatModel) -> bool:
        """Remember a fallback answered; return whether the chat still needs a notice."""
        if index == 0:
            return False
        if turn.served.get(key) != index:
            name = get_model_identifier(model) or type(model).__name__
            logger.warning("Model fallback %s answered in place of the primary model", name)
        turn.served[key] = index
        notify = not turn.notified
        turn.notified = True
        return notify


def is_retryable(exc: Exception) -> bool:
    """Classify an error as transient, so the same request may succeed if repeated.

    Args:
        exc: Error raised by a graph invocation or a model call.

    Returns:
        Whether repeating the work might succeed.
    """
    if isinstance(exc, ModelFallbackExhaustedError):
        return False
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True

    text = str(exc).lower()
    status_code = status_code_of(exc)
    if status_code in _RETRYABLE_STATUS_CODES:
        return True
    if status_code == _BAD_REQUEST_STATUS_CODE:
        return _contains_marker(text, _RETRYABLE_BAD_REQUEST_MARKERS)
    return _contains_marker(text, _RETRYABLE_MESSAGE_MARKERS)


def status_code_of(exc: BaseException) -> int | None:
    """Find an HTTP status code on a provider error or its response.

    Args:
        exc: Error to inspect, including exception groups.

    Returns:
        The first status code found, or `None`.
    """
    for source in (exc, getattr(exc, "response", None)):
        if source is None:
            continue
        for attr in ("status_code", "status"):
            value = getattr(source, attr, None)
            if isinstance(value, int):
                return value
    if isinstance(exc, BaseExceptionGroup):
        for item in exc.exceptions:
            value = status_code_of(item)
            if value is not None:
                return value
    return None


def _falls_back(error: Exception) -> bool:
    """Whether a model error is worth retrying here and then handing to another model.

    Context overflows are left alone: the outer summarization middleware recovers from
    them by compacting the conversation, which it cannot do if this layer swallows them.
    """
    if isinstance(error, (GraphBubbleUp, ContextOverflowError)):
        return False
    if _contains_marker(str(error).lower(), _CONTEXT_MARKERS):
        return False
    return is_retryable(error)


def _model_key(model: BaseChatModel) -> str:
    return f"{get_model_provider(model)}:{get_model_identifier(model) or type(model).__name__}"


def _backoff(attempt: int) -> int:
    return min(2**attempt, _MAX_BACKOFF_SECONDS)


def _contains_marker(text: str, markers: Sequence[str]) -> bool:
    return any(marker in text for marker in markers)
