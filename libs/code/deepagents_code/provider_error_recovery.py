"""Middleware that converts raw provider exceptions into user-facing messages.

Provider SDKs (`openai`, `langchain_fireworks`, etc.) raise transport-level
errors directly through the model call. Without this middleware, a 403
`PermissionDeniedError` or a transient 5xx propagates out of every downstream
middleware and aborts the run with no catch — the user sees an empty AI bubble
and no signal of what went wrong.

This middleware sits just inside `ConfigurableModelMiddleware` so it sees raw
provider exceptions but its translated `AIMessage` reaches all outer middleware
(TodoListMiddleware, SubAgentMiddleware, etc.) as a normal completion.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


logger = logging.getLogger(__name__)


_AUTH_STATUS_RE = re.compile(r"Error code:\s*40[13]\b")
_AUTH_TEXT_RE = re.compile(r"\b(Forbidden|Unauthorized)\b", re.IGNORECASE)
_TRANSIENT_STATUS_RE = re.compile(r"Error code:\s*5\d{2}\b")

_TRANSIENT_RETRY_BACKOFF_S: float = 1.0


def _exception_chain(exc: BaseException) -> list[BaseException]:
    """Walk `__cause__`/`__context__` so wrapped provider errors are visible.

    Returns:
        The exception itself plus every cause/context found by walking the chain.
    """
    seen: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and current not in seen:
        seen.append(current)
        current = current.__cause__ or current.__context__
    return seen


def _matches_class(exc: BaseException, dotted: str) -> bool:
    """Return True when any class in `exc`'s MRO matches `module.ClassName`."""
    module, _, name = dotted.rpartition(".")
    for cls in type(exc).__mro__:
        if cls.__name__ == name and (not module or cls.__module__.startswith(module)):
            return True
    return False


def _classify(exc: BaseException) -> str:
    """Classify a provider exception as 'auth', 'transient', or 'other'.

    'auth' = 401/403 — never retry, surface to user.
    'transient' = 5xx / connection / read error — one-shot retry.

    Returns:
        One of `'auth'`, `'transient'`, `'other'`.
    """
    for inner in _exception_chain(exc):
        message = str(inner)
        if (
            _matches_class(inner, "openai.PermissionDeniedError")
            or _matches_class(inner, "openai.AuthenticationError")
            or _AUTH_STATUS_RE.search(message)
            or _AUTH_TEXT_RE.search(message)
        ):
            return "auth"
        if (
            _matches_class(inner, "openai.APIConnectionError")
            or _matches_class(inner, "httpx.ConnectError")
            or _matches_class(inner, "httpx.ReadError")
            or _matches_class(inner, "fireworks.ServiceUnavailableError")
            or _TRANSIENT_STATUS_RE.search(message)
        ):
            return "transient"
    return "other"


def _model_id(request: ModelRequest) -> str:
    """Best-effort human-readable model id for error messages.

    Returns:
        The first non-empty `model_name` / `model` / `model_id` attribute, or
            the model's class name as a last-resort label.
    """
    model = request.model
    for attr in ("model_name", "model", "model_id"):
        value = getattr(model, attr, None)
        if isinstance(value, str) and value:
            return value
    return type(model).__name__


def _fallback_spec(request: ModelRequest) -> str | None:
    """Return a `fallback_model` spec from `runtime.context`, if configured."""
    runtime = request.runtime
    if runtime is None:
        return None
    ctx = getattr(runtime, "context", None)
    if not isinstance(ctx, dict):
        return None
    spec = ctx.get("fallback_model")
    return spec if isinstance(spec, str) and spec else None


def _swap_to_fallback(request: ModelRequest, spec: str) -> ModelRequest | None:
    """Resolve `spec` via `create_model` and return a request bound to it.

    Returns:
        A new request with `model` overridden, or `None` when `spec` cannot be
            resolved (missing creds, unknown provider, etc.).
    """
    from deepagents_code.config import create_model
    from deepagents_code.model_config import ModelConfigError

    try:
        result = create_model(spec)
    except ModelConfigError:
        logger.exception("Failed to resolve fallback model '%s'", spec)
        return None
    return request.override(model=result.model)


def _auth_error_message(model_id: str) -> str:
    return (
        f"Your configured model `{model_id}` returned 403 Forbidden. "
        "This usually means your API key doesn't have access to this model. "
        "Switch models with /model or check your provider account."
    )


def _transient_error_message(model_id: str) -> str:
    return (
        f"Your configured model `{model_id}` is temporarily unavailable "
        "(provider returned a 5xx / connection error after one retry). "
        "Try again in a moment, or switch models with /model."
    )


def _error_response(text: str) -> ModelResponse[Any]:
    return ModelResponse(result=[AIMessage(content=text)])


class ProviderErrorRecoveryMiddleware(AgentMiddleware):
    """Catch raw provider exceptions and surface them as assistant messages.

    - 401/403 → no retry; optional one-shot fallback model; user-facing message.
    - 5xx / connection errors → one-shot retry with 1s backoff, then typed
      "temporarily unavailable" message.
    - Anything else propagates unchanged.
    """

    def wrap_model_call(  # noqa: PLR6301
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Sync variant of the recovery wrapper.

        Returns:
            The downstream `ModelResponse`, or a synthetic `AIMessage` when a
                provider auth/transient error is caught and translated.
        """
        try:
            return handler(request)
        except Exception as exc:
            kind = _classify(exc)
            if kind == "auth":
                logger.warning("Provider auth error for %s", _model_id(request))
                spec = _fallback_spec(request)
                if spec:
                    swapped = _swap_to_fallback(request, spec)
                    if swapped is not None:
                        try:
                            return handler(swapped)
                        except Exception:
                            logger.exception("Fallback model '%s' also failed", spec)
                return _error_response(_auth_error_message(_model_id(request)))
            if kind == "transient":
                logger.warning(
                    "Transient provider error for %s; retrying once",
                    _model_id(request),
                )
                time.sleep(_TRANSIENT_RETRY_BACKOFF_S)
                try:
                    return handler(request)
                except Exception:
                    logger.exception("Retry after transient error failed")
                    return _error_response(_transient_error_message(_model_id(request)))
            raise

    async def awrap_model_call(  # noqa: PLR6301
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Async variant of the recovery wrapper.

        Returns:
            The downstream `ModelResponse`, or a synthetic `AIMessage` when a
                provider auth/transient error is caught and translated.
        """
        try:
            return await handler(request)
        except Exception as exc:
            kind = _classify(exc)
            if kind == "auth":
                logger.warning("Provider auth error for %s", _model_id(request))
                spec = _fallback_spec(request)
                if spec:
                    swapped = _swap_to_fallback(request, spec)
                    if swapped is not None:
                        try:
                            return await handler(swapped)
                        except Exception:
                            logger.exception("Fallback model '%s' also failed", spec)
                return _error_response(_auth_error_message(_model_id(request)))
            if kind == "transient":
                logger.warning(
                    "Transient provider error for %s; retrying once",
                    _model_id(request),
                )
                await asyncio.sleep(_TRANSIENT_RETRY_BACKOFF_S)
                try:
                    return await handler(request)
                except Exception:
                    logger.exception("Retry after transient error failed")
                    return _error_response(_transient_error_message(_model_id(request)))
            raise
