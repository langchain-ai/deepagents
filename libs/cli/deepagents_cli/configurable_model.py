"""CLI middleware for runtime model selection via LangGraph runtime context.

Allows switching the model per invocation by passing a ``CLIContext``
via ``context=`` on ``agent.astream()`` / ``agent.invoke()`` without
recompiling the graph.

Per-invocation model settings can be merged from
``CLIContext.model_params``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from deepagents._models import model_matches_spec, resolve_model  # noqa: PLC2701
from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain_core.language_models import BaseChatModel


logger = logging.getLogger(__name__)


@dataclass
class CLIContext:
    """Runtime context passed via ``context=`` to the LangGraph graph.

    Carries per-invocation overrides that ``ConfigurableModelMiddleware``
    reads from ``request.runtime.context``.
    """

    model: str | None = None
    """Model spec to swap at runtime (e.g. ``"openai:gpt-4o"``)."""

    model_params: dict[str, Any] = field(default_factory=dict)
    """Invocation params (e.g. ``temperature``, ``max_tokens``) to merge
    into ``model_settings``."""


class ConfigurableModelMiddleware(AgentMiddleware):
    """Swap the model or per-call settings from ``runtime.context``."""

    _deepagents_prepend = True

    @staticmethod
    def _get_context(request: ModelRequest) -> CLIContext | None:
        """Extract ``CLIContext`` from the runtime, if present.

        Args:
            request: The current model request.

        Returns:
            The ``CLIContext`` instance, or ``None`` when unavailable.
        """
        runtime = request.runtime
        if runtime is None:
            return None
        ctx = runtime.context
        if isinstance(ctx, CLIContext):
            return ctx
        return None

    def _get_override_model(self, request: ModelRequest) -> BaseChatModel | None:
        """Read the model override from runtime context, if present.

        Args:
            request: The current model request.

        Returns:
            A resolved ``BaseChatModel`` if an override is specified, else ``None``.
        """
        ctx = self._get_context(request)
        if ctx is None or ctx.model is None:
            return None

        if model_matches_spec(request.model, ctx.model):
            return None

        logger.debug("Overriding model to %s", ctx.model)
        return resolve_model(ctx.model)

    def _get_model_params(self, request: ModelRequest) -> dict[str, Any] | None:
        """Read invocation param overrides from runtime context.

        Args:
            request: The current model request.

        Returns:
            A non-empty dict of params to merge into ``model_settings``,
            or ``None`` if no overrides are configured.
        """
        ctx = self._get_context(request)
        if ctx is None or not ctx.model_params:
            return None
        return ctx.model_params

    def _apply_overrides(self, request: ModelRequest) -> ModelRequest:
        """Apply model and param overrides from runtime context.

        Args:
            request: The original model request.

        Returns:
            A (possibly new) request with model and/or ``model_settings``
            overrides applied.  Returns the original request unchanged if
            no overrides are configured.
        """
        overrides: dict[str, Any] = {}

        override_model = self._get_override_model(request)
        if override_model is not None:
            overrides["model"] = override_model

        params = self._get_model_params(request)
        if params is not None:
            merged = {**request.model_settings, **params}
            overrides["model_settings"] = merged

        if overrides:
            return request.override(**overrides)
        return request

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Swap the model / merge invocation params before the handler executes.

        Args:
            request: The model request.
            handler: The next handler in the middleware chain.

        Returns:
            The model response from the handler.
        """
        return handler(self._apply_overrides(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Async version of ``wrap_model_call``.

        Args:
            request: The model request.
            handler: The next async handler in the middleware chain.

        Returns:
            The model response from the handler.
        """
        return await handler(self._apply_overrides(request))
