"""Middleware for runtime model selection via LangGraph config.

Allows switching the model per-invocation by passing the model spec
in `config["configurable"]["model"]` without recompiling the graph.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain.chat_models import init_chat_model

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


def _resolve_model_from_spec(spec: str) -> BaseChatModel:
    """Resolve a model spec string to a chat model instance.

    Handles the `openai:` prefix to enable the Responses API,
    matching the behavior of `deepagents.graph.resolve_model`.

    Args:
        spec: Model specification (e.g., `'anthropic:claude-sonnet-4-6'`).

    Returns:
        Resolved `BaseChatModel` instance.
    """
    if spec.startswith("openai:"):
        return init_chat_model(spec, use_responses_api=True)
    return init_chat_model(spec)


class ConfigurableModelMiddleware(AgentMiddleware):
    """Swap the model at runtime based on `config["configurable"]["model"]`.

    When the configurable key is absent, the graph's original model is used.
    This middleware should be placed early in the middleware stack so that
    downstream middleware (e.g., prompt caching, summarization) sees the
    correct model.

    Example:
        ```python
        agent = create_deep_agent(model="anthropic:claude-sonnet-4-6", ...)

        # Switch model at runtime — no graph recompilation needed:
        agent.ainvoke(
            {"messages": [...]},
            config={"configurable": {"model": "openai:gpt-4o"}},
        )
        ```
    """

    @staticmethod
    def _get_configurable(request: ModelRequest) -> dict[str, Any]:
        """Extract `configurable` dict from the runtime config.

        Args:
            request: The current model request.

        Returns:
            The `configurable` sub-dict (empty dict if absent).
        """
        config = getattr(request.runtime, "config", None) or {}
        return config.get("configurable", {})

    def _get_override_model(self, request: ModelRequest) -> BaseChatModel | None:
        """Read the model override from runtime config, if present.

        Args:
            request: The current model request.

        Returns:
            A resolved `BaseChatModel` if an override is specified, else `None`.
        """
        configurable = self._get_configurable(request)
        model_spec = configurable.get("model")
        if model_spec is None:
            return None

        current = getattr(request.model, "model_name", None) or getattr(request.model, "model", None)
        if model_spec == current:
            return None
        # Handle provider-prefixed specs (e.g., "anthropic:claude-sonnet-4-6")
        # matching model_name without the prefix ("claude-sonnet-4-6").
        if ":" in model_spec and model_spec.split(":", 1)[1] == current:
            return None

        logger.debug("Overriding model to %s (was %s)", model_spec, current)
        return _resolve_model_from_spec(model_spec)

    @staticmethod
    def _get_model_params(request: ModelRequest) -> dict[str, Any] | None:
        """Read invocation param overrides from runtime config.

        Args:
            request: The current model request.

        Returns:
            A non-empty dict of params to merge into `model_settings`,
            or `None` if no overrides are configured.
        """
        configurable = ConfigurableModelMiddleware._get_configurable(request)
        params: dict[str, Any] | None = configurable.get("model_params")
        return params or None

    def _apply_overrides(self, request: ModelRequest) -> ModelRequest:
        """Apply model and param overrides from runtime config.

        Args:
            request: The original model request.

        Returns:
            A (possibly new) request with model and/or `model_settings` overrides
            applied. Returns the original request unchanged if no overrides are
            configured.
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
        """Async version of `wrap_model_call`.

        Args:
            request: The model request.
            handler: The next async handler in the middleware chain.

        Returns:
            The model response from the handler.
        """
        return await handler(self._apply_overrides(request))
