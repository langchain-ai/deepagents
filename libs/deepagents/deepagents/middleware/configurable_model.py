"""Middleware for runtime model selection via LangGraph config.

Allows switching the model per-invocation by passing the model spec
in `config["configurable"]["model"]` without recompiling the graph.

Per-invocation model settings can be merged from
`config["configurable"]["model_params"]`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol, TypedDict, cast

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langgraph.config import get_config

from deepagents._models import model_matches_spec, resolve_model

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain_core.language_models import BaseChatModel
    from langchain_core.runnables import RunnableConfig


logger = logging.getLogger(__name__)


class _RuntimeWithConfig(Protocol):
    """Protocol for runtimes that expose a `config` attribute."""

    config: RunnableConfig | None


class _ConfigurableModelConfig(TypedDict, total=False):
    """Supported `config["configurable"]` keys for model overrides."""

    model: str
    model_params: dict[str, Any]


class ConfigurableModelMiddleware(AgentMiddleware):
    """Swap the model or per-call settings from `config["configurable"]`.

    When the configurable key is absent, the graph's original model and request
    settings are used.

    Supported keys:

    - `model`: provider-prefixed model spec, resolved with the same rules as
        `create_deep_agent()`
    - `model_params`: per-invocation settings merged into
        `ModelRequest.model_settings`

    This middleware should be placed early in the middleware stack so that
    downstream middleware (e.g., prompt caching, summarization) sees the
    correct model.

    Example:
        ```python
        agent = create_deep_agent(model="anthropic:claude-sonnet-4-6", ...)

        # Switch model and tune per-call settings at runtime:
        agent.ainvoke(
            {"messages": [...]},
            config={
                "configurable": {
                    "model": "openai:gpt-4o",
                    "model_params": {"temperature": 0.2, "max_tokens": 1024},
                }
            },
        )
        ```

    `model_params` are merged into the request's `model_settings`, so the
    supported keys depend on the selected provider and model.
    """

    @staticmethod
    def _get_runnable_config(request: ModelRequest) -> RunnableConfig:
        """Extract runnable config for the current model request.

        LangGraph recommends `get_config()` for middleware access. We still
        accept `request.runtime.config` as a fallback because tests and some
        call sites inject it directly.

        Args:
            request: The current model request.

        Returns:
            Runnable config for this request, or an empty dict when unavailable.
        """
        empty_config: RunnableConfig = {}

        try:
            return get_config()
        except RuntimeError:
            runtime = request.runtime
            if runtime is None:
                return empty_config

            try:
                runtime_config = cast("_RuntimeWithConfig", runtime).config
            except AttributeError:
                return empty_config

            if runtime_config is None:
                return empty_config
            return runtime_config

    @classmethod
    def _get_configurable(cls, request: ModelRequest) -> _ConfigurableModelConfig:
        """Extract validated `configurable` settings from runnable config.

        Args:
            request: The current model request.

        Returns:
            The validated `configurable` settings.

        Raises:
            TypeError: If the runnable config is not a dict.
            TypeError: If `config["configurable"]` is not a dict.
        """
        config = cls._get_runnable_config(request)
        if not isinstance(config, dict):
            msg = "`config` must be a dictionary."
            raise TypeError(msg)
        configurable = config.get("configurable", {})
        if not isinstance(configurable, dict):
            msg = "`config['configurable']` must be a dictionary."
            raise TypeError(msg)
        return cast("_ConfigurableModelConfig", configurable)

    def _get_override_model(self, request: ModelRequest) -> BaseChatModel | None:
        """Read the model override from runtime config, if present.

        Args:
            request: The current model request.

        Returns:
            A resolved `BaseChatModel` if an override is specified, else `None`.
        """
        configurable = self._get_configurable(request)
        raw_model_spec = configurable.get("model")
        if raw_model_spec is None:
            return None
        if not isinstance(raw_model_spec, str):
            msg = "`config['configurable']['model']` must be a string."
            raise TypeError(msg)

        if model_matches_spec(request.model, raw_model_spec):
            return None

        logger.debug("Overriding model to %s", raw_model_spec)
        return resolve_model(raw_model_spec)

    @staticmethod
    def _get_model_params(request: ModelRequest) -> dict[str, Any] | None:
        """Read invocation param overrides from runtime config.

        Args:
            request: The current model request.

        Returns:
            A non-empty dict of params to merge into `model_settings`,
                or `None` if no overrides are configured.

        Raises:
            TypeError: If `model_params` is not a dict.
        """
        configurable = ConfigurableModelMiddleware._get_configurable(request)
        raw_params = configurable.get("model_params")
        if raw_params is None:
            return None
        if not isinstance(raw_params, dict):
            msg = "`config['configurable']['model_params']` must be a dictionary."
            raise TypeError(msg)
        return raw_params or None

    def _apply_overrides(self, request: ModelRequest) -> ModelRequest:
        """Apply model and param overrides from runtime config.

        Args:
            request: The original model request.

        Returns:
            A (possibly new) request with model and/or `model_settings`
                overrides applied.

                Returns the original request unchanged if no overrides are
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
