"""CLI middleware for runtime model selection via LangGraph runtime context.

Allows switching the model per invocation by passing a `CLIContext` via
`context=` on `agent.astream()` / `agent.invoke()` without recompiling
the graph.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from deepagents._models import model_matches_spec  # noqa: PLC2701
from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from typing_extensions import TypedDict

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


logger = logging.getLogger(__name__)


class CLIContext(TypedDict, total=False):
    """Runtime context passed via `context=` to the LangGraph graph.

    Carries per-invocation overrides that `ConfigurableModelMiddleware`
    reads from `request.runtime.context`.
    """

    model: str | None
    """Model spec to swap at runtime (e.g. `'openai:gpt-4o'`)."""

    model_params: dict[str, Any]
    """Invocation params (e.g. `temperature`, `max_tokens`) to merge
    into `model_settings`."""


def _is_anthropic_model(model: object) -> bool:
    """Check whether a resolved model reports `'anthropic'` as its provider.

    Uses `_get_ls_params` from `BaseChatModel` to read the provider name.

    Returns:
        `True` if the model's `ls_provider` is `'anthropic'`.
    """
    try:
        ls_params = model._get_ls_params()  # type: ignore[attr-defined]
    except (AttributeError, TypeError, RuntimeError):
        logger.debug(
            "_get_ls_params raised for %s; assuming non-Anthropic",
            type(model).__name__,
        )
        return False
    return isinstance(ls_params, dict) and ls_params.get("ls_provider") == "anthropic"


_ANTHROPIC_ONLY_SETTINGS: set[str] = {"cache_control"}
"""Keys injected by Anthropic-specific middleware (e.g.
`AnthropicPromptCachingMiddleware`) that are not accepted by other providers and
must be stripped on cross-provider swap."""


def _apply_overrides(request: ModelRequest) -> ModelRequest:
    """Apply model/param overrides from `CLIContext` on the runtime.

    Returns:
        The original request unchanged when no `CLIContext` is present or it
            contains no overrides, otherwise a new request with overrides.
    """
    runtime = request.runtime
    if runtime is None:
        return request

    ctx = runtime.context
    if not isinstance(ctx, dict):
        return request

    overrides: dict[str, Any] = {}

    # Model swap
    new_model = None
    model = ctx.get("model")
    if model and not model_matches_spec(request.model, model):
        from deepagents_cli.config import create_model
        from deepagents_cli.model_config import ModelConfigError

        logger.debug("Overriding model to %s", model)
        try:
            new_model = create_model(model).model
        except ModelConfigError:
            logger.exception(
                "Failed to resolve runtime model override '%s'; "
                "continuing with current model",
                model,
            )
            return request
        overrides["model"] = new_model

    # Param merge
    model_params = ctx.get("model_params", {})
    if model_params:
        overrides["model_settings"] = {**request.model_settings, **model_params}

    if not overrides:
        return request

    # When switching away from Anthropic, strip provider-specific settings
    # that would cause errors on other providers (e.g. cache_control passed
    # to the OpenAI SDK raises TypeError).
    if new_model is not None and not _is_anthropic_model(new_model):
        settings = overrides.get("model_settings", request.model_settings)
        dropped = settings.keys() & _ANTHROPIC_ONLY_SETTINGS
        if dropped:
            logger.debug(
                "Stripped Anthropic-only settings %s for non-Anthropic model",
                dropped,
            )
            overrides["model_settings"] = {
                k: v for k, v in settings.items() if k not in dropped
            }

    return request.override(**overrides)


class ConfigurableModelMiddleware(AgentMiddleware):
    """Swap the model or per-call settings from `runtime.context`."""

    def wrap_model_call(  # noqa: PLR6301
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Apply runtime overrides and delegate to the next handler."""  # noqa: DOC201
        return handler(_apply_overrides(request))

    async def awrap_model_call(  # noqa: PLR6301
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Apply runtime overrides and delegate to the next async handler."""  # noqa: DOC201
        return await handler(_apply_overrides(request))
