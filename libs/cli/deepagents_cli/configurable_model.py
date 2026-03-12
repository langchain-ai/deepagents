"""CLI middleware for runtime model selection via LangGraph runtime context.

Allows switching the model per invocation by passing a `CLIContext` via
`context=` on `agent.astream()` / `agent.invoke()` without recompiling
the graph.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from deepagents._models import model_matches_spec, resolve_model  # noqa: PLC2701
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
    model = ctx.get("model")
    if model and not model_matches_spec(request.model, model):
        logger.debug("Overriding model to %s", model)
        overrides["model"] = resolve_model(model)

    # Param merge
    model_params = ctx.get("model_params", {})
    if model_params:
        overrides["model_settings"] = {**request.model_settings, **model_params}

    return request.override(**overrides) if overrides else request


class ConfigurableModelMiddleware(AgentMiddleware):
    """Swap the model or per-call settings from `runtime.context`."""

    _deepagents_prepend = True

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
