"""CLI middleware for runtime model selection via LangGraph runtime context.

Allows switching the model per invocation by passing a ``CLIContext``
via ``context=`` on ``agent.astream()`` / ``agent.invoke()`` without
recompiling the graph.
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


def _apply_overrides(request: ModelRequest) -> ModelRequest:
    """Apply model/param overrides from ``CLIContext`` on the runtime.

    Returns the original request unchanged when no ``CLIContext`` is present
    or it contains no overrides.
    """
    runtime = request.runtime
    if runtime is None:
        return request
    ctx = runtime.context
    if not isinstance(ctx, CLIContext):
        return request

    overrides: dict[str, Any] = {}

    # Model swap
    if ctx.model is not None and not model_matches_spec(request.model, ctx.model):
        logger.debug("Overriding model to %s", ctx.model)
        overrides["model"] = resolve_model(ctx.model)

    # Param merge
    if ctx.model_params:
        overrides["model_settings"] = {**request.model_settings, **ctx.model_params}

    return request.override(**overrides) if overrides else request


class ConfigurableModelMiddleware(AgentMiddleware):
    """Swap the model or per-call settings from ``runtime.context``."""

    _deepagents_prepend = True

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        return handler(_apply_overrides(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        return await handler(_apply_overrides(request))
