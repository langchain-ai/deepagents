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


def _is_anthropic_model(model: object) -> bool:
    """Check whether a resolved model is an Anthropic `ChatAnthropic` instance.

    Returns `False` if `langchain-anthropic` is not installed.

    Args:
        model: A model instance to inspect.

            Typed as `object` (rather than `BaseChatModel`) so the caller can
            pass any model without an import-time dependency on a specific
            provider package.

    Args:
        model: A model instance to inspect.

            Typed as `object` (rather than `BaseChatModel`) so the caller can
            pass any model without an import-time dependency on a specific
            provider package.

    Returns:
        `True` if the model is a `ChatAnthropic` instance.
    """
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        logger.debug("langchain_anthropic not installed; assuming non-Anthropic model")
        return False
    return isinstance(model, ChatAnthropic)


_ANTHROPIC_ONLY_SETTINGS: set[str] = {"cache_control"}
"""Keys injected by Anthropic-specific middleware (e.g.
`AnthropicPromptCachingMiddleware`) that are not accepted by other providers and
must be stripped on cross-provider swap."""


def _apply_overrides(request: ModelRequest) -> ModelRequest:
    """Apply model/param overrides from `CLIContext` on the runtime.

    Reads `'model'` and `'model_params'` from `runtime.context` and, when
    present, swaps the model and/or merges extra settings into the request.
    On a cross-provider swap away from Anthropic, Anthropic-only settings
    (e.g. `cache_control`) are stripped. The `### Model Identity` section
    in the system prompt is also patched to reflect the new model.

    Args:
        request: The incoming model request from the middleware chain.

    Returns:
        The original request unchanged when no `CLIContext` is present or it
            contains no overrides, otherwise a new request with overrides
            applied via `request.override()`.
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
        logger.debug("Overriding model to %s", model)
        new_model = resolve_model(model)
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

    # Patch the Model Identity section in the system prompt so the new model
    # sees its own name/provider/context-limit, not the original's.
    # We read metadata from model_result (not the CLI settings singleton)
    # because the middleware runs in the server subprocess where settings
    # are never updated by /model.
    if new_model is not None and request.system_prompt:
        from deepagents_cli.agent import (
            MODEL_IDENTITY_RE,
            build_model_identity_section,
        )

        prompt = request.system_prompt
        new_identity = build_model_identity_section(
            model_result.model_name,
            provider=model_result.provider,
            context_limit=model_result.context_limit,
            unsupported_modalities=model_result.unsupported_modalities,
        )
        patched = MODEL_IDENTITY_RE.sub(new_identity, prompt, count=1)
        if patched != prompt:
            overrides["system_prompt"] = patched
        elif "### Model Identity" in prompt:
            logger.warning(
                "System prompt contains '### Model Identity' but regex "
                "did not match; identity section was NOT updated for "
                "model '%s'. The regex may be out of sync with the "
                "prompt template.",
                model_result.model_name,
            )

    return request.override(**overrides)


class ConfigurableModelMiddleware(AgentMiddleware):
    """Swap the model or per-call settings from `runtime.context`.

    Reads two optional keys from the runtime context dict:

    - `'model'` — a `provider:model` spec (e.g. `"openai:gpt-5"`).
        When present and different from the current model, the request is
        re-routed to the new model.
    - `'model_params'` — a dict of extra model settings (e.g.
        `{"temperature": 0}`) that are shallow-merged into the
        request's `model_settings`.

    This middleware is typically the outermost layer so it intercepts every
    model call before provider-specific middleware (like
    `AnthropicPromptCachingMiddleware`) runs.
    """

    def wrap_model_call(  # noqa: PLR6301
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Apply runtime overrides and delegate to the next handler.

        Returns:
            The `ModelResponse` produced by the downstream handler.
        """
        return handler(_apply_overrides(request))

    async def awrap_model_call(  # noqa: PLR6301
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Apply runtime overrides and delegate to the next async handler.

        Returns:
            The `ModelResponse` produced by the downstream handler.
        """
        return await handler(_apply_overrides(request))
