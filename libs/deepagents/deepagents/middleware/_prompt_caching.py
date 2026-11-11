"""Provider-specific prompt-caching middleware helpers."""

from __future__ import annotations

import logging
import sys
from importlib import import_module
from typing import TYPE_CHECKING, Any, cast

from langchain.agents.middleware.types import AgentMiddleware, TracePolicy, omit_payload

from deepagents.middleware._utils import is_anthropic_chat_model

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain.agents.middleware.types import ModelCallResult, ModelRequest, ModelResponse

logger = logging.getLogger(__name__)

_ANTHROPIC_PROMPT_CACHING_MODULE = "langchain_anthropic.middleware.prompt_caching"


class _LazyAnthropicPromptCachingMiddleware(AgentMiddleware[Any, Any, Any]):
    """Stand-in for `AnthropicPromptCachingMiddleware` that imports it on the first Anthropic request.

    Importing `langchain_anthropic` loads the `anthropic` SDK, which is slow. A
    request model can only be a `ChatAnthropic` once that module is loaded, and
    for any other model the real middleware passes the request through
    unchanged. So the stand-in defers the import until a request needs it, and
    still caches when a session switches to an Anthropic model.
    """

    trace_policy = TracePolicy(process_inputs=omit_payload)

    def __init__(self) -> None:
        """Initialize without importing `langchain_anthropic`."""
        super().__init__()
        self._delegate: AgentMiddleware[Any, Any, Any] | None = None

    @property
    def name(self) -> str:
        """Use the real middleware's name, so name-based replacement and exclusion still match."""
        return "AnthropicPromptCachingMiddleware"

    @property
    def excluded_as(self) -> type[AgentMiddleware[Any, Any, Any]]:
        """Return the class that a class-form `excluded_middleware` entry must name to drop this middleware.

        A caller that names `AnthropicPromptCachingMiddleware` has imported it,
        so a `sys.modules` lookup finds it without importing.
        """
        module = sys.modules.get(_ANTHROPIC_PROMPT_CACHING_MODULE)
        return module.AnthropicPromptCachingMiddleware if module is not None else type(self)

    def _resolve(self, request: ModelRequest) -> AgentMiddleware[Any, Any, Any] | None:
        """Return the real middleware if `request` targets an Anthropic model, else `None`."""
        if not is_anthropic_chat_model(request.model):
            return None
        if self._delegate is None:
            # Cheap: `langchain_anthropic` is already loaded for the model.
            from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware  # noqa: PLC0415

            self._delegate = AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore")
        return self._delegate

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Apply Anthropic prompt caching when the request model is Anthropic."""
        delegate = self._resolve(request)
        if delegate is None:
            return handler(request)
        return delegate.wrap_model_call(request, handler)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Apply Anthropic prompt caching when the request model is Anthropic (async version)."""
        delegate = self._resolve(request)
        if delegate is None:
            return await handler(request)
        return await delegate.awrap_model_call(request, handler)


def _create_bedrock_prompt_caching_middleware() -> AgentMiddleware[Any, Any, Any] | None:
    """Create Bedrock prompt caching middleware when `langchain-aws` is installed."""
    module_name = "langchain_aws.middleware.prompt_caching"
    try:
        module = import_module(module_name)
    except ImportError as exc:
        if exc.name not in {"langchain_aws", "langchain_aws.middleware", module_name}:
            raise
        logger.debug("Bedrock prompt caching middleware is unavailable.", exc_info=exc)
        return None
    middleware_cls = module.BedrockPromptCachingMiddleware
    return cast("AgentMiddleware[Any, Any, Any]", middleware_cls(unsupported_model_behavior="ignore"))


def _create_fireworks_prompt_caching_middleware() -> AgentMiddleware[Any, Any, Any] | None:
    """Create Fireworks prompt caching middleware when `langchain-fireworks` is installed."""
    module_name = "langchain_fireworks.middleware.prompt_caching"
    try:
        module = import_module(module_name)
    except ImportError as exc:
        if exc.name not in {"langchain_fireworks", "langchain_fireworks.middleware", module_name}:
            raise
        logger.debug("Fireworks prompt caching middleware is unavailable.", exc_info=exc)
        return None
    middleware_cls = module.FireworksPromptCachingMiddleware
    return cast("AgentMiddleware[Any, Any, Any]", middleware_cls(unsupported_model_behavior="ignore"))


def append_prompt_caching_middleware(middleware: list[AgentMiddleware[Any, Any, Any]]) -> None:
    """Append provider-specific prompt caching middleware."""
    middleware.append(_LazyAnthropicPromptCachingMiddleware())
    bedrock_middleware = _create_bedrock_prompt_caching_middleware()
    if bedrock_middleware is not None:
        middleware.append(bedrock_middleware)
    fireworks_middleware = _create_fireworks_prompt_caching_middleware()
    if fireworks_middleware is not None:
        middleware.append(fireworks_middleware)
