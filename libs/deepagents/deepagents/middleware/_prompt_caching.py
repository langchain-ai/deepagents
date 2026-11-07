"""Lazy provider-specific prompt caching behind a single middleware."""

from __future__ import annotations

import logging
import sys
from importlib import import_module
from typing import TYPE_CHECKING, Any, cast

from langchain.agents.middleware.types import AgentMiddleware, TracePolicy, omit_payload

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain.agents.middleware.types import ModelCallResult, ModelRequest, ModelResponse

logger = logging.getLogger(__name__)

_PROVIDERS = {
    "AnthropicPromptCachingMiddleware": ("langchain_anthropic", ("ChatAnthropic",)),
    "BedrockPromptCachingMiddleware": ("langchain_aws", ("ChatBedrock", "ChatBedrockConverse")),
    "FireworksPromptCachingMiddleware": ("langchain_fireworks", ("ChatFireworks",)),
}


class _CachingMiddleware(AgentMiddleware[Any, Any, Any]):
    """Dispatch caching lazily using the actual request model, including after model switches."""

    trace_policy = TracePolicy(process_inputs=omit_payload)
    provider_names = frozenset(_PROVIDERS)

    def __init__(self) -> None:
        """Initialize without importing any provider integrations."""
        super().__init__()
        self._delegates: dict[str, AgentMiddleware[Any, Any, Any] | None] = {}
        self._excluded: set[str] = set()

    @property
    def name(self) -> str:
        """Expose a stable name for replacing or excluding all caching."""
        return "CachingMiddleware"

    def exclude(self, entries: set[type[AgentMiddleware[Any, Any, Any]] | str]) -> set[type[AgentMiddleware[Any, Any, Any]] | str]:
        """Disable individual providers while preserving legacy exclusions and replacements."""
        matched: set[type[AgentMiddleware[Any, Any, Any]] | str] = set()
        for name, (package, _) in _PROVIDERS.items():
            module = sys.modules.get(f"{package}.middleware.prompt_caching")
            cls = vars(module).get(name) if module is not None else None
            for entry in entries:
                if entry == name or (cls is not None and entry is cls):
                    self._excluded.add(name)
                    matched.add(entry)
        return matched

    def _resolve(self, request: ModelRequest) -> AgentMiddleware[Any, Any, Any] | None:
        """Resolve only the matching provider, remembering unavailable middleware separately."""
        for name, (package, models) in _PROVIDERS.items():
            if name in self._excluded:
                continue
            module = sys.modules.get(package)
            classes = tuple(cls for model in models if module is not None and isinstance(cls := vars(module).get(model), type))
            if isinstance(request.model, classes):
                if name not in self._delegates:
                    self._delegates[name] = _create_delegate(package, name)
                return self._delegates[name]
        return None

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelCallResult:
        """Apply the matching provider's caching behavior synchronously."""
        delegate = self._resolve(request)
        return handler(request) if delegate is None else delegate.wrap_model_call(request, handler)

    async def awrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]) -> ModelCallResult:
        """Apply the matching provider's caching behavior asynchronously."""
        delegate = self._resolve(request)
        return await handler(request) if delegate is None else await delegate.awrap_model_call(request, handler)


def _create_delegate(package: str, name: str) -> AgentMiddleware[Any, Any, Any] | None:
    """Load optional caching middleware without hiding broken transitive dependencies."""
    module_name = f"{package}.middleware.prompt_caching"
    try:
        module = import_module(module_name)
    except ImportError as exc:
        if exc.name not in {package, f"{package}.middleware", module_name}:
            raise
        logger.debug("%s is unavailable.", name, exc_info=exc)
        return None
    return cast("AgentMiddleware[Any, Any, Any]", getattr(module, name)(unsupported_model_behavior="ignore"))


def append_prompt_caching_middleware(middleware: list[AgentMiddleware[Any, Any, Any]]) -> None:
    """Append one caching dispatcher without importing provider integrations."""
    middleware.append(_CachingMiddleware())
