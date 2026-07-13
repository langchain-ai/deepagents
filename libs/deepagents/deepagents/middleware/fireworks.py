"""Middleware for Fireworks prompt-cache session affinity.

Fireworks prompt caching is enabled by default, but for serverless and
multi-replica deployments Fireworks recommends passing a stable session hint so
related requests route to the same replica and reuse its warm cache. This
middleware sets that hint automatically from the active
`config.configurable.thread_id`, improving cache hit rate without any consumer
configuration.

## Overview

The middleware is a no-op unless the resolved model reports
`ls_provider == "fireworks"`. When it applies, it injects both a
`prompt_cache_key` and an `x-session-affinity` header derived from the thread
ID. If the caller already manages affinity (via a non-empty `user` or
`prompt_cache_key`, or an `x-session-affinity` header), the request is left
untouched.

## Usage

```python
from deepagents.middleware.fireworks import FireworksPromptCachingMiddleware

agent = create_deep_agent(middleware=[FireworksPromptCachingMiddleware()])
```

`create_deep_agent` already includes this middleware automatically in its
prompt-caching stack, so most consumers do not need to add it explicitly.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langgraph.config import get_config

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)

_FIREWORKS_PROVIDER = "fireworks"
"""LangSmith provider name reported by Fireworks chat models."""

_SESSION_AFFINITY_HEADER = "x-session-affinity"
"""Fireworks prompt-cache affinity header populated from the active thread ID."""

_USER_MANAGED_SETTINGS = ("user", "prompt_cache_key")
"""Model settings whose non-empty values signal caller-managed affinity."""


def _get_ls_provider(model: object) -> str | None:
    """Return the LangSmith provider name reported by a chat model.

    Reads `_get_ls_params()` (implemented by `BaseChatModel`) so detection is
    provider-metadata based and does not require importing `langchain-fireworks`.

    Returns:
        The `ls_provider` string when the model reports one, otherwise `None`
            (including when `_get_ls_params` is missing, raises, or yields a
            non-string provider).
    """
    try:
        ls_params = model._get_ls_params()  # ty: ignore[unresolved-attribute]
    except (AttributeError, TypeError, RuntimeError, NotImplementedError):
        logger.debug("_get_ls_params raised for %s", type(model).__name__)
        return None
    if isinstance(ls_params, dict):
        provider = ls_params.get("ls_provider")
        if isinstance(provider, str):
            return provider
    return None


def _get_thread_id() -> str | None:
    """Return a non-empty `config.configurable.thread_id`, if present.

    Returns:
        The thread ID string when a non-empty string is configured, otherwise
            `None` (including when not running inside a runnable context).
    """
    try:
        config = get_config()
    except RuntimeError:
        return None
    thread_id = config.get("configurable", {}).get("thread_id")
    if isinstance(thread_id, str) and thread_id:
        return thread_id
    return None


def _has_session_affinity_header(headers: Mapping[Any, Any]) -> bool:
    """Return whether `headers` already contains an `x-session-affinity` key.

    Comparison is case-insensitive.

    Returns:
        `True` if a string key case-insensitively equal to `x-session-affinity`
            is present.
    """
    return any(isinstance(key, str) and key.lower() == _SESSION_AFFINITY_HEADER for key in headers)


class FireworksPromptCachingMiddleware(AgentMiddleware):
    """Set Fireworks prompt-cache session affinity from the active thread ID.

    Fireworks prompt caching is enabled by default; this middleware improves
    cache hit rate by pinning session affinity to `config.configurable.thread_id`
    so related requests route to the same replica and reuse its warm cache.

    This is closer to "session affinity for prompt caching" than the
    `cache_control` block marking used for Anthropic/Bedrock, but the name is
    kept for discoverability alongside the other prompt-caching middleware.

    The middleware is a no-op unless the resolved model reports
    `ls_provider == "fireworks"`. Detection is provider-metadata based (via the
    model's `_get_ls_params()`), so it does not require `langchain-fireworks` to
    be installed. It also no-ops when:

    - no non-empty string `thread_id` is present in the config, or
    - the caller already manages affinity with a non-empty `user` or
        `prompt_cache_key`, or an `x-session-affinity` header (case-insensitive),
        or
    - `extra_headers` is present but is not a mapping (a warning is logged).

    Otherwise it injects both `prompt_cache_key` and
    `extra_headers["x-session-affinity"]`, set to the thread ID. Existing
    `model_settings` and headers are preserved, and caller-provided dicts are
    never mutated.
    """

    def _apply_session_affinity(self, request: ModelRequest) -> ModelRequest | None:
        """Return a request with session affinity applied, or `None` to no-op.

        Returns:
            A new `ModelRequest` with `prompt_cache_key` and the
                `x-session-affinity` header injected, or `None` when the request
                should be left unchanged.
        """
        if _get_ls_provider(request.model) != _FIREWORKS_PROVIDER:
            return None

        thread_id = _get_thread_id()
        if thread_id is None:
            return None

        model_settings = request.model_settings
        if any(model_settings.get(key) for key in _USER_MANAGED_SETTINGS):
            return None

        raw_headers = model_settings.get("extra_headers")
        if raw_headers is None:
            headers: dict[Any, Any] = {}
        elif isinstance(raw_headers, Mapping):
            if _has_session_affinity_header(raw_headers):
                return None
            headers = dict(raw_headers)
        else:
            logger.warning(
                "Cannot set Fireworks session affinity because extra_headers is %s",
                type(raw_headers).__name__,
            )
            return None

        headers[_SESSION_AFFINITY_HEADER] = thread_id
        new_settings = {
            **model_settings,
            "prompt_cache_key": thread_id,
            "extra_headers": headers,
        }
        # No thread ID in the message: it is treated as a sensitive session
        # identifier. The line's presence alone confirms injection ran.
        logger.debug("Set Fireworks prompt-cache session affinity")
        return request.override(model_settings=new_settings)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Inject Fireworks session affinity before delegating to the handler.

        Returns:
            The model response from the handler.
        """
        return handler(self._apply_session_affinity(request) or request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Inject Fireworks session affinity before delegating (async version).

        Returns:
            The model response from the handler.
        """
        return await handler(self._apply_session_affinity(request) or request)
