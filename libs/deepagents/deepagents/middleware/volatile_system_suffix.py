"""Middleware that appends a volatile (un-cached) trailing system block.

## Overview

Anthropic prompt caching works by tagging a content block with
`cache_control`; everything up to and including that block forms the cached
prefix. `AnthropicPromptCachingMiddleware` (and, when memory is configured,
`MemoryMiddleware`) place that breakpoint on the *last* system content block.

That is the wrong place for content that changes every turn -- the current
date, the signed-in user's identity, and similar per-request context. If such
content is the last block, it carries the breakpoint and shifts on every turn,
invalidating the prefix cache.

`VolatileSystemSuffixMiddleware` solves this by appending the volatile content
as a trailing system block *without* `cache_control`. Wired innermost by
`create_deep_agent` (after `AnthropicPromptCachingMiddleware` and
`MemoryMiddleware`), it runs last, so the breakpoint those middlewares set
stays on the stable block (the base prompt, or `AGENTS.md` when memory is
configured) and the volatile block simply trails it, outside the cached prefix.

## Usage

```python
from deepagents import create_deep_agent

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    volatile_system_suffix="Current date: 2026-06-01. You are speaking with Ada.",
)
```

## Trust boundary

The suffix is appended verbatim as system-role content. Callers that fold in
user-controlled data (names, emails, free text) are responsible for wrapping it
in explicit boundary markers (e.g. `<user_context>...</user_context>`) so the
model treats it as reference material rather than instructions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import ContentBlock, SystemMessage

from deepagents.middleware._utils import append_to_system_message


class VolatileSystemSuffixMiddleware(AgentMiddleware):
    """Append a trailing system content block that is never cached.

    Designed to run innermost -- after `AnthropicPromptCachingMiddleware` and
    `MemoryMiddleware` -- so the prompt-cache breakpoint they set stays on the
    stable block and the volatile block trails it without `cache_control`.

    The middleware never adds `cache_control` itself, so the appended content is
    always outside the cached prefix regardless of the model provider. On
    non-Anthropic models no caching metadata exists at all and the block is
    simply serialized as the final part of the single system instruction.
    """

    def __init__(self, suffix: str | list[ContentBlock]) -> None:
        """Initialize the middleware.

        Args:
            suffix: Volatile content to append after the cache breakpoint.

                A `str` is appended as a single trailing text block (separated
                from existing content by a blank line). A `list` of content
                blocks is appended verbatim, in order, as the trailing blocks.

                An empty string or empty list is a no-op.
        """
        self._suffix = suffix

    def modify_request(self, request: ModelRequest) -> ModelRequest:
        """Append the volatile suffix as the final system content block(s).

        Args:
            request: Model request to modify.

        Returns:
            Request whose system message ends with the volatile suffix, or the
            original request unchanged when the suffix is empty.
        """
        suffix = self._suffix
        if not suffix:
            return request

        if isinstance(suffix, str):
            new_system_message = append_to_system_message(request.system_message, suffix)
        else:
            blocks: list[ContentBlock] = list(request.system_message.content_blocks) if request.system_message else []
            blocks.extend(suffix)
            new_system_message = SystemMessage(content_blocks=blocks)

        return request.override(system_message=new_system_message)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Append the volatile suffix before delegating to the handler.

        Args:
            request: Model request being processed.
            handler: Handler function to call with the modified request.

        Returns:
            Model response from the handler.
        """
        return handler(self.modify_request(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Append the volatile suffix before delegating to the async handler.

        Args:
            request: Model request being processed.
            handler: Async handler function to call with the modified request.

        Returns:
            Model response from the handler.
        """
        return await handler(self.modify_request(request))
