"""Micro-compaction middleware for lightweight token savings without LLM calls.

This module provides `MicroCompactionMiddleware`, which replaces the content of
stale tool result messages with short stubs based on elapsed time. Unlike
`SummarizationMiddleware`, it requires no LLM call — content replacement is
purely rule-based.

## How it works

- Tool results older than `max_age_seconds` (default 5 min) have their
  `content` replaced with a short stub.
- Message structure (`role`, `tool_call_id`) is preserved for API compatibility.
- Error results (`status="error"`) are always preserved.
- The most recent `preserve_recent` messages are always kept intact.
- `read_file` results use a specific stub: `[file content cleared — re-read if needed]`

## Middleware stack placement

Place `MicroCompactionMiddleware` *before* `SummarizationToolMiddleware`:

```
TokenStateMiddleware → ... → MicroCompactionMiddleware → SummarizationToolMiddleware
```

This reduces token usage so that full LLM-based compaction is needed less often.

## Usage

```python
from deepagents import create_deep_agent
from deepagents.middleware.micro_compaction import MicroCompactionMiddleware

agent = create_deep_agent(
    middleware=[MicroCompactionMiddleware(max_age_seconds=300, preserve_recent=10)],
)
```
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import AnyMessage, ToolMessage

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain.agents.middleware.types import ModelRequest, ModelResponse

_STALE_STUB = "[content cleared — stale tool result]"
_FILE_READ_STUB = "[file content cleared — re-read if needed]"

# Tool names whose results use the file-specific stub
_FILE_READ_TOOLS = frozenset({"read_file", "glob", "grep", "ls"})


class MicroCompactionMiddleware(AgentMiddleware):
    """Lightweight middleware that clears stale tool result content without LLM calls.

    Tool results older than `max_age_seconds` have their content replaced with a
    short stub. Message structure (`role`, `tool_call_id`) is preserved for API
    compatibility. Error results are always preserved. The most recent
    `preserve_recent` messages are always kept intact regardless of age.

    This is a zero-cost alternative to `SummarizationMiddleware` that reduces
    context window usage for long-running sessions without incurring an LLM call.

    Args:
        max_age_seconds: Seconds after first observation at which a tool result
            is considered stale and eligible for compaction.
        preserve_recent: Number of most-recent messages to always keep intact,
            regardless of age.
    """

    def __init__(self, *, max_age_seconds: int = 300, preserve_recent: int = 10) -> None:
        """Initialize the micro-compaction middleware.

        Args:
            max_age_seconds: Seconds after first observation at which a tool result
                is considered stale and eligible for compaction.
            preserve_recent: Number of most-recent messages to always keep intact,
                regardless of age.
        """
        self._max_age_seconds = max_age_seconds
        self._preserve_recent = preserve_recent
        # Maps message id -> monotonic timestamp of first observation
        self._seen_at: dict[str, float] = {}

    def _compact_messages(self, messages: list[AnyMessage]) -> list[AnyMessage]:
        """Replace content of stale ToolMessages with lightweight stubs.

        Records the first-seen timestamp for any new ToolMessages, then replaces
        the content of those older than `max_age_seconds` with a short stub.
        The most recent `preserve_recent` messages are always left untouched.

        Args:
            messages: The current message list to process.

        Returns:
            Messages with stale tool result content replaced by stubs.
        """
        now = time.monotonic()

        # Record first-seen timestamp for any new ToolMessages
        for msg in messages:
            if isinstance(msg, ToolMessage) and msg.id:
                self._seen_at.setdefault(msg.id, now)

        # Determine recent messages by object identity (last N messages unconditionally kept)
        recent_slice = messages[-self._preserve_recent :] if self._preserve_recent > 0 else []
        recent_obj_ids = {id(m) for m in recent_slice}

        result: list[AnyMessage] = []
        for msg in messages:
            if (
                isinstance(msg, ToolMessage)
                and id(msg) not in recent_obj_ids
                and msg.id is not None
                and msg.status != "error"
                and (now - self._seen_at.get(msg.id, now)) > self._max_age_seconds
            ):
                stub = _FILE_READ_STUB if msg.name in _FILE_READ_TOOLS else _STALE_STUB
                kwargs: dict[str, Any] = {
                    "content": stub,
                    "tool_call_id": msg.tool_call_id,
                    "name": msg.name,
                    "id": msg.id,
                }
                if msg.status is not None:
                    kwargs["status"] = msg.status
                msg = ToolMessage(**kwargs)  # noqa: PLW2901  # intentional replacement in loop
            result.append(msg)
        return result

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Compact stale tool results before the model call.

        Args:
            request: Model request being processed.
            handler: Handler function to call with modified request.

        Returns:
            Model response from handler.
        """
        return handler(request.override(messages=self._compact_messages(request.messages)))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Compact stale tool results before the async model call.

        Args:
            request: Model request being processed.
            handler: Async handler function to call with modified request.

        Returns:
            Model response from handler.
        """
        return await handler(request.override(messages=self._compact_messages(request.messages)))
