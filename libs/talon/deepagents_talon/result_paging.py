"""Page oversized text results through a reader scoped to one fresh graph invocation."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING
from uuid import uuid4

from deepagents.backends import CompositeBackend
from deepagents.middleware._message_eviction import (
    _build_evicted_content,
    _build_evicted_tool_message,
    _create_content_preview,
    _extract_text_from_message,
    _render_preview_stub,
)
from deepagents.middleware.filesystem import (
    NUM_CHARS_PER_TOKEN,
    TOOLS_EXCLUDED_FROM_EVICTION,
    _routes_to_state_backend,
)
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from deepagents.backends.protocol import BackendProtocol
    from langchain.agents.middleware.types import ModelRequest, ModelResponse, ToolCallRequest
    from langgraph.types import Command


_EVICTION_THRESHOLD = 20_000 * NUM_CHARS_PER_TOKEN


class ToolResultPaging(AgentMiddleware):
    """Archive text and expose a reader only for artifacts written by this task."""

    reader_name = "read_tool_result"

    def __init__(self, backend: BackendProtocol) -> None:
        """Keep the successful writes for this invocation as its read allowlist."""
        self._backend = backend
        root = backend.artifacts_root if isinstance(backend, CompositeBackend) else "/"
        self._prefix = f"{root.rstrip('/')}/large_tool_results"
        self._paths: set[str] = set()
        self._reader = tool(self.reader_name)(self._read)

    async def _read(self, path: str, offset: int = 0, column: int = 0) -> str:
        """Read up to 50 lines from this task's result, starting at a zero-based line and column."""
        if path not in self._paths:
            return "Unknown result for this task."
        if offset < 0 or column < 0:
            return "Offset and column must be nonnegative."
        page = await self._backend.aread(path, offset=offset, limit=50)
        if page.error or page.file_data is None:
            return "The saved result could not be read."
        raw = page.file_data["content"]
        first, separator, rest = raw.partition("\n")
        text = first[column:] + separator + rest
        shown = text[:8000]
        if len(shown) < len(text):
            lines = shown.count("\n")
            next_column = len(shown.rsplit("\n", 1)[-1]) if lines else column + len(shown)
            return f"{shown}\n[Continue at offset={offset + lines}, column={next_column}]"
        if page.next_offset is not None:
            return f"{shown}\n[Continue at offset={page.next_offset}, column=0]"
        return shown

    async def awrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
    ) -> ModelResponse:
        """Offer recovery after the first successful archive write."""
        if self._paths:
            request = request.override(tools=[*request.tools, self._reader])
        return await handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Archive result messages while preserving other tool state updates."""
        name = request.tool_call["name"]
        if name == self.reader_name and self._paths:
            return await handler(request.override(tool=self._reader))
        result = await handler(request)
        if name in TOOLS_EXCLUDED_FROM_EVICTION:
            return result
        if isinstance(result, ToolMessage):
            return await self._archive(result)
        if isinstance(result.update, dict) and isinstance(
            messages := result.update.get("messages"), list
        ):
            messages = [
                await self._archive(m) if isinstance(m, ToolMessage) else m for m in messages
            ]
            return replace(result, update={**result.update, "messages": messages})
        return result

    async def _archive(self, message: ToolMessage) -> ToolMessage:
        text = _extract_text_from_message(message)
        if len(text) <= _EVICTION_THRESHOLD:
            return message
        path = f"{self._prefix}/{uuid4().hex}"
        if _routes_to_state_backend(self._backend, path):
            return message
        try:
            written = await self._backend.awrite(path, text)
        except Exception:  # noqa: BLE001 -- an archive failure must preserve the completed tool result
            return message
        if written is None or written.error:
            return message
        self._paths.add(path)
        stub = _render_preview_stub(
            "Full result: {path}\nUse read_tool_result(path, offset, column) during this task.\n"
            "{preview_note}\n{content_sample}",
            _create_content_preview(text),
            path=path,
        )
        return _build_evicted_tool_message(message, _build_evicted_content(message, stub))
