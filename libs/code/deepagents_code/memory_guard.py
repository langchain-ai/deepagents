"""Protect machine-managed memory blocks from agent edits.

The onboarding flow writes the user's preferred name into the user `AGENTS.md`
inside a marker-delimited block (see `onboarding.ONBOARDING_NAME_MEMORY_START` /
`ONBOARDING_NAME_MEMORY_END`). `MemoryMiddleware` strips HTML comments before
injecting memory, so the model never sees those markers and has no way to know
the region is off-limits. Since the same prompt tells the model to `edit_file`
that file to persist learnings, nothing stops it from rewriting the managed
block.

This middleware intercepts `write_file`/`edit_file` calls targeting the guarded
file(s). When a call would change or remove the managed block, the model's other
edits are kept but the managed block is restored to its prior content, and an
error is returned so the model learns the region is machine-managed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware.types import AgentMiddleware

from deepagents_code.onboarding import (
    _upsert_onboarding_name_memory,
    extract_onboarding_name_block,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain_core.messages import ToolMessage
    from langgraph.prebuilt.tool_node import ToolCallRequest
    from langgraph.types import Command

logger = logging.getLogger(__name__)

_GUARDED_TOOLS: frozenset[str] = frozenset({"write_file", "edit_file"})

_REJECTION_MESSAGE = (
    "The region between the `deepagents:onboarding-name` markers in "
    "{path} is machine-managed and must not be edited. Your other changes "
    "to the file were kept, but the managed block was restored to its "
    "previous content. Do not modify content between those markers."
)


class ManagedMemoryGuardMiddleware(AgentMiddleware):
    """Revert agent edits to the managed onboarding-name memory block.

    Guards a fixed set of memory files. A `write_file`/`edit_file` that leaves
    the managed block untouched passes through; one that alters or drops it has
    the block restored (other edits preserved) and returns an error.
    """

    def __init__(self, guarded_paths: list[str]) -> None:
        """Initialize the guard with the memory files to protect.

        Args:
            guarded_paths: Absolute paths whose managed onboarding-name block
                must be protected from agent edits.
        """
        super().__init__()
        self._guarded: set[Path] = set()
        for raw in guarded_paths:
            try:
                self._guarded.add(Path(raw).expanduser().resolve())
            except (OSError, RuntimeError, ValueError):
                logger.debug("Could not resolve guarded memory path %r", raw)

    def _guarded_path(self, request: ToolCallRequest) -> Path | None:
        """Return the resolved guarded path targeted by the call, if any.

        Returns:
            The matching guarded `Path`, or `None` when the call is unrelated.
        """
        if request.tool_call["name"] not in _GUARDED_TOOLS:
            return None
        args = request.tool_call.get("args") or {}
        file_path = args.get("file_path")
        if not isinstance(file_path, str) or not file_path:
            return None
        try:
            resolved = Path(file_path).expanduser().resolve()
        except (OSError, RuntimeError, ValueError):
            return None
        return resolved if resolved in self._guarded else None

    @staticmethod
    def _read(path: Path) -> str | None:
        """Read `path` as UTF-8, returning `None` on failure.

        Returns:
            File content, or `None` when the file is missing or unreadable.
        """
        try:
            return path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None

    def _restore(self, path: Path, before_block: str) -> bool:
        """Re-apply `before_block` into `path`, preserving other edits.

        Returns:
            `True` when the managed block was restored, otherwise `False`.
        """
        after = self._read(path)
        if after is None:
            return False
        if extract_onboarding_name_block(after) == before_block:
            return False
        try:
            path.write_text(
                _upsert_onboarding_name_memory(after, before_block),
                encoding="utf-8",
            )
        except OSError:
            logger.warning(
                "Could not restore managed memory block at %s", path, exc_info=True
            )
            return False
        return True

    @staticmethod
    def _error(request: ToolCallRequest, path: Path) -> ToolMessage:
        """Build the error result returned after a reverted managed-block edit.

        Returns:
            An error-status `ToolMessage` explaining the managed region.
        """
        from langchain_core.messages import ToolMessage as LCToolMessage

        return LCToolMessage(
            content=_REJECTION_MESSAGE.format(path=path),
            name=request.tool_call["name"],
            tool_call_id=request.tool_call["id"],
            status="error",
        )

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Restore the managed block when a sync edit would change it.

        Returns:
            The tool result, or an error `ToolMessage` when the managed block
                was altered and restored.
        """
        path = self._guarded_path(request)
        if path is None:
            return handler(request)
        before = self._read(path)
        before_block = (
            extract_onboarding_name_block(before) if before is not None else None
        )
        result = handler(request)
        if before_block is None:
            return result
        if self._restore(path, before_block):
            return self._error(request, path)
        return result

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Restore the managed block when an async edit would change it.

        Returns:
            The tool result, or an error `ToolMessage` when the managed block
                was altered and restored.
        """
        path = self._guarded_path(request)
        if path is None:
            return await handler(request)
        before = self._read(path)
        before_block = (
            extract_onboarding_name_block(before) if before is not None else None
        )
        result = await handler(request)
        if before_block is None:
            return result
        if self._restore(path, before_block):
            return self._error(request, path)
        return result
