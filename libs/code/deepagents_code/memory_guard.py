"""Protect machine-managed memory blocks from agent edits.

The onboarding flow writes the user's preferred name into the user `AGENTS.md`
inside a marker-delimited block (see `onboarding.ONBOARDING_NAME_MEMORY_START` /
`ONBOARDING_NAME_MEMORY_END`). `MemoryMiddleware` strips HTML comments before
injecting memory, so the model never sees those markers and has no way to know
the region is off-limits. Since the same prompt tells the model to `edit_file`
that file to persist learnings, nothing stops it from rewriting the managed
block.

This middleware intercepts `write_file`/`edit_file` calls targeting the guarded
file(s), and `delete` calls that would remove them. When a write or edit would
change or remove the managed block, the model's other edits are kept (though
surrounding whitespace may be normalized, and a fully removed block is
re-appended rather than restored in place) while the managed block is restored,
and an error is returned so the model learns the region is machine-managed. A
`delete` call that would remove an existing managed block is rejected before the
tool runs; a `delete` of a guarded file that exists but cannot be read is also
rejected, failing closed rather than removing a file we cannot inspect. When the
block was altered but the restore could not be completed, an error is still
returned so the failure is never silent.
"""

from __future__ import annotations

import asyncio
import logging
import os
from difflib import SequenceMatcher
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import ToolMessage

from deepagents_code.onboarding import (
    _upsert_onboarding_name_memory,
    extract_onboarding_name_block,
    strip_onboarding_name_markers,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable

    from langgraph.prebuilt.tool_node import ToolCallRequest
    from langgraph.types import Command

logger = logging.getLogger(__name__)

_GUARDED_TOOLS: frozenset[str] = frozenset({"write_file", "edit_file", "delete"})
"""Tool names whose calls can mutate a guarded file and must be inspected."""

_REJECTION_MESSAGE = (
    "The region between the `deepagents:onboarding-name:start` and "
    "`deepagents:onboarding-name:end` markers in {path} is machine-managed and "
    "must not be edited. Your other changes to the file were kept, but the "
    "managed block was restored to its previous content. Do not modify content "
    "between those markers."
)
"""Error returned when a managed-block edit was reverted (`{path}` formatted in)."""

_RESTORE_FAILED_MESSAGE = (
    "The region between the `deepagents:onboarding-name:start` and "
    "`deepagents:onboarding-name:end` markers in {path} is machine-managed and "
    "must not be edited. Your edit changed it and the previous content could "
    "not be restored, so the managed block may now be corrupted. Do not modify "
    "content between those markers, and do not rely on this edit having "
    "succeeded."
)
"""Error returned when a managed-block edit could not be reverted."""

_DELETE_REJECTION_MESSAGE = (
    "The guarded memory file {path} contains a machine-managed region between "
    "the `deepagents:onboarding-name:start` and `deepagents:onboarding-name:end` "
    "markers and must not be deleted. Do not delete this file or a parent "
    "directory that contains it."
)
"""Error returned when a delete would remove a managed memory block."""


class _RestoreOutcome(Enum):
    """Result of attempting to restore a managed block after a tool call."""

    UNCHANGED = "unchanged"
    """The managed block was not altered; nothing to restore."""

    RESTORED = "restored"
    """The managed block was altered and successfully restored."""

    FAILED = "failed"
    """The managed block was altered but could not be restored."""


class ManagedMemoryGuardMiddleware(AgentMiddleware):
    """Revert agent edits to the managed onboarding-name memory block.

    Guards the managed onboarding-name block in a fixed set of memory files. A
    `write_file`/`edit_file` that leaves the managed block untouched passes
    through; one that alters or drops it has the block restored (other edits
    kept) and returns an error. A `delete` targeting a guarded file (or a parent
    directory that contains one) is rejected outright before the tool runs when
    the file holds a managed block or exists but cannot be read. If the restore
    itself fails, an error is still returned so the failure is never silent.
    """

    def __init__(self, guarded_paths: Iterable[str | Path]) -> None:
        """Initialize the guard with the memory files to protect.

        Args:
            guarded_paths: Paths whose managed onboarding-name block must be
                protected from agent edits. Resolved to absolute form for
                matching; unresolvable entries are skipped.
        """
        super().__init__()
        requested = list(guarded_paths)
        resolved: set[Path] = set()
        for raw in requested:
            try:
                resolved.add(Path(raw).expanduser().resolve())
            except (OSError, RuntimeError, ValueError):
                logger.warning(
                    "Could not resolve guarded memory path %r", raw, exc_info=True
                )
        self._guarded: frozenset[Path] = frozenset(resolved)
        if requested and not self._guarded:
            # Every configured path failed to resolve, so this guard now
            # protects nothing. That nullifies an integrity control, so surface
            # it loudly rather than letting protection silently disappear.
            logger.error(
                "ManagedMemoryGuardMiddleware resolved no guarded paths from %r; "
                "managed memory-block protection is disabled",
                requested,
            )

    def _guarded_path(self, request: ToolCallRequest) -> Path | None:
        """Return the resolved guarded path targeted by the call, if any.

        Returns:
            The matching guarded `Path`, or `None` when the call is unrelated.
        """
        tool_name = request.tool_call["name"]
        if tool_name not in _GUARDED_TOOLS:
            return None
        args = request.tool_call.get("args") or {}
        file_path = args.get("file_path")
        if not isinstance(file_path, str) or not file_path:
            return None
        try:
            resolved = Path(file_path).expanduser().resolve()
        except (OSError, RuntimeError, ValueError):
            # A guarded-tool call whose path won't resolve could be an attempt
            # to slip past the set-membership match, so leave a trail.
            logger.warning(
                "Could not resolve target path %r for %s",
                file_path,
                tool_name,
                exc_info=True,
            )
            return None
        if tool_name == "delete":
            # `is_relative_to` is True when the guarded file is the delete
            # target itself or lives under a directory being deleted.
            for guarded in self._guarded:
                if guarded.is_relative_to(resolved):
                    return guarded
            return None
        return resolved if resolved in self._guarded else None

    @staticmethod
    def _read(path: Path) -> str | None:
        """Read `path` as UTF-8, returning `None` on failure.

        Returns:
            File content, or `None` when the file is missing or unreadable.
        """
        try:
            fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            with os.fdopen(fd, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            # Expected when the guarded file has not been created yet.
            return None
        except (OSError, UnicodeDecodeError):
            # An existing-but-unreadable guarded file would otherwise silently
            # disable protection for this call, so make it visible.
            logger.warning("Could not read guarded memory file %s", path, exc_info=True)
            return None

    @staticmethod
    def _write(path: Path, content: str) -> None:
        """Write `content` to `path` without following symlinks."""
        flags = os.O_WRONLY | os.O_TRUNC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(path, flags)
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(content)

    @staticmethod
    def _line_range_for_block(before: str, before_block: str) -> tuple[int, int] | None:
        """Return the line range occupied by `before_block` in `before`.

        Returns:
            A `(start, end)` line range, or `None` when the block is absent.
        """
        block_start = before.find(before_block)
        if block_start == -1:
            return None
        block_end = block_start + len(before_block)
        start_line: int | None = None
        end_line: int | None = None
        offset = 0
        for line_number, line in enumerate(before.splitlines(keepends=True)):
            line_end = offset + len(line)
            if start_line is None and offset <= block_start < line_end:
                start_line = line_number
            if offset < block_end <= line_end:
                end_line = line_number + 1
                break
            offset = line_end
        if start_line is None or end_line is None:
            return None
        return start_line, end_line

    @staticmethod
    def _without_managed_block_edits(
        before: str, after: str, before_block: str
    ) -> str | None:
        """Remove post-edit lines that originated from the managed block.

        Returns:
            `after` with lines mapped from `before_block` removed, or `None` when
                the old block cannot be located in `before`.
        """
        block_range = ManagedMemoryGuardMiddleware._line_range_for_block(
            before, before_block
        )
        if block_range is None:
            return None
        block_start, block_end = block_range
        # Use line-level matching so a damaged marker cannot leave the old
        # managed memory body behind as regular user-editable memory.
        before_lines = before.splitlines(keepends=True)
        after_lines = after.splitlines(keepends=True)
        ranges: list[tuple[int, int]] = []
        matcher = SequenceMatcher(None, before_lines, after_lines, autojunk=False)
        for (
            tag,
            before_start,
            before_end,
            after_start,
            after_end,
        ) in matcher.get_opcodes():
            overlaps = before_start < block_end and block_start < before_end
            if tag == "insert":
                if block_start < before_start < block_end:
                    ranges.append((after_start, after_end))
                continue
            if not overlaps or tag == "delete":
                continue
            if tag == "equal":
                start = max(before_start, block_start)
                end = min(before_end, block_end)
                ranges.append(
                    (
                        after_start + start - before_start,
                        after_start + end - before_start,
                    )
                )
            else:
                ranges.append((after_start, after_end))

        if not ranges:
            return after
        parts: list[str] = []
        cursor = 0
        for start, end in sorted(ranges):
            range_start = start
            range_end = end
            if range_start < cursor:
                range_end = max(range_end, cursor)
                range_start = cursor
            parts.extend(after_lines[cursor:range_start])
            cursor = range_end
        parts.extend(after_lines[cursor:])
        return "".join(parts)

    def _restore(self, path: Path, before: str, before_block: str) -> _RestoreOutcome:
        """Re-apply `before_block` into `path`, preserving other edits.

        The restored content is verified before it is written, so a malformed
        re-insertion (for example from a partially deleted block) is reported as
        a failure instead of being persisted.

        Returns:
            `UNCHANGED` when the block was untouched, `RESTORED` when it was
                altered and successfully restored, or `FAILED` when it was
                altered but could not be restored.
        """
        after = self._read(path)
        if after is None:
            # The file vanished or became unreadable after the edit, so the
            # block cannot be restored. Treat as a failure rather than passing
            # the clobbering edit through as a success.
            logger.warning(
                "Guarded memory file %s is unreadable after edit; "
                "cannot restore managed block",
                path,
            )
            return _RestoreOutcome.FAILED
        block_after = extract_onboarding_name_block(after)
        if block_after == before_block:
            return _RestoreOutcome.UNCHANGED
        if block_after is not None:
            source = after
        else:
            source = self._without_managed_block_edits(before, after, before_block)
            if source is None:
                logger.error(
                    "Could not locate previous managed block in %s; leaving the "
                    "edited file untouched",
                    path,
                )
                return _RestoreOutcome.FAILED
            source = strip_onboarding_name_markers(source)
        restored = _upsert_onboarding_name_memory(source, before_block)
        if extract_onboarding_name_block(restored) != before_block:
            logger.error(
                "Restored content for %s did not reproduce the managed block; "
                "leaving the edited file untouched",
                path,
            )
            return _RestoreOutcome.FAILED
        try:
            self._write(path, restored)
        except (OSError, UnicodeEncodeError):
            logger.warning(
                "Could not restore managed memory block at %s", path, exc_info=True
            )
            return _RestoreOutcome.FAILED
        return _RestoreOutcome.RESTORED

    @staticmethod
    def _error(
        request: ToolCallRequest, path: Path, *, restore_failed: bool
    ) -> ToolMessage:
        """Build the error result returned after a managed-block edit.

        Returns:
            An error-status `ToolMessage` explaining the managed region.
        """
        template = _RESTORE_FAILED_MESSAGE if restore_failed else _REJECTION_MESSAGE
        return ToolMessage(
            content=template.format(path=path),
            name=request.tool_call["name"],
            tool_call_id=request.tool_call["id"],
            status="error",
        )

    @staticmethod
    def _reject_delete(path: Path, before: str | None) -> bool:
        """Return whether a delete targeting a guarded path must be rejected.

        The caller has already matched `path` as a guarded file (or a file
        inside a directory being deleted). The delete is rejected when:

        - the guarded file currently holds a managed block, or
        - the guarded file exists but its content could not be read.

        The second case fails closed on purpose: `_read` returns `None` for
        both a missing file and an existing-but-unreadable one (a permission
        error, or a symlink swap caught by `O_NOFOLLOW`). A missing guarded
        file has nothing to protect and is safe to remove, but an existing
        file we cannot inspect must not be deleted irreversibly on the
        assumption that it lacks a managed block.

        Returns:
            `True` when the delete must be blocked, `False` when it may proceed.
        """
        if before is not None:
            return extract_onboarding_name_block(before) is not None
        return path.exists()

    @staticmethod
    def _delete_error(request: ToolCallRequest, path: Path) -> ToolMessage:
        """Build the error result returned when a delete would remove memory.

        Returns:
            An error-status `ToolMessage` explaining the protected file.
        """
        return ToolMessage(
            content=_DELETE_REJECTION_MESSAGE.format(path=path),
            name=request.tool_call["name"],
            tool_call_id=request.tool_call["id"],
            status="error",
        )

    def _result_after_restore(
        self,
        request: ToolCallRequest,
        path: Path,
        before: str,
        before_block: str,
        result: ToolMessage | Command[Any],
    ) -> ToolMessage | Command[Any]:
        """Restore the managed block and pick the result to return.

        Returns:
            The original `result` when the block was untouched, otherwise an
                error `ToolMessage` describing the restore.
        """
        outcome = self._restore(path, before, before_block)
        if outcome is _RestoreOutcome.UNCHANGED:
            return result
        return self._error(
            request, path, restore_failed=outcome is _RestoreOutcome.FAILED
        )

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Restore the managed block when a sync edit would change it.

        Returns:
            The tool result, or an error `ToolMessage` when the managed block
                was altered.
        """
        path = self._guarded_path(request)
        if path is None:
            return handler(request)
        before = self._read(path)
        if request.tool_call["name"] == "delete":
            if self._reject_delete(path, before):
                return self._delete_error(request, path)
            return handler(request)
        before_block = (
            extract_onboarding_name_block(before) if before is not None else None
        )
        if before is None or before_block is None:
            return handler(request)
        result = handler(request)
        return self._result_after_restore(request, path, before, before_block, result)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Restore the managed block when an async edit would change it.

        Returns:
            The tool result, or an error `ToolMessage` when the managed block
                was altered.
        """
        path = await asyncio.to_thread(self._guarded_path, request)
        if path is None:
            return await handler(request)
        before = await asyncio.to_thread(self._read, path)
        if request.tool_call["name"] == "delete":
            if await asyncio.to_thread(self._reject_delete, path, before):
                return self._delete_error(request, path)
            return await handler(request)
        before_block = (
            extract_onboarding_name_block(before) if before is not None else None
        )
        if before is None or before_block is None:
            return await handler(request)
        result = await handler(request)
        return await asyncio.to_thread(
            self._result_after_restore, request, path, before, before_block, result
        )
