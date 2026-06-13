"""Normalize empty `ls`/`glob` tool output for the model.

!!! warning "Temporary workaround"

    This middleware exists only because the SDK's `ls`/`glob` tools currently
    serialize a successful-but-empty result as the literal `"[]"`. Upstream is
    expected to start returning useful empty-result content directly, at which
    point this middleware becomes redundant and should be removed. The canary
    test `test_sdk_still_returns_bracket_for_empty_listing` in
    `tests/unit_tests/test_filesystem_empty_result.py` fails loudly when that
    upstream change lands, signalling that this module can be deleted.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain.agents.middleware.types import AgentMiddleware

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain_core.messages import ToolMessage
    from langgraph.prebuilt.tool_node import ToolCallRequest
    from langgraph.types import Command

# The SDK's `ls`/`glob` tools serialize a successful-but-empty result as
# `str([])` -> the literal "[]", which the model reads as opaque/ambiguous
# content rather than "the directory is empty". Rewrite it to the same
# human-readable phrasing the SDK already uses for empty results elsewhere
# (see `_glob_search_files` in `deepagents.backends.utils`) so the agent
# gets a consistent, unambiguous "empty" signal.
_EMPTY_FILE_LIST_SENTINEL = "No files found"
_FILE_LIST_TOOLS: frozenset[str] = frozenset({"ls", "glob"})


class _FilesystemEmptyResultMiddleware(AgentMiddleware):
    """Rewrite empty `ls`/`glob` output to a readable sentinel for the model.

    A successful `ls`/`glob` with no entries serializes to the literal "[]",
    which is ambiguous to the model. This middleware rewrites only that exact
    case (success status, list tool, "[]" content) to `No files found`; error
    results, non-list tools, and non-empty output pass through untouched.

    !!! warning "Temporary workaround"

        This middleware exists only because the SDK's `ls`/`glob` tools currently
        serialize a successful-but-empty result as the literal `"[]"`. Upstream is
        expected to start returning useful empty-result content directly, at which
        point this middleware becomes redundant and should be removed. The canary
        test `test_sdk_still_returns_bracket_for_empty_listing` in
        `tests/unit_tests/test_filesystem_empty_result.py` fails loudly when that
        upstream change lands, signalling that this module can be deleted.
    """

    @staticmethod
    def _normalize_result(
        result: ToolMessage | Command[Any],
    ) -> ToolMessage | Command[Any]:
        # Narrow to `ToolMessage` so the rewrite is type-checked and `Command`
        # results pass through explicitly rather than via silent `getattr`
        # defaults. `ToolMessage` is a `TYPE_CHECKING`-only import, so import
        # the concrete class locally for the runtime `isinstance` check.
        from langchain_core.messages import ToolMessage as LCToolMessage

        if (
            isinstance(result, LCToolMessage)
            and result.name in _FILE_LIST_TOOLS
            and result.status == "success"
            and result.content == "[]"
        ):
            result.content = _EMPTY_FILE_LIST_SENTINEL
        return result

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Normalize empty list outputs after sync filesystem tool calls.

        Returns:
            The tool result, with empty `ls`/`glob` output rewritten to the
                sentinel; all other results are returned unchanged.
        """
        return self._normalize_result(handler(request))

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Normalize empty list outputs after async filesystem tool calls.

        Returns:
            The tool result, with empty `ls`/`glob` output rewritten to the
                sentinel; all other results are returned unchanged.
        """
        return self._normalize_result(await handler(request))
