"""Stateless middleware that detects edit loops on the same file.

NOT YET WIRED IN — add ``LoopDetectionMiddleware()`` to the middleware
list in ``agent.py:create_cli_agent()`` to enable.

Soft warning at 8 edits (appended to tool output), hard warning at 12
(injected HumanMessage + ``jump_to: "model"``).  All counts derived
from ``state["messages"]`` — no mutable instance state.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    hook_config,
)
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain.tools.tool_node import ToolCallRequest
    from langgraph.runtime import Runtime
    from langgraph.types import Command

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Warning templates
# ---------------------------------------------------------------------------

LOOP_WARNING_SOFT = """
**NOTE**: You've edited `{file_path}` {count} times. If you're stuck, consider:
- Is there a fundamentally different approach?
- Can you use a different library or tool?
- Should you step back and re-read the original requirements?
"""

LOOP_WARNING_HARD = """
You've edited `{file_path}` {count} times without resolving the issue.

Stop and ask the user what to do — describe what you've tried and what's
not working so they can help you course-correct.
"""

# Tool names we track
_EDIT_TOOL_NAMES = frozenset({"edit_file", "write_file"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_file_edits(messages: list[Any]) -> dict[str, int]:
    """Count per-file edit_file/write_file calls from AIMessage tool_calls."""
    counts: dict[str, int] = {}
    for msg in messages:
        if not isinstance(msg, AIMessage):
            continue
        for tc in getattr(msg, "tool_calls", []) or []:
            if tc.get("name") not in _EDIT_TOOL_NAMES:
                continue
            args = tc.get("args") or {}
            raw_path = args.get("file_path") or args.get("path") or "unknown"
            path = str(Path(raw_path).resolve())
            counts[path] = counts.get(path, 0) + 1
    return counts


def _soft_warning_already_shown(messages: list[Any], file_path: str) -> bool:
    """Check if a soft warning for *file_path* was already appended."""
    marker = f"**NOTE**: You've edited `{file_path}`"
    for msg in messages:
        if isinstance(msg, ToolMessage) and marker in str(msg.content):
            return True
    return False


def _hard_warning_already_shown(messages: list[Any], file_path: str) -> bool:
    """Check if a hard warning for *file_path* was already injected."""
    marker = f"You've edited `{file_path}`"
    for msg in messages:
        if isinstance(msg, HumanMessage) and marker in str(msg.content):
            return True
    return False


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


class LoopDetectionMiddleware(AgentMiddleware):
    """Detect edit loops on the same file via message scanning."""

    DEFAULT_SOFT_THRESHOLD = 8
    DEFAULT_HARD_THRESHOLD = 12

    def __init__(
        self,
        *,
        soft_threshold: int = DEFAULT_SOFT_THRESHOLD,
        hard_threshold: int = DEFAULT_HARD_THRESHOLD,
    ) -> None:
        """Initialize with edit-count thresholds.

        Raises:
            ValueError: If thresholds are invalid.
        """
        if soft_threshold < 1:
            msg = "soft_threshold must be >= 1"
            raise ValueError(msg)
        if hard_threshold <= soft_threshold:
            msg = "hard_threshold must be greater than soft_threshold"
            raise ValueError(msg)

        self.soft_threshold = soft_threshold
        self.hard_threshold = hard_threshold

    # -- tool call hook -----------------------------------------------------

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Track file edits and inject soft warnings at threshold."""
        result = handler(request)
        return self._maybe_append_soft_warning(request, result)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Async version of wrap_tool_call."""
        result = await handler(request)
        return self._maybe_append_soft_warning(request, result)

    def _maybe_append_soft_warning(
        self,
        request: ToolCallRequest,
        result: ToolMessage | Command[Any],
    ) -> ToolMessage | Command[Any]:
        """Append soft warning to tool result if threshold is reached."""
        tool_name = request.tool_call.get("name", "")
        if tool_name not in _EDIT_TOOL_NAMES:
            return result

        args = request.tool_call.get("args") or {}
        raw_path = args.get("file_path") or args.get("path") or "unknown"
        file_path = str(Path(raw_path).resolve())

        messages = request.state.get("messages", [])
        counts = _count_file_edits(messages)
        # The current tool call may not yet be reflected in messages,
        # so add 1 for this invocation.
        count = counts.get(file_path, 0) + 1

        if count < self.soft_threshold:
            return result

        if _soft_warning_already_shown(messages, file_path):
            return result

        logger.info(
            "Loop detection: %s edited %d times (soft warning)",
            file_path,
            count,
        )
        if isinstance(result, ToolMessage):
            result.content = str(result.content) + LOOP_WARNING_SOFT.format(
                file_path=file_path, count=count
            )

        return result

    # -- after model hook ---------------------------------------------------

    @hook_config(can_jump_to=["model"])
    def after_model(
        self,
        state: AgentState[Any],
        runtime: Runtime[Any],  # noqa: ARG002
    ) -> dict[str, Any] | None:
        """Inject escalation prompt at hard threshold."""
        return self._check_hard_threshold(state)

    @hook_config(can_jump_to=["model"])
    async def aafter_model(
        self,
        state: AgentState[Any],
        runtime: Runtime[Any],  # noqa: ARG002
    ) -> dict[str, Any] | None:
        """Async version of after_model."""
        return self._check_hard_threshold(state)

    def _check_hard_threshold(self, state: AgentState[Any]) -> dict[str, Any] | None:
        """Check if any file has hit the hard threshold and inject warning."""
        messages = state.get("messages", [])

        # Don't inject between AI tool_call and tool_result
        if messages:
            last_msg = messages[-1]
            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                return None

        counts = _count_file_edits(messages)

        for file_path, count in counts.items():
            if count < self.hard_threshold:
                continue
            if _hard_warning_already_shown(messages, file_path):
                continue

            logger.warning(
                "Loop detection: %s edited %d times (hard warning)",
                file_path,
                count,
            )
            return {
                "messages": [
                    HumanMessage(
                        content=LOOP_WARNING_HARD.format(
                            file_path=file_path, count=count
                        )
                    )
                ],
                "jump_to": "model",
            }

        return None


__all__ = ["LoopDetectionMiddleware"]
