"""Custom middleware for Harbor benchmark runs."""

import logging
import re
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import Annotated, Any, NotRequired

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
    PrivateStateAttr,
    ToolCallRequest,
)
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command

logger = logging.getLogger(__name__)


# The checklist reminder message - injected when agent appears to be finishing
CHECKLIST_REMINDER = """**STOP - Before you finish, walk through the PRE-COMPLETION CHECKLIST:**

☐ Did I EXECUTE my code (not just write it)?
☐ Did I check the ACTUAL OUTPUT matches requirements?
☐ Are field names/paths EXACTLY as specified?
☐ If task asked for ALL solutions, did I find ALL of them?
☐ If server task, is the server actually responding?
☐ If I compiled code, did I clean up intermediate files (.o, .a, build/)?
☐ Will automated verification pass immediately after I finish?

If ANY item is unchecked, go back and fix it now. Then respond with your final answer."""


class PreCompletionCheckState(AgentState):
    """State schema for PreCompletionCheckMiddleware.

    Attributes:
        checklist_reminder_shown: Whether we've already shown the checklist reminder.
            Private so it's not included in the final agent state.
    """

    checklist_reminder_shown: NotRequired[Annotated[bool, PrivateStateAttr]]


class PreCompletionCheckMiddleware(AgentMiddleware):
    """Middleware that injects a checklist reminder before the agent finishes.

    Detects when the agent is about to finish (produces an AI message without
    tool calls) and injects a reminder to verify the pre-completion checklist.

    Only triggers ONCE per conversation to avoid infinite loops.
    """

    state_schema = PreCompletionCheckState

    def after_model(
        self,
        state: PreCompletionCheckState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Check if agent is finishing without tool calls and inject reminder.

        Triggers when:
        1. The AI response has no tool calls (agent is "finishing")
        2. We haven't already shown the checklist reminder

        Args:
            state: Current agent state
            runtime: Runtime context

        Returns:
            State update with reminder message, or None if no action needed
        """
        # Don't trigger if we've already shown the reminder
        if state.get("checklist_reminder_shown", False):
            return None

        # Get the last message from state (new API - response is in state)
        messages = state.get("messages", [])
        if not messages:
            return None

        last_message = messages[-1]

        # Check if it's an AI message without tool calls (agent is finishing)
        if isinstance(last_message, AIMessage):
            has_tool_calls = bool(last_message.tool_calls)

            if not has_tool_calls:
                # Agent is about to finish - inject the checklist reminder
                return {
                    "messages": [HumanMessage(content=CHECKLIST_REMINDER)],
                    "checklist_reminder_shown": True,
                }

        return None


# =============================================================================
# Loop Detection Middleware
# =============================================================================

LOOP_WARNING_SOFT = """
**NOTE**: You've edited `{file_path}` {count} times. If you're stuck in a debugging loop, consider:
- Is there a fundamentally different approach?
- Can you use a different library or tool?
- Should you step back and verify your understanding of the problem?
"""

LOOP_WARNING_STRONG = """
**WARNING**: You've edited `{file_path}` {count} times - this strongly suggests you're stuck.

STOP and reconsider:
1. What is the ROOT CAUSE of the errors you're seeing?
2. Is there a simpler solution you're missing?
3. Would a completely different approach work better?
4. Are you trying to fix symptoms instead of the underlying problem?

Do NOT continue with the same approach. Try something fundamentally different.
"""


class LoopDetectionState(AgentState):
    """State schema for LoopDetectionMiddleware."""

    file_edit_counts: NotRequired[Annotated[dict[str, int], PrivateStateAttr]]
    error_pattern_counts: NotRequired[Annotated[dict[str, int], PrivateStateAttr]]
    loop_warnings_shown: NotRequired[Annotated[set[str], PrivateStateAttr]]


class LoopDetectionMiddleware(AgentMiddleware):
    """Middleware that detects when agent is stuck in an editing loop.

    Tracks how many times each file is edited and injects warnings when
    the count exceeds thresholds. This helps prevent infinite debugging
    loops where the agent keeps editing the same file without progress.
    """

    state_schema = LoopDetectionState

    # Thresholds for warnings
    SOFT_WARNING_THRESHOLD = 8
    STRONG_WARNING_THRESHOLD = 15

    def __init__(self):
        self._file_edit_counts: dict[str, int] = defaultdict(int)
        self._warnings_shown: set[str] = set()

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Wrap tool calls to track file edits and inject warnings."""
        # Execute the tool first
        result = await handler(request)

        # Check if this was an edit operation
        tool_name = request.tool_call.get("name", "")
        if tool_name in ("edit_file", "write_file"):
            args = request.tool_call.get("args", {})
            file_path = args.get("file_path") or args.get("path", "unknown")

            self._file_edit_counts[file_path] += 1
            count = self._file_edit_counts[file_path]

            # Inject warnings at thresholds
            warning_key = f"{file_path}:{count}"
            if count == self.SOFT_WARNING_THRESHOLD and warning_key not in self._warnings_shown:
                self._warnings_shown.add(warning_key)
                logger.warning(f"Loop detection: {file_path} edited {count} times")
                # Append warning to tool result
                if isinstance(result, ToolMessage):
                    result.content = str(result.content) + LOOP_WARNING_SOFT.format(
                        file_path=file_path, count=count
                    )

            elif count == self.STRONG_WARNING_THRESHOLD and warning_key not in self._warnings_shown:
                self._warnings_shown.add(warning_key)
                logger.warning(f"Loop detection STRONG: {file_path} edited {count} times")
                if isinstance(result, ToolMessage):
                    result.content = str(result.content) + LOOP_WARNING_STRONG.format(
                        file_path=file_path, count=count
                    )

        return result


# =============================================================================
# Context Budget Middleware
# =============================================================================

CONTEXT_WARNING = """
**NOTE**: Your conversation is at approximately {percent}% of context capacity.
Consider:
- Summarizing your findings so far
- Being more concise in future responses
- Avoiding re-reading files you've already seen
"""

TRUNCATION_NOTICE = """
[OUTPUT TRUNCATED: {original_lines} lines -> {kept_lines} lines]
[Full output saved to: {save_path}]
"""


class ContextBudgetMiddleware(AgentMiddleware):
    """Middleware that manages context budget to prevent overflow.

    Features:
    1. Truncates oversized tool outputs (>MAX_OUTPUT_LINES)
    2. Warns when total context approaches limit
    3. Saves full output to temp file when truncating
    """

    MAX_OUTPUT_LINES = 200  # Max lines per tool output
    WARN_THRESHOLD_PERCENT = 70  # Warn at 70% of context

    def __init__(self, max_context_tokens: int = 128000):
        self.max_context_tokens = max_context_tokens
        self._estimated_tokens = 0
        self._warning_shown = False
        self._truncation_counter = 0

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimate: ~4 chars per token."""
        return len(text) // 4

    def _truncate_output(self, content: str) -> str:
        """Truncate oversized output and save full version to temp file."""
        lines = content.split("\n")
        if len(lines) <= self.MAX_OUTPUT_LINES:
            return content

        # Keep first and last portions
        keep_start = self.MAX_OUTPUT_LINES // 2
        keep_end = self.MAX_OUTPUT_LINES // 2

        truncated_lines = (
            lines[:keep_start]
            + [f"\n... [{len(lines) - self.MAX_OUTPUT_LINES} lines omitted] ...\n"]
            + lines[-keep_end:]
        )

        # Save full output to temp file
        self._truncation_counter += 1
        save_path = f"/tmp/tool_output_{self._truncation_counter}.txt"
        try:
            with open(save_path, "w") as f:
                f.write(content)
        except Exception as e:
            logger.warning(f"Failed to save truncated output: {e}")
            save_path = "(save failed)"

        truncated = "\n".join(truncated_lines)
        truncated += TRUNCATION_NOTICE.format(
            original_lines=len(lines),
            kept_lines=self.MAX_OUTPUT_LINES,
            save_path=save_path,
        )

        return truncated

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Wrap tool calls to truncate oversized outputs."""
        result = await handler(request)

        if isinstance(result, ToolMessage) and result.content:
            original_content = str(result.content)
            truncated_content = self._truncate_output(original_content)

            if truncated_content != original_content:
                result.content = truncated_content
                logger.info(
                    f"Truncated tool output: {len(original_content)} -> {len(truncated_content)} chars"
                )

            # Update token estimate
            self._estimated_tokens += self._estimate_tokens(truncated_content)

        return result

    def after_model(
        self,
        state: AgentState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Check context usage after model response and warn if high."""
        # Update estimate with model response (new API - messages in state)
        messages = state.get("messages", [])
        # Only count the last message (the new one from model)
        if messages:
            last_msg = messages[-1]
            if hasattr(last_msg, "content"):
                self._estimated_tokens += self._estimate_tokens(str(last_msg.content))

        # Check if we should warn
        usage_percent = (self._estimated_tokens / self.max_context_tokens) * 100

        if usage_percent >= self.WARN_THRESHOLD_PERCENT and not self._warning_shown:
            self._warning_shown = True
            logger.warning(f"Context budget at {usage_percent:.0f}%")
            return {
                "messages": [
                    HumanMessage(content=CONTEXT_WARNING.format(percent=int(usage_percent)))
                ]
            }

        return None


# =============================================================================
# API Error Recovery Middleware
# =============================================================================

CONTENT_FILTER_RECOVERY = """
**Your last request was blocked by the model's content policy filter.**

This is likely a false positive. Try:
1. Rephrasing your request using more neutral/technical language
2. Breaking the task into smaller steps
3. Avoiding phrases that might seem like you're trying to replicate or copy content

Continue with a different approach.
"""

INVALID_IMAGE_RECOVERY = """
**The image data you tried to send was invalid.**

For video/media processing tasks:
- Use programmatic approaches (OpenCV, ffmpeg, PIL) instead of vision APIs
- Extract data computationally rather than visually analyzing frames
- Process files as binary data, not as images to "look at"

Continue with a programmatic approach.
"""

CONTEXT_OVERFLOW_RECOVERY = """
**Your request exceeded the model's context window.**

To recover:
1. Summarize your findings so far in a few sentences
2. Focus on the specific next step needed
3. Avoid re-reading large files - reference your summary instead
4. Be more concise in your responses

Continue with a summarized context.
"""


class APIErrorRecoveryMiddleware(AgentMiddleware):
    """Middleware that catches API errors and helps the agent recover.

    Instead of crashing on certain API errors, this middleware:
    1. Catches the error
    2. Injects a helpful recovery message
    3. Lets the agent try a different approach

    Handles:
    - OpenAI BadRequestError with specific codes
    - Anthropic InvalidRequestError
    - Context length exceeded errors
    - Content policy violations
    - Invalid image data errors
    """

    # Known error codes that we can recover from
    CONTENT_FILTER_CODES = {"content_filter", "invalid_prompt", "content_policy_violation"}
    CONTEXT_OVERFLOW_CODES = {"context_length_exceeded", "string_above_max_length"}
    INVALID_IMAGE_CODES = {"invalid_image_format", "invalid_base64"}

    def _classify_error(self, error: Exception) -> str | None:
        """Classify an error into a recoverable category.

        Returns:
            One of: 'content_filter', 'context_overflow', 'invalid_image', or None
        """
        # Try to get error code from OpenAI-style exceptions
        error_code = getattr(error, "code", None)
        if error_code:
            if error_code in self.CONTENT_FILTER_CODES:
                return "content_filter"
            if error_code in self.CONTEXT_OVERFLOW_CODES:
                return "context_overflow"
            if error_code in self.INVALID_IMAGE_CODES:
                return "invalid_image"

        # Fall back to message inspection for errors without codes
        # This handles Anthropic errors and edge cases
        error_msg = str(error).lower()

        # Content filter patterns
        if any(p in error_msg for p in ["usage policy", "content filter", "flagged", "content_policy"]):
            return "content_filter"

        # Context overflow patterns
        if any(p in error_msg for p in [
            "context_length_exceeded",
            "prompt is too long",
            "exceeds the context window",
            "maximum context length",
        ]):
            return "context_overflow"

        # Invalid image patterns
        if any(p in error_msg for p in [
            "invalid image",
            "not represent a valid image",
            "could not process image",
        ]):
            return "invalid_image"

        return None

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Wrap model calls to catch and recover from API errors."""
        try:
            return await handler(request)
        except Exception as e:
            error_type = self._classify_error(e)

            if error_type == "content_filter":
                logger.warning(f"Content filter triggered (recoverable): {e}")
                return {"messages": [HumanMessage(content=CONTENT_FILTER_RECOVERY)]}

            if error_type == "invalid_image":
                logger.warning(f"Invalid image error (recoverable): {e}")
                return {"messages": [HumanMessage(content=INVALID_IMAGE_RECOVERY)]}

            if error_type == "context_overflow":
                logger.warning(f"Context overflow (recoverable): {e}")
                return {"messages": [HumanMessage(content=CONTEXT_OVERFLOW_RECOVERY)]}

            # Not a recoverable error - re-raise
            raise
