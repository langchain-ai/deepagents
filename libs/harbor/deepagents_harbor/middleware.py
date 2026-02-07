"""Custom middleware for Harbor benchmark runs."""

import logging
import re
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import Annotated, Any, NotRequired

from langchain.agents.middleware import hook_config
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


def _message_has_pending_tool_calls(message: Any) -> bool:
    """Return True when an AI message still expects tool results next.

    Anthropic/OpenAI tool-call payloads can appear in different shapes:
    - `AIMessage.tool_calls`
    - `AIMessage.additional_kwargs["tool_calls"]`
    - `AIMessage.content` blocks (`type == "tool_call"` / `"tool_use"`)

    We must detect all of these before injecting HumanMessages, otherwise we
    can violate provider ordering requirements:
    `assistant(tool_use) -> user/human -> tool_result` (invalid).
    """
    if not isinstance(message, AIMessage):
        return False

    if getattr(message, "tool_calls", None):
        return True

    additional_kwargs = getattr(message, "additional_kwargs", None)
    if isinstance(additional_kwargs, dict) and additional_kwargs.get("tool_calls"):
        return True

    content = getattr(message, "content", None)
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") in {"tool_call", "tool_use"}:
                return True

    return False


# System-level reflection prompt — injected when agent appears to be finishing.
# Framed as a thinking exercise rather than a command, which reasoning models
# engage with more naturally than imperative checklists.
COMPLETION_REFLECTION = """STOP. You must verify your work before finishing.

RIGHT NOW, do this:
1. Run the EXACT command or script that produces the deliverable. Read the actual output.
2. If the task specifies expected values, formats, or constraints — compare yours against the spec.
3. If a server/process must be running for evaluation, confirm it responds (curl, ps, etc.).

You MUST use a tool call to verify. Do not just review your code in your head. If you find any issue, fix it."""


class PreCompletionCheckState(AgentState):
    """State schema for PreCompletionCheckMiddleware.

    Attributes:
        checklist_reminder_shown: Whether we've already shown the checklist reminder.
            Private so it's not included in the final agent state.
    """

    checklist_reminder_shown: NotRequired[Annotated[bool, PrivateStateAttr]]
    activity_enforcement_shown: NotRequired[Annotated[bool, PrivateStateAttr]]
    test_enforcement_shown: NotRequired[Annotated[bool, PrivateStateAttr]]


class PreCompletionCheckMiddleware(AgentMiddleware):
    """Middleware that injects a checklist reminder before the agent finishes.

    Detects when the agent is about to finish (produces an AI message without
    tool calls) and injects a reminder to verify the pre-completion checklist.

    Only triggers ONCE per conversation to avoid infinite loops.
    """

    state_schema = PreCompletionCheckState

    def __init__(self, model: Any = None) -> None:
        """Track execute commands so we can enforce a real verifier run.

        Args:
            model: Optional chat model instance. If provided and the model has a
                mutable ``reasoning_effort`` attribute, the middleware will
                temporarily boost reasoning to ``"xhigh"`` for the reflection
                turn — the most important self-review moment.
        """
        self._model = model
        self._prior_effort: str | None = None
        self._executed_commands: list[str] = []
        self._written_paths: list[str] = []

    def before_agent(
        self,
        state: PreCompletionCheckState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Reset per-run tracking state."""
        self._executed_commands = []
        self._written_paths = []
        return None

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Capture execute commands so finish gating can verify test execution."""
        tool_name = request.tool_call.get("name", "")
        if tool_name == "execute":
            args = request.tool_call.get("args", {})
            command = args.get("command") or args.get("cmd")
            if isinstance(command, str) and command.strip():
                self._executed_commands.append(command)
        elif tool_name in {"write_file", "edit_file"}:
            args = request.tool_call.get("args", {})
            file_path = args.get("file_path") or args.get("path")
            if isinstance(file_path, str) and file_path.strip():
                self._written_paths.append(file_path)
        return await handler(request)

    @staticmethod
    def _extract_recommended_test_commands(messages: list[Any]) -> list[str]:
        """Extract test commands from injected environment context messages."""
        discovered: list[str] = []
        for message in messages:
            if not isinstance(message, HumanMessage):
                continue
            content = message.content if isinstance(message.content, str) else str(message.content)
            if not content:
                continue
            # Example format from LocalContextMiddleware:
            # "**Run Tests**: `pytest`"
            for match in re.findall(r"\*\*Run Tests\*\*:\s*`([^`]+)`", content):
                cmd = match.strip()
                if cmd and cmd not in discovered:
                    discovered.append(cmd)
            if "/tests/test_outputs.py" in content:
                if "python /tests/test_outputs.py" not in discovered:
                    discovered.append("python /tests/test_outputs.py")
        return discovered

    @staticmethod
    def _has_executed_verifier(
        commands: list[str],
        recommended: list[str],
    ) -> bool:
        """Return True if any executed command appears to run tests/verifier."""
        for command in commands:
            normalized = command.lower()
            if any(
                needle in normalized
                for needle in (
                    "pytest",
                    "test_outputs.py",
                    "make test",
                    "uv run",
                )
            ):
                return True
            for rec in recommended:
                if rec.lower() in normalized:
                    return True
        return False

    @hook_config(can_jump_to=["model"])
    def after_model(
        self,
        state: PreCompletionCheckState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Check if agent is finishing without tool calls and inject reminder.

        Triggers when:
        1. The AI response has no tool calls (agent is "finishing")
        2. We haven't already shown the checklist reminder

        Uses jump_to="model" to route back to the model node so the agent
        actually sees and acts on the checklist. Without this, the graph
        routes to END because the last AIMessage has no tool_calls.

        Args:
            state: Current agent state
            runtime: Runtime context

        Returns:
            State update with reminder message and jump_to, or None
        """
        # Get the last message from state (new API - response is in state)
        messages = state.get("messages", [])
        if not messages:
            return None

        last_message = messages[-1]

        # Check if it's an AI message without tool calls (agent is finishing)
        if isinstance(last_message, AIMessage):
            has_tool_calls = _message_has_pending_tool_calls(last_message)

            if not has_tool_calls:
                if not state.get("checklist_reminder_shown", False):
                    # Agent is about to finish — inject a reflection prompt to
                    # force reconsideration of blind spots before completing.
                    # Boost reasoning to xhigh for this critical self-review turn.
                    if self._model and hasattr(self._model, "reasoning_effort"):
                        self._prior_effort = getattr(self._model, "reasoning_effort", None)
                        self._model.reasoning_effort = "xhigh"
                        logger.info(
                            "PreCompletionCheck: boosted reasoning %s → xhigh for reflection",
                            self._prior_effort,
                        )
                    logger.info("PreCompletionCheck: agent finishing, injecting reflection prompt")
                    return {
                        "messages": [HumanMessage(content=COMPLETION_REFLECTION)],
                        "checklist_reminder_shown": True,
                        "jump_to": "model",
                    }

                # Prevent read-only early exits (observed in failed traces where the
                # agent reads files and finishes without making any changes or running
                # any command that could produce the required artifacts).
                if not state.get("activity_enforcement_shown", False):
                    if not self._executed_commands and not self._written_paths:
                        return {
                            "messages": [
                                HumanMessage(
                                    content=(
                                        "You are attempting to finish after read-only exploration. "
                                        "Do actual task work now: run commands and/or write required "
                                        "files, then continue."
                                    )
                                )
                            ],
                            "activity_enforcement_shown": True,
                            "jump_to": "model",
                        }

                # Checklist was already shown. Enforce one verifier/test execution
                # before allowing completion when a recommended test command exists.
                if not state.get("test_enforcement_shown", False):
                    recommended = self._extract_recommended_test_commands(messages)
                    if recommended and not self._has_executed_verifier(
                        self._executed_commands,
                        recommended,
                    ):
                        suggested = " OR ".join(f"`{cmd}`" for cmd in recommended[:2])
                        return {
                            "messages": [
                                HumanMessage(
                                    content=(
                                        "You are attempting to finish without running verifier/tests. "
                                        f"Run one now ({suggested}), inspect the output, fix issues if any, "
                                        "then continue."
                                    )
                                )
                            ],
                            "test_enforcement_shown": True,
                            "jump_to": "model",
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

LOOP_REFLECTION_PROMPT = """
You've edited `{file_path}` {count} times. This strongly suggests you're stuck in a loop.

Before your next action, briefly reflect:
1. What approach have you been trying?
2. Why isn't it working?
3. What's a fundamentally different approach?

State your new approach, then continue working autonomously. Do not ask for permission - just pivot and keep going.
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

    At the soft threshold, appends a warning to the tool output.
    At the hard threshold, forces a reflection prompt via jump_to to ensure
    the agent acknowledges being stuck and pivots to a different approach.
    """

    state_schema = LoopDetectionState

    # Default thresholds for warnings
    DEFAULT_SOFT_WARNING_THRESHOLD = 7
    DEFAULT_HARD_REFLECTION_THRESHOLD = 12

    def __init__(
        self,
        *,
        soft_warning_threshold: int = DEFAULT_SOFT_WARNING_THRESHOLD,
        hard_reflection_threshold: int = DEFAULT_HARD_REFLECTION_THRESHOLD,
    ) -> None:
        """Initialize loop detection thresholds.

        Args:
            soft_warning_threshold: Edit count before appending a soft warning.
            hard_reflection_threshold: Edit count before forcing reflection.
        """
        if soft_warning_threshold < 1:
            msg = "soft_warning_threshold must be >= 1"
            raise ValueError(msg)
        if hard_reflection_threshold <= soft_warning_threshold:
            msg = "hard_reflection_threshold must be greater than soft_warning_threshold"
            raise ValueError(msg)

        self.soft_warning_threshold = soft_warning_threshold
        self.hard_reflection_threshold = hard_reflection_threshold
        self._file_edit_counts: dict[str, int] = defaultdict(int)
        self._warnings_shown: set[str] = set()
        self._pending_reflection: tuple[str, int] | None = None  # (file_path, count)

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

            # Soft warning at threshold - append to tool output
            warning_key = f"{file_path}:soft"
            if count == self.soft_warning_threshold and warning_key not in self._warnings_shown:
                self._warnings_shown.add(warning_key)
                logger.warning(f"Loop detection: {file_path} edited {count} times")
                if isinstance(result, ToolMessage):
                    result.content = str(result.content) + LOOP_WARNING_SOFT.format(
                        file_path=file_path, count=count
                    )

            # Hard reflection at threshold - flag for after_model hook
            reflection_key = f"{file_path}:hard"
            if count == self.hard_reflection_threshold and reflection_key not in self._warnings_shown:
                self._warnings_shown.add(reflection_key)
                logger.warning(f"Loop detection HARD: {file_path} edited {count} times - forcing reflection")
                self._pending_reflection = (file_path, count)

        return result

    @hook_config(can_jump_to=["model"])
    def after_model(
        self,
        state: LoopDetectionState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Inject reflection prompt after hard threshold and jump back to model.

        When we detect the agent is stuck in a loop, we inject a HumanMessage
        forcing them to reflect on their approach and pivot. The jump_to ensures
        the agent continues working rather than stopping.
        """
        if self._pending_reflection:
            # IMPORTANT: Don't inject a HumanMessage if the AI made tool calls!
            # The next message MUST be tool_result, not a human message.
            # Injecting here would break: AI(tool_call) -> HumanMessage -> tool_result
            # Which violates Anthropic's API contract.
            messages = state.get("messages", [])
            if messages:
                last_msg = messages[-1]
                if _message_has_pending_tool_calls(last_msg):
                    # AI made tool calls - don't inject yet, wait until after tools complete
                    logger.info("Loop detection: deferring reflection (pending tool calls)")
                    return None

            file_path, count = self._pending_reflection
            self._pending_reflection = None
            logger.info(f"Injecting reflection prompt for {file_path} after {count} edits")
            return {
                "messages": [
                    HumanMessage(content=LOOP_REFLECTION_PROMPT.format(
                        file_path=file_path, count=count
                    ))
                ],
                "jump_to": "model",
            }
        return None


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
    4. Estimates usage from live message state (avoids cumulative drift)
    """

    DEFAULT_MAX_OUTPUT_LINES = 200  # Max lines per tool output
    DEFAULT_WARN_THRESHOLD_PERCENT = 70  # Warn at 70% of context

    def __init__(
        self,
        max_context_tokens: int = 128000,
        *,
        max_output_lines: int = DEFAULT_MAX_OUTPUT_LINES,
        warn_threshold_percent: int = DEFAULT_WARN_THRESHOLD_PERCENT,
    ) -> None:
        """Initialize context budget controls.

        Args:
            max_context_tokens: Estimated total token budget.
            max_output_lines: Max lines to keep per tool output before truncation.
            warn_threshold_percent: Usage percent at which to inject warning.
        """
        if max_output_lines < 1:
            msg = "max_output_lines must be >= 1"
            raise ValueError(msg)
        if warn_threshold_percent < 1 or warn_threshold_percent > 100:
            msg = "warn_threshold_percent must be between 1 and 100"
            raise ValueError(msg)

        self.max_context_tokens = max_context_tokens
        self.max_output_lines = max_output_lines
        self.warn_threshold_percent = warn_threshold_percent
        self._estimated_tokens = 0
        self._warning_shown = False
        self._truncation_counter = 0

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimate: ~4 chars per token."""
        return len(text) // 4

    def _estimate_message_tokens(self, message: Any) -> int:
        """Estimate tokens for one message, including content and tool-call payloads."""
        estimate = 0

        content = getattr(message, "content", None)
        if content:
            estimate += self._estimate_tokens(str(content))

        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            estimate += self._estimate_tokens(str(tool_calls))

        additional_kwargs = getattr(message, "additional_kwargs", None)
        if isinstance(additional_kwargs, dict):
            pending_tool_calls = additional_kwargs.get("tool_calls")
            if pending_tool_calls:
                estimate += self._estimate_tokens(str(pending_tool_calls))

        return estimate

    def _estimate_state_tokens(self, messages: list[Any]) -> int:
        """Estimate total tokens from the current live message list."""
        return sum(self._estimate_message_tokens(message) for message in messages)

    def _truncate_output(self, content: str) -> str:
        """Truncate oversized output and save full version to temp file."""
        lines = content.split("\n")
        if len(lines) <= self.max_output_lines:
            return content

        # Keep first and last portions
        keep_start = self.max_output_lines // 2
        keep_end = self.max_output_lines // 2

        truncated_lines = (
            lines[:keep_start]
            + [f"\n... [{len(lines) - self.max_output_lines} lines omitted] ...\n"]
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
            kept_lines=self.max_output_lines,
            save_path=save_path,
        )

        return truncated

    def _process_tool_message_output(self, message: ToolMessage) -> None:
        """Truncate a single tool message output in-place when oversized."""
        if not message.content:
            return

        original_content = str(message.content)
        truncated_content = self._truncate_output(original_content)

        if truncated_content != original_content:
            message.content = truncated_content
            logger.info(
                f"Truncated tool output: {len(original_content)} -> {len(truncated_content)} chars"
            )

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Wrap tool calls to truncate oversized outputs."""
        result = await handler(request)

        if isinstance(result, ToolMessage):
            self._process_tool_message_output(result)
            return result

        if isinstance(result, Command):
            update = result.update
            if isinstance(update, dict):
                for message in update.get("messages", []):
                    if isinstance(message, ToolMessage):
                        self._process_tool_message_output(message)

        return result

    def after_model(
        self,
        state: AgentState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Check context usage after model response and warn if high."""
        messages = state.get("messages", [])
        self._estimated_tokens = self._estimate_state_tokens(messages)

        # Check if we should warn
        usage_percent = (self._estimated_tokens / self.max_context_tokens) * 100

        if usage_percent >= self.warn_threshold_percent and not self._warning_shown:
            # IMPORTANT: Don't inject a HumanMessage if the AI made tool calls!
            # The next message MUST be tool_result, not a human message.
            # Injecting here would break: AI(tool_call) -> HumanMessage -> tool_result
            # Which violates Anthropic's API contract.
            if messages:
                last_msg = messages[-1]
                if _message_has_pending_tool_calls(last_msg):
                    # AI made tool calls - don't inject, wait until after tools complete
                    logger.warning(f"Context budget at {usage_percent:.0f}% but skipping warning (pending tool calls)")
                    return None

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

CONTINUE_AFTER_RECOVERY = """Now continue working on the original task using a different approach."""


class LocalContextMiddleware(AgentMiddleware):
    """Middleware that injects sandbox environment context and re-injects after summarization.

    Uses before_agent to inject a one-time HumanMessage containing
    environment discovery results (current directory, language, tools,
    test files, directory tree, etc.). This keeps the system prompt lean
    with only static rules, while dynamic per-task context is injected
    into the message history where it can be summarized if needed.

    When SummarizationMiddleware evicts old messages (including the env
    context), before_model detects this and re-injects the context so
    the agent retains sandbox awareness throughout long conversations.
    """

    _ENV_CONTEXT_SOURCE = "environment_context"

    def __init__(self, environment_context: str) -> None:
        """Initialize with pre-formatted environment context string.

        Args:
            environment_context: Formatted string describing the sandbox
                environment (directory, tools, files, etc.)
        """
        self._environment_context = environment_context

    def before_agent(
        self,
        state: AgentState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Inject environment context as a HumanMessage at agent start.

        Fires once before the agent begins processing. The environment
        context becomes part of the message history, giving the model
        awareness of the sandbox without bloating the system prompt.

        Tags the message with lc_source so before_model can detect if
        summarization evicted it.
        """
        if not self._environment_context:
            return None
        return {
            "messages": [HumanMessage(
                content=self._environment_context,
                additional_kwargs={"lc_source": self._ENV_CONTEXT_SOURCE},
            )],
        }

    def before_model(
        self,
        state: AgentState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Re-inject environment context if summarization evicted it.

        SummarizationMiddleware (earlier in the middleware stack) may
        replace old messages with a summary, evicting the env context
        HumanMessage. This hook detects that by checking for the tagged
        message and re-injects the full environment context so the agent
        retains awareness of the sandbox (working directory, test files,
        available tools, etc.).
        """
        if not self._environment_context:
            return None

        messages = state.get("messages", [])
        if not messages:
            return None

        # Don't inject if AI has pending tool calls (would break message ordering)
        last_msg = messages[-1]
        if _message_has_pending_tool_calls(last_msg):
            return None

        # Check if our tagged env context message still exists
        for msg in messages:
            if (
                isinstance(msg, HumanMessage)
                and msg.additional_kwargs.get("lc_source") == self._ENV_CONTEXT_SOURCE
            ):
                return None  # Still present, no action needed

        # Only re-inject if summarization has actually occurred
        has_summary = any(
            isinstance(msg, HumanMessage)
            and msg.additional_kwargs.get("lc_source") == "summarization"
            for msg in messages
        )
        if not has_summary:
            return None

        logger.info("Re-injecting environment context after summarization")
        return {
            "messages": [HumanMessage(
                content=(
                    "**[Environment context re-injected after conversation summarization]**\n\n"
                    + self._environment_context
                ),
                additional_kwargs={"lc_source": self._ENV_CONTEXT_SOURCE},
            )],
        }


class AdaptiveReasoningMiddleware(AgentMiddleware):
    """Switches reasoning effort from xhigh to high partway through the time budget.

    xhigh reasoning produces better first-attempt solutions (plans upfront,
    diagnoses errors, verifies robustly) but each API call is slower, yielding
    fewer total turns in a fixed wall-clock budget. This middleware starts with
    xhigh for the critical planning/implementation phase, then downgrades to
    high for the iterative debugging/verification phase where turn count matters
    more than per-turn reasoning depth.

    Only activates when the model's initial reasoning_effort is "xhigh".
    """

    def __init__(
        self,
        model: Any,
        total_budget_sec: float,
        *,
        switch_fraction: float = 0.5,
        fallback_effort: str = "high",
    ) -> None:
        """Initialize adaptive reasoning controls.

        Args:
            model: The chat model instance (must have a mutable reasoning_effort attr).
            total_budget_sec: Total wall-clock time budget in seconds.
            switch_fraction: Fraction of budget after which to downgrade (default 0.5).
            fallback_effort: Reasoning effort to switch to (default "high").
        """
        self._model = model
        self._total_budget_sec = total_budget_sec
        self._switch_fraction = switch_fraction
        self._fallback_effort = fallback_effort
        self._start_time: float | None = None
        self._switched = False
        self._initial_effort: str | None = None

    def before_agent(
        self,
        state: AgentState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Record the start time and initial reasoning effort."""
        self._start_time = time.monotonic()
        self._switched = False
        self._initial_effort = getattr(self._model, "reasoning_effort", None)
        if self._initial_effort and self._initial_effort != "xhigh":
            # Not starting at xhigh — nothing to adapt
            self._switched = True  # Disable further checks
        return None

    def before_model(
        self,
        state: AgentState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Check elapsed time and downgrade reasoning effort if past threshold."""
        if self._switched or self._start_time is None:
            return None

        elapsed = time.monotonic() - self._start_time
        threshold = self._total_budget_sec * self._switch_fraction

        if elapsed >= threshold:
            self._switched = True
            self._model.reasoning_effort = self._fallback_effort
            logger.info(
                "AdaptiveReasoning: %s → %s at %.0fs (%.0f%% of %.0fs budget)",
                self._initial_effort,
                self._fallback_effort,
                elapsed,
                (elapsed / self._total_budget_sec) * 100,
                self._total_budget_sec,
            )

        return None


class APIErrorRecoveryMiddleware(AgentMiddleware):
    """Middleware that catches API errors and helps the agent recover.

    Instead of crashing on certain API errors, this middleware:
    1. Catches the error
    2. Returns an AIMessage with recovery guidance
    3. Injects a follow-up HumanMessage to prompt the agent to continue

    This two-step approach ensures the agent doesn't stop after seeing the
    recovery message, but instead continues trying with a different approach.

    Handles recoverable API failures:
    - OpenAI/Anthropic content policy violations
    - Invalid image payload errors

    Context overflow is deliberately NOT recovered here. It must bubble up
    so Harbor can fail/retry the trial attempt with a fresh context.
    """

    # Known error codes that we can recover from
    CONTENT_FILTER_CODES = {"content_filter", "invalid_prompt", "content_policy_violation"}
    CONTEXT_OVERFLOW_CODES = {"context_length_exceeded", "string_above_max_length"}
    INVALID_IMAGE_CODES = {"invalid_image_format", "invalid_base64"}

    def __init__(self) -> None:
        """Initialize the middleware with recovery tracking state."""
        self._recovery_pending = False

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
            self._recovery_pending = False  # Reset on successful call attempt
            return await handler(request)
        except Exception as e:
            error_type = self._classify_error(e)

            if error_type == "content_filter":
                logger.warning(f"Content filter triggered (recoverable): {e}")
                self._recovery_pending = True
                return AIMessage(content=CONTENT_FILTER_RECOVERY)

            if error_type == "invalid_image":
                logger.warning(f"Invalid image error (recoverable): {e}")
                self._recovery_pending = True
                return AIMessage(content=INVALID_IMAGE_RECOVERY)

            if error_type == "context_overflow":
                logger.warning(f"Context overflow (non-recoverable, bubbling): {e}")
                raise

            # Not a recoverable error - re-raise
            raise

    @hook_config(can_jump_to=["model"])
    def after_model(
        self,
        state: AgentState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Inject a follow-up message after error recovery and jump back to model.

        When we recover from an API error, we return an AIMessage with guidance.
        But since that AIMessage has no tool_calls, the agent would normally stop.
        This hook:
        1. Injects a HumanMessage to prompt the agent to continue
        2. Uses jump_to='model' to explicitly route back to the model node
        """
        if self._recovery_pending:
            self._recovery_pending = False
            logger.info("Injecting continuation prompt after API error recovery")
            return {
                "messages": [HumanMessage(content=CONTINUE_AFTER_RECOVERY)],
                "jump_to": "model",  # Force loop to continue
            }
        return None
