"""Step-into subagent support: context stack and conversation tracking.

This module is imported lazily (not at CLI startup) to avoid adding
import-time overhead to the lightweight config path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from deepagents_cli.config import SessionState

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class ConversationContext:
    """A single conversation context in the stack.

    Used for tracking nested subagent conversations when
    the user "steps into" a subagent.
    """

    thread_id: str
    subagent_type: str  # "root" for main conversation, or subagent type name
    task_description: str  # Original task from parent (empty for root)
    summary_path: Path | None  # Where summary will be written (None for root)
    parent_tool_call_id: str | None  # For ToolMessage to parent
    parent_auto_approve: bool | None = None  # Saved parent auto_approve on push


class StepIntoSessionState(SessionState):
    """SessionState with context stack for step-into subagent navigation.

    Extends the base SessionState with a stack of ConversationContext objects
    that track nested subagent sessions.  The root context is always at
    index 0.
    """

    def __init__(
        self,
        *,
        auto_approve: bool = False,
        no_splash: bool = False,
        thread_id: str | None = None,
    ) -> None:
        """Initialize with a root conversation context.

        Args:
            auto_approve: Whether to auto-approve tool calls.
            no_splash: Whether to skip the splash screen.
            thread_id: Optional thread ID override (generates one if None).
        """
        super().__init__(auto_approve=auto_approve, no_splash=no_splash)
        # Replace the plain thread_id with a context stack
        root_tid = thread_id or self.thread_id
        self.context_stack: list[ConversationContext] = [
            ConversationContext(
                thread_id=root_tid,
                subagent_type="root",
                task_description="",
                summary_path=None,
                parent_tool_call_id=None,
            )
        ]

    @property
    def thread_id(self) -> str:
        """Get the current context's thread_id."""
        if hasattr(self, "context_stack") and self.context_stack:
            return self.context_stack[-1].thread_id
        # During __init__ before context_stack exists
        return self.__dict__.get("thread_id", "")

    @thread_id.setter
    def thread_id(self, value: str) -> None:
        """Set the current context's thread_id."""
        if hasattr(self, "context_stack") and self.context_stack:
            self.context_stack[-1].thread_id = value
        else:
            # During __init__ before context_stack exists
            self.__dict__["thread_id"] = value

    @property
    def current_context(self) -> ConversationContext:
        """Get the current (topmost) conversation context."""
        return self.context_stack[-1]

    @property
    def depth(self) -> int:
        """Get the current nesting depth (0 = root)."""
        return len(self.context_stack) - 1

    @property
    def is_in_subagent(self) -> bool:
        """Check if currently in a stepped-into subagent context."""
        return self.depth > 0

    def push_context(self, ctx: ConversationContext) -> None:
        """Push a new context onto the stack (entering a subagent).

        Saves the current auto_approve state so it can be restored on pop.
        Resets auto_approve to False for the new context.
        """
        ctx.parent_auto_approve = self.auto_approve
        self.auto_approve = False
        self.context_stack.append(ctx)

    def pop_context(self) -> ConversationContext:
        """Pop the current context from the stack (returning from subagent).

        Restores the parent's auto_approve state.

        Returns:
            The popped context.

        Raises:
            ValueError: If trying to pop the root context.
        """
        if self.depth == 0:
            msg = "Cannot pop root context"
            raise ValueError(msg)
        ctx = self.context_stack.pop()
        if ctx.parent_auto_approve is not None:
            self.auto_approve = ctx.parent_auto_approve
        return ctx

    def reset_to_root(self) -> None:
        """Reset to a fresh root context (used by /clear)."""
        from deepagents_cli.sessions import generate_thread_id

        self.context_stack = [
            ConversationContext(
                thread_id=generate_thread_id(),
                subagent_type="root",
                task_description="",
                summary_path=None,
                parent_tool_call_id=None,
            )
        ]
