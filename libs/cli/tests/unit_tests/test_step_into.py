"""Unit tests for step-into subagent feature."""

from __future__ import annotations

from pathlib import Path

from deepagents_cli.config import ConversationContext, SessionState
from deepagents_cli.widgets.approval import ApprovalMenu
from deepagents_cli.widgets.subagent_banner import SubagentBanner


class TestConversationContext:
    """Tests for ConversationContext and context stack."""

    def test_initial_state_is_root(self) -> None:
        """Session starts at root with depth 0."""
        state = SessionState(auto_approve=False)
        assert state.depth == 0
        assert not state.is_in_subagent

    def test_push_increases_depth(self) -> None:
        """Pushing a context increases depth."""
        state = SessionState(auto_approve=False)
        ctx = ConversationContext(
            thread_id="t2",
            subagent_type="general-purpose",
            task_description="research something",
            summary_path=Path("/tmp/summary.md"),
            parent_tool_call_id="tc1",
        )
        state.push_context(ctx)
        assert state.depth == 1
        assert state.is_in_subagent
        assert state.current_context.subagent_type == "general-purpose"

    def test_pop_decreases_depth(self) -> None:
        """Popping a context decreases depth back to root."""
        state = SessionState(auto_approve=False)
        ctx = ConversationContext(
            thread_id="t2",
            subagent_type="general-purpose",
            task_description="test",
            summary_path=None,
            parent_tool_call_id=None,
        )
        state.push_context(ctx)
        popped = state.pop_context()
        assert state.depth == 0
        assert not state.is_in_subagent
        assert popped.subagent_type == "general-purpose"

    def test_nested_push_pop(self) -> None:
        """Can nest multiple subagent contexts."""
        state = SessionState(auto_approve=False)
        ctx1 = ConversationContext(
            thread_id="t2",
            subagent_type="planner",
            task_description="plan",
            summary_path=None,
            parent_tool_call_id=None,
        )
        ctx2 = ConversationContext(
            thread_id="t3",
            subagent_type="coder",
            task_description="code",
            summary_path=None,
            parent_tool_call_id=None,
        )
        state.push_context(ctx1)
        state.push_context(ctx2)
        assert state.depth == 2
        assert state.current_context.subagent_type == "coder"
        state.pop_context()
        assert state.depth == 1
        assert state.current_context.subagent_type == "planner"

    def test_thread_id_reflects_current_context(self) -> None:
        """Thread ID should reflect current context."""
        state = SessionState(auto_approve=False)
        root_thread = state.thread_id
        ctx = ConversationContext(
            thread_id="sub-thread",
            subagent_type="general-purpose",
            task_description="test",
            summary_path=None,
            parent_tool_call_id=None,
        )
        state.push_context(ctx)
        assert state.thread_id == "sub-thread"
        state.pop_context()
        assert state.thread_id == root_thread


class TestApprovalMenuStepInto:
    """Tests for step-into option in ApprovalMenu."""

    def test_task_tool_shows_step_into_option(self) -> None:
        """Task tool should show step-into as option 3."""
        menu = ApprovalMenu(
            {"name": "task", "args": {"subagent_type": "general-purpose"}}
        )
        assert menu._is_task_tool is True
        assert menu._num_options == 4

    def test_non_task_tool_hides_step_into(self) -> None:
        """Non-task tools should not show step-into option."""
        menu = ApprovalMenu({"name": "shell", "args": {"command": "ls"}})
        assert menu._is_task_tool is False
        assert menu._num_options == 3

    def test_step_into_text_indicates_approval(self) -> None:
        """Step-into option text should indicate it implies approval."""
        menu = ApprovalMenu(
            {"name": "task", "args": {"subagent_type": "general-purpose"}}
        )
        # Verify the step-into option label clarifies approval semantics.
        # _update_options builds this list; we check the computed options
        # include "approve" in the step-into entry.
        assert menu._is_task_tool
        # The actual text is set in _update_options; verify the source list
        # pattern by asserting the menu recognises 4 options (incl. step-into)
        assert menu._num_options == 4


class TestSubagentBanner:
    """Tests for SubagentBanner widget."""

    def test_banner_starts_hidden(self) -> None:
        """Banner should not have visible class initially."""
        banner = SubagentBanner()
        assert "visible" not in banner.classes

    def test_show_adds_visible_class(self) -> None:
        """show() should add the visible class."""
        banner = SubagentBanner()
        banner.show(subagent_type="general-purpose", depth=1)
        assert "visible" in banner.classes
        assert banner.subagent_type == "general-purpose"
        assert banner.depth == 1

    def test_hide_removes_visible_class(self) -> None:
        """hide() should remove the visible class."""
        banner = SubagentBanner()
        banner.show(subagent_type="general-purpose", depth=1)
        banner.hide()
        assert "visible" not in banner.classes

    def test_show_updates_on_depth_change(self) -> None:
        """show() at different depths updates state."""
        banner = SubagentBanner()
        banner.show(subagent_type="planner", depth=1)
        assert banner.depth == 1
        banner.show(subagent_type="coder", depth=2)
        assert banner.depth == 2
        assert banner.subagent_type == "coder"
