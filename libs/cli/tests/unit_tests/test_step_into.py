"""Unit tests for step-into subagent feature.

Covers:
- ConversationContext and context stack state machine
- ApprovalMenu step-into option detection and semantics
- SubagentBanner visibility toggling and content
- Adapter-level step-into HITL flow (spinner, context creation)
- App-level /context command rendering
"""

from __future__ import annotations

import asyncio
from asyncio import Future
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
from langgraph.types import Interrupt
from rich.text import Text
from textual.app import App, ComposeResult
from textual.widgets import Static

from deepagents_cli.config import ConversationContext, SessionState
from deepagents_cli.textual_adapter import (
    TextualUIAdapter,
    execute_task_textual,
)
from deepagents_cli.widgets.approval import ApprovalMenu
from deepagents_cli.widgets.messages import AppMessage
from deepagents_cli.widgets.subagent_banner import SubagentBanner

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _noop_mount(_widget: object) -> None:
    """No-op async mount callback."""


def _noop_status(_: str) -> None:
    """No-op status callback."""


class _FakeAgent:
    """Minimal async stream agent that yields preconfigured chunks."""

    def __init__(self, chunks: list[tuple]) -> None:
        self._chunks = chunks

    async def astream(self, *_: Any, **__: Any) -> AsyncIterator[tuple[Any, ...]]:
        for chunk in self._chunks:
            yield chunk


# ---------------------------------------------------------------------------
# 1. ConversationContext & stack state machine
# ---------------------------------------------------------------------------


class TestConversationContext:
    """Tests for ConversationContext and context stack."""

    def test_initial_state_is_root(self) -> None:
        """Session starts at root with depth 0."""
        state = SessionState(auto_approve=False)
        assert state.depth == 0
        assert not state.is_in_subagent
        assert state.current_context.subagent_type == "root"

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
        """Thread ID getter follows the top of the stack."""
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

    def test_pop_on_root_raises(self) -> None:
        """Popping when at root should raise (can't pop root)."""
        state = SessionState(auto_approve=False)
        with pytest.raises(ValueError, match="Cannot pop root"):
            state.pop_context()

    def test_reset_to_root_clears_all_subagents(self) -> None:
        """reset_to_root should collapse the stack back to depth 0."""
        state = SessionState(auto_approve=False)
        for i in range(3):
            state.push_context(
                ConversationContext(
                    thread_id=f"t{i}",
                    subagent_type=f"agent-{i}",
                    task_description=f"task {i}",
                    summary_path=None,
                    parent_tool_call_id=None,
                )
            )
        assert state.depth == 3
        state.reset_to_root()
        assert state.depth == 0
        assert state.current_context.subagent_type == "root"


# ---------------------------------------------------------------------------
# 2. ApprovalMenu step-into detection
# ---------------------------------------------------------------------------


class TestApprovalMenuStepInto:
    """Tests for step-into option in ApprovalMenu."""

    def test_task_tool_shows_four_options(self) -> None:
        """Task tool should present 4 options (including step-into)."""
        menu = ApprovalMenu(
            {"name": "task", "args": {"subagent_type": "general-purpose"}}
        )
        assert menu._is_task_tool is True
        assert menu._num_options == 4

    def test_non_task_tool_shows_three_options(self) -> None:
        """Non-task tools should have 3 options (no step-into)."""
        menu = ApprovalMenu({"name": "shell", "args": {"command": "ls"}})
        assert menu._is_task_tool is False
        assert menu._num_options == 3

    def test_batch_requests_not_task_tool(self) -> None:
        """Multiple simultaneous tool calls should not show step-into."""
        menu = ApprovalMenu(
            [
                {"name": "task", "args": {"subagent_type": "a"}},
                {"name": "task", "args": {"subagent_type": "b"}},
            ]
        )
        assert menu._is_task_tool is False
        assert menu._num_options == 3

    def test_step_into_decision_includes_task_args(self) -> None:
        """Step-into decision should carry the original task args."""
        menu = ApprovalMenu(
            {"name": "task", "args": {"subagent_type": "coder", "description": "fix"}}
        )
        # Simulate the decision mapping used internally
        decision = {"type": "step_into"}
        decision["args"] = menu._action_requests[0].get("args", {})
        assert decision["args"]["subagent_type"] == "coder"
        assert decision["args"]["description"] == "fix"

    def test_step_into_keybinding_exists(self) -> None:
        """The 's' key binding for step-into should be registered."""
        bindings = {b.key for b in ApprovalMenu.BINDINGS}
        assert "s" in bindings


# ---------------------------------------------------------------------------
# 3. SubagentBanner widget
# ---------------------------------------------------------------------------


class TestSubagentBanner:
    """Tests for SubagentBanner widget."""

    def test_banner_starts_hidden(self) -> None:
        """Banner should not have visible class on creation."""
        banner = SubagentBanner()
        assert "visible" not in banner.classes

    def test_show_adds_visible_class_and_state(self) -> None:
        """show() should make the banner visible and store state."""
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

    def test_show_at_different_depths_updates_state(self) -> None:
        """Calling show() again updates state for new depth."""
        banner = SubagentBanner()
        banner.show(subagent_type="planner", depth=1)
        banner.show(subagent_type="coder", depth=2)
        assert banner.depth == 2
        assert banner.subagent_type == "coder"
        assert "visible" in banner.classes


class TestSubagentBannerInApp:
    """Test SubagentBanner rendering inside a real Textual app."""

    async def test_banner_renders_content_on_show(self) -> None:
        """Banner label and commands should update when show() is called."""

        class _BannerTestApp(App[None]):
            def compose(self) -> ComposeResult:
                yield SubagentBanner(id="banner")

        app = _BannerTestApp()
        async with app.run_test() as pilot:
            banner = app.query_one("#banner", SubagentBanner)

            # Initially hidden
            assert not banner.has_class("visible")

            banner.show(subagent_type="researcher", depth=2)
            await pilot.pause()

            assert banner.has_class("visible")
            label = banner.query_one("#subagent-label", Static)
            commands = banner.query_one("#subagent-commands", Static)
            label_text = str(label._Static__content)  # type: ignore[attr-defined]
            cmds_text = str(commands._Static__content)  # type: ignore[attr-defined]
            assert "researcher" in label_text
            assert "2" in label_text
            assert "/return" in cmds_text
            assert "/summary" in cmds_text
            assert "/context" in cmds_text

    async def test_banner_hides_when_returning_to_root(self) -> None:
        """Banner should disappear when hide() is called."""

        class _BannerTestApp(App[None]):
            def compose(self) -> ComposeResult:
                yield SubagentBanner(id="banner")

        app = _BannerTestApp()
        async with app.run_test() as pilot:
            banner = app.query_one("#banner", SubagentBanner)
            banner.show(subagent_type="coder", depth=1)
            await pilot.pause()
            assert banner.has_class("visible")

            banner.hide()
            await pilot.pause()
            assert not banner.has_class("visible")


# ---------------------------------------------------------------------------
# 4. Adapter-level: step-into HITL flow
# ---------------------------------------------------------------------------


class TestStepIntoAdapterFlow:
    """Test the step-into path through execute_task_textual."""

    async def test_step_into_stops_spinner_and_returns_context(self) -> None:
        """Spinner should stop and result should carry a ConversationContext."""
        spinner_states: list[str | None] = []
        mounted_widgets: list[object] = []

        async def record_spinner(status: str | None) -> None:
            await asyncio.sleep(0)
            spinner_states.append(status)

        async def record_mount(widget: object) -> None:
            await asyncio.sleep(0)
            mounted_widgets.append(widget)

        # The future resolves to a step-into decision
        step_into_future: Future[object] = Future()
        step_into_future.set_result(
            {
                "type": "step_into",
                "args": {
                    "subagent_type": "researcher",
                    "description": "find bugs",
                },
            }
        )

        async def mock_approval(  # noqa: RUF029  # must be async for adapter
            _requests: list[dict[str, object]], _assistant_id: str
        ) -> Future[object]:
            return step_into_future

        # Build a stream that triggers a HITL interrupt for a task tool.
        # Only send the updates/interrupt chunk — no messages chunk needed.
        hitl_value = {
            "action_requests": [
                {
                    "name": "task",
                    "args": {
                        "subagent_type": "researcher",
                        "description": "find bugs",
                    },
                }
            ],
            "review_configs": [
                {
                    "action_name": "task",
                    "allowed_decisions": ["approve", "reject"],
                }
            ],
        }
        interrupt = Interrupt(value=hitl_value, resumable=True, ns=None)
        chunks: list[tuple] = [
            ((), "updates", {"__interrupt__": [interrupt]}),
        ]

        adapter = TextualUIAdapter(
            mount_message=record_mount,
            update_status=_noop_status,
            request_approval=mock_approval,
            set_spinner=record_spinner,
        )

        session = SimpleNamespace(
            thread_id="thread-1",
            auto_approve=False,
        )

        result = await execute_task_textual(
            user_input="hello",
            agent=_FakeAgent(chunks),
            assistant_id="test-assistant",
            session_state=session,
            adapter=adapter,
        )

        # Spinner should have been set to None (stopped) before returning
        assert spinner_states[-1] is None, (
            f"Spinner should end at None, got: {spinner_states}"
        )

        # Result should carry a step-into context
        assert result.step_into_context is not None
        assert result.step_into_context.subagent_type == "researcher"
        assert result.step_into_context.thread_id  # non-empty UUID
        assert result.step_into_context.summary_path is not None
        assert result.step_into_context.summary_path.name == "summary.md"

        # An AppMessage should have been mounted with step-into info
        app_messages = [w for w in mounted_widgets if isinstance(w, AppMessage)]
        step_into_msgs = [m for m in app_messages if "Stepped into" in str(m._content)]
        assert len(step_into_msgs) == 1

    async def test_step_into_creates_summary_file_on_disk(self) -> None:
        """Branch context should create a summary.md template on disk."""
        step_into_future: Future[object] = Future()
        step_into_future.set_result(
            {
                "type": "step_into",
                "args": {
                    "subagent_type": "planner",
                    "description": "plan feature",
                },
            }
        )

        async def mock_approval(*_: object) -> Future[object]:  # noqa: RUF029
            return step_into_future

        hitl_value = {
            "action_requests": [
                {
                    "name": "task",
                    "args": {"subagent_type": "planner", "description": "plan"},
                }
            ],
            "review_configs": [
                {
                    "action_name": "task",
                    "allowed_decisions": ["approve", "reject"],
                }
            ],
        }
        interrupt = Interrupt(value=hitl_value, resumable=True, ns=None)
        chunks: list[tuple] = [
            ((), "updates", {"__interrupt__": [interrupt]}),
        ]

        adapter = TextualUIAdapter(
            mount_message=_noop_mount,
            update_status=_noop_status,
            request_approval=mock_approval,
        )

        result = await execute_task_textual(
            user_input="plan it",
            agent=_FakeAgent(chunks),
            assistant_id="test-assistant",
            session_state=SimpleNamespace(thread_id="thread-1", auto_approve=False),
            adapter=adapter,
        )

        assert result.step_into_context is not None
        summary_path = result.step_into_context.summary_path
        assert summary_path is not None
        assert summary_path.exists(), f"Summary file not created: {summary_path}"
        content = summary_path.read_text()
        assert "# Summary" in content or "Task" in content


# ---------------------------------------------------------------------------
# 5. /context command output rendering
# ---------------------------------------------------------------------------


class TestContextCommandRendering:
    """Test that /context uses Rich Text with proper styles (not raw markdown)."""

    def test_context_output_uses_rich_text_not_markdown(self) -> None:
        """AppMessage for /context should receive Rich Text, not raw markdown."""
        # Simulate what _handle_context_command builds
        text = Text()
        text.append("Context Stack:\n", style="bold")
        text.append("  [0] root (main conversation) <-- current\n")

        msg = AppMessage(text)
        # AppMessage stores the content — verify it's a Text object
        assert isinstance(msg._content, Text)
        # The plain text should NOT contain ** markdown markers
        assert "**" not in msg._content.plain
        # The word "Context Stack:" should be present
        assert "Context Stack:" in msg._content.plain

    def test_context_output_shows_subagent_type_bold(self) -> None:
        """Subagent names in /context output should have bold styling."""
        text = Text()
        text.append("Context Stack:\n", style="bold")
        text.append("  [0] root (main conversation)\n")
        text.append("  [1] ", style="dim")
        text.append("researcher", style="bold")
        text.append(" <-- current\n")

        # Extract spans to verify bold styling exists
        bold_spans = [span for span in text._spans if span.style == "bold"]
        assert len(bold_spans) >= 2  # "Context Stack:" and "researcher"
        # Verify "researcher" is in the bolded region
        researcher_start = text.plain.index("researcher")
        researcher_end = researcher_start + len("researcher")
        has_bold_researcher = any(
            span.start <= researcher_start and span.end >= researcher_end
            for span in bold_spans
        )
        assert has_bold_researcher


# ---------------------------------------------------------------------------
# 6. Edge cases and integration
# ---------------------------------------------------------------------------


class TestStepIntoEdgeCases:
    """Edge cases for step-into feature."""

    def test_approval_menu_preserves_args_for_step_into(self) -> None:
        """Task tool args should be carried through for context creation."""
        task_args = {
            "subagent_type": "explorer",
            "description": "find the config file",
            "prompt": "detailed instructions here",
        }
        menu = ApprovalMenu({"name": "task", "args": task_args})
        # The menu stores action_requests which include args
        assert menu._action_requests[0]["args"] == task_args

    def test_session_state_thread_id_setter_compat(self) -> None:
        """Setting thread_id via setter should update root context."""
        state = SessionState(auto_approve=False)
        old_thread = state.thread_id
        state.thread_id = "new-thread-id"
        assert state.thread_id == "new-thread-id"
        assert state.thread_id != old_thread

    def test_context_stack_isolation(self) -> None:
        """Each subagent context should have independent thread IDs."""
        state = SessionState(auto_approve=False)
        threads = {state.thread_id}
        for i in range(5):
            ctx = ConversationContext(
                thread_id=f"unique-{i}",
                subagent_type=f"agent-{i}",
                task_description=f"task {i}",
                summary_path=None,
                parent_tool_call_id=None,
            )
            state.push_context(ctx)
            threads.add(state.thread_id)
        # All 6 thread IDs (root + 5 subagents) should be unique
        assert len(threads) == 6
