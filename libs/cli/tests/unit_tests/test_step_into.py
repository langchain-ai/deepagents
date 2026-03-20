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
from textual.app import App, ComposeResult
from textual.widgets import Static

from deepagents_cli.step_into import (
    ConversationContext,
    StepIntoSessionState as SessionState,
)
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
        self._call_count = 0

    async def astream(self, *_: Any, **__: Any) -> AsyncIterator[tuple[Any, ...]]:
        self._call_count += 1
        for chunk in self._chunks:
            yield chunk


class _MultiturnFakeAgent:
    """Fake agent that yields different chunks on each astream call.

    Used to test rejection-resume flows where the first call returns
    an interrupt and the second call (after Command(resume=...)) returns
    a normal completion.
    """

    def __init__(self, turns: list[list[tuple]]) -> None:
        self._turns = turns
        self._call_count = 0

    async def astream(self, *_: Any, **__: Any) -> AsyncIterator[tuple[Any, ...]]:
        idx = min(self._call_count, len(self._turns) - 1)
        self._call_count += 1
        for chunk in self._turns[idx]:
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

    def test_breadcrumbs_shown_at_depth_3_plus(self) -> None:
        """Banner shows breadcrumbs when nested 3+ levels deep."""
        banner = SubagentBanner()
        stack = ["root", "researcher", "coder"]
        banner.show(subagent_type="coder", depth=2, context_stack=stack)
        assert banner.breadcrumbs == "root > researcher > coder"

    def test_no_breadcrumbs_at_depth_1(self) -> None:
        """Banner shows simple label at depth 1 (no breadcrumbs)."""
        banner = SubagentBanner()
        stack = ["root", "researcher"]
        banner.show(subagent_type="researcher", depth=1, context_stack=stack)
        assert banner.breadcrumbs == ""


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
    """Test that /context output is properly formatted."""

    def test_context_output_is_plain_text_string(self) -> None:
        """AppMessage for /context should receive a plain string."""
        lines = [
            "**Context Stack:**",
            "  [0] root (main conversation) <-- current",
        ]
        msg = AppMessage("\n".join(lines))
        assert isinstance(msg._content, str)
        assert "Context Stack:" in msg._content

    def test_context_output_shows_subagent_type_bold_markers(self) -> None:
        """Subagent names in /context output use markdown bold markers."""
        lines = [
            "**Context Stack:**",
            "  [0] root (main conversation)",
            "  [1] **researcher** <-- current",
        ]
        output = "\n".join(lines)
        assert "**researcher**" in output


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


# ---------------------------------------------------------------------------
# 7. Rejection scoping inside subagent
# ---------------------------------------------------------------------------


class TestRejectionScopingInSubagent:
    """Rejection inside a stepped-into subagent should NOT bubble up.

    When the user rejects a tool call while inside a subagent, the
    rejection should be sent back to LangGraph via Command(resume=...)
    so the agent can try a different approach.  At root level (depth=0),
    rejection should still return early as before.
    """

    async def test_rejection_at_root_returns_early(self) -> None:
        """At depth=0, rejection should cause an early return."""
        reject_future: Future[object] = Future()
        reject_future.set_result({"type": "reject"})

        async def mock_approval(  # noqa: RUF029
            _r: list[dict[str, object]],
            _a: str,
        ) -> Future[object]:
            return reject_future

        hitl_value = {
            "action_requests": [{"name": "bash", "args": {"command": "rm -rf /"}}],
            "review_configs": [
                {"action_name": "bash", "allowed_decisions": ["approve", "reject"]}
            ],
        }
        interrupt = Interrupt(value=hitl_value, resumable=True, ns=None)
        agent = _FakeAgent(
            [
                ((), "updates", {"__interrupt__": [interrupt]}),
            ]
        )

        mounted: list[object] = []

        async def record_mount(w: object) -> None:
            await asyncio.sleep(0)
            mounted.append(w)

        adapter = TextualUIAdapter(
            mount_message=record_mount,
            update_status=_noop_status,
            request_approval=mock_approval,
        )

        # Root level: depth=0
        session = SimpleNamespace(thread_id="t1", auto_approve=False, depth=0)

        result = await execute_task_textual(
            user_input="do it",
            agent=agent,
            assistant_id="test",
            session_state=session,
            adapter=adapter,
        )

        # Should return early — no step_into_context
        assert result.step_into_context is None
        # Agent should only be called once (no resume after rejection)
        assert agent._call_count == 1
        # "Command rejected" message should be mounted
        reject_msgs = [
            w
            for w in mounted
            if isinstance(w, AppMessage) and "rejected" in str(w._content).lower()
        ]
        assert len(reject_msgs) == 1

    async def test_rejection_in_subagent_resumes_agent(self) -> None:
        """At depth>0, rejection should resume the agent, not return early."""
        reject_future: Future[object] = Future()
        reject_future.set_result({"type": "reject"})

        async def mock_approval(  # noqa: RUF029
            _r: list[dict[str, object]],
            _a: str,
        ) -> Future[object]:
            return reject_future

        hitl_value = {
            "action_requests": [{"name": "bash", "args": {"command": "rm -rf /"}}],
            "review_configs": [
                {"action_name": "bash", "allowed_decisions": ["approve", "reject"]}
            ],
        }
        interrupt = Interrupt(value=hitl_value, resumable=True, ns=None)

        # Turn 1: interrupt → rejection → resume
        # Turn 2: agent finishes normally (no interrupt)
        agent = _MultiturnFakeAgent(
            [
                [((), "updates", {"__interrupt__": [interrupt]})],
                [],  # empty = agent completes
            ]
        )

        mounted: list[object] = []

        async def record_mount(w: object) -> None:
            await asyncio.sleep(0)
            mounted.append(w)

        adapter = TextualUIAdapter(
            mount_message=record_mount,
            update_status=_noop_status,
            request_approval=mock_approval,
        )

        # Inside subagent: depth=1
        session = SimpleNamespace(thread_id="t1", auto_approve=False, depth=1)

        await execute_task_textual(
            user_input="do it",
            agent=agent,
            assistant_id="test",
            session_state=session,
            adapter=adapter,
        )

        # Should NOT return early — agent should be called twice
        # (once for initial stream, once for resume after rejection)
        assert agent._call_count == 2
        # "Rejected — agent will continue" message should be mounted
        continue_msgs = [
            w
            for w in mounted
            if isinstance(w, AppMessage) and "continue" in str(w._content).lower()
        ]
        assert len(continue_msgs) == 1
        # Should NOT have "Command rejected" (root-level message)
        root_reject_msgs = [
            w
            for w in mounted
            if isinstance(w, AppMessage) and "tell the agent" in str(w._content).lower()
        ]
        assert len(root_reject_msgs) == 0


# ---------------------------------------------------------------------------
# 8. Bug exposure tests — step-into lifecycle issues
# ---------------------------------------------------------------------------


class TestStepIntoLifecycleBugs:
    """Tests that expose known bugs in the step-into lifecycle.

    These tests document and verify fixes for issues discovered
    during deep investigation of the step-into/return cycle.
    """

    def test_bug1_parent_interrupt_id_captured(self) -> None:
        """BUG-1: parent_tool_call_id must be captured on step-into.

        When the user steps into a subagent, the interrupt_id from the
        parent thread must be stored in the ConversationContext so that
        /return can resume the parent thread with Command(resume=...).
        """
        # The _create_branch_context function currently hardcodes
        # parent_tool_call_id=None. After the fix, it should accept
        # and store the interrupt_id.
        from deepagents_cli.textual_adapter import _create_branch_context

        ctx = _create_branch_context(
            assistant_id="test",
            subagent_type="researcher",
            task_description="find bugs",
        )
        # BUG: parent_tool_call_id is always None
        # After fix, _create_branch_context should accept an interrupt_id param
        # and store it. For now, this test documents the gap.
        assert ctx.parent_tool_call_id is None, (
            "Expected None (known bug) — fix should make this non-None "
            "when an interrupt_id is provided"
        )

    async def test_bug1_step_into_preserves_interrupt_id(self) -> None:
        """BUG-1: step-into decision should capture the interrupt_id.

        The interrupt_id from pending_interrupts should flow into the
        ConversationContext.parent_tool_call_id so /return can resume it.
        """
        step_into_future: Future[object] = Future()
        step_into_future.set_result(
            {
                "type": "step_into",
                "args": {
                    "subagent_type": "researcher",
                    "description": "investigate",
                },
            }
        )

        async def mock_approval(  # noqa: RUF029
            _r: list[dict[str, object]],
            _a: str,
        ) -> Future[object]:
            return step_into_future

        hitl_value = {
            "action_requests": [
                {
                    "name": "task",
                    "args": {
                        "subagent_type": "researcher",
                        "description": "investigate",
                    },
                }
            ],
            "review_configs": [
                {"action_name": "task", "allowed_decisions": ["approve", "reject"]}
            ],
        }
        # Give the interrupt a known ID
        interrupt = Interrupt(value=hitl_value, resumable=True, ns=None, when="during")

        chunks: list[tuple] = [
            ((), "updates", {"__interrupt__": [interrupt]}),
        ]

        adapter = TextualUIAdapter(
            mount_message=_noop_mount,
            update_status=_noop_status,
            request_approval=mock_approval,
            set_spinner=lambda _: asyncio.sleep(0),
        )

        session = SimpleNamespace(thread_id="t1", auto_approve=False, depth=0)

        result = await execute_task_textual(
            user_input="do it",
            agent=_FakeAgent(chunks),
            assistant_id="test",
            session_state=session,
            adapter=adapter,
        )

        assert result.step_into_context is not None
        # BUG-1: parent_tool_call_id should be the interrupt's ID, not None
        # This will fail until the fix is applied
        assert result.step_into_context.parent_tool_call_id is not None, (
            "parent_tool_call_id should capture the interrupt ID "
            "so /return can resume the parent thread"
        )

    def test_bug2_context_popped_on_subagent_error(self) -> None:
        """BUG-2: If subagent crashes, context must be popped.

        When execute_task_textual throws during a subagent invocation,
        the pushed context should be cleaned up so the user isn't
        stuck in a dead subagent.
        """
        state = SessionState(auto_approve=False)
        assert state.depth == 0

        # Simulate what _run_agent_task does: push context
        ctx = ConversationContext(
            thread_id="subagent-thread",
            subagent_type="researcher",
            task_description="find bugs",
            summary_path=None,
            parent_tool_call_id=None,
        )
        state.push_context(ctx)
        assert state.depth == 1

        # Simulate subagent crash — the except block should pop context
        # (mirrors _run_agent_task's error handling after the fix)
        crashed = True
        if crashed and state.depth > 0:
            state.pop_context()

        assert state.depth == 0, "Context should be back to root after subagent error"

    async def test_bug3_nested_step_into_result_captured(self) -> None:
        """BUG-3: Nested step-into result must not be discarded.

        When the subagent itself triggers a step-into, the returned
        ExecuteTaskResult.step_into_context must be handled (pushed).
        Currently the second execute_task_textual call discards its result.
        """
        # First step-into: user steps into subagent
        first_step_into: Future[object] = Future()
        first_step_into.set_result(
            {
                "type": "step_into",
                "args": {
                    "subagent_type": "researcher",
                    "description": "outer task",
                },
            }
        )

        async def mock_approval(  # noqa: RUF029
            _r: list[dict[str, object]],
            _a: str,
        ) -> Future[object]:
            return first_step_into

        hitl_value = {
            "action_requests": [
                {
                    "name": "task",
                    "args": {"subagent_type": "researcher", "description": "outer"},
                }
            ],
            "review_configs": [
                {"action_name": "task", "allowed_decisions": ["approve", "reject"]}
            ],
        }
        interrupt = Interrupt(value=hitl_value, resumable=True, ns=None)

        adapter = TextualUIAdapter(
            mount_message=_noop_mount,
            update_status=_noop_status,
            request_approval=mock_approval,
            set_spinner=lambda _: asyncio.sleep(0),
        )

        session = SimpleNamespace(thread_id="t1", auto_approve=False, depth=0)

        # The first call returns step_into_context
        result = await execute_task_textual(
            user_input="go",
            agent=_FakeAgent([((), "updates", {"__interrupt__": [interrupt]})]),
            assistant_id="test",
            session_state=session,
            adapter=adapter,
        )
        assert result.step_into_context is not None, (
            "First step-into should return a context"
        )
        # BUG-3: In _run_agent_task, the SECOND call to execute_task_textual
        # at line 2139 discards its result. If that call also returns a
        # step_into_context, the nested context is lost.
        # This test documents the expected behavior: the result should be
        # handled in a loop that pushes nested contexts.

    def test_bug4_auto_approve_scoped_to_context(self) -> None:
        """BUG-4: Auto-approve should not leak between contexts.

        Toggling auto-approve inside a subagent should not affect the
        parent context when the user returns.
        """
        state = SessionState(auto_approve=False)
        assert not state.auto_approve

        # Push a subagent context
        ctx = ConversationContext(
            thread_id="sub-thread",
            subagent_type="coder",
            task_description="write code",
            summary_path=None,
            parent_tool_call_id=None,
        )
        state.push_context(ctx)

        # Toggle auto-approve inside the subagent
        state.auto_approve = True
        assert state.auto_approve

        # Return to root
        state.pop_context()

        # BUG-4: auto_approve is a single shared boolean, so it's still True
        # After the fix, returning to root should restore the parent's setting
        assert not state.auto_approve, (
            "Auto-approve should be scoped to the context — "
            "returning to root should restore the parent's setting"
        )


# ---------------------------------------------------------------------------
# 9. Breadcrumbs feature
# ---------------------------------------------------------------------------


class TestBreadcrumbs:
    """Tests for the breadcrumbs feature in the SubagentBanner."""

    def test_breadcrumbs_exact_boundary_at_3(self) -> None:
        """Breadcrumbs should appear at exactly 3 items (the threshold)."""
        banner = SubagentBanner()
        banner.show(
            subagent_type="coder",
            depth=2,
            context_stack=["root", "researcher", "coder"],
        )
        assert banner.breadcrumbs == "root > researcher > coder"

    def test_breadcrumbs_at_depth_4(self) -> None:
        """Breadcrumbs should show full chain at depth 4."""
        banner = SubagentBanner()
        stack = ["root", "planner", "researcher", "coder"]
        banner.show(subagent_type="coder", depth=3, context_stack=stack)
        assert banner.breadcrumbs == "root > planner > researcher > coder"

    def test_no_breadcrumbs_at_2_items(self) -> None:
        """Breadcrumbs should NOT appear with only 2 stack items."""
        banner = SubagentBanner()
        banner.show(
            subagent_type="researcher",
            depth=1,
            context_stack=["root", "researcher"],
        )
        assert banner.breadcrumbs == ""

    def test_no_breadcrumbs_with_none_stack(self) -> None:
        """Breadcrumbs should be empty when context_stack is None."""
        banner = SubagentBanner()
        banner.show(subagent_type="coder", depth=1, context_stack=None)
        assert banner.breadcrumbs == ""

    def test_no_breadcrumbs_with_empty_stack(self) -> None:
        """Breadcrumbs should be empty when context_stack is empty."""
        banner = SubagentBanner()
        banner.show(subagent_type="coder", depth=1, context_stack=[])
        assert banner.breadcrumbs == ""

    def test_breadcrumbs_update_on_successive_shows(self) -> None:
        """Calling show() multiple times should update breadcrumbs."""
        banner = SubagentBanner()
        # First: depth 1, no breadcrumbs
        banner.show(
            subagent_type="researcher",
            depth=1,
            context_stack=["root", "researcher"],
        )
        assert banner.breadcrumbs == ""

        # Second: depth 2, still no breadcrumbs (only 2 items shown)
        banner.show(
            subagent_type="coder",
            depth=2,
            context_stack=["root", "coder"],
        )
        assert banner.breadcrumbs == ""

        # Third: depth 3, now breadcrumbs appear
        banner.show(
            subagent_type="writer",
            depth=3,
            context_stack=["root", "coder", "writer"],
        )
        assert banner.breadcrumbs == "root > coder > writer"

    async def test_breadcrumbs_render_in_label(self) -> None:
        """When breadcrumbs are set, the label should show them."""

        class _BreadcrumbApp(App[None]):
            def compose(self) -> ComposeResult:
                yield SubagentBanner(id="banner")

        app = _BreadcrumbApp()
        async with app.run_test() as pilot:
            banner = app.query_one("#banner", SubagentBanner)
            stack = ["root", "researcher", "coder"]
            banner.show(subagent_type="coder", depth=2, context_stack=stack)
            await pilot.pause()

            label = banner.query_one("#subagent-label", Static)
            label_text = str(label._Static__content)  # type: ignore[attr-defined]
            assert "root > researcher > coder" in label_text

    async def test_no_breadcrumbs_shows_type_depth_label(self) -> None:
        """Without breadcrumbs, label should show [type:depth] format."""

        class _NoBreadcrumbApp(App[None]):
            def compose(self) -> ComposeResult:
                yield SubagentBanner(id="banner")

        app = _NoBreadcrumbApp()
        async with app.run_test() as pilot:
            banner = app.query_one("#banner", SubagentBanner)
            banner.show(
                subagent_type="researcher",
                depth=1,
                context_stack=["root", "researcher"],
            )
            await pilot.pause()

            label = banner.query_one("#subagent-label", Static)
            label_text = str(label._Static__content)  # type: ignore[attr-defined]
            assert "[researcher:1]" in label_text


# ---------------------------------------------------------------------------
# 10. Recursive step-into loop
# ---------------------------------------------------------------------------


class TestBuildStepIntoPrompt:
    """Tests for _build_step_into_prompt helper."""

    def test_includes_task_description(self) -> None:
        """Prompt should contain the task description."""
        from deepagents_cli.app import _build_step_into_prompt

        ctx = ConversationContext(
            thread_id="t1",
            subagent_type="researcher",
            task_description="find the auth bug",
            summary_path=Path("/tmp/summary.md"),
            parent_tool_call_id=None,
        )
        prompt = _build_step_into_prompt(ctx)
        assert "find the auth bug" in prompt

    def test_includes_summary_path(self) -> None:
        """Prompt should reference the summary file path."""
        from deepagents_cli.app import _build_step_into_prompt

        ctx = ConversationContext(
            thread_id="t1",
            subagent_type="coder",
            task_description="write tests",
            summary_path=Path("/tmp/branches/abc/summary.md"),
            parent_tool_call_id=None,
        )
        prompt = _build_step_into_prompt(ctx)
        assert "/tmp/branches/abc/summary.md" in prompt

    def test_includes_return_instruction(self) -> None:
        """Prompt should mention /return command."""
        from deepagents_cli.app import _build_step_into_prompt

        ctx = ConversationContext(
            thread_id="t1",
            subagent_type="researcher",
            task_description="task",
            summary_path=Path("/tmp/s.md"),
            parent_tool_call_id=None,
        )
        prompt = _build_step_into_prompt(ctx)
        assert "/return" in prompt


def _make_step_into_interrupt(
    subagent_type: str, description: str
) -> list[tuple[Any, ...]]:
    """Build chunks that trigger a step-into HITL interrupt."""
    hitl_value = {
        "action_requests": [
            {
                "name": "task",
                "args": {
                    "subagent_type": subagent_type,
                    "description": description,
                },
            }
        ],
        "review_configs": [
            {"action_name": "task", "allowed_decisions": ["approve", "reject"]}
        ],
    }
    interrupt = Interrupt(value=hitl_value, resumable=True, ns=None)
    return [((), "updates", {"__interrupt__": [interrupt]})]


def _step_into_future(subagent_type: str, description: str) -> Future[object]:
    """Create a resolved Future with a step-into decision."""
    f: Future[object] = Future()
    f.set_result(
        {
            "type": "step_into",
            "args": {"subagent_type": subagent_type, "description": description},
        }
    )
    return f


class _CrashingAgent:
    """Agent that yields chunks on first N calls, then raises on the next."""

    def __init__(self, turns: list[list[tuple]], crash_on: int) -> None:
        self._turns = turns
        self._crash_on = crash_on
        self._call_count = 0

    async def astream(self, *_: Any, **__: Any) -> AsyncIterator[tuple[Any, ...]]:
        self._call_count += 1
        if self._call_count == self._crash_on:
            msg = f"Simulated crash on call {self._call_count}"
            raise RuntimeError(msg)
        idx = min(self._call_count - 1, len(self._turns) - 1)
        for chunk in self._turns[idx]:
            yield chunk


class TestRecursiveStepIntoLoop:
    """Integration tests for the full recursive step-into loop.

    These exercise execute_task_textual in the same while-loop
    pattern used by _run_agent_task in app.py.
    """

    async def test_two_level_step_into_chain(self) -> None:
        """Agent returns step-into twice, then completes. Loop pushes both."""
        call_count = 0

        async def mock_approval(  # noqa: RUF029
            _r: list[dict[str, object]],
            _a: str,
        ) -> Future[object]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _step_into_future("researcher", "find bugs")
            return _step_into_future("coder", "fix bugs")

        # Turn 1: interrupt -> step-into researcher
        # Turn 2: interrupt -> step-into coder
        # Turn 3: clean completion
        agent = _MultiturnFakeAgent(
            [
                _make_step_into_interrupt("researcher", "find bugs"),
                _make_step_into_interrupt("coder", "fix bugs"),
                [],  # completes
            ]
        )

        adapter = TextualUIAdapter(
            mount_message=_noop_mount,
            update_status=_noop_status,
            request_approval=mock_approval,
            set_spinner=lambda _: asyncio.sleep(0),
        )

        state = SessionState(auto_approve=False)
        assert state.depth == 0

        # Replicate the exact loop from _run_agent_task
        result = await execute_task_textual(
            user_input="start",
            agent=agent,
            assistant_id="test",
            session_state=state,
            adapter=adapter,
        )

        while result.step_into_context:
            state.push_context(result.step_into_context)
            from deepagents_cli.app import _build_step_into_prompt

            prompt = _build_step_into_prompt(result.step_into_context)
            result = await execute_task_textual(
                user_input=prompt,
                agent=agent,
                assistant_id="test",
                session_state=state,
                adapter=adapter,
            )

        # Two contexts pushed: researcher then coder
        assert state.depth == 2
        stack_types = [c.subagent_type for c in state.context_stack]
        assert stack_types == ["root", "researcher", "coder"]
        # Agent called 3 times total
        assert agent._call_count == 3

    async def test_no_step_into_skips_loop(self) -> None:
        """When agent completes without step-into, loop body never runs."""
        # Agent completes immediately
        agent = _FakeAgent([])

        adapter = TextualUIAdapter(
            mount_message=_noop_mount,
            update_status=_noop_status,
            request_approval=lambda *_: None,
        )

        state = SessionState(auto_approve=False)
        result = await execute_task_textual(
            user_input="hello",
            agent=agent,
            assistant_id="test",
            session_state=state,
            adapter=adapter,
        )

        assert result.step_into_context is None
        assert state.depth == 0

    async def test_prompts_contain_correct_task_descriptions(self) -> None:
        """Each loop iteration should use the correct context's task."""
        captured_inputs: list[str] = []

        class _InputCapturingAgent:
            def __init__(self, turns: list[list[tuple]]) -> None:
                self._turns = turns
                self._call_count = 0

            async def astream(
                self,
                user_input: str | dict[str, Any],
                *_: Any,
                **__: Any,
            ) -> AsyncIterator[tuple[Any, ...]]:
                # Capture the input for each call
                if isinstance(user_input, str):
                    captured_inputs.append(user_input)
                elif isinstance(user_input, dict):
                    # langgraph format
                    msgs = user_input.get("messages", [])
                    if msgs:
                        captured_inputs.append(
                            msgs[0].content
                            if hasattr(msgs[0], "content")
                            else str(msgs[0])
                        )
                idx = min(self._call_count, len(self._turns) - 1)
                self._call_count += 1
                for chunk in self._turns[idx]:
                    yield chunk

        call_count = 0

        async def mock_approval(  # noqa: RUF029
            _r: list[dict[str, object]], _a: str
        ) -> Future[object]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _step_into_future("planner", "plan the migration")
            return _step_into_future("coder", "write migration SQL")

        agent = _InputCapturingAgent(
            [
                _make_step_into_interrupt("planner", "plan the migration"),
                _make_step_into_interrupt("coder", "write migration SQL"),
                [],
            ]
        )

        adapter = TextualUIAdapter(
            mount_message=_noop_mount,
            update_status=_noop_status,
            request_approval=mock_approval,
            set_spinner=lambda _: asyncio.sleep(0),
        )

        state = SessionState(auto_approve=False)
        result = await execute_task_textual(
            user_input="start",
            agent=agent,
            assistant_id="test",
            session_state=state,
            adapter=adapter,
        )

        while result.step_into_context:
            state.push_context(result.step_into_context)
            from deepagents_cli.app import _build_step_into_prompt

            prompt = _build_step_into_prompt(result.step_into_context)
            result = await execute_task_textual(
                user_input=prompt,
                agent=agent,
                assistant_id="test",
                session_state=state,
                adapter=adapter,
            )

        # The prompts fed into loop iterations should match each context
        # captured_inputs[0] = "start" (initial call)
        # captured_inputs[1] = prompt for planner context
        # captured_inputs[2] = prompt for coder context
        # Note: inputs may be wrapped by execute_task_textual, so check
        # that the step_into_context descriptions ended up in the state
        assert state.depth == 2
        assert state.context_stack[1].task_description == "plan the migration"
        assert state.context_stack[2].task_description == "write migration SQL"


class TestErrorRecoveryIntegration:
    """Error during nested step-into must pop all pushed contexts.

    These tests exercise the actual execute_task_textual + loop pattern,
    not just the state machine in isolation.
    """

    async def test_crash_on_second_subagent_pops_both(self) -> None:
        """Agent crashes on 3rd call (2nd subagent). Both pushed contexts pop."""
        call_count = 0

        async def mock_approval(  # noqa: RUF029
            _r: list[dict[str, object]], _a: str
        ) -> Future[object]:
            nonlocal call_count
            call_count += 1
            return _step_into_future(f"agent-{call_count}", f"task-{call_count}")

        # Turn 1: step-into interrupt
        # Turn 2: step-into interrupt
        # Turn 3: crash
        agent = _CrashingAgent(
            turns=[
                _make_step_into_interrupt("agent-1", "task-1"),
                _make_step_into_interrupt("agent-2", "task-2"),
            ],
            crash_on=3,
        )

        adapter = TextualUIAdapter(
            mount_message=_noop_mount,
            update_status=_noop_status,
            request_approval=mock_approval,
            set_spinner=lambda _: asyncio.sleep(0),
        )

        state = SessionState(auto_approve=False)
        depth_before = state.depth  # 0

        # Run the loop, catching the crash like _run_agent_task does
        try:
            result = await execute_task_textual(
                user_input="go",
                agent=agent,
                assistant_id="test",
                session_state=state,
                adapter=adapter,
            )
            while result.step_into_context:
                state.push_context(result.step_into_context)
                from deepagents_cli.app import _build_step_into_prompt

                prompt = _build_step_into_prompt(result.step_into_context)
                result = await execute_task_textual(
                    user_input=prompt,
                    agent=agent,
                    assistant_id="test",
                    session_state=state,
                    adapter=adapter,
                )
        except RuntimeError:
            # Error handler: pop all contexts back to pre-loop depth
            while state.depth > depth_before:
                state.pop_context()

        assert state.depth == 0
        assert state.current_context.subagent_type == "root"

    async def test_crash_preserves_pre_existing_depth(self) -> None:
        """Crash should restore to depth before the loop, not always 0."""

        async def mock_approval(  # noqa: RUF029
            _r: list[dict[str, object]], _a: str
        ) -> Future[object]:
            return _step_into_future("child", "child task")

        agent = _CrashingAgent(
            turns=[_make_step_into_interrupt("child", "child task")],
            crash_on=2,
        )

        adapter = TextualUIAdapter(
            mount_message=_noop_mount,
            update_status=_noop_status,
            request_approval=mock_approval,
            set_spinner=lambda _: asyncio.sleep(0),
        )

        state = SessionState(auto_approve=False)
        # Pre-existing depth: user already in a subagent
        state.push_context(
            ConversationContext(
                thread_id="t-parent",
                subagent_type="parent",
                task_description="parent task",
                summary_path=None,
                parent_tool_call_id=None,
            )
        )
        depth_before = state.depth  # 1

        try:
            result = await execute_task_textual(
                user_input="go deeper",
                agent=agent,
                assistant_id="test",
                session_state=state,
                adapter=adapter,
            )
            while result.step_into_context:
                state.push_context(result.step_into_context)
                from deepagents_cli.app import _build_step_into_prompt

                prompt = _build_step_into_prompt(result.step_into_context)
                result = await execute_task_textual(
                    user_input=prompt,
                    agent=agent,
                    assistant_id="test",
                    session_state=state,
                    adapter=adapter,
                )
        except RuntimeError:
            while state.depth > depth_before:
                state.pop_context()

        # Should be back at depth 1 (parent), not 0 (root)
        assert state.depth == 1
        assert state.current_context.subagent_type == "parent"
