"""Approval widget for HITL - using standard Textual patterns."""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar

from textual import events
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Container, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Static

from deepagents_cli.widgets.tool_renderers import get_renderer


class ApprovalMenu(Container):
    """Approval menu using standard Textual patterns.

    Key design decisions (following mistral-vibe reference):
    - Container base class with compose()
    - BINDINGS for key handling (not on_key)
    - can_focus_children = False to prevent focus theft
    - Simple Static widgets for options
    - Standard message posting
    - Tool-specific widgets via renderer pattern
    """

    can_focus = True
    can_focus_children = False

    # CSS is in app.tcss - no DEFAULT_CSS needed

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("up", "move_up", "Up", show=False),
        Binding("k", "move_up", "Up", show=False),
        Binding("down", "move_down", "Down", show=False),
        Binding("j", "move_down", "Down", show=False),
        Binding("enter", "select", "Select", show=False),
        Binding("1", "select_approve", "Approve", show=False),
        Binding("y", "select_approve", "Approve", show=False),
        Binding("2", "select_reject", "Reject", show=False),
        Binding("n", "select_reject", "Reject", show=False),
        Binding("3", "select_auto", "Auto-approve", show=False),
        Binding("a", "select_auto", "Auto-approve", show=False),
    ]

    class Decided(Message):
        """Message sent when user makes a decision."""

        def __init__(self, decision: dict[str, str]) -> None:
            super().__init__()
            self.decision = decision

    # Tools that don't need detailed info display (already shown in tool call)
    _MINIMAL_TOOLS: ClassVar[set[str]] = {"bash", "shell"}

    def __init__(
        self,
        action_requests: list[dict[str, Any]] | dict[str, Any],
        _assistant_id: str | None = None,
        id: str | None = None,  # noqa: A002
        **kwargs: Any,
    ) -> None:
        super().__init__(id=id or "approval-menu", classes="approval-menu", **kwargs)
        # Support both single request (legacy) and list of requests (batch)
        if isinstance(action_requests, dict):
            self._action_requests = [action_requests]
        else:
            self._action_requests = action_requests

        # For display purposes, get tool names
        self._tool_names = [r.get("name", "unknown") for r in self._action_requests]
        self._selected = 0
        self._future: asyncio.Future[dict[str, str]] | None = None
        self._option_widgets: list[Static] = []
        self._tool_info_container: Vertical | None = None
        # Minimal display if ALL tools are bash/shell
        self._is_minimal = all(name in self._MINIMAL_TOOLS for name in self._tool_names)

    def set_future(self, future: asyncio.Future[dict[str, str]]) -> None:
        """Set the future to resolve when user decides."""
        self._future = future

    def compose(self) -> ComposeResult:
        """Compose the widget with Static children.

        Layout: Tool info first (what's being approved), then options at bottom.
        For bash/shell, skip tool info since it's already shown in tool call.
        """
        # Title - show count if multiple tools
        count = len(self._action_requests)
        if count == 1:
            title = f">>> {self._tool_names[0]} Requires Approval <<<"
        else:
            title = f">>> {count} Tool Calls Require Approval <<<"
        yield Static(title, classes="approval-title")

        # Tool info - only for non-minimal tools (diffs, writes show actual content)
        if not self._is_minimal:
            with VerticalScroll(classes="tool-info-scroll"):
                self._tool_info_container = Vertical(classes="tool-info-container")
                yield self._tool_info_container

            # Separator between tool details and options
            yield Static("─" * 40, classes="approval-separator")

        # Options container at bottom
        with Container(classes="approval-options-container"):
            # Options - create 3 Static widgets
            for i in range(3):
                widget = Static("", classes="approval-option")
                self._option_widgets.append(widget)
                yield widget

        # Help text at the very bottom
        yield Static(
            "↑/↓ navigate • Enter select • y/n/a quick keys",
            classes="approval-help",
        )

    async def on_mount(self) -> None:
        """Focus self on mount and update tool info."""
        if not self._is_minimal:
            await self._update_tool_info()
        self._update_options()
        self.focus()

    async def _update_tool_info(self) -> None:
        """Mount the tool-specific approval widgets for all tools."""
        if not self._tool_info_container:
            return

        # Clear existing content
        await self._tool_info_container.remove_children()

        # Mount info for each tool
        for i, action_request in enumerate(self._action_requests):
            tool_name = action_request.get("name", "unknown")
            tool_args = action_request.get("args", {})

            # Add tool header if multiple tools
            if len(self._action_requests) > 1:
                header = Static(f"[bold]{i + 1}. {tool_name}[/bold]")
                await self._tool_info_container.mount(header)

            # Get the appropriate renderer for this tool
            renderer = get_renderer(tool_name)
            widget_class, data = renderer.get_approval_widget(tool_args)
            approval_widget = widget_class(data)
            await self._tool_info_container.mount(approval_widget)

    def _update_options(self) -> None:
        """Update option widgets based on selection."""
        count = len(self._action_requests)
        if count == 1:
            options = [
                "1. Approve (y)",
                "2. Reject (n)",
                "3. Auto-approve all this session (a)",
            ]
        else:
            options = [
                f"1. Approve all {count} (y)",
                f"2. Reject all {count} (n)",
                "3. Auto-approve all this session (a)",
            ]

        for i, (text, widget) in enumerate(zip(options, self._option_widgets, strict=True)):
            cursor = "› " if i == self._selected else "  "
            widget.update(f"{cursor}{text}")

            # Update classes
            widget.remove_class("approval-option-selected")
            if i == self._selected:
                widget.add_class("approval-option-selected")

    def action_move_up(self) -> None:
        """Move selection up."""
        self._selected = (self._selected - 1) % 3
        self._update_options()

    def action_move_down(self) -> None:
        """Move selection down."""
        self._selected = (self._selected + 1) % 3
        self._update_options()

    def action_select(self) -> None:
        """Select current option."""
        self._handle_selection(self._selected)

    def action_select_approve(self) -> None:
        """Select approve option."""
        self._selected = 0
        self._update_options()
        self._handle_selection(0)

    def action_select_reject(self) -> None:
        """Select reject option."""
        self._selected = 1
        self._update_options()
        self._handle_selection(1)

    def action_select_auto(self) -> None:
        """Select auto-approve option."""
        self._selected = 2
        self._update_options()
        self._handle_selection(2)

    def _handle_selection(self, option: int) -> None:
        """Handle the selected option."""
        decision_map = {
            0: "approve",
            1: "reject",
            2: "auto_approve_all",
        }
        decision = {"type": decision_map[option]}

        # Resolve the future
        if self._future and not self._future.done():
            self._future.set_result(decision)

        # Post message
        self.post_message(self.Decided(decision))

    def on_blur(self, event: events.Blur) -> None:
        """Re-focus on blur to keep focus trapped until decision is made."""
        self.call_after_refresh(self.focus)
