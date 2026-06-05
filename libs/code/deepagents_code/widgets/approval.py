"""Approval widget for HITL - using standard Textual patterns."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar

from textual.binding import Binding, BindingType
from textual.containers import Container, Vertical, VerticalScroll
from textual.content import Content
from textual.message import Message
from textual.widgets import Input, Static

if TYPE_CHECKING:
    import asyncio

    from textual import events
    from textual.app import ComposeResult

from deepagents_code import theme
from deepagents_code.config import (
    get_glyphs,
    is_ascii_mode,
)
from deepagents_code.unicode_security import (
    check_url_safety,
    detect_dangerous_unicode,
    format_warning_detail,
    iter_string_values,
    looks_like_url_key,
    render_with_unicode_markers,
    strip_dangerous_unicode,
    summarize_issues,
)
from deepagents_code.widgets.tool_renderers import get_renderer

logger = logging.getLogger(__name__)

# Max length for truncated shell command display
_SHELL_COMMAND_TRUNCATE_LENGTH: int = 120
# Max number of lines for truncated shell command display
_SHELL_COMMAND_TRUNCATE_LINES: int = 5
_WARNING_PREVIEW_LIMIT: int = 3
_WARNING_TEXT_TRUNCATE_LENGTH: int = 220
# Must match the "reject" entry in `_handle_selection`'s decision map.
_REJECT_OPTION_INDEX: int = 2


def _is_command_too_long(command: str) -> bool:
    """Whether a shell command exceeds the display thresholds (char or line).

    Args:
        command: The shell command string to check.

    Returns:
        `True` if the command is longer than `_SHELL_COMMAND_TRUNCATE_LENGTH`
        characters or has more than `_SHELL_COMMAND_TRUNCATE_LINES` lines.
    """
    if len(command) > _SHELL_COMMAND_TRUNCATE_LENGTH:
        return True
    return command.count("\n") + 1 > _SHELL_COMMAND_TRUNCATE_LINES


def _truncate_command(command: str) -> str:
    """Truncate a shell command for compact display.

    Applies line truncation first (keeping at most `_SHELL_COMMAND_TRUNCATE_LINES`
    lines), then character truncation, so multi-line commands collapse before
    long single lines are cut. A single ellipsis is appended at the end.

    Args:
        command: The shell command string to truncate.

    Returns:
        The truncated command string, with a trailing ellipsis if any truncation
        was applied; otherwise the original command unchanged.
    """
    ellipsis = get_glyphs().ellipsis
    lines = command.split("\n")
    truncated = len(lines) > _SHELL_COMMAND_TRUNCATE_LINES
    if truncated:
        command = "\n".join(lines[:_SHELL_COMMAND_TRUNCATE_LINES])
    if len(command) > _SHELL_COMMAND_TRUNCATE_LENGTH:
        command = command[:_SHELL_COMMAND_TRUNCATE_LENGTH]
        truncated = True
    return command + ellipsis if truncated else command


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
        Binding("2", "select_auto", "Auto-approve", show=False),
        Binding("a", "select_auto", "Auto-approve", show=False),
        Binding("3", "select_reject", "Reject", show=False),
        Binding("n", "select_reject", "Reject", show=False),
        Binding("e", "toggle_expand", "Expand command", show=False),
        Binding("tab", "reject_with_reason", "Reject with reason", show=False),
    ]

    class Decided(Message):
        """Message sent when user makes a decision."""

        def __init__(self, decision: dict[str, str]) -> None:
            """Initialize a Decided message with the user's decision.

            Args:
                decision: Dictionary containing the decision type (e.g., 'approve',
                    'reject', or 'auto_approve_all').
            """
            super().__init__()
            self.decision = decision

    # Tools that don't need detailed info display (already shown in tool call)
    _MINIMAL_TOOLS: ClassVar[frozenset[str]] = frozenset({"execute"})

    def __init__(
        self,
        action_requests: list[dict[str, Any]] | dict[str, Any],
        _assistant_id: str | None = None,
        id: str | None = None,  # noqa: A002  # Textual widget constructor uses `id` parameter
        **kwargs: Any,
    ) -> None:
        """Initialize the ApprovalMenu widget.

        Args:
            action_requests: A single action request dictionary or a list of action
                request dictionaries requiring approval. Each dictionary should
                contain 'name' (tool name) and 'args' (tool arguments).
            _assistant_id: Optional assistant ID (currently unused, reserved for
                future use).
            id: Optional widget ID. Defaults to 'approval-menu'.
            **kwargs: Additional keyword arguments passed to the Container base class.
        """
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
        # Minimal display if ALL tools are shell-execution tools
        self._is_minimal = all(name in self._MINIMAL_TOOLS for name in self._tool_names)
        # For expandable shell commands
        self._command_expanded = False
        self._command_widget: Static | None = None
        self._has_expandable_command = self._check_expandable_command()
        self._security_warnings = self._collect_security_warnings()
        # Free-text reject mode state (Tab on Reject opens an inline Input).
        self._reason_input: Input | None = None
        self._reason_input_active = False
        self._help_widget: Static | None = None

    def set_future(self, future: asyncio.Future[dict[str, str]]) -> None:
        """Set the future to resolve when user decides."""
        self._future = future

    def _check_expandable_command(self) -> bool:
        """Check if there's a shell command that can be expanded.

        Returns:
            Whether the single action request is an expandable shell command.
        """
        if len(self._action_requests) != 1:
            return False
        req = self._action_requests[0]
        if req.get("name", "") != "execute":
            return False
        command = str(req.get("args", {}).get("command", ""))
        return _is_command_too_long(command)

    def _get_command_display(self, *, expanded: bool) -> Content:
        """Get the command display content (truncated or full).

        Args:
            expanded: Whether to show the full command or truncated version.

        Returns:
            Styled Content for the command display.

        Raises:
            RuntimeError: If called with empty action_requests.
        """
        if not self._action_requests:
            msg = "_get_command_display called with empty action_requests"
            raise RuntimeError(msg)
        req = self._action_requests[0]
        command_raw = str(req.get("args", {}).get("command", ""))
        command = strip_dangerous_unicode(command_raw)
        issues = detect_dangerous_unicode(command_raw)

        too_long = _is_command_too_long(command)
        if expanded or not too_long:
            command_display = command
        else:
            command_display = _truncate_command(command)

        if not expanded and too_long:
            display = Content.from_markup(
                "[bold]$cmd[/bold] [dim](press 'e' to expand)[/dim]",
                cmd=command_display,
            )
        else:
            display = Content.from_markup("[bold]$cmd[/bold]", cmd=command_display)

        if not issues:
            return display

        raw_with_markers = render_with_unicode_markers(command_raw)
        if not expanded and len(raw_with_markers) > _WARNING_TEXT_TRUNCATE_LENGTH:
            raw_with_markers = (
                raw_with_markers[:_WARNING_TEXT_TRUNCATE_LENGTH] + get_glyphs().ellipsis
            )

        return Content.assemble(
            display,
            Content.from_markup(
                "\n[yellow]Warning:[/yellow] hidden chars detected ($summary)\n"
                "[dim]raw: $raw[/dim]",
                summary=summarize_issues(issues),
                raw=raw_with_markers,
            ),
        )

    def compose(self) -> ComposeResult:
        """Compose the widget with Static children.

        Layout: Tool info first (what's being approved), then options at bottom.
        For bash/shell, skip tool info since it's already shown in tool call.

        Yields:
            Widgets for title, tool info, options, and help text.
        """
        # Title - show count if multiple tools
        count = len(self._action_requests)
        if count == 1:
            title = Content.from_markup(
                ">>> $name Requires Approval <<<", name=self._tool_names[0]
            )
        else:
            title = Content(f">>> {count} Tool Calls Require Approval <<<")
        yield Static(title, classes="approval-title")

        if self._security_warnings:
            parts: list[Content] = [
                Content.from_markup(
                    "[yellow]Warning:[/yellow] Potentially deceptive text"
                ),
            ]
            parts.extend(
                Content.from_markup("\n[dim]- $w[/dim]", w=warning)
                for warning in self._security_warnings[:_WARNING_PREVIEW_LIMIT]
            )
            if len(self._security_warnings) > _WARNING_PREVIEW_LIMIT:
                remaining = len(self._security_warnings) - _WARNING_PREVIEW_LIMIT
                parts.append(Content.styled(f"\n- +{remaining} more warning(s)", "dim"))
            yield Static(
                Content.assemble(*parts),
                classes="approval-security-warning",
            )

        # For shell commands, show the command (expandable if long)
        if self._is_minimal and len(self._action_requests) == 1:
            self._command_widget = Static(
                self._get_command_display(expanded=self._command_expanded),
                classes="approval-command",
            )
            yield self._command_widget

        # Tool info - only for non-minimal tools (diffs, writes show actual content)
        if not self._is_minimal:
            with VerticalScroll(classes="tool-info-scroll"):
                self._tool_info_container = Vertical(classes="tool-info-container")
                yield self._tool_info_container

            # Separator between tool details and options
            glyphs = get_glyphs()
            yield Static(glyphs.box_horizontal * 40, classes="approval-separator")

        # Options container at bottom
        with Container(classes="approval-options-container"):
            # Options - create 3 Static widgets
            for i in range(3):  # noqa: B007  # Loop variable unused - iterating for count only
                widget = Static("", classes="approval-option")
                self._option_widgets.append(widget)
                yield widget

        # Free-text reject reason input (hidden until activated via Tab)
        self._reason_input = Input(
            placeholder="Reason (Enter to submit, Esc to cancel)",
            classes="approval-reason-input",
            id="approval-reason-input",
        )
        self._reason_input.display = False
        yield self._reason_input

        # Help text at the very bottom
        self._help_widget = Static(self._compose_help_text(), classes="approval-help")
        yield self._help_widget

    def _compose_help_text(self) -> str:
        """Build the help-line content for the current mode.

        Returns:
            Help text for either the normal menu or the reject-reason input.
        """
        glyphs = get_glyphs()
        if self._reason_input_active:
            return (
                f"Enter submit {glyphs.bullet} Esc cancel {glyphs.bullet} "
                "leave blank to reject without a reason"
            )
        help_parts = [
            (
                f"{glyphs.arrow_up}/{glyphs.arrow_down} navigate "
                f"{glyphs.bullet} Enter select {glyphs.bullet} y/a/n quick keys"
            ),
        ]
        if self._selected == _REJECT_OPTION_INDEX:
            help_parts.append("Tab amend")
        help_parts.append("Esc reject")
        help_text = f" {glyphs.bullet} ".join(help_parts)
        if self._has_expandable_command:
            help_text += f" {glyphs.bullet} e expand"
        return help_text

    async def on_mount(self) -> None:
        """Focus self on mount and update tool info."""
        if is_ascii_mode():
            colors = theme.get_theme_colors(self)
            self.styles.border = ("ascii", colors.warning)

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
                header = Static(
                    Content.from_markup(
                        "[bold]$num. $name[/bold]",
                        num=i + 1,
                        name=tool_name,
                    )
                )
                await self._tool_info_container.mount(header)

            # Show description if present
            description = action_request.get("description")
            if description:
                desc_widget = Static(
                    Content.from_markup("[dim]$desc[/dim]", desc=description),
                    classes="approval-description",
                )
                await self._tool_info_container.mount(desc_widget)

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
                "2. Auto-approve for this thread (a)",
                "3. Reject (n)",
            ]
        else:
            options = [
                f"1. Approve all {count} (y)",
                "2. Auto-approve for this thread (a)",
                f"3. Reject all {count} (n)",
            ]

        for i, (text, widget) in enumerate(
            zip(options, self._option_widgets, strict=True)
        ):
            cursor = f"{get_glyphs().cursor} " if i == self._selected else "  "
            widget.update(f"{cursor}{text}")

            # Update classes
            widget.remove_class("approval-option-selected")
            if i == self._selected:
                widget.add_class("approval-option-selected")
        if self._help_widget is not None:
            self._help_widget.update(self._compose_help_text())

    def action_move_up(self) -> None:
        """Move selection up."""
        if self._reason_input_active:
            return
        self._selected = (self._selected - 1) % 3
        self._update_options()

    def action_move_down(self) -> None:
        """Move selection down."""
        if self._reason_input_active:
            return
        self._selected = (self._selected + 1) % 3
        self._update_options()

    def action_select(self) -> None:
        """Select current option."""
        self._handle_selection(self._selected)

    def action_select_approve(self) -> None:
        """Submit approve option."""
        self._handle_selection(0)

    def action_select_auto(self) -> None:
        """Submit auto-approve option."""
        self._handle_selection(1)

    def action_select_reject(self) -> None:
        """Submit reject option.

        When the free-text reject input is open, the first press cancels the
        input instead of rejecting, so the user can back out without losing
        their unsubmitted reason.
        """
        if self._reason_input_active:
            self._exit_reason_input_mode()
            return
        self._handle_selection(2)

    def action_toggle_expand(self) -> None:
        """Toggle shell command expansion."""
        if not self._has_expandable_command or not self._command_widget:
            return
        self._command_expanded = not self._command_expanded
        self._command_widget.update(
            self._get_command_display(expanded=self._command_expanded)
        )

    def _handle_selection(
        self, option: int, *, reject_message: str | None = None
    ) -> None:
        """Handle the selected option.

        Args:
            option: Index of the chosen option (0 approve, 1 auto-approve, 2 reject).
            reject_message: Optional free-text reason. Only attached when the
                user rejects with a non-empty message via `action_reject_with_reason`.
        """
        decision_map = {
            0: "approve",
            1: "auto_approve_all",
            2: "reject",
        }
        decision: dict[str, str] = {"type": decision_map[option]}
        if option == _REJECT_OPTION_INDEX and reject_message:
            decision["message"] = reject_message

        self.display = False

        # Resolve the future
        if self._future and not self._future.done():
            self._future.set_result(decision)

        # Post message
        self.post_message(self.Decided(decision))

    def action_reject_with_reason(self) -> None:
        """Enter free-text reject mode if Reject is currently selected.

        No-op unless the cursor is on the Reject option. Mounts an inline
        `Input` whose value is sent as `RejectDecision.message` on submit.
        """
        if self._reason_input_active:
            return
        if self._selected != _REJECT_OPTION_INDEX:
            return
        if self._reason_input is None:
            # Lifecycle bug: Tab fired before `compose()` populated the Input ref.
            # Logging makes the silent no-op debuggable instead of invisible.
            logger.warning(
                "action_reject_with_reason: _reason_input is None; menu may not "
                "be mounted yet"
            )
            return
        self._reason_input_active = True
        self._reason_input.value = ""
        self._reason_input.display = True
        if self._help_widget is not None:
            self._help_widget.update(self._compose_help_text())
        self._reason_input.focus()

    def _exit_reason_input_mode(self) -> None:
        """Close the reason input and return focus to the menu without deciding."""
        if not self._reason_input_active or self._reason_input is None:
            return
        self._reason_input_active = False
        self._reason_input.display = False
        if self._help_widget is not None:
            self._help_widget.update(self._compose_help_text())
        self.focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Submit the reject decision with the typed reason (if any)."""
        # Stop before the guard so a stray submit (e.g. queued after Esc closed
        # the input) cannot bubble to a parent and be re-interpreted, and so a
        # foreign Input's submission is never misrouted through this handler.
        if event.input is not self._reason_input:
            return
        event.stop()
        if not self._reason_input_active:
            logger.debug(
                "on_input_submitted fired with inactive reason input; dropping"
            )
            return
        reason = event.value.strip()
        self._reason_input_active = False
        self._handle_selection(2, reject_message=reason or None)

    def _collect_security_warnings(self) -> list[str]:
        """Collect warning strings for suspicious Unicode and URL values.

        Recursively inspects all nested string values in action arguments.

        Returns:
            Warning strings for the current action request batch.
        """
        warnings: list[str] = []
        for action_request in self._action_requests:
            tool_name = str(action_request.get("name", "unknown"))
            args = action_request.get("args", {})
            if not isinstance(args, dict):
                continue
            for arg_path, text in iter_string_values(args):
                issues = detect_dangerous_unicode(text)
                if issues:
                    warnings.append(
                        f"{tool_name}.{arg_path}: hidden Unicode "
                        f"({summarize_issues(issues)})"
                    )
                if looks_like_url_key(arg_path):
                    result = check_url_safety(text)
                    if result.safe:
                        continue
                    detail = format_warning_detail(result.warnings)
                    if result.decoded_domain:
                        detail = f"{detail}; decoded host: {result.decoded_domain}"
                    warnings.append(f"{tool_name}.{arg_path}: {detail}")
        return warnings

    def on_blur(self, event: events.Blur) -> None:  # noqa: ARG002  # Textual event handler signature
        """Re-focus on blur to keep focus trapped until decision is made.

        Skipped while the free-text reject input is active so the `Input`
        widget can keep keyboard focus.
        """
        if self._reason_input_active:
            return
        self.call_after_refresh(self.focus)
