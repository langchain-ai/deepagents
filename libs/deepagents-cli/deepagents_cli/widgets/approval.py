"""Approval screen for Human-in-the-Loop tool confirmation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option

from deepagents_cli.file_ops import build_approval_preview
from deepagents_cli.widgets.diff import EnhancedDiff

if TYPE_CHECKING:
    from textual.app import ComposeResult


class ApprovalScreen(ModalScreen[dict[str, str]]):
    """Modal screen for approving or rejecting tool actions.

    Uses ModalScreen which properly captures all input.
    """

    DEFAULT_CSS = """
    ApprovalScreen {
        align: center middle;
    }

    ApprovalScreen > Vertical {
        width: 80%;
        max-width: 100;
        height: auto;
        max-height: 80%;
        background: $surface;
        border: solid $warning;
        padding: 1 2;
    }

    ApprovalScreen .approval-header {
        color: $warning;
        text-style: bold;
        text-align: center;
        margin-bottom: 1;
    }

    ApprovalScreen .approval-title {
        text-style: bold;
    }

    ApprovalScreen .approval-details {
        color: $text-muted;
        margin-bottom: 1;
    }

    ApprovalScreen OptionList {
        height: auto;
        max-height: 5;
        margin: 1 0;
        background: $surface;
    }

    ApprovalScreen .approval-hint {
        color: $text-muted;
        text-style: italic;
        margin-top: 1;
    }

    ApprovalScreen .diff-section {
        margin-top: 1;
        max-height: 20;
        overflow-y: auto;
    }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("y", "approve", "Approve", show=False),
        Binding("1", "approve", "Approve", show=False),
        Binding("n", "reject", "Reject", show=False),
        Binding("2", "reject", "Reject", show=False),
        Binding("a", "auto_approve", "Auto-approve", show=False),
        Binding("3", "auto_approve", "Auto-approve", show=False),
        Binding("escape", "reject", "Cancel", show=False),
    ]

    def __init__(
        self,
        action_request: dict[str, Any],
        assistant_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._action_request = action_request
        self._assistant_id = assistant_id

        self._tool_name = action_request.get("name", "unknown")
        self._description = action_request.get("description", "Tool action requires approval")
        self._args = action_request.get("args", {})

        self._preview = None
        if self._tool_name in ("write_file", "edit_file"):
            self._preview = build_approval_preview(
                self._tool_name, self._args, assistant_id
            )

    def compose(self) -> ComposeResult:
        """Compose the approval screen."""
        with Vertical():
            yield Static(
                "[bold yellow]>>> Tool Action Requires Approval <<<[/bold yellow]",
                classes="approval-header",
            )

            # Tool info
            if self._preview:
                yield Static(f"[bold]{self._preview.title}[/bold]", classes="approval-title")
                details = " | ".join(self._preview.details[:2])
                yield Static(f"[dim]{details}[/dim]", classes="approval-details")
            else:
                yield Static(f"[bold]{self._tool_name}[/bold]", classes="approval-title")
                yield Static(f"[dim]{self._description}[/dim]", classes="approval-details")

            # Option list - built-in widget that handles keyboard properly
            yield OptionList(
                Option("1. Approve (y)", id="approve"),
                Option("2. Reject (n)", id="reject"),
                Option("3. Auto-approve all writes this session (a)", id="auto_approve_all"),
                id="approval-options",
            )

            yield Static(
                "[dim]↑/↓ navigate, Enter select, or press y/n/a[/dim]",
                classes="approval-hint",
            )

            # Diff preview
            if self._preview and self._preview.diff and not self._preview.error:
                yield EnhancedDiff(
                    diff=self._preview.diff,
                    title=self._preview.diff_title or "Changes",
                    max_lines=15,
                    classes="diff-section",
                )
            elif self._preview and self._preview.error:
                yield Static(f"[red]{self._preview.error}[/red]", classes="diff-section")
            elif self._tool_name == "write_file":
                # Show content preview for new files
                content = self._args.get("content", "")
                if content:
                    # Create a simple "all additions" diff
                    lines = content.split("\n")[:20]  # First 20 lines
                    preview_lines = [f"[green]+ {line}[/green]" for line in lines]
                    if len(content.split("\n")) > 20:
                        preview_lines.append("[dim]... (truncated)[/dim]")
                    yield Static(
                        "[bold cyan]═══ New File Content ═══[/bold cyan]\n" + "\n".join(preview_lines),
                        classes="diff-section",
                    )

    def on_mount(self) -> None:
        """Focus the option list when mounted."""
        try:
            option_list = self.query_one("#approval-options", OptionList)
            option_list.focus()
        except Exception:  # noqa: BLE001
            pass

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Handle option selection."""
        option_id = event.option.id
        if option_id:
            self.dismiss({"type": str(option_id)})

    def action_approve(self) -> None:
        """Quick approve action."""
        self.dismiss({"type": "approve"})

    def action_reject(self) -> None:
        """Quick reject action."""
        self.dismiss({"type": "reject"})

    def action_auto_approve(self) -> None:
        """Quick auto-approve action."""
        self.dismiss({"type": "auto_approve_all"})


# Keep the old class name for backwards compatibility with app.py imports
ApprovalWidget = ApprovalScreen
