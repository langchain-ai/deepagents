"""Trust prompt for project hooks discovered after a cwd switch."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal

from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.content import Content
from textual.screen import ModalScreen
from textual.widgets import Static

if TYPE_CHECKING:
    from textual.app import ComposeResult

HookTrustChoice = Literal["allow_once", "always_allow", "deny"]


class HookTrustScreen(ModalScreen[HookTrustChoice | None]):
    """Ask how project hooks in a newly entered workspace should be trusted."""

    can_focus = True
    can_focus_children = False

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("enter", "allow_once", "Allow once", show=False, priority=True),
        Binding("a", "always_allow", "Always allow", show=False, priority=True),
        Binding("escape", "deny", "Deny", show=False, priority=True),
    ]

    CSS = """
    HookTrustScreen {
        align: center middle;
    }

    HookTrustScreen > Vertical {
        width: 76;
        max-width: 90%;
        height: auto;
        background: $surface;
        border: solid $warning;
        padding: 1 2;
    }

    HookTrustScreen .hook-trust-title {
        text-style: bold;
        color: $warning;
        text-align: center;
        margin-bottom: 1;
    }

    HookTrustScreen .hook-trust-body {
        height: auto;
        color: $text;
        margin-bottom: 1;
    }

    HookTrustScreen .hook-trust-help {
        height: 1;
        color: $text-muted;
        text-style: italic;
        text-align: center;
    }
    """

    def __init__(self, *, project_root: str, config_path: str) -> None:
        """Initialize the project-hooks trust prompt.

        Args:
            project_root: Workspace root governing the trust decision.
            config_path: Project hooks file that may execute commands.
        """
        super().__init__()
        self._project_root = project_root
        self._config_path = config_path

    def compose(self) -> ComposeResult:
        """Compose the project-hooks trust dialog.

        Yields:
            Title, warning body, and keyboard help widgets.
        """
        with Vertical():
            yield Static(
                "Project hooks can execute commands",
                classes="hook-trust-title",
                markup=False,
            )
            yield Static(
                Content.from_markup(
                    "The workspace [bold]$root[/bold] contains project hooks at "
                    "[bold]$path[/bold]. Only allow hooks for projects you trust. "
                    "Always allow also trusts future edits to this file.",
                    root=self._project_root,
                    path=self._config_path,
                ),
                classes="hook-trust-body",
                markup=False,
            )
            yield Static(
                "Enter: allow once · A: always allow · Esc: deny",
                classes="hook-trust-help",
                markup=False,
            )

    def on_mount(self) -> None:
        """Focus the modal so its bindings receive keyboard input."""
        self.focus()

    def action_allow_once(self) -> None:
        """Approve the current file contents for this session."""
        self.dismiss("allow_once")

    def action_always_allow(self) -> None:
        """Approve this workspace persistently."""
        self.dismiss("always_allow")

    def action_deny(self) -> None:
        """Deny project hooks in this workspace."""
        self.dismiss("deny")

    def action_cancel(self) -> None:
        """Treat app-level cancellation as deny."""
        self.action_deny()
