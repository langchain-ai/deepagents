"""Confirmation modal offered after a restart-capable `/install`.

Provider and sandbox extras (and `--package` installs) are imported by the
app-owned LangGraph server subprocess, so a `/restart` loads them without
exiting the TUI. Rather than make the user type `/restart` by hand, this
modal offers to run that restart immediately while leaving deferral one
keypress away.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal

from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.content import Content
from textual.screen import ModalScreen
from textual.widgets import Static

from deepagents_code.config import get_glyphs

if TYPE_CHECKING:
    from textual.app import ComposeResult


RestartChoice = Literal["restart", "later"]
"""Outcome of the prompt: restart the server now or defer."""


class RestartPromptScreen(ModalScreen[RestartChoice]):
    """Modal asking whether to restart the server after a successful install.

    Dismisses with `"restart"` when the user accepts and `"later"` when the
    user defers. Esc is treated as "later" so the user is never forced into a
    restart they did not explicitly choose.
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("enter", "restart", "Restart", show=False, priority=True),
        Binding("escape", "later", "Later", show=False, priority=True),
    ]

    CSS = """
    RestartPromptScreen {
        align: center middle;
    }

    RestartPromptScreen > Vertical {
        width: 64;
        max-width: 90%;
        height: auto;
        background: $surface;
        border: solid $primary;
        padding: 1 2;
    }

    RestartPromptScreen .restart-prompt-title {
        text-style: bold;
        color: $primary;
        text-align: center;
        margin-bottom: 1;
    }

    RestartPromptScreen .restart-prompt-body {
        height: auto;
        color: $text;
        margin-bottom: 1;
    }

    RestartPromptScreen .restart-prompt-help {
        height: 1;
        color: $text-muted;
        text-style: italic;
        text-align: center;
    }
    """

    _DEFAULT_BODY = "Restart the server to load it now, or defer with `/restart`."

    def __init__(
        self,
        label: str,
        *,
        verb: str = "Installed",
        body: str | None = None,
    ) -> None:
        """Initialize the prompt.

        Args:
            label: The subject surfaced in the title (e.g. an installed extra
                name, or a saved credential like ``"Tavily API key"``).
            verb: Past-tense action shown before `label` in the title. Defaults
                to `"Installed"` for the post-install flow; callers that saved a
                key rather than installed a package can pass e.g. `"Saved"`.
            body: Optional override for the explanatory line under the title.
                Defaults to the post-install copy.
        """
        super().__init__()
        self._label = label
        self._verb = verb
        self._body = body or self._DEFAULT_BODY

    def compose(self) -> ComposeResult:
        """Compose the confirmation dialog.

        Yields:
            Title, body, and help-row widgets parented inside a `Vertical`.
        """
        glyphs = get_glyphs()
        with Vertical():
            yield Static(
                Content.from_markup(
                    "$check $verb [bold]$name[/bold]",
                    check=glyphs.checkmark,
                    verb=self._verb,
                    name=self._label,
                ),
                classes="restart-prompt-title",
                markup=False,
            )
            yield Static(
                self._body,
                classes="restart-prompt-body",
                markup=False,
            )
            yield Static(
                "Enter to restart, Esc to defer",
                classes="restart-prompt-help",
                markup=False,
            )

    def action_restart(self) -> None:
        """Dismiss with `"restart"`."""
        self.dismiss("restart")

    def action_later(self) -> None:
        """Dismiss with `"later"`."""
        self.dismiss("later")

    def action_cancel(self) -> None:
        """Alias for `action_later` so Esc resolves to a deliberate defer.

        The app's `action_interrupt` (`escape` binding, `priority=True`)
        fires before this screen's own `escape` binding. When the active
        screen is a `ModalScreen`, it dispatches to `action_cancel` if
        present, else falls through to `dismiss(None)`. The current caller
        no-ops on both `None` and `"later"`, but defining this alias pins Esc
        to an explicit `"later"` — matching the sibling reconnect/cwd-switch
        modals and keeping the outcome unambiguous for any future caller that
        branches on a deliberate defer versus a programmatic dismiss.
        """
        self.action_later()
