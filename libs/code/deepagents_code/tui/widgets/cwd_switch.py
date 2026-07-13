"""Prompt for switching cwd when resuming or switching threads."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal, assert_never, cast

from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Static

from deepagents_code.sessions import format_path

if TYPE_CHECKING:
    from textual.app import ComposeResult

    from deepagents_code.app import DeepAgentsApp


CwdSwitchChoice = Literal["switch", "stay", "abort"]
"""Outcome of the cwd switch prompt.

`"abort"` is only offered when the prompt is opened with an `abort` mode set;
its meaning depends on that mode (see `CwdSwitchAbortMode`).
"""

CwdSwitchAbortMode = Literal["resume", "thread_switch"]
"""Which flow opened an abort-capable prompt, selecting the abort wording.

Passed as the prompt's `abort` argument; `None` there means abort is not
offered. `"resume"` is the launch-time `-r` resume (abort starts a new
session); `"thread_switch"` is the in-session `/threads` switcher (abort keeps
the current thread). Its members are kept disjoint from `CwdSwitchChoice`'s as a
naming convention -- not a type guarantee (these are distinct `Literal` types
used at distinct sites, so a checker already keeps them apart) -- so a mode token
is never mistaken for an outcome token in a log, test, or debugger.
`test_abort_mode_tokens_disjoint_from_choice` enforces it.
"""


class CwdSwitchPromptScreen(ModalScreen[CwdSwitchChoice]):
    """Modal asking whether to switch cwd when resuming or switching to a thread."""

    can_focus = True
    can_focus_children = False

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("enter", "switch", "Switch", show=False, priority=True),
        Binding("escape", "stay", "Stay", show=False, priority=True),
        Binding("a", "abort", "Abort", show=False, priority=True),
        Binding(
            "ctrl+c",
            "quit_or_interrupt",
            "Quit/Interrupt",
            show=False,
            priority=True,
        ),
        Binding("ctrl+d", "quit_app", "Quit", show=False, priority=True),
    ]

    CSS = """
    CwdSwitchPromptScreen {
        align: center middle;
    }

    CwdSwitchPromptScreen > Vertical {
        width: 72;
        max-width: 90%;
        height: auto;
        background: $surface;
        border: solid $warning;
        padding: 1 2;
    }

    CwdSwitchPromptScreen .cwd-switch-title {
        text-style: bold;
        color: $warning;
        text-align: center;
        margin-bottom: 1;
    }

    CwdSwitchPromptScreen .cwd-switch-body {
        height: auto;
        color: $text;
        margin-bottom: 1;
    }

    CwdSwitchPromptScreen .cwd-switch-help {
        height: 1;
        color: $text-muted;
        text-style: italic;
        text-align: center;
    }
    """

    def __init__(
        self,
        *,
        current_cwd: str,
        thread_cwd: str,
        project_settings_change_detected: bool = False,
        abort: CwdSwitchAbortMode | None = None,
    ) -> None:
        """Initialize the prompt."""
        super().__init__()
        self._current_cwd = current_cwd
        self._thread_cwd = thread_cwd
        self._project_settings_change_detected = project_settings_change_detected
        self._abort: CwdSwitchAbortMode | None = abort

    def _title_text(self) -> str:
        """Return the title, phrased for the flow that opened the prompt.

        The in-session `/threads` switcher (`"thread_switch"`) asks about
        switching; every other flow (launch-time resume, or no abort mode) asks
        about resuming. Structured for `assert_never` exhaustiveness so a new
        mode fails statically here rather than silently inheriting the resume
        wording.
        """
        if self._abort is None or self._abort == "resume":
            return "Resume from the thread's original directory?"
        if self._abort == "thread_switch":
            return "Switch to the thread's original directory?"
        assert_never(self._abort)

    def _body_text(self) -> str:
        """Return the prompt body text."""
        current = format_path(self._current_cwd)
        target = format_path(self._thread_cwd)
        settings_note = (
            "\n\nSwitching may also reload project-specific config like .env, "
            "MCP, skills, and AGENTS.md."
            if self._project_settings_change_detected
            else ""
        )
        if self._abort is None:
            abort_note = ""
        elif self._abort == "resume":
            abort_note = "\n\nOr abort to start a new session instead of resuming."
        elif self._abort == "thread_switch":
            abort_note = (
                "\n\nOr abort to keep your current thread instead of switching."
            )
        else:
            assert_never(self._abort)
        return (
            "This thread was last used from:\n"
            f"  {target}\n\n"
            "You're currently in:\n"
            f"  {current}\n\n"
            "Switch if you want local context, project instructions, skills, "
            "MCP config, and env files to match the original directory. Stay "
            "here if you intentionally want to continue this thread against "
            f"the current directory.{settings_note}{abort_note}"
        )

    def _help_text(self) -> str:
        """Return the help line text, naming the mode's abort action if offered."""
        help_text = "Enter: switch · Esc: stay here"
        if self._abort is None:
            return help_text
        if self._abort == "resume":
            abort_help = "A: don't resume"
        elif self._abort == "thread_switch":
            abort_help = "A: don't switch"
        else:
            assert_never(self._abort)
        return f"{help_text} · {abort_help}"

    def compose(self) -> ComposeResult:
        """Compose the confirmation dialog.

        Yields:
            Widgets for the cwd switch prompt.
        """
        with Vertical():
            yield Static(
                self._title_text(),
                classes="cwd-switch-title",
                markup=False,
            )
            yield Static(
                self._body_text(),
                classes="cwd-switch-body",
                markup=False,
            )
            yield Static(
                self._help_text(),
                classes="cwd-switch-help",
                markup=False,
            )

    def on_mount(self) -> None:
        """Focus the modal so screen bindings work after nested modal flows."""
        self.focus()

    def check_action(
        self,
        action: str,
        parameters: tuple[object, ...],  # noqa: ARG002  # required by Textual's DOMNode.check_action override signature
    ) -> bool | None:
        """Disable the `abort` binding unless the prompt was opened for it.

        Textual gates a binding's action on a truthy `check_action` result, so
        both `False` and `None` stop `a` from dispatching `action_abort` -- in
        neither case does the key fire the action, and it falls through the same
        way. They differ only in footer presentation (`False` hides the binding,
        `None` shows it grayed out), which is moot here anyway: every binding is
        declared `show=False` and the modal renders its own help line instead of
        a `Footer`. We return `False` to mark the disabled state explicitly; the
        actual inertness backstop is `action_abort`'s own `self._abort is None`
        guard, should the action ever be dispatched.

        Returns:
            `self._abort is not None` for the `abort` action, so the binding is
                enabled only when abort was offered; `True` for every other action.
        """
        if action == "abort":
            return self._abort is not None
        return True

    def action_switch(self) -> None:
        """Dismiss with `switch`."""
        self.dismiss("switch")

    def action_stay(self) -> None:
        """Dismiss with `stay`."""
        self.dismiss("stay")

    def action_abort(self) -> None:
        """Dismiss with `abort` to skip the resume/switch, when the prompt allows it."""
        if self._abort is None:
            return
        self.dismiss("abort")

    def action_cancel(self) -> None:
        """Treat cancellation as staying in the current cwd."""
        self.action_stay()

    def action_quit_or_interrupt(self) -> None:
        """Delegate Ctrl+C to the app-level quit/interrupt handler."""
        cast("DeepAgentsApp", self.app).action_quit_or_interrupt()

    def action_quit_app(self) -> None:
        """Delegate Ctrl+D to the app-level quit handler."""
        cast("DeepAgentsApp", self.app).action_quit_app()
