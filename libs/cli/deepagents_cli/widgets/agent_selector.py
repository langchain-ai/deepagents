"""Interactive agent selector screen for `/agents` command."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import TYPE_CHECKING, ClassVar

from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.content import Content
from textual.screen import ModalScreen
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option

if TYPE_CHECKING:
    from textual.app import ComposeResult

from deepagents_cli import theme
from deepagents_cli.config import Glyphs, get_glyphs, is_ascii_mode
from deepagents_cli.model_config import clear_default_agent, save_default_agent

logger = logging.getLogger(__name__)


class AgentSelectorScreen(ModalScreen[str | None]):
    """Modal dialog for switching between available agents.

    Displays agents found in `~/.deepagents/` in an `OptionList`. Returns the
    selected agent name on Enter, or `None` on Esc (no change).
    `Ctrl+S` toggles the highlighted agent as the persisted default
    (`[agents].default`), mirroring the model selector's affordance.
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("tab", "cursor_down", "Next", show=False, priority=True),
        Binding("shift+tab", "cursor_up", "Previous", show=False, priority=True),
        Binding("ctrl+s", "set_default", "Set default", show=False, priority=True),
    ]
    """Key bindings for the selector.

    Esc dismisses without switching agents. Arrow keys, Enter, and letter
    navigation are handled natively by the embedded `OptionList`; Tab /
    Shift+Tab are bound here to advance the cursor for consistency with
    other selector screens. Ctrl+S toggles the highlighted agent as the
    persisted default.
    """

    CSS = """
    AgentSelectorScreen {
        align: center middle;
        background: transparent;
    }

    AgentSelectorScreen > Vertical {
        width: 60;
        max-width: 90%;
        height: auto;
        max-height: 80%;
        background: $surface;
        border: solid $primary;
        padding: 1 2;
    }

    AgentSelectorScreen .agent-selector-title {
        text-style: bold;
        color: $primary;
        text-align: center;
        margin-bottom: 1;
    }

    AgentSelectorScreen .agent-selector-subtitle {
        height: auto;
        color: $text-muted;
        text-align: center;
        margin-bottom: 1;
    }

    AgentSelectorScreen OptionList {
        height: auto;
        max-height: 16;
        background: $background;
    }

    AgentSelectorScreen .agent-selector-help {
        height: auto;
        color: $text-muted;
        text-style: italic;
        margin-top: 1;
        text-align: center;
    }
    """
    """Styling for the centered modal shell, title, option list, and help footer."""

    def __init__(
        self,
        current_agent: str | None,
        agent_names: list[str],
        *,
        default_agent: str | None = None,
    ) -> None:
        """Initialize the `AgentSelectorScreen`.

        Args:
            current_agent: The name of the currently active agent (to
                highlight).

                May be `None` when no agent is active.
            agent_names: Sorted list of available agent names to display.
            default_agent: The persisted default agent name from
                `[agents].default`, or `None` if no default is set.
        """
        super().__init__()
        self._current_agent = current_agent
        self._agent_names = agent_names
        self._default_agent = default_agent

    def compose(self) -> ComposeResult:
        """Compose the screen layout.

        Yields:
            Widgets for the agent selector UI.
        """
        glyphs = get_glyphs()

        with Vertical():
            yield Static("Select Agent", classes="agent-selector-title")
            if self._agent_names:
                yield Static(
                    "Switching restarts the agent and starts a new thread.",
                    classes="agent-selector-subtitle",
                )
                option_list = OptionList(
                    *self._build_options(),
                    id="agent-options",
                )
                option_list.highlighted = self._current_index()
                yield option_list
                help_text = self._help_text(glyphs)
            else:
                yield Static(
                    "No agents found in ~/.deepagents/.\n"
                    "Run the CLI with -a <name> to create one.",
                    classes="agent-selector-help",
                )
                help_text = f"{glyphs.bullet} Esc close"
            yield Static(help_text, classes="agent-selector-help", id="agent-help")

    def _build_options(self) -> list[Option]:
        """Build option entries with `(current)` / `(default)` suffixes.

        Render labels via `Content.from_markup` so agent directory names
        containing Rich markup characters (e.g. `[`) don't break rendering.

        Returns:
            One `Option` per agent name in `self._agent_names`.
        """
        return [Option(self._format_label(name), id=name) for name in self._agent_names]

    def _format_label(self, name: str) -> Content:
        """Render an agent's label with `(current)` / `(default)` markers.

        Args:
            name: The agent directory name.

        Returns:
            Styled `Content` label.
        """
        is_current = name == self._current_agent
        is_default = name == self._default_agent
        if is_current and is_default:
            return Content.from_markup(
                "$name [dim](current,[/dim] [bold]default[/bold][dim])[/dim]",
                name=name,
            )
        if is_current:
            return Content.from_markup("$name [dim](current)[/dim]", name=name)
        if is_default:
            return Content.from_markup(
                "$name [dim]([/dim][bold]default[/bold][dim])[/dim]", name=name
            )
        return Content.from_markup("$name", name=name)

    def _current_index(self) -> int:
        """Return the index of the current agent in the option list, or 0."""
        if self._current_agent is None:
            return 0
        try:
            return self._agent_names.index(self._current_agent)
        except ValueError:
            return 0

    @staticmethod
    def _help_text(glyphs: Glyphs) -> str:
        r"""Build the help-line text shown beneath the option list.

        Split into two balanced rows joined by `\n` so the wrap is
        predictable at the modal's fixed 60-column width — Textual's
        default word-wrap might otherwise break mid-phrase (e.g.,
        between "set" and "default"), which reads as a bug. The Static
        host has `height: auto` and `text-align: center`, so each row
        centers on its own line.

        Args:
            glyphs: Glyph set for the active terminal mode.

        Returns:
            Two-line help string describing the available key bindings.
        """
        return (
            f"{glyphs.arrow_up}/{glyphs.arrow_down} or Tab switch"
            f" {glyphs.bullet} Enter select\n"
            f"Ctrl+S set default {glyphs.bullet} Esc cancel"
        )

    def on_mount(self) -> None:
        """Apply ASCII border if needed."""
        if is_ascii_mode():
            container = self.query_one(Vertical)
            colors = theme.get_theme_colors(self)
            container.styles.border = ("ascii", colors.success)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Dismiss with the selected agent name.

        Args:
            event: The option selected event.
        """
        name = event.option.id
        self.dismiss(name)

    def action_cancel(self) -> None:
        """Cancel without switching agents."""
        self.dismiss(None)

    def action_cursor_down(self) -> None:
        """Move the option list cursor down (Tab)."""
        option_list = self._option_list()
        if option_list is not None:
            option_list.action_cursor_down()

    def action_cursor_up(self) -> None:
        """Move the option list cursor up (Shift+Tab)."""
        option_list = self._option_list()
        if option_list is not None:
            option_list.action_cursor_up()

    async def action_set_default(self) -> None:
        """Toggle the highlighted agent as the persisted default.

        If the highlighted agent is already the default, clears it.
        Otherwise sets it as the new default. Disk I/O is offloaded to a
        thread so the modal stays responsive. The help line shows a
        transient confirmation/error message; the option list is rebuilt
        in place so the `(default)` marker tracks the new state.

        Robustness notes:
            * The `to_thread` call is wrapped in `try/except Exception` so
                an unexpected error inside the persistence functions
                (e.g., `tomli_w.dump` raising a `TypeError`) is treated as
                a normal save failure rather than killing the modal.
            * Post-await `query_one` is guarded against `NoMatches` so a
                user dismissing the modal mid-flight does not surface a
                Textual callback error.
            * The option-list rebuild is staged in `_refresh_options`,
                which builds the new options first and only swaps on
                success. A failure mid-rebuild leaves the existing list
                intact and shows an error in the help line.
        """
        from textual.css.query import NoMatches

        option_list = self._option_list()
        if option_list is None or not self._agent_names:
            return

        highlighted = option_list.highlighted
        if highlighted is None or not (0 <= highlighted < len(self._agent_names)):
            return
        name = self._agent_names[highlighted]

        if name == self._default_agent:
            new_default: str | None = None
            success_msg = "Default cleared"
            failure_msg = "Failed to clear default"
            try:
                ok = await asyncio.to_thread(clear_default_agent)
            except Exception:
                logger.exception("clear_default_agent raised unexpectedly")
                ok = False
        else:
            new_default = name
            success_msg = f"Default set to {name}"
            failure_msg = "Failed to save default"
            try:
                ok = await asyncio.to_thread(save_default_agent, name)
            except Exception:
                logger.exception("save_default_agent raised unexpectedly for %r", name)
                ok = False

        # The user may have dismissed the modal while the disk I/O was in
        # flight; don't try to mutate widgets on an unmounted screen.
        try:
            help_widget = self.query_one("#agent-help", Static)
        except NoMatches:
            return

        if not ok:
            help_widget.update(
                Content.styled(
                    failure_msg,
                    f"bold {theme.get_theme_colors(self).error}",
                )
            )
            self.set_timer(3.0, self._restore_help_text)
            return

        # Apply the rebuild before mutating `_default_agent` so a failure
        # leaves the picker visually consistent with on-disk state.
        if not self._refresh_options(option_list, highlighted, new_default):
            help_widget.update(
                Content.styled(
                    "Failed to refresh agent list",
                    f"bold {theme.get_theme_colors(self).error}",
                )
            )
            self.set_timer(3.0, self._restore_help_text)
            return

        self._default_agent = new_default
        help_widget.update(Content.styled(success_msg, "bold"))
        self.set_timer(3.0, self._restore_help_text)

    def _refresh_options(
        self,
        option_list: OptionList,
        highlighted: int,
        new_default: str | None,
    ) -> bool:
        """Rebuild option labels in place to track the new `(default)` state.

        Builds the new option list with `new_default` applied first and
        only mutates the live `OptionList` once construction succeeds.
        Failure mid-build leaves the existing list intact, avoiding the
        catastrophic empty-picker state where `clear_options()` had
        succeeded but `add_options(...)` raised.

        `OptionList.clear_options()` resets the highlight to `None`, so
        restore it explicitly to the previously selected row to avoid
        appearing to lose the cursor after a Ctrl+S toggle.

        Args:
            option_list: The live `OptionList` to rebuild in place.
            highlighted: Index to restore as the highlighted row.
            new_default: The agent name about to become the default
                (`None` for clear).

                Passed in rather than read from `self._default_agent`
                so the caller can decide whether to commit the new
                default based on the rebuild's success.

        Returns:
            `True` when the rebuild applied cleanly, `False` if option
                construction or mounting raised. On `False`, the live
                option list has been left untouched.
        """
        previous_default = self._default_agent
        self._default_agent = new_default
        try:
            new_options = self._build_options()
        except Exception:
            logger.exception("Failed to build new agent picker options")
            self._default_agent = previous_default
            return False

        try:
            option_list.clear_options()
            option_list.add_options(new_options)
        except Exception:
            logger.exception("Failed to mount rebuilt agent picker options")
            self._default_agent = previous_default
            # Best-effort restore of the prior options so the user is
            # not left staring at an empty picker.
            with contextlib.suppress(Exception):
                option_list.clear_options()
                option_list.add_options(self._build_options())
            return False

        if 0 <= highlighted < len(self._agent_names):
            option_list.highlighted = highlighted
        # `_default_agent` is set on entry so `_build_options` reflects
        # the new state; revert it here so the caller can commit only
        # after the rebuild has fully succeeded.
        self._default_agent = previous_default
        return True

    def _restore_help_text(self) -> None:
        """Restore the default help text after a transient message.

        Guarded against the user having dismissed the modal during the
        3-second timer; without the guard, `query_one` would raise
        `NoMatches` and Textual would surface it as a callback error.
        """
        from textual.css.query import NoMatches

        try:
            help_widget = self.query_one("#agent-help", Static)
        except NoMatches:
            return
        help_widget.update(self._help_text(get_glyphs()))

    def _option_list(self) -> OptionList | None:
        """Return the agent `OptionList`, or `None` when the screen is empty."""
        from textual.css.query import NoMatches

        try:
            return self.query_one("#agent-options", OptionList)
        except NoMatches:
            return None
