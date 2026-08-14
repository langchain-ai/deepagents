"""Confirmation modal for an expensive cold prompt-cache turn."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, ClassVar

from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.content import Content
from textual.screen import ModalScreen
from textual.widgets import Static

from deepagents_code._session_stats import format_cost_estimate, format_token_count
from deepagents_code.cold_cache import (
    PromptCachePolicy,
    RewarmEstimate,
    format_cache_age,
    format_cache_window,
)
from deepagents_code.config import get_glyphs

if TYPE_CHECKING:
    from textual.app import ComposeResult
    from textual.events import Click


class ColdCacheChoice(Enum):
    """How to resolve a cold prompt-cache warning."""

    SEND = "send"
    """Send this turn; keep warning on future cold-cache turns."""

    SEND_SUPPRESS_SESSION = "send_suppress_session"
    """Send this turn; skip the warning until the app restarts."""

    SEND_SUPPRESS_ALWAYS = "send_suppress_always"
    """Send this turn; persistently suppress the warning in config.toml."""

    CANCEL = "cancel"
    """Keep the draft instead of sending."""


class _ChoiceOption(Static):
    """Clickable single-line choice row."""

    def __init__(self, choice: ColdCacheChoice, label: str) -> None:
        """Initialize the choice row widget.

        Args:
            choice: The choice this row resolves to.
            label: User-facing row text.
        """
        super().__init__(classes="cold-cache-choice")
        self._choice = choice
        self._label = label
        self._is_selected = False
        self.update(self._render())

    @property
    def choice(self) -> ColdCacheChoice:
        """Underlying choice."""
        return self._choice

    def set_selected(self, selected: bool) -> None:
        """Toggle selection styling.

        Args:
            selected: Whether this row is currently under the cursor.
        """
        if self._is_selected == selected:
            return
        self._is_selected = selected
        self.set_class(selected, "-selected")
        self.update(self._render())

    def _render(self) -> Content:
        glyphs = get_glyphs()
        cursor = glyphs.cursor if self._is_selected else " "
        return Content(f"{cursor} {self._label}")

    def on_click(self, event: Click) -> None:  # noqa: PLR6301  # Textual event handler
        """Swallow the click without activating.

        Clicks are intentionally disabled so an accidental mouse press
        cannot authorize spend or persist a suppression. Activation is
        keyboard-only (enter), matching the update-available modal.
        """
        event.stop()


class ColdCacheWarningScreen(ModalScreen[ColdCacheChoice | None]):
    """Ask whether to send a turn whose prompt cache may be cold.

    Dismisses with the chosen `ColdCacheChoice`, or `None` on a
    programmatic pop. Esc is mapped to `CANCEL` so the user is never
    forced into a spend they did not explicitly choose. `None` and
    `CANCEL` are both non-send outcomes, and callers must treat any
    non-send value as cancel so the dialog fails closed.
    """

    can_focus = True

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Cancel", show=False, priority=True),
        Binding("up", "move_up", "Up", show=False, priority=True),
        Binding("k", "move_up", "Up", show=False, priority=True),
        Binding("down", "move_down", "Down", show=False, priority=True),
        Binding("j", "move_down", "Down", show=False, priority=True),
        Binding("tab", "move_down", "Next", show=False, priority=True),
        Binding("shift+tab", "move_up", "Previous", show=False, priority=True),
        Binding("enter", "activate", "Select", show=False, priority=True),
    ]

    CSS = """
    ColdCacheWarningScreen {
        align: center middle;
    }

    ColdCacheWarningScreen > Vertical {
        width: 72;
        max-width: 90%;
        height: auto;
        background: $surface;
        border: solid $warning;
        padding: 1 2;
    }

    ColdCacheWarningScreen .cold-cache-title {
        text-style: bold;
        color: $warning;
        text-align: center;
        margin-bottom: 1;
    }

    ColdCacheWarningScreen .cold-cache-body {
        height: auto;
        color: $text;
        margin-bottom: 1;
    }

    ColdCacheWarningScreen .cold-cache-choice {
        height: auto;
        padding: 0 1;
        color: $text;
    }

    ColdCacheWarningScreen .cold-cache-choice.-selected {
        background: $surface-lighten-1;
    }

    ColdCacheWarningScreen .cold-cache-help {
        height: 1;
        color: $text-muted;
        text-style: italic;
        text-align: center;
        margin-top: 1;
    }
    """

    def __init__(
        self,
        *,
        policy: PromptCachePolicy,
        estimate: RewarmEstimate,
        context_tokens: int,
        age_seconds: float,
        identity_changed: bool = False,
    ) -> None:
        """Initialize the warning from validated policy and pricing data."""
        super().__init__()
        self._policy = policy
        self._estimate = estimate
        self._context_tokens = context_tokens
        self._age_seconds = age_seconds
        self._identity_changed = identity_changed
        self._options: list[_ChoiceOption] = []
        self._selected = 0

    def _body(self) -> str:
        """Build provider-aware warning copy.

        Returns:
            Plain-text warning body.
        """
        age = format_cache_age(self._age_seconds)
        window = format_cache_window(self._policy.window_seconds)
        if self._identity_changed:
            status = (
                "The active model or prompt-cache settings differ from the last "
                "successful turn, so the previous cached prefix cannot be reused."
            )
        elif self._policy.confidence == "expired":
            status = (
                f"This thread has been idle for {age}, longer than "
                f"{self._policy.provider_name}'s {window} prompt-cache lifetime. "
                "The cached conversation prefix has likely expired."
            )
        else:
            status = (
                f"This thread has been idle for {age}, longer than "
                f"{self._policy.provider_name}'s {window} minimum cache-retention "
                "window. The provider may still have retained the cache."
            )
        # Both figures are worst-case estimates from synthetic usage payloads:
        # the cache may be partially warm and the actual spend lower, so the
        # modal rounds them and frames them as "up to" bounds. Under the
        # `may_be_cold` branch the cache may also be fully intact, so the cost
        # sentence stays conditional on the cache having expired.
        conditional = (
            "If the cache has expired, re-processing"
            if self._policy.confidence == "may_be_cold" and not self._identity_changed
            else "Re-processing"
        )
        cost = (
            f"{conditional} approximately "
            f"{format_token_count(self._context_tokens)} history tokens may "
            f"cost up to {format_cost_estimate(self._estimate.cold_cost_usd)} "
            f"in input tokens, roughly "
            f"{format_cost_estimate(self._estimate.incremental_cost_usd)} more "
            "than a warm cache hit."
        )
        return f"{status}\n\n{cost}"

    def compose(self) -> ComposeResult:
        """Compose the warning dialog.

        Yields:
            Title, warning copy, one row per choice, and keyboard help.
        """
        glyphs = get_glyphs()
        with Vertical():
            yield Static(
                "Warning: cache may be cold",
                classes="cold-cache-title",
                markup=False,
            )
            yield Static(self._body(), classes="cold-cache-body", markup=False)
            for choice, label in (
                (ColdCacheChoice.SEND, "Send anyway"),
                (
                    ColdCacheChoice.SEND_SUPPRESS_SESSION,
                    "Send and don't warn again this session",
                ),
                (
                    ColdCacheChoice.SEND_SUPPRESS_ALWAYS,
                    "Send and never warn again",
                ),
                (ColdCacheChoice.CANCEL, "Don't send (keep draft)"),
            ):
                option = _ChoiceOption(choice, label)
                self._options.append(option)
                yield option
            help_text = (
                f"{glyphs.arrow_up}/{glyphs.arrow_down} or Tab navigate "
                f"{glyphs.bullet} Enter select "
                f"{glyphs.bullet} Esc cancel"
            )
            yield Static(help_text, classes="cold-cache-help", markup=False)

    def on_mount(self) -> None:
        """Focus the modal and default the cursor to the send row."""
        self.focus()
        self._set_selected(0)

    def _set_selected(self, new_index: int) -> None:
        """Move the selection cursor to *new_index*."""
        if not self._options:
            return
        if new_index != self._selected:
            self._options[self._selected].set_selected(selected=False)
        self._selected = new_index
        self._options[new_index].set_selected(selected=True)

    def action_move_up(self) -> None:
        """Move the cursor up one row (wraps at the top)."""
        if not self._options:
            return
        self._set_selected((self._selected - 1) % len(self._options))

    def action_move_down(self) -> None:
        """Move the cursor down one row (wraps at the bottom)."""
        if not self._options:
            return
        self._set_selected((self._selected + 1) % len(self._options))

    def action_activate(self) -> None:
        """Resolve with the highlighted choice."""
        if not self._options:
            self.dismiss(None)
            return
        self.dismiss(self._options[self._selected].choice)

    def action_cancel(self) -> None:
        """Cancel the pending send, keeping the draft.

        The method name must stay `cancel`: the app owns a priority `escape`
        binding that, for an active `ModalScreen`, dispatches to `action_cancel`
        if present and otherwise falls through to `dismiss(None)`. Renaming this
        would silently regress Esc to a `None` dismiss instead of an explicit
        cancel.
        """
        self.dismiss(ColdCacheChoice.CANCEL)
