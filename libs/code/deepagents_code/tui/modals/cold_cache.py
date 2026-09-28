"""Modal for a cold prompt cache: send-time confirmation or expiry handoff."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, ClassVar, assert_never

from textual.binding import Binding, BindingType
from textual.containers import Vertical, VerticalScroll
from textual.content import Content
from textual.screen import ModalScreen
from textual.widgets import Static

from deepagents_code._session_stats import format_cost_estimate
from deepagents_code.cold_cache import format_cache_age
from deepagents_code.config import get_glyphs
from deepagents_code.tui.key_hints import modal_navigation_hint

if TYPE_CHECKING:
    from textual.app import ComposeResult
    from textual.events import Click

    from deepagents_code.cold_cache import ColdCacheWarning


class ColdCacheChoice(Enum):
    """How to resolve a cold prompt-cache warning."""

    HANDOFF = "handoff"
    """Start a summarized new thread; never sends, so not in `SEND_CHOICES`."""

    SEND = "send"
    """Send this turn; keep warning on future cold-cache turns."""

    SEND_SUPPRESS_SESSION = "send_suppress_session"
    """Send this turn; skip the warning until the app restarts."""

    SEND_SUPPRESS_ALWAYS = "send_suppress_always"
    """Send this turn; persistently suppress the warning in config.toml."""

    CANCEL = "cancel"
    """Keep the draft instead of sending."""


SEND_CHOICES: frozenset[ColdCacheChoice] = frozenset(
    {
        ColdCacheChoice.SEND,
        ColdCacheChoice.SEND_SUPPRESS_SESSION,
        ColdCacheChoice.SEND_SUPPRESS_ALWAYS,
    }
)
"""Choices that authorize sending the submitted turn.

Lives with the enum so callers restate the set in one place only. A new variant
is excluded until added here, which fails closed: an unlisted choice is treated
as cancel rather than silently sending.

Membership is the required spelling rather than a property on the enum, because
the value under test is `ColdCacheChoice | None` -- a programmatic pop dismisses
with `None`, and `None in SEND_CHOICES` is safely `False` where an attribute
access would raise.
"""


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
    """Offer actions for a turn whose prompt cache may be cold.

    Dismisses with the chosen `ColdCacheChoice`, or `None` on a
    programmatic pop. Esc is mapped to `CANCEL` so the user is never
    forced into a spend they did not explicitly choose. `None` and
    `CANCEL` are both non-send outcomes. `HANDOFF` authorizes summarization
    without sending the submitted turn.
    """

    can_focus = True

    BINDINGS: ClassVar[list[BindingType]] = [
        # Esc and shift+tab are both claimed by app-level priority bindings
        # that dispatch to `action_cancel` / `action_move_up` directly, so
        # these two entries never fire in the running app. Kept so the screen
        # is self-contained under a bare `App` test host and stays correct if
        # the app-level routing is ever narrowed.
        Binding("escape", "cancel", "Cancel", show=False, priority=True),
        Binding("up", "move_up", "Up", show=False, priority=True),
        Binding("k", "move_up", "Up", show=False, priority=True),
        Binding("down", "move_down", "Down", show=False, priority=True),
        Binding("j", "move_down", "Down", show=False, priority=True),
        Binding("tab", "move_down", "Next", show=False, priority=True),
        Binding("shift+tab", "move_up", "Previous", show=False, priority=True),
        Binding("enter", "activate", "Select", show=False, priority=True),
        Binding(
            "ctrl+c", "quit_or_interrupt", "Quit/Interrupt", show=False, priority=True
        ),
        Binding("ctrl+d", "quit_app", "Quit", show=False, priority=True),
    ]

    CSS = """
    ColdCacheWarningScreen {
        align: center middle;
    }

    ColdCacheWarningScreen > Vertical {
        width: 72;
        max-width: 90%;
        height: auto;
        max-height: 100%;
        background: $surface;
        border: solid $warning;
        padding: 1 2;
    }

    ColdCacheWarningScreen .cold-cache-title {
        dock: top;
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

    ColdCacheWarningScreen #cold-cache-body-scroll {
        height: auto;
        max-height: 100%;
        min-height: 1;
    }

    ColdCacheWarningScreen #cold-cache-actions {
        dock: bottom;
        height: auto;
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
        warning: ColdCacheWarning | None,
        *,
        handoff: bool = False,
        allow_send: bool = False,
    ) -> None:
        """Initialize the warning from validated policy and pricing data.

        Takes the whole `ColdCacheWarning` rather than its fields separately so
        the `age_seconds`/`reason` pairing it enforces cannot be split apart at
        this boundary. Passing them individually let an `age_unknown` warning
        arrive with a defaulted `reason`, which rendered "idle for 0m" -- an
        idle duration the caller had explicitly determined it did not know.

        Args:
            warning: Validated policy, pricing, and cause for this turn, or
                `None` when no reliable estimate exists. The body then shows
                a generic expiry notice.
            handoff: Offer a summarized thread as the default action.
            allow_send: Include a send action in the handoff menu when a
                message has been submitted.
        """
        super().__init__()
        self._handoff = handoff
        self._allow_send = allow_send
        self._warning = warning
        self._options: list[_ChoiceOption] = []
        self._selected = 0

    def _body(self) -> str:
        """Explain the possible extra cost and why it applies.

        Returns:
            Plain-text warning body.
        """
        if self._warning is None:
            return (
                "This conversation's cache may have expired. Continuing could "
                "cost more, but an estimate isn't available."
            )
        expires = self._warning.policy.confidence == "expired"
        # `certain` is set by the arm that already knows the answer rather than
        # re-tested afterwards, so the cost sentence cannot drift out of step
        # with the status sentence above it.
        match self._warning.reason:
            case "identity_changed":
                # Model settings covers model, endpoint, and cache parameter
                # changes without claiming that the model itself changed.
                certain = True
                status = (
                    "Your model settings changed, so the conversation needs "
                    "to be processed again."
                )
            case "age_unknown":
                certain = False
                status = "We can't tell whether this conversation is still cached."
            case "idle":
                certain = expires
                age = format_cache_age(self._warning.age_seconds or 0.0)
                if expires:
                    status = (
                        f"After {age} of inactivity, this conversation's cache "
                        "has likely expired."
                    )
                else:
                    status = (
                        f"After {age} of inactivity, this conversation's cache "
                        "may have expired."
                    )
            case _:  # pragma: no cover - exhaustiveness guard
                assert_never(self._warning.reason)
        # Both figures are worst-case estimates from synthetic usage payloads:
        # the cache may be partially warm and the actual spend lower, so the
        # modal rounds them and frames the total as an "up to" bound and the
        # delta as an "about" figure. Only `identity_changed` and an expired
        # window are certainties; `may_be_cold` and `age_unknown` both leave
        # open that the cache is intact, so the cost sentence stays conditional
        # on it having expired.
        conditional = "Rereading" if certain else "If the cache has expired, rereading"
        estimate = self._warning.estimate
        cost = (
            f"{conditional} this conversation could cost up to "
            f"{format_cost_estimate(estimate.cold_cost_usd)}, about "
            f"{format_cost_estimate(estimate.incremental_cost_usd)} extra. "
            "This excludes the reply."
        )
        return f"{status}\n\n{cost}"

    def _choices(self) -> tuple[tuple[ColdCacheChoice, str], ...]:
        """Return the available actions in navigation order."""
        if self._handoff:
            choices = ((ColdCacheChoice.HANDOFF, "Start new thread with summary"),)
            if self._allow_send:
                choices += ((ColdCacheChoice.SEND, "Send in current thread"),)
            cancel = (
                "Cancel (keep draft)" if self._allow_send else "Stay in current thread"
            )
            return (*choices, (ColdCacheChoice.CANCEL, cancel))
        return (
            (ColdCacheChoice.SEND, "Send anyway"),
            (
                ColdCacheChoice.SEND_SUPPRESS_SESSION,
                "Send and don't warn again this session",
            ),
            (ColdCacheChoice.SEND_SUPPRESS_ALWAYS, "Send and never warn again"),
            (ColdCacheChoice.CANCEL, "Don't send (keep draft)"),
        )

    def compose(self) -> ComposeResult:
        """Compose the warning dialog.

        Yields:
            Title, warning copy, action rows, and keyboard help.
        """
        glyphs = get_glyphs()
        with Vertical():
            yield Static(
                "Continuing may cost more",
                classes="cold-cache-title",
                markup=False,
            )
            with VerticalScroll(id="cold-cache-body-scroll"):
                yield Static(self._body(), classes="cold-cache-body", markup=False)
                if self._handoff:
                    yield Static(
                        "Start a new thread with a summary of this conversation. "
                        "Your original thread stays available, and your draft "
                        "won't be sent. Creating the summary also costs money; "
                        "overall savings aren't guaranteed.",
                        classes="cold-cache-body",
                        markup=False,
                    )
            with Vertical(id="cold-cache-actions"):
                for choice, label in self._choices():
                    option = _ChoiceOption(choice, label)
                    self._options.append(option)
                    yield option
                cancel_hint = (
                    "stay" if self._handoff and not self._allow_send else "cancel"
                )
                help_text = (
                    f"{modal_navigation_hint(glyphs)} "
                    f"{glyphs.bullet} Enter select "
                    f"{glyphs.bullet} Esc {cancel_hint}"
                )
                yield Static(help_text, classes="cold-cache-help", markup=False)
                if self._handoff:
                    yield Static(
                        "Manage this warning in /notifications",
                        classes="cold-cache-help",
                        markup=False,
                    )

    def on_mount(self) -> None:
        """Focus the scrollable copy and select the first action.

        The body receives Page Up/Down while the modal's priority bindings
        keep Tab and arrow keys navigating the pinned actions.
        """
        self.query_one("#cold-cache-body-scroll", VerticalScroll).focus()
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
        """Cancel the pending send or decline the handoff, keeping the draft.

        The method name must stay `cancel`: the app owns a priority `escape`
        binding that, for an active `ModalScreen`, dispatches to `action_cancel`
        if present and otherwise falls through to `dismiss(None)`. Renaming this
        would silently regress Esc to a `None` dismiss instead of an explicit
        cancel.
        """
        self.dismiss(ColdCacheChoice.CANCEL)
