"""Confirmation modal for an expensive cold prompt-cache turn."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from deepagents_code._session_stats import format_cost, format_token_count
from deepagents_code.cold_cache import (
    PromptCachePolicy,
    RewarmEstimate,
    format_cache_age,
    format_cache_window,
)

if TYPE_CHECKING:
    from textual.app import ComposeResult


class ColdCacheWarningScreen(ModalScreen[bool]):
    """Ask whether to send a turn whose prompt cache may be cold."""

    can_focus = True

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Cancel", show=False, priority=True),
        Binding("s", "send", "Send anyway", show=False, priority=True),
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

    ColdCacheWarningScreen .cold-cache-actions {
        width: 100%;
        height: auto;
        align-horizontal: center;
    }

    ColdCacheWarningScreen Button {
        margin: 0 1;
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
                "The cached conversation prefix has expired."
            )
        else:
            status = (
                f"This thread has been idle for {age}, longer than "
                f"{self._policy.provider_name}'s {window} minimum cache-retention "
                "window. The provider may still have retained the cache."
            )
        return (
            f"{status}\n\n"
            f"Re-processing approximately {format_token_count(self._context_tokens)} "
            f"history tokens may cost {format_cost(self._estimate.cold_cost_usd)} "
            "in input tokens, roughly "
            f"{format_cost(self._estimate.incremental_cost_usd)} more than a warm "
            "cache hit. Output cost is not included.\n\n"
            "Cancel and use /clear to start without the old context."
        )

    def compose(self) -> ComposeResult:
        """Compose warning copy and safe-default actions.

        Yields:
            Warning text and explicit cancel/send controls.
        """
        with Vertical():
            yield Static(
                "Prompt cache may be cold",
                classes="cold-cache-title",
                markup=False,
            )
            yield Static(self._body(), classes="cold-cache-body", markup=False)
            with Horizontal(classes="cold-cache-actions"):
                yield Button("Cancel", id="cold-cache-cancel")
                yield Button(
                    "Send anyway",
                    id="cold-cache-send",
                    variant="warning",
                )

    def on_mount(self) -> None:
        """Focus Cancel so an extra Enter cannot authorize spend."""
        self.query_one("#cold-cache-cancel", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Resolve the modal from its explicit actions."""
        event.stop()
        if event.button.id == "cold-cache-send":
            self.dismiss(True)
        else:
            self.dismiss(False)

    def action_cancel(self) -> None:
        """Cancel the pending send."""
        self.dismiss(False)

    def action_send(self) -> None:
        """Authorize the pending send."""
        self.dismiss(True)
