"""Confirmation modal for an expensive cold prompt-cache turn."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Static

from deepagents_code._session_stats import format_cost, format_token_count
from deepagents_code.cold_cache import (
    PromptCachePolicy,
    RewarmEstimate,
    format_cache_age,
    format_cache_window,
)
from deepagents_code.config import get_glyphs

if TYPE_CHECKING:
    from textual.app import ComposeResult


class ColdCacheWarningScreen(ModalScreen[bool | None]):
    """Ask whether to send a turn whose prompt cache may be cold.

    Dismisses with `True` when the user sends and `False` when the user
    keeps the draft. Esc is treated as cancel so the user is never forced
    into a spend they did not explicitly choose.

    Typed `bool | None` rather than `bool`: a programmatic pop can yield
    `None`. The caller's `if send:` collapses `None` and `False` to cancel,
    so both dismiss values fail closed.
    """

    can_focus = True

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("enter", "send", "Send", show=False, priority=True),
        Binding("escape", "cancel", "Cancel", show=False, priority=True),
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

    ColdCacheWarningScreen .cold-cache-help {
        height: 1;
        color: $text-muted;
        text-style: italic;
        text-align: center;
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
                "The cached conversation prefix has likely expired."
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
            "cache hit."
        )

    def compose(self) -> ComposeResult:
        """Compose the warning dialog.

        Yields:
            Title, warning copy, and keyboard help.
        """
        with Vertical():
            yield Static(
                "Prompt cache may be cold",
                classes="cold-cache-title",
                markup=False,
            )
            yield Static(self._body(), classes="cold-cache-body", markup=False)
            yield Static(
                f" {get_glyphs().bullet} ".join(
                    ("Enter: send anyway", "Esc: keep draft")
                ),
                classes="cold-cache-help",
                markup=False,
            )

    def on_mount(self) -> None:
        """Focus the modal so its bindings receive keyboard input."""
        self.focus()

    def action_send(self) -> None:
        """Authorize the pending send."""
        self.dismiss(True)

    def action_cancel(self) -> None:
        """Cancel the pending send, keeping the draft.

        The method name must stay `cancel`: the app owns a priority `escape`
        binding that, for an active `ModalScreen`, dispatches to `action_cancel`
        if present and otherwise falls through to `dismiss(None)`. Renaming this
        would silently regress Esc to a `None` dismiss instead of an explicit
        cancel.
        """
        self.dismiss(False)
