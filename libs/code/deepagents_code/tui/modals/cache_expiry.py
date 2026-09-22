"""Confirmation before starting a summarized thread after cache expiry."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, override

from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Static

from deepagents_code.config import get_glyphs

if TYPE_CHECKING:
    from textual.app import ComposeResult


class CacheExpiryScreen(ModalScreen[bool]):
    """Offer an explicit fresh-thread handoff without submitting the draft."""

    can_focus = True
    CSS_PATH = "cache_expiry.tcss"
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("enter", "handoff", "New thread", show=False, priority=True),
        Binding("escape", "bypass", "Continue", show=False, priority=True),
        Binding(
            "ctrl+c", "quit_or_interrupt", "Quit/Interrupt", show=False, priority=True
        ),
        Binding("ctrl+d", "quit_app", "Quit", show=False, priority=True),
    ]

    @override
    def compose(self) -> ComposeResult:
        """Yield the cache-expiry choices."""
        with Vertical():
            yield Static("Cache retention timer expired", classes="title")
            yield Static(
                "The next turn may need to re-warm the prompt cache. Start a new "
                "thread with an LLM-generated summary? This uses a model call. "
                "The summary includes the previous thread ID and a transcript "
                "path for recovering details; the original thread is preserved.",
                classes="body",
                markup=False,
            )
            yield Static(
                f" {get_glyphs().bullet} ".join(
                    ("Enter: summarize into a new thread", "Esc: continue this thread")
                ),
                classes="choices",
                markup=False,
            )
            yield Static(
                "Configure warnings.cache_expiry_prompt in /config",
                classes="hint",
                markup=False,
            )

    def on_mount(self) -> None:
        """Focus the modal rather than the underlying chat input."""
        self.focus()

    def action_handoff(self) -> None:
        """Confirm a summarized new thread."""
        self.dismiss(True)

    def action_bypass(self) -> None:
        """Keep the current thread unchanged."""
        self.dismiss(False)
