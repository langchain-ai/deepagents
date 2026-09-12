"""First-enable confirmation modal for Auto mode.

Shown at most once per install (per notice version) before Auto activates.
Enter confirms the switch and records the notice; Esc cancels and leaves the
notice unsaved so it can appear again next time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Markdown, Static

from deepagents_code._markdown import escape_markdown
from deepagents_code.config import get_glyphs
from deepagents_code.tui.widgets._links import open_checked_url_async

if TYPE_CHECKING:
    from textual.app import ComposeResult

AUTO_MODE_DOCS_URL = (
    "https://docs.langchain.com/oss/python/deepagents/code/approval-modes"
)
"""Canonical docs page for Manual / Auto / YOLO behavior."""

AUTO_MODE_NOTICE_MODEL_ANCHOR = "**classifier model**"
"""Phrase in `AUTO_MODE_NOTICE_BODY` that the model description replaces.

`build_auto_mode_notice_body` interpolates against this exact substring, so the
constant and the body sentence must be edited together. Rewording the sentence
without updating this value would silently drop the disclosure — hence the
`ValueError` in the builder rather than a quiet no-op `str.replace`.
"""

AUTO_MODE_NOTICE_BODY = (
    "You switched to **Auto**. The agent can approve **routine gated actions** "
    "without asking first — for example, file edits and read-only Git "
    "commands.\n\n"
    "Anything uncertain is reviewed by "
    f"{AUTO_MODE_NOTICE_MODEL_ANCHOR}. If review keeps failing, you're asked "
    "to approve.\n\n"
    "This is **not a sandbox**. The agent still runs on this machine and can "
    "change files, run commands, and use tools when Auto allows them.\n\n"
    "This notice appears **once** on this machine after you continue.\n\n"
    f"[Learn more about approval modes]({AUTO_MODE_DOCS_URL})"
)
"""Default Markdown body shown on first successful Auto enable.

Auto involves two distinct roles — the model writing your code and the model
reviewing its gated actions — and `--auto-classifier-model` (or
`[models].auto_classifier`) makes them different models. This modal is the only
place that is disclosed, so the review sentence says which model reviews *and*
whether it is the one writing the code; "active model" would read as the latter.
"""


def build_auto_mode_notice_body(
    model_label: str | None, *, distinct_from_main_model: bool
) -> str:
    """Describe the reviewing model in the Auto notice body.

    Args:
        model_label: Spec of the model that reviews gated actions, or `None`
            when it is not known (no resolved main-model spec yet).
        distinct_from_main_model: Whether that model differs from the one writing
            code. Drives the wording, because naming a model without saying
            which role it plays is what made the old copy misleading.

    Returns:
        Markdown notice body containing the safely escaped model label.

    Raises:
        ValueError: If `AUTO_MODE_NOTICE_BODY` no longer contains the
            interpolation anchor, which would drop the disclosure silently.
    """
    if AUTO_MODE_NOTICE_MODEL_ANCHOR not in AUTO_MODE_NOTICE_BODY:
        msg = (
            "AUTO_MODE_NOTICE_BODY is missing "
            f"{AUTO_MODE_NOTICE_MODEL_ANCHOR!r}; the Auto classifier model "
            "would not be disclosed"
        )
        raise ValueError(msg)
    named = f" ({escape_markdown(model_label)})" if model_label else ""
    if distinct_from_main_model:
        description = (
            f"a separate {AUTO_MODE_NOTICE_MODEL_ANCHOR}{named} — not the model "
            "writing your code"
        )
    else:
        description = (
            f"the {AUTO_MODE_NOTICE_MODEL_ANCHOR}, which is the same model "
            f"writing your code{named}"
        )
    return AUTO_MODE_NOTICE_BODY.replace(AUTO_MODE_NOTICE_MODEL_ANCHOR, description, 1)


class AutoModeNoticeScreen(ModalScreen[bool]):
    """In-TUI first-run notice describing what Auto mode does.

    Dismisses with `True` on Enter (confirm switch to Auto) and `False` on
    Esc (cancel, stay in current mode). Programmatic dismiss may yield `None`;
    callers treat that like cancel so Auto is never activated without an
    explicit confirm.
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("enter", "confirm", "Keep Auto", show=False, priority=True),
        Binding("escape", "cancel", "Manual", show=False, priority=True),
    ]

    CSS = """
    AutoModeNoticeScreen {
        align: center middle;
    }

    AutoModeNoticeScreen > Vertical {
        width: 72;
        max-width: 90%;
        height: auto;
        background: $surface;
        border: solid $warning;
        padding: 1 2;
    }

    AutoModeNoticeScreen .auto-mode-notice-title {
        text-style: bold;
        color: $warning;
        text-align: center;
        margin-bottom: 1;
    }

    AutoModeNoticeScreen .auto-mode-notice-body {
        height: auto;
        color: $text;
        margin-bottom: 1;
        padding: 0;
    }

    AutoModeNoticeScreen .auto-mode-notice-body > * {
        margin: 0 0 1 0;
    }

    AutoModeNoticeScreen .auto-mode-notice-body > *:last-child {
        margin-bottom: 0;
    }

    AutoModeNoticeScreen .auto-mode-notice-help {
        height: 1;
        color: $text-muted;
        text-style: italic;
        text-align: center;
        margin-top: 1;
    }
    """

    # The screen must be the focus target for its own priority Enter/Esc
    # bindings to fire (see `on_mount`); without this the keys reach no handler.
    can_focus = True

    def __init__(
        self,
        body: str | None = None,
        *,
        model_label: str | None = None,
        distinct_from_main_model: bool | None = None,
    ) -> None:
        """Initialize the notice.

        Args:
            body: Optional Markdown body under the title. Defaults to
                `AUTO_MODE_NOTICE_BODY`. Links open in a browser.
            model_label: Spec of the model that reviews gated actions, added to
                the default body. Ignored when `body` is supplied.
            distinct_from_main_model: Whether the reviewing model differs from
                the model writing code. Required alongside `model_label`;
                omitting both leaves the body's generic anchor in place.
        """
        super().__init__()
        if body is not None:
            self._body = body
        elif distinct_from_main_model is not None:
            self._body = build_auto_mode_notice_body(
                model_label, distinct_from_main_model=distinct_from_main_model
            )
        else:
            self._body = AUTO_MODE_NOTICE_BODY

    def on_mount(self) -> None:
        """Take focus so priority bindings receive Enter/Esc."""
        self.focus()

    def compose(self) -> ComposeResult:
        """Compose the Auto first-enable notice.

        Yields:
            Title, body, and help-row widgets parented inside a `Vertical`.
        """
        with Vertical():
            yield Static(
                "Auto mode",
                classes="auto-mode-notice-title",
                markup=False,
            )
            # open_links=False so we own the click path (toast feedback + shared
            # URL safety). Assistant message widgets use the same pattern.
            yield Markdown(
                self._body,
                classes="auto-mode-notice-body",
                open_links=False,
            )
            yield Static(
                f"Enter switch to Auto {get_glyphs().separator} Esc cancel",
                classes="auto-mode-notice-help",
                markup=False,
            )

    async def on_markdown_link_clicked(self, event: Markdown.LinkClicked) -> None:
        """Open docs (or any body link) with the shared URL helper."""
        event.stop()
        await open_checked_url_async(event.href, app=self.app, notify_on_success=True)

    def action_confirm(self) -> None:
        """Confirm switch to Auto and mark the notice dismissed without re-showing."""
        self.dismiss(True)

    def action_cancel(self) -> None:
        """Cancel the Auto switch without persisting the notice.

        The method name must stay `cancel`: the app owns a priority `escape`
        binding that, for an active `ModalScreen`, dispatches to
        `action_cancel` if present and otherwise falls through to
        `dismiss(None)`. Renaming this would silently regress Esc handling.
        """
        self.dismiss(False)
