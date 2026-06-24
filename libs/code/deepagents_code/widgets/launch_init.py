"""Onboarding screens for the interactive TUI."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar

from textual.app import ScreenStackError
from textual.binding import Binding, BindingType
from textual.containers import Vertical, VerticalScroll
from textual.content import Content
from textual.screen import ModalScreen
from textual.widgets import Input, Static

if TYPE_CHECKING:
    from textual.app import ComposeResult
    from textual.screen import Screen

    from deepagents_code.extras_info import ExtraDependencyStatus

from deepagents_code import theme
from deepagents_code.config import get_glyphs, is_ascii_mode
from deepagents_code.extras_info import (
    MODEL_PROVIDER_EXTRAS,
    SANDBOX_EXTRAS,
    STANDALONE_EXTRAS,
)

logger = logging.getLogger(__name__)


def _normalize_name(value: str) -> str:
    """Normalize submitted onboarding names for display.

    Args:
        value: Raw submitted name.

    Returns:
        The stripped name, title-cased when it was entered in lowercase.
    """
    name = value.strip()
    if name.islower():
        return name.title()
    return name


class LaunchNameScreen(ModalScreen[str | None]):
    """First-step onboarding screen that asks for the user's name.

    Dismissal values:

    - Non-empty stripped/title-cased name when the user submits one.
    - `""` when the user submits an empty input (continue, but skip name memory).
    - `None` when the user dismisses with Escape (skip remaining onboarding).
    """

    AUTO_FOCUS = "#launch-name-input"

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "skip", "Skip", show=False, priority=True),
    ]

    CSS = """
    LaunchNameScreen {
        align: center middle;
    }

    LaunchNameScreen > Vertical {
        width: 64;
        max-width: 90%;
        height: auto;
        background: $surface;
        border: solid $primary;
        padding: 1 2;
    }

    LaunchNameScreen .launch-init-title {
        text-style: bold;
        color: $primary;
        text-align: center;
        margin-bottom: 1;
    }

    LaunchNameScreen .launch-init-copy {
        height: auto;
        color: $text;
        margin-bottom: 1;
    }

    LaunchNameScreen #launch-name-input {
        margin-bottom: 1;
        border: solid $primary-lighten-2;
    }

    LaunchNameScreen #launch-name-input:focus {
        border: solid $primary;
    }

    LaunchNameScreen .launch-init-help {
        height: 1;
        color: $text-muted;
        text-style: italic;
        text-align: center;
    }
    """

    def compose(self) -> ComposeResult:  # noqa: PLR6301  # Textual override
        """Compose the name-entry screen.

        Yields:
            Widgets for the modal content.
        """
        glyphs = get_glyphs()
        with Vertical():
            yield Static("Welcome to Deep Agents", classes="launch-init-title")
            yield Static(
                Content.assemble("What should Deep Agents call you?"),
                classes="launch-init-copy",
            )
            yield Input(
                placeholder="Your name (optional)",
                id="launch-name-input",
            )
            yield Static(
                f"Enter to continue {glyphs.bullet} Esc skip setup",
                classes="launch-init-help",
            )

    def on_mount(self) -> None:
        """Apply ASCII border when needed."""
        if is_ascii_mode():
            container = self.query_one(Vertical)
            colors = theme.get_theme_colors(self)
            container.styles.border = ("ascii", colors.success)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Dismiss with the submitted name.

        Args:
            event: The input submission event.
        """
        event.stop()
        value = _normalize_name(event.value)
        self.dismiss(value)

    def action_skip(self) -> None:
        """Skip the onboarding sequence."""
        self.dismiss(None)

    def action_cancel(self) -> None:
        """Alias for `action_skip` invoked by the global Esc binding.

        Textual's `Screen.action_cancel` is the conventional cancel hook used
        by the app-level Esc handler in `DeepAgentsApp`; routing it to
        `action_skip` keeps the screen-specific binding and the global path
        in sync.
        """
        self.action_skip()


class LaunchDependenciesScreen(ModalScreen[bool | None]):
    """Onboarding screen that summarizes installed optional integrations."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("enter", "continue", "Continue", show=False, priority=True),
        Binding("escape", "skip", "Skip", show=False, priority=True),
    ]

    CSS = """
    LaunchDependenciesScreen {
        align: center middle;
    }

    LaunchDependenciesScreen > Vertical {
        width: 76;
        max-width: 90%;
        height: auto;
        background: $surface;
        border: solid $primary;
        padding: 1 2;
    }

    LaunchDependenciesScreen .launch-init-title {
        text-style: bold;
        color: $primary;
        text-align: center;
        margin-bottom: 1;
    }

    LaunchDependenciesScreen .launch-init-copy {
        height: auto;
        color: $text-muted;
        margin-bottom: 1;
    }

    LaunchDependenciesScreen #launch-dependencies-body {
        height: auto;
        max-height: 16;
        margin-bottom: 1;
    }

    LaunchDependenciesScreen .launch-dependencies-section {
        height: auto;
        color: $text;
    }

    LaunchDependenciesScreen .launch-dependencies-section.is-available {
        margin-top: 1;
    }

    LaunchDependenciesScreen .launch-init-help {
        height: 1;
        color: $text-muted;
        text-style: italic;
        text-align: center;
    }
    """

    def __init__(
        self,
        statuses: tuple[ExtraDependencyStatus, ...] | None = None,
        *,
        continue_screen: Screen[Any] | None = None,
    ) -> None:
        """Initialize the dependency summary screen.

        Args:
            statuses: Optional dependency statuses to display. When omitted,
                the status is read from the installed package metadata.
            continue_screen: Optional screen to switch to when the user
                continues, avoiding an intermediate base-screen frame.
        """
        super().__init__()
        if statuses is None:
            from deepagents_code.extras_info import get_optional_dependency_status

            statuses = get_optional_dependency_status()
        self._statuses = statuses
        self._continue_screen = continue_screen

    def compose(self) -> ComposeResult:
        """Compose the dependency summary screen.

        Yields:
            Widgets for the modal content.
        """
        glyphs = get_glyphs()
        with Vertical():
            yield Static("Installed Integrations", classes="launch-init-title")
            yield Static(
                "Model providers and sandboxes are enabled by optional add-on "
                "packages. The ones already present in your environment are "
                "ready to use now.",
                classes="launch-init-copy",
            )
            if self._statuses:
                with VerticalScroll(id="launch-dependencies-body"):
                    yield Static(
                        self._format_section(
                            title="Ready now",
                            ready=True,
                            glyph=glyphs.checkmark,
                            empty="Nothing installed yet — add one below.",
                        ),
                        classes="launch-dependencies-section",
                    )
                    yield Static(
                        self._format_section(
                            title="Available to add",
                            ready=False,
                            glyph=glyphs.circle_empty,
                            empty="Everything is installed.",
                        ),
                        classes="launch-dependencies-section is-available",
                    )
                yield Static(
                    "Pick a model on the next screen to install its provider "
                    "automatically, or add any integration anytime with "
                    "`/install <name>` (e.g. `/install daytona`).",
                    classes="launch-init-copy",
                )
            else:
                # `get_optional_dependency_status` returns an empty tuple when
                # `importlib.metadata` cannot find the distribution (editable
                # install renamed, dev checkout without dist-info). Render a
                # single explanatory line instead of "none detected" twice.
                yield Static(
                    "Could not read installed dependency metadata. Reinstall "
                    "with `/install <extra>` to populate.",
                    classes="launch-dependencies-section",
                )
            yield Static(
                f"Enter to continue {glyphs.bullet} Esc skip setup",
                classes="launch-init-help",
            )

    def on_mount(self) -> None:
        """Apply ASCII border when needed."""
        if is_ascii_mode():
            container = self.query_one(Vertical)
            colors = theme.get_theme_colors(self)
            container.styles.border = ("ascii", colors.success)

    def _format_section(
        self, *, title: str, ready: bool, glyph: str, empty: str
    ) -> str:
        """Format one status section as per-extra rows grouped by category.

        Every matching extra is listed (no truncation); each category that
        has matches is shown under a sub-header, and the section title carries
        a total count.

        Args:
            title: Section title.
            ready: Whether to include ready or not-yet-ready extras.
            glyph: Status glyph rendered before each extra name.
            empty: Placeholder line shown when the section has no extras.

        Returns:
            Multi-line section text.
        """
        groups: tuple[tuple[str, frozenset[str]], ...] = (
            ("Model providers", MODEL_PROVIDER_EXTRAS),
            ("Sandboxes", SANDBOX_EXTRAS),
            ("Other", STANDALONE_EXTRAS),
        )
        grouped = [
            (label, self._extra_names(names, ready=ready)) for label, names in groups
        ]
        total = sum(len(extras) for _, extras in grouped)
        lines = [f"{title} ({total})"]
        if total == 0:
            lines.append(f"  {empty}")
            return "\n".join(lines)
        for label, extras in grouped:
            if not extras:
                continue
            lines.append(f"  {label}")
            lines.extend(f"    {glyph} {name}" for name in extras)
        return "\n".join(lines)

    def _extra_names(self, names: frozenset[str], *, ready: bool) -> list[str]:
        """Return sorted extra names matching a category and readiness state.

        Args:
            names: Category names to include.
            ready: Desired readiness state.

        Returns:
            Sorted matching extra names.
        """
        return sorted(
            status.name
            for status in self._statuses
            if status.name in names and status.ready is ready
        )

    def action_continue(self) -> None:
        """Continue onboarding."""
        if self._continue_screen is not None:
            try:
                self.app.switch_screen(self._continue_screen)
            except ScreenStackError:
                # Stack was torn down (app exiting, screen popped under us).
                # Fall back to dismissal so the launch-init task can finish
                # rather than leaving the user staring at this modal.
                logger.warning(
                    "Could not switch to continue screen; dismissing instead",
                    exc_info=True,
                )
                self.app.notify(
                    "Could not open the model selector. Use /model to pick "
                    "one when you're ready.",
                    severity="warning",
                    markup=False,
                )
                self.dismiss(True)
            return
        self.dismiss(True)

    def action_skip(self) -> None:
        """Skip the remaining onboarding sequence."""
        self.dismiss(None)

    def action_cancel(self) -> None:
        """See `LaunchNameScreen.action_cancel`."""
        self.action_skip()
