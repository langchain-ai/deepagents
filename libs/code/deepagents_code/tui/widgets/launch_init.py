"""Onboarding screens for the interactive TUI."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar

from textual.app import ScreenStackError
from textual.binding import Binding, BindingType
from textual.containers import Vertical, VerticalScroll
from textual.content import Content
from textual.css.query import NoMatches
from textual.screen import ModalScreen
from textual.widgets import Input, Static

if TYPE_CHECKING:
    from collections.abc import Callable

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

_DEPENDENCY_BODY_MAX_HEIGHT = 16
"""Upper bound (in cells) for the scrollable dependency list.

Keep in sync with the `max-height: 16` in the `#launch-dependencies-body` CSS;
Textual CSS cannot reference Python constants, so the static cap and the
runtime `_fit_dependencies_body` clamp must agree.
"""
_DEPENDENCY_BODY_MIN_HEIGHT = 1
"""Floor (in cells) so the list never collapses to zero on tiny terminals."""


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

    def __init__(
        self,
        *,
        continue_screen: Screen[Any] | None = None,
        on_continue: Callable[[str], None] | None = None,
        on_continue_failed: Callable[[str], None] | None = None,
    ) -> None:
        """Initialize the name-entry screen.

        Args:
            continue_screen: Optional screen to switch to after submitting a name.
            on_continue: Optional callback invoked with the submitted name before
                switching to `continue_screen`.
            on_continue_failed: Optional callback invoked with the submitted
                name when switching to `continue_screen` fails.
        """
        super().__init__()
        self._continue_screen = continue_screen
        self._on_continue = on_continue
        self._on_continue_failed = on_continue_failed

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
        with Vertical():
            yield Static("Welcome to dcode", classes="launch-init-title")
            yield Static(
                Content.assemble("What should Deep Agents call you?"),
                classes="launch-init-copy",
            )
            yield Input(
                placeholder="Your name (optional)",
                id="launch-name-input",
            )
            yield Static(
                "Enter to continue",
                classes="launch-init-help",
            )

    def on_mount(self) -> None:
        """Apply ASCII border when needed."""
        if is_ascii_mode():
            container = self.query_one(Vertical)
            colors = theme.get_theme_colors(self)
            container.styles.border = ("ascii", colors.success)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Continue with the submitted name.

        Args:
            event: The input submission event.
        """
        event.stop()
        value = _normalize_name(event.value)
        if self._continue_screen is None:
            self.dismiss(value)
            return
        if self._on_continue is not None:
            self._on_continue(value)
        try:
            self.app.switch_screen(self._continue_screen)
        except ScreenStackError:
            logger.warning(
                "Could not switch from launch name screen; dismissing instead",
                exc_info=True,
            )
            if self._on_continue_failed is not None:
                self._on_continue_failed(value)
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
        max-height: 16;  /* keep in sync with `_DEPENDENCY_BODY_MAX_HEIGHT` */
        scrollbar-gutter: stable;
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
        on_done: Callable[[bool | None], None] | None = None,
    ) -> None:
        """Initialize the dependency summary screen.

        Args:
            statuses: Optional dependency statuses to display. When omitted,
                the status is read from the installed package metadata.
            continue_screen: Optional screen to switch to when the user
                continues, avoiding an intermediate base-screen frame.
            on_done: Optional callback invoked when this screen finishes without
                switching to `continue_screen`.
        """
        super().__init__()
        if statuses is None:
            from deepagents_code.extras_info import get_optional_dependency_status

            statuses = get_optional_dependency_status()
        self._statuses = statuses
        self._continue_screen = continue_screen
        self._on_done = on_done

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
                            empty="All bundled integrations are installed.",
                        ),
                        classes="launch-dependencies-section is-available",
                    )
                yield Static(
                    "Pick a model on the next screen and its provider installs "
                    "automatically. Add more anytime with `/install`.",
                    classes="launch-init-copy",
                )
            else:
                # `get_optional_dependency_status` returns an empty tuple when
                # `importlib.metadata` cannot find the distribution (editable
                # install renamed, dev checkout without dist-info). Render a
                # single explanatory line rather than empty status sections.
                yield Static(
                    "Could not read installed dependency metadata. Reinstall "
                    "with `/install <extra>` to populate.",
                    classes="launch-dependencies-section",
                )
            yield Static(
                "Enter to continue",
                classes="launch-init-help",
            )

    def on_mount(self) -> None:
        """Apply ASCII border when needed."""
        if is_ascii_mode():
            container = self.query_one(Vertical)
            colors = theme.get_theme_colors(self)
            container.styles.border = ("ascii", colors.success)
        self.call_after_refresh(self._fit_dependencies_body)

    def on_resize(self) -> None:
        """Refit the scroll body when terminal dimensions change."""
        self.call_after_refresh(self._fit_dependencies_body)

    def _fit_dependencies_body(self) -> None:
        """Cap the dependency list height so modal controls stay in view."""
        # `#launch-dependencies-body` is only composed when statuses are
        # non-empty (see `compose`); skip the structural always-empty case
        # here. The `NoMatches` catch below still handles the teardown race.
        if not self._statuses:
            return

        try:
            container = self.query_one(Vertical)
            body = self.query_one("#launch-dependencies-body", VerticalScroll)
        except NoMatches:
            # This runs deferred via `call_after_refresh`; the screen may have
            # been popped or recomposed before it fires (e.g. a resize racing
            # dismissal). Sizing is cosmetic, so skip quietly but leave a
            # breadcrumb rather than letting it surface in the event loop.
            logger.debug(
                "Skipping dependency-body refit; widgets not mounted",
                exc_info=True,
            )
            return
        non_body_height = max(0, container.region.height - body.region.height)
        available_height = self.size.height - non_body_height
        max_height = max(
            _DEPENDENCY_BODY_MIN_HEIGHT,
            min(_DEPENDENCY_BODY_MAX_HEIGHT, available_height),
        )
        current = body.styles.max_height
        if current is not None and current.cells == max_height:
            return
        body.styles.max_height = max_height

    def _format_section(
        self, *, title: str, ready: bool, glyph: str, empty: str
    ) -> str:
        """Format one status section as per-extra rows grouped by category.

        Every matching extra is listed (no truncation); each category that
        has matches is shown under a sub-header, and the section title carries
        a total count. When nothing matches, the `empty` placeholder is shown
        in place of the sub-headers.

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
                self._finish(True)
            return
        self._finish(True)

    def action_skip(self) -> None:
        """Skip the remaining onboarding sequence."""
        self._finish(None)

    def _finish(self, result: bool | None) -> None:
        """Resolve the screen-specific callback before dismissing."""
        if self._on_done is not None:
            self._on_done(result)
        self.dismiss(result)

    def action_cancel(self) -> None:
        """See `LaunchNameScreen.action_cancel`."""
        self.action_skip()
