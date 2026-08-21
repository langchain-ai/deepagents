"""Layout guards for the shared modal navigation footer.

`modal_navigation_hint` is long enough to wrap once a modal narrows. Wrapping
is the intended behavior -- the alternative is truncating the trailing hints --
but a footer that grows inside a `height: auto` container capped by
`max-height` gets laid out past the modal's bottom edge, where the compositor
never paints it. These tests assert the footer's *region* stays inside the
modal, which is the only thing that distinguishes "wrapped and visible" from
"wrapped and silently clipped"; asserting the string is set cannot see it.

`(50, 20)` and `(60, 20)` are sizes where the hint wraps. They mirror the
existing guard at `tui/widgets/test_mcp_viewer.py::test_footer_hints_stay_on_screen`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from textual.app import App
from textual.containers import Vertical, VerticalGroup
from textual.widgets import Static

if TYPE_CHECKING:
    from collections.abc import Callable

    from textual.screen import ModalScreen
    from textual.widget import Widget

from deepagents_code.cold_cache import (
    ColdCacheWarning,
    PromptCachePolicy,
    RewarmEstimate,
)
from deepagents_code.notifications import (
    ActionId,
    MissingDepPayload,
    NotificationAction,
    PendingNotification,
    UpdateAvailablePayload,
)
from deepagents_code.tui.modals.cold_cache import ColdCacheWarningScreen
from deepagents_code.tui.widgets.agent_selector import AgentSelectorScreen
from deepagents_code.tui.widgets.effort_selector import EffortSelectorScreen
from deepagents_code.tui.widgets.launch_init import (
    LaunchGoalCriteriaPreferenceScreen,
)
from deepagents_code.tui.widgets.notification_center import NotificationCenterScreen
from deepagents_code.tui.widgets.notification_detail import NotificationDetailScreen
from deepagents_code.tui.widgets.notification_settings import (
    NotificationSettingsScreen,
)
from deepagents_code.tui.widgets.theme_selector import ThemeSelectorScreen
from deepagents_code.tui.widgets.update_available import UpdateAvailableScreen

FOOTER_SIZES = [(80, 24), (60, 20), (50, 20)]


def _update_entry() -> PendingNotification:
    return PendingNotification(
        key="update:available",
        title="Update available",
        body="v2.0.0 is available.\nCurrently installed: 1.0.0.",
        actions=(
            NotificationAction(ActionId.INSTALL, "Install now", primary=True),
            NotificationAction(ActionId.SKIP_ONCE, "Remind me next launch"),
            NotificationAction(ActionId.SKIP_VERSION, "Skip this version"),
        ),
        payload=UpdateAvailablePayload(
            latest="2.0.0", upgrade_cmd="uv tool upgrade deepagents-code"
        ),
    )


def _dep_entry() -> PendingNotification:
    return PendingNotification(
        key="dep:ripgrep",
        title="ripgrep is not installed",
        body="Install with: brew install ripgrep",
        actions=(
            NotificationAction(
                ActionId.COPY_INSTALL, "Copy install command", primary=True
            ),
            NotificationAction(ActionId.SUPPRESS, "Don't show notification again"),
        ),
        payload=MissingDepPayload(
            tool="ripgrep", install_command="brew install ripgrep"
        ),
    )


def _cold_cache_warning() -> ColdCacheWarning:
    return ColdCacheWarning(
        policy=PromptCachePolicy(
            provider_name="OpenAI",
            window_seconds=1800,
            confidence="may_be_cold",
            minimum_tokens=1024,
            write_bucket="generic",
        ),
        estimate=RewarmEstimate(cold_cost_usd=0.35, incremental_cost_usd=0.25),
        context_tokens=50_000,
        age_seconds=3600,
        reason="idle",
    )


# (id, screen factory, footer selector)
FOOTER_CASES: list[tuple[str, Callable[[], ModalScreen], str]] = [
    (
        "cold_cache",
        lambda: ColdCacheWarningScreen(_cold_cache_warning()),
        ".cold-cache-help",
    ),
    (
        "notification_center",
        lambda: NotificationCenterScreen([_dep_entry(), _update_entry()]),
        ".nc-help",
    ),
    (
        "notification_detail",
        lambda: NotificationDetailScreen(_update_entry()),
        ".nd-help",
    ),
    (
        "notification_settings",
        lambda: NotificationSettingsScreen(suppressed=set()),
        ".ns-help",
    ),
    (
        "update_available",
        lambda: UpdateAvailableScreen(_update_entry()),
        ".ua-help",
    ),
    (
        "launch_goal_preference",
        LaunchGoalCriteriaPreferenceScreen,
        ".launch-init-help",
    ),
    (
        "theme_selector",
        lambda: ThemeSelectorScreen(current_theme="langchain"),
        ".theme-selector-help",
    ),
    (
        "agent_selector",
        lambda: AgentSelectorScreen(
            current_agent="general",
            agent_names=["general", "research"],
            default_agent=None,
        ),
        ".agent-selector-help",
    ),
    (
        "effort_selector",
        lambda: EffortSelectorScreen(
            model_spec="anthropic:claude-sonnet-4-5",
            efforts=("low", "medium", "high"),
        ),
        ".effort-selector-help",
    ),
]


def _modal_container(screen: ModalScreen) -> Widget:
    """Return the modal's outer container, whichever vertical type it uses."""
    for node in screen.query(Vertical):
        return node
    return screen.query_one(VerticalGroup)


@pytest.mark.parametrize("size", FOOTER_SIZES, ids=lambda s: f"{s[0]}x{s[1]}")
@pytest.mark.parametrize(
    ("factory", "selector"),
    [(factory, selector) for _, factory, selector in FOOTER_CASES],
    ids=[case_id for case_id, _, _ in FOOTER_CASES],
)
async def test_navigation_footer_stays_inside_the_modal(
    size: tuple[int, int],
    factory: Callable[[], ModalScreen],
    selector: str,
) -> None:
    """Every hint row renders inside the modal, however narrow the window.

    The bug guarded here is a complete footer string laid out below the
    modal's bottom edge: `height: auto` lets the hint wrap, and without
    `dock: bottom` the wrapped row is pushed out of the container's content
    region and never painted.
    """
    screen = factory()
    app: App[None] = App()
    async with app.run_test(size=size) as pilot:
        app.push_screen(screen)
        await pilot.pause()
        await pilot.pause()

        footer = screen.query_one(selector, Static)
        container = _modal_container(screen)

        assert footer in app.screen._compositor.visible_widgets
        assert container.content_region.contains_region(footer.region), (
            f"footer {footer.region} escapes {container.content_region} at {size}"
        )
        assert container.region.y >= 0
