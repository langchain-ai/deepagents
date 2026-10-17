"""Layout guards for the shared modal navigation footer.

`modal_navigation_hint` is long enough to wrap once a modal narrows. Wrapping
is the intended behavior -- the alternative is truncating the trailing hints --
but the extra row has to come from somewhere, and there are three distinct ways
for it not to. Each assertion below covers one, because passing any two of them
still leaves a footer the user cannot act on:

1. The row is laid out past the container's content region and never painted.
2. The row is painted *over* a sibling -- `dock: bottom` does not reserve space
   inside a `height: auto` container, because the docked child is excluded from
   the parent's auto-height, so it lands on top of the last siblings.
3. The container itself outgrows the viewport, carrying an in-container footer
   off-screen with it.

Containment inside the container is therefore necessary but not sufficient; the
overlap and viewport checks are what make this test able to fail.

`(60, 20)` and `(50, 20)` are sizes where the hint wraps. They mirror the guard
at `tui/widgets/test_mcp_viewer.py::test_footer_hints_stay_on_screen`.

`ColdCacheWarningScreen` and `LaunchGoalCriteriaPreferenceScreen` are absent on
purpose: their body text grows as the window narrows (the cold-cache body alone
wants seven rows at any width, and ten at 50 columns), so there is no room for a
second hint row without hiding cost or policy text the user needs more than the
hint. Both keep a single-row footer that clips sideways instead, which is what
they did before the shared hint landed. Capping their body height was tried and
rejected: `max-height` applies at every window size, so it hid two rows of the
warning even on a large terminal.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from textual.app import App
from textual.containers import Vertical, VerticalGroup
from textual.geometry import Region
from textual.widgets import Static

if TYPE_CHECKING:
    from collections.abc import Callable

    from textual.screen import ModalScreen
    from textual.widget import Widget

from deepagents_code.notifications import (
    ActionId,
    MissingDepPayload,
    NotificationAction,
    PendingNotification,
    UpdateAvailablePayload,
)
from deepagents_code.tui.widgets.agent_selector import AgentSelectorScreen
from deepagents_code.tui.widgets.effort_selector import EffortSelectorScreen
from deepagents_code.tui.widgets.notification_center import NotificationCenterScreen
from deepagents_code.tui.widgets.notification_detail import NotificationDetailScreen
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


# (id, screen factory, footer selector)
FOOTER_CASES: list[tuple[str, Callable[[], ModalScreen], str]] = [
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
        "update_available",
        lambda: UpdateAvailableScreen(_update_entry()),
        ".ua-help",
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


def _overlaps(a: Region, b: Region) -> bool:
    """Report whether two regions share at least one cell."""
    return not (a.right <= b.x or b.right <= a.x or a.bottom <= b.y or b.bottom <= a.y)


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
    """Every hint row is painted, in the modal, and over nothing else."""
    screen = factory()
    app: App[None] = App()
    async with app.run_test(size=size) as pilot:
        app.push_screen(screen)
        await pilot.pause()
        await pilot.pause()

        footer = screen.query_one(selector, Static)
        container = _modal_container(screen)
        viewport = Region(0, 0, *size)

        # 1. Painted at all.
        assert footer in app.screen._compositor.visible_widgets, (
            f"footer {footer.region} is not painted at {size}"
        )

        # 2. Inside the modal.
        assert container.content_region.contains_region(footer.region), (
            f"footer {footer.region} escapes {container.content_region} at {size}"
        )

        # 3. Not on top of a sibling. A docked footer inside a `height: auto`
        #    container is excluded from the parent's auto-height, so it lands
        #    over the last children instead of pushing them up.
        collisions = [
            " ".join(sibling.classes) or type(sibling).__name__
            for sibling in container.children
            if sibling is not footer and _overlaps(footer.region, sibling.region)
        ]
        assert not collisions, (
            f"footer {footer.region} paints over {collisions} at {size}"
        )

        # 4. The container cannot carry an in-container footer off-screen.
        assert viewport.contains_region(container.region), (
            f"modal {container.region} escapes the {size} viewport"
        )
