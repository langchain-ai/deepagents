"""Tests for the plugin manager modal structure."""

import asyncio
import inspect
import re
from pathlib import Path
from typing import get_args
from unittest.mock import AsyncMock, MagicMock

import pytest
from textual.containers import Vertical
from textual.widgets import Input, OptionList, Static

from deepagents_code.app import DeepAgentsApp
from deepagents_code.config import get_glyphs
from deepagents_code.plugins.models import PluginMarketplace
from deepagents_code.tui.modals.plugin_manager import PluginManagerScreen
from deepagents_code.tui.modals.plugin_manager.content import (
    _installed_details_options,
    _installed_plugin_details_content,
    _load_state_label,
    _marketplace_label,
    _plugin_details_content,
    _plugin_options,
    _plugin_prompt,
    _status_lines,
)
from deepagents_code.tui.modals.plugin_manager.models import (
    PluginTab,
    _ManagerState,
    _MarketplaceRow,
    _PluginRow,
)
from deepagents_code.tui.modals.plugin_manager.tabs import TAB_LABELS


def test_plugin_manager_css_is_colocated_with_screen() -> None:
    screen_file = Path(inspect.getfile(PluginManagerScreen))
    css_path = screen_file.parent / PluginManagerScreen.CSS_PATH
    assert PluginManagerScreen.CSS_PATH == "plugin_manager.tcss"
    assert css_path.is_file(), f"expected colocated CSS at {css_path}"


def test_plugin_options_preserve_selectable_rows_and_spacers() -> None:
    rows = (
        _PluginRow(
            plugin_id="first@source",
            description="First",
            enabled=False,
            version=None,
            author=None,
            display_name="First",
        ),
        _PluginRow(
            plugin_id="second@source",
            description="Second",
            enabled=True,
            version="1.0",
            author="Author",
            display_name="Second",
        ),
    )

    options = _plugin_options(rows, action="detail", status=None)

    assert [option.id for option in options] == [
        "detail:first@source",
        "spacer:1",
        "detail:second@source",
    ]
    assert options[1].disabled


def test_installed_details_separate_back_action() -> None:
    row = _PluginRow(
        plugin_id="linear@tools",
        description="Linear plugin",
        enabled=True,
        version="1.0.0",
        author=None,
    )

    divider_width = 8
    options = _installed_details_options(row, divider_width=divider_width)

    assert [option.id for option in options] == [
        "action:toggle-enabled",
        "action:uninstall",
        "details-divider",
        "details-back",
    ]
    assert options[2].disabled
    assert str(options[2].prompt) == get_glyphs().box_horizontal * divider_width


async def test_plugin_manager_closes_without_mcp_reconnect() -> None:
    app = DeepAgentsApp(agent=MagicMock(), thread_id="t")
    screen = PluginManagerScreen()
    on_close = MagicMock()

    async with app.run_test(size=(120, 40)) as pilot:
        app.push_screen(screen, on_close)
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()

    on_close.assert_called_once_with(None)


async def test_plugin_search_filters_and_clears() -> None:
    app = DeepAgentsApp(agent=MagicMock(), thread_id="t")
    screen = PluginManagerScreen()
    state = _ManagerState(
        available_plugins=(
            _PluginRow(
                plugin_id="docs@official",
                description="Read/write documentation",
                enabled=False,
                version=None,
                author=None,
            ),
            _PluginRow(
                plugin_id="tests@official",
                description="Run the test suite",
                enabled=False,
                version=None,
                author=None,
            ),
        ),
        installed_plugins=(),
        marketplaces=(_MarketplaceRow("official", "owner/official", 2, 0),),
        errors=(),
    )

    async with app.run_test(size=(120, 40)) as pilot:
        app.push_screen(screen)
        await pilot.pause()
        screen._state = state
        screen._refresh_view()
        search = screen.query_one("#plugin-manager-search", Input)
        options = screen.query_one("#plugin-manager-options", OptionList)

        await pilot.press("/")
        assert search.has_focus
        await pilot.press("r", "e", "a", "d", "/", "w", "r", "i", "t", "e")
        assert search.value == "read/write"
        assert options.option_count == 1
        assert options.get_option_at_index(0).id == "detail:docs@official"

        await pilot.press("ctrl+a", "x")
        assert options.option_count == 1
        assert options.get_option_at_index(0).prompt == "No plugins match your search."
        assert options.get_option_at_index(0).disabled
        assert options.highlighted is None

        await pilot.press("escape")
        assert search.value == ""
        assert search.has_focus
        assert options.option_count == 3
        await pilot.press("escape")
        assert options.has_focus


async def test_plugin_search_and_footer_fit_standard_terminal() -> None:
    """Search must not push the plugin list or footer outside an 80x24 modal."""
    app = DeepAgentsApp(agent=MagicMock(), thread_id="t")
    screen = PluginManagerScreen()
    state = _ManagerState(
        available_plugins=(
            _PluginRow(
                plugin_id="docs@official",
                description="Search documentation",
                enabled=False,
                version=None,
                author=None,
            ),
        ),
        installed_plugins=(),
        marketplaces=(_MarketplaceRow("official", "owner/official", 1, 0),),
        errors=(),
    )

    async with app.run_test(size=(80, 24)) as pilot:
        app.push_screen(screen)
        await pilot.pause()
        screen._state = state
        screen._refresh_view()
        await pilot.pause()

        container = screen.query_one(Vertical)
        search = screen.query_one("#plugin-manager-search", Input)
        options = screen.query_one("#plugin-manager-options", OptionList)
        help_text = screen.query_one("#plugin-manager-help", Static)

        assert container.region.y >= 0
        assert container.region.bottom <= app.size.height
        assert search.display is True
        assert options.region.height >= 5
        assert options.region.bottom <= container.content_region.bottom
        assert help_text.region.height >= 1
        assert help_text.region.bottom <= container.content_region.bottom


async def test_search_arrows_move_cursor_only_when_nonempty() -> None:
    """Left/right move the caret only once the search box has text."""
    app = DeepAgentsApp(agent=MagicMock(), thread_id="t")
    screen = PluginManagerScreen()
    state = _ManagerState(
        available_plugins=(
            _PluginRow(
                plugin_id="docs@official",
                description="Search documentation",
                enabled=False,
                version=None,
                author=None,
            ),
        ),
        installed_plugins=(
            _PluginRow(
                plugin_id="tests@official",
                description="Run the test suite",
                enabled=True,
                version="1.0.0",
                author=None,
            ),
        ),
        marketplaces=(_MarketplaceRow("official", "owner/official", 2, 1),),
        errors=(),
    )

    async with app.run_test(size=(120, 40)) as pilot:
        app.push_screen(screen)
        await pilot.pause()
        screen._state = state
        screen._refresh_view()
        search = screen.query_one("#plugin-manager-search", Input)
        options = screen.query_one("#plugin-manager-options", OptionList)

        await pilot.press("/")
        assert search.has_focus
        assert search.value == ""
        await pilot.press("right")
        assert screen._tab == "installed"
        assert search.display is True

        screen._select_tab("discover")
        assert search.has_focus
        await pilot.press("d", "o", "c", "s")
        assert search.value == "docs"
        assert search.cursor_position == 4
        assert screen._tab == "discover"

        await pilot.press("left")
        assert screen._tab == "discover"
        assert search.cursor_position == 3
        assert search.has_focus

        await pilot.press("right")
        assert screen._tab == "discover"
        assert search.cursor_position == 4

        await pilot.press("escape", "escape")
        assert options.has_focus
        await pilot.press("right")
        assert screen._tab == "installed"


async def test_plugin_tabs_are_mouse_clickable() -> None:
    """Clicking a tab label switches to that tab."""
    app = DeepAgentsApp(agent=MagicMock(), thread_id="t")
    screen = PluginManagerScreen()
    state = _ManagerState(
        available_plugins=(),
        installed_plugins=(
            _PluginRow(
                plugin_id="docs@official",
                description="Search documentation",
                enabled=True,
                version=None,
                author=None,
            ),
        ),
        marketplaces=(),
        errors=(),
    )

    async with app.run_test(size=(120, 40)) as pilot:
        app.push_screen(screen)
        await pilot.pause()
        screen._state = state
        screen._refresh_view()
        assert screen._tab == "discover"

        await pilot.click("#plugin-tab-installed")
        await pilot.pause()
        assert screen._tab == "installed"
        assert screen.query_one("#plugin-tab-installed", Static).has_class("active")
        assert not screen.query_one("#plugin-tab-discover", Static).has_class("active")


async def test_plugin_tabs_fit_and_are_clickable_in_narrow_terminal() -> None:
    """Every tab remains inside the modal when horizontal space is limited."""
    app = DeepAgentsApp(agent=MagicMock(), thread_id="t")
    screen = PluginManagerScreen()

    async with app.run_test(size=(50, 30)) as pilot:
        app.push_screen(screen)
        await pilot.pause()
        tabs = screen.query_one("#plugin-manager-tabs")
        tab_labels = list(tabs.query(".plugin-manager-tab"))

        assert tab_labels
        assert all(
            label.region.x >= tabs.region.x and label.region.right <= tabs.region.right
            for label in tab_labels
        )

        await pilot.click("#plugin-tab-marketplaces")
        await pilot.pause()
        assert screen._tab == "marketplaces"

        await pilot.click("#plugin-tab-errors")
        await pilot.pause()
        assert screen._tab == "errors"


async def test_installed_plugin_search_filters_and_handles_no_match() -> None:
    app = DeepAgentsApp(agent=MagicMock(), thread_id="t")
    screen = PluginManagerScreen()
    screen._tab = "installed"
    state = _ManagerState(
        available_plugins=(),
        installed_plugins=(
            _PluginRow(
                plugin_id="docs@official",
                description="Search documentation",
                enabled=True,
                version="1.0.0",
                author=None,
            ),
            _PluginRow(
                plugin_id="tests@official",
                description="Run the test suite",
                enabled=True,
                version="1.0.0",
                author=None,
            ),
        ),
        marketplaces=(_MarketplaceRow("official", "owner/official", 2, 2),),
        errors=(),
    )

    async with app.run_test(size=(120, 40)) as pilot:
        app.push_screen(screen)
        await pilot.pause()
        screen._state = state
        screen._refresh_view()
        search = screen.query_one("#plugin-manager-search", Input)
        options = screen.query_one("#plugin-manager-options", OptionList)

        await pilot.press("/")
        assert search.has_focus
        await pilot.press("d", "o", "c", "s")
        assert options.option_count == 1
        assert options.get_option_at_index(0).id == "installed:docs@official"

        await pilot.press("ctrl+a", "x")
        assert options.option_count == 1
        placeholder = options.get_option_at_index(0)
        assert placeholder.prompt == "No installed plugins match your search."
        assert placeholder.disabled
        assert options.highlighted is None

        await pilot.press("enter")
        assert screen._mode == "list"
        assert search.has_focus


async def test_enter_from_search_opens_highlighted_plugin_details() -> None:
    app = DeepAgentsApp(agent=MagicMock(), thread_id="t")
    screen = PluginManagerScreen()
    state = _ManagerState(
        available_plugins=(
            _PluginRow(
                plugin_id="docs@official",
                description="Search documentation",
                enabled=False,
                version=None,
                author=None,
            ),
        ),
        installed_plugins=(),
        marketplaces=(_MarketplaceRow("official", "owner/official", 1, 0),),
        errors=(),
    )

    async with app.run_test(size=(120, 40)) as pilot:
        app.push_screen(screen)
        await pilot.pause()
        screen._state = state
        screen._refresh_view()

        await pilot.press("/", "d", "o", "c", "s", "enter")
        await pilot.pause()

        assert screen._mode == "plugin_details"


async def test_enter_from_search_activates_installed_row() -> None:
    """Enter on a matched installed row opens its details, not just `detail:`."""
    app = DeepAgentsApp(agent=MagicMock(), thread_id="t")
    screen = PluginManagerScreen()
    screen._tab = "installed"
    state = _ManagerState(
        available_plugins=(),
        installed_plugins=(
            _PluginRow(
                plugin_id="docs@official",
                description="Search documentation",
                enabled=True,
                version="1.0.0",
                author=None,
            ),
        ),
        marketplaces=(_MarketplaceRow("official", "owner/official", 1, 1),),
        errors=(),
    )

    async with app.run_test(size=(120, 40)) as pilot:
        app.push_screen(screen)
        await pilot.pause()
        screen._state = state
        screen._refresh_view()

        await pilot.press("/", "d", "o", "c", "s", "enter")
        await pilot.pause()

        assert screen._mode == "installed_details"


async def test_search_query_resets_when_switching_tabs() -> None:
    """A query typed on one tab does not silently filter another tab."""
    app = DeepAgentsApp(agent=MagicMock(), thread_id="t")
    screen = PluginManagerScreen()
    state = _ManagerState(
        available_plugins=(
            _PluginRow(
                plugin_id="docs@official",
                description="Search documentation",
                enabled=False,
                version=None,
                author=None,
            ),
        ),
        installed_plugins=(
            _PluginRow(
                plugin_id="tests@official",
                description="Run the test suite",
                enabled=True,
                version="1.0.0",
                author=None,
            ),
        ),
        marketplaces=(_MarketplaceRow("official", "owner/official", 1, 1),),
        errors=(),
    )

    async with app.run_test(size=(120, 40)) as pilot:
        app.push_screen(screen)
        await pilot.pause()
        screen._state = state
        screen._refresh_view()
        search = screen.query_one("#plugin-manager-search", Input)
        options = screen.query_one("#plugin-manager-options", OptionList)

        await pilot.press("/", "d", "o", "c", "s")
        assert screen._search_query == "docs"

        # Click switches tabs regardless of focus; the installed list must not
        # inherit the "docs" query (which would hide "tests@official").
        await pilot.click("#plugin-tab-installed")
        await pilot.pause()
        assert screen._tab == "installed"
        assert screen._search_query == ""
        assert search.value == ""
        assert options.option_count == 1
        assert options.get_option_at_index(0).id == "installed:tests@official"


async def test_search_query_persists_through_details_roundtrip() -> None:
    """Opening a filtered row and returning keeps the query (uses _refresh_view)."""
    app = DeepAgentsApp(agent=MagicMock(), thread_id="t")
    screen = PluginManagerScreen()
    state = _ManagerState(
        available_plugins=(
            _PluginRow(
                plugin_id="docs@official",
                description="Search documentation",
                enabled=False,
                version=None,
                author=None,
            ),
            _PluginRow(
                plugin_id="tests@official",
                description="Run the test suite",
                enabled=False,
                version=None,
                author=None,
            ),
        ),
        installed_plugins=(),
        marketplaces=(_MarketplaceRow("official", "owner/official", 2, 0),),
        errors=(),
    )

    async with app.run_test(size=(120, 40)) as pilot:
        app.push_screen(screen)
        await pilot.pause()
        screen._state = state
        screen._refresh_view()
        options = screen.query_one("#plugin-manager-options", OptionList)

        await pilot.press("/", "d", "o", "c", "s", "enter")
        await pilot.pause()
        assert screen._mode == "plugin_details"

        await pilot.press("escape")
        await pilot.pause()
        assert screen._mode == "list"
        assert screen._search_query == "docs"
        assert options.option_count == 1
        assert options.get_option_at_index(0).id == "detail:docs@official"


async def test_tab_click_from_details_returns_to_list() -> None:
    """Clicking another tab while in a details view drops back to list mode."""
    app = DeepAgentsApp(agent=MagicMock(), thread_id="t")
    screen = PluginManagerScreen()
    state = _ManagerState(
        available_plugins=(
            _PluginRow(
                plugin_id="docs@official",
                description="Search documentation",
                enabled=False,
                version=None,
                author=None,
            ),
        ),
        installed_plugins=(),
        marketplaces=(_MarketplaceRow("official", "owner/official", 1, 0),),
        errors=(),
    )

    async with app.run_test(size=(120, 40)) as pilot:
        app.push_screen(screen)
        await pilot.pause()
        screen._state = state
        screen._refresh_view()

        await pilot.press("enter")
        await pilot.pause()
        assert screen._mode == "plugin_details"
        assert screen._selected_plugin is not None

        await pilot.click("#plugin-tab-marketplaces")
        await pilot.pause()
        assert screen._mode == "list"
        assert screen._tab == "marketplaces"
        assert screen._selected_plugin is None
        assert screen._selected_marketplace is None


def test_tab_labels_cover_every_plugin_tab() -> None:
    """TAB_LABELS must stay in sync with the PluginTab literal.

    `dict[PluginTab, str]` only constrains keys, not completeness, so a new tab
    could otherwise KeyError at compose time.
    """
    assert set(TAB_LABELS) == set(get_args(PluginTab))


def test_rendered_tabs_cover_every_plugin_tab() -> None:
    """`_tabs` must stay in sync with the PluginTab literal.

    The tab set is duplicated across PluginTab, TAB_LABELS, and `_tabs`. A tab
    missing from `_tabs` would silently never render (compose iterates `_tabs`),
    and one absent from TAB_LABELS would KeyError at compose time.
    """
    assert set(PluginManagerScreen._tabs) == set(get_args(PluginTab))


async def test_refresh_state_clears_search_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mutating reload drops the query so reloaded results are not filtered."""
    app = DeepAgentsApp(agent=MagicMock(), thread_id="t")
    screen = PluginManagerScreen()
    fresh = _ManagerState(
        available_plugins=(
            _PluginRow(
                plugin_id="docs@official",
                description="Read/write documentation",
                enabled=False,
                version=None,
                author=None,
            ),
            _PluginRow(
                plugin_id="tests@official",
                description="Run the test suite",
                enabled=False,
                version=None,
                author=None,
            ),
        ),
        installed_plugins=(),
        marketplaces=(_MarketplaceRow("official", "owner/official", 2, 0),),
        errors=(),
    )
    monkeypatch.setattr(
        "deepagents_code.tui.modals.plugin_manager._load_manager_state",
        lambda _info, **_kwargs: fresh,
    )

    async with app.run_test(size=(120, 40)) as pilot:
        app.push_screen(screen)
        await pilot.pause()
        options = screen.query_one("#plugin-manager-options", OptionList)
        search = screen.query_one("#plugin-manager-search", Input)

        await pilot.press("/", "d", "o", "c", "s")
        await pilot.pause()
        assert screen._search_query == "docs"
        assert options.option_count == 1

        await screen._refresh_state()
        await pilot.pause()

        assert screen._search_query == ""
        assert search.value == ""
        # The cleared query restores every plugin, not just the "docs" match.
        ids = {options.get_option_at_index(i).id for i in range(options.option_count)}
        assert {"detail:docs@official", "detail:tests@official"} <= ids


async def test_tab_switch_ignored_during_add_marketplace() -> None:
    """Switching tabs must not discard an in-progress marketplace source entry."""
    app = DeepAgentsApp(agent=MagicMock(), thread_id="t")
    screen = PluginManagerScreen()

    async with app.run_test(size=(120, 40)) as pilot:
        app.push_screen(screen)
        await pilot.pause()
        screen._mode = "add_marketplace"
        screen._refresh_view()
        await pilot.pause()

        screen._select_tab("installed")
        assert screen._mode == "add_marketplace"
        assert screen._tab == "discover"


async def test_marketplace_add_stays_locked_during_state_refresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A successful add cannot be submitted again while state is reloading."""
    app = DeepAgentsApp(agent=MagicMock(), thread_id="t")
    screen = PluginManagerScreen()
    refresh_started = asyncio.Event()
    release_refresh = asyncio.Event()

    async def refresh_state() -> None:
        refresh_started.set()
        await release_refresh.wait()

    async with app.run_test(size=(120, 40)) as pilot:
        app.push_screen(screen)
        await pilot.pause()
        screen._mode = "add_marketplace"
        screen._adding_marketplace = True
        screen._refresh_view()
        source = screen.query_one("#plugin-marketplace-source", Input)
        source.value = "owner/repo"
        source.disabled = True
        add_marketplace = MagicMock()
        monkeypatch.setattr(screen, "_add_marketplace", add_marketplace)
        monkeypatch.setattr(screen, "_refresh_state", refresh_state)
        marketplace = PluginMarketplace(
            name="official",
            root=tmp_path,
            manifest_path=tmp_path / "marketplace.json",
            metadata={},
            plugins=(),
        )

        finish = asyncio.create_task(screen._finish_marketplace_add(marketplace, None))
        await refresh_started.wait()

        assert screen._adding_marketplace is True
        assert source.disabled is True
        screen.on_input_submitted(Input.Submitted(source, source.value))
        add_marketplace.assert_not_called()

        release_refresh.set()
        await finish
        assert screen._adding_marketplace is False
        assert source.disabled is False


async def test_search_hidden_without_filterable_plugins() -> None:
    app = DeepAgentsApp(agent=MagicMock(), thread_id="t")
    screen = PluginManagerScreen()

    async with app.run_test(size=(120, 40)) as pilot:
        app.push_screen(screen)
        await pilot.pause()
        search = screen.query_one("#plugin-manager-search", Input)

        screen._state = _ManagerState((), (), (), ())
        screen._refresh_view()
        assert search.display is False

        screen._state = _ManagerState(
            (), (), (_MarketplaceRow("official", "owner/official", 0, 0),), ()
        )
        screen._refresh_view()
        assert search.display is False

        screen._tab = "installed"
        screen._refresh_view()
        assert search.display is False

        # Marketplaces and errors tabs never filter, even when populated.
        screen._state = _ManagerState(
            available_plugins=(),
            installed_plugins=(),
            marketplaces=(_MarketplaceRow("official", "owner/official", 2, 0),),
            errors=("boom",),
        )
        screen._tab = "marketplaces"
        screen._refresh_view()
        assert search.display is False
        assert screen.check_action("focus_search", ()) is False

        screen._tab = "errors"
        screen._refresh_view()
        assert search.display is False
        assert screen.check_action("focus_search", ()) is False


def test_focus_search_binding_enabled_only_when_filter_visible() -> None:
    """`check_action` enables `/` only when the search filter is shown."""
    screen = PluginManagerScreen()

    assert screen.check_action("focus_search", ()) is False

    screen._state = _ManagerState(
        available_plugins=(
            _PluginRow(
                plugin_id="docs@official",
                description="Search documentation",
                enabled=False,
                version=None,
                author=None,
            ),
        ),
        installed_plugins=(),
        marketplaces=(_MarketplaceRow("official", "owner/official", 1, 0),),
        errors=(),
    )
    assert screen.check_action("focus_search", ()) is True

    screen._tab = "marketplaces"
    assert screen.check_action("focus_search", ()) is False

    screen._tab = "discover"
    screen._mode = "add_marketplace"
    assert screen.check_action("focus_search", ()) is False

    screen._mode = "list"
    screen._state = _ManagerState((), (), (), ())
    assert screen.check_action("focus_search", ()) is False
    assert screen.check_action("cancel", ()) is True


async def test_slash_does_not_focus_hidden_search() -> None:
    """`/` must not steal focus when the filter is hidden (empty discover)."""
    app = DeepAgentsApp(agent=MagicMock(), thread_id="t")
    screen = PluginManagerScreen()

    async with app.run_test(size=(120, 40)) as pilot:
        app.push_screen(screen)
        await pilot.pause()
        screen._state = _ManagerState((), (), (), ())
        screen._refresh_view()
        search = screen.query_one("#plugin-manager-search", Input)
        options = screen.query_one("#plugin-manager-options", OptionList)
        assert search.display is False
        assert options.has_focus

        await pilot.press("/")
        assert options.has_focus
        assert not search.has_focus

        await pilot.press("enter")
        await pilot.pause()
        assert screen._mode == "add_marketplace"


async def test_slash_remains_typeable_in_add_marketplace_source() -> None:
    """`/` must reach the marketplace source field (owner/repo, urls, paths)."""
    app = DeepAgentsApp(agent=MagicMock(), thread_id="t")
    screen = PluginManagerScreen()

    async with app.run_test(size=(120, 40)) as pilot:
        app.push_screen(screen)
        await pilot.pause()
        screen._mode = "add_marketplace"
        screen._refresh_view()
        await pilot.pause()

        source = screen.query_one("#plugin-marketplace-source", Input)
        assert source.has_focus

        await pilot.press("o", "w", "n", "e", "r", "/", "r", "e", "p", "o")
        assert source.value == "owner/repo"


def test_filtered_plugins_matches_display_label() -> None:
    screen = PluginManagerScreen()
    screen._search_query = "convex"
    rows = (
        _PluginRow(
            plugin_id="cx-backend@tools",
            description="Database backend",
            enabled=False,
            version=None,
            author=None,
            display_name="Convex",
        ),
        _PluginRow(
            plugin_id="linear@tools",
            description="Issue tracker",
            enabled=False,
            version=None,
            author=None,
        ),
    )

    assert [row.plugin_id for row in screen._filtered_plugins(rows)] == [
        "cx-backend@tools"
    ]


def test_filtered_plugins_matches_description_case_insensitively() -> None:
    screen = PluginManagerScreen()
    screen._search_query = "DATA WAREHOUSE"
    rows = (
        _PluginRow(
            plugin_id="analytics@tools",
            description="Query the data warehouse",
            enabled=False,
            version=None,
            author=None,
            display_name="Analytics",
        ),
        _PluginRow(
            plugin_id="linear@tools",
            description="Issue tracker",
            enabled=False,
            version=None,
            author=None,
        ),
    )

    assert [row.plugin_id for row in screen._filtered_plugins(rows)] == [
        "analytics@tools"
    ]


def test_plugin_row_label_prefers_display_name() -> None:
    row = _PluginRow(
        plugin_id="convex@tools",
        description="Backend plugin",
        enabled=False,
        version=None,
        author=None,
        display_name="Convex",
    )
    assert row.label == "Convex"
    assert (
        _PluginRow(
            plugin_id="linear@tools",
            description="",
            enabled=False,
            version=None,
            author=None,
        ).label
        == "linear"
    )


def test_marketplace_options_omit_divider_when_empty() -> None:
    screen = PluginManagerScreen()
    screen._tab = "marketplaces"
    screen._state = _ManagerState(
        available_plugins=(),
        installed_plugins=(),
        marketplaces=(),
        errors=(),
    )

    options = screen._current_options()

    assert [option.id for option in options] == ["add-marketplace"]


def test_marketplace_options_pad_between_entries() -> None:
    """Marketplace entries are separated by disabled spacers, like the plugins list."""
    screen = PluginManagerScreen()
    screen._tab = "marketplaces"
    screen._state = _ManagerState(
        available_plugins=(),
        installed_plugins=(),
        marketplaces=(
            _MarketplaceRow("first", "owner/first", 1, 0),
            _MarketplaceRow("second", "owner/second", 1, 0),
            _MarketplaceRow("third", "owner/third", 1, 0),
        ),
        errors=(),
    )

    options = screen._current_options()

    ids = [option.id for option in options]
    assert ids == [
        "add-marketplace",
        "marketplace-divider",
        "marketplace:first",
        "marketplace-spacer:1",
        "marketplace:second",
        "marketplace-spacer:2",
        "marketplace:third",
    ]
    spacers = [
        option
        for option in options
        if option.id is not None and option.id.startswith("marketplace-spacer:")
    ]
    assert all(spacer.disabled for spacer in spacers)


def test_healthy_marketplace_label_shows_available_plugins() -> None:
    row = _MarketplaceRow("healthy", "owner/healthy", 3, 0)

    label = _marketplace_label(row)

    assert not row.has_error
    assert "3 available" in label.plain
    assert get_glyphs().error not in label.plain


def test_plugin_details_content_lists_preview_components() -> None:
    row = _PluginRow(
        plugin_id="quality@tools",
        description="Quality review",
        enabled=False,
        version="1.0.0",
        author="Team",
        skill_count=1,
        skill_names=("quality@tools:review",),
        mcp_server_names=("docs",),
    )

    content = str(_plugin_details_content(row))

    assert "Will install:" in content
    assert "Skills: quality@tools:review" in content
    assert "MCP: docs" in content
    assert "Components will be discovered at installation." not in content


def test_installed_details_explain_unsupported_components() -> None:
    row = _PluginRow(
        plugin_id="pr-review-toolkit@official",
        description="PR review",
        enabled=True,
        version="1.0.0",
        author=None,
        skill_count=0,
        unsupported_components=("agents", "commands"),
        session_loaded=True,
    )

    content = str(_installed_plugin_details_content(row))

    assert f"Status: {get_glyphs().checkmark} Enabled" in content
    assert "No supported components (skills/MCP)." in content
    assert "agents/" in content
    assert "commands/" in content
    assert "No components discovered." not in content


def test_installed_details_pending_reload_not_enabled() -> None:
    row = _PluginRow(
        plugin_id="quality@tools",
        description="Quality",
        enabled=True,
        version="1.0.0",
        author=None,
        skill_count=1,
        skill_names=("quality@tools:review",),
        session_loaded=False,
    )

    content = str(_installed_plugin_details_content(row))

    assert "Status: Installed · pending /reload" in content
    assert "Run /reload" in content
    assert f"{get_glyphs().checkmark} Enabled" not in content


def test_installed_details_pending_reload_after_disable() -> None:
    row = _PluginRow(
        plugin_id="quality@tools",
        description="Quality",
        enabled=False,
        version="1.0.0",
        author=None,
        skill_count=1,
        skill_names=("quality@tools:review",),
        session_loaded=True,
    )

    assert row.load_state == "pending_reload"
    content = str(_installed_plugin_details_content(row))

    assert "Status: Disabled · pending /reload" in content
    assert "unload this plugin" in content
    assert "Status: Disabled\n" not in content


def test_installed_details_error_state_shows_reason_and_fix() -> None:
    row = _PluginRow(
        plugin_id="broken@tools",
        description="Broken",
        enabled=True,
        version=None,
        author=None,
        session_loaded=False,
        load_error="install cache missing at /x; re-run install",
    )

    assert row.load_state == "error"
    content = str(_installed_plugin_details_content(row))

    assert "Status: Error — install cache missing at /x; re-run install" in content
    assert "Fix the error, then run /reload." in content


def test_installed_details_disabled_state_copy() -> None:
    row = _PluginRow(
        plugin_id="quality@tools",
        description="Quality",
        enabled=False,
        version="1.0.0",
        author=None,
        skill_count=1,
        skill_names=("quality@tools:review",),
        session_loaded=False,
    )

    assert row.load_state == "disabled"
    content = str(_installed_plugin_details_content(row))

    assert "Status: Disabled" in content
    assert "Enable the plugin, then run /reload to load it." in content


def test_installed_details_enabled_flags_mcp_restart() -> None:
    row = _PluginRow(
        plugin_id="quality@tools",
        description="Quality",
        enabled=True,
        version="1.0.0",
        author=None,
        skill_count=1,
        skill_names=("quality@tools:review",),
        mcp_connected=False,
        session_loaded=True,
    )

    content = str(_installed_plugin_details_content(row))

    assert f"Status: {get_glyphs().checkmark} Enabled" in content
    assert "MCP servers need a server restart (/reload) to connect." in content


def test_status_lines_enabled_is_success_and_companions_stay_dim() -> None:
    """The enabled status renders green ($success); companion lines stay dim."""
    row = _PluginRow(
        plugin_id="quality@tools",
        description="Quality",
        enabled=True,
        version="1.0.0",
        author=None,
        skill_count=1,
        skill_names=("quality@tools:review",),
        mcp_connected=False,  # forces the second (dim) status line
        session_loaded=True,
    )

    lines = _status_lines(row)

    assert lines[0].plain == f"Status: {get_glyphs().checkmark} Enabled"
    assert [span.style for span in lines[0].spans] == ["$success"]
    # The MCP-restart companion line must NOT inherit the success color.
    assert lines[1].plain.startswith("MCP servers need a server restart")
    assert [span.style for span in lines[1].spans] == ["dim"]


def test_status_lines_non_enabled_states_stay_dim() -> None:
    """Every line of a non-enabled status keeps the plain dim styling."""
    row = _PluginRow(
        plugin_id="broken@tools",
        description="Broken",
        enabled=True,
        version=None,
        author=None,
        session_loaded=False,
        load_error="boom",  # load_state == "error"
    )

    assert row.load_state == "error"
    assert all(
        span.style == "dim" for line in _status_lines(row) for span in line.spans
    )


def test_load_state_label_matches_state() -> None:
    def _row(
        *, enabled: bool, session_loaded: bool, load_error: str | None = None
    ) -> _PluginRow:
        return _PluginRow(
            plugin_id="p@m",
            description="",
            enabled=enabled,
            version=None,
            author=None,
            session_loaded=session_loaded,
            load_error=load_error,
        )

    checkmark = get_glyphs().checkmark
    assert _load_state_label(_row(enabled=True, session_loaded=True)) == (
        f"{checkmark} enabled"
    )
    assert (
        _load_state_label(_row(enabled=True, session_loaded=False)) == "pending /reload"
    )
    assert _load_state_label(_row(enabled=False, session_loaded=False)) is None
    assert (
        _load_state_label(_row(enabled=True, session_loaded=True, load_error="boom"))
        == "error"
    )


def test_plugin_prompt_enabled_connection_hints() -> None:
    checkmark = get_glyphs().checkmark

    connected = _PluginRow(
        plugin_id="quality@tools",
        description="Quality",
        enabled=True,
        version="1.0.0",
        author=None,
        mcp_connected=True,
        session_loaded=True,
    )
    prompt = str(_plugin_prompt(connected, status=None))
    assert f"{checkmark} enabled" in prompt
    assert f"{checkmark} connected" in prompt
    assert "restart to connect" not in prompt

    needs_restart = _PluginRow(
        plugin_id="quality@tools",
        description="Quality",
        enabled=True,
        version="1.0.0",
        author=None,
        mcp_connected=False,
        session_loaded=True,
    )
    prompt = str(_plugin_prompt(needs_restart, status=None))
    assert "restart to connect" in prompt
    assert f"{checkmark} connected" not in prompt


def test_plugin_prompt_pending_reload_keeps_connected_when_still_loaded() -> None:
    """A disabled-but-still-loaded plugin keeps its live connection hint."""
    checkmark = get_glyphs().checkmark
    row = _PluginRow(
        plugin_id="quality@tools",
        description="Quality",
        enabled=False,
        version="1.0.0",
        author=None,
        mcp_connected=True,
        session_loaded=True,
    )

    assert row.load_state == "pending_reload"
    prompt = str(_plugin_prompt(row, status=None))

    assert "pending /reload" in prompt
    assert f"{checkmark} connected" in prompt


async def test_install_keeps_manager_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    screen = PluginManagerScreen()
    screen._selected_plugin = _PluginRow(
        plugin_id="linear@tools",
        description="Linear plugin",
        enabled=False,
        version=None,
        author=None,
        display_name="Linear",
    )
    monkeypatch.setattr(
        "deepagents_code.tui.modals.plugin_manager.install_plugin",
        lambda _plugin_id: object(),
    )
    monkeypatch.setattr(
        "deepagents_code.plugins.adapters.mcp.plugin_mcp_server_entries",
        lambda _instance: (),
    )
    monkeypatch.setattr(screen, "_refresh_state", AsyncMock())
    monkeypatch.setattr(screen, "notify", MagicMock())
    dismiss = MagicMock()
    monkeypatch.setattr(screen, "dismiss", dismiss)

    await screen._install_selected_plugin()

    dismiss.assert_not_called()


async def test_install_preserves_mcp_login_guidance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    screen = PluginManagerScreen()
    screen._selected_plugin = _PluginRow(
        plugin_id="linear@tools",
        description="Linear plugin",
        enabled=False,
        version=None,
        author=None,
        display_name="Linear",
    )
    monkeypatch.setattr(
        "deepagents_code.tui.modals.plugin_manager.install_plugin",
        lambda _plugin_id: object(),
    )
    monkeypatch.setattr(
        "deepagents_code.plugins.adapters.mcp.plugin_mcp_server_entries",
        lambda _instance: (("linear", "plugin__linear_tools__linear", True),),
    )
    monkeypatch.setattr(screen, "_refresh_state", AsyncMock())
    monkeypatch.setattr(screen, "notify", MagicMock())

    await screen._install_selected_plugin()

    assert screen._status is not None
    assert "After reload, sign in to Linear via /mcp." in screen._status


@pytest.mark.parametrize(
    ("key", "expected_option_id"),
    [("right", "action:install"), ("left", "details-back")],
)
async def test_details_navigation_starts_at_matching_edge(
    key: str, expected_option_id: str
) -> None:
    app = DeepAgentsApp(agent=MagicMock(), thread_id="t")
    async with app.run_test() as pilot:
        screen = PluginManagerScreen()
        app.push_screen(screen)
        await pilot.pause()

        screen._selected_plugin = _PluginRow(
            plugin_id="linear@tools",
            description="Linear plugin",
            enabled=False,
            version=None,
            author=None,
        )
        screen._mode = "plugin_details"
        screen._refresh_view()
        options = screen.query_one("#plugin-manager-options", OptionList)
        options.highlighted = None

        await pilot.press(key)

        highlighted = options.highlighted
        assert highlighted is not None
        assert options.get_option_at_index(highlighted).id == expected_option_id


async def test_installed_details_divider_refits_on_resize() -> None:
    app = DeepAgentsApp(agent=MagicMock(), thread_id="t")
    async with app.run_test(size=(120, 40)) as pilot:
        screen = PluginManagerScreen()
        app.push_screen(screen)
        await pilot.pause()

        screen._selected_plugin = _PluginRow(
            plugin_id="linear@tools",
            description="Linear plugin",
            enabled=True,
            version="1.0.0",
            author=None,
        )
        screen._mode = "installed_details"
        screen._refresh_view()
        await pilot.pause()

        options = screen.query_one("#plugin-manager-options", OptionList)
        assert options.get_option_at_index(2).id == "details-divider"
        box = get_glyphs().box_horizontal
        wide_width = str(options.get_option_at_index(2).prompt).count(box)
        assert wide_width > 0

        await pilot.resize_terminal(60, 40)
        await pilot.pause()

        narrow_width = str(options.get_option_at_index(2).prompt).count(box)
        assert 0 < narrow_width < wide_width


async def test_marketplace_divider_refits_on_resize() -> None:
    """The width-sized divider shrinks with the modal on a narrower terminal."""
    app = DeepAgentsApp(agent=MagicMock(), thread_id="t")
    async with app.run_test(size=(120, 40)) as pilot:
        screen = PluginManagerScreen()
        app.push_screen(screen)
        await pilot.pause()

        screen._tab = "marketplaces"
        screen._state = _ManagerState(
            available_plugins=(),
            installed_plugins=(),
            marketplaces=(_MarketplaceRow("tools", "owner/repo", 2, 0),),
            errors=(),
        )
        screen._refresh_view()
        await pilot.pause()

        options = screen.query_one("#plugin-manager-options", OptionList)
        assert options.get_option_at_index(1).id == "marketplace-divider"
        box = get_glyphs().box_horizontal
        wide_width = str(options.get_option_at_index(1).prompt).count(box)
        assert wide_width > 0

        await pilot.resize_terminal(60, 40)
        await pilot.pause()

        narrow_width = str(options.get_option_at_index(1).prompt).count(box)
        assert 0 < narrow_width < wide_width


async def test_plugin_manager_overlays_underlying_content() -> None:
    """Dimmed modal backdrop must composite the screen underneath."""
    app = DeepAgentsApp(agent=MagicMock(), thread_id="t")
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.theme = "textual-dark"
        await pilot.pause()

        await app.mount(Static("TOP_MARKER_VISIBLE", id="top-marker"))
        marker = app.query_one("#top-marker")
        marker.styles.dock = "top"
        marker.styles.height = 1
        await pilot.pause()

        app.push_screen(PluginManagerScreen())
        await pilot.pause()

        # Inherit the default ModalScreen dim backdrop instead of a fully
        # transparent one. The alpha is in (0, 1) only under a non-ansi theme
        # (hence the "textual-dark" pin above); it degrades to transparent
        # under ansi themes.
        assert 0 < app.screen.styles.background.a < 1
        plain = re.sub(r"<[^>]+>", " ", app.export_screenshot())
        assert "TOP_MARKER_VISIBLE" in plain
        assert "Plugins" in plain
