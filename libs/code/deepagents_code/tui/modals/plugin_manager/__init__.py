"""Interactive plugin manager screen."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, ClassVar

from textual import work
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical
from textual.content import Content
from textual.css.query import NoMatches
from textual.screen import ModalScreen
from textual.widgets import Input, OptionList, Rule, Static
from textual.widgets.option_list import Option, OptionDoesNotExist

from deepagents_code.tui.widgets.loading import Spinner

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence, Set as AbstractSet

    from textual.app import ComposeResult
    from textual.events import Key
    from textual.timer import Timer

    from deepagents_code.mcp_tools import MCPServerInfo
    from deepagents_code.plugins.models import PluginMarketplace
    from deepagents_code.tui.modals.plugin_manager.models import (
        PluginClosePhase,
        PluginManagerView,
        _MarketplaceRow,
        _PluginRow,
    )

from deepagents_code import theme
from deepagents_code.config import get_glyphs, is_ascii_mode
from deepagents_code.model_config import _save_toml_field
from deepagents_code.plugins import (
    add_marketplace_source,
    install_plugin,
    remove_marketplace,
    set_installed_plugin_enabled,
    uninstall_plugin,
)
from deepagents_code.plugins.discovery import plugin_auto_update_setting
from deepagents_code.plugins.marketplace import MarketplaceError
from deepagents_code.tui.modals.plugin_manager.content import (
    _confirm_marketplace_removal_options,
    _install_details_options,
    _installed_details_options,
    _installed_plugin_details_content,
    _marketplace_details_content,
    _marketplace_details_options,
    _marketplace_label,
    _marketplace_removal_content,
    _plugin_details_content,
    _plugin_options,
)
from deepagents_code.tui.modals.plugin_manager.models import (
    PluginManagerResult,
    PluginTab,
    _ManagerState,
)
from deepagents_code.tui.modals.plugin_manager.state import _load_manager_state
from deepagents_code.tui.modals.plugin_manager.tabs import (
    TAB_LABELS,
    PluginTabLabel,
    PluginTabSelected,
)

logger = logging.getLogger(__name__)  # noqa: RUF067  # module-level logger

_CONNECTION_REFRESH_ERROR = (  # noqa: RUF067  # module-level constant
    "Could not refresh MCP connection status. Reopen /plugins."
)
"""Banner for a failed connection refresh, cleared when a later one succeeds."""

# Bounds the close check so a stalled plugin scan cannot leave the manager
# mounted with its keys locked out. Deliberately shorter than the app's modal
# watchdog: nothing here waits on the user, only on a filesystem snapshot.
_CLOSE_CHECK_TIMEOUT_SECONDS = 30.0  # noqa: RUF067  # module-level constant


class PluginManagerScreen(ModalScreen[PluginManagerResult]):  # noqa: RUF067
    """Arrow-key navigable plugin manager for `/plugins`.

    Closing runs the optional `check_reload_required` callback while this screen
    stays mounted. Changed plugin state transitions in place to a reload prompt
    and dismisses with `"reload"` or `"later"`; a failed or timed-out check
    dismisses with `"check_failed"`; unchanged state, or no callback at all,
    dismisses with `None` and shows nothing.
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Close", show=False, priority=True),
        Binding("enter", "reload_plugins", "Reload", show=False, priority=True),
        # Separate from tab/shift+tab so check_action can release arrows to a
        # non-empty Input for caret movement while keeping Tab as tab cycling.
        Binding(
            "left", "arrow_previous_tab", "Previous tab", show=False, priority=True
        ),
        Binding("right", "arrow_next_tab", "Next tab", show=False, priority=True),
        Binding("tab", "next_tab", "Next tab", show=False, priority=True),
        Binding("shift+tab", "previous_tab", "Previous tab", show=False, priority=True),
        Binding("up", "cursor_up", "Up", show=False, priority=True),
        Binding("down", "cursor_down", "Down", show=False, priority=True),
        Binding("/", "focus_search", "Search", show=False, priority=True),
    ]

    CSS_PATH = "plugin_manager.tcss"
    # Prefer the option list over the search Input so Enter activates rows on
    # open; `/` still focuses search explicitly.
    AUTO_FOCUS = "#plugin-manager-options"

    # Divider width used before the options list has been laid out (e.g. in unit
    # tests that build options off-screen). At render time the divider is sized to
    # the measured options width instead so it never wraps on a narrower modal.
    _DIVIDER_FALLBACK_WIDTH: ClassVar[int] = 72

    _tabs: ClassVar[tuple[PluginTab, ...]] = (
        "discover",
        "installed",
        "marketplaces",
        "errors",
        "settings",
    )

    def __init__(
        self,
        *,
        mcp_server_info: Sequence[MCPServerInfo] = (),
        mcp_connecting: bool = False,
        loaded_plugin_ids: AbstractSet[str] | None = None,
        on_auto_update_enabled: Callable[[], None] | None = None,
        check_reload_required: Callable[[], Awaitable[bool | None]] | None = None,
    ) -> None:
        """Initialize the plugin manager.

        Args:
            mcp_server_info: Live MCP server metadata from the running session,
                used to show connection status for plugins that declare MCP
                servers.
            mcp_connecting: Whether MCP connection status is still loading.
            loaded_plugin_ids: Plugin ids loaded into the current session.
                Plugins whose enabled state differs from this set (enabled but
                not loaded, or disabled but still loaded) are shown as pending
                reload.
            on_auto_update_enabled: Called after auto-update is enabled.
            check_reload_required: Awaited on close to decide whether plugin
                state changed since the manager opened. `True` shows the inline
                reload prompt, `False` closes silently, and `None` means the
                check failed. When omitted, the manager closes without
                prompting.
        """
        super().__init__()
        self._tab: PluginTab = "discover"
        self._mode: PluginManagerView = "list"
        self._mcp_server_info = mcp_server_info
        self._mcp_connecting = mcp_connecting
        self._loaded_plugin_ids: frozenset[str] = frozenset(loaded_plugin_ids or ())
        self._on_auto_update_enabled = on_auto_update_enabled
        self._check_reload_required = check_reload_required
        self._close_phase: PluginClosePhase = "browsing"
        self._state = _ManagerState((), (), (), ())
        self._status: str | None = None
        self._error: str | None = None
        self._selected_plugin: _PluginRow | None = None
        self._selected_marketplace: _MarketplaceRow | None = None
        self._adding_marketplace = False
        self._marketplace_spinner = Spinner()
        self._marketplace_spinner_timer: Timer | None = None
        self._search_query = ""
        self._auto_update_enabled = False
        self._auto_update_source = "default"
        # State loads run in worker threads. Keep their snapshots and view
        # updates ordered so an older connecting snapshot cannot overwrite a
        # later settled connection snapshot.
        self._refresh_lock = asyncio.Lock()
        self._refresh_tasks: set[asyncio.Task[None]] = set()
        """Strong references to in-flight fire-and-forget refresh tasks."""

        self._dismissed = False
        """Whether the screen has been unmounted.

        Textual leaves `is_mounted` `True` and `is_attached` `False` after a
        dismiss, and `is_attached` is also `False` before the first mount, so
        neither flag distinguishes a dismissed screen from a fresh one.
        """

    def compose(self) -> ComposeResult:
        """Compose the manager screen.

        Yields:
            Widgets for the plugin manager UI.
        """
        with Vertical():
            yield Static(
                "Plugins", id="plugin-manager-title", classes="plugin-manager-title"
            )
            with Horizontal(id="plugin-manager-tabs", classes="plugin-manager-tabs"):
                for tab in self._tabs:
                    yield PluginTabLabel(tab, TAB_LABELS[tab])
            yield Rule(
                line_style="heavy" if not is_ascii_mode() else "ascii",
                id="plugin-manager-divider",
                classes="plugin-manager-divider",
            )
            yield Static(
                "",
                id="plugin-manager-status",
                classes="plugin-manager-status",
                markup=False,
            )
            yield Static(
                "",
                id="plugin-manager-error",
                classes="plugin-manager-error",
                markup=False,
            )
            yield Input(
                placeholder="Search plugins...",
                select_on_focus=False,
                id="plugin-manager-search",
            )
            yield OptionList(id="plugin-manager-options")
            yield Input(
                placeholder="",
                id="plugin-marketplace-source",
            )
            yield Static("", id="plugin-manager-help", classes="plugin-manager-help")

    async def on_mount(self) -> None:
        """Apply initial render, then load plugin state off the UI thread."""
        if is_ascii_mode():
            container = self.query_one(Vertical)
            colors = theme.get_theme_colors(self)
            container.styles.border = ("ascii", colors.success)
        self._status = "Loading plugins..."
        self._refresh_view()
        await self._refresh_state()
        if self._status == "Loading plugins...":
            self._status = None
            self._refresh_view()

    def on_unmount(self) -> None:
        """Stop accepting deferred refreshes once the screen is torn down."""
        self._dismissed = True

    def on_resize(self) -> None:
        """Refit width-sized dividers when the modal resizes."""
        if self._close_phase != "browsing":
            return
        marketplace_divider_visible = (
            self._mode == "list"
            and self._tab == "marketplaces"
            and bool(self._state.marketplaces)
        )
        details_divider_visible = (
            self._mode == "installed_details" and self._selected_plugin is not None
        )
        if marketplace_divider_visible or details_divider_visible:
            self._refresh_view()

    def _update_tab_labels(self) -> None:
        """Refresh active styling on each clickable tab label."""
        for tab in self._tabs:
            self.query_one(f"#plugin-tab-{tab}", PluginTabLabel).set_active(
                tab == self._tab
            )

    def _select_tab(self, tab: PluginTab) -> None:
        """Activate `tab`, exiting details into list mode when needed.

        Args:
            tab: Tab to show.
        """
        if self._close_phase != "browsing" or self._mode == "add_marketplace":
            return
        if self._details_mode_active():
            self._mode = "list"
            self._selected_plugin = None
            self._selected_marketplace = None
        if tab != self._tab:
            # A query typed on one tab should not silently filter another.
            self._search_query = ""
        self._tab = tab
        self._error = None
        self._refresh_view()

    def _search_available(self) -> bool:
        """Whether the plugin filter should be shown and focusable.

        Returns:
            `True` when list mode has plugins that can be filtered.
        """
        if self._mode != "list":
            return False
        if self._tab == "discover":
            return bool(self._state.marketplaces and self._state.available_plugins)
        if self._tab == "installed":
            return bool(self._state.installed_plugins)
        return False

    def _filtered_plugins(self, rows: Sequence[_PluginRow]) -> tuple[_PluginRow, ...]:
        query = self._search_query.strip().casefold()
        if not query:
            return tuple(rows)
        return tuple(
            row
            for row in rows
            if query in row.plugin_id.casefold()
            or query in row.label.casefold()
            or query in row.description.casefold()
        )

    def _current_options(self) -> list[Option]:
        glyphs = get_glyphs()
        if self._tab == "discover":
            if not self._state.marketplaces:
                return [
                    Option(
                        "No marketplaces installed. Add one to discover plugins.",
                        id="empty",
                        disabled=True,
                    ),
                    Option("+ Add marketplace", id="add-marketplace"),
                ]
            if not self._state.available_plugins:
                return [Option("All available plugins are installed.", id="empty")]
            rows = self._filtered_plugins(self._state.available_plugins)
            if not rows:
                return [
                    Option("No plugins match your search.", id="empty", disabled=True)
                ]
            return _plugin_options(rows, action="detail", status=None)
        if self._tab == "installed":
            if not self._state.installed_plugins:
                return [Option("No plugins installed.", id="empty")]
            rows = self._filtered_plugins(self._state.installed_plugins)
            if not rows:
                return [
                    Option(
                        "No installed plugins match your search.",
                        id="empty",
                        disabled=True,
                    )
                ]
            return _plugin_options(rows, action="installed", status=None)
        if self._tab == "marketplaces":
            options = [Option("+ Add marketplace", id="add-marketplace")]
            if self._state.marketplaces:
                options.append(
                    Option(
                        Content.styled(
                            glyphs.box_horizontal * self._divider_width(), "dim"
                        ),
                        id="marketplace-divider",
                        disabled=True,
                    )
                )
            for index, row in enumerate(self._state.marketplaces):
                if index > 0:
                    options.append(
                        Option(" ", id=f"marketplace-spacer:{index}", disabled=True)
                    )
                options.append(
                    Option(
                        _marketplace_label(row),
                        id=f"marketplace:{row.name}",
                    )
                )
            return options
        if self._tab == "settings":
            label = "enabled" if self._auto_update_enabled else "disabled"
            env_override = self._auto_update_source.startswith("env (")
            suffix = " (set by environment)" if env_override else ""
            return [
                Option(
                    f"Auto-update plugins: {label}{suffix}",
                    id="action:toggle-auto-update",
                    disabled=env_override,
                )
            ]
        if not self._state.errors:
            return [Option("No plugin errors.", id="empty")]
        return [Option(Content(error), id="empty") for error in self._state.errors]

    def _divider_width(self) -> int:
        """Width for option-list dividers, sized to the available content.

        The options list respects the modal's `max-width`, so a fixed width wraps on
        terminals narrower than the full modal. Measure the laid-out content width when
        available and fall back to a constant before the first layout (e.g. in tests).

        Returns:
            The measured options content width, or `_DIVIDER_FALLBACK_WIDTH` if the
            options list is not mounted or has not been laid out yet.
        """
        try:
            width = self.query_one(
                "#plugin-manager-options", OptionList
            ).content_size.width
        except NoMatches:
            return self._DIVIDER_FALLBACK_WIDTH
        return width if width > 0 else self._DIVIDER_FALLBACK_WIDTH

    @staticmethod
    def _nearest_enabled_index(options: OptionList, candidate: int) -> int | None:
        """Return the nearest selectable option index to `candidate`.

        Scope-group header rows (and spacers) are disabled options, so a
        highlighted index carried over from a previous tab/refresh can land
        on one after the option count or grouping changes. Scans forward
        first, then backward, so the cursor always rests on a real row.

        Args:
            options: Option list to scan.
            candidate: Preferred index (already clamped to bounds).

        Returns:
            `candidate` if selectable, the nearest selectable index otherwise,
            or `None` if every option (or the list itself) is disabled/empty.
        """
        if not options.option_count:
            return None
        if not options.get_option_at_index(candidate).disabled:
            return candidate
        for index in range(candidate + 1, options.option_count):
            if not options.get_option_at_index(index).disabled:
                return index
        for index in range(candidate - 1, -1, -1):
            if not options.get_option_at_index(index).disabled:
                return index
        return None

    def _details_mode_active(self) -> bool:
        return self._mode in {
            "plugin_details",
            "installed_details",
            "marketplace_details",
            "confirm_remove_marketplace",
        }

    def _refresh_view(self) -> None:
        # _show_close_state hides the browsing chrome one way; repainting it
        # here would restore the tabs and option list underneath the reload
        # prompt while its key bindings are still active.
        if self._close_phase != "browsing":
            return
        title = self.query_one("#plugin-manager-title", Static)
        tabs = self.query_one("#plugin-manager-tabs", Horizontal)
        divider = self.query_one("#plugin-manager-divider", Rule)
        self._update_tab_labels()
        status_widget = self.query_one("#plugin-manager-status", Static)
        if self._mode == "plugin_details" and self._selected_plugin is not None:
            status_widget.update(_plugin_details_content(self._selected_plugin))
        elif self._mode == "installed_details" and self._selected_plugin is not None:
            status_widget.update(
                _installed_plugin_details_content(self._selected_plugin)
            )
        elif (
            self._mode == "marketplace_details"
            and self._selected_marketplace is not None
        ):
            status_widget.update(
                _marketplace_details_content(self._selected_marketplace)
            )
        elif (
            self._mode == "confirm_remove_marketplace"
            and self._selected_marketplace is not None
        ):
            status_widget.update(
                _marketplace_removal_content(self._selected_marketplace)
            )
        else:
            status_widget.update(self._status or "")
        error = self._error or ""
        self.query_one("#plugin-manager-error", Static).update(error)

        options = self.query_one("#plugin-manager-options", OptionList)
        search_input = self.query_one("#plugin-manager-search", Input)
        source_input = self.query_one("#plugin-marketplace-source", Input)
        help_text = self.query_one("#plugin-manager-help", Static)
        glyphs = get_glyphs()

        if self._mode == "add_marketplace":
            title.update("Add Marketplace")
            tabs.display = False
            divider.display = False
            if self._status is None:
                status_widget.update(
                    "Enter marketplace source:\n"
                    "\n"
                    "Examples:\n"
                    f"  {glyphs.bullet} owner/repo (GitHub)\n"
                    f"  {glyphs.bullet} git@github.com:owner/repo.git (SSH)\n"
                    f"  {glyphs.bullet} https://example.com/marketplace.json\n"
                    f"  {glyphs.bullet} ./path/to/marketplace"
                )
            options.display = False
            search_input.display = False
            source_input.display = True
            source_input.focus()
            help_text.update(f"Enter to add {glyphs.bullet} Esc to cancel")
            return

        title.update("Plugins")
        tabs.display = True
        divider.display = True

        if self._details_mode_active():
            search_input.display = False
            source_input.display = False
            options.display = True
            highlighted = options.highlighted
            highlighted_id = (
                None
                if highlighted is None
                else options.get_option_at_index(highlighted).id
            )
            options.clear_options()
            detail_options = self._active_details_options()
            for option in detail_options:
                options.add_option(option)
            if options.option_count:
                candidate = 0
                if highlighted_id is not None:
                    try:
                        candidate = options.get_option_index(highlighted_id)
                    except OptionDoesNotExist:
                        candidate = 0
                options.highlighted = self._nearest_enabled_index(options, candidate)
            options.focus()
            help_text.update(
                f"{glyphs.arrow_up}/{glyphs.arrow_down} select {glyphs.bullet} "
                f"Enter choose {glyphs.bullet} Esc back"
            )
            return

        source_input.display = False
        options.display = True
        search_input.display = self._search_available()
        if search_input.display and search_input.value != self._search_query:
            search_input.value = self._search_query
        highlighted = options.highlighted
        options.clear_options()
        for option in self._current_options():
            options.add_option(option)
        if options.option_count:
            candidate = (
                0 if highlighted is None else min(highlighted, options.option_count - 1)
            )
            options.highlighted = self._nearest_enabled_index(options, candidate)
        if not search_input.has_focus:
            options.focus()

        if self._tab == "marketplaces":
            help_text.update(
                f"{glyphs.arrow_up}/{glyphs.arrow_down} select {glyphs.bullet} "
                f"Enter add/view {glyphs.bullet} "
                f"Left/Right tabs {glyphs.bullet} Esc close"
            )
        elif self._tab in {"discover", "installed"}:
            if self._tab == "installed":
                action = "view"
            elif not self._state.marketplaces:
                action = "add marketplace"
            else:
                action = "install"
            search_hint = (
                f"/ search {glyphs.bullet} " if self._search_available() else ""
            )
            help_text.update(
                f"{glyphs.arrow_up}/{glyphs.arrow_down} select {glyphs.bullet} "
                f"Enter {action} {glyphs.bullet} {search_hint}Left/Right tabs "
                f"{glyphs.bullet} Esc close"
            )
        else:
            help_text.update(f"Left/Right tabs {glyphs.bullet} Esc close")

    def _active_details_options(self) -> list[Option]:
        if self._mode == "plugin_details":
            return _install_details_options()
        if self._mode == "installed_details" and self._selected_plugin is not None:
            return _installed_details_options(
                self._selected_plugin, divider_width=self._divider_width()
            )
        if (
            self._mode == "marketplace_details"
            and self._selected_marketplace is not None
        ):
            return _marketplace_details_options()
        if (
            self._mode == "confirm_remove_marketplace"
            and self._selected_marketplace is not None
        ):
            return _confirm_marketplace_removal_options(self._selected_marketplace)
        return [Option("Back to plugin list", id="details-back")]

    def update_connection_state(
        self,
        mcp_server_info: Sequence[MCPServerInfo],
        *,
        mcp_connecting: bool,
    ) -> None:
        """Update the live MCP connection snapshot and re-render.

        The constructor only receives a snapshot of the app's connection
        state, so a manager opened while a connection is still settling would
        otherwise keep `mcp_connecting=True` forever — suppressing, for
        session-loaded plugins that declare MCP servers, both the "connected"
        indicator and a legitimate "run /reload" hint until the modal is
        closed and reopened. The app calls this whenever a connection attempt
        settles — `ServerReady` or a startup failure, including reconnects —
        so the open manager reflects the final connection state.

        Args:
            mcp_server_info: Settled MCP server metadata from the session.
            mcp_connecting: Whether MCP connection status is still loading.
                While `True`, per-plugin MCP status is omitted entirely rather
                than rendered as disconnected.
        """
        if self._dismissed:
            # The app resolves its handle synchronously but this refresh is
            # deferred, so a dismiss landing in between would otherwise leave
            # `_refresh_view` querying a torn-down DOM.
            return
        self._mcp_server_info = mcp_server_info
        self._mcp_connecting = mcp_connecting
        task = asyncio.create_task(self._refresh_state(clear_search=False))
        # `asyncio` keeps only a weak reference to a running task, so hold a
        # strong one until it completes or the refresh can vanish mid-flight.
        self._refresh_tasks.add(task)
        task.add_done_callback(self._refresh_tasks.discard)
        task.add_done_callback(self._settle_refresh_result)

    def _settle_refresh_result(self, task: asyncio.Task[None]) -> None:
        """Report a failed connection-state refresh instead of dropping it.

        Logging alone would leave the manager showing the pre-settle snapshot
        with no on-screen signal — indistinguishable from the stale state this
        refresh exists to clear. Route through `_error` like every other
        failure path in this screen, and retire that banner once a later
        refresh succeeds.

        Only this callback's own message is cleared: a reconnect must not wipe
        a marketplace or install error the user still needs to read.
        """
        if task.cancelled():
            return
        exc = task.exception()
        if not self.is_attached:
            # Unmounted mid-refresh: a failure here means the DOM is gone, not
            # that the plugin state is unreadable — and there is nothing left
            # to render a message into either way.
            if exc is not None:
                logger.warning(
                    "Plugin manager connection refresh failed after unmount",
                    exc_info=exc,
                )
            return
        if exc is None:
            if self._error == _CONNECTION_REFRESH_ERROR:
                self._error = None
                self._refresh_view()
            return
        logger.warning(
            "Failed to refresh plugin manager connection state",
            exc_info=exc,
        )
        self._error = _CONNECTION_REFRESH_ERROR
        self._refresh_view()

    async def _refresh_state(self, *, clear_search: bool = True) -> None:
        """Load and apply plugin state without allowing refreshes to overlap.

        Args:
            clear_search: Whether this refresh follows a plugin mutation that
                should show the complete, newly loaded list.
        """
        async with self._refresh_lock:
            # A mutating action (install/toggle/uninstall/marketplace change)
            # reloads state and shows a fresh list, so a leftover query would
            # hide results. Connection updates do not change plugin data and
            # must retain the user's active filter.
            if clear_search:
                self._search_query = ""

            # Take the snapshot only after acquiring the lock. A settled
            # connection update queued behind the initial load therefore uses
            # the current state rather than re-applying the connecting one.
            mcp_server_info = tuple(self._mcp_server_info)
            mcp_connecting = self._mcp_connecting
            loaded_plugin_ids = self._loaded_plugin_ids
            self._state = await asyncio.to_thread(
                _load_manager_state,
                mcp_server_info,
                mcp_connecting=mcp_connecting,
                loaded_plugin_ids=loaded_plugin_ids,
            )
            (
                self._auto_update_enabled,
                self._auto_update_source,
            ) = await asyncio.to_thread(plugin_auto_update_setting)
            if self._selected_plugin is not None:
                refreshed = self._find_installed_plugin(self._selected_plugin.plugin_id)
                if refreshed is None:
                    refreshed = self._find_available_plugin(
                        self._selected_plugin.plugin_id
                    )
                self._selected_plugin = refreshed
            if self._selected_marketplace is not None:
                self._selected_marketplace = self._find_marketplace(
                    self._selected_marketplace.name
                )
            self._refresh_view()

    def check_action(
        self,
        action: str,
        parameters: tuple[object, ...],  # noqa: ARG002  # required by Textual's DOMNode.check_action override signature
    ) -> bool | None:
        """Gate priority bindings that would otherwise steal Input keystrokes.

        `/` is enabled only when the plugin filter is visible and not focused,
        so it remains typeable in the filter and Add Marketplace source field,
        and cannot steal focus while the filter is hidden.
        Left/right release to a focused Input once it has at least one character,
        so caret movement works while empty-field arrows keep switching tabs.

        Args:
            action: Textual action name being considered for dispatch.
            parameters: Parameters associated with the action.

        Returns:
            `False` to step a binding aside so the focused widget receives the
                key; `True` to allow the action.
        """
        if self._close_phase != "browsing":
            prompting = self._close_phase == "reload_prompt"
            if action == "reload_plugins":
                return prompting
            return action == "cancel" and prompting
        if action == "reload_plugins":
            return False
        if action in {"arrow_previous_tab", "arrow_next_tab"}:
            focused = self.focused
            return not (isinstance(focused, Input) and bool(focused.value))
        if action == "focus_search":
            if not self._search_available():
                return False
            try:
                return not self.query_one("#plugin-manager-search", Input).has_focus
            except NoMatches:
                return True
        return True

    def on_key(self, event: Key) -> None:
        """Focus plugin search when a letter is typed from another control.

        Seed the character into the filter before focusing so the input never
        paints empty with a caret flash (and so `select_on_focus` cannot leave
        the inserted text selected for the next keypress).
        """
        if self._close_phase != "browsing" or not self._search_available():
            return

        search_input = self.query_one("#plugin-manager-search", Input)
        if search_input.has_focus:
            return

        character = event.character
        if not character or not character.isalpha():
            return

        # Mutate first, then focus, so the first focused frame already shows the
        # typed letter — matching type-to-search in the thread filter.
        new_value = f"{search_input.value}{character}"
        search_input.value = new_value
        search_input.selection = type(search_input.selection).cursor(len(new_value))
        search_input.focus()
        event.stop()

    def on_plugin_tab_selected(self, event: PluginTabSelected) -> None:
        """Switch tabs from a mouse click on a tab label.

        Args:
            event: Tab selection message from `PluginTabLabel`.
        """
        self._select_tab(event.tab)

    def _show_close_state(self, title: str, body: str, help_text: str) -> None:
        """Switch the modal to a one-way terminal close state.

        Hides the browsing chrome and repaints the title, status and help lines.
        There is no path back to the list; `_refresh_view` is gated on the close
        phase so nothing restores what this hides.

        Args:
            title: Replacement modal title.
            body: Status text shown in place of the plugin list.
            help_text: Key hints; an empty string hides the help line.
        """
        self.add_class("plugin-close-state")
        self.query_one("#plugin-manager-title", Static).update(title)
        self.query_one("#plugin-manager-tabs", Horizontal).display = False
        self.query_one("#plugin-manager-divider", Rule).display = False
        status = self.query_one("#plugin-manager-status", Static)
        status.update(body)
        status.display = True
        self.query_one("#plugin-manager-error", Static).display = False
        self.query_one("#plugin-manager-search", Input).display = False
        self.query_one("#plugin-manager-options", OptionList).display = False
        self.query_one("#plugin-marketplace-source", Input).display = False
        help_widget = self.query_one("#plugin-manager-help", Static)
        help_widget.update(help_text)
        help_widget.display = bool(help_text)

    async def _close_or_prompt(self) -> None:
        """Run the close check, then dismiss or switch to the reload prompt.

        Runs as a worker so the manager stays painted while the check is in
        flight. Every path out either dismisses or leaves the reload prompt
        showing, because the `checking` phase locks out the keys that would
        otherwise close the modal.

        Raises:
            CancelledError: Re-raised after releasing the key lock so a
                cancelled worker cannot strand the manager.
        """
        check_reload_required = self._check_reload_required
        if check_reload_required is None:
            # Unreachable via action_cancel, which dismisses before starting
            # this worker; kept so the call below is unconditionally safe.
            self.dismiss(None)
            return
        try:
            try:
                reload_required = await asyncio.wait_for(
                    check_reload_required(), _CLOSE_CHECK_TIMEOUT_SECONDS
                )
            except TimeoutError:
                logger.warning("Plugin state check timed out before manager close")
                self.dismiss("check_failed")
                return
            except Exception:
                logger.exception("Plugin state check raised before manager close")
                self.dismiss("check_failed")
                return
            if reload_required is None:
                self.dismiss("check_failed")
            elif reload_required:
                # Enter the prompt phase only once the repaint has succeeded, so
                # a failed paint cannot leave the prompt bindings live over a
                # half-updated manager.
                self._show_close_state(
                    "Reload plugins?",
                    "Reload to apply changes to plugin skills and MCP tools.",
                    "Enter to reload, Esc for later",
                )
                self._close_phase = "reload_prompt"
            else:
                self.dismiss(None)
        except asyncio.CancelledError:
            # Release the key lock so a cancelled worker cannot strand the
            # manager in a phase that ignores Escape.
            self._close_phase = "browsing"
            raise
        except Exception:
            # The worker runs with exit_on_error=False, so an unhandled repaint
            # or dismiss failure would otherwise be swallowed and latch the
            # modal shut. Fall back to the deferred-reload reminder instead.
            logger.exception("Failed to finish plugin manager close")
            self.dismiss("later")

    def action_reload_plugins(self) -> None:
        """Apply pending plugin changes from the inline reload prompt."""
        if self._close_phase == "reload_prompt":
            self.dismiss("reload")

    def action_cancel(self) -> None:
        """Clear a query, back out of a view, answer the reload prompt, or close.

        Answering the reload prompt dismisses the manager with `"later"`; the
        final branch only *starts* the close check, so the screen may stay
        mounted afterwards.
        """
        if self._adding_marketplace or self._close_phase == "checking":
            return
        if self._close_phase == "reload_prompt":
            self.dismiss("later")
            return
        search_input = self.query_one("#plugin-manager-search", Input)
        if search_input.has_focus:
            if self._search_query:
                self._search_query = ""
                search_input.value = ""
                self._refresh_view()
                search_input.focus()
            else:
                self.query_one("#plugin-manager-options", OptionList).focus()
            return
        if self._mode == "add_marketplace":
            self._mode = "list"
            self._error = None
            self._refresh_view()
            return
        if self._details_mode_active():
            if self._mode == "confirm_remove_marketplace":
                self._mode = "marketplace_details"
                self._error = None
                self._refresh_view()
                return
            self._mode = "list"
            self._selected_plugin = None
            self._selected_marketplace = None
            self._error = None
            self._refresh_view()
            return
        if self._check_reload_required is None:
            self.dismiss(None)
            return
        # Keep the manager painted as-is while the check runs; only repaint
        # when the result is a reload prompt so an unchanged close shows no
        # intermediate state. The `checking` phase also locks out keys and
        # option-row activation for the duration (see `check_action`, `on_key`,
        # `on_option_list_option_selected`) so nothing mutates plugin state
        # after the snapshot is taken.
        self._close_phase = "checking"
        self.run_worker(
            self._close_or_prompt(),
            exclusive=True,
            group="plugin-manager-close",
            exit_on_error=False,
        )

    def action_focus_search(self) -> None:
        """Focus the plugin filter when it is visible."""
        if self._search_available():
            self.query_one("#plugin-manager-search", Input).focus()

    def _cycle_details_option(self, step: int) -> None:
        options = self.query_one("#plugin-manager-options", OptionList)
        enabled = [
            index
            for index in range(options.option_count)
            if not options.get_option_at_index(index).disabled
        ]
        if not enabled:
            return
        current = options.highlighted
        if current is not None and current in enabled:
            position = enabled.index(current)
            options.highlighted = enabled[(position + step) % len(enabled)]
        else:
            # Nothing highlighted yet: step forward to the first option, back to
            # the last.
            options.highlighted = enabled[0] if step > 0 else enabled[-1]
        options.focus()

    def action_arrow_next_tab(self) -> None:
        """Switch tabs via right arrow when the caret is not editing text."""
        self.action_next_tab()

    def action_arrow_previous_tab(self) -> None:
        """Switch tabs via left arrow when the caret is not editing text."""
        self.action_previous_tab()

    def action_next_tab(self) -> None:
        """Switch tabs or focus the next details option."""
        if self._details_mode_active():
            self._cycle_details_option(1)
            return
        if self._mode != "list":
            return
        index = self._tabs.index(self._tab)
        self._select_tab(self._tabs[(index + 1) % len(self._tabs)])

    def action_previous_tab(self) -> None:
        """Switch tabs or focus the previous details option."""
        if self._details_mode_active():
            self._cycle_details_option(-1)
            return
        if self._mode != "list":
            return
        index = self._tabs.index(self._tab)
        self._select_tab(self._tabs[(index - 1) % len(self._tabs)])

    def action_cursor_down(self) -> None:
        """Move the option-list cursor down."""
        if self._mode in {
            "list",
            "plugin_details",
            "installed_details",
            "marketplace_details",
            "confirm_remove_marketplace",
        }:
            self.query_one("#plugin-manager-options", OptionList).action_cursor_down()

    def action_cursor_up(self) -> None:
        """Move the option-list cursor up."""
        if self._mode in {
            "list",
            "plugin_details",
            "installed_details",
            "marketplace_details",
            "confirm_remove_marketplace",
        }:
            self.query_one("#plugin-manager-options", OptionList).action_cursor_up()

    async def on_option_list_option_selected(
        self, event: OptionList.OptionSelected
    ) -> None:
        """Handle row activation."""
        # Closing snapshots persisted state asynchronously. Ignore OptionList
        # messages that arrive during that interval so a row cannot open a
        # details view and mutate plugin state after the snapshot completes.
        if self._close_phase != "browsing":
            return
        option_id = event.option.id
        if option_id is None or option_id == "empty":
            return
        if option_id == "add-marketplace":
            self._mode = "add_marketplace"
            self._status = None
            self._error = None
            self.query_one("#plugin-marketplace-source", Input).value = ""
            self._refresh_view()
            return
        if option_id.startswith("marketplace:"):
            name = option_id.removeprefix("marketplace:")
            row = self._find_marketplace(name)
            if row is None:
                return
            self._selected_marketplace = row
            self._selected_plugin = None
            self._mode = "marketplace_details"
            self._status = None
            self._error = None
            self._refresh_view()
            return
        if option_id.startswith("detail:"):
            plugin_id = option_id.removeprefix("detail:")
            row = self._find_available_plugin(plugin_id)
            if row is None:
                return
            self._selected_plugin = row
            self._mode = "plugin_details"
            self._error = None
            self._refresh_view()
            return
        if option_id.startswith("installed:"):
            plugin_id = option_id.removeprefix("installed:")
            row = self._find_installed_plugin(plugin_id)
            if row is None:
                return
            self._selected_plugin = row
            self._mode = "installed_details"
            self._error = None
            self._status = None
            self._refresh_view()
            return
        if option_id == "action:toggle-auto-update":
            await self._toggle_auto_update()
            return
        if option_id == "action:install":
            await self._install_selected_plugin()
            return
        if option_id == "action:toggle-enabled":
            await self._toggle_selected_plugin_enabled()
            return
        if option_id == "action:uninstall":
            await self._uninstall_selected_plugin()
            return
        if option_id == "action:remove-marketplace":
            self._mode = "confirm_remove_marketplace"
            self._error = None
            self._refresh_view()
            return
        if option_id == "action:confirm-remove-marketplace":
            await self._remove_selected_marketplace()
            return
        if option_id == "details-back":
            self._mode = (
                "marketplace_details"
                if self._mode == "confirm_remove_marketplace"
                else "list"
            )
            self._selected_plugin = None
            if self._mode == "list":
                self._selected_marketplace = None
            self._refresh_view()

    def _find_available_plugin(self, plugin_id: str) -> _PluginRow | None:
        return next(
            (
                row
                for row in self._state.available_plugins
                if row.plugin_id == plugin_id
            ),
            None,
        )

    def _find_installed_plugin(self, plugin_id: str) -> _PluginRow | None:
        return next(
            (
                row
                for row in self._state.installed_plugins
                if row.plugin_id == plugin_id
            ),
            None,
        )

    def _find_marketplace(self, name: str) -> _MarketplaceRow | None:
        return next(
            (row for row in self._state.marketplaces if row.name == name),
            None,
        )

    async def _toggle_auto_update(self) -> None:
        enabled = not self._auto_update_enabled
        if not await asyncio.to_thread(
            _save_toml_field, "plugins", "auto_update", enabled
        ):
            self._error = "Could not save the plugin auto-update setting."
            self._status = None
            self._refresh_view()
            return
        self._auto_update_enabled = enabled
        self._auto_update_source = "config.toml"
        self._status = f"Plugin auto-updates {'enabled' if enabled else 'disabled'}."
        self._error = None
        self._refresh_view()
        if enabled and self._on_auto_update_enabled is not None:
            self._on_auto_update_enabled()

    async def _install_selected_plugin(self) -> None:
        row = self._selected_plugin
        if row is None:
            return
        try:
            instance = await asyncio.to_thread(install_plugin, row.plugin_id)
        except (MarketplaceError, FileNotFoundError, OSError, ValueError) as exc:
            self._error = str(exc)
            self._status = None
            self._refresh_view()
            return
        from deepagents_code.plugins.adapters.mcp import plugin_mcp_server_entries

        label = row.label
        needs_login = any(
            needs_login
            for _server_label, _scoped, needs_login in plugin_mcp_server_entries(
                instance
            )
        )
        self.notify(f"Installed {label}", timeout=5, markup=False)
        self._mode = "list"
        self._tab = "installed"
        self._selected_plugin = None
        self._status = f"Installed {label}. Run /reload to activate."
        if needs_login:
            self._status += f" After reload, sign in to {label} via /mcp."
        self._error = None
        await self._refresh_state()

    async def _toggle_selected_plugin_enabled(self) -> None:
        row = self._selected_plugin
        if row is None:
            return
        try:
            await asyncio.to_thread(
                set_installed_plugin_enabled, row.plugin_id, enabled=not row.enabled
            )
            if row.enabled:
                self._status = f"Disabled {row.label}. Run /reload to unload."
                self._mode = "list"
                self._selected_plugin = None
            else:
                self._status = f"Enabled {row.label}. Run /reload to activate."
                self._mode = "list"
                self._tab = "installed"
                self._selected_plugin = None
        except OSError as exc:
            self._error = f"Could not update plugin state: {exc}"
            self._status = None
            self._refresh_view()
            return
        self._error = None
        await self._refresh_state()

    async def _uninstall_selected_plugin(self) -> None:
        row = self._selected_plugin
        if row is None:
            return
        try:
            await asyncio.to_thread(uninstall_plugin, row.plugin_id)
        except OSError as exc:
            self._error = f"Could not uninstall plugin: {exc}"
            self._status = None
            self._refresh_view()
            return
        self._mode = "list"
        self._selected_plugin = None
        reload_hint = " Run /reload to unload." if row.enabled else ""
        self._status = f"Uninstalled {row.label}.{reload_hint}"
        self._error = None
        await self._refresh_state()

    async def _remove_selected_marketplace(self) -> None:
        row = self._selected_marketplace
        if row is None:
            return
        self._status = f"Removing marketplace {row.name}..."
        self._error = None
        try:
            removed = await asyncio.to_thread(remove_marketplace, row.name)
        except OSError as exc:
            self._status = None
            self._error = f"Could not remove marketplace: {exc}"
            self._refresh_view()
            return
        if not removed:
            self._status = None
            self._error = f"Marketplace {row.name} is no longer configured."
            await self._refresh_state()
            return
        plugin_label = "plugin" if row.installed_count == 1 else "plugins"
        self._mode = "list"
        self._tab = "marketplaces"
        self._selected_marketplace = None
        self._status = (
            f"Removed marketplace {row.name} and uninstalled "
            f"{row.installed_count} {plugin_label}."
        )
        self._error = None
        await self._refresh_state()

    def on_input_changed(self, event: Input.Changed) -> None:
        """Filter the current plugin list as the search query changes."""
        if event.input.id != "plugin-manager-search":
            return
        self._search_query = event.value
        self._refresh_view()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Activate a search result or start adding a marketplace."""
        if event.input.id == "plugin-manager-search":
            event.stop()
            options = self.query_one("#plugin-manager-options", OptionList)
            highlighted = options.highlighted
            if highlighted is None:
                return
            option_id = options.get_option_at_index(highlighted).id
            if option_id is None or not option_id.startswith(("detail:", "installed:")):
                return
            options.focus()
            options.action_select()
            return
        if event.input.id != "plugin-marketplace-source" or self._adding_marketplace:
            return
        source = event.value.strip()
        if not source:
            self._error = "Please enter a marketplace source."
            self._refresh_view()
            return
        self._adding_marketplace = True
        event.input.disabled = True
        self._error = None
        self._marketplace_spinner_timer = self.set_interval(
            0.1, self._tick_marketplace_spinner
        )
        self._tick_marketplace_spinner()
        self._add_marketplace(source)

    def _tick_marketplace_spinner(self) -> None:
        self._status = f"{self._marketplace_spinner.next_frame()} Adding marketplace..."
        self._refresh_view()

    @work(thread=True, exclusive=True, exit_on_error=False)
    def _add_marketplace(self, source: str) -> None:
        try:
            marketplace = add_marketplace_source(source)
        except (MarketplaceError, OSError, RuntimeError) as exc:
            self.app.call_from_thread(self._finish_marketplace_add, None, str(exc))
            return
        except Exception as exc:
            # Any exception outside the expected set must still route through
            # _finish_marketplace_add: with exit_on_error=False the worker
            # crash is otherwise swallowed, leaving _adding_marketplace latched
            # (spinner running, input disabled, Escape blocked by action_cancel)
            # and the manager permanently unrecoverable.
            logger.exception("Unexpected error adding marketplace source")
            self.app.call_from_thread(
                self._finish_marketplace_add, None, f"Unexpected error: {exc}"
            )
            return
        self.app.call_from_thread(self._finish_marketplace_add, marketplace, None)

    async def _finish_marketplace_add(
        self, marketplace: PluginMarketplace | None, error: str | None
    ) -> None:
        if self._marketplace_spinner_timer is not None:
            self._marketplace_spinner_timer.stop()
            self._marketplace_spinner_timer = None
        source_input = self.query_one("#plugin-marketplace-source", Input)

        def release_guard() -> None:
            """Re-enable the input and unblock Escape now the add has settled."""
            self._adding_marketplace = False
            source_input.disabled = False

        if error is not None:
            release_guard()
            self._status = None
            self._error = f"Could not add marketplace: {error}"
            self._refresh_view()
            source_input.focus()
            return
        if marketplace is None:
            release_guard()
            return
        self._mode = "list"
        self._tab = "discover"
        self._status = (
            f"Added marketplace {marketplace.name} "
            f"({len(marketplace.plugins)} plugin(s))."
        )
        self._error = None
        # Keep the guard set until the refresh finishes so Escape stays blocked
        # while _refresh_state() runs.
        try:
            await self._refresh_state()
        finally:
            release_guard()
